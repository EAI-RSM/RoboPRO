"""Ghost-futures branching videos as a first-class lookahead capability.

Per scene (task, task_config, seed) this renders one composited MP4: playback
follows the search's COMMITTED trajectory; at each decision point (a committed
chunk boundary) the video freezes and STAYS frozen while the K candidate futures
are revealed ONE AT A TIME as translucent ghosts color-coded by their
held-to-terminal outcome tier (green = success, red = collision, gray = plain
fail); each revealed ghost persists as a faint trail, the committed future is
revealed LAST and popped to full opacity with a "selected" tag, then the fan fades
and playback resumes from the frozen frame along it.

Three stages, each rerunnable; viz NEVER re-runs the search:
  1. CAPTURE (GPU, sim + policy): :func:`capture_trace` — the ONLY stage that
     touches the candidate policy/fitness. Re-based entirely on lookahead
     primitives (:func:`rollback.snapshot` / :func:`rollback.restore` /
     :func:`rollback.apply_chunk`, ``CandidateSource.propose``,
     ``Fitness.outcome``): at every committed chunk boundary of a
     :class:`~lookahead.search.SearchResult` it holds each non-chosen candidate
     index to the terminal and records the RAW ACTION ROWS + scored outcome
     (no frames). Emits a :class:`~lookahead.trace.SearchTrace` — the shared
     policy-free artifact for viz + replay + Q-data.
  2. RENDER (GPU, sim only — deterministic, POLICY-FREE): :func:`render_trace` —
     consumes a saved SearchTrace exactly like replay consumes a plan: pure
     raw-action playback of the saved spine/ghost rows via
     ``rollback.apply_chunk`` grabbing camera frames. No ``candidates.propose``,
     no fitness, no model. Writes one bundle dir (``frames.npz`` + ``meta.json``)
     per scene, with tiers pulled from the trace (never re-scored).
  3. COMPOSITE (CPU, numpy + cv2): :func:`compose_scene_video` — ported verbatim
     from the tuned reference implementation. Because all futures share a fixed
     camera, a per-decision median plate isolates the moving arm/objects; ghosts
     are masked, tinted, and alpha-blended on the frozen frame.

The bundle schema is byte-compatible with the reference bundles
(``meta.json``: task/config/seed/instr/chosen_mode/spine_chunks/spine_boundaries/
spine_tier/decisions/fans; ``frames.npz``: ``spine``, ``fan_t{t}_k{k}``,
``settle``), so either side can consume the other stack's output.

No jax at import time — the candidate model only enters through the
``CandidateSource``, exactly as in :mod:`lookahead.run_search`. No
``eval_b3h6`` / ``per_step_oracle`` / ``outcome_diversity`` imports.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import rollback
from .candidates import CandidateSource
from .fitness import Fitness, tier_of
from .plan import Plan, TaskSpec
from .search import SearchResult
from .trace import SearchTrace

TIER_CODE = {"HARD": 0, "SOFT": 1, "COLLIDED_FAIL": 2, "FAIL": 3}
GREEN, RED, GRAY = (60, 210, 120), (240, 80, 70), (118, 122, 134)


# ==================================================================== record phase
def grab_frame(task: Any, camera_name: str = "countertop_camera") -> np.ndarray:
    """One static-camera rgb frame (uint8 HWC) — renders ONE camera instead of the
    full get_obs (3 cams rgb + 3 depth + state), which dominates recording cost."""
    task._update_render()
    idx = task.cameras.static_camera_name.index(camera_name)
    cam = task.cameras.static_camera_list[idx]
    cam.take_picture()
    rgba = cam.get_picture("Color")
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255).astype(np.uint8)


def _hold_actions(task: Any, candidates: CandidateSource, mode: int, k: int,
                  depth0: int, horizon: int, *,
                  first_chunks: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Hold candidate index ``mode`` from the CURRENT (restored boundary) state to
    the terminal, returning the RAW ACTION ROWS taken (float32 ``[T, adim]``) —
    no frames; the render stage regenerates them deterministically.

    SEMANTIC NOTE — "holding mode k" here means: at EVERY depth re-propose the K
    candidates from the current state and commit index ``k`` (a greedy
    same-candidate-index hold). This replaces the old stack's fixed-noise
    convention (``pso.cnoise(seed, decision, mode, depth)``). For a WTA proposer
    (``mode_propose`` / a policy's ``propose_candidates`` hook) index k IS latent
    mode z=k, so this is a true mode hold; for the default ``noise_propose`` index
    k maps to the k-th seeded noise, i.e. a fixed-noise hold re-inferred at each
    new state. ``first_chunks`` (the proposal already made AT the boundary) is
    reused for the first depth so every ghost of a decision shares one proposal.
    Rows a mid-chunk success turns into no-ops are still recorded — replay
    re-latches ``eval_success`` at the same row, so playback stays faithful.
    """
    rows: List[np.ndarray] = []
    for depth in range(depth0, horizon):
        if depth > depth0 and task.check_success():
            break
        chunks = (first_chunks if depth == depth0 and first_chunks is not None
                  else candidates.propose(task, k))
        chunk_rows = np.asarray(chunks[mode], dtype=np.float32)
        rollback.apply_chunk(task, chunk_rows)
        rows.append(chunk_rows)
    return np.concatenate(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def capture_trace(task: Any, candidates: CandidateSource, fitness: Fitness,
                  committed: SearchResult, *, horizon: int,
                  task_spec: TaskSpec, k: Optional[int] = None,
                  meta: Optional[Dict[str, Any]] = None) -> SearchTrace:
    """POLICY+SIM stage: expand each committed boundary's K-fan into a SearchTrace.

    The ONLY viz stage that touches the candidate policy / fitness. The task must
    be AT THE ROOT state (t0) of ``committed`` when this is called (the caller
    restores its pre-search snapshot). The committed spine is replayed by
    raw-action playback of ``committed.actions`` split into ``candidates.chunk``-
    row chunks; every chunk boundary is a decision point (the terminal is not
    one). At each decision the boundary is snapshotted and every NON-chosen
    candidate index is held to the terminal (:func:`_hold_actions`), recording its
    raw action rows and terminal ``fitness.outcome`` / tier. The chosen entry's
    actions reference the committed continuation (``actions[d * chunk:]``) with
    the spine's terminal tier — the chosen index is NOT re-held, mirroring the
    reference bundles' t>0 convention and keeping the "selected" ghost consistent
    with the trajectory the video actually commits to.

    Returns the :class:`~lookahead.trace.SearchTrace` (no frames — those are the
    render stage's job).
    """
    ctx = fitness.make_context(task)
    root = rollback.snapshot(task)
    # defensive: restore the root we just snapshotted — a no-op state-wise, but it
    # clears any latched eval_success (take_action no-ops while it is set)
    rollback.restore(task, root)
    chunk = int(candidates.chunk)
    k = int(k if k is not None else getattr(candidates, "k", 6))
    actions = np.asarray(committed.actions, dtype=np.float32)
    chosen = [int(i) for i in committed.candidate_indices]
    n_chunks = len(chosen)
    if n_chunks == 0 or actions.shape[0] <= (n_chunks - 1) * chunk:
        raise ValueError(
            f"committed SearchResult is inconsistent: {actions.shape[0]} action rows "
            f"for {n_chunks} committed chunks of {chunk}")
    spine_chunks = [actions[d * chunk:(d + 1) * chunk] for d in range(n_chunks)]
    spine_outcome = dict(committed.outcome or {})
    spine_tier = spine_outcome.get("tier") or tier_of(
        bool(committed.success), bool(spine_outcome.get("is_collision", False)))
    committed_plan = Plan(
        task_spec=task_spec,
        actions=actions.tolist(),
        fingerprints={"t0": np.asarray(committed.root_fingerprint).tolist(),
                      "terminal": np.asarray(committed.terminal_fingerprint).tolist()},
        meta={"candidate_indices": chosen, "outcome": spine_outcome,
              "success": bool(committed.success), "score": list(committed.score),
              "depth": int(committed.depth)})
    print(f"[trace] {task_spec.task_name} {task_spec.task_config} "
          f"seed{task_spec.seed}: K={k} chunk={chunk} spine_chunks={n_chunks} "
          f"spine_tier={spine_tier} indices={chosen}", flush=True)

    fans: List[List[Dict[str, Any]]] = []
    for d in range(n_chunks):
        boundary = rollback.snapshot(task)
        # one proposal AT the boundary, shared as every ghost's first chunk
        first_chunks = candidates.propose(task, k)
        entries: List[Dict[str, Any]] = []
        for m in range(k):
            if m == chosen[d]:
                entries.append({"mode": m, "actions": actions[d * chunk:],
                                "tier": spine_tier, "outcome": spine_outcome,
                                "chosen": True})
                continue
            rollback.restore(task, boundary)
            rows = _hold_actions(task, candidates, m, k, d, horizon,
                                 first_chunks=first_chunks)
            out = fitness.outcome(task, ctx)
            tier = out.get("tier") or tier_of(
                bool(out.get("success", False)), bool(out.get("is_collision", False)))
            entries.append({"mode": m, "actions": rows, "tier": tier,
                            "outcome": out, "chosen": False})
            print(f"[trace]   t={d} mode {m}: {tier} rows={rows.shape[0]}", flush=True)
        fans.append(entries)
        rollback.restore(task, boundary)
        rollback.apply_chunk(task, spine_chunks[d])   # advance along the spine
    return SearchTrace(task_spec=task_spec, chunk=chunk, horizon=int(horizon),
                       committed=committed_plan, fans=fans, meta=dict(meta or {}))


def render_trace(task: Any, trace: SearchTrace, *, out_dir: str,
                 frame_every: int = 3, fan_record_chunks: int = 2,
                 camera: str = "countertop_camera",
                 settle_steps: int = 24, settle_every: int = 2) -> str:
    """DETERMINISTIC, POLICY-FREE render stage: SearchTrace -> scene bundle.

    Consumes a saved trace exactly like replay consumes a plan: the task must be
    a FRESH scene at the trace's (task, task_config, seed) root, and everything
    is pure raw-action playback — the committed spine chunk-by-chunk, and at each
    boundary every non-chosen ghost's SAVED rows (only the first
    ``fan_record_chunks`` chunks are replayed/recorded; frames beyond that are
    never composited). NO ``candidates.propose``, NO fitness, NO model — tiers
    come from the trace, never re-scored. Determinism: ``take_action`` re-latches
    ``eval_success`` at the same row it did during capture, so the saved rows
    reproduce the searched trajectories exactly.

    Writes ``frames.npz`` + ``meta.json`` in the reference bundle schema that
    :func:`compose_scene_video` reads, and returns ``out_dir``.
    """
    spec = trace.task_spec
    chunk = int(trace.chunk)
    chosen = trace.candidate_indices
    spine_chunks = trace.spine_chunks()
    n_chunks = len(trace.fans)
    if chosen and len(chosen) != n_chunks:
        raise ValueError(f"trace is inconsistent: {len(chosen)} candidate_indices "
                         f"vs {n_chunks} fans")
    spine_outcome = trace.committed.meta.get("outcome") or {}
    spine_tier = spine_outcome.get("tier") or tier_of(
        bool(trace.committed.meta.get("success", False)),
        bool(spine_outcome.get("is_collision", False)))
    root = rollback.snapshot(task)
    rollback.restore(task, root)             # clears any latched eval_success
    print(f"[render] {spec.task_name} {spec.task_config} seed{spec.seed}: "
          f"chunk={chunk} spine_chunks={n_chunks} spine_tier={spine_tier}", flush=True)

    arrays: Dict[str, np.ndarray] = {}
    fans_meta: List[Dict[str, Any]] = []
    spine_frames = [grab_frame(task, camera)]
    bounds = [0]
    step = 0
    for d in range(n_chunks):
        boundary = rollback.snapshot(task)
        modes_meta: List[Dict[str, Any]] = []
        for e in trace.fans[d]:
            m = int(e["mode"])
            tier = TIER_CODE[e["tier"]] if isinstance(e["tier"], str) else int(e["tier"])
            if e["chosen"]:                  # committed continuation == the spine
                modes_meta.append({"mode": m, "tier": tier, "chosen": True})
                continue
            rollback.restore(task, boundary)
            frames = [grab_frame(task, camera)]
            gstep = 0
            rows = np.asarray(e["actions"])[:fan_record_chunks * chunk]
            for a in rows:                   # saved-row playback, frame-gated
                task.take_action(a)
                gstep += 1
                if gstep % frame_every == 0:
                    frames.append(grab_frame(task, camera))
            arrays[f"fan_t{d}_k{m}"] = np.stack(frames)
            modes_meta.append({"mode": m, "tier": tier, "chosen": False})
        fans_meta.append({"t": d, "modes": modes_meta})
        rollback.restore(task, boundary)
        for a in spine_chunks[d]:            # advance along the committed spine
            task.take_action(a)
            step += 1
            if step % frame_every == 0:
                spine_frames.append(grab_frame(task, camera))
        bounds.append(len(spine_frames))
    arrays["spine"] = np.stack(spine_frames)

    settle_frames = []                       # settle at the terminal
    for i in range(int(settle_steps)):
        task.scene.step()
        if i % max(1, int(settle_every)) == 0:
            settle_frames.append(grab_frame(task, camera))
    arrays["settle"] = np.stack(settle_frames or [spine_frames[-1]])

    meta = {"task": spec.task_name, "config": spec.task_config,
            "seed": int(spec.seed), "instr": trace.meta.get("instruction"),
            "chosen_mode": chosen[0] if chosen else 0, "spine_chunks": n_chunks,
            "spine_boundaries": bounds, "spine_tier": spine_tier,
            "decisions": list(range(n_chunks)), "fans": fans_meta,
            # extra provenance (ignored by the composite)
            "candidate_indices": chosen, "horizon": int(trace.horizon),
            "frame_every": int(frame_every), "camera": camera}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "frames.npz"), **arrays)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[render] wrote {out_dir}", flush=True)
    return out_dir


# ================================================================ composite phase
def outcome_color(tier):
    return {0: GREEN, 1: GREEN, 2: RED, 3: GRAY}[int(tier)]


def build_plate(frame_sets, base, stride=5):
    """Pixelwise median background plate over the base + subsampled future frames."""
    sample = [base.astype(np.float32)]
    for frames in frame_sets:
        sample.extend(f.astype(np.float32) for f in frames[::stride])
    return np.median(np.stack(sample), axis=0)


def ghost_mask(frame, plate, lo=18.0, hi=55.0, blur=3):
    """Soft 0..1 mask (HxWx1) of where the future differs from the background plate.
    Thresholds sit above the soft-shadow level so only the arm/objects ghost."""
    diff = np.abs(frame.astype(np.float32) - plate).sum(axis=-1) / 3.0
    m = np.clip((diff - lo) / (hi - lo), 0.0, 1.0)
    return cv2.GaussianBlur(m, (blur, blur), 0)[..., None]


def overlay_ghost(comp, frame, mask, color, alpha, tint):
    ghost = (1.0 - tint) * frame.astype(np.float32) + tint * np.asarray(color, np.float32)
    a = alpha * mask
    return comp * (1.0 - a) + ghost * a


def draw_label_box(img, lines, org, pad=6, fg=(235, 238, 245), dot_colors=None):
    """Small semi-transparent label box with optional colored dots per line (BGR-free:
    img is RGB uint8, modified in place)."""
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    sizes = [cv2.getTextSize(l, font, fs, th)[0] for l in lines]
    w = max(s[0] for s in sizes) + 2 * pad + (16 if dot_colors else 0)
    h = sum(s[1] + 8 for s in sizes) + 2 * pad - 2
    x, y = org
    roi = img[y:y + h, x:x + w].astype(np.float32)
    img[y:y + h, x:x + w] = (roi * 0.32 + np.array([16, 18, 24], np.float32) * 0.68).astype(np.uint8)
    cy = y + pad + sizes[0][1]
    for i, l in enumerate(lines):
        cx = x + pad
        if dot_colors:
            cv2.circle(img, (cx + 5, cy - 4), 4, dot_colors[i], -1, cv2.LINE_AA)
            cx += 16
        cv2.putText(img, l, (cx, cy), font, fs, fg, th, cv2.LINE_AA)
        cy += sizes[i][1] + 8


def annotate(img, scene_tag, phase_text, phase_color=(235, 238, 245)):
    """Persistent overlays: scene tag (top-left), outcome legend (top-right), and an
    optional phase caption (bottom-left). img: RGB uint8 (upscaled), in place."""
    draw_label_box(img, [scene_tag], (10, 10))
    legend = ["success", "collision", "fail"]
    draw_label_box(img, legend, (img.shape[1] - 108, 10), dot_colors=[GREEN, RED, GRAY])
    if phase_text:
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1
        (tw, tht), _ = cv2.getTextSize(phase_text, font, fs, th)
        x, y = 10, img.shape[0] - 14
        roi = img[y - tht - 8:y + 6, x - 2:x + tw + 10].astype(np.float32)
        img[y - tht - 8:y + 6, x - 2:x + tw + 10] = \
            (roi * 0.32 + np.array([16, 18, 24], np.float32) * 0.68).astype(np.uint8)
        cv2.putText(img, phase_text, (x + 4, y), font, fs, phase_color, th, cv2.LINE_AA)


def upscale(frame, factor=2):
    return cv2.resize(frame, (frame.shape[1] * factor, frame.shape[0] * factor),
                      interpolation=cv2.INTER_CUBIC)


def draw_ghost_outline(img, mask, color, factor=2, thickness=1):
    """Thin outline of a ghost silhouette on the upscaled frame — used when a ghost's
    fill is mostly hidden behind the frozen arm so it stays visible."""
    binary = (mask[..., 0] > 0.5).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c * factor for c in contours if cv2.contourArea(c) > 150]
    cv2.drawContours(img, big, -1, color, thickness, cv2.LINE_AA)


def mask_centroid(mask):
    """(x, y) intensity-weighted centroid of a HxWx1 soft mask, or None if empty."""
    m = mask[..., 0]
    total = float(m.sum())
    if total < 1.0:
        return None
    ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    return int((xs * m).sum() / total), int((ys * m).sum() / total)


def compose_scene_video(bundle_dir, scene_tag, alpha=0.5, tint=0.62,
                        freeze_frames=8, reveal_anim=8, reveal_hold=3,
                        reveal_step=3, trail_alpha=0.2, select_pop=3,
                        select_hold=12, release_frames=4):
    """Assemble the final annotated frame sequence for one scene bundle. At each
    decision point the video freezes and STAYS frozen for the whole beat: the K
    candidate futures are revealed one at a time (each ghost animates forward over
    reveal_anim frames stepping reveal_step poses per frame, holds reveal_hold
    frames, then persists as a faint trail at trail_alpha); the committed future is
    revealed LAST, popped to full opacity with a "selected" tag over
    select_pop+select_hold frames, the fan fades out over release_frames, and
    playback resumes from the frozen frame along the chosen future. Returns
    (frames_uint8_list, meta, previews) where previews maps name -> frame for the
    inspection PNGs (mid-reveal of each decision + the success hold)."""
    meta = json.load(open(os.path.join(bundle_dir, "meta.json")))
    arrays = np.load(os.path.join(bundle_dir, "frames.npz"))
    spine, settle = arrays["spine"], arrays["settle"]
    bounds = meta["spine_boundaries"]
    # EVERY decision boundary gets a beat. The reveal happens entirely on the frozen
    # frame and playback resumes AT the decision frame, so consecutive beats always
    # play strictly forward (no rewind, no overshoot).
    decisions = [t for t in meta["decisions"]
                 if any(x["t"] == t for x in meta["fans"])]
    n_dec = len(decisions)
    meta["decisions"] = decisions               # rendered decision points
    out_frames, cursor, previews = [], 0, {}
    last_pose = reveal_step * (reveal_anim - 1)  # final ghost pose index of a reveal
    tier_word = {0: "success", 1: "success", 2: "collision", 3: "fail"}

    for di, t in enumerate(decisions):
        b = bounds[t]
        for i in range(cursor, b):                       # spine playback up to the decision
            f = upscale(spine[i])
            annotate(f, scene_tag, "")
            out_frames.append(f)
        fan = next(x for x in meta["fans"] if x["t"] == t)
        futures = []                                     # non-chosen first, chosen LAST
        for m in sorted(fan["modes"], key=lambda x: (x["chosen"], x["mode"])):
            if m["chosen"]:
                frames = spine[b + 1:]                          # committed continuation
            else:
                frames = arrays[f"fan_t{t}_k{m['mode']}"][1:]   # drop the restored-base frame
            futures.append({"frames": frames, "tier": m["tier"], "chosen": m["chosen"],
                            "mode": m["mode"]})
        K = len(futures)
        base = spine[b].astype(np.float32)
        plate = build_plate([fu["frames"][:last_pose + 2] for fu in futures], spine[b])
        # mostly avoid painting ghosts over the frozen frame's own content (the parked
        # arm): without this, futures that moved away "erase" the frozen arm with
        # tinted background pulled from the contaminated median plate. 0.85 (not 1.0)
        # lets low-motion futures still faintly tint the arm they overlap.
        keep_base = 1.0 - 0.85 * ghost_mask(spine[b], plate)

        probe_masks = [ghost_mask(fu["frames"][min(12, len(fu["frames"]) - 1)], plate)
                       for fu in futures]                # divergence locus for the marker
        locus = mask_centroid(np.max(np.stack(probe_masks), axis=0))

        dec_text = f"decision {di + 1}/{n_dec} - exploring K={K} futures"
        print(f"[compose] {scene_tag} decision {di + 1}: freeze@{len(out_frames)}", flush=True)
        for i in range(freeze_frames):                   # freeze on the decision frame
            f = upscale(spine[b])
            annotate(f, scene_tag, dec_text if i >= freeze_frames // 3 else "")
            if locus is not None:                        # pulsing decision-point marker
                r = 16 + int(8 * np.sin(i * 0.9))
                cv2.circle(f, (locus[0] * 2, locus[1] * 2), r, (255, 255, 255), 1, cv2.LINE_AA)
            out_frames.append(f)

        trail_comp = base.copy()      # frozen frame + faint trails of revealed futures
        trail_outlines = []           # mostly-hidden trails keep a thin outline
        sel_comp = base
        for ci, fu in enumerate(futures):
            color = outcome_color(fu["tier"])
            n = len(fu["frames"])
            cap = (f"decision {di + 1}/{n_dec} - future {ci + 1}/{K}: "
                   f"{tier_word[int(fu['tier'])]}")
            print(f"[compose]   reveal {ci + 1}/{K} mode={fu['mode']} "
                  f"tier={fu['tier']} chosen={fu['chosen']} @{len(out_frames)}", flush=True)
            for j in range(reveal_anim):                 # this future's ghost plays forward
                fr = fu["frames"][min(reveal_step * j, n - 1)]
                m = ghost_mask(fr, plate)
                vis = m * keep_base
                a = alpha * min(1.0, (j + 1) / 4.0)      # quick fade-in of the new ghost
                comp = overlay_ghost(trail_comp.copy(), fr, vis, color, a, tint)
                f = upscale(np.clip(comp, 0, 255).astype(np.uint8))
                if j >= 3 and float(vis.sum()) < 0.55 * max(float(m.sum()), 1.0):
                    draw_ghost_outline(f, m, color)      # mostly-hidden ghost -> outline
                for tm, tc in trail_outlines:
                    draw_ghost_outline(f, tm, tc)
                annotate(f, scene_tag, cap, phase_color=color)
                out_frames.append(f)
            fr_last = fu["frames"][min(last_pose, n - 1)]
            m_last = ghost_mask(fr_last, plate)
            if not fu["chosen"]:
                for _ in range(reveal_hold):             # hold the fully-revealed ghost
                    out_frames.append(f)
                trail_comp = overlay_ghost(trail_comp, fr_last, m_last * keep_base,
                                           color, trail_alpha, tint)
                if float((m_last * keep_base).sum()) < 0.55 * max(float(m_last.sum()), 1.0):
                    trail_outlines.append((m_last, color))
                if ci == 2:                              # mid-reveal: some shown, some not
                    previews[f"decision{di + 1}_fan"] = f
                continue
            # chosen future (always last): pop to full opacity + "selected" tag,
            # held on the frozen frame over the whole faint fan.
            ch_pt = mask_centroid(m_last)
            print(f"[compose]   selected beat @{len(out_frames)}", flush=True)
            for g in range(select_pop + select_hold):
                w = min(1.0, (g + 1) / select_pop)
                hl_mask = m_last * (keep_base * (1.0 - w) + w)
                comp = overlay_ghost(trail_comp.copy(), fr_last, hl_mask, color,
                                     alpha + (1.0 - alpha) * w, tint * (1.0 - w) + 0.15 * w)
                f = upscale(np.clip(comp, 0, 255).astype(np.uint8))
                for tm, tc in trail_outlines:
                    draw_ghost_outline(f, tm, tc)
                draw_ghost_outline(f, m_last, color, thickness=2)
                annotate(f, scene_tag, "selected - committing this future",
                         phase_color=GREEN)
                if ch_pt is not None and w >= 1.0:       # "selected" tag on the winner
                    x, y = ch_pt[0] * 2, ch_pt[1] * 2
                    cv2.circle(f, (x, y), 22, (255, 255, 255), 1, cv2.LINE_AA)
                    tx, ty = min(x + 28, f.shape[1] - 96), max(y - 14, 18)
                    cv2.putText(f, "selected", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(f, "selected", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                out_frames.append(f)
                sel_comp = comp
        for g in range(release_frames):                  # fan fades out on the frozen frame
            w = (g + 1) / (release_frames + 1.0)
            f = upscale(np.clip(sel_comp * (1.0 - w) + base * w, 0, 255).astype(np.uint8))
            annotate(f, scene_tag, "committing best future", phase_color=GREEN)
            out_frames.append(f)
        print(f"[compose]   resume @{len(out_frames)}", flush=True)
        cursor = b + 1                # unfreeze: chosen future IS the spine from here

    for i in range(cursor, len(spine)):                  # final stretch to task success
        f = upscale(spine[i])
        annotate(f, scene_tag, "")
        out_frames.append(f)
    for i, sf in enumerate(list(settle) + [settle[-1]] * 18):
        f = upscale(sf)
        annotate(f, scene_tag, "task success", phase_color=GREEN)
        out_frames.append(f)
    previews["success"] = out_frames[-1]
    for g in range(8):                                   # crossfade to the first frame -> loopable
        w = (g + 1) / 9.0
        f = (out_frames[-1].astype(np.float32) * (1 - w)
             + out_frames[0].astype(np.float32) * w).astype(np.uint8)
        out_frames.append(f)
    return out_frames, meta, previews


def write_mp4(frames, path, fps=15, crf=27):
    """Encode via ffmpeg (H.264, yuv420p, faststart) for a small data-URI-able file."""
    tmp = tempfile.mkdtemp(prefix="branchvid_")
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, f"f{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
                    "-i", os.path.join(tmp, "f%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                   check=True, capture_output=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"[mp4] {path}: {os.path.getsize(path) / 1e6:.2f} MB, "
          f"{len(frames) / fps:.1f}s", flush=True)


def write_fragment(clip_paths, out_html):
    """Self-contained slide fragment: VIDEO-ONLY (every label is baked into the
    frames). All scene clips sit side by side, full-bleed centered, embedded as
    base64 data URIs (strict-CSP safe, no external refs)."""
    vids = []
    for path in clip_paths:
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        vids.append(f'<video autoplay muted playsinline controls preload="auto">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')
    html = f"""<section class="slide">
<div class="inner">
  <style>
    .branchfx{{display:flex;gap:16px;flex-wrap:wrap;align-items:center;
      justify-content:center;width:100%;height:100%;min-height:70vh}}
    .branchfx video{{display:block;flex:1 1 420px;min-width:300px;max-width:640px;
      width:100%;max-height:82vh;height:auto;border-radius:12px;
      box-shadow:0 14px 40px -18px rgba(0,0,0,.55)}}
  </style>
  <div class="branchfx">{''.join(vids)}</div>
</div>
</section>
"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[fragment] {out_html}: {os.path.getsize(out_html) / 1e6:.2f} MB", flush=True)
