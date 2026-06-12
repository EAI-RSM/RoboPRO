"""
relabel_collision_dataset.py — offline label pass for the paired collision dataset.

Reads what the collector stored (see data_generation_documentation.md):
    <dataset>/data/episode_N.hdf5      qpos (joint_action/vector), attrs seed/source
    <dataset>/scene/seed_S/scene.npz   surface samples + normals + per-sample obj_id
    <dataset>/scene/seed_S/objects.json  tags (target/container/furniture/clutter/...)
    <dataset>/metrics/seedS_<source>_contacts.jsonl
    <dataset>/fk_basis.npz             serialized FK + sphere decomposition
    <dataset>/index.jsonl              one line per episode

and writes, per episode:
    HDF5 datasets   link_clearance/d            (T, L, K)   top-K clearance per link, clipped
                    link_clearance/u            (T, L, K, 3) unit away-vectors (obstacle→link)
                    link_clearance/obj          (T, L, K)   per_scene_id (-1 = empty slot)
                    link_clearance/sphere_idx   (T, L, K)   argmin sphere WITHIN the link
                    link_clearance/names        (L,)        link name per row
                    link_margin                 (T, L)      top-1 clearance < margin
                    in_contact_window           (T,)        within ±window of an engine contact
    metrics/..._contacts_attributed.jsonl       contacts + nearest-sphere attribution
    data/episode_N.meta.json                    contact windows, counts, min clearance

All labels are evaluated at each step's MEASURED qpos (never lookahead).
Distances are exact KD-tree point distances to the stored surface samples —
no voxel smoothing.  Exclusions happen here (by tag), not at collection.

Usage:
    python relabel_collision_dataset.py <dataset_dir> \
        [--margin 0.025] [--dmax 0.3] [--topk 3] [--knn 32] [--window 50] \
        [--exclude-tags target,container,furniture,skipped]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

_CUROBO_SAVE_FREQ = 15   # must match the collector


# ── FK from the serialized basis (numpy, batched over T) ───────────────────

def _rot_about_axis_np(axis, point, q):
    """(T,) angles → (T,4,4) rotations about the line (point, axis)."""
    T = len(q)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    s = np.sin(q)[:, None, None]
    c = np.cos(q)[:, None, None]
    R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    M = np.zeros((T, 4, 4))
    M[:, :3, :3] = R
    M[:, :3, 3] = point - np.einsum("tij,j->ti", R, point)
    M[:, 3, 3] = 1.0
    return M


class SerializedArm:
    def __init__(self, z, side):
        g = lambda k: z[f"{side}_{k}"]
        self.T_base = g("T_base")
        self.O      = g("O")          # (6,4,4)
        self.axes   = g("axes")       # (6,3)
        self.points = g("points")     # (6,3)
        self.chain  = [str(x) for x in g("chain_links")]
        self.ee_names   = [str(x) for x in g("ee_names")]
        self.ee_offsets = g("ee_offsets")          # (K,4,4)
        self.sph_link   = [str(x) for x in g("sph_link")]
        self.sph_type   = [str(x) for x in g("sph_slot_type")]
        self.sph_idx    = g("sph_slot_idx").astype(int)
        self.sph_center = g("sph_center")          # (S,3)
        self.sph_radius = g("sph_radius")          # (S,)

    def fk(self, Q):
        """Q (T,6) → list of 6 (T,4,4) world transforms."""
        T = self.T_base[None].repeat(len(Q), axis=0)
        out = []
        for i in range(6):
            M = _rot_about_axis_np(self.axes[i], self.points[i], Q[:, i])
            T = T @ self.O[i][None] @ M
            out.append(T)
        return out

    def sphere_centers(self, Q):
        """Q (T,6) → centers (T,S,3); spheres ordered as serialized."""
        Ts = self.fk(Q)
        slotT = {("chain", i): Ts[i] for i in range(6)}
        for k, name in enumerate(self.ee_names):
            slotT[("ee", k)] = Ts[5] @ self.ee_offsets[k][None]
        cols = []
        for s in range(len(self.sph_radius)):
            Tm = slotT[(self.sph_type[s], int(self.sph_idx[s]))]
            c = self.sph_center[s]
            cols.append(np.einsum("tij,j->ti", Tm[:, :3, :3], c) + Tm[:, :3, 3])
        return np.stack(cols, axis=1)   # (T,S,3)


def load_robot(fk_path):
    z = np.load(fk_path, allow_pickle=False)
    arms = [SerializedArm(z, "left"), SerializedArm(z, "right")]
    links, link_of_sphere, within_idx, radii = [], [], [], []
    for arm in arms:
        seen = {}
        for s, ln in enumerate(arm.sph_link):
            if ln not in seen:
                seen[ln] = 0
                links.append(ln)
            link_of_sphere.append(links.index(ln))
            within_idx.append(seen[ln])
            seen[ln] += 1
            radii.append(arm.sph_radius[s])
    return arms, links, np.asarray(link_of_sphere), np.asarray(within_idx), np.asarray(radii)


# ── scene ──────────────────────────────────────────────────────────────────

def load_scene(scene_dir, exclude_tags, exclude_names=frozenset(),
               include_names=frozenset()):
    """Tag-level exclusion with per-name overrides: include_names force-keeps
    an object despite excluded tags; exclude_names masks additional objects.
    Nothing is ever removed from disk — masking is a relabel-time decision."""
    z = np.load(scene_dir / "scene.npz")
    samples, obj_id = z["samples"].astype(np.float64), z["obj_id"]
    objects = json.loads((scene_dir / "objects.json").read_text())
    excluded = {o["per_scene_id"] for o in objects
                if (set(o["tags"]) & exclude_tags or o["name"] in exclude_names)
                and o["name"] not in include_names}
    keep = ~np.isin(obj_id, sorted(excluded))
    if not keep.any():
        raise RuntimeError(f"{scene_dir}: every sample excluded")
    return samples[keep], obj_id[keep], objects


def _pose_mats(P):
    """(...,7) p+q(wxyz, SAPIEN) → (...,4,4)."""
    from scipy.spatial.transform import Rotation
    P = np.asarray(P, dtype=np.float64)
    T = np.tile(np.eye(4), P.shape[:-1] + (1, 1))
    T[..., :3, 3] = P[..., :3]
    q = P[..., [4, 5, 6, 3]]                      # wxyz → xyzw for scipy
    T[..., :3, :3] = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(
        P.shape[:-1] + (3, 3))
    return T


def scene_versions(h5, samples0, sample_obj, drift=0.03):
    """Split the episode into segments with (near-)constant scene geometry.

    Objects move mid-episode (uniquely per run).  Using the recorded
    object_poses track, start a new version whenever any object drifts more
    than `drift` from the current version's reference pose, rigidly re-posing
    that object's samples (x_t = R_t R_0ᵀ (x_0 − p_0) + p_t relative to the
    episode's FIRST frame, which matches the exported scene).  Episodes
    without the pose track (legacy) get a single static version.
    Returns [(t0, t1_exclusive, tree, samples_world), ...].
    """
    T_total = len(h5["joint_action/vector"])
    if "object_poses" not in h5:
        return [(0, T_total, cKDTree(samples0), samples0)]
    ids   = h5["object_poses/ids"][:]
    poses = h5["object_poses/pose"][:]            # (T, n, 7)
    id_to_col = {int(i): k for k, i in enumerate(ids)}
    # samples belonging to ids not in the pose track (articulation boxes,
    # negative ids) stay static.
    sample_col = np.array([id_to_col.get(int(o), -1) for o in sample_obj])

    def _ang(qa, qb):                              # quat angle, wxyz
        qdot = np.abs(np.sum(qa * qb, axis=1))
        return 2 * np.arccos(np.clip(qdot, -1.0, 1.0))

    versions, t0, ref = [], 0, poses[0]
    for t in range(1, len(poses)):
        delta = np.linalg.norm(poses[t, :, :3] - ref[:, :3], axis=1)
        if delta.max() > drift or _ang(poses[t, :, 3:], ref[:, 3:]).max() > 0.2:
            versions.append((t0, t, ref))
            t0, ref = t, poses[t]
    versions.append((t0, len(poses), ref))

    out, M0_inv = [], np.linalg.inv(_pose_mats(poses[0]))     # (n,4,4)
    for (a, b, ref) in versions:
        moved = (np.linalg.norm(ref[:, :3] - poses[0, :, :3], axis=1) > 1e-4) \
                | (_ang(ref[:, 3:], poses[0, :, 3:]) > 1e-3)
        if moved.any():
            Mrel = _pose_mats(ref) @ M0_inv                   # (n,4,4)
            S = samples0.copy()
            for k in np.where(moved)[0]:
                m = sample_col == k
                if m.any():
                    S[m] = S[m] @ Mrel[k, :3, :3].T + Mrel[k, :3, 3]
        else:
            S = samples0
        out.append((a, b, cKDTree(S), S))
    return out


# ── per-episode labeling ───────────────────────────────────────────────────

def label_episode(qpos, arms, link_of_sphere, within_idx, radii,
                  versions, sample_obj, n_links,
                  topk, knn, dmax, margin, batch=100):
    """versions: [(t0, t1, tree, samples_world), ...] from scene_versions() —
    each step is labeled against its own scene geometry."""
    T = len(qpos)
    QL, QR = qpos[:, 0:6].astype(np.float64), qpos[:, 7:13].astype(np.float64)
    D   = np.full((T, n_links, topk), dmax, dtype=np.float32)
    U   = np.zeros((T, n_links, topk, 3), dtype=np.float32)
    OBJ = np.full((T, n_links, topk), -1, dtype=np.int32)
    SPH = np.full((T, n_links, topk), -1, dtype=np.int16)

    spheres_of_link = [np.where(link_of_sphere == l)[0] for l in range(n_links)]

    spans = [(max(v0, t0), min(v1, t0 + batch), tree, samples)
             for (v0, v1, tree, samples) in versions
             for t0 in range(v0, v1, batch)]
    for t0, t1, tree, samples in spans:
        cL = arms[0].sphere_centers(QL[t0:t1])
        cR = arms[1].sphere_centers(QR[t0:t1])
        centers = np.concatenate([cL, cR], axis=1)          # (B,S,3)
        B, S, _ = centers.shape
        d_nn, i_nn = tree.query(centers.reshape(-1, 3), k=knn, workers=-1)
        d_nn = d_nn.reshape(B, S, knn)
        i_nn = i_nn.reshape(B, S, knn)
        for b in range(B):
            for l in range(n_links):
                sph = spheres_of_link[l]
                d_cand = d_nn[b, sph] - radii[sph, None]     # (|sph|, knn) surface dist
                o_cand = sample_obj[i_nn[b, sph]]            # (|sph|, knn)
                flat = np.argsort(d_cand, axis=None)
                seen, slot = set(), 0
                for f in flat:
                    si, ni = divmod(int(f), knn)
                    o = int(o_cand[si, ni])
                    if o in seen:
                        continue
                    seen.add(o)
                    dist = float(d_cand[si, ni])
                    if dist >= dmax:
                        break
                    gs = int(sph[si])                        # global sphere index
                    p = samples[i_nn[b, gs, ni]]
                    v = centers[b, gs] - p
                    nrm = np.linalg.norm(v)
                    D[t0 + b, l, slot]   = dist
                    U[t0 + b, l, slot]   = (v / nrm) if nrm > 1e-9 else 0.0
                    OBJ[t0 + b, l, slot] = o
                    SPH[t0 + b, l, slot] = int(within_idx[gs])
                    slot += 1
                    if slot == topk:
                        break
    return D, U, OBJ, SPH


# ── contacts ───────────────────────────────────────────────────────────────

def frame_of_contact(line, source, n_frames):
    if source == "pi05_base" and line.get("ta", -1) >= 0:
        f = line["ta"]
    else:
        f = line.get("t", 0) // _CUROBO_SAVE_FREQ
    return int(np.clip(f, 0, n_frames - 1))


def attribute_contacts(lines, frames, qpos, arms, link_of_sphere, within_idx,
                       links):
    """Append nearest-sphere-within-link index to every contact pair."""
    out = []
    by_frame = {}
    for ln, fr in zip(lines, frames):
        by_frame.setdefault(fr, []).append(ln)
    for fr, group in sorted(by_frame.items()):
        cL = arms[0].sphere_centers(qpos[fr:fr + 1, 0:6].astype(np.float64))[0]
        cR = arms[1].sphere_centers(qpos[fr:fr + 1, 7:13].astype(np.float64))[0]
        centers = np.concatenate([cL, cR], axis=0)
        for ln in group:
            new_pairs = []
            for pair in ln["pairs"]:
                toucher, obj, imp, pos = pair[0], pair[1], pair[2], pair[3]
                sph_i = -1
                if toucher in links:
                    l = links.index(toucher)
                    cand = np.where(link_of_sphere == l)[0]
                    dd = np.linalg.norm(centers[cand] - np.asarray(pos), axis=1)
                    sph_i = int(within_idx[cand[int(np.argmin(dd))]])
                new_pairs.append([toucher, obj, imp, pos, sph_i])
            out.append({**ln, "frame": fr, "pairs": new_pairs})
    return out


def contact_windows(attributed, n_frames):
    """Contiguous per-object contact runs with peak impulse."""
    per_obj = {}
    for ln in attributed:
        for p in ln["pairs"]:
            per_obj.setdefault(p[1], []).append((ln["frame"], p[2]))
    windows = []
    for obj, hits in per_obj.items():
        hits.sort()
        t0, prev, peak = hits[0][0], hits[0][0], hits[0][1]
        for fr, imp in hits[1:]:
            if fr > prev + 3:
                windows.append({"object": obj, "t0": int(t0), "t1": int(prev),
                                "peak_impulse": round(float(peak), 4)})
                t0, peak = fr, 0.0
            prev, peak = fr, max(peak, imp)
        windows.append({"object": obj, "t0": int(t0), "t1": int(prev),
                        "peak_impulse": round(float(peak), 4)})
    return sorted(windows, key=lambda w: w["t0"])


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--margin", type=float, default=0.025)
    ap.add_argument("--dmax",   type=float, default=0.30)
    ap.add_argument("--topk",   type=int,   default=3)
    ap.add_argument("--knn",    type=int,   default=32)
    ap.add_argument("--window", type=int,   default=50)
    ap.add_argument("--force-thresh", type=float, default=1e-3,
                    help="impulse above which a contact counts for windows / "
                         "in_contact_window (zero-impulse proximity pairs are "
                         "recorded but do not define contact regions); 0 = all")
    ap.add_argument("--exclude-tags", default="target,container,furniture,skipped")
    ap.add_argument("--exclude-names", default="",
                    help="additional object NAMEs to mask (comma-separated)")
    ap.add_argument("--include-names", default="",
                    help="object NAMEs to force-include despite excluded tags")
    args = ap.parse_args()

    ds = args.dataset
    exclude_tags  = set(t for t in args.exclude_tags.split(",") if t)
    exclude_names = set(t for t in args.exclude_names.split(",") if t)
    include_names = set(t for t in args.include_names.split(",") if t)
    arms, links, link_of_sphere, within_idx, radii = load_robot(ds / "fk_basis.npz")
    n_links = len(links)
    print(f"robot: {len(radii)} spheres on {n_links} links: {links}")

    index = [json.loads(l) for l in open(ds / "index.jsonl")] \
        if (ds / "index.jsonl").exists() else None
    if index is None:
        index = []
        for sub in ("curobo_collision_free", "pi05_rollout", "curobo_collision", "data"):
            for f in sorted((ds / sub).glob("episode_*.hdf5")) if (ds / sub).is_dir() else []:
                with h5py.File(f) as h:
                    if "seed" not in h.attrs:
                        print(f"skip {f.name}: no seed attr (pre-triplet episode)")
                        continue
                    index.append({"file": str(f.relative_to(ds)),
                                  "seed": int(h.attrs["seed"]),
                                  "source": h.attrs["source"].decode()
                                  if isinstance(h.attrs["source"], bytes)
                                  else str(h.attrs["source"])})

    scenes = {}
    settings = {"margin": args.margin, "dmax": args.dmax, "topk": args.topk,
                "knn": args.knn, "window": args.window,
                "force_thresh": args.force_thresh,
                "exclude_tags": sorted(exclude_tags),
                "exclude_names": sorted(exclude_names),
                "include_names": sorted(include_names)}

    for ent in index:
        f = ds / ent["file"]
        if not f.exists():                      # legacy single-folder layout
            f = ds / "data" / Path(ent["file"]).name
        seed, source = ent["seed"], ent["source"]
        if seed not in scenes:
            scenes[seed] = load_scene(ds / "scene" / f"seed_{seed}",
                                      exclude_tags, exclude_names,
                                      include_names)[:2]
        samples, sample_obj = scenes[seed]

        with h5py.File(f, "a") as h:
            qpos = np.asarray(h["joint_action/vector"], dtype=np.float64)
            T = len(qpos)
            # per-episode scene versions: objects displaced mid-run get their
            # samples re-posed from the recorded object_poses track
            versions = scene_versions(h, samples, sample_obj)
            D, U, OBJ, SPH = label_episode(
                qpos, arms, link_of_sphere, within_idx, radii,
                versions, sample_obj, n_links,
                args.topk, args.knn, args.dmax, args.margin)

            # per-source layout first, legacy global metrics/ as fallback
            contacts_path = f.parent / "metrics" / f"seed{seed}_contacts.jsonl"
            if not contacts_path.exists():
                contacts_path = ds / "metrics" / f"seed{seed}_{source}_contacts.jsonl"
            attributed, windows, in_window = [], [], np.zeros(T, dtype=bool)
            if contacts_path.exists():
                lines = [json.loads(l) for l in open(contacts_path)]
                frames = [frame_of_contact(l, source, T) for l in lines]
                attributed = attribute_contacts(lines, frames, qpos, arms,
                                                link_of_sphere, within_idx, links)
                # Windows are FORCE-gated: PhysX reports zero-impulse proximity
                # pairs on most steps near clutter, which would saturate
                # in_contact_window (observed: 224/224 steps).  Real contacts
                # only define contact regions; resting pairs stay in the
                # attributed stream as boundary/calibration data.
                forced = [ln for ln in attributed
                          if ln.get("max_impulse", 0) > args.force_thresh]
                windows = contact_windows(forced, T)
                for ln in forced:
                    fr = ln["frame"]
                    in_window[max(0, fr - args.window):fr + args.window + 1] = True
                _attr_path = contacts_path.with_name(
                    contacts_path.name.replace("_contacts.jsonl",
                                               "_contacts_attributed.jsonl"))
                with open(_attr_path, "w") as fo:
                    for ln in attributed:
                        fo.write(json.dumps(ln) + "\n")

            for k in ("link_clearance", "link_margin", "in_contact_window"):
                if k in h:
                    del h[k]
            g = h.create_group("link_clearance")
            g.create_dataset("d", data=D)
            g.create_dataset("u", data=U)
            g.create_dataset("obj", data=OBJ)
            g.create_dataset("sphere_idx", data=SPH)
            g.create_dataset("names", data=np.array(links, dtype="S"))
            h.create_dataset("link_margin", data=(D[:, :, 0] < args.margin))
            h.create_dataset("in_contact_window", data=in_window)
            h.attrs["label_settings"] = json.dumps(settings)

            n_contact = len({ln["frame"] for ln in attributed})
            n_force   = len({ln["frame"] for ln in attributed
                             if ln.get("max_impulse", 0) > args.force_thresh})
            meta = {
                "file": ent["file"], "seed": seed, "source": source,
                "success": bool(h.attrs.get("success", False)),
                "episode_len": T,
                "contact_steps": n_contact, "force_steps": n_force,
                "contact_windows": windows,
                "min_clearance_m": round(float(D[:, :, 0].min()), 4),
                "scene_versions": len(versions),
                "expert_valid": (None if source != "planner_collision_aware"
                                 else n_force == 0),
                "label_settings": settings,
            }
        with open(f.with_suffix(".meta.json"), "w") as fo:
            json.dump(meta, fo, indent=1)
        print(f"{ent['file']:>20} [{source:>26}] T={T:4d} "
              f"min_clear={meta['min_clearance_m']*100:6.1f}cm "
              f"contact={n_contact:3d} force={n_force:3d} "
              f"windows={len(windows)} scene_versions={len(versions)}")

    print("done.")


if __name__ == "__main__":
    main()
