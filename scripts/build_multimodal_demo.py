#!/usr/bin/env python
"""Build a SINGLE self-contained HTML demo of the multimodal PoC.

For each task it embeds:
  * an INTERACTIVE 3D plot (plotly) of the per-mode end-effector trajectories
    (rotate/zoom) — the multimodality, in 3D;
  * the rollout VIDEOS for each mode, inlined as base64 (no server, no files).
plotly.js is inlined once. Open the result straight from disk.

  python build_multimodal_demo.py   # -> ../multimodal_data/multimodal_demo.html
"""
import base64
import json
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "multimodal_data" / "runs"
OUT = REPO / "multimodal_data" / "multimodal_demo.html"

import sys
sys.path.insert(0, str(HERE))
from robo_tools.multimodal import MODES, MODE_ORDER

COLORS = {"direct": "#555555", "high_arc": "#1f77b4", "swing_left": "#2ca02c", "swing_right": "#d62728"}
OUTCOME_CLS = {"mode_success": "ok", "mode_unrealized": "warn", "mode_failed": "fail"}


def active_ee(f):
    best, best_len = None, -1.0
    for arm in ("left_endpose", "right_endpose"):
        k = f"endpose/{arm}"
        if k in f:
            xyz = np.asarray(f[k][:])[:, :3]
            plen = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
            if plen > best_len:
                best, best_len = xyz, plen
    return best


def target_path(f):
    if "targeted_state" not in f:
        return None
    g = f["targeted_state"]
    names = [n.decode() if isinstance(n, bytes) else n for n in g["actor_names"][:]]
    ent = json.loads(g.attrs["entities_json"]) if "entities_json" in g.attrs else {}
    t = ent.get("target")
    return g["actor_pos"][:, names.index(t), :] if t in names else None


def datauri(p):
    return "data:video/mp4;base64," + base64.b64encode(Path(p).read_bytes()).decode("ascii")


def fig3d(task_dir, title):
    fig = go.Figure()
    grasp = place = None
    for mode in MODE_ORDER:
        hp = task_dir / mode / "data" / "episode0.hdf5"
        if not hp.exists():
            continue
        with h5py.File(hp, "r") as f:
            ee = active_ee(f)
            tp = target_path(f)
            outcome = f["multimodal"].attrs.get("outcome", "?") if "multimodal" in f else "?"
        if ee is None:
            continue
        c = COLORS[mode]
        fig.add_trace(go.Scatter3d(x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode="lines",
                                   line=dict(color=c, width=6), name=f"{mode} ({outcome})"))
        if tp is not None:
            fig.add_trace(go.Scatter3d(x=tp[:, 0], y=tp[:, 1], z=tp[:, 2], mode="lines",
                                       line=dict(color=c, width=2, dash="dot"),
                                       opacity=0.5, showlegend=False))
        grasp = ee[0] if grasp is None else grasp
        place = ee[-1]
    if grasp is not None:
        fig.add_trace(go.Scatter3d(x=[grasp[0]], y=[grasp[1]], z=[grasp[2]], mode="markers",
                                   marker=dict(size=6, color="black", symbol="square"), name="grasp"))
    if place is not None:
        fig.add_trace(go.Scatter3d(x=[place[0]], y=[place[1]], z=[place[2]], mode="markers",
                                   marker=dict(size=7, color="black", symbol="diamond"), name="place"))
    fig.update_layout(
        height=460, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0))),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        paper_bgcolor="rgba(0,0,0,0)")
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": True, "responsive": True})


def video_cells(task_dir):
    cells = []
    for mode in MODE_ORDER:
        d = task_dir / mode
        epj = d / "episode.json"
        mp4 = d / "video" / "episode0.mp4"
        if not epj.exists():
            continue
        ep = json.loads(epj.read_text())
        oc = ep.get("outcome", "?")
        cls = OUTCOME_CLS.get(oc, "warn")
        vid = (f'<video src="{datauri(mp4)}" muted loop autoplay playsinline></video>'
               if mp4.exists() else '<div class="novid">no video</div>')
        cells.append(
            f'<div class="cell"><div class="mt"><span class="dot" style="background:{COLORS[mode]}"></span>'
            f'{mode}</div>{vid}<span class="badge {cls}">{oc}</span>'
            f'<div class="desc">{MODES[mode]["desc"]}</div></div>')
    return "".join(cells)


def main():
    tasks = sorted([p.name for p in RUNS.iterdir() if p.is_dir()]) if RUNS.exists() else []
    sections = []
    for task in tasks:
        td = RUNS / task
        instr = task.replace("_", " ").replace(" ks", "")
        sections.append(
            f'<section><h2>{instr}</h2><div class="sub">{task} · one scene, 4 planner modes</div>'
            f'<div class="plot">{fig3d(td, task)}</div>'
            f'<div class="row">{video_cells(td)}</div></section>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Multimodal Actions — RoboPRO</title>
<script>{get_plotlyjs()}</script>
<style>
 :root{{--bg:#0d1117;--panel:#161b22;--line:#2a3140;--txt:#e6edf3;--dim:#9aa7b4;--accent:#58a6ff;
       --ok:#3fb950;--fail:#f85149;--warn:#d29922;}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--txt);
   font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
 .wrap{{max-width:1280px;margin:0 auto;padding:28px 22px 80px}}
 h1{{font-size:29px;margin:0 0 4px}} .lead{{color:var(--dim);font-size:16px;max-width:900px}}
 .thesis{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:20px 0}}
 .thesis b{{color:var(--accent)}}
 section{{margin:34px 0;border-top:1px solid var(--line);padding-top:18px}}
 h2{{font-size:20px;margin:0;text-transform:capitalize}} .sub{{color:var(--dim);font-size:13px;margin-bottom:8px}}
 .plot{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px;margin-bottom:14px}}
 .row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
 @media(max-width:900px){{.row{{grid-template-columns:repeat(2,1fr)}}}}
 .cell{{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;
        display:flex;flex-direction:column}}
 .cell video{{width:100%;aspect-ratio:4/3;background:#000;object-fit:cover}}
 .cell .novid{{aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;color:var(--dim)}}
 .mt{{padding:8px 10px 0;font-weight:600;font-size:13.5px}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:middle}}
 .badge{{align-self:flex-start;margin:7px 0 0 10px;font-size:11.5px;font-weight:700;padding:2px 8px;border-radius:999px}}
 .badge.ok{{background:rgba(63,185,80,.16);color:var(--ok)}} .badge.warn{{background:rgba(210,153,34,.16);color:var(--warn)}}
 .badge.fail{{background:rgba(248,81,73,.15);color:var(--fail)}}
 .desc{{color:var(--dim);font-size:12px;padding:6px 10px 10px}}
 a{{color:var(--accent)}}
</style></head><body><div class="wrap">
 <h1>Multimodal Actions</h1>
 <div class="lead">Many distinct behaviors for the same task on the same scene — from a parameterized classical planner.</div>
 <div class="thesis">RoboPRO collects ~1 demo per scene, so conditioned on the scene the action distribution looks
 <b>unimodal</b>. Here the planner is re-parameterized into several <b>modes</b>: each injects a different
 transport <b>waypoint</b> while keeping the grasp and place <b>endpoints fixed</b>, so the task still succeeds but
 the arm takes a visibly different path. Same scene → multiple valid solutions. Rotate the 3D plots; the videos are
 the actual rollouts. A mode the planner can't reach is honestly labelled <b>mode_unrealized</b>.</div>
 {''.join(sections)}
 <p style="color:var(--dim);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px">
 RoboPRO · multimodal_actions PoC · trajectories are the recorded end-effector <code>endpose</code>; the
 dotted lines are the manipulated object. Self-contained — plotly.js + videos embedded.</p>
</div></body></html>"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {len(tasks)} tasks)")


if __name__ == "__main__":
    main()
