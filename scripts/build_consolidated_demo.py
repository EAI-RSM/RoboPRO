#!/usr/bin/env python
"""Build ONE self-contained HTML consolidating both demo ideas as a TREE per scene:
the seed scene is the ROOT, and its sibling variations branch out of it — the clean
+ multimodal ALTERNATIVE GOOD actions (green branches) and the targeted-negative
BAD actions (red branches, with the causal graph). Bezier edges connect the root to
each variation; an interactive 3D end-effector overlay shows the multimodality.

Reads per-episode runner outputs (episode.json + data/episode0.hdf5 + video/
episode0.mp4); inlines plotly.js + all videos + a scene thumbnail. No server.

  python build_consolidated_demo.py --root <runs-dir> --out demo.html
"""
import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robo_tools" / "src"))
import robo_tools as rt

GOOD_ORDER = ["baseline", "high_arc", "swing_left", "swing_right"]
BAD_ORDER = ["shift_object", "shift_target", "shift_obstacle", "hide_obstacle"]
GOOD_COLORS = {"baseline": "#3fb950", "high_arc": "#1f77b4", "swing_left": "#2ca02c", "swing_right": "#a371f7"}
GOOD_DESC = {"baseline": "clean — the planner's default solution", "high_arc": "lift high, arc over",
             "swing_left": "swing wide one way", "swing_right": "swing wide the other way"}
SUCCESS = {"clean_success", "mode_success"}


def load(run_dir):
    epj = run_dir / "episode.json"
    if not epj.exists():
        return None
    rec = json.loads(epj.read_text())
    mp4 = run_dir / "video" / "episode0.mp4"
    h5 = run_dir / "data" / "episode0.hdf5"
    return {"record": rec, "outcome": rec.get("outcome"), "name": run_dir.name,
            "mp4": mp4 if mp4.exists() else None, "h5": h5 if h5.exists() else None}


def active_ee(h5):
    if h5 is None or not h5.exists():
        return None
    with h5py.File(h5, "r") as f:
        best, bl = None, -1.0
        for arm in ("left_endpose", "right_endpose"):
            if f"endpose/{arm}" in f:
                xyz = np.asarray(f[f"endpose/{arm}"][:])[:, :3]
                plen = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
                if plen > bl:
                    best, bl = xyz, plen
        return best


def vid_uri(mp4):
    return ("data:video/mp4;base64," + base64.b64encode(mp4.read_bytes()).decode("ascii")) if mp4 else None


def thumb_uri(mp4):
    """First frame of the baseline video as a base64 jpg — the static seed scene."""
    if mp4 is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
            out = t.name
        subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-vframes", "1", "-q:v", "3", out],
                       capture_output=True, check=False)
        data = Path(out).read_bytes()
        Path(out).unlink(missing_ok=True)
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def fig3d(task_dir):
    fig = go.Figure()
    for name in GOOD_ORDER:
        d = load(task_dir / name)
        if d is None or d["outcome"] not in SUCCESS:
            continue
        ee = active_ee(d["h5"])
        if ee is None:
            continue
        fig.add_trace(go.Scatter3d(x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode="lines",
                                   line=dict(color=GOOD_COLORS[name], width=6), name=name))
        fig.add_trace(go.Scatter3d(x=[ee[0, 0]], y=[ee[0, 1]], z=[ee[0, 2]], mode="markers",
                                   marker=dict(size=4, color=GOOD_COLORS[name]), showlegend=False))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=6, b=0),
                      scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z (m)",
                                 camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0))),
                      legend=dict(orientation="h", y=1.0, x=0), paper_bgcolor="rgba(0,0,0,0)")
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def good_node(name, d):
    if d is None:
        return ""
    oc = d["outcome"]
    cls = "ok" if oc in SUCCESS else "warn"
    vid = f'<video src="{vid_uri(d["mp4"])}" muted loop autoplay playsinline></video>' if d["mp4"] else '<div class="novid">—</div>'
    return (f'<div class="node good" data-kind="good" data-color="{GOOD_COLORS[name]}">{vid}'
            f'<div class="cap"><span class="dot" style="background:{GOOD_COLORS[name]}"></span><b>{name}</b>'
            f'<span class="badge {cls}">{oc}</span><div class="d">{GOOD_DESC.get(name,"")}</div></div></div>')


def bad_node(name, d):
    if d is None:
        return ""
    oc = d["outcome"]
    failed = oc not in ("clean_success",)
    cls = "fail" if failed else "warn"
    vid = f'<video src="{vid_uri(d["mp4"])}" muted loop autoplay playsinline></video>' if d["mp4"] else '<div class="novid">—</div>'
    c = rt.build_causal_annotation(d["record"])["nodes"]
    iv, mech, med = c["intervention"], c["mechanism"], c["mediators"]
    mag = iv.get("magnitude_cm")
    chain = (f'<div class="causal"><code>{iv.get("type")}</code>'
             f'{(" %.0fcm" % mag) if isinstance(mag,(int,float)) else ""} '
             f'&rarr; belief <code>{mech.get("planner_belief")}</code> '
             f'&rarr; <code>plan={med.get("plan_success")}</code> &rarr; <code>{oc}</code></div>')
    label = "FAILURE" if failed else "absorbed"
    return (f'<div class="node bad" data-kind="bad">{vid}'
            f'<div class="cap"><span class="dot" style="background:#f85149"></span><b>{name}</b>'
            f'<span class="badge {cls}">{label}</span>{chain}</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    tasks = sorted([p.name for p in root.iterdir() if p.is_dir()]) if root.exists() else []

    sections, n_good, n_bad = [], 0, 0
    for ti, task in enumerate(tasks):
        td = root / task
        instr = task.replace("_", " ").replace(" ks", "")
        base = load(td / "baseline")
        thumb = thumb_uri(base["mp4"]) if base else None
        root_vis = f'<img src="{thumb}"/>' if thumb else '<div class="novid">scene</div>'
        good = "".join(good_node(n, load(td / n)) for n in GOOD_ORDER if (td / n).exists())
        bad = "".join(bad_node(n, load(td / n)) for n in BAD_ORDER if (td / n).exists())
        n_good += sum(1 for n in GOOD_ORDER if (td / n).exists())
        n_bad += sum(1 for n in BAD_ORDER if (td / n).exists())
        sections.append(
            f'<section><h2>{instr}</h2><div class="sub">{task} · seed 0</div>'
            f'<div class="tree" id="tree-{ti}">'
            f'  <div class="rootwrap"><div class="node root">{root_vis}'
            f'    <div class="cap"><b>\U0001f331 seed scene</b><div class="d">one initial state</div></div></div></div>'
            f'  <div class="branches">'
            f'    <div class="grouplbl g">good actions — clean + multimodal alternatives</div>{good}'
            f'    <div class="grouplbl b">bad actions — causally-labelled failures</div>{bad}'
            f'  </div>'
            f'  <svg class="edges"></svg>'
            f'</div>'
            f'<div class="plot"><div class="plbl">end-effector trajectories of the good actions — '
            f'same grasp &amp; place, different paths (drag to rotate)</div>{fig3d(td)}</div>'
            f'</section>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RoboPRO — scene variation tree</title>
<script>{get_plotlyjs()}</script>
<style>
 :root{{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#2a3140;--txt:#e6edf3;--dim:#9aa7b4;
       --accent:#58a6ff;--ok:#3fb950;--fail:#f85149;--warn:#d29922}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--txt);
   font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
 .wrap{{max-width:1280px;margin:0 auto;padding:28px 22px 90px}}
 h1{{font-size:29px;margin:0 0 4px}} .lead{{color:var(--dim);font-size:16px;max-width:940px}}
 .thesis{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:20px 0}}
 .thesis b{{color:var(--accent)}}
 section{{margin:38px 0;border-top:1px solid var(--line);padding-top:18px}}
 h2{{font-size:21px;margin:0;text-transform:capitalize}} .sub{{color:var(--dim);font-size:13px;margin-bottom:14px}}
 /* tree */
 .tree{{position:relative;display:flex;gap:90px;align-items:center}}
 .rootwrap{{flex:0 0 230px;display:flex;align-items:center}}
 .branches{{flex:1;display:flex;flex-direction:column;gap:11px;position:relative;z-index:1}}
 .edges{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:0}}
 .grouplbl{{font-size:12px;font-weight:700;letter-spacing:.3px;margin:4px 0 -2px}}
 .grouplbl.g{{color:var(--ok)}} .grouplbl.b{{color:var(--fail)}}
 .node{{background:var(--panel);border:1px solid var(--line);border-radius:11px;overflow:hidden;
        display:flex;align-items:stretch;z-index:1;transition:border-color .15s,transform .15s}}
 .node:hover{{border-color:var(--accent);transform:translateX(2px)}}
 .node.root{{flex-direction:column;width:230px;border-color:#3a4658;background:linear-gradient(180deg,var(--panel),var(--panel2))}}
 .node.root img,.node.root .novid{{width:100%;aspect-ratio:4/3;object-fit:cover;background:#000}}
 .branches .node{{min-height:96px}}
 .branches .node video,.branches .node .novid{{width:128px;flex:0 0 128px;aspect-ratio:4/3;object-fit:cover;background:#000}}
 .novid{{display:flex;align-items:center;justify-content:center;color:var(--dim);font-size:12px}}
 .cap{{padding:9px 12px;display:flex;flex-direction:column;gap:4px;justify-content:center;font-size:13px}}
 .dot{{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:6px;vertical-align:middle}}
 .badge{{align-self:flex-start;font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px}}
 .badge.ok{{background:rgba(63,185,80,.16);color:var(--ok)}} .badge.warn{{background:rgba(210,153,34,.16);color:var(--warn)}}
 .badge.fail{{background:rgba(248,81,73,.15);color:var(--fail)}}
 .cap .d{{color:var(--dim);font-size:11.5px}}
 .causal{{color:var(--dim);font-size:11px;line-height:1.5}} .causal code{{color:#c9d1d9}}
 .plot{{margin-top:16px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 8px 4px}}
 .plbl{{color:var(--dim);font-size:12.5px;padding:2px 6px 4px}}
 @media(max-width:820px){{.tree{{flex-direction:column;gap:14px}}.edges{{display:none}}.rootwrap{{flex:none}}}}
</style></head><body><div class="wrap">
 <h1>RoboPRO — scene variation tree</h1>
 <div class="lead">One seed scene at the root; its labelled sibling actions branch out.</div>
 <div class="thesis">Each scene's <b>seed</b> is the root. The enriched collection pipeline grows the
 branches: the <b>multimodal</b> runtime re-parameterizes the planner into <b>alternative good actions</b>
 (green — same grasp &amp; place, different path), and the <b>targeted-negative</b> runtime desyncs the
 planner into <b>bad actions</b> (red — each a <b>causally-labelled failure</b>: intervention &rarr; belief
 &rarr; outcome). Every clip is a real rollout; the 3D plot shows the good actions' end-effector paths.</div>
 {''.join(sections)}
 <p style="color:var(--dim);font-size:12.5px;margin-top:44px;border-top:1px solid var(--line);padding-top:16px">
 RoboPRO · {len(tasks)} seed scenes · {n_good} good + {n_bad} bad branches · self-contained.</p>
</div>
<script>
function drawTree(tree){{
  const root=tree.querySelector('.node.root'), svg=tree.querySelector('.edges');
  const nodes=tree.querySelectorAll('.branches .node'); const tr=tree.getBoundingClientRect();
  svg.setAttribute('viewBox',`0 0 ${{tr.width}} ${{tr.height}}`); svg.innerHTML='';
  const rr=root.getBoundingClientRect(); const x1=rr.right-tr.left, y1=rr.top-tr.top+rr.height/2;
  nodes.forEach(n=>{{
    const nr=n.getBoundingClientRect(); const x2=nr.left-tr.left, y2=nr.top-tr.top+nr.height/2;
    const mx=(x1+x2)/2; const col=n.dataset.kind==='bad'?'#f85149':(n.dataset.color||'#3fb950');
    const p=document.createElementNS('http://www.w3.org/2000/svg','path');
    p.setAttribute('d',`M${{x1}},${{y1}} C${{mx}},${{y1}} ${{mx}},${{y2}} ${{x2}},${{y2}}`);
    p.setAttribute('stroke',col); p.setAttribute('fill','none'); p.setAttribute('stroke-width','2.5'); p.setAttribute('opacity','.65');
    const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
    dot.setAttribute('cx',x2); dot.setAttribute('cy',y2); dot.setAttribute('r','3.5'); dot.setAttribute('fill',col);
    svg.appendChild(p); svg.appendChild(dot);
  }});
}}
function drawAll(){{document.querySelectorAll('.tree').forEach(drawTree);}}
window.addEventListener('load',drawAll); window.addEventListener('resize',drawAll);
setTimeout(drawAll,400); setTimeout(drawAll,1200);
</script>
</body></html>"""
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB, {len(tasks)} scenes, "
          f"{n_good} good + {n_bad} bad branches)")


if __name__ == "__main__":
    main()
