"""Visualize negative-data demo episodes with their annotations as an HTML gallery.

Scans a data root for `episode.json` + rollout video, computes the annotation
(WHAT flag-set / WHY / quality, per the demo.ipynb notebook spec) offline from the
recorded signals, and writes a self-contained `index.html` that embeds each demo
video next to its annotation, grouping twins (baseline vs perturbed) side by side.

No sim imports. Run (from the repo root):
    python visualize_negative_data.py \
        --root <collection-dir> [--out path.html]

Serving / viewing the gallery
-----------------------------
index.html references the rollout videos by *relative* path, so it must be
served (or opened) with its --root as the working/server directory.

Local machine:
    open file://<root>/index.html          # macOS; or just double-click it

Remote node (this repo runs on the cluster, e.g. london-2-embody-ai-node-8):
    # 1) on the node, start a static server rooted at the demo dir:
    cd <collection-dir>
    python -m http.server 8011 --bind 0.0.0.0    # all interfaces, for the tunnel
    # 2) on your laptop, forward the port over SSH:
    ssh -L 8011:localhost:8011 <user>@<node-host>
    # 3) browse to:
    http://localhost:8011/
Pick any free port if 8011 is taken. Stop the server with Ctrl-C (or kill <pid>
if backgrounded). --bind 0.0.0.0 is only needed for the tunnel/direct-IP case;
omit it (defaults to localhost) when serving and viewing on the same machine.
"""
import argparse
import html
import json
import os
import sys
from pathlib import Path

# robo_negative is an installed editable package providing both the annotation and
# the offline metrics (sapien is imported lazily there, so no sim is needed here).
try:
    import robo_negative  # type: ignore
    labels = metrics = robo_negative
except Exception:  # noqa: BLE001
    labels = metrics = None

# The annotation (WHAT flag-set / WHY / quality) comes from labels.annotate;
# these are just display colors.
QUALITY_COLOR = {"good": "#2e7d32", "suboptimal": "#f9a825", "bad": "#c62828", "?": "#607d8b"}
WHAT_COLOR = {"ok": "#2e7d32", "absorbed": "#2e7d32", "grasp_failure": "#c62828",
              "placement_failure": "#ad1457", "collision": "#e65100",
              "planning_failure": "#6a1b9a", "?": "#607d8b"}
SCENE_ORDER = {"clean": 0, "d6": 1, "d8": 2, "d10": 3, "d12": 4, "d15": 5}
CASE_ORDER = {"baseline": 0, "shift_object": 1, "shift_target": 2, "shift_obstacle": 1}

# Draws each per-frame progress curve and animates a marker in lockstep with the
# video: the playhead maps currentTime/duration -> frame index, so it tracks the
# real playback speed. The played portion is drawn solid, the rest light.
CHART_JS = r"""
function curIdx(id,n){
  const vid=document.querySelector('video[data-ep="'+id+'"]');
  if(vid&&vid.duration&&isFinite(vid.duration)) return Math.min(n-1,Math.max(0,Math.round(vid.currentTime/vid.duration*(n-1))));
  return n-1;
}
function drawSeries(cv, arr, idx, o){
  if(!cv||!arr||!arr.length) return;
  const ctx=cv.getContext('2d'), W=cv.width,H=cv.height,padL=34,padR=10,padT=10,padB=14, n=arr.length;
  const X=i=>padL+(i/Math.max(1,n-1))*(W-padL-padR);
  const Y=v=>padT+(1-Math.max(0,Math.min(o.vmax,v))/o.vmax)*(H-padT-padB);
  ctx.clearRect(0,0,W,H);
  ctx.font='10px sans-serif'; ctx.textAlign='right'; ctx.textBaseline='middle';
  o.ticks.forEach(t=>{ const y=Y(t.v);
    ctx.strokeStyle=t.label?'#dcdcdc':'#f3f3f3'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(W-padR,y); ctx.stroke();
    if(t.label){ ctx.fillStyle='#999'; ctx.fillText(t.label, padL-4, y); }
  });
  ctx.strokeStyle=o.light; ctx.lineWidth=1.5; ctx.beginPath();
  for(let i=0;i<n;i++){const x=X(i),y=Y(arr[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);} ctx.stroke();
  ctx.strokeStyle=o.color; ctx.lineWidth=2.5; ctx.beginPath();
  for(let i=0;i<=idx;i++){const x=X(i),y=Y(arr[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);} ctx.stroke();
  const cx=X(idx),cy=Y(arr[idx]);
  ctx.strokeStyle='rgba(120,120,120,.3)'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(cx,padT); ctx.lineTo(cx,H-padB); ctx.stroke();
  ctx.fillStyle=o.color; ctx.beginPath(); ctx.arc(cx,cy,4,0,2*Math.PI); ctx.fill();
  ctx.font='bold 11px sans-serif'; ctx.textBaseline='bottom'; ctx.textAlign=cx>W-46?'right':'left';
  ctx.fillText(o.fmt(arr[idx]), cx+(cx>W-46?-6:6), cy-3);
}
const PROG={vmax:100,light:'#cfd8dc',color:'#1565c0',fmt:v=>v.toFixed(0)+'%',
  ticks:[{v:0,label:'0%'},{v:25},{v:50,label:'50%'},{v:75},{v:100,label:'100%'}]};
const SAFETY={vmax:1,light:'#ffccbc',color:'#e65100',fmt:v=>v.toFixed(2),
  ticks:[{v:0,label:'0'},{v:0.25},{v:0.5,label:'0.5'},{v:0.75},{v:1,label:'1.0'}]};
function draw(id){
  const d=DATA[id]; if(!d) return;
  if(d.p&&d.p.length){ const i=curIdx(id,d.p.length);
    drawSeries(document.getElementById(id+'_c'), d.p, i, PROG);
    const e=document.getElementById(id+'_v');
    if(e){ const dc=d.d?('  ·  '+d.d[i].toFixed(1)+' cm'):''; e.textContent='progress '+d.p[i].toFixed(1)+'%'+dc+'  ·  frame '+i+'/'+(d.p.length-1); }
  }
  if(d.s&&d.s.length){ const i=curIdx(id,d.s.length);
    drawSeries(document.getElementById(id+'_s'), d.s, i, SAFETY);
    const e=document.getElementById(id+'_sv');
    if(e) e.textContent='safety '+d.s[i].toFixed(2)+' / 1.0  ·  frame '+i+'/'+(d.s.length-1);
  }
}
function attach(id){
  const vid=document.querySelector('video[data-ep="'+id+'"]'); draw(id); if(!vid) return;
  let raf=null; const loop=()=>{draw(id); if(!vid.paused&&!vid.ended) raf=requestAnimationFrame(loop);};
  vid.addEventListener('loadedmetadata',()=>draw(id));
  vid.addEventListener('play',()=>{ if(raf)cancelAnimationFrame(raf); loop(); });
  vid.addEventListener('pause',()=>{ if(raf)cancelAnimationFrame(raf); draw(id); });
  vid.addEventListener('seeked',()=>draw(id));
  vid.addEventListener('timeupdate',()=>{ if(vid.paused) draw(id); });
  vid.addEventListener('ended',()=>draw(id));
}
Object.keys(DATA).forEach(attach);
"""


def scene_key(cfg: str) -> str:
    if not cfg:
        return "?"
    tail = cfg.replace("bench_demo_office_", "")
    return tail if tail.startswith("d") and tail[1:].isdigit() else ("clean" if "clean" in tail else tail)


def derive(ep: dict) -> dict:
    ann = labels.annotate(ep)
    what = ann["what"] or (["absorbed"] if ann["why"] != "none" else ["ok"])
    cmd = ep.get("commanded") or {}
    ach = ep.get("achieved") or {}
    sig = ep.get("signals") or {}
    cmetrics = sig.get("collision_metrics") or {}
    return {
        "what": what,                       # flag-set
        "why": ann["why"],
        "why_role": ann["why_role"],
        "axis": ann["why_axis"],
        "magnitude_cm": cmd.get("magnitude_cm"),
        "achieved_cm": ach.get("achieved_magnitude_cm"),
        "quality": ann["quality"],
        "task_success": ann["task_success"],
        "object_moved_cm": sig.get("object_moved_cm"),
        "place_error_cm": sig.get("place_error_cm"),
        "perturbed_actor": ep.get("perturbed_actor"),
        "collision_bodies": cmetrics.get("robot_to_static_object_names") or [],
        "contact_frames": ann["what_frames"].get("collision", []),
        "instruction": ep.get("instruction"),
        "pair_id": ep.get("pair_id"),
    }


def badge(text, color):
    return (f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:10px;'
            f'font-size:12px;font-weight:600">{html.escape(str(text))}</span>')


def num(v, suffix="", n=2):
    return f"{v:.{n}f}{suffix}" if isinstance(v, (int, float)) else "—"


def card_html(root: Path, ep_dir: Path, ep: dict, d: dict, ep_id: str, cd) -> str:
    vids = sorted((ep_dir / "video").glob("*.mp4")) if (ep_dir / "video").is_dir() else []
    vid_rel = os.path.relpath(vids[0], root) if vids else None
    video = (f'<video controls loop muted width="340" preload="metadata" data-ep="{ep_id}">'
             f'<source src="{html.escape(vid_rel)}" type="video/mp4"></video>'
             if vid_rel else '<div class="novid">no video</div>')
    chart = ''
    if cd and cd.get("p"):
        chart += (f'<canvas class="chart" id="{ep_id}_c" width="340" height="110"></canvas>'
                  f'<div class="pval" id="{ep_id}_v"></div>')
    if cd and cd.get("s"):
        chart += (f'<canvas class="chart" id="{ep_id}_s" width="340" height="110"></canvas>'
                  f'<div class="pval" id="{ep_id}_sv"></div>')

    role = ep.get("pair_role") or "?"
    title = (f'{role} · ' + (ep.get("perturbation_type") or "no perturbation"))
    rows = [
        ("WHAT", " ".join(badge(w, WHAT_COLOR.get(w, "#607d8b")) for w in d["what"])),
        ("WHY", f'{badge(d["why"], "#1565c0")} '
                + (f'<small>{d["why_role"]} · {d["axis"] or "?"}</small>' if d["why"] != "none" else "")),
        ("quality", badge(d["quality"], QUALITY_COLOR.get(d["quality"], "#607d8b"))),
        ("task_success", "✅" if d["task_success"] else ("❌" if d["task_success"] is False else "—")),
    ]
    if d["why"] != "none":
        rows.append(("error magnitude", f'{num(d["magnitude_cm"]," cm")} '
                     f'<small>(achieved {num(d["achieved_cm"]," cm")})</small>'))
        if d["perturbed_actor"]:
            rows.append(("faulted entity", f'<code>{html.escape(str(d["perturbed_actor"]))}</code>'))
    rows.append(("object_moved", num(d["object_moved_cm"], " cm")))
    rows.append(("place_error", num(d["place_error_cm"], " cm")))
    if d["contact_frames"]:
        frames = d["contact_frames"]
        bodies = ", ".join(html.escape(b) for b in d["collision_bodies"]) or "—"
        rows.append(("collision", f'{badge(len(frames), "#e65100")} frame(s) '
                     f'<small>@ {frames[:8]}{"…" if len(frames) > 8 else ""} · {bodies}</small>'))
    body = "".join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rows)
    return (f'<div class="card"><div class="cap">{html.escape(title)}</div>{video}'
            f'{chart}<table>{body}</table></div>')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent  # repo root (this script lives here)
    ap.add_argument("--root", default=str(here / "targetted_dataset"),
                    help="collection dir to scan for episode.json + rollout videos")
    ap.add_argument("--out", default=None, help="output html (default: <root>/index.html)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out) if args.out else root / "index.html"
    eps = []
    chart_data = {}  # ep_id -> {p:[per-frame], m:label} for the synced graphs
    for i, ej in enumerate(sorted(root.rglob("episode.json"))):
        try:
            ep = json.loads(ej.read_text(encoding="utf-8"))
        except Exception:
            continue
        ep_id = f"ep{i}"
        cd = {}
        if metrics is not None:  # all metrics computed OFFLINE from logged raw state
            hs = sorted((ej.parent / "data").glob("*.hdf5")) if (ej.parent / "data").is_dir() else []
            try:
                st = metrics.load_frame_state(hs[0]) if hs else None
            except Exception:  # noqa: BLE001
                st = None
            if st:
                prog = metrics.compute_progress(ep, st)
                if prog:
                    cd.update(p=prog["values"], d=prog.get("distance_cm"), m=prog["metric"])
                safety = metrics.compute_safety(ep, st)
                if safety:
                    cd.update(s=safety["values"], cutoff=safety["cutoff_cm"])
        if cd:
            chart_data[ep_id] = cd
        eps.append((ej.parent, ep, derive(ep), ep_id, cd or None))
    if not eps:
        raise SystemExit(f"no episode.json found under {root}")

    # group: scene -> seed -> [(case_order, dir, ep, derived, ep_id, progress)]
    groups: dict = {}
    for ep_dir, ep, d, ep_id, prog in eps:
        sk = scene_key(ep.get("task_config", ""))
        seed = ep.get("scene_seed")
        case = ep.get("perturbation_type") or "baseline"
        groups.setdefault(sk, {}).setdefault(seed, []).append(
            (CASE_ORDER.get(case, 9), ep_dir, ep, d, ep_id, prog))

    # tallies
    q_tally, ww_tally = {}, {}
    for _, _, d, _, _ in eps:
        q_tally[d["quality"]] = q_tally.get(d["quality"], 0) + 1
        key = f'{d["why"]} → {"+".join(d["what"])}'
        ww_tally[key] = ww_tally.get(key, 0) + 1

    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#fafafa;color:#222}
    h1{margin-bottom:4px} h2{margin-top:34px;border-bottom:2px solid #ddd;padding-bottom:4px}
    .seedrow{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0}
    .card{background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:10px;width:360px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
    .cap{font-weight:700;margin-bottom:6px;font-size:14px}
    .novid{width:340px;height:200px;display:flex;align-items:center;justify-content:center;background:#eee;color:#888;border-radius:6px}
    table{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}
    td{padding:3px 6px;border-top:1px solid #f0f0f0;vertical-align:top}
    td:first-child{color:#666;width:120px;font-weight:600}
    small{color:#888} code{background:#f3f3f3;padding:1px 4px;border-radius:4px}
    .legend{background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:12px 16px;display:inline-block}
    .legend span{margin-right:6px}
    canvas.chart{display:block;margin-top:8px;background:#fcfcfc;border:1px solid #eee;border-radius:6px}
    .pval{font-size:12px;color:#444;margin-top:2px;font-variant-numeric:tabular-nums}
    """
    parts = [f'<!doctype html><html><head><meta charset="utf-8"><title>Negative-data demo</title>',
             f'<style>{css}</style></head><body>',
             '<h1>Negative-data demo — annotated</h1>',
             '<p>Twin-pair desync episodes. WHAT (symptom, from sim) · WHY (cause, known by '
             'construction) · quality — derived from the recorded labels (annotation '
             'spec lives in the demo.ipynb notebook). Under each video, two graphs (computed '
             'offline from logged per-frame state, animated in sync with playback): '
             '<b>task progress</b> 0–100% (object→destination), and <b>safety</b> 0–1 '
             '(1 = clear; dips as the end-effector nears non-target objects or moves fast).</p>',
             '<div class="legend"><b>quality:</b> ',
             " ".join(badge(q, c) for q, c in QUALITY_COLOR.items() if q != "?"),
             '<br><br><b>tally:</b> ',
             " · ".join(f"{badge(q, QUALITY_COLOR.get(q,'#607d8b'))} {n}" for q, n in sorted(q_tally.items())),
             '<br><br><b>why → what:</b> ',
             " · ".join(f"{html.escape(k)} ×{v}" for k, v in sorted(ww_tally.items())),
             '</div>']

    for sk in sorted(groups, key=lambda s: SCENE_ORDER.get(s, 99)):
        parts.append(f'<h2>scene: {html.escape(sk)}</h2>')
        for seed in sorted(groups[sk]):
            cards = sorted(groups[sk][seed], key=lambda x: x[0])
            parts.append(f'<div><b>seed {seed}</b> <small>(twin pair)</small></div>')
            parts.append('<div class="seedrow">')
            for _, ep_dir, ep, d, ep_id, _cd in cards:
                parts.append(card_html(root, ep_dir, ep, d, ep_id, chart_data.get(ep_id)))
            parts.append('</div>')

    parts.append('<script>const DATA=' + json.dumps(chart_data) + ';' + CHART_JS + '</script>')
    parts.append('</body></html>')
    out.write_text("".join(parts), encoding="utf-8")
    print(f"[viz] {len(eps)} episodes across {len(groups)} scenes -> {out}")
    print(f"[viz] quality tally: {q_tally}")
    # Videos are referenced relatively, so serve with --root as the server dir.
    # Local: open file://...  Remote: bind 0.0.0.0 + SSH -L tunnel (see module docstring).
    print(f"[viz] view (local):  open file://{out}")
    print(f"[viz] view (remote): cd {root} && python -m http.server 8011 --bind 0.0.0.0"
          f"  then  ssh -L 8011:localhost:8011 <user>@<node>  ->  http://localhost:8011/")


if __name__ == "__main__":
    main()
