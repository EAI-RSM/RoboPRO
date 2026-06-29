#!/usr/bin/env python
"""Build the targeted-negative DEMO web bundle from a collected run tree.

Reads  enriched_negative_demo/runs/<task>/{baseline,shift_object,shift_target,
shift_obstacle,hide_obstacle}/{episode.json, video/episode0.mp4}
Writes web/videos/<task>__<label>.mp4  and  web/manifest.json
(then serve web/ — see serve note). No simulator, pure projection of labels.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RUNS = REPO / "enriched_negative_demo" / "runs"
WEB = ROOT / "web"
VID = WEB / "videos"

PTYPES = ["shift_object", "shift_target", "shift_obstacle", "hide_obstacle"]
PTYPE_META = {
    "shift_object": {"title": "Object mislocalized",
                     "cause": "The target object is moved a few cm AFTER its grasp is planned — "
                              "the planner closes the gripper on stale geometry.",
                     "icon": "✋"},
    "shift_target": {"title": "Destination mislocalized",
                     "cause": "The destination is moved before placement — the planner places the "
                              "object at where the target used to be.",
                     "icon": "\U0001f3af"},
    "shift_obstacle": {"title": "Obstacle mislocalized",
                       "cause": "An obstacle sits in the path, but the planner believes it is "
                                "somewhere else — it plans straight through the real one.",
                       "icon": "\U0001f4e6"},
    "hide_obstacle": {"title": "Obstacle undetected",
                      "cause": "An obstacle sits in the path, but is hidden from the planner — it "
                               "plans as if the path were clear.",
                      "icon": "\U0001f47b"},
}


def load(task, label):
    d = RUNS / task / label
    epj = d / "episode.json"
    mp4 = d / "video" / "episode0.mp4"
    if not epj.exists():
        return None
    ep = json.loads(epj.read_text())
    out = {"label": label, "status": ep.get("status"), "outcome": ep.get("outcome"),
           "n_frames": ep.get("n_frames"), "instruction": ep.get("instruction"),
           "perceptual_failure_class": ep.get("perceptual_failure_class"),
           "obstacle_source": ep.get("obstacle_source"),
           "commanded": ep.get("commanded"), "video": None}
    if mp4.exists():
        name = f"{task}__{label}.mp4"
        shutil.copy(mp4, VID / name)
        out["video"] = f"videos/{name}"
    return out


def main():
    VID.mkdir(parents=True, exist_ok=True)
    tasks = sorted([p.name for p in RUNS.iterdir() if p.is_dir()]) if RUNS.exists() else []
    manifest = {"tasks": [], "ptype_meta": PTYPE_META, "ptype_order": PTYPES}
    for task in tasks:
        base = load(task, "baseline")
        perts = [load(task, pt) for pt in PTYPES]
        perts = [p for p in perts if p is not None]
        if base is None and not perts:
            continue
        instr = (base or perts[0]).get("instruction") or task.replace("_", " ").replace(" ks", "")
        manifest["tasks"].append({
            "task": task, "instruction": instr,
            "baseline": base, "perturbations": perts,
        })
    (WEB / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_vid = len(list(VID.glob("*.mp4")))
    print(f"built {len(manifest['tasks'])} tasks, {n_vid} videos -> {WEB/'manifest.json'}")
    # quick outcome table
    for t in manifest["tasks"]:
        b = t["baseline"]
        print(f"  {t['task']:30s} baseline={b['outcome'] if b else '-':>14}  "
              + "  ".join(f"{p['label'].split('_')[0][:5]}:{str(p['outcome'])[:14]}" for p in t["perturbations"]))


if __name__ == "__main__":
    main()
