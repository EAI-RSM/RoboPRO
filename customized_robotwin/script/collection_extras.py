"""Pipeline glue for collect_data.py — built on the robo_tools library.

Two additions to the standard pipeline:
  * ENRICHMENT (default): every collected episode gets the per-frame
    `targeted_state` pose trace + a clean annotation written into its HDF5
    (additive groups; nothing existing changes).
  * VARIATIONS (toggles): when `negative` / `multimodal` are enabled for a task,
    each scene seed is re-run through the tested per-episode runners
    (scripts/run_targeted_episode.py, run_mode_episode.py) as ISOLATED
    subprocesses to produce sibling episodes — parallel annotations of one scene
    (clean good action + bad actions + alternative good actions). Negatives get a
    causal-graph annotation. Unsupported (task, feature) pairs are SKIPPED.
"""
import json
import os
import subprocess
import sys

import robo_tools as rt


# ----------------------------------------------------------------- enrichment (default)
def attach_enrichment(env):
    """Attach a fresh runtime that logs the per-frame state trace. The logger is
    idempotent, so calling this every episode in the reused-env loop is safe."""
    runtime = rt.TargetedRuntime()
    runtime.attach(env)
    return runtime


def write_clean_enrichment(hdf5_path, runtime, env, args, seed):
    """Write the trace + clean annotation into a freshly-merged episode HDF5."""
    if not os.path.exists(hdf5_path):
        return
    record = rt.clean_record(env, args, seed)
    rt.write_frame_state(hdf5_path, runtime.frame_state, record["progress_entities"])
    rt.write_labels_hdf5(hdf5_path, record)


# ----------------------------------------------------------------- variations (toggles)
def _write_causal(hdf5_path, causal):
    import h5py
    if not os.path.exists(hdf5_path):
        return
    with h5py.File(hdf5_path, "a") as f:
        g = f.require_group("causal")
        g.attrs["causal_json"] = json.dumps(causal)
        nodes = causal.get("nodes", {})
        g.attrs["intervention_type"] = str(nodes.get("intervention", {}).get("type"))
        g.attrs["outcome"] = str(nodes.get("outcome", {}).get("outcome"))


def _bench_subdir(task_name):
    entry = rt.task_entry(task_name) or rt.multimodal_entry(task_name)
    return entry["bench_subdir"] if entry else None


def collect_scene_variations(task_name, task_config, args, seed_list, repo_root):
    """Per scene seed: collect the planned sibling variations (one subprocess
    each), annotate negatives with the causal graph, append a scene-index row."""
    if not rt.variations_enabled(args):
        return
    summary = rt.supported_summary(task_name, args)
    subdir = _bench_subdir(task_name)
    if subdir is None:
        print(f"[variations] '{task_name}' not in any registry "
              f"(negative={summary['negative_supported']}, "
              f"multimodal={summary['multimodal_supported']}) -> skipping variations")
        return
    print(f"[variations] {task_name}: negative_supported={summary['negative_supported']} "
          f"multimodal_supported={summary['multimodal_supported']}")

    save_path = args["save_path"]
    var_root = os.path.join(save_path, "variations")
    os.makedirs(var_root, exist_ok=True)
    index_path = os.path.join(var_root, "scene_index.jsonl")
    runners = {
        "run_targeted_episode": os.path.join(repo_root, "scripts", "run_targeted_episode.py"),
        "run_mode_episode": os.path.join(repo_root, "scripts", "run_mode_episode.py"),
    }
    env = dict(os.environ)
    n = min(len(seed_list), int(args["episode_num"]))
    for idx in range(n):
        seed = int(seed_list[idx])
        variations = rt.plan_variations(task_name, args, seed)
        if not variations:
            continue
        scene_row = {"scene_id": f"{task_name}__seed{seed:05d}", "seed": seed,
                     "clean": f"data/episode{idx}.hdf5", "variations": []}
        for var in variations:
            out_dir = os.path.join(var_root, f"episode{idx}", var["name"])
            cmd = [sys.executable, "-u", runners[var["runner"]],
                   "--task-name", task_name, "--task-config", task_config,
                   "--bench-subdir", subdir, "--seed", str(seed), "--out-dir", out_dir]
            if var["kind"] == "negative":
                cmd += ["--role", "perturbed", "--ptype", var["ptype"],
                        "--params-json", json.dumps(var["params"]),
                        "--pair-id", f"{task_name}__seed{seed:05d}"]  # link causal counterfactual to the clean
            else:
                cmd += ["--mode", var["mode"]]
            print(f"[variations] scene{idx} (seed {seed}) -> {var['kind']}:{var['name']}")
            subprocess.run(cmd, env=env, check=False)

            epj = os.path.join(out_dir, "episode.json")
            rec = json.loads(open(epj).read()) if os.path.exists(epj) else {}
            hdf5 = os.path.join(out_dir, "data", "episode0.hdf5")
            if var["kind"] == "negative" and os.path.exists(hdf5):
                _write_causal(hdf5, rt.build_causal_annotation(rec))
            scene_row["variations"].append({
                "name": var["name"], "kind": var["kind"], "quality": var["quality"],
                "status": rec.get("status"), "outcome": rec.get("outcome"),
                "path": os.path.relpath(hdf5, save_path) if os.path.exists(hdf5) else None})
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(scene_row) + "\n")
    print(f"[variations] done -> {var_root}")
