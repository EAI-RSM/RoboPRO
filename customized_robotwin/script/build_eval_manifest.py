#!/usr/bin/env python3
"""Build a portable eval manifest from a completed training collection run.

Read-only over the dataset. Emits a few-KB JSON carrying the per-episode seeds
and scene hashes needed to replay the exact training initial states on another
machine, without shipping the dataset itself.

Usage:
    python script/build_eval_manifest.py <task_name> <task_config> \
        [--data-root ./data/bench_data] [--out <path>]

Run from customized_robotwin/ (or anywhere, if you pass absolute paths).
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCHEMA = 1


def _repo_roots():
    """Locate BENCH_ROOT / repo root, preferring set_env.sh's exports."""
    bench = os.environ.get("BENCH_ROOT")
    if bench:
        bench = Path(bench)
        return bench.parent, bench
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "benchmark").is_dir() and (p / "customized_robotwin").is_dir():
            return p, p / "benchmark"
    sys.exit("cannot locate repo root; run `source set_env.sh` first")


def _sha256_of_pyfiles(root: Path) -> str:
    """Fingerprint the env code: content hash over every .py under root."""
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return "sha256:" + h.hexdigest()


def _sha256_of_asset_listing(root: Path) -> str:
    """Fingerprint assets by name+size only -- hashing contents is far too slow."""
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                entries.append(f"{p.relative_to(root)}:{p.stat().st_size}")
            except OSError:
                continue
    for e in sorted(entries):
        h.update(e.encode())
    return "sha256:" + h.hexdigest()


def _episode_index(path: Path) -> int:
    m = re.search(r"episode[_]?(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _seeds_from_hdf5(data_dir: Path):
    """Primary source: the seed stamped into each surviving episode.

    Preferred over seed.txt because collect_data.py DELETES the hdf5 of any
    episode that ultimately failed its success check, while leaving that seed
    in seed.txt. Only episodes with a surviving hdf5 became training data.
    """
    if not data_dir.is_dir():
        return {}
    try:
        import h5py
    except ImportError:
        print("[warn] h5py unavailable; falling back to seed.txt", file=sys.stderr)
        return {}
    out = {}
    for f in sorted(data_dir.glob("episode*.hdf5"), key=_episode_index):
        idx = _episode_index(f)
        if idx < 0:
            continue
        try:
            with h5py.File(f, "r") as fh:
                if "seed" in fh.attrs:
                    out[idx] = int(fh.attrs["seed"])
                else:
                    print(f"[warn] {f.name}: no seed attr", file=sys.stderr)
        except Exception as e:
            print(f"[warn] {f.name}: {e}", file=sys.stderr)
    return out


def _seeds_from_seedfile(run_dir: Path):
    """Fallback: seed.txt, an ordered list where position == episode index."""
    sf = run_dir / "seed.txt"
    if not sf.exists():
        return {}
    seeds = [int(t) for t in sf.read_text().split() if t]
    return {i: s for i, s in enumerate(seeds)}


def _scene_hash(run_dir: Path, idx: int):
    p = run_dir / "scene" / f"episode{idx}" / "scene_hash.txt"
    if not p.exists():
        return None
    return p.read_text().strip().split()[0]


def _instruction_pool_sha(run_dir: Path, idx: int, desc_type: str):
    """Hash the instruction pool the trainer actually consumed.

    policy/pi05/scripts/process_data.py reads instructions/episode{i}.json
    and saves the WHOLE list under desc_type, so the pool -- not any single
    string -- is the unit of language fidelity.
    """
    p = run_dir / "instructions" / f"episode{idx}.json"
    if not p.exists():
        return None, None
    try:
        pool = json.loads(p.read_text()).get(desc_type)
    except Exception:
        return None, None
    if not pool:
        return None, None
    h = hashlib.sha1(json.dumps(sorted(pool), ensure_ascii=False).encode()).hexdigest()
    return "sha1:" + h, len(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name")
    ap.add_argument("task_config")
    ap.add_argument("--data-root", default="./data/bench_data",
                    help="collection save_path root (default matches bench configs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--desc-type", default="seen",
                    help="instruction pool key the checkpoint trained on")
    ap.add_argument("--tag", default="train")
    args = ap.parse_args()

    repo_root, bench_root = _repo_roots()
    run_dir = Path(args.data_root).expanduser().resolve() / args.task_name / args.task_config
    if not run_dir.is_dir():
        sys.exit(f"no collection run at {run_dir}")

    # --- config metadata -------------------------------------------------
    cfg_path = bench_root / "bench_task_config" / f"{args.task_config}.yml"
    if not cfg_path.exists():
        cfg_path = repo_root / "customized_robotwin" / "task_config" / f"{args.task_config}.yml"
    embodiment, language_num = None, None
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        embodiment = cfg.get("embodiment")
        language_num = cfg.get("language_num")
    else:
        print(f"[warn] task config not found: {args.task_config}.yml", file=sys.stderr)

    # --- seeds -----------------------------------------------------------
    seeds = _seeds_from_hdf5(run_dir / "data")
    source = "hdf5_attrs"
    if not seeds:
        seeds = _seeds_from_seedfile(run_dir)
        source = "seed.txt"
        if seeds:
            print("[warn] using seed.txt: it may include seeds whose episode was "
                  "later deleted for failing its success check", file=sys.stderr)
    if not seeds:
        sys.exit(f"no seeds recoverable from {run_dir} (no episode*.hdf5, no seed.txt)")

    # --- per-episode records ---------------------------------------------
    episodes, missing_hash = [], 0
    for idx in sorted(seeds):
        h = _scene_hash(run_dir, idx)
        if h is None:
            missing_hash += 1
        episodes.append({"episode": idx, "seed": seeds[idx], "scene_hash": h})

    pool_sha, pool_size = _instruction_pool_sha(run_dir, episodes[0]["episode"], args.desc_type)

    manifest = {
        "schema": SCHEMA,
        "kind": "train_replay",
        "task_name": args.task_name,
        "task_config": args.task_config,
        "embodiment": embodiment,
        "seed_source": source,
        "env_fingerprint": _sha256_of_pyfiles(bench_root / "bench_envs"),
        "assets_fingerprint": _sha256_of_asset_listing(bench_root / "assets"),
        "instruction": {
            "source": "generated",
            "instruction_type": args.desc_type,
            "language_num": language_num,
            "pool_size": pool_size,
            "pool_sha": pool_sha,
        },
        "episodes": episodes,
    }

    out = Path(args.out) if args.out else (
        bench_root / "eval_manifests" / args.task_name / args.task_config / f"{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {out}  ({out.stat().st_size} B)")
    print(f"  episodes    : {len(episodes)}  (seeds from {source})")
    print(f"  seed range  : {min(seeds.values())}..{max(seeds.values())}")
    print(f"  scene hashes: {len(episodes) - missing_hash}/{len(episodes)}")
    if missing_hash:
        print(f"  [warn] {missing_hash} episode(s) have no scene_hash -- those cannot "
              f"be verified on the eval machine, only replayed")
    if pool_sha is None:
        print("  [warn] no instruction pool found; language fidelity unverifiable")


if __name__ == "__main__":
    main()
