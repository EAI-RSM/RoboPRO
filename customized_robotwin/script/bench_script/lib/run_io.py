"""Run-directory, logging, and timing helpers."""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

CLEARANCE_RESULTS_DIR = (
    Path(__file__).resolve().parents[4]
    / "scripts" / "validation" / "results" / "clearance_metric_3d"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_out_dir(args):
    """Resolve this run's output directory. main() has already rewritten
    args.out_dir to <out-dir>/<type>/<timestamp> for a live run (or left it as the
    user-supplied folder for --plot-only), so this is now a plain passthrough that
    every downstream helper (run() + the analysis pass) shares to hit the SAME
    folder. Shadows the analyze_natural_visibility helper of the same name, which
    used a `_rollout` sibling-suffix scheme instead."""
    return Path(args.out_dir)


class _Tee:
    """Fan writes out to several streams at once, so a rollout's stdout/stderr lands
    in its per-episode log file while STILL showing up live on the console."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        # Best-effort per stream: a rollout prints an emoji line ("Video is saved...")
        # from images_to_video, and if ANY sink (e.g. an ASCII-encoded log file, or a
        # non-UTF-8 console) can't encode it, a raw st.write would raise mid-rollout --
        # which previously propagated out of merge_pkl_to_hdf5_video and skipped the
        # success/fail bucketing entirely. Never let a logging write abort the rollout.
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                try:
                    st.write(s.encode("ascii", "replace").decode("ascii"))
                except Exception:
                    pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def _prune_empty_topdirs(out_dir):
    """_bucket_rollout_artifacts moves every episode's hdf5/mp4 into success/ or
    fail/, leaving the top-level data/ and video/ folders (created fresh by the
    save_data machinery each rollout) empty. Remove them so those two top-layer
    folders don't sit around empty."""
    for name in ("data", "video"):
        d = Path(out_dir) / name
        try:
            d.rmdir()          # only succeeds if empty; leaves anything unexpected in place
        except OSError:
            pass


def atomic_write_text(path, text):
    """Durably replace one small text artifact without exposing a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_jsonl_fsync(path, value):
    """Append one complete JSON record and force it through the OS page cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class Timings:
    """Reusable per-component timer (project convention: time every script by phase and save the
    breakdown with the run). Use `with tm.section("name"):` around each logical phase; call
    `tm.save(out_dir)` at the end to print a summary table and write timings.json into the run folder."""

    def __init__(self):
        self.records = []
        self._wall0 = time.perf_counter()

    @contextmanager
    def section(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.records.append((name, dt))
            print(f"[time] {name}: {dt:.2f}s")

    def save(self, out_dir, off_seconds=0.0, filename="timings.json"):
        total = time.perf_counter() - self._wall0
        projected = total - off_seconds                       # cost if the OFF pass were skipped (--free-only)
        data = {"components": [{"name": n, "seconds": round(s, 3)} for n, s in self.records],
                "total_seconds": round(total, 3),
                "off_pass_seconds": round(off_seconds, 3),
                "projected_free_only_seconds": round(projected, 3)}
        atomic_write_json(Path(out_dir) / filename, data)
        width = max((len(n) for n, _ in self.records), default=10)
        print("[time] ---------------- component timing ----------------")
        for n, s in self.records:
            print(f"[time]   {n:<{width}}  {s:8.2f}s  ({100 * s / total if total else 0:4.1f}%)")
        print(f"[time]   {'TOTAL':<{width}}  {total:8.2f}s")
        if off_seconds > 0:
            print(f"[time] projected WITH --free-only: {projected:.2f}s  "
                  f"(OFF pass {off_seconds:.2f}s = {100 * off_seconds / total if total else 0:.1f}% would be skipped)")
        else:
            print("[time] projected WITH --free-only: equals TOTAL above (--free-only already active)")
        print(f"[time] wrote {filename}")
        return total
