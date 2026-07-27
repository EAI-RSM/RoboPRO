"""Download the RoboPRO JAX (orbax) checkpoint and lay it out for the PI0 loader.

Fetches only the `jax_30000/` subtree of mzxuan/robopro_jax_30000 into a staging
dir under checkpoints/, then symlinks it into the layout pi_model.PI0 expects:

    checkpoints/<train_config>/<model_name>/<ckpt_id>/{params, assets/<repo_id>}

Public repo, no token needed. Idempotent + resumable (snapshot_download resumes).
"""
import os
import pathlib
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download

REPO_ID = "mzxuan/robopro_jax_30000"
SUBDIR = "jax_30000"
TRAIN_CONFIG = "pi05_robopro_top_cam_jax"
MODEL_NAME = "robopro"
CKPT_ID = "30000"

PI05 = pathlib.Path(__file__).resolve().parents[1]  # policy/pi05
CKPT_ROOT = PI05 / "checkpoints"
STAGING = CKPT_ROOT / "_hf" / "robopro_jax_30000"
LAYOUT_DIR = CKPT_ROOT / TRAIN_CONFIG / MODEL_NAME
LINK = LAYOUT_DIR / CKPT_ID


def main() -> int:
    STAGING.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[download] {REPO_ID}:{SUBDIR}/** -> {STAGING}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{SUBDIR}/**"],
        local_dir=str(STAGING),
        max_workers=8,
    )
    dt = time.time() - t0
    ckpt_src = STAGING / SUBDIR
    print(f"[download] done in {dt:.0f}s -> {ckpt_src}", flush=True)

    # Sanity: the loader needs params/ and assets/<repo_id>/
    params = ckpt_src / "params"
    assets = ckpt_src / "assets"
    assert params.is_dir(), f"missing {params}"
    assert assets.is_dir(), f"missing {assets}"
    repo_ids = [p.name for p in assets.iterdir() if p.is_dir()]
    print(f"[download] assets repo ids: {repo_ids}", flush=True)

    # Symlink into loader layout (absolute target, idempotent).
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    if LINK.is_symlink() or LINK.exists():
        if LINK.is_symlink() and os.readlink(LINK) == str(ckpt_src):
            print(f"[layout] link already correct: {LINK} -> {ckpt_src}", flush=True)
        else:
            print(f"[layout] WARNING: {LINK} exists and differs; leaving as-is", flush=True)
    else:
        LINK.symlink_to(ckpt_src, target_is_directory=True)
        print(f"[layout] linked {LINK} -> {ckpt_src}", flush=True)

    # Final verification against the exact paths pi_model / eval_double_env check.
    assert (LINK / "params").exists(), "params/ not visible through link"
    assert (LINK / "assets").is_dir(), "assets/ not visible through link"
    print("[ok] checkpoint staged and laid out:", flush=True)
    print(f"     train_config={TRAIN_CONFIG} model_name={MODEL_NAME} checkpoint_id={CKPT_ID}", flush=True)
    print(f"     {LINK}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
