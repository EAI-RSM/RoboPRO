"""CPU guard for the gitignored vendored-CuRobo seed trajectory patch."""

from pathlib import Path


CUSTOMIZED_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = CUSTOMIZED_ROOT.parent
MOTION_GEN = (
    CUSTOMIZED_ROOT
    / "envs/curobo/src/curobo/wrap/reacher/motion_gen.py"
)
REPAIR_COMMAND = (
    f"cd {REPO_ROOT} && git -C customized_robotwin/envs/curobo apply "
    "../../script/bench_script/curobo_seed_traj.patch"
)
PATCH_SENTINELS = (
    "seed_traj: Optional[torch.Tensor] = None",
    "seed_traj=self.seed_traj",
    'getattr(plan_config, "seed_traj", None) is not None',
    "trajopt_seed_traj = ext_seed.contiguous()",
)


def test_seed_traj_patch_is_installed() -> None:
    try:
        source = MOTION_GEN.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssertionError(
            f"cannot read vendored CuRobo motion_gen.py at {MOTION_GEN}: {exc}\n"
            f"Repair with:\n  {REPAIR_COMMAND}"
        ) from exc

    missing = [sentinel for sentinel in PATCH_SENTINELS if sentinel not in source]
    assert not missing, (
        "vendored CuRobo is missing the RoboPRO seed_traj patch sentinels:\n  "
        + "\n  ".join(missing)
        + f"\nRepair with:\n  {REPAIR_COMMAND}"
    )


if __name__ == "__main__":
    test_seed_traj_patch_is_installed()
    print("vendored CuRobo seed_traj patch: present")
