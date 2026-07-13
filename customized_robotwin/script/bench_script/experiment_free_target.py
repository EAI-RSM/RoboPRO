"""
Experiment: does an UNRESTRICTED target (mouse) location cause curobo/expert
failures? Normally put_mouse_on_pad constrains the mouse to |x|>=0.3 and
y in [-0.23, 0.05] (graspability). Here we let the mouse start anywhere on the
table (only avoiding the destination pad), run the expert rollout, and record
whether it succeeded.

Outputs (in --out-dir):
  - rollouts/video/episode<i>.mp4   one video per rollout (success or fail)
  - results.jsonl                   per-rollout: x, y, success, plan_success, error
  - target_success_scatter.png      x vs y, green=success / red=failure

USAGE (from the benchmark folder):
    cd benchmark
    source set_env.sh
    export ROBOTWIN_BENCH_TASK=bench
    python script/bench_script/experiment_free_target.py \
        --num-rollouts 50 --seed-start 0 \
        --out-dir ../scripts/validation/results/free_target_curobo

    # re-plot from an existing results.jsonl without re-running rollouts:
    python script/bench_script/experiment_free_target.py --plot-only \
        --out-dir ../scripts/validation/results/free_target_curobo
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

from setup_paths import setup_paths
setup_paths()

bench_root = Path(os.environ["BENCH_ROOT"])
robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
os.chdir(robotwin_root)

import yaml
from envs import CONFIGS_PATH
from envs.utils import rand_pose, create_actor, create_box
from visualize_task_scene import get_env_class, get_embodiment_config

# --- sampling region (near-full table; table is x[-0.6,0.6] y[-0.35,0.35]) ---
# y capped at 0.15 (back of the table is out of scope for these scenes).
X_LIM = (-0.55, 0.55)
Y_LIM = (-0.32, 0.15)
PAD_XY = (0.0, -0.10)          # fixed destination pad
PAD_CLEAR = 0.12               # min mouse-to-pad distance (avoid spawning on the bin)
# the original (restricted) mouse region, drawn on the scatter for reference
ORIG_X = [(-0.45, -0.30), (0.30, 0.45)]
ORIG_Y = (-0.23, 0.05)
# the region where the OCCLUDER-scene bottle usually spawns (analyze_occluder_visibility.py
# TARGET_XLIM / TARGET_YLIM) -- drawn for reference.
BOTTLE_XLIM = (-0.15, 0.15)
BOTTLE_YLIM = (0.10, 0.15)


def make_free_target_class():
    Base = get_env_class("put_mouse_on_pad", bench_subdir="office")

    class FreeTargetTask(Base):
        # Remove the back furniture (shelf/cabinet/wooden-box/file-holder) to match the
        # current benchmark scenes, which don't include it. With it off, the office base
        # appends only off-table rects to prohibited_area["table"], so the _mouse_in_furniture
        # guard below simply never fires (every sampled position gets a real rollout).
        SPAWN_BACK_FURNITURE = False
        # play_once here is the STOCK put_mouse_on_pad expert (dev's vanilla
        # grasp -> lift -> place) -- i.e. the baseline planner, no subgoal routing.
        forced_xy = (0.0, 0.2)
        fixed_pad_xy = PAD_XY

        def load_actors(self):
            x, y = self.forced_xy
            # create_static_elements() ran before load_actors(), so prohibited_area["table"]
            # already holds the furniture footprints (shelf/cabinet/file-holder). Flag if the
            # target would spawn inside one -> that's a furniture overlap, not a reach test.
            self._mouse_in_furniture = any(
                (r[0] <= x <= r[2] and r[1] <= y <= r[3])
                for r in self.prohibited_area.get("table", [])
            )
            mouse_pose = rand_pose(xlim=[x], ylim=[y], qpos=[0.5, 0.5, 0.5, 0.5],
                                   rotate_rand=True, rotate_lim=[0, 3.14, 0])
            self.mouse_id = np.random.choice(self._target_ids("office", "047_mouse"))
            self.target_obj = create_actor(
                scene=self, pose=mouse_pose, modelname="047_mouse", convex=True,
                model_id=self.mouse_id,
                scale=self.item_info["scales"]["047_mouse"].get(f"{self.mouse_id}", None),
            )
            self.target_obj.set_mass(0.05)

            px, py = self.fixed_pad_xy
            pad_pose = rand_pose(xlim=[px], ylim=[py], qpos=[1, 0, 0, 0], rotate_rand=False)
            colors = {
                "Red": (1, 0, 0), "Green": (0, 1, 0), "Blue": (0, 0, 1), "Yellow": (1, 1, 0),
                "Cyan": (0, 1, 1), "Magenta": (1, 0, 1), "Black": (0, 0, 0), "Gray": (0.5, 0.5, 0.5),
            }
            items = list(colors.items())
            self.color_name, self.color_value = items[np.random.choice(len(items))]
            self.des_obj = create_box(scene=self, pose=pad_pose, half_size=[0.06, 0.06, 0.0005],
                                      color=self.color_value, name="box", is_static=True)
            self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
            self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
            self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
            self.des_obj_pose[2] += 0.02
            # NOTE: no clutter / milk-box branch here (clean scene by design)

    return FreeTargetTask


def build_cfg(task_name, base_config, seed, ep_num, save_path):
    config_path = bench_root / "bench_task_config" / f"{base_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = 0
    cfg["now_ep_num"] = ep_num
    cfg["seed"] = int(seed)
    cfg["need_plan"] = True
    cfg["save_data"] = True
    cfg["save_path"] = str(save_path)
    cfg.setdefault("data_type", {"rgb": True})
    # clean scene, no clutter
    cfg.setdefault("domain_randomization", {})
    cfg["domain_randomization"].update({"cluttered_table": False, "obstacle_density": 0,
                                        "clean_background_rate": 0})

    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(name):
        rf = emb[name]["file_path"]
        if rf is None:
            raise SystemExit("missing embodiment files")
        return rf

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = cfg["right_robot_file"] = emb_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    else:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])
    return cfg


def sample_xy(rng):
    while True:
        x = rng.uniform(*X_LIM)
        y = rng.uniform(*Y_LIM)
        if np.hypot(x - PAD_XY[0], y - PAD_XY[1]) >= PAD_CLEAR:
            return float(x), float(y)


def run(args):
    save_root = Path(args.out_dir)
    (save_root).mkdir(parents=True, exist_ok=True)
    rollout_dir = save_root / "rollouts"
    jsonl_path = save_root / "results.jsonl"
    rng = np.random.default_rng(args.seed_start)

    FreeTargetTask = make_free_target_class()
    env = FreeTargetTask()  # reuse one instance across rollouts (collect_data pattern)

    print(f"running {args.num_rollouts} rollouts; videos+data -> {save_root}\n")
    with open(jsonl_path, "w") as fout:
        for i in range(args.num_rollouts):
            x, y = sample_xy(rng)
            env.forced_xy = (x, y)
            env.fixed_pad_xy = PAD_XY
            seed = args.seed_start + i

            success = False
            plan_success = False
            in_furniture = False
            err = None
            try:
                env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed, i, rollout_dir))
                in_furniture = bool(getattr(env, "_mouse_in_furniture", False))
                if in_furniture:
                    env.close_env()  # spawned in furniture footprint: skip the rollout
                else:
                    env.play_once()
                    plan_success = bool(getattr(env, "plan_success", False))
                    success = bool(plan_success and env.check_success())
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            # save the rollout video (skip when there was no rollout)
            if not in_furniture:
                try:
                    env.close_env(clear_cache=True)
                    env.merge_pkl_to_hdf5_video()
                    env.remove_data_cache()
                except Exception as e:
                    if err is None:
                        err = f"video:{type(e).__name__}: {e}"
                    try:
                        env.close_env()
                    except Exception:
                        pass

            rec = {"i": i, "seed": seed, "x": x, "y": y, "success": success,
                   "plan_success": plan_success, "in_furniture": in_furniture, "error": err}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            tag = "FURN" if in_furniture else ("OK  " if success else "FAIL")
            print(f"[{i+1}/{args.num_rollouts}] {tag} x={x:+.3f} y={y:+.3f} "
                  f"plan={plan_success} {'' if err is None else '(' + err + ')'}")

    try:
        env.close_env()
    except Exception:
        pass
    print(f"\ndone -> {jsonl_path}")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in open(Path(args.out_dir) / "results.jsonl") if l.strip()]
    if not recs:
        raise SystemExit("no records to plot")
    furn = [r for r in recs if r.get("in_furniture")]
    rollouts = [r for r in recs if not r.get("in_furniture")]   # actually-tested positions
    sx = [r["x"] for r in rollouts if r["success"]]
    sy = [r["y"] for r in rollouts if r["success"]]
    fx = [r["x"] for r in rollouts if not r["success"]]
    fy = [r["y"] for r in rollouts if not r["success"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sx, sy, c="#2ca02c", s=42, label=f"success ({len(sx)})", edgecolors="white")
    ax.scatter(fx, fy, c="#d62728", s=42, marker="X", label=f"failure ({len(fx)})", edgecolors="white")
    ax.scatter([r["x"] for r in furn], [r["y"] for r in furn], c="0.6", s=30, marker="s",
               label=f"in furniture, skipped ({len(furn)})", edgecolors="white")
    # reference: table bounds, fixed pad, and the original restricted mouse region
    ax.add_patch(plt.Rectangle((-0.6, -0.35), 1.2, 0.7, fill=False, ec="0.6", lw=1.5))
    for (x0, x1) in ORIG_X:
        ax.add_patch(plt.Rectangle((x0, ORIG_Y[0]), x1 - x0, ORIG_Y[1] - ORIG_Y[0],
                                   fill=False, ec="navy", ls="--", lw=1.5))
    # region where the occluder-scene bottle usually spawns (labelled -> shown in legend)
    ax.add_patch(plt.Rectangle((BOTTLE_XLIM[0], BOTTLE_YLIM[0]),
                               BOTTLE_XLIM[1] - BOTTLE_XLIM[0], BOTTLE_YLIM[1] - BOTTLE_YLIM[0],
                               fill=False, ec="darkorange", ls="--", lw=1.8,
                               label="bottle spawn region"))
    ax.scatter([PAD_XY[0]], [PAD_XY[1]], c="black", marker="s", s=90, label="pad (destination)")
    ax.set_xlabel("mouse x (m)")
    ax.set_ylabel("mouse y (m)")
    ax.set_title(f"Expert success vs unrestricted target location (n={len(recs)})\n"
                 "dashed navy = original allowed mouse region; "
                 "dashed orange = bottle spawn region")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    out = Path(args.out_dir) / "target_success_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    n_ok, n_roll = len(sx), len(rollouts)
    rate = f"{n_ok}/{n_roll} ({n_ok/n_roll:.0%})" if n_roll else "n/a"
    print(f"saved {out}   success rate on clear-table positions = {rate}; "
          f"{len(furn)} skipped (spawned in furniture)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-rollouts", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--out-dir", default="./free_target_curobo")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        run(args)
    plot(args)


if __name__ == "__main__":
    main()
