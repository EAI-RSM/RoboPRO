import sys
from pathlib import Path

sys.path.append("./")

from script.bench_script.setup_paths import setup_paths
setup_paths()

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import h5py
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
bench_root = Path(os.environ["BENCH_ROOT"])


def class_decorator(task_name):
    envs_module = None
    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        for mod_path in [f"bench_envs.{task_name}", f"bench_envs.study.{task_name}", f"bench_envs.office.{task_name}", f"bench_envs.kitchenl.{task_name}", f"bench_envs.kitchens.{task_name}"]:
            try:
                envs_module = importlib.import_module(mod_path)
                break
            except ModuleNotFoundError:
                continue
    if envs_module is None:
        envs_module = importlib.import_module(f"envs.{task_name}")

    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except AttributeError:
        raise SystemExit("No such task")

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)

    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        task_config_path = f"{bench_root}/bench_task_config/{task_config}.yml"
    else:
        task_config_path = f"./task_config/{task_config}.yml"
        
    with open(task_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def _banner(text, char="─", width=74):
    line = char * width
    print(f"\n\033[96m{line}\n{text}\n{line}\033[0m")


def _to_jsonable(x):
    # numpy-safe conversion for scene_info.json payloads (collision metrics etc.)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            return str(x)
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            return str(x)
    return x


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    # Debug mode: instead of iterating over seeds (reopening Sapien),
    # hold the viewer open on the first failure so targets/scene can be inspected.
    # Enable with:  RT_DEBUG_STUCK=1 python -u script/collect_data.py ...
    debug_stuck = os.getenv("RT_DEBUG_STUCK") == "1"

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    if debug_stuck:
        print("\033[93m[RT_DEBUG_STUCK=1] will hold viewer on failure; no seed iteration\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        save_seed_file = args.get("save_seed", True)

        if save_seed_file and os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            print(f"\n\033[90m── seed attempt {epid} · successes {suc_num}/{args['episode_num']} "
                  + "─" * 40 + "\033[0m")
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                if hasattr(TASK_ENV, "_maybe_apply_language_perturbation"):
                    TASK_ENV._maybe_apply_language_perturbation()
                TASK_ENV.play_once()

                accept = TASK_ENV.plan_success and TASK_ENV.check_success()
                cf_note = ""
                if accept and args.get("seed_require_collision_free", False):
                    # pure-S seed acceptance: bank only collision-free successes so
                    # the kept positives hit episode_num exactly (requires
                    # enable_collision_metrics: true)
                    try:
                        if bool(TASK_ENV.get_collision_metrics().get("is_collision", False)):
                            accept = False
                            cf_note = " — task ok but COLLIDED (pure-S required)"
                    except Exception as e:
                        print(f"seed collision check unavailable: {e}")
                if accept:
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid}){cf_note}")
                    fail_num += 1
                    if debug_stuck:
                        reason = f"plan_success={TASK_ENV.plan_success} check_success={TASK_ENV.check_success()} seed={epid}"
                        TASK_ENV.debug_hold_viewer(reason=reason)
                        return

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                if debug_stuck:
                    TASK_ENV.debug_hold_viewer(reason=f"UnStableError: {e} (seed={epid})")
                    return
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(stack_trace)
                print(" -------------")
                fail_num += 1
                if debug_stuck:
                    try:
                        TASK_ENV.debug_hold_viewer(reason=f"Exception: {e} (seed={epid})")
                    except Exception:
                        pass
                    return
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            if save_seed_file:
                with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                    for sed in seed_list:
                        file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        # replan_on_collect: plan fresh at collection time instead of replaying the
        # seed-phase trajectories (negative samples: same seeds, planner re-plans
        # while blind to obstacles)
        replan = bool(args.get("replan_on_collect", False))
        args["need_plan"] = replan
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        outcome_counts, deleted_num = {}, 0

        for episode_idx in range(st_idx, args["episode_num"]):
            if exist_hdf5(episode_idx):
                continue
            _banner(f"COLLECT episode {episode_idx} · seed {seed_list[episode_idx]} · "
                    f"{args['task_name']} / {args['task_config']}")

            try:
                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)
                if hasattr(TASK_ENV, "_maybe_apply_language_perturbation"):
                    TASK_ENV._maybe_apply_language_perturbation()

                if not replan:
                    traj_data = TASK_ENV.load_tran_data(episode_idx)
                    args["left_joint_path"] = traj_data["left_joint_path"]
                    args["right_joint_path"] = traj_data["right_joint_path"]
                    TASK_ENV.set_path_lst(args)

                info_file_path = os.path.join(args["save_path"], "scene_info.json")

                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)

                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                info = TASK_ENV.play_once()
                if info is None:
                    info = getattr(TASK_ENV, "info", None) or {}
                # grounding sidecar: seg-id -> name map + task-designated role names
                # (must be captured while the scene is still alive)
                if hasattr(TASK_ENV, "get_actor_id_map"):
                    info["actor_id_map"] = TASK_ENV.get_actor_id_map()
                if hasattr(TASK_ENV, "get_role_names"):
                    info["role_names"] = TASK_ENV.get_role_names()

                # outcome label (2x2 grid: success x collision) — measured while the
                # scene is still alive, never assumed
                success = bool(TASK_ENV.check_success())
                metrics = None
                if getattr(TASK_ENV, "enable_collision_metrics", False) and hasattr(TASK_ENV, "get_collision_metrics"):
                    try:
                        metrics = TASK_ENV.get_collision_metrics()
                    except Exception as e:
                        print(f"collision metrics unavailable: {e}")
                if metrics is not None:
                    collision = bool(metrics.get("is_collision", False))
                    label_code = (1 if success and not collision else
                                  2 if success and collision else
                                  3 if collision else 4)
                    label = {1: "success", 2: "success_with_accident",
                             3: "crashed_and_failed", 4: "failed_no_accident"}[label_code]
                else:
                    collision, label_code = None, None
                    label = "success" if success else "failed"
                planner_blind = getattr(TASK_ENV, "planner_exclude_obstacles", None)
                if planner_blind is None:
                    planner_blind = bool(getattr(TASK_ENV, "enable_collision_metrics", False))

                # only episodes with saved frames become an hdf5 (blind negative plans
                # sometimes produce no executable motion -> nothing to save, not a crash)
                cache_dir = os.path.join(args["save_path"], ".cache", f"episode{episode_idx}")
                has_frames = os.path.isdir(cache_dir) and any(
                    fn.endswith(".pkl") for fn in os.listdir(cache_dir))

                info["outcome"] = {
                    "success": success,
                    "collision": collision,
                    "label_code": label_code,
                    "label": label,
                    "plan_success": bool(getattr(TASK_ENV, "plan_success", True)),
                    "planner_blind_to_obstacles": bool(planner_blind),
                    "has_trajectory": bool(has_frames),
                    "collision_metrics": _to_jsonable(metrics) if metrics is not None else None,
                }
                print(f"\033[96mepisode {episode_idx}: outcome = {label} "
                      f"(success={success}, collision={collision}, frames={has_frames})\033[0m")
                outcome_counts[label] = outcome_counts.get(label, 0) + 1
                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4, default=str)

                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                if has_frames:
                    TASK_ENV.merge_pkl_to_hdf5_video()
                else:
                    print(f"\033[93mepisode {episode_idx}: no frames saved "
                          f"(planner produced no executable motion) — no hdf5 written\033[0m")
                TASK_ENV.remove_data_cache()

                # dataset filter: keep_labels (e.g. pos -> [success],
                # neg -> [success_with_accident, crashed_and_failed]);
                # absent -> legacy rule (keep successes / keep_failed keeps all)
                keep_labels = args.get("keep_labels", None)
                if keep_labels is not None:
                    keep = has_frames and (label in keep_labels)
                else:
                    keep = success or bool(args.get("keep_failed_episodes", False))
                info["outcome"]["kept"] = bool(keep)
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4, default=str)

                if not keep:
                    deleted_num += 1
                    why = (f"label '{label}' not in keep_labels {keep_labels}"
                           if keep_labels is not None else "task failed")
                    print(f"\033[91mepisode {episode_idx} filtered out ({why}) — removing files\033[0m")
                    for ext_path in [
                        os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5"),
                        os.path.join(args["save_path"], "video", f"episode{episode_idx}.mp4"),
                    ]:
                        if os.path.exists(ext_path):
                            os.remove(ext_path)
                    continue

                # stamp the kept episode's HDF5 with its outcome -> self-contained
                # for the training dataloader (no sidecar dependency)
                hdf5_path = os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5")
                if has_frames and os.path.exists(hdf5_path):
                    with h5py.File(hdf5_path, "a") as hf:
                        hf.attrs["label"] = label
                        hf.attrs["label_code"] = -1 if label_code is None else int(label_code)
                        hf.attrs["success"] = bool(success)
                        hf.attrs["collision"] = -1 if collision is None else int(collision)
                        hf.attrs["planner_blind_to_obstacles"] = bool(planner_blind)
                        hf.attrs["seed"] = int(seed_list[episode_idx])
                        hf.attrs["task_name"] = str(args["task_name"])
                        hf.attrs["task_config"] = str(args["task_config"])
                        if metrics is not None:
                            hf.attrs["collision_metrics_json"] = json.dumps(
                                _to_jsonable(metrics), default=str)
            except Exception as e:
                # one bad episode (CuRobo/mesh crash, missing frames, etc.) must NOT
                # kill the whole run — log it, record it, clean up, move on.
                print(f"\033[91m[episode {episode_idx}] crashed during collection "
                      f"(seed={seed_list[episode_idx]}): {e}\033[0m")
                print(traceback.format_exc())
                outcome_counts["error"] = outcome_counts.get("error", 0) + 1
                try:
                    info_file_path = os.path.join(args["save_path"], "scene_info.json")
                    err_db = {}
                    if os.path.exists(info_file_path):
                        with open(info_file_path, "r", encoding="utf-8") as f:
                            err_db = json.load(f)
                    err_db[f"episode_{episode_idx}"] = {"outcome": {
                        "label": "error", "error": str(e),
                        "seed": int(seed_list[episode_idx])}}
                    with open(info_file_path, "w", encoding="utf-8") as f:
                        json.dump(err_db, f, ensure_ascii=False, indent=4, default=str)
                except Exception:
                    pass
                for _cleanup in (TASK_ENV.remove_data_cache, TASK_ENV.close_env):
                    try:
                        _cleanup()
                    except Exception:
                        pass
                for ext_path in [
                    os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5"),
                    os.path.join(args["save_path"], "video", f"episode{episode_idx}.mp4"),
                ]:
                    if os.path.exists(ext_path):
                        try:
                            os.remove(ext_path)
                        except Exception:
                            pass
                continue

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)

        _banner(f"COLLECTION SUMMARY · {args['task_name']} / {args['task_config']}")
        for name in ("success", "success_with_accident", "crashed_and_failed",
                     "failed_no_accident", "failed", "error"):
            if name in outcome_counts:
                print(f"  {name:26s} {outcome_counts[name]}")
        total = sum(outcome_counts.values())
        print(f"  {'episodes attempted':26s} {total}")
        print(f"  {'kept in dataset':26s} {total - deleted_num}")
        if deleted_num:
            print(f"  {'filtered out (removed)':26s} {deleted_num}")


if __name__ == "__main__":
    pass  # Skip Sapien_TEST — ray tracing uses too much VRAM for CuRobo

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
