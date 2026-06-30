"""Pipeline glue for the data-collection entrypoint (customized_robotwin/script/
collect_data.py): plan the sibling variations to collect for a scene, and turn a
recorded perturbation into a causal-graph annotation.

Pure logic — no sim, no subprocess. The entrypoint reads the per-task toggles,
calls `plan_variations` (which SKIPS unsupported tasks via the registries), and
runs each variation; the causal annotation is attached to negative variations.
"""
from . import core
from . import multimodal as mm

# Perturbation types whose magnitude comes from the stratified sampler (vs the
# corridor-form obstacle perturbations which inject their own obstacle).
_SAMPLED_PTYPES = {"shift_object", "shift_target"}


def variations_enabled(args: dict) -> bool:
    return bool((args.get("negative") or {}).get("enabled")
                or (args.get("multimodal") or {}).get("enabled"))


def plan_variations(task_name: str, args: dict, seed: int) -> list[dict]:
    """Sibling variations to collect for one scene (seed). The clean episode is
    collected by the standard pipeline; these are the alternatives:
      negative  -> bad actions (causally-labelled failures)
      multimodal-> other good actions (distinct successful behaviors)
    Unsupported (task, feature) pairs are skipped (empty contribution)."""
    variations: list[dict] = []

    neg = args.get("negative") or {}
    if neg.get("enabled"):
        entry = core.task_entry(task_name)          # registry gate
        if entry is not None:
            ptypes = neg.get("ptypes") or sorted(entry["ptypes"])
            n = int(neg.get("n_per_scene", 1))
            for pt in ptypes:
                if pt not in entry["ptypes"]:
                    continue
                for i in range(n):
                    params = (core.sample_shift_params(pt, seed, i) if pt in _SAMPLED_PTYPES
                              else {"mode": "corridor"})
                    variations.append({
                        "kind": "negative", "name": f"{pt}_v{i}", "quality": "bad",
                        "runner": "run_targeted_episode",
                        "ptype": pt, "params": params})

    mmc = args.get("multimodal") or {}
    if mmc.get("enabled"):
        mentry = mm.multimodal_entry(task_name)     # registry gate
        if mentry is not None:
            modes = mmc.get("modes") or [m for m in mm.MODE_ORDER if m in mentry["modes"]]
            for mode in modes:
                if mode == "direct" or mode not in mentry["modes"]:
                    continue                        # "direct" == the clean episode
                variations.append({
                    "kind": "multimodal", "name": mode, "quality": "good",
                    "runner": "run_mode_episode", "mode": mode})

    return variations


def supported_summary(task_name: str, args: dict) -> dict:
    """What this task will produce — for logging / the scene index."""
    neg_ok = (args.get("negative") or {}).get("enabled") and core.task_entry(task_name) is not None
    mm_ok = (args.get("multimodal") or {}).get("enabled") and mm.multimodal_entry(task_name) is not None
    return {"negative_supported": bool(neg_ok), "multimodal_supported": bool(mm_ok)}


def clean_record(env, args: dict, seed: int) -> dict:
    """Base annotation for a CLEAN (unperturbed) episode — the 'good action'
    reference each scene's siblings are compared against. Defensive: assumes no
    task-specific attrs beyond the common target_obj/des_obj convention."""
    sig = {"plan_success": bool(getattr(env, "plan_success", False))}
    try:
        sig["success_flag"] = bool(env.check_success())
    except Exception:  # noqa: BLE001
        sig["success_flag"] = None
    tgt, dst = getattr(env, "target_obj", None), getattr(env, "des_obj", None)
    entities = {"target": tgt.get_name() if tgt is not None and hasattr(tgt, "get_name") else None,
                "destination": dst.get_name() if dst is not None and hasattr(dst, "get_name") else None}
    try:
        instruction = env.get_instruction()
    except Exception:  # noqa: BLE001
        instruction = None
    task_name = args.get("task_name")
    return {
        "label_schema_version": core.LABEL_SCHEMA_VERSION,
        "task_name": task_name, "task_config": args.get("task_config"),
        "scene_seed": int(seed), "executor_type": "scripted_expert",
        "pair_id": f"{task_name}__seed{int(seed):05d}",
        "pair_role": "clean", "variant": "clean", "quality": "good",
        "perturbation_type": None, "status": "recorded",
        "outcome": ("clean_success" if sig["success_flag"]
                    else "clean_failure" if sig["success_flag"] is not None else "unknown"),
        "signals": sig, "progress_entities": entities, "instruction": instruction,
    }


def build_causal_annotation(record: dict) -> dict:
    """Structure a recorded perturbation into a causal-graph annotation —
    intervention (cause, known by construction) -> mechanism -> mediators ->
    outcome, plus the clean counterfactual and scene context. Rich enough to
    reconstruct a causal graph for diagnostics / assessment."""
    sig = record.get("signals") or {}
    cmd = record.get("commanded") or {}
    cm = sig.get("collision_metrics") or {}
    ptype = record.get("perturbation_type")

    intervention = {                       # the manipulated variable (do-operator)
        "type": ptype,
        "actor": record.get("perturbed_actor"),
        "trigger": record.get("trigger"),
        "perceptual_class": record.get("perceptual_failure_class"),
        "magnitude_cm": cmd.get("magnitude_cm"),
        "direction_world": cmd.get("shift_world"),
        "camera_frame": cmd.get("camera_frame"),
        "sampler": record.get("sampler"),
    }
    mechanism = {                          # how the desync enters the planner
        "planner_belief": ("undetected" if ptype == "hide_obstacle"
                           else "mislocated" if ptype in ("shift_object", "shift_target", "shift_obstacle")
                           else None),
        "obstacle_source": record.get("obstacle_source"),
        "achieved": record.get("achieved"),
    }
    mediators = {                          # observable intermediates on the path to the outcome
        "plan_success": sig.get("plan_success"),
        "object_moved_cm": sig.get("object_moved_cm"),
        "place_error_cm": sig.get("place_error_cm"),
        "is_collision": cm.get("is_collision"),
        "n_contacts": len(record.get("contact_log") or []),
    }
    outcome = {                            # the effect
        "outcome": record.get("outcome"),
        "success": sig.get("success_flag"),
        "failure_reason": None if sig.get("success_flag") else record.get("failure_reason"),
    }
    return {
        "schema": "causal_v1",
        "nodes": {"intervention": intervention, "mechanism": mechanism,
                  "mediators": mediators, "outcome": outcome},
        "edges": [["intervention", "mechanism"], ["mechanism", "mediators"],
                  ["mediators", "outcome"]],
        "counterfactual": {"variant": "clean", "pair_id": record.get("pair_id"),
                           "note": "same scene_seed without the intervention -> clean_success"},
        "scene": {"target": (record.get("progress_entities") or {}).get("target"),
                  "destination": (record.get("progress_entities") or {}).get("destination"),
                  "planner_visible_actors": record.get("planner_visible_actors")},
    }
