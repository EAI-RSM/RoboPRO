"""Supported-task registry for targeted data collection.

Tasks are enabled incrementally: a task enters this registry only after its
desync behavior has been verified for each listed perturbation type (locked
quantity does not follow the shift, `check_success` reads live state — see the
design doc's desync audit). Collectors and runners must SKIP unsupported
(task, ptype) combinations with a logged reason, never crash or silently
mislabel.
"""

# Scene-pairing rule: object/target desyncs are collected in clean scenes only;
# obstacle desyncs need clutter, so they run in cluttered scenes only.
REQUIRED_SCENE_BY_PTYPE = {"shift_object": "clean", "shift_target": "clean",
                           "shift_obstacle": "cluttered", "hide_obstacle": "cluttered"}

SUPPORTED_TASKS: dict[str, dict] = {
    # Verified 2026-06-10/12 (PoC batch: 84 episodes + demo quad):
    #   shift_object  after_grasp_plan, boundary ~2.9-3.5 cm (plan_aborted)
    #   shift_target  immediate, flips at the 2 cm check_success epsilon
    #   shift_obstacle corridor form, milk box, success_with_collision
    "put_mouse_on_pad": {
        "bench_subdir": "office",
        "ptypes": frozenset({"shift_object", "shift_target", "shift_obstacle"}),
        "clean_config": "bench_demo_office_clean",
        "cluttered_config": "bench_demo_office_d6",
    },
}


def task_entry(task_name: str) -> dict | None:
    return SUPPORTED_TASKS.get(task_name)


def is_supported(task_name: str, ptype: str | None = None) -> bool:
    entry = SUPPORTED_TASKS.get(task_name)
    if entry is None:
        return False
    return ptype is None or ptype in entry["ptypes"]
