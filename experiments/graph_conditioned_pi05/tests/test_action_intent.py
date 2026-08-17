from dataclasses import replace

from experiments.graph_conditioned_pi05.action_intent import (
    ActionIntent,
    IntentOperation,
    MotionDirection,
    PlacementRelation,
)
from experiments.graph_conditioned_pi05.graph_replanning import GraspSubstage


def test_grasp_substages_render_atomic_instructions():
    fallback = ActionIntent(IntentOperation.GRASP, "sauce can")
    assert fallback.render_stage_instruction() == (
        "Move the gripper toward the sauce can."
    )
    selected = replace(fallback, preferred_arm="left")
    assert selected.render_stage_instruction() == (
        "Use the left gripper to approach the sauce can."
    )
    warning = replace(
        selected,
        blocked_arm="right",
        obstacle_label="kettle",
        collision_imminent=True,
    )
    assert warning.render_stage_instruction() == (
        "Collision risk: the kettle blocks the right gripper. "
        "Use the left gripper to approach the sauce can."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.ALIGN
    ).render_stage_instruction() == "Align the left gripper with the sauce can."
    assert replace(
        selected, grasp_substage=GraspSubstage.CLOSE
    ).render_stage_instruction() == "Close the left gripper to grasp the sauce can."


def test_place_and_release_rendering_have_legacy_parity():
    placement = ActionIntent(
        IntentOperation.PLACE,
        "sauce can",
        destination_label="basket",
        placement_relation=PlacementRelation.IN,
        motion_directions=(MotionDirection.FORWARD, MotionDirection.LEFT),
    )
    assert placement.render_stage_instruction() == (
        "Keep holding the object. Move it forward and left into the basket."
    )
    assert replace(placement, motion_directions=()).render_stage_instruction() == (
        "Place the held object into the basket."
    )
    release = ActionIntent(
        IntentOperation.RELEASE,
        "mouse",
        destination_label="book",
        placement_relation=PlacementRelation.ON,
    )
    assert release.render_stage_instruction() == (
        "Release the held object on the book."
    )


def test_intent_is_serializable_and_self_consistent():
    intent = ActionIntent(
        IntentOperation.PLACE,
        "container",
        destination_label="plate",
        placement_relation=PlacementRelation.ON,
        motion_directions=(MotionDirection.RIGHT,),
    )
    assert intent.phase == "placement"
    assert intent.as_dict() == {
        "operation": "place",
        "target_label": "container",
        "destination_label": "plate",
        "placement_relation": "on",
        "motion_directions": ["right"],
        "preferred_arm": None,
        "blocked_arm": None,
        "obstacle_label": None,
        "collision_imminent": False,
        "grasp_substage": None,
        "phase": "placement",
    }


def main():
    test_grasp_substages_render_atomic_instructions()
    test_place_and_release_rendering_have_legacy_parity()
    test_intent_is_serializable_and_self_consistent()
    print("3 action-intent checks passed")


if __name__ == "__main__":
    main()
