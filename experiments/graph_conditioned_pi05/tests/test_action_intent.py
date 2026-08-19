from dataclasses import replace

from experiments.graph_conditioned_pi05.action_intent import (
    ActionIntent,
    IntentOperation,
    MotionDirection,
    PlacementRelation,
    PlacementSubstage,
)
from experiments.graph_conditioned_pi05.graph_replanning import GraspSubstage


def test_grasp_substages_render_atomic_instructions():
    fallback = ActionIntent(IntentOperation.GRASP, "sauce can")
    assert fallback.render_stage_instruction() == (
        "Align the gripper with the sauce can."
    )
    selected = replace(fallback, preferred_arm="left")
    assert selected.render_stage_instruction() == (
        "Align the left gripper with the sauce can."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.GRASP_APPROACH
    ).render_stage_instruction() == (
        "Approach the middle of the sauce can with the open left gripper. "
        "Keep moving toward it until the gripper is centered for grasping."
    )
    warning = replace(
        selected,
        blocked_arm="right",
        obstacle_label="kettle",
        collision_imminent=True,
    )
    assert warning.render_stage_instruction() == (
        "Collision risk: the kettle blocks the right gripper. "
        "Align the left gripper with the sauce can."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.MOVE_DOWN
    ).render_stage_instruction() == (
        "Move the left gripper down to align it with the sauce can."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.MOVE_UP
    ).render_stage_instruction() == (
        "Move the left gripper up to align it with the sauce can."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.MOVE_CLOSER
    ).render_stage_instruction() == (
        "Move the left gripper closer to the sauce can for grasping."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.FINAL_APPROACH
    ).render_stage_instruction() == (
        "Move the left gripper straight toward the sauce can. "
        "Keep the left gripper open and move closer before closing."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.FINAL_APPROACH_DOWN
    ).render_stage_instruction() == (
        "Move the left gripper down and toward the sauce can. "
        "Keep the left gripper open and move closer before closing."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.FINAL_APPROACH_UP
    ).render_stage_instruction() == (
        "Move the left gripper up and toward the sauce can. "
        "Keep the left gripper open and move closer before closing."
    )
    assert replace(
        selected, grasp_substage=GraspSubstage.CLOSE
    ).render_stage_instruction() == "Close the left gripper to grasp the sauce can."


def test_place_and_release_rendering_uses_destination_substages():
    placement = ActionIntent(
        IntentOperation.PLACE,
        "sauce can",
        destination_label="basket",
        placement_relation=PlacementRelation.IN,
        motion_directions=(MotionDirection.FORWARD, MotionDirection.LEFT),
    )
    assert placement.render_stage_instruction() == (
        "Keep holding the sauce can. Move it over the center of the basket."
    )
    descent = replace(
        placement, placement_substage=PlacementSubstage.FINAL_DESCENT
    )
    assert descent.render_stage_instruction() == (
        "Keep holding the sauce can. Lower it into the basket."
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
        "placement_substage": "align_destination",
        "preferred_arm": None,
        "blocked_arm": None,
        "obstacle_label": None,
        "collision_imminent": False,
        "grasp_substage": None,
        "phase": "placement",
    }


def main():
    test_grasp_substages_render_atomic_instructions()
    test_place_and_release_rendering_uses_destination_substages()
    test_intent_is_serializable_and_self_consistent()
    print("3 action-intent checks passed")


if __name__ == "__main__":
    main()
