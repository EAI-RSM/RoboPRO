"""Structured action intent and its behavior-compatible language renderer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .graph_replanning import GraspSubstage


class IntentOperation(str, Enum):
    GRASP = "grasp"
    PLACE = "place"
    RELEASE = "release"


class PlacementRelation(str, Enum):
    IN = "in"
    ON = "on"


class MotionDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class ActionIntent:
    """Machine-readable atomic manipulation goal for the current stage."""

    operation: IntentOperation
    target_label: str
    destination_label: str | None = None
    placement_relation: PlacementRelation | None = None
    motion_directions: tuple[MotionDirection, ...] = ()
    preferred_arm: str | None = None
    blocked_arm: str | None = None
    obstacle_label: str | None = None
    collision_imminent: bool = False
    grasp_substage: GraspSubstage | None = None

    def __post_init__(self) -> None:
        if not self.target_label:
            raise ValueError("action intent requires a target label")
        if self.preferred_arm not in {None, "left", "right"}:
            raise ValueError(f"invalid preferred arm: {self.preferred_arm}")
        if self.blocked_arm not in {None, "left", "right"}:
            raise ValueError(f"invalid blocked arm: {self.blocked_arm}")
        placement = self.operation in {IntentOperation.PLACE, IntentOperation.RELEASE}
        if placement and (self.destination_label is None or self.placement_relation is None):
            raise ValueError("place/release intent requires destination and relation")
        if not placement and (self.destination_label or self.placement_relation):
            raise ValueError("grasp intent cannot carry a placement destination")
        if self.operation is not IntentOperation.PLACE and self.motion_directions:
            raise ValueError("only place intent can carry motion directions")
        if self.collision_imminent and not (self.blocked_arm and self.obstacle_label):
            raise ValueError("collision warning requires blocked arm and obstacle")
        if self.operation is IntentOperation.GRASP and self.grasp_substage is None:
            object.__setattr__(self, "grasp_substage", GraspSubstage.APPROACH)
        if self.operation is not IntentOperation.GRASP and self.grasp_substage is not None:
            raise ValueError("only grasp intent can carry a grasp substage")

    @property
    def phase(self) -> str:
        return {
            IntentOperation.GRASP: "grasp",
            IntentOperation.PLACE: "placement",
            IntentOperation.RELEASE: "release",
        }[self.operation]

    def render_stage_instruction(self) -> str:
        """Render the exact stage language used before this refactor."""
        if self.operation is IntentOperation.GRASP:
            arm = f"{self.preferred_arm} gripper" if self.preferred_arm else "gripper"
            if self.grasp_substage is GraspSubstage.CLOSE:
                instruction = f"Close the {arm} to grasp the {self.target_label}."
            elif self.grasp_substage is GraspSubstage.ALIGN:
                instruction = (
                    f"Align the {arm} with the {self.target_label}. "
                    f"Move the {arm} closer to the {self.target_label} for grasping."
                )
            elif self.preferred_arm:
                instruction = (
                    f"Use the {self.preferred_arm} gripper to approach the "
                    f"{self.target_label}."
                )
            else:
                instruction = f"Move the gripper toward the {self.target_label}."
            if self.collision_imminent:
                return (
                    f"Collision risk: the {self.obstacle_label} blocks the "
                    f"{self.blocked_arm} gripper. {instruction}"
                )
            return instruction

        preposition = (
            "onto" if self.placement_relation is PlacementRelation.ON else "into"
        )
        if self.operation is IntentOperation.RELEASE:
            destination_preposition = (
                "on" if self.placement_relation is PlacementRelation.ON else "in"
            )
            return (
                f"Release the held object {destination_preposition} "
                f"the {self.destination_label}."
            )
        if self.motion_directions:
            motion = " and ".join(direction.value for direction in self.motion_directions)
            return (
                f"Keep holding the object. Move it {motion} "
                f"{preposition} the {self.destination_label}."
            )
        return f"Place the held object {preposition} the {self.destination_label}."

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["operation"] = self.operation.value
        result["placement_relation"] = (
            self.placement_relation.value if self.placement_relation else None
        )
        result["motion_directions"] = [value.value for value in self.motion_directions]
        result["grasp_substage"] = (
            self.grasp_substage.value if self.grasp_substage else None
        )
        result["phase"] = self.phase
        return result
