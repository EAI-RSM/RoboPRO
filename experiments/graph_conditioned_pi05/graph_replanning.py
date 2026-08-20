"""Typed, debounced graph-delta decisions for graph-conditioned control."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class PromptPhase(str, Enum):
    GRASP = "grasp"
    PLACEMENT = "placement"
    RELEASE = "release"


class GraspSubstage(str, Enum):
    ALIGN = "align"
    ORIENTATION_ALIGN = "orientation_align"
    MOVE_DOWN = "move_down"
    MOVE_UP = "move_up"
    MOVE_CLOSER = "move_closer"
    GRASP_APPROACH = "grasp_approach"
    CLOSE = "close"


OPEN_GRIPPER_SUBSTAGES = frozenset({
    GraspSubstage.MOVE_CLOSER,
    GraspSubstage.GRASP_APPROACH,
    GraspSubstage.ORIENTATION_ALIGN,
})


class Evidence(str, Enum):
    UNKNOWN = "unknown"
    FALSE = "false"
    TRUE = "true"


@dataclass(frozen=True)
class TaskGoal:
    target_ids: tuple[int, ...]
    destination_ids: tuple[int, ...]
    relation: str

    def __post_init__(self):
        if not self.target_ids or not self.destination_ids:
            raise ValueError("task goal requires target and destination IDs")
        if self.relation not in {"in", "on"}:
            raise ValueError(f"unsupported placement relation: {self.relation}")


class GraphEvent(str, Enum):
    GRASP_ACQUIRED = "grasp_acquired"
    GRASP_LOST = "grasp_lost"
    DESTINATION_ALIGNED = "destination_aligned"
    DESTINATION_MISALIGNED = "destination_misaligned"
    RELEASE_READY = "release_ready"
    GOAL_REACHED = "goal_reached"
    GOAL_LOST = "goal_lost"
    PATH_BLOCKED = "path_blocked"
    PATH_CLEARED = "path_cleared"
    REACHABILITY_LOST = "reachability_lost"
    REACHABILITY_RESTORED = "reachability_restored"
    VISIBILITY_LOST = "visibility_lost"
    VISIBILITY_RESTORED = "visibility_restored"


@dataclass(frozen=True)
class ActionGraphState:
    held: Evidence = Evidence.UNKNOWN
    held_contact: Evidence = Evidence.UNKNOWN
    destination_aligned: Evidence = Evidence.UNKNOWN
    release_ready: Evidence = Evidence.UNKNOWN
    goal_satisfied: Evidence = Evidence.UNKNOWN
    path_blocked: Evidence = Evidence.UNKNOWN
    reachable: Evidence = Evidence.UNKNOWN
    visible: Evidence = Evidence.UNKNOWN
    held_arm: str | None = None
    grasp_substage: GraspSubstage = GraspSubstage.ALIGN
    grasp_arm: str | None = None
    grasp_close_immediate: bool = False
    grasp_target_contact: bool = False
    grasp_height_aligned: bool = False
    grasp_orientation_aligned: bool = False
    grasp_lateral_error_m: float = float("nan")

    def predicates(self) -> Mapping[str, Evidence]:
        # A momentary simulator ``held_by`` relation is not enough to begin
        # transport. Acquisition is valid only while the held arm also has
        # target contact; the delta detector then applies temporal persistence.
        validated_hold = self.held
        if self.held is Evidence.TRUE:
            validated_hold = self.held_contact
        return {
            "held_by": validated_hold,
            "destination_aligned": self.destination_aligned,
            "release_ready": self.release_ready,
            "goal_relation": self.goal_satisfied,
            "blocks": self.path_blocked,
            "reachable_by": self.reachable,
            "visible_to": self.visible,
        }


@dataclass(frozen=True)
class GraphDelta:
    events: tuple[GraphEvent, ...] = ()
    changed_relations: tuple[str, ...] = ()
    persistence: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplanDecision:
    requires_replan: bool
    next_phase: PromptPhase
    event: GraphEvent | None = None
    reason: str = ""


_TRANSITIONS = {
    ("held_by", Evidence.FALSE, Evidence.TRUE): GraphEvent.GRASP_ACQUIRED,
    ("held_by", Evidence.TRUE, Evidence.FALSE): GraphEvent.GRASP_LOST,
    ("destination_aligned", Evidence.FALSE, Evidence.TRUE): GraphEvent.DESTINATION_ALIGNED,
    ("destination_aligned", Evidence.TRUE, Evidence.FALSE): GraphEvent.DESTINATION_MISALIGNED,
    ("release_ready", Evidence.FALSE, Evidence.TRUE): GraphEvent.RELEASE_READY,
    ("goal_relation", Evidence.FALSE, Evidence.TRUE): GraphEvent.GOAL_REACHED,
    ("goal_relation", Evidence.TRUE, Evidence.FALSE): GraphEvent.GOAL_LOST,
    ("blocks", Evidence.FALSE, Evidence.TRUE): GraphEvent.PATH_BLOCKED,
    ("blocks", Evidence.TRUE, Evidence.FALSE): GraphEvent.PATH_CLEARED,
    ("reachable_by", Evidence.TRUE, Evidence.FALSE): GraphEvent.REACHABILITY_LOST,
    ("reachable_by", Evidence.FALSE, Evidence.TRUE): GraphEvent.REACHABILITY_RESTORED,
    ("visible_to", Evidence.TRUE, Evidence.FALSE): GraphEvent.VISIBILITY_LOST,
    ("visible_to", Evidence.FALSE, Evidence.TRUE): GraphEvent.VISIBILITY_RESTORED,
}


class GraphDeltaDetector:
    """Stabilize graph predicates and emit only effective typed changes."""

    def __init__(self, thresholds: Mapping[GraphEvent, int] | None = None):
        defaults = {
            GraphEvent.GRASP_ACQUIRED: 3,
            GraphEvent.GRASP_LOST: 2,
            GraphEvent.DESTINATION_ALIGNED: 2,
            GraphEvent.DESTINATION_MISALIGNED: 2,
            GraphEvent.RELEASE_READY: 2,
            GraphEvent.GOAL_REACHED: 2,
            GraphEvent.GOAL_LOST: 2,
            GraphEvent.PATH_BLOCKED: 3,
            GraphEvent.PATH_CLEARED: 2,
            GraphEvent.REACHABILITY_LOST: 3,
            GraphEvent.REACHABILITY_RESTORED: 2,
            GraphEvent.VISIBILITY_LOST: 3,
            GraphEvent.VISIBILITY_RESTORED: 2,
        }
        if thresholds:
            defaults.update(thresholds)
        if any(value < 1 for value in defaults.values()):
            raise ValueError("graph-delta thresholds must be positive")
        self.thresholds = defaults
        self._stable: dict[str, Evidence] = {}
        self._candidate: dict[str, tuple[Evidence, int]] = {}

    def observe(self, state: ActionGraphState) -> GraphDelta:
        events: list[GraphEvent] = []
        changed: list[str] = []
        persistence: dict[str, int] = {}
        for relation, observed in state.predicates().items():
            if observed is Evidence.UNKNOWN:
                self._candidate.pop(relation, None)
                continue
            if relation not in self._stable:
                self._stable[relation] = observed
                continue
            previous = self._stable[relation]
            if observed is previous:
                self._candidate.pop(relation, None)
                continue
            candidate, count = self._candidate.get(relation, (observed, 0))
            count = count + 1 if candidate is observed else 1
            self._candidate[relation] = (observed, count)
            event = _TRANSITIONS.get((relation, previous, observed))
            threshold = self.thresholds.get(event, 2)
            persistence[relation] = count
            if count < threshold:
                continue
            self._stable[relation] = observed
            self._candidate.pop(relation, None)
            changed.append(relation)
            if event is not None:
                events.append(event)
        return GraphDelta(tuple(events), tuple(changed), persistence)


class ReplanPolicy:
    """Map stable graph events to phase-aware chunk interruptions."""

    _PRIORITY = (
        GraphEvent.GOAL_REACHED,
        GraphEvent.RELEASE_READY,
        GraphEvent.DESTINATION_ALIGNED,
        GraphEvent.DESTINATION_MISALIGNED,
        GraphEvent.GRASP_LOST,
        GraphEvent.GRASP_ACQUIRED,
        GraphEvent.GOAL_LOST,
        GraphEvent.PATH_BLOCKED,
        GraphEvent.REACHABILITY_LOST,
        GraphEvent.VISIBILITY_LOST,
        GraphEvent.PATH_CLEARED,
        GraphEvent.REACHABILITY_RESTORED,
        GraphEvent.VISIBILITY_RESTORED,
    )
    _SAFETY_EVENTS = {
        GraphEvent.GRASP_ACQUIRED,
        GraphEvent.GRASP_LOST,
        GraphEvent.RELEASE_READY,
        GraphEvent.GOAL_REACHED,
    }

    def __init__(self, minimum_actions: int = 4):
        if minimum_actions < 0:
            raise ValueError("minimum_actions cannot be negative")
        self.minimum_actions = minimum_actions

    def evaluate(
        self,
        phase: PromptPhase | str,
        delta: GraphDelta,
        actions_since_replan: int,
        state: ActionGraphState | None = None,
    ) -> ReplanDecision:
        phase = PromptPhase(phase)
        present = set(delta.events)
        for event in self._PRIORITY:
            if event not in present:
                continue
            next_phase = self._next_phase(phase, event)
            if (
                event is GraphEvent.GOAL_REACHED
                and phase is PromptPhase.PLACEMENT
                and state is not None
                and state.held is Evidence.TRUE
                and state.release_ready is not Evidence.TRUE
            ):
                continue
            if (
                event is GraphEvent.GOAL_LOST
                and phase is PromptPhase.RELEASE
                and state is not None
                and state.held is Evidence.FALSE
            ):
                next_phase = PromptPhase.GRASP
            if next_phase is None:
                continue
            if event not in self._SAFETY_EVENTS and actions_since_replan < self.minimum_actions:
                continue
            return ReplanDecision(True, next_phase, event, event.value)
        return ReplanDecision(False, phase)

    @staticmethod
    def _next_phase(phase: PromptPhase, event: GraphEvent) -> PromptPhase | None:
        if event is GraphEvent.GRASP_ACQUIRED and phase is PromptPhase.GRASP:
            return PromptPhase.PLACEMENT
        if event is GraphEvent.GRASP_LOST and phase is PromptPhase.PLACEMENT:
            return PromptPhase.GRASP
        if event is GraphEvent.GOAL_REACHED and phase is PromptPhase.PLACEMENT:
            return PromptPhase.RELEASE
        if event is GraphEvent.RELEASE_READY and phase is PromptPhase.PLACEMENT:
            return PromptPhase.RELEASE
        if event is GraphEvent.GOAL_LOST and phase is PromptPhase.RELEASE:
            return PromptPhase.PLACEMENT
        if event in {
            GraphEvent.DESTINATION_ALIGNED,
            GraphEvent.DESTINATION_MISALIGNED,
        } and phase is PromptPhase.PLACEMENT:
            return phase
        if event in {GraphEvent.PATH_BLOCKED, GraphEvent.PATH_CLEARED} and phase in {
            PromptPhase.GRASP,
            PromptPhase.PLACEMENT,
        }:
            return phase
        if event in {
            GraphEvent.REACHABILITY_LOST,
            GraphEvent.REACHABILITY_RESTORED,
            GraphEvent.VISIBILITY_LOST,
            GraphEvent.VISIBILITY_RESTORED,
        } and phase is PromptPhase.GRASP:
            return phase
        return None


@dataclass
class GraphControllerState:
    phase: PromptPhase = PromptPhase.GRASP
    held_arm: str | None = None
    actions_since_replan: int = 0
    frame: int = 0
    detector: GraphDeltaDetector = field(default_factory=GraphDeltaDetector)
    policy: ReplanPolicy = field(default_factory=ReplanPolicy)
    pending_events: set[GraphEvent] = field(default_factory=set)
    grasp_substage: GraspSubstage = GraspSubstage.ALIGN
    grasp_arm: str | None = None
    _grasp_candidate: GraspSubstage | None = None
    _grasp_candidate_count: int = 0
    close_latch_steps: int = 15
    close_max_attempts: int = 2
    close_latch_remaining: int = 0
    close_attempt_count: int = 0
    validated_contact_steps: int = 2
    contact_close_lateral_error_max_m: float = 0.018
    _contact_candidate_arm: str | None = None
    _contact_candidate_count: int = 0

    def _validated_contact_close(self, state: ActionGraphState) -> bool:
        """Validate stable, aligned target contact before relaxing centering."""
        qualified = (
            self.phase is PromptPhase.GRASP
            and state.grasp_arm is not None
            and state.grasp_target_contact
            and state.grasp_height_aligned
            and state.grasp_orientation_aligned
            and state.grasp_lateral_error_m
            <= self.contact_close_lateral_error_max_m
        )
        if not qualified:
            self._contact_candidate_arm = None
            self._contact_candidate_count = 0
            return False
        if state.grasp_arm == self._contact_candidate_arm:
            self._contact_candidate_count += 1
        else:
            self._contact_candidate_arm = state.grasp_arm
            self._contact_candidate_count = 1
        return self._contact_candidate_count >= self.validated_contact_steps

    @property
    def enforced_close_arm(self) -> str | None:
        """Arm whose close command is currently latched, if any."""
        if (
            self.phase is PromptPhase.GRASP
            and self.grasp_substage is GraspSubstage.CLOSE
            and self.close_latch_remaining > 0
        ):
            return self.grasp_arm
        return None

    def consume_close_action(self) -> None:
        """Account for one low-level action executed under the close latch."""
        if self.enforced_close_arm is not None:
            self.close_latch_remaining -= 1

    def _update_grasp_substage(self, state: ActionGraphState) -> bool:
        """Debounce grasp substages, including recoverable close attempts."""
        if self.phase is not PromptPhase.GRASP:
            return False
        contact_close = self._validated_contact_close(state)
        desired = GraspSubstage.CLOSE if contact_close else state.grasp_substage
        retry_exhausted = False
        # Once a geometrically authorized close begins, transient pose changes
        # must not cancel it before the gripper has had time to respond.
        if (
            self.grasp_substage is GraspSubstage.CLOSE
            and self.close_latch_remaining > 0
            and desired is not GraspSubstage.CLOSE
        ):
            return False
        # A close prompt that did not establish a validated hold gets one
        # bounded retry. After that, request a small corrective approach.
        if (
            self.grasp_substage is GraspSubstage.CLOSE
            and self.close_latch_remaining == 0
            and desired is GraspSubstage.CLOSE
            and state.held is not Evidence.TRUE
        ):
            if self.close_attempt_count < self.close_max_attempts:
                self.close_attempt_count += 1
                self.close_latch_remaining = self.close_latch_steps
                return False
            desired = GraspSubstage.GRASP_APPROACH
            retry_exhausted = True
        if desired is self.grasp_substage:
            self._grasp_candidate = None
            self._grasp_candidate_count = 0
            return False
        if desired is self._grasp_candidate:
            self._grasp_candidate_count += 1
        else:
            self._grasp_candidate = desired
            self._grasp_candidate_count = 1
        threshold = (
            1
            if retry_exhausted
            or (
                desired is GraspSubstage.CLOSE
                and (state.grasp_close_immediate or contact_close)
            )
            else 2
        )
        if self._grasp_candidate_count < threshold:
            return False
        self.grasp_substage = desired
        self.grasp_arm = state.grasp_arm
        if desired is GraspSubstage.CLOSE:
            self.close_attempt_count = 1
            self.close_latch_remaining = self.close_latch_steps
        else:
            self.close_latch_remaining = 0
            self.close_attempt_count = 0
        self._grasp_candidate = None
        self._grasp_candidate_count = 0
        return True

    def observe(self, state: ActionGraphState, remaining_actions: int) -> tuple[ReplanDecision, dict]:
        self.frame += 1
        prior_substage = self.grasp_substage
        substage_changed = self._update_grasp_substage(state)
        if state.held is Evidence.TRUE and state.held_arm is not None:
            self.held_arm = state.held_arm
            if (
                state.held_contact is Evidence.TRUE
                and self.phase is PromptPhase.GRASP
                and self.grasp_substage is GraspSubstage.CLOSE
            ):
                # Keep closure active until the delta detector confirms the
                # hold. One provisional frame must never expose the policy's
                # open command before the persistence window completes.
                self.close_latch_remaining = max(
                    self.close_latch_remaining, 1
                )
        delta = self.detector.observe(state)
        opposites = {
            GraphEvent.PATH_BLOCKED: GraphEvent.PATH_CLEARED,
            GraphEvent.PATH_CLEARED: GraphEvent.PATH_BLOCKED,
            GraphEvent.REACHABILITY_LOST: GraphEvent.REACHABILITY_RESTORED,
            GraphEvent.REACHABILITY_RESTORED: GraphEvent.REACHABILITY_LOST,
            GraphEvent.VISIBILITY_LOST: GraphEvent.VISIBILITY_RESTORED,
            GraphEvent.VISIBILITY_RESTORED: GraphEvent.VISIBILITY_LOST,
            GraphEvent.GOAL_REACHED: GraphEvent.GOAL_LOST,
            GraphEvent.DESTINATION_ALIGNED: GraphEvent.DESTINATION_MISALIGNED,
            GraphEvent.DESTINATION_MISALIGNED: GraphEvent.DESTINATION_ALIGNED,
            GraphEvent.GOAL_LOST: GraphEvent.GOAL_REACHED,
            GraphEvent.GRASP_ACQUIRED: GraphEvent.GRASP_LOST,
            GraphEvent.GRASP_LOST: GraphEvent.GRASP_ACQUIRED,
        }
        for event in delta.events:
            opposite = opposites.get(event)
            if opposite is not None:
                self.pending_events.discard(opposite)
            self.pending_events.add(event)
        effective_delta = GraphDelta(
            tuple(self.pending_events), delta.changed_relations, delta.persistence
        )
        decision = self.policy.evaluate(
            self.phase, effective_delta, self.actions_since_replan, state=state
        )
        if not decision.requires_replan and substage_changed and self.frame > 1:
            decision = ReplanDecision(
                True,
                self.phase,
                reason=f"grasp_substage:{self.grasp_substage.value}",
            )
        record = {
            "frame": self.frame,
            "prior_phase": self.phase.value,
            "events": [event.value for event in delta.events],
            "changed_relations": list(delta.changed_relations),
            "persistence": dict(delta.persistence),
            "requires_replan": decision.requires_replan,
            "trigger_event": decision.event.value if decision.event else None,
            "reason": decision.reason,
            "discarded_actions": remaining_actions if decision.requires_replan else 0,
            "next_phase": decision.next_phase.value,
            "prior_grasp_substage": prior_substage.value,
            "next_grasp_substage": self.grasp_substage.value,
            "grasp_arm": self.grasp_arm,
            "close_latch_remaining": self.close_latch_remaining,
            "close_attempt_count": self.close_attempt_count,
            "validated_contact_arm": self._contact_candidate_arm,
            "validated_contact_count": self._contact_candidate_count,
            "validated_contact_close": (
                self._contact_candidate_count >= self.validated_contact_steps
            ),
        }
        if decision.requires_replan:
            self.phase = decision.next_phase
            self.actions_since_replan = 0
            self.pending_events.clear()
            if self.phase is PromptPhase.GRASP:
                self.held_arm = None
                if decision.event is GraphEvent.GRASP_LOST:
                    self.grasp_substage = GraspSubstage.ALIGN
                    self.grasp_arm = None
                    self._grasp_candidate = None
                    self._grasp_candidate_count = 0
                    self.close_latch_remaining = 0
                    self.close_attempt_count = 0
                    self._contact_candidate_arm = None
                    self._contact_candidate_count = 0
            else:
                # Placement immediately takes over with its existing held-arm
                # latch, so the close-attempt bookkeeping can be retired.
                self.close_latch_remaining = 0
                self.close_attempt_count = 0
                self._contact_candidate_arm = None
                self._contact_candidate_count = 0
            record["next_grasp_substage"] = self.grasp_substage.value
            record["grasp_arm"] = self.grasp_arm
        return decision, record
