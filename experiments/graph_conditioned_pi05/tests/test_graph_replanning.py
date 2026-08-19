from experiments.graph_conditioned_pi05.graph_replanning import (
    ActionGraphState,
    Evidence,
    GraphControllerState,
    GraphDelta,
    GraphDeltaDetector,
    GraphEvent,
    GraspSubstage,
    PromptPhase,
    ReplanPolicy,
)


def state(**values):
    defaults = {
        "held": Evidence.FALSE,
        "destination_aligned": Evidence.FALSE,
        "release_ready": Evidence.FALSE,
        "goal_satisfied": Evidence.FALSE,
        "path_blocked": Evidence.FALSE,
        "reachable": Evidence.TRUE,
        "visible": Evidence.TRUE,
    }
    defaults.update(values)
    return ActionGraphState(**defaults)


def test_detector_debounces_and_unknown_is_not_false():
    detector = GraphDeltaDetector()
    assert detector.observe(state()).events == ()
    validated = dict(held=Evidence.TRUE, held_contact=Evidence.TRUE)
    # A transient held frame must not start placement.
    assert detector.observe(state(**validated)).events == ()
    assert detector.observe(state(held=Evidence.FALSE)).events == ()
    # Held without simultaneous contact is not a validated acquisition.
    assert detector.observe(
        state(held=Evidence.TRUE, held_contact=Evidence.FALSE)
    ).events == ()
    # Acquisition requires three consecutive held-plus-contact frames.
    assert detector.observe(state(**validated)).events == ()
    assert detector.observe(state(**validated)).events == ()
    assert detector.observe(state(**validated)).events == (
        GraphEvent.GRASP_ACQUIRED,
    )
    # Unknown frames neither emit loss nor count toward its two-frame threshold.
    assert detector.observe(state(held=Evidence.UNKNOWN)).events == ()
    assert detector.observe(state(held=Evidence.FALSE)).events == ()
    assert detector.observe(state(held=Evidence.UNKNOWN)).events == ()
    assert detector.observe(state(held=Evidence.FALSE)).events == ()
    assert detector.observe(state(held=Evidence.FALSE)).events == (
        GraphEvent.GRASP_LOST,
    )


def test_path_delta_uses_hysteresis():
    detector = GraphDeltaDetector()
    detector.observe(state())
    assert detector.observe(state(path_blocked=Evidence.TRUE)).events == ()
    assert detector.observe(state(path_blocked=Evidence.TRUE)).events == ()
    assert detector.observe(state(path_blocked=Evidence.TRUE)).events == (
        GraphEvent.PATH_BLOCKED,
    )


def test_policy_is_phase_aware_and_safety_bypasses_cooldown():
    policy = ReplanPolicy(minimum_actions=4)
    blocked = GraphDelta((GraphEvent.PATH_BLOCKED,))
    assert not policy.evaluate(PromptPhase.PLACEMENT, blocked, 2).requires_replan
    decision = policy.evaluate(PromptPhase.PLACEMENT, blocked, 4)
    assert decision.requires_replan and decision.next_phase is PromptPhase.PLACEMENT
    grasped = policy.evaluate(
        PromptPhase.GRASP, GraphDelta((GraphEvent.GRASP_ACQUIRED,)), 1
    )
    assert grasped.requires_replan and grasped.next_phase is PromptPhase.PLACEMENT


def test_controller_retains_cooldown_event_and_logs_discarded_actions():
    controller = GraphControllerState(policy=ReplanPolicy(minimum_actions=4))
    controller.observe(state(), 0)
    controller.actions_since_replan = 2
    controller.detector = GraphDeltaDetector({GraphEvent.PATH_BLOCKED: 1})
    controller.detector.observe(state())
    decision, _ = controller.observe(state(path_blocked=Evidence.TRUE), 9)
    assert not decision.requires_replan
    controller.actions_since_replan = 4
    decision, record = controller.observe(state(path_blocked=Evidence.TRUE), 7)
    assert decision.requires_replan
    assert record["discarded_actions"] == 7
    assert record["prior_phase"] == record["next_phase"] == "grasp"


def test_goal_has_priority_over_grasp_loss_during_placement():
    policy = ReplanPolicy()
    delta = GraphDelta((GraphEvent.GRASP_LOST, GraphEvent.GOAL_REACHED))
    decision = policy.evaluate(PromptPhase.PLACEMENT, delta, 20)
    assert decision.event is GraphEvent.GOAL_REACHED
    assert decision.next_phase is PromptPhase.RELEASE


def test_destination_alignment_replans_within_placement():
    detector = GraphDeltaDetector()
    detector.observe(state(destination_aligned=Evidence.FALSE))
    assert detector.observe(
        state(destination_aligned=Evidence.TRUE)
    ).events == ()
    aligned = detector.observe(state(destination_aligned=Evidence.TRUE))
    assert aligned.events == (GraphEvent.DESTINATION_ALIGNED,)
    decision = ReplanPolicy().evaluate(PromptPhase.PLACEMENT, aligned, 20)
    assert decision.requires_replan
    assert decision.next_phase is PromptPhase.PLACEMENT


def test_goal_relation_cannot_bypass_safe_descent_while_held():
    decision = ReplanPolicy().evaluate(
        PromptPhase.PLACEMENT,
        GraphDelta((GraphEvent.GOAL_REACHED,)),
        20,
        state(
            held=Evidence.TRUE,
            release_ready=Evidence.FALSE,
            goal_satisfied=Evidence.TRUE,
        ),
    )
    assert not decision.requires_replan


def test_releasing_does_not_turn_release_readiness_into_goal_loss():
    detector = GraphDeltaDetector()
    detector.observe(state(held=Evidence.TRUE))
    assert detector.observe(
        state(held=Evidence.TRUE, release_ready=Evidence.TRUE)
    ).events == ()
    ready = detector.observe(
        state(held=Evidence.TRUE, release_ready=Evidence.TRUE)
    )
    assert ready.events == (GraphEvent.RELEASE_READY,)
    decision = ReplanPolicy().evaluate(PromptPhase.PLACEMENT, ready, 20)
    assert decision.requires_replan
    assert decision.next_phase is PromptPhase.RELEASE

    # Opening the gripper necessarily makes release_ready false. Since the
    # strict in/on relation remained false throughout, this must not be
    # misreported as GOAL_LOST or trigger an immediate re-grasp.
    detector.observe(state(held=Evidence.FALSE, release_ready=Evidence.FALSE))
    after_open = detector.observe(
        state(held=Evidence.FALSE, release_ready=Evidence.FALSE)
    )
    assert GraphEvent.GOAL_LOST not in after_open.events
    assert not ReplanPolicy().evaluate(
        PromptPhase.RELEASE, after_open, 2,
        state=state(held=Evidence.FALSE),
    ).requires_replan


def test_controller_accepts_one_way_release_ready_event():
    controller = GraphControllerState(phase=PromptPhase.PLACEMENT)
    controller.observe(state(held=Evidence.TRUE), 0)
    controller.actions_since_replan = 20
    held = dict(held=Evidence.TRUE, held_contact=Evidence.TRUE)
    controller.observe(state(**held, release_ready=Evidence.TRUE), 10)
    decision, record = controller.observe(
        state(**held, release_ready=Evidence.TRUE), 9
    )
    assert decision.event is GraphEvent.RELEASE_READY
    assert controller.phase is PromptPhase.RELEASE
    assert record["trigger_event"] == "release_ready"


def test_grasp_substages_correct_alignment_and_recover_close():
    controller = GraphControllerState()
    initial, _ = controller.observe(
        state(grasp_substage=GraspSubstage.ALIGN), 0
    )
    assert not initial.requires_replan
    pending, _ = controller.observe(
        state(grasp_substage=GraspSubstage.MOVE_CLOSER, grasp_arm="left"), 31
    )
    assert not pending.requires_replan
    move_closer, record = controller.observe(
        state(grasp_substage=GraspSubstage.MOVE_CLOSER, grasp_arm="left"), 30
    )
    assert move_closer.requires_replan
    assert move_closer.reason == "grasp_substage:move_closer"
    assert controller.grasp_substage is GraspSubstage.MOVE_CLOSER
    assert record["discarded_actions"] == 30
    # Losing height alignment returns to align after two-frame persistence.
    pending_correction, _ = controller.observe(
        state(grasp_substage=GraspSubstage.ALIGN), 30
    )
    assert not pending_correction.requires_replan
    correction, _ = controller.observe(
        state(grasp_substage=GraspSubstage.ALIGN), 29
    )
    assert correction.requires_replan
    assert correction.reason == "grasp_substage:align"
    assert controller.grasp_substage is GraspSubstage.ALIGN
    close, _ = controller.observe(
        state(
            grasp_substage=GraspSubstage.CLOSE,
            grasp_arm="left",
            grasp_close_immediate=True,
        ),
        28,
    )
    assert close.requires_replan and close.reason == "grasp_substage:close"
    assert controller.grasp_substage is GraspSubstage.CLOSE
    # Once the bounded close attempt ends, invalid geometry recovers after
    # two-frame persistence.
    controller.close_latch_remaining = 0
    controller.close_attempt_count = controller.close_max_attempts
    pending_recovery, _ = controller.observe(
        state(grasp_substage=GraspSubstage.MOVE_CLOSER, grasp_arm="left"), 27
    )
    assert not pending_recovery.requires_replan
    recovery, _ = controller.observe(
        state(grasp_substage=GraspSubstage.MOVE_CLOSER, grasp_arm="left"), 26
    )
    assert recovery.requires_replan
    assert recovery.reason == "grasp_substage:move_closer"
    assert controller.grasp_substage is GraspSubstage.MOVE_CLOSER
    # A tolerated (8--10 cm) close entry is also debounced for two frames.
    tolerated_pending, _ = controller.observe(
        state(grasp_substage=GraspSubstage.CLOSE, grasp_arm="left"), 25
    )
    assert not tolerated_pending.requires_replan
    tolerated_close, _ = controller.observe(
        state(grasp_substage=GraspSubstage.CLOSE, grasp_arm="left"), 24
    )
    assert tolerated_close.requires_replan
    assert controller.grasp_substage is GraspSubstage.CLOSE


def test_close_is_latched_then_recovers_after_bounded_attempts():
    controller = GraphControllerState(close_latch_steps=2, close_max_attempts=2)
    controller.observe(state(grasp_substage=GraspSubstage.ALIGN), 0)
    close, _ = controller.observe(state(
        grasp_substage=GraspSubstage.CLOSE,
        grasp_arm="left",
        grasp_close_immediate=True,
    ), 10)
    assert close.requires_replan
    assert controller.enforced_close_arm == "left"

    # Geometry noise cannot cancel an active physical close attempt.
    controller.consume_close_action()
    held, _ = controller.observe(state(
        grasp_substage=GraspSubstage.MOVE_DOWN, grasp_arm="left"
    ), 9)
    assert not held.requires_replan
    assert controller.grasp_substage is GraspSubstage.CLOSE
    controller.consume_close_action()

    # A still-valid pose gets one more bounded attempt.
    retry, _ = controller.observe(state(
        grasp_substage=GraspSubstage.CLOSE, grasp_arm="left"
    ), 8)
    assert not retry.requires_replan
    assert controller.enforced_close_arm == "left"
    controller.consume_close_action()
    controller.consume_close_action()

    # Failure to establish held_by recovers through a corrective approach.
    pending, _ = controller.observe(state(
        grasp_substage=GraspSubstage.CLOSE, grasp_arm="left"
    ), 7)
    assert not pending.requires_replan
    recovery, _ = controller.observe(state(
        grasp_substage=GraspSubstage.CLOSE, grasp_arm="left"
    ), 6)
    assert recovery.requires_replan
    assert recovery.reason == "grasp_substage:grasp_approach"
    assert controller.grasp_substage is GraspSubstage.GRASP_APPROACH


def test_provisional_hold_keeps_close_latched_until_grasp_is_confirmed():
    controller = GraphControllerState(close_latch_steps=1, close_max_attempts=1)
    controller.observe(state(grasp_substage=GraspSubstage.ALIGN), 0)
    controller.observe(state(
        grasp_substage=GraspSubstage.CLOSE,
        grasp_arm="left",
        grasp_close_immediate=True,
    ), 10)
    controller.consume_close_action()
    assert controller.enforced_close_arm is None

    validated = dict(
        held=Evidence.TRUE,
        held_contact=Evidence.TRUE,
        held_arm="left",
        grasp_substage=GraspSubstage.CLOSE,
        grasp_arm="left",
    )
    first, _ = controller.observe(state(**validated), 9)
    assert not first.requires_replan
    assert controller.enforced_close_arm == "left"
    controller.consume_close_action()
    second, _ = controller.observe(state(**validated), 8)
    assert not second.requires_replan
    assert controller.enforced_close_arm == "left"
    controller.consume_close_action()
    confirmed, _ = controller.observe(state(**validated), 7)
    assert confirmed.event is GraphEvent.GRASP_ACQUIRED
    assert controller.phase is PromptPhase.PLACEMENT
    assert controller.held_arm == "left"
    assert controller.enforced_close_arm is None


def test_transient_hold_does_not_leave_a_stale_verification_latch():
    controller = GraphControllerState(close_latch_steps=1, close_max_attempts=1)
    controller.observe(state(grasp_substage=GraspSubstage.ALIGN), 0)
    controller.observe(state(
        grasp_substage=GraspSubstage.CLOSE,
        grasp_arm="left",
        grasp_close_immediate=True,
    ), 10)
    controller.consume_close_action()
    controller.observe(state(
        held=Evidence.TRUE,
        held_contact=Evidence.TRUE,
        held_arm="left",
        grasp_substage=GraspSubstage.CLOSE,
        grasp_arm="left",
    ), 9)
    assert controller.enforced_close_arm == "left"
    controller.consume_close_action()
    lost, _ = controller.observe(state(
        held=Evidence.FALSE,
        grasp_substage=GraspSubstage.MOVE_CLOSER,
        grasp_arm="left",
    ), 8)
    assert not lost.requires_replan
    assert controller.enforced_close_arm is None
    assert controller.phase is PromptPhase.GRASP


def test_directional_alignment_corrections_are_debounced():
    controller = GraphControllerState()
    controller.observe(state(grasp_substage=GraspSubstage.ALIGN), 0)
    first_down, _ = controller.observe(
        state(grasp_substage=GraspSubstage.MOVE_DOWN, grasp_arm="left"), 20
    )
    assert not first_down.requires_replan
    down, _ = controller.observe(
        state(grasp_substage=GraspSubstage.MOVE_DOWN, grasp_arm="left"), 19
    )
    assert down.requires_replan
    assert down.reason == "grasp_substage:move_down"
    assert controller.grasp_substage is GraspSubstage.MOVE_DOWN
    first_up, _ = controller.observe(
        state(grasp_substage=GraspSubstage.MOVE_UP, grasp_arm="left"), 18
    )
    assert not first_up.requires_replan
    up, _ = controller.observe(
        state(grasp_substage=GraspSubstage.MOVE_UP, grasp_arm="left"), 17
    )
    assert up.requires_replan
    assert up.reason == "grasp_substage:move_up"
    assert controller.grasp_substage is GraspSubstage.MOVE_UP
