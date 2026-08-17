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
    assert detector.observe(state(held=Evidence.TRUE)).events == (
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
    controller.observe(
        state(held=Evidence.TRUE, release_ready=Evidence.TRUE), 10
    )
    decision, record = controller.observe(
        state(held=Evidence.TRUE, release_ready=Evidence.TRUE), 9
    )
    assert decision.event is GraphEvent.RELEASE_READY
    assert controller.phase is PromptPhase.RELEASE
    assert record["trigger_event"] == "release_ready"


def test_grasp_substages_advance_monotonically_and_interrupt_chunks():
    controller = GraphControllerState()
    initial, _ = controller.observe(
        state(grasp_substage=GraspSubstage.APPROACH, grasp_arm="left"), 0
    )
    assert not initial.requires_replan
    pending, _ = controller.observe(
        state(grasp_substage=GraspSubstage.ALIGN, grasp_arm="left"), 31
    )
    assert not pending.requires_replan
    align, record = controller.observe(
        state(grasp_substage=GraspSubstage.ALIGN, grasp_arm="left"), 30
    )
    assert align.requires_replan and align.reason == "grasp_substage:align"
    assert controller.grasp_substage is GraspSubstage.ALIGN
    assert record["discarded_actions"] == 30
    # Noisy distance evidence cannot regress align back to approach.
    regressed, _ = controller.observe(
        state(grasp_substage=GraspSubstage.APPROACH, grasp_arm="left"), 30
    )
    assert not regressed.requires_replan
    assert controller.grasp_substage is GraspSubstage.ALIGN
    close, _ = controller.observe(
        state(grasp_substage=GraspSubstage.CLOSE, grasp_arm="left"), 29
    )
    assert close.requires_replan and close.reason == "grasp_substage:close"
    assert controller.grasp_substage is GraspSubstage.CLOSE
