from experiments.graph_conditioned_pi05.graph_replanning import (
    ActionGraphState,
    Evidence,
    GraphControllerState,
    GraphDelta,
    GraphDeltaDetector,
    GraphEvent,
    PromptPhase,
    ReplanPolicy,
)


def state(**values):
    defaults = {
        "held": Evidence.FALSE,
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
