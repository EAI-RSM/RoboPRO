from customized_robotwin.policy.pi05.deploy_policy import (
    _truncate_actions_to_step_limit,
)


class Task:
    def __init__(self, count=None, limit=None):
        if count is not None:
            self.take_action_cnt = count
        if limit is not None:
            self.step_lim = limit


def test_action_chunk_is_truncated_to_remaining_episode_steps():
    actions = [[0.0] * 14 for _ in range(50)]
    assert len(_truncate_actions_to_step_limit(Task(598, 600), actions)) == 2
    assert len(_truncate_actions_to_step_limit(Task(600, 600), actions)) == 0
    assert len(_truncate_actions_to_step_limit(Task(601, 600), actions)) == 0


def test_missing_step_budget_preserves_actions():
    actions = [[0.0] * 14 for _ in range(50)]
    assert _truncate_actions_to_step_limit(Task(), actions) is actions


def main():
    test_action_chunk_is_truncated_to_remaining_episode_steps()
    test_missing_step_budget_preserves_actions()
    print("2 deploy-policy checks passed")


if __name__ == "__main__":
    main()
