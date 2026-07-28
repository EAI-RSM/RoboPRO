import unittest

import numpy as np

from envs._base_task import Base_Task
from envs.utils.action import Action


class FakeEntity:
    def __init__(self, object_id):
        self.per_scene_id = object_id


class FakeArticulation:
    per_scene_id = 90

    def __init__(self, qpos):
        self.qpos = qpos

    def get_qpos(self):
        return [self.qpos]


class FakeScene:
    def __init__(self, articulation):
        self.articulation = articulation

    def get_all_articulations(self):
        return [self.articulation]


class ActionGroundingStateTest(unittest.TestCase):
    def setUp(self):
        self.task = Base_Task.__new__(Base_Task)
        self.task.FRAME_IDX = 0
        self.task._benchmark_action_nodes = []
        self.task._benchmark_held_object_ids = {"left": None, "right": None}
        self.task._benchmark_held_object_state_known = {"left": False, "right": False}
        self.task._benchmark_successful_placement_pairs = set()

    def finish(self, action, succeeded=True):
        action_id = self.task._begin_benchmark_action(action)
        self.task.FRAME_IDX += 1
        self.task._finish_benchmark_action(action_id, succeeded)
        return self.task._benchmark_action_nodes[action_id]

    @staticmethod
    def action(arm, primitive, action_type, target=None):
        kwargs = {"benchmark_action": action_type}
        if target is not None:
            kwargs["benchmark_target_object_id"] = target
        if primitive == "move":
            return Action(arm, primitive, target_pose=[0] * 7, **kwargs)
        return Action(arm, primitive, target_gripper_pos=0.0 if primitive == "close" else 1.0, **kwargs)

    def test_successful_grasp_propagates_until_successful_release(self):
        self.finish(self.action("right", "close", "grasp", 17))
        lift = self.finish(self.action("right", "move", "lift"))
        transport = self.finish(self.action("right", "move", "transport"))
        release = self.finish(self.action("right", "open", "release"))
        after_release = self.finish(self.action("right", "move", "transport"))

        self.assertEqual(lift["target_object_id"], 17)
        self.assertEqual(transport["target_object_id"], 17)
        self.assertEqual(release["target_object_id"], 17)
        self.assertIsNone(after_release["target_object_id"])

    def test_failed_grasp_does_not_replace_held_object(self):
        self.finish(self.action("left", "close", "grasp", 11))
        self.finish(self.action("left", "close", "grasp", 22), succeeded=False)
        transport = self.finish(self.action("left", "move", "transport"))
        self.assertEqual(transport["target_object_id"], 11)

    def test_failed_release_does_not_clear_held_object(self):
        self.finish(self.action("left", "close", "grasp", 11))
        self.finish(self.action("left", "open", "release"), succeeded=False)
        transport = self.finish(self.action("left", "move", "transport"))
        self.assertEqual(transport["target_object_id"], 11)

    def test_articulation_action_records_joint_position_effect(self):
        articulation = FakeArticulation(1.2)
        self.task.scene = FakeScene(articulation)
        action = Action(
            "left", "move", target_pose=[0] * 7,
            benchmark_action="close_articulation",
            benchmark_target_object_id=90,
            interaction_part="door_handle_contact_point_0",
            articulation_joint_index=0,
        )
        action_id = self.task._begin_benchmark_action(action)
        articulation.qpos = 0.4
        self.task.FRAME_IDX += 1
        self.task._finish_benchmark_action(action_id, True)

        node = self.task._benchmark_action_nodes[action_id]
        self.assertEqual(node["target_object_id"], 90)
        self.assertEqual(node["parameters"]["interaction_part"], "door_handle_contact_point_0")
        self.assertEqual(node["observed_effects"][0]["attribute"], "joint_position")
        self.assertAlmostEqual(node["observed_effects"][0]["delta"], -0.8)

    def test_move_to_pose_forwards_explicit_place_grounding(self):
        _, actions = self.task.move_to_pose(
            "right",
            [0] * 7,
            benchmark_action="place",
            benchmark_target_entity=FakeEntity(17),
            benchmark_destination_entity=FakeEntity(23),
            benchmark_phase="final_descent",
        )

        self.assertEqual(actions[0].args["benchmark_action"], "place")
        self.assertEqual(actions[0].args["benchmark_phase"], "final_descent")
        self.assertEqual(actions[0].args["benchmark_target_object_id"], 17)
        self.assertEqual(actions[0].args["benchmark_destination_object_id"], 23)

    def test_open_gripper_forwards_explicit_release_grounding(self):
        _, actions = self.task.open_gripper(
            "right",
            benchmark_action="release",
            benchmark_target_entity=FakeEntity(17),
            benchmark_destination_entity=FakeEntity(23),
            benchmark_phase="final_descent",
        )

        self.assertEqual(actions[0].args["benchmark_action"], "release")
        self.assertEqual(actions[0].args["benchmark_phase"], "final_descent")
        self.assertEqual(actions[0].args["benchmark_target_object_id"], 17)
        self.assertEqual(actions[0].args["benchmark_destination_object_id"], 23)

    def test_destination_contact_is_intentional_during_transport(self):
        action = Action(
            "left", "move", target_pose=[0] * 7,
            benchmark_action="transport",
            benchmark_target_object_id=17,
            benchmark_destination_object_id=23,
        )
        self.task._begin_benchmark_action(action)

        self.assertIn((17, 23), self.task._benchmark_intentional_contact_pairs())

    def test_successful_place_intent_persists_while_target_remains_in_destination(self):
        action = Action(
            "left", "move", target_pose=[0] * 7,
            benchmark_action="place",
            benchmark_target_object_id=17,
            benchmark_destination_object_id=23,
        )
        self.finish(action)
        contained = np.zeros((2, 2), dtype=np.bool_)
        contained[0, 1] = True

        pairs = self.task._benchmark_intentional_contact_pairs(
            in_relations=contained,
            index_by_id={17: 0, 23: 1},
        )

        self.assertIn((17, 23), pairs)
        self.assertIn((17, 23), self.task._benchmark_successful_placement_pairs)

    def test_placement_intent_expires_after_target_leaves_destination(self):
        self.task._benchmark_successful_placement_pairs = {(17, 23)}
        not_contained = np.zeros((2, 2), dtype=np.bool_)

        pairs = self.task._benchmark_intentional_contact_pairs(
            in_relations=not_contained,
            index_by_id={17: 0, 23: 1},
        )

        self.assertNotIn((17, 23), pairs)
        self.assertEqual(self.task._benchmark_successful_placement_pairs, set())

    def test_successful_regrasp_clears_prior_placement_intent(self):
        self.task._benchmark_successful_placement_pairs = {(17, 23)}

        self.finish(self.action("left", "close", "grasp", 17))

        self.assertEqual(self.task._benchmark_successful_placement_pairs, set())

    def test_arms_track_different_objects_independently(self):
        self.finish(self.action("left", "close", "grasp", 11))
        self.finish(self.action("right", "close", "grasp", 22))
        left = self.finish(self.action("left", "move", "lift"))
        right = self.finish(self.action("right", "move", "lift"))
        self.assertEqual(left["target_object_id"], 11)
        self.assertEqual(right["target_object_id"], 22)


if __name__ == "__main__":
    unittest.main()
