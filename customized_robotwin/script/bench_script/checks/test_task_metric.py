#!/usr/bin/env python3
"""Focused CPU checks for task roles, nominal waypoints, and Stage C2 artifacts."""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import analyze_metric_distribution as dist
from task_metric import _save_scene_images
from lib.scene_provenance import scene_fingerprint, scene_id
from lib.task_roles import resolve_task_roles
from lib.waypoints import canonical_legs, canonical_waypoints


class FakePose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)


class FakeActor:
    def __init__(self, name, p, contact=None):
        self._name = name
        self._pose = FakePose(p)
        self._contact = contact
        self.scale = 1.0

    def get_name(self):
        return self._name

    def get_pose(self):
        return self._pose

    def get_contact_point(self, index, ret):
        assert index == 0 and ret == "matrix"
        return self._contact.copy()


class FakeRobot:
    def get_left_ee_pose(self):
        return [-0.4, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]

    def get_right_ee_pose(self):
        return [0.4, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]


class FakeCameras:
    static_camera_name = ["countertop_camera", "demo_camera", "demo_camera_2"]

    def __init__(self):
        self.rendered = None

    def update_picture(self, camera_names=None):
        self.rendered = list(camera_names)

    def get_rgb(self, camera_names=None):
        return {
            name: {"rgb": np.full((8, 10, 3), index / 4.0, dtype=float)}
            for index, name in enumerate(camera_names, 1)
        }


def _fake_env(root):
    for name, model_id in (("021_cup", "2"), ("019_coaster", "0"), ("block", "3")):
        path = root / "assets" / "objects" / name / "collision"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"base{model_id}.glb").write_bytes(b"mesh")
    contact = np.eye(4)
    contact[:3, 3] = [0.2, -0.1, 0.9]
    target = FakeActor("021_cup", [0.2, -0.1, 0.82], contact)
    destination = FakeActor("019_coaster", [-0.2, 0.15, 0.82])
    obstacle = FakeActor("block", [0.0, 0.0, 0.85])
    return SimpleNamespace(
        target_obj=target,
        target_name="021_cup",
        target_id=2,
        des_obj=destination,
        des_obj_id=0,
        des_obj_pose=[-0.2, 0.10, 0.82, 1.0, 0.0, 0.0, 0.0],
        side_to_place="right",
        robot=FakeRobot(),
        collision_list=[
            {"actor": target, "collision_path": str(root / "assets/objects/021_cup/collision/base2.glb")},
            {"actor": destination, "collision_path": str(root / "assets/objects/019_coaster/collision/base0.glb")},
            {"actor": obstacle, "collision_path": str(root / "assets/objects/block/collision/base3.glb"), "is_obstacle": True},
        ],
    )


def test_roles_and_waypoints():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        previous = os.environ.get("BENCH_ROOT")
        os.environ["BENCH_ROOT"] = str(root)
        try:
            env = _fake_env(root)
            roles = resolve_task_roles(env, "put_cup_on_coaster")
            assert roles.target.actor is env.target_obj
            assert roles.destination.actor is env.des_obj
            assert [role.name for role in roles.obstacles] == ["block"]
            first = canonical_waypoints(env, "put_cup_on_coaster")
            second = canonical_waypoints(env, "put_cup_on_coaster")
        finally:
            if previous is None:
                os.environ.pop("BENCH_ROOT", None)
            else:
                os.environ["BENCH_ROOT"] = previous
    assert first == second
    legs = canonical_legs(first)
    assert [leg.kind for leg in legs] == ["pre_grasp", "grasp", "carry", "place"]
    assert [leg.gripper_state for leg in legs] == ["empty", "empty", "holding", "holding"]
    assert first[0].xyz == (0.4, 0.0, 0.9)
    print("  [1] explicit roles and deterministic planning-free waypoint states     PASS")


def _record(scene, seed, values, clutter, outcome=False):
    legs = []
    kinds = ["pre_grasp", "grasp", "carry", "place"]
    for index, (kind, value) in enumerate(zip(kinds, values)):
        legs.append(
            {
                "index": index,
                "kind": kind,
                "eps_geom": None if np.isposinf(value) else float(value),
                "eps_geom_unbounded": bool(np.isposinf(value)),
            }
        )
    minimum = min(values)
    row = {
        "schema": "robopro.task-metric.v1",
        "status": "ok",
        "scene_id": scene,
        "seed": seed,
        "obstacle_density": 10,
        "clutter_count": clutter,
        "eps_geom_min": None if np.isposinf(minimum) else float(minimum),
        "eps_geom_min_unbounded": bool(np.isposinf(minimum)),
        "legs": legs,
    }
    if outcome:
        row["hard_success"] = True
    return row


def test_distribution_artifacts_and_rejection():
    records = [
        _record("scene-a", 1, [0.02, 0.03, 0.04, 0.05], 8),
        _record("scene-b", 2, [0.02, 0.03, 0.04, 0.06], 8),
        _record("scene-c", 3, [np.inf, np.inf, np.inf, np.inf], 9),
    ]
    summary = dist.summarize_records(records)
    assert summary["n"] == 3 and summary["positive_infinity_n"] == 1
    assert summary["tie_group_count"] == 1
    assert summary["realized_clutter_count_frequencies"] == {"8": 2, "9": 1}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        source.mkdir()
        (source / "records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        (source / "config.json").write_text("{}", encoding="utf-8")
        out = dist.run(SimpleNamespace(records=str(source), out_dir=str(root / "out")))
        expected = {
            "eps_geom_min_distribution.png",
            "eps_geom_by_leg.png",
            "eps_geom_min_by_scene.png",
            "distribution_summary.json",
            "source_records.json",
            "timings.json",
        }
        assert expected <= {path.name for path in out.iterdir()}
        assert not (out / "bucket_spec.json").exists()

        bad = root / "bad.jsonl"
        bad.write_text(json.dumps(_record("scene-z", 9, [0.1] * 4, 8, outcome=True)) + "\n")
        try:
            dist.read_metric_records(bad)
        except ValueError as exc:
            assert "outcome-bearing" in str(exc)
        else:
            raise AssertionError("Stage C2 accepted an outcome-bearing record")

        all_inf = [_record("scene-inf", 4, [np.inf] * 4, 8)]
        dist.plot_primary_distribution(all_inf, root / "all_inf.png")
        dist.plot_by_leg(all_inf, root / "all_inf_legs.png")
        assert (root / "all_inf.png").is_file() and (root / "all_inf_legs.png").is_file()
    print("  [2] finite/tied/+inf plots and strict outcome-field rejection          PASS")


def test_scene_images():
    cameras = FakeCameras()
    env = SimpleNamespace(cameras=cameras, _update_render=lambda: None)
    with tempfile.TemporaryDirectory() as tmp:
        paths = _save_scene_images(
            env,
            Path(tmp),
            7,
            ("countertop_camera", "demo_camera", "demo_camera_2"),
        )
        assert cameras.rendered == ["countertop_camera", "demo_camera", "demo_camera_2"]
        assert all((Path(tmp) / path).is_file() for path in paths)
        assert paths == [
            "scene_images/seed0007_countertop_camera.png",
            "scene_images/seed0007_demo_camera.png",
            "scene_images/seed0007_demo_camera_2.png",
        ]
    print("  [3] initialized scene views render once and save three PNGs            PASS")


def test_scene_identity_stability():
    intent = {"task": "put_cup_on_coaster", "seed": 3, "density": 10}
    assert scene_id(intent) == scene_id(dict(reversed(list(intent.items()))))
    first_hash, first_source = scene_fingerprint({"p": [0.1, 0.2, 0.3], **intent})
    second_hash, second_source = scene_fingerprint({**intent, "p": np.array([0.1, 0.2, 0.3])})
    assert first_hash == second_hash and first_source == second_source
    print("  [4] canonical scene ID and exact fingerprint serialization            PASS")


def main():
    print("task metric Stage C/C2 -- CPU checks")
    test_roles_and_waypoints()
    test_distribution_artifacts_and_rejection()
    test_scene_images()
    test_scene_identity_stability()
    print("ALL PASS")


if __name__ == "__main__":
    main()
