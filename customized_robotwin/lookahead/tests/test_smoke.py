"""Sim-free smoke tests for the lookahead core.

Exercises the PURE modules (``rollback``, ``fitness``, ``search``, ``plan``,
``pins``) end to end against a tiny FAKE task that mimics just the RoboTwin surface
the core touches: a PhysX ``pack``/``unpack``, actors/articulations with poses, a
latched ``eval_success`` flag, ``take_action`` and ``check_success``. No jax, no
sapien, no GPU — this proves search + pins + plan save/load work without the sim.

Runnable two ways::

    python -m lookahead.tests.test_smoke          # from customized_robotwin
    python lookahead/tests/test_smoke.py          # direct
    pytest lookahead/tests/test_smoke.py          # if pytest is available
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

# resolve the ``lookahead`` package whether run via -m or as a bare file
try:
    from lookahead import fitness as fit_mod
    from lookahead import pins as pins_mod
    from lookahead import plan as plan_mod
    from lookahead import rollback as rb
    from lookahead import search as search_mod
    from lookahead import candidates as cand_mod
    from lookahead.candidates import CandidateSource
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from lookahead import fitness as fit_mod
    from lookahead import pins as pins_mod
    from lookahead import plan as plan_mod
    from lookahead import rollback as rb
    from lookahead import search as search_mod
    from lookahead import candidates as cand_mod
    from lookahead.candidates import CandidateSource


# ============================================================ the fake task
class _Pose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


class _Joint:
    """A drive joint whose targets are (de)restored by the rollback core."""
    def __init__(self):
        self._t, self._v = 0.0, 0.0

    def get_drive_target(self):
        return self._t

    def get_drive_velocity_target(self):
        return self._v

    def set_drive_target(self, t):
        self._t = t

    def set_drive_velocity_target(self, v):
        self._v = v


class _Actor:
    def __init__(self, name, pose_fn):
        self._name = name
        self._pose_fn = pose_fn

    def get_name(self):
        return self._name

    def get_pose(self):
        return self._pose_fn()


class _Articulation:
    def __init__(self, task):
        self._task = task
        self._joints = [_Joint()]

    def get_root_pose(self):
        return _Pose([0.0, 0.0, 0.0])

    def get_qpos(self):
        return np.array([self._task.pos], dtype=np.float64)

    def get_qvel(self):
        return np.array([0.0], dtype=np.float64)

    def get_active_joints(self):
        return self._joints


class _Physx:
    def __init__(self, task):
        self._task = task

    def pack(self):
        # opaque blob: full mutable dynamic state (here just the 1-D "pos")
        return {"pos": float(self._task.pos)}

    def unpack(self, blob):
        self._task.pos = float(blob["pos"])


class _Scene:
    def __init__(self, task):
        self._task = task
        self._physx = _Physx(task)
        target = _Actor("target", lambda: _Pose([task.pos, 0.0, 0.1]))
        dest = _Actor("dest", lambda: _Pose([task.goal, 0.0, 0.1]))
        clutter = _Actor("clutter", lambda: _Pose([-0.5, 0.5, 0.1]))
        self._actors = [target, dest, clutter]
        self._arts = [_Articulation(task)]

    def get_physx_system(self):
        return self._physx

    def get_all_actors(self):
        return self._actors

    def get_all_articulations(self):
        return self._arts

    def step(self):
        pass


class FakeTask:
    """1-D reach task: ``take_action`` nudges ``pos`` toward ``goal``.

    Exposes exactly the ``TaskAdapter`` surface plus the optional
    ``target_obj``/``des_obj``/object-pose hooks the OracleFitness uses.
    """

    def __init__(self, goal=1.0):
        self.pos = 0.0
        self.goal = float(goal)
        self.eval_success = False
        self.step_lim = 1000
        self.take_action_cnt = 0
        self.scene = _Scene(self)
        self.target_obj = self.scene._actors[0]
        self.des_obj = self.scene._actors[1]

    # -- TaskAdapter surface --
    def take_action(self, action):
        if self.eval_success:
            return                       # latched: no-op after success (as in RoboTwin)
        self.take_action_cnt += 1
        self.pos += float(np.asarray(action).ravel()[0])
        if self.check_success():
            self.eval_success = True

    def check_success(self):
        return self.pos >= self.goal

    def get_obs(self):
        return {"pos": self.pos}

    def _update_render(self):
        pass

    # -- optional fitness hooks --
    def get_object_poses(self):
        ids = [0, 1, 2]
        poses = np.array([
            [self.pos, 0.0, 0.1, 1, 0, 0, 0],
            [self.goal, 0.0, 0.1, 1, 0, 0, 0],
            [-0.5, 0.5, 0.1, 1, 0, 0, 0],
        ], dtype=np.float64)
        return ids, poses

    def get_actor_id_map(self):
        return {0: "target", 1: "dest", 2: "clutter"}

    def close_env(self):
        pass


class FakeCandidates(CandidateSource):
    """K constant chunks with increasing per-step deltas (larger = more progress)."""
    def __init__(self, base=0.08, chunk_len=4, action_dim=14):
        self.base = base
        self.chunk_len = chunk_len
        self.action_dim = action_dim

    def propose(self, task, k):
        out = []
        for i in range(k):
            delta = self.base * (i + 1)
            out.append(np.full((self.chunk_len, self.action_dim), delta, dtype=np.float32))
        return out


class _FakePolicy:
    """Minimal deploy-policy .policy: infer() varies with noise and latent z."""
    def __init__(self):
        self._sample_kwargs = {}

    def infer(self, obs, noise=None):
        z = float(self._sample_kwargs.get("z", -1))
        base = float(np.asarray(noise).mean()) if noise is not None else 0.0
        # each chunk is constant-valued so distinct (base, z) => distinct chunks
        return {"actions": np.full((50, 32), base + 10.0 * z, dtype=np.float32)}


class _FakeModel:
    """Minimal deploy-policy model surface PolicyCandidates duck-types."""
    def __init__(self):
        self.observation_window = None
        self.policy = _FakePolicy()
        self.rendered = 0

    def update_observation_window(self, rgb, state):
        self.observation_window = {"rgb": rgb, "state": state}
        self.rendered += 1


def _encode_obs(obs):
    return [obs], obs


# ============================================================ tests
def test_snapshot_restore_roundtrip():
    task = FakeTask()
    snap = rb.snapshot(task)
    fp0 = snap["fingerprint"].copy()
    rb.apply_chunk(task, np.full((4, 14), 0.1, dtype=np.float32))
    assert task.pos > 0.0
    assert not np.allclose(rb.state_fingerprint(task), fp0)
    task.eval_success = True
    rb.restore(task, snap)
    assert task.pos == 0.0
    assert task.eval_success is False                    # restore MUST clear the latch
    assert np.allclose(rb.state_fingerprint(task), fp0)


def test_apply_chunk_latches_success():
    task = FakeTask(goal=0.4)
    # a chunk that overshoots: take_action must stop advancing pos after success
    rb.apply_chunk(task, np.full((10, 14), 0.1, dtype=np.float32))
    assert task.check_success()
    assert task.eval_success is True
    assert task.pos < 0.5                                # stopped at first success (not all 10 steps)
    assert abs(task.pos - 0.4) < 1e-5                    # ~0.4 (float32 accumulation)


def test_oracle_fitness_outcome_and_score():
    task = FakeTask(goal=1.0)
    fit = fit_mod.OracleFitness()
    ctx = fit.make_context(task)
    o0 = fit.outcome(task, ctx)
    assert o0["success"] is False and o0["tier"] == "FAIL"
    assert o0["collided"] is False                       # excluded target moved; clutter didn't
    assert abs(o0["dist_to_goal"] - 1.0) < 1e-9
    # advance closer -> smaller dist -> better (lower) score
    rb.apply_chunk(task, np.full((5, 14), 0.1, dtype=np.float32))
    o1 = fit.outcome(task, ctx)
    assert fit.score(o1) < fit.score(o0)


def test_beam_search_finds_success_and_replays():
    task = FakeTask(goal=1.0)
    root = rb.snapshot(task)
    cands = FakeCandidates(base=0.08, chunk_len=4)
    fit = fit_mod.OracleFitness()
    results = search_mod.BeamSearch(width=3, k=4, depth=4).search(task, root, cands, fit)
    assert isinstance(results, list) and len(results) == 1     # m defaults to 1
    res = results[0]
    assert res.success is True
    assert res.depth >= 1
    assert res.actions.shape[1] == 14
    assert res.score[0] == 0.0                           # HARD tier

    # FROM-START RAW-ACTION REPLAY reproduces the searched terminal (the whole point)
    rb.restore(task, root)
    rb.apply_chunk(task, res.actions)
    assert task.check_success()
    assert np.allclose(rb.state_fingerprint(task), res.terminal_fingerprint, atol=1e-6)


def test_top_m_ranked_distinct_ordered():
    task = FakeTask(goal=1.0)
    root = rb.snapshot(task)
    cands = FakeCandidates(base=0.08, chunk_len=4)
    fit = fit_mod.OracleFitness()
    M = 5
    results = search_mod.BeamSearch(width=3, k=4, depth=4).search(
        task, root, cands, fit, m=M)
    assert 1 < len(results) <= M                          # <= M plans
    # score-ordered (best first), non-decreasing
    scores = [r.score for r in results]
    assert scores == sorted(scores)
    # distinct by committed candidate-index path (no near-identical plans)
    paths = [tuple(r.candidate_indices) for r in results]
    assert len(set(paths)) == len(paths)
    # rank-0 is the single-best plan (identical to m=1)
    best1 = search_mod.BeamSearch(width=3, k=4, depth=4).search(task, root, cands, fit)[0]
    assert results[0].score == best1.score
    assert np.allclose(results[0].actions, best1.actions, atol=1e-6)

    # MonteCarlo + FullTree also honour m (<= M, distinct, ordered)
    for strat in (search_mod.MonteCarloSearch(n_samples=20, k=4, depth=4, seed=1),
                  search_mod.FullTreeSearch(k=3, depth=3, max_leaves=64)):
        t = FakeTask(goal=1.0)
        r = rb.snapshot(t)
        rs = strat.search(t, r, cands, fit, m=M)
        assert 1 <= len(rs) <= M
        assert [x.score for x in rs] == sorted(x.score for x in rs)
        ps = [tuple(x.candidate_indices) for x in rs]
        assert len(set(ps)) == len(ps)


def test_policy_candidates_proposer_routing():
    # default proposer = noise: K distinct chunks (K different noises), obs rendered
    model = _FakeModel()
    pc = cand_mod.PolicyCandidates(model, _encode_obs, k=4, chunk=50)
    assert pc.propose_fn is cand_mod.noise_propose        # noise is the default
    chunks = pc.propose(FakeTask(), 4)
    assert model.rendered == 1                            # rendered the obs once
    assert len(chunks) == 4 and chunks[0].shape == (50, 32)
    vals = {round(float(c[0, 0]), 6) for c in chunks}
    assert len(vals) == 4                                 # 4 distinct noise samples

    # a provided propose_fn is used instead of the default
    sentinel_used = {"n": 0}

    def my_propose(m, obs, k):
        sentinel_used["n"] += 1
        return [np.full((50, 32), 99.0, dtype=np.float32) for _ in range(k)]

    pc2 = cand_mod.PolicyCandidates(model, _encode_obs, k=3, propose_fn=my_propose, chunk=10)
    chunks2 = pc2.propose(FakeTask(), 3)
    assert sentinel_used["n"] == 1                        # our proposer was called
    assert len(chunks2) == 3 and chunks2[0].shape == (10, 32)  # chunk slicing applied
    assert all(float(c[0, 0]) == 99.0 for c in chunks2)

    # mode_propose helper varies the latent z -> distinct chunks
    model.update_observation_window([0], 0)
    modes = cand_mod.mode_propose(model, model.observation_window, 5)
    assert len({round(float(c[0, 0]), 6) for c in modes}) == 5


def test_montecarlo_and_fulltree():
    fit = fit_mod.OracleFitness()
    cands = FakeCandidates(base=0.12, chunk_len=4)

    task = FakeTask(goal=1.0)
    root = rb.snapshot(task)
    mc = search_mod.MonteCarloSearch(n_samples=12, k=4, depth=4, seed=0).search(
        task, root, cands, fit)[0]
    assert mc.success is True

    task2 = FakeTask(goal=1.0)
    root2 = rb.snapshot(task2)
    ft = search_mod.FullTreeSearch(k=2, depth=3, max_leaves=64).search(
        task2, root2, cands, fit)[0]
    assert ft.success is True
    # replay verify for fulltree too
    rb.restore(task2, root2)
    rb.apply_chunk(task2, ft.actions)
    assert np.allclose(rb.state_fingerprint(task2), ft.terminal_fingerprint, atol=1e-6)


def test_fulltree_guard():
    fit = fit_mod.OracleFitness()
    cands = FakeCandidates()
    task = FakeTask()
    root = rb.snapshot(task)
    try:
        search_mod.FullTreeSearch(k=6, depth=6, max_leaves=100).search(task, root, cands, fit)
        assert False, "expected FullTreeSearch to guard against a huge tree"
    except ValueError:
        pass


def test_plan_save_load_roundtrip():
    task = FakeTask(goal=1.0)
    root = rb.snapshot(task)
    res = search_mod.BeamSearch(width=3, k=4, depth=4).search(
        task, root, FakeCandidates(), fit_mod.OracleFitness())[0]
    spec = plan_mod.TaskSpec(
        task_name="fake_reach", task_config="unit", seed=42,
        embodiment={"embodiment": ["aloha"], "embodiment_name": "aloha"},
        domain_randomization={"cluttered_table": False, "random_table_height": 0},
        control={"action_dim": 14, "chunk": 4, "control_freq": None},
        provenance={"robotwin_commit": "deadbeef", "bench_config_hash": "abc123",
                    "created_at": "2026-07-20T00:00:00+00:00"})
    plan = plan_mod.Plan(
        task_spec=spec, actions=res.actions.tolist(),
        fingerprints={"t0": root["fingerprint"].tolist(),
                      "terminal": res.terminal_fingerprint.tolist()},
        meta={"strategy": "beam", "fitness": "oracle", "candidate_policy": "fake",
              "score": list(res.score), "success": res.success})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fake_reach", "unit", "seed42.json")
        plan_mod.save(plan, path)
        loaded = plan_mod.load(path)
    assert loaded.task_spec.task_name == "fake_reach"
    assert loaded.task_spec.seed == 42
    assert np.allclose(loaded.actions_array(), res.actions, atol=1e-6)
    assert np.allclose(loaded.fingerprints["terminal"], res.terminal_fingerprint, atol=1e-6)
    assert loaded.meta["strategy"] == "beam"


def test_pins_enforce_warn_off_and_atol():
    # default policy: all enforce
    pol = pins_mod.default_policy()
    assert pol.check("task_config", ("t", "c"), ("t", "c")) is True
    try:
        pol.check("task_config", ("t", "c"), ("t", "OTHER"))
        assert False, "enforce mismatch should raise"
    except pins_mod.PinViolation:
        pass

    # numeric pin honours atol
    a = [1.0, 2.0, 3.0]
    b = [1.0 + 1e-7, 2.0, 3.0]
    assert pol.check("fingerprint_t0", a, b) is True                 # within 1e-5
    try:
        pol.check("fingerprint_t0", a, [1.1, 2.0, 3.0])
        assert False, "large numeric mismatch should raise under enforce"
    except pins_mod.PinViolation:
        pass

    # relax a single pin from config (bare string) + a {level, atol} dict
    relaxed = pins_mod.PinPolicy.from_dict({"provenance": "off",
                                            "fingerprint_t0": {"level": "warn", "atol": 1e-9}})
    assert relaxed.check("provenance", "aaa", "bbb") is True          # off -> skip
    assert relaxed.check("fingerprint_t0", a, [1.2, 2.0, 3.0]) is False  # warn -> no raise
    # unknown pin name is rejected
    try:
        pins_mod.PinPolicy.from_dict({"nope": "off"})
        assert False, "unknown pin should raise"
    except KeyError:
        pass


# ============================================================ runner
_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
