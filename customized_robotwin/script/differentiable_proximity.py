"""
differentiable_proximity.py — Differentiable ground-truth proximity cost f(s, a_chunk).

Clean "Option A": an exactly differentiable collision/clutter objective over an
action chunk, cheap enough to optimize per chunk at inference (guidance) and
designed to port unchanged into a training graph later.

    f(a_{1:H}) = w_p · Σ_{k,s} h²·(1 + γ·h/margin) / 2β
                   with h = relu(r_s + margin − d(c_s(a_k)))
               + λ · Σ_k ‖a_{k+1} − a_k‖²            (smoothness)
               + μ · ‖a − a₀‖²                        (trust region)

    The squared hinge is EXACTLY zero outside the margin shell (a softplus
    tail summed over ~6500 sphere-waypoint terms otherwise dominates the
    objective and biases the arms away from far obstacles).

Components
----------
ArmFK          Analytic FK for one 6-DoF arm.  The kinematic model (joint
               origins + screw axes) is extracted NUMERICALLY from the live
               SAPIEN articulation at poses q=0 and q=δ·e_i, so it is robust
               to URDF conventions and mounting transforms, and is validated
               against SAPIEN before use.  Forward pass is batched torch.
collision spheres
               CuRobo sphere decomposition shipped with the embodiment
               (collision_aloha_{left,right}.yml) — per-link centers + radii.
SDFGrid        Per-episode signed-distance voxel grid.  Obstacle surfaces are
               sampled (with normals); grid distances come from a cKDTree
               nearest-neighbour query, signed by the normal dot product.
               Exclusions (targets, container, table, ground) are applied AT
               BUILD TIME, so the function definitionally cannot penalize
               task-required proximity.  Queries are trilinear-interpolated
               and differentiable.
ProximityCost  Ties the above together; `optimize_chunk` runs a few Adam steps
               on the (H,14) chunk.  Gripper columns (6, 13) are never touched.

Env-var knobs (all optional).  Defaults live ONLY in the constants block
below the imports — that block is the single place to change them (other
modules import the constants; docstrings and launchers do not repeat them):
    SDF_VOXEL        grid resolution in metres
    SDF_PAD          grid padding around obstacles
    SDF_MARGIN       hinge margin beyond sphere surface (activation distance)
    SDF_BETA         softplus temperature
    SDF_STEPS        Adam steps per chunk
    SDF_LR           Adam learning rate
    SDF_W_PROX       proximity weight
    SDF_LAM_SMOOTH   smoothness weight
    SDF_MU_ANCHOR    trust-region weight
    SDF_DEPTH_GAIN   extra hinge steepening at small clearance (0 = off)
    SDF_MAX_DQ       hard per-joint displacement cap
    SDF_DEVICE       torch device
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

_VOXEL      = float(os.environ.get("SDF_VOXEL",      0.01))
_PAD        = float(os.environ.get("SDF_PAD",        0.30))
_MARGIN     = float(os.environ.get("SDF_MARGIN",     0.025))
_BETA       = float(os.environ.get("SDF_BETA",       0.02))
# steps × lr ≈ total per-joint authority (rad).  Finer steps at the same
# budget converge more consistently chunk-to-chunk (less correction noise)
# than few coarse strides that oscillate around the hinge boundary.
_STEPS      = int(os.environ.get("SDF_STEPS",        32))
_LR         = float(os.environ.get("SDF_LR",         0.005))
_W_PROX     = float(os.environ.get("SDF_W_PROX",     1.0))
_LAM_SMOOTH = float(os.environ.get("SDF_LAM_SMOOTH", 2.0))   # velocity term
_LAM_ACC    = float(os.environ.get("SDF_LAM_ACC",    4.0))   # acceleration term
_MU_ANCHOR  = float(os.environ.get("SDF_MU_ANCHOR",  0.05))
_MU_CONT    = float(os.environ.get("SDF_MU_CONT",    4.0))   # chunk-start continuity
# Depth weighting of the hinge: cost = h²·(1 + depth_gain·h/margin)/2β.
# Near the margin boundary (h→0) this is the plain squared hinge — far
# obstacles get the same gentle nudge as before; at contact (h=margin) the
# gradient is (2+3·depth_gain)/2 × the plain hinge, so close threats bulge
# the trajectory out harder.  0 disables (pure squared hinge).
_DEPTH_GAIN = float(os.environ.get("SDF_DEPTH_GAIN", 1.0))
_MAX_DQ     = float(os.environ.get("SDF_MAX_DQ",     0.20))
_DEVICE     = os.environ.get("SDF_DEVICE", "cpu")

_FK_DELTA       = 0.5     # rad perturbation for screw-axis extraction
_FK_ERR_WARN    = 1e-3    # m — warn above this FK validation error
_FK_ERR_FAIL    = 5e-3    # m — refuse to use the model above this
_MAX_VOXELS     = 4_000_000
_REBUILD_DRIFT  = 0.03    # m — rebuild SDF if any obstacle moved this much
_SAMPLE_DENSITY = 0.25e-4  # m² per surface sample ((0.5 cm)² — keep spacing ≈ voxel/2)


def _pose_to_mat(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


def _extract_screw(M: np.ndarray, delta: float):
    """Rotation axis (unit) and a point on the axis from a screw displacement M.

    M is the child-frame displacement T(0)⁻¹T(δ) of a revolute joint: a pure
    rotation by ±δ about some line.  Returns (axis(3,), point(3,)) such that
    M ≈ Rot(axis, signed_delta) about the line through `point`.
    """
    R = M[:3, :3]
    t = M[:3, 3]
    rotvec = Rotation.from_matrix(R).as_rotvec()
    angle  = np.linalg.norm(rotvec)
    if angle < 1e-6:
        raise ValueError("joint perturbation produced no rotation")
    axis = rotvec / angle
    # Align sign so model rotation angle equals +q (not −q)
    if abs(angle - delta) > abs(angle + delta) - 2 * delta:  # angle ≈ −delta case
        pass  # rotvec already signed; angle is positive by construction
    sign = 1.0 if abs(angle - delta) < abs(-angle - delta) else -1.0
    axis = axis * sign
    # Point on axis: (I − R) p = t   (min-norm solution ⊥ axis)
    p, *_ = np.linalg.lstsq(np.eye(3) - R, t, rcond=None)
    return axis, p


def _rot_about_axis(axis: torch.Tensor, point: torch.Tensor, q: torch.Tensor):
    """(B,4,4) rotation by angles q about the line (point, axis). Differentiable in q."""
    B = q.shape[0]
    K = torch.zeros(3, 3, dtype=axis.dtype, device=axis.device)
    K[0, 1], K[0, 2] = -axis[2],  axis[1]
    K[1, 0], K[1, 2] =  axis[2], -axis[0]
    K[2, 0], K[2, 1] = -axis[1],  axis[0]
    s = torch.sin(q).view(B, 1, 1)
    c = torch.cos(q).view(B, 1, 1)
    I = torch.eye(3, dtype=axis.dtype, device=axis.device)
    R = I + s * K + (1.0 - c) * (K @ K)                       # (B,3,3)
    t = point - torch.einsum('bij,j->bi', R, point)           # (B,3)
    M = torch.zeros(B, 4, 4, dtype=axis.dtype, device=axis.device)
    M[:, :3, :3] = R
    M[:, :3, 3]  = t
    M[:, 3, 3]   = 1.0
    return M


class ArmFK:
    """Batched analytic FK for one 6-DoF arm chain of a SAPIEN articulation."""

    def __init__(self, entity, dof_indices, device=_DEVICE):
        self.device      = device
        self.dof_indices = list(dof_indices)
        self._build(entity)

    def _build(self, entity):
        joints      = entity.get_active_joints()
        arm_joints  = [joints[i] for i in self.dof_indices]
        child_links = [j.get_child_link() for j in arm_joints]
        base_link   = arm_joints[0].get_parent_link()
        self.chain_link_names = [lk.get_name() for lk in child_links]

        q_save = np.asarray(entity.get_qpos()).copy()
        q0 = q_save.copy()
        q0[self.dof_indices] = 0.0
        try:
            entity.set_qpos(q0)
            T_base = _pose_to_mat(base_link.get_pose())
            T0     = [_pose_to_mat(lk.get_pose()) for lk in child_links]

            O, axes, points = [], [], []
            prev = T_base
            for i, T in enumerate(T0):
                O.append(np.linalg.inv(prev) @ T)
                prev = T

            for i, ji in enumerate(self.dof_indices):
                qp = q0.copy()
                qp[ji] += _FK_DELTA
                entity.set_qpos(qp)
                Td = _pose_to_mat(child_links[i].get_pose())
                Om = np.linalg.inv(T0[i]) @ Td
                a, p = _extract_screw(Om, _FK_DELTA)
                axes.append(a)
                points.append(p)

            # Gripper links (link7/8): rigid offsets from link6 at the current
            # gripper opening — good to ~2 cm; absorbed by sphere radius padding.
            entity.set_qpos(q0)
            self.ee_offsets = {}
            prefix = self.chain_link_names[-1][:3]          # 'fl_' / 'fr_'
            link_map = {lk.get_name(): lk for lk in entity.get_links()}
            T6_inv = np.linalg.inv(T0[-1])
            for suffix in ("link7", "link8"):
                name = prefix + suffix
                if name in link_map:
                    self.ee_offsets[name] = T6_inv @ _pose_to_mat(link_map[name].get_pose())
        finally:
            entity.set_qpos(q_save)

        dt = torch.float32
        self.T_base = torch.tensor(T_base, dtype=dt, device=self.device)
        self.O      = [torch.tensor(o, dtype=dt, device=self.device) for o in O]
        self.axes   = [torch.tensor(a, dtype=dt, device=self.device) for a in axes]
        self.points = [torch.tensor(p, dtype=dt, device=self.device) for p in points]
        self.ee_offsets_t = {k: torch.tensor(v, dtype=dt, device=self.device)
                             for k, v in self.ee_offsets.items()}

    def fk(self, q: torch.Tensor):
        """q: (B,6) → list of 6 (B,4,4) world transforms for chain links 1..6."""
        B = q.shape[0]
        T = self.T_base.unsqueeze(0).expand(B, 4, 4)
        out = []
        for i in range(6):
            M = _rot_about_axis(self.axes[i], self.points[i], q[:, i])
            T = T @ self.O[i].unsqueeze(0) @ M
            out.append(T)
        return out

    def validate(self, entity, n_trials=5, rng=None) -> float:
        """Max position error [m] of model FK vs SAPIEN over random configs."""
        rng = rng or np.random.default_rng(0)
        q_save = np.asarray(entity.get_qpos()).copy()
        link_map = {lk.get_name(): lk for lk in entity.get_links()}
        max_err = 0.0
        try:
            for _ in range(n_trials):
                q_arm = rng.uniform(-1.5, 1.5, size=6)
                qp = q_save.copy()
                qp[self.dof_indices] = q_arm
                entity.set_qpos(qp)
                Ts = self.fk(torch.tensor(q_arm[None], dtype=torch.float32,
                                          device=self.device))
                for i, name in enumerate(self.chain_link_names):
                    p_true  = np.asarray(link_map[name].get_pose().p, dtype=np.float64)
                    p_model = Ts[i][0, :3, 3].detach().cpu().numpy()
                    max_err = max(max_err, float(np.linalg.norm(p_true - p_model)))
        finally:
            entity.set_qpos(q_save)
        return max_err


def load_collision_spheres(prefix: str):
    """CuRobo sphere decomposition {link_name: [(center(3,), radius), ...]}.

    prefix: 'fl_' loads collision_aloha_left.yml, 'fr_' the right one.
    """
    side = "left" if prefix == "fl_" else "right"
    path = os.path.join(os.environ.get("BENCH_ROOT", ""),
                        "assets/embodiments/aloha-agilex",
                        f"collision_aloha_{side}.yml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"collision sphere yaml not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)["collision_spheres"]
    out = {}
    for link, entries in raw.items():
        out[link] = [(np.asarray(e["center"], dtype=np.float64), float(e["radius"]))
                     for e in entries]
    return out


class SDFGrid:
    """Signed-distance voxel grid with differentiable trilinear queries."""

    def __init__(self, origin, voxel, data: torch.Tensor):
        self.origin = origin            # (3,) torch
        self.voxel  = voxel             # float
        self.data   = data              # (nx,ny,nz) torch
        self.size   = torch.tensor(data.shape, dtype=torch.float32,
                                   device=data.device)

    @staticmethod
    def build(meshes, voxel=_VOXEL, pad=_PAD, device=_DEVICE, signed=True):
        """meshes: list of world-frame trimesh.Trimesh obstacles.

        signed=False stores unsigned distance-to-surface (used as fallback
        when flipped mesh normals corrupt the sign — penetrations then read
        as small positive distances, safe for a few-cm repulsion margin).

        Normal-based signing is only trustworthy for CLOSED geometry, so
        samples from non-watertight meshes never produce negative distances
        (per-sample sign mask) — open shells / inconsistent winding cannot
        flip the field around themselves.
        """
        samples, normals, sign_ok = [], [], []
        for m in meshes:
            n = int(np.clip(m.area / _SAMPLE_DENSITY, 300, 4000))
            pts, fidx = trimesh.sample.sample_surface(m, n)
            samples.append(np.asarray(pts))
            normals.append(np.asarray(m.face_normals[fidx]))
            watertight = bool(getattr(m, "is_watertight", False))
            sign_ok.append(np.full(len(pts), watertight))
        if not samples:
            raise ValueError("no obstacle meshes to build SDF from")
        S    = np.concatenate(samples).astype(np.float64)
        Nrm  = np.concatenate(normals).astype(np.float64)
        Sok  = np.concatenate(sign_ok)

        lo = S.min(axis=0) - pad
        hi = S.max(axis=0) + pad
        extent = hi - lo
        vox = voxel
        while np.prod(np.ceil(extent / vox)) > _MAX_VOXELS:
            vox *= 1.26
        dims = np.maximum(np.ceil(extent / vox).astype(int) + 1, 2)

        xs = lo[0] + np.arange(dims[0]) * vox
        ys = lo[1] + np.arange(dims[1]) * vox
        zs = lo[2] + np.arange(dims[2]) * vox
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        tree = cKDTree(S)
        data = np.empty(len(pts), dtype=np.float32)
        B = 500_000
        for s in range(0, len(pts), B):
            chunk = pts[s:s + B]
            dist, idx = tree.query(chunk, workers=-1)
            if signed:
                # Sign from surface normal: outside if (p − q)·n̂_q > 0.
                # Only samples from watertight meshes may go negative.
                sign = np.sign(np.einsum("ij,ij->i", chunk - S[idx], Nrm[idx]))
                sign[sign == 0] = 1.0
                sign[~Sok[idx]] = 1.0
                data[s:s + B] = (dist * sign).astype(np.float32)
            else:
                data[s:s + B] = dist.astype(np.float32)

        grid = torch.tensor(data.reshape(dims), device=device)
        origin = torch.tensor(lo, dtype=torch.float32, device=device)
        return SDFGrid(origin, float(vox), grid), {
            "n_meshes": len(meshes), "n_samples": len(S),
            "dims": tuple(int(d) for d in dims), "voxel": float(vox),
            "signed": signed, "_samples": S.astype(np.float32),
        }

    def query(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: (...,3) → (...,) signed distance. Differentiable in pts.

        Outside the grid, returns the edge value plus the Euclidean excess
        (conservative, smooth, pushes queries back toward the grid).
        """
        shape = pts.shape[:-1]
        p = pts.reshape(-1, 3)
        u = (p - self.origin) / self.voxel
        u_cl = torch.clamp(u, torch.zeros_like(self.size), self.size - 1.001)
        excess = torch.linalg.norm((u - u_cl) * self.voxel, dim=-1)

        i0 = u_cl.floor().long()
        f  = u_cl - i0.float()
        nx, ny, nz = self.data.shape
        i0x, i0y, i0z = i0[:, 0], i0[:, 1], i0[:, 2]
        i1x = torch.clamp(i0x + 1, max=nx - 1)
        i1y = torch.clamp(i0y + 1, max=ny - 1)
        i1z = torch.clamp(i0z + 1, max=nz - 1)
        d = self.data
        c000 = d[i0x, i0y, i0z]; c100 = d[i1x, i0y, i0z]
        c010 = d[i0x, i1y, i0z]; c110 = d[i1x, i1y, i0z]
        c001 = d[i0x, i0y, i1z]; c101 = d[i1x, i0y, i1z]
        c011 = d[i0x, i1y, i1z]; c111 = d[i1x, i1y, i1z]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        out = c0 * (1 - fz) + c1 * fz + excess
        return out.reshape(shape)


def _actor_key(actor, fallback_idx):
    """Stable per-entity key — actor NAMES are duplicated in cluttered scenes."""
    for attr in ("per_scene_id", "global_id", "gid"):
        v = getattr(actor, attr, None)
        if v is not None:
            return f"{actor.get_name()}#{v}"
    return f"{actor.get_name()}@{fallback_idx}"


def _gather_obstacle_meshes(task_env, exclude_names):
    """World-frame trimesh per non-excluded obstacle; AABB-box fallback.

    Returns (meshes, pose_snapshot {name: (3,) position}) — the snapshot is
    used to detect obstacle movement and trigger an SDF rebuild.
    """
    mesh_cache = getattr(task_env, "_proximity_mesh_cache", {})
    robot = task_env.robot
    robot_entities = {robot.left_entity, robot.right_entity}

    meshes, poses = [], {}
    for i, actor in enumerate(task_env.scene.get_all_actors()):
        name = actor.get_name()
        if name in exclude_names:
            continue
        aabb = None
        for comp in actor.get_components():
            if hasattr(comp, "get_global_aabb_fast"):
                try:
                    aabb = comp.get_global_aabb_fast()
                except RuntimeError:
                    pass
                break
        if aabb is None:
            continue
        T = _pose_to_mat(actor.get_pose())
        if name in mesh_cache:
            m = mesh_cache[name].copy()
            m.apply_transform(T)
            # Collision GLBs frequently ship with inverted winding; flipped
            # normals invert the sign of the distance field around that
            # object (free space reads "inside"), so repair them.
            try:
                m.fix_normals()
            except Exception:
                pass
        else:
            lo = np.asarray(aabb[0], dtype=np.float64)
            hi = np.asarray(aabb[1], dtype=np.float64)
            if np.any(hi - lo < 1e-6):
                continue
            box_T = np.eye(4); box_T[:3, 3] = (lo + hi) / 2
            m = trimesh.creation.box(extents=hi - lo, transform=box_T)
        meshes.append(m)
        # Scenes contain duplicate actor NAMES (three 108_block, two
        # 090_trophy, ...) — key the pose snapshot by entity id, never by
        # name alone, or move-detection compares different objects and
        # rebuilds the grid every chunk.
        poses[_actor_key(actor, i)] = np.asarray(actor.get_pose().p,
                                                 dtype=np.float64)

    # Articulated furniture (bookcase, fridge, microwave, cabinet): AABB boxes
    # per collision link.  Two exclusion rules:
    #   1. name in exclude_names (explicit).
    #   2. the articulation's volume CONTAINS a task target — then it is the
    #      container the task must reach INTO (bottle inside the fridge, book
    #      inside the bookcase).  AABB boxes would mark its open cavity as
    #      solid and guidance would fight the task (kitchen: guided rollouts
    #      failed exactly this way), so skip it entirely.
    target_names = frozenset(getattr(task_env, 'target_object_names', set()))
    target_pos = [np.asarray(a.get_pose().p, dtype=np.float64)
                  for a in task_env.scene.get_all_actors()
                  if a.get_name() in target_names]

    for art in task_env.scene.get_all_articulations():
        if art in robot_entities:
            continue
        try:
            art_name = art.get_name() or ""
        except Exception:
            art_name = ""
        if art_name in exclude_names:
            continue

        boxes = []
        for link in art.get_links():
            if not link.collision_shapes:
                continue
            for comp in link.entity.get_components():
                if hasattr(comp, "get_global_aabb_fast"):
                    try:
                        aabb = comp.get_global_aabb_fast()
                    except RuntimeError:
                        break
                    lo = np.asarray(aabb[0], dtype=np.float64)
                    hi = np.asarray(aabb[1], dtype=np.float64)
                    if np.all(hi - lo > 1e-6):
                        boxes.append((lo, hi))
                    break
        if not boxes:
            continue

        art_lo = np.min([b[0] for b in boxes], axis=0) - 0.02
        art_hi = np.max([b[1] for b in boxes], axis=0) + 0.02
        if any(np.all(p >= art_lo) and np.all(p <= art_hi) for p in target_pos):
            print(f"[sdf] articulation '{art_name or 'unnamed'}' excluded — "
                  f"contains a task target (structural container)")
            continue

        for lo, hi in boxes:
            box_T = np.eye(4); box_T[:3, 3] = (lo + hi) / 2
            meshes.append(trimesh.creation.box(extents=hi - lo,
                                               transform=box_T))
    return meshes, poses


class ProximityCost:
    """Differentiable proximity objective + chunk optimizer for aloha-agilex."""

    def __init__(self, task_env, left_dofs, right_dofs, exclude_names,
                 device=_DEVICE, margin=_MARGIN, beta=_BETA,
                 depth_gain=_DEPTH_GAIN):
        self.device = device
        self.margin = margin
        self.beta   = beta
        self.depth_gain = depth_gain
        robot = task_env.robot
        self._left_entity  = robot.left_entity
        self._right_entity = robot.right_entity

        self.fk_left  = ArmFK(robot.left_entity,  left_dofs,  device)
        self.fk_right = ArmFK(robot.right_entity, right_dofs, device)
        self.fk_err_left  = self.fk_left.validate(robot.left_entity)
        self.fk_err_right = self.fk_right.validate(robot.right_entity)
        worst = max(self.fk_err_left, self.fk_err_right)
        if worst > _FK_ERR_FAIL:
            raise RuntimeError(f"FK validation failed: max err {worst*1000:.2f} mm")
        if worst > _FK_ERR_WARN:
            print(f"[sdf] WARNING: FK validation error {worst*1000:.2f} mm")

        self._sphere_setup()
        self._exclude_names = frozenset(exclude_names)
        self._build_sdf(task_env)

    # ── robot sphere model ────────────────────────────────────────────────
    def _sphere_setup(self):
        self._arm_specs = []      # (fk, [(slot, center(3,), radius)]) per arm
        self._sphere_labels = []  # parallel to _sphere_centers concat order
        for fk, prefix in ((self.fk_left, "fl_"), (self.fk_right, "fr_")):
            spheres = load_collision_spheres(prefix)
            entries = []
            for link, lst in spheres.items():
                if link == f"{prefix}base_link":
                    continue                      # static — zero action gradient
                if link in fk.ee_offsets_t:
                    slot = ("ee", link)           # rigid offset from link6
                    pad  = 0.01                   # absorbs gripper opening
                elif link in fk.chain_link_names:
                    slot = ("chain", fk.chain_link_names.index(link))
                    pad  = 0.0
                else:
                    continue
                for c, r in lst:
                    entries.append((slot, c, r + pad))
                    self._sphere_labels.append(link)
            self._arm_specs.append((fk, entries))
        # Radii in _sphere_centers concat order (for solid-geometry exports)
        self._radii_np = np.array(
            [r for _, entries in self._arm_specs for _, _, r in entries],
            dtype=np.float64)

    def _sphere_centers(self, qL: torch.Tensor, qR: torch.Tensor):
        """(B,6),(B,6) → centers (B,S,3), radii (S,)."""
        all_centers, all_radii = [], []
        for (fk, entries), q in zip(self._arm_specs, (qL, qR)):
            Ts = fk.fk(q)                                   # 6 × (B,4,4)
            T_by_slot = {("chain", i): Ts[i] for i in range(6)}
            for name, off in fk.ee_offsets_t.items():
                T_by_slot[("ee", name)] = Ts[5] @ off.unsqueeze(0)
            for slot, c, r in entries:
                T = T_by_slot[slot]
                c_t = torch.tensor(c, dtype=torch.float32, device=self.device)
                p = torch.einsum("bij,j->bi", T[:, :3, :3], c_t) + T[:, :3, 3]
                all_centers.append(p.unsqueeze(1))          # (B,1,3)
                all_radii.append(r)
        centers = torch.cat(all_centers, dim=1)             # (B,S,3)
        radii = torch.tensor(all_radii, dtype=torch.float32, device=self.device)
        return centers, radii

    # ── scene SDF ─────────────────────────────────────────────────────────
    def _current_clearance(self):
        """Worst clearance of the robot's CURRENT pose — must be ≥ 0."""
        qL = np.asarray(self._left_entity.get_qpos())[self.fk_left.dof_indices]
        qR = np.asarray(self._right_entity.get_qpos())[self.fk_right.dof_indices]
        with torch.no_grad():
            centers, radii = self._sphere_centers(
                torch.tensor(qL[None], dtype=torch.float32, device=self.device),
                torch.tensor(qR[None], dtype=torch.float32, device=self.device))
            d = self.sdf.query(centers)
            return float((d - radii).min())

    def _build_sdf(self, task_env):
        t0 = time.time()
        meshes, poses = _gather_obstacle_meshes(task_env, self._exclude_names)
        signed = not getattr(self, "_force_unsigned", False)
        self.sdf, info = SDFGrid.build(meshes, device=self.device, signed=signed)

        # Sanity: the robot's actual pose is in free space by definition.  If
        # the signed field claims it is deep inside an obstacle, some mesh has
        # irreparably flipped/inconsistent normals (fix_normals can't repair
        # non-watertight geometry) — rebuild UNSIGNED rather than optimize an
        # inverted field that PULLS the arm toward that obstacle.
        if signed:
            clr = self._current_clearance()
            if clr < -0.02:
                print(f"\033[33m[sdf] WARNING: signed field puts the current "
                      f"robot pose {clr*100:.1f} cm inside an obstacle — "
                      f"flipped mesh normals; rebuilding unsigned\033[0m")
                self._force_unsigned = True
                self.sdf, info = SDFGrid.build(meshes, device=self.device,
                                               signed=False)

        self._obstacle_poses = poses
        self._dbg_samples = info.pop("_samples", None)
        self._dbg_meshes  = meshes
        info["build_s"] = time.time() - t0
        self.sdf_info = info

        dbg = os.environ.get("SDF_DEBUG_VIZ", "")
        if dbg and dbg.lower() not in ("0", "false", "no"):
            out_dir = dbg if dbg.lower() not in ("1", "true", "yes") else "sdf_debug"
            try:
                self.export_debug_viz(out_dir)
            except Exception as e:
                print(f"[sdf] debug viz failed: {type(e).__name__}: {e}")

    def force_unsigned(self, task_env):
        """Permanently switch this episode's field to unsigned distances."""
        self._force_unsigned = True
        self._build_sdf(task_env)

    # ── debug visualization (SDF_DEBUG_VIZ=1 or =<dir>) ──────────────────
    def export_debug_viz(self, out_dir="sdf_debug"):
        """Dump the grid as PNG z-slices + PLY point clouds.

        sdf_slices_*.png    — horizontal slices: blue=free, red=inside,
                              black contour = obstacle surface (d=0),
                              green dots = robot spheres at the CURRENT pose.
        sdf_negative_*.ply  — every voxel with d<0 (the claimed 'inside'
                              volume); open in Meshlab/CloudCompare.  Phantom
                              geometry shows up as red blobs in free space.
        sdf_samples_*.ply   — the obstacle surface samples the grid is
                              built from.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        tag    = time.strftime("%H%M%S")
        data   = self.sdf.data.cpu().numpy()
        origin = self.sdf.origin.cpu().numpy()
        vox    = self.sdf.voxel
        nx, ny, nz = data.shape

        # Robot spheres at the current pose (green overlay)
        qL = np.asarray(self._left_entity.get_qpos())[self.fk_left.dof_indices]
        qR = np.asarray(self._right_entity.get_qpos())[self.fk_right.dof_indices]
        with torch.no_grad():
            centers, _ = self._sphere_centers(
                torch.tensor(qL[None], dtype=torch.float32, device=self.device),
                torch.tensor(qR[None], dtype=torch.float32, device=self.device))
        sph = centers[0].cpu().numpy()

        z_lo, z_hi = origin[2], origin[2] + (nz - 1) * vox
        slice_zs = np.linspace(z_lo + 0.02, z_hi - 0.02, 6)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        ext = (origin[0], origin[0] + (nx - 1) * vox,
               origin[1], origin[1] + (ny - 1) * vox)
        for ax, z in zip(axes.ravel(), slice_zs):
            k = int(round((z - origin[2]) / vox))
            sl = data[:, :, k]
            im = ax.imshow(sl.T, origin="lower", extent=ext, cmap="RdBu",
                           vmin=-0.15, vmax=0.15)
            ax.contour(np.linspace(ext[0], ext[1], nx),
                       np.linspace(ext[2], ext[3], ny),
                       sl.T, levels=[0.0], colors="k", linewidths=1.0)
            near = sph[np.abs(sph[:, 2] - z) < 0.03]
            if len(near):
                ax.scatter(near[:, 0], near[:, 1], s=12, c="lime",
                           edgecolors="k", linewidths=0.3)
            ax.set_title(f"z = {z:.2f} m")
            ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                     label="signed distance [m]")
        fig.suptitle(f"SDF slices (signed={self.sdf_info.get('signed', '?')}, "
                     f"voxel={vox*100:.1f} cm) — red=inside, green=robot spheres")
        png = os.path.join(out_dir, f"sdf_slices_{tag}.png")
        fig.savefig(png, dpi=110)
        plt.close(fig)

        # Only the slice PNG is written at build time — event files
        # (sdf_event_*) are the sole PLY output, exported on demand.
        neg_count = int((data < 0.0).sum())
        print(f"[sdf] debug viz written: {png}  (negative voxels: {neg_count})")

    def _solid_scene(self, sphere_sets, subdiv=1, include_obstacles=True):
        """One colored mesh: obstacle meshes (gray) + real-radius spheres.

        sphere_sets: list of (centers (S,3), radii (S,), color (4,) or (S,4)).
        subdiv: icosphere subdivisions (0 = 20 faces/sphere, for big sweeps).
        Returns a single trimesh.Trimesh with face colors — opens cleanly in
        Meshlab / CloudCompare / any glTF-style viewer.
        """
        parts = []
        if include_obstacles:
            for m in getattr(self, "_dbg_meshes", None) or []:
                mm = m.copy()
                mm.visual.face_colors = [170, 170, 170, 255]
                parts.append(mm)
        for centers, radii, colors in sphere_sets:
            colors = np.asarray(colors, dtype=np.uint8)
            single = colors.ndim == 1
            for i, (c, r) in enumerate(zip(centers, radii)):
                s = trimesh.creation.icosphere(subdivisions=subdiv,
                                               radius=float(r))
                s.apply_translation(np.asarray(c, dtype=np.float64))
                s.visual.face_colors = colors if single else colors[i]
                parts.append(s)
        return trimesh.util.concatenate(parts)

    @staticmethod
    def _clearance_colors(clear, margin=None):
        """Two colors only: green = safe sphere, red = planned penetration."""
        col = np.tile(np.array([[0, 180, 0, 255]], np.uint8), (len(clear), 1))
        col[clear < 0.0] = [255, 0, 0, 255]
        return col

    def _export_event(self, out_dir, tag, centers0, clear0, centers1=None,
                      worst_k=None):
        """Exactly TWO solid-geometry files per event.

        sdf_event_{tag}.ply       gray obstacles + ONLY the contacting link's
                                  spheres swept across all waypoints:
                                  red  = planned (non-optimized) positions
                                  blue = after optimization (when present)
        sdf_event_{tag}_k{K}.ply  one frame at the worst waypoint K: the full
                                  robot, green spheres with red where the
                                  plan penetrates (+ blue after optimization).
        """
        os.makedirs(out_dir, exist_ok=True)
        cl_np = clear0.cpu().numpy()                       # (B,S)
        flat  = int(np.argmin(cl_np))
        k0, s0 = divmod(flat, cl_np.shape[1])
        if worst_k is None:
            worst_k = k0
        link = self._sphere_labels[s0]
        idx  = np.array([i for i, l in enumerate(self._sphere_labels)
                         if l == link])

        c0 = centers0.cpu().numpy()                        # (B,S,3)
        c1 = centers1.cpu().numpy() if centers1 is not None else None
        B  = c0.shape[0]
        r_link = self._radii_np[idx]

        # 1) Contact-link sweep: that link's spheres at every waypoint.
        sweep_sets = [(c0[:, idx].reshape(-1, 3), np.tile(r_link, B),
                       np.array([255, 0, 0, 255]))]        # planned = red
        if c1 is not None:
            sweep_sets.append((c1[:, idx].reshape(-1, 3), np.tile(r_link, B),
                               np.array([0, 90, 255, 255])))  # optimized = blue
        path = os.path.join(out_dir, f"sdf_event_{tag}.ply")
        self._solid_scene(sweep_sets, subdiv=0).export(path)

        # 2) Single worst-waypoint frame, full robot.
        frame_sets = [(c0[worst_k], self._radii_np,
                       self._clearance_colors(cl_np[worst_k]))]
        if c1 is not None:
            frame_sets.append((c1[worst_k], self._radii_np,
                               np.array([0, 90, 255, 120])))
        fpath = os.path.join(out_dir, f"sdf_event_{tag}_k{int(worst_k)}.ply")
        self._solid_scene(frame_sets, subdiv=1).export(fpath)

        print(f"[sdf] event viz: {os.path.basename(path)} (link={link}, "
              f"red=planned, blue=optimized) + {os.path.basename(fpath)}")

    def maybe_rebuild(self, task_env):
        """Rebuild the SDF if any tracked obstacle moved > _REBUILD_DRIFT."""
        for i, actor in enumerate(task_env.scene.get_all_actors()):
            key = _actor_key(actor, i)
            ref = self._obstacle_poses.get(key)
            if ref is None:
                continue
            if np.linalg.norm(np.asarray(actor.get_pose().p) - ref) > _REBUILD_DRIFT:
                print(f"[sdf] obstacle '{key}' moved — rebuilding grid")
                self._build_sdf(task_env)
                return True
        return False

    # ── objective + optimizer ────────────────────────────────────────────
    def proximity_cost(self, qL, qR):
        """Depth-weighted squared-hinge penetration cost over spheres × waypoints.

        cost = Σ h²·(1 + γ·h/margin) / (2β)  with  h = relu((radius + margin) − d)
        and γ = depth_gain.

        EXACTLY zero whenever every sphere clears the margin (unlike softplus,
        whose tail summed over ~6500 sphere-waypoint pairs dominated the
        objective and pushed the arms away from obstacles that were never a
        threat).  Gradient is continuous at the boundary; near it (h→0) it
        matches the plain squared hinge (far obstacles: small nudge), and the
        cubic term steepens it as clearance shrinks, so imminent collisions
        bulge the trajectory out disproportionately harder.
        """
        centers, radii = self._sphere_centers(qL, qR)        # (B,S,3),(S,)
        d = self.sdf.query(centers)                          # (B,S)
        pen = (radii + self.margin) - d
        h = F.relu(pen)
        depth_w = 1.0 + self.depth_gain * h / max(self.margin, 1e-9)
        cost = (h ** 2 * depth_w).sum() / (2.0 * self.beta)
        min_clear = (d - radii).min()
        return cost, min_clear

    def evaluate_chunk(self, chunk: np.ndarray):
        """Forward-only FK-lookahead of a chunk — predict, never modify.

        Used on UNGUIDED rollouts to test the predictive validity of the
        proximity function: does a negative predicted clearance actually
        materialize as a real collision when the chunk executes?
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        aL = torch.tensor(chunk[:, 0:6],  device=self.device)
        aR = torch.tensor(chunk[:, 7:13], device=self.device)
        with torch.no_grad():
            prox0, clear0 = self.proximity_cost(aL, aR)
            centers0, radii0 = self._sphere_centers(aL, aR)
            clr_t = self.sdf.query(centers0) - radii0
            flat  = int(torch.argmin(clr_t))
            k0, s0 = divmod(flat, clr_t.shape[1])
        return {"prox0": float(prox0), "clear0": float(clear0),
                "worst": {"link": self._sphere_labels[s0],
                          "waypoint": int(k0),
                          "pos": [round(float(x), 3)
                                  for x in centers0[k0, s0].tolist()]},
                "centers0": centers0, "clr_t": clr_t}

    def export_actual(self, out_dir, tag):
        """Solid snapshot of the robot's ACTUAL current pose vs obstacles.

        Written when a previously PREDICTED collision's execution step
        arrives — pair it with the matching sdf_event_pred_* files to see
        whether the FK prediction came true (red spheres = real penetration
        of the model right now).
        """
        os.makedirs(out_dir, exist_ok=True)
        qL = np.asarray(self._left_entity.get_qpos())[self.fk_left.dof_indices]
        qR = np.asarray(self._right_entity.get_qpos())[self.fk_right.dof_indices]
        with torch.no_grad():
            centers, radii = self._sphere_centers(
                torch.tensor(qL[None], dtype=torch.float32, device=self.device),
                torch.tensor(qR[None], dtype=torch.float32, device=self.device))
            clr = (self.sdf.query(centers) - radii)[0].cpu().numpy()
        solid = self._solid_scene([(centers[0].cpu().numpy(), self._radii_np,
                                    self._clearance_colors(clr))], subdiv=1)
        path = os.path.join(out_dir, f"sdf_actual_{tag}.ply")
        solid.export(path)
        print(f"[sdf] actual-pose snapshot: {path}  "
              f"(real min clearance {clr.min()*100:.1f} cm)")

    def optimize_chunk(self, chunk: np.ndarray,
                       n_steps=_STEPS, lr=_LR,
                       w_prox=_W_PROX, lam=_LAM_SMOOTH, lam_acc=_LAM_ACC,
                       mu=_MU_ANCHOR, mu_cont=_MU_CONT,
                       max_dq=_MAX_DQ, event_dir=None, event_tag=""):
        """Optimize the arm columns of an (H,14) chunk. Returns (chunk', info).

        ALL regularizers act on the CORRECTION FIELD d = a − a₀, never on the
        absolute trajectory: a zero correction has zero regularizer cost, so
        the optimizer cannot "improve" the loss by flattening or slowing the
        policy's own motion.  (Penalizing ‖a[k+1]−a[k]‖² of the absolute
        trajectory at high weight let the optimizer velocity-damp the plan
        itself on every active chunk — arms stalled mid-task and success
        collapsed 9/10 → 5/10 while looking "smooth".)

        Terms: prox(a) + λ·vel(d) + λ_acc·acc(d) + μ·‖d‖² + μ_cont-ramp·d²
        (corrections ramp in from zero over the first ~10 waypoints so a chunk
        never starts with a step jump).

        event_dir: when set and the incoming chunk has a planned penetration
        (clearance < 0), a colored point-cloud snapshot of the event is
        exported there (see _export_event).
        """
        t0 = time.time()
        chunk = np.asarray(chunk, dtype=np.float32).copy()
        H  = len(chunk)
        aL = torch.tensor(chunk[:, 0:6],  device=self.device, requires_grad=True)
        aR = torch.tensor(chunk[:, 7:13], device=self.device, requires_grad=True)
        a0L, a0R = aL.detach().clone(), aR.detach().clone()

        # Chunk-start ramp weights: corrections must START at ~zero (decay ~6 wp)
        w_cont = (mu_cont * torch.exp(-torch.arange(
            H, dtype=torch.float32, device=self.device) / 4.0)).view(H, 1)

        with torch.no_grad():
            prox0, clear0 = self.proximity_cost(a0L, a0R)
            # Identify the worst (sphere, waypoint) pair — names the obstacle
            # region driving the cost, which makes phantom geometry (flipped
            # normals, bad meshes) immediately visible in the logs.
            centers0, radii0 = self._sphere_centers(a0L, a0R)
            clr_t = self.sdf.query(centers0) - radii0          # (B,S)
            flat  = int(torch.argmin(clr_t))
            k0, s0 = divmod(flat, clr_t.shape[1])
            worst = {"link": self._sphere_labels[s0], "waypoint": int(k0),
                     "pos": [round(float(x), 3)
                             for x in centers0[k0, s0].tolist()]}

        if float(prox0) < 1e-6:        # nothing near anything — skip the solve
            return chunk, {"prox0": float(prox0), "prox1": float(prox0),
                           "clear0": float(clear0), "clear1": float(clear0),
                           "max_dq": 0.0, "ms": (time.time() - t0) * 1e3,
                           "skipped": True, "worst": worst}

        opt = torch.optim.Adam([aL, aR], lr=lr)
        for _ in range(n_steps):
            prox, _ = self.proximity_cost(aL, aR)
            dL = aL - a0L
            dR = aR - a0R
            vel = ((dL[1:] - dL[:-1]) ** 2).sum() + ((dR[1:] - dR[:-1]) ** 2).sum()
            acc = ((dL[2:] - 2 * dL[1:-1] + dL[:-2]) ** 2).sum() + \
                  ((dR[2:] - 2 * dR[1:-1] + dR[:-2]) ** 2).sum()
            anchor = (dL ** 2).sum() + (dR ** 2).sum()
            cont = (w_cont * dL ** 2).sum() + (w_cont * dR ** 2).sum()
            loss = (w_prox * prox + lam * vel + lam_acc * acc
                    + mu * anchor + cont)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            # Bound the correction by RESCALING it uniformly per arm — an
            # elementwise clamp clips some waypoints and not their neighbours,
            # which injects exactly the kinks the smoothness terms remove.
            for a, a0 in ((aL, a0L), (aR, a0R)):
                d = a - a0
                m = float(d.abs().max())
                if m > max_dq:
                    a.data = a0 + d * (max_dq / m)
            prox1, clear1 = self.proximity_cost(aL, aR)
            max_disp = float(torch.max(torch.abs(torch.cat(
                [aL - a0L, aR - a0R], dim=1))))

        event_written = False
        if event_dir is not None and float(clear0) < 0.0:
            try:
                with torch.no_grad():
                    centers1, _ = self._sphere_centers(aL.detach(), aR.detach())
                self._export_event(event_dir, event_tag, centers0, clr_t,
                                   centers1, worst_k=int(k0))
                event_written = True
            except Exception as e:
                print(f"[sdf] event viz failed: {type(e).__name__}: {e}")

        chunk[:, 0:6]  = aL.detach().cpu().numpy()
        chunk[:, 7:13] = aR.detach().cpu().numpy()
        return chunk, {"prox0": float(prox0), "prox1": float(prox1),
                       "clear0": float(clear0), "clear1": float(clear1),
                       "max_dq": max_disp, "ms": (time.time() - t0) * 1e3,
                       "skipped": False, "worst": worst,
                       "event_written": event_written}


# ── dataset export helpers (collection-time; the offline relabel pass reads
#    these — see data_generation_documentation.md) ─────────────────────────

_COARSE_DENSITY = 16e-4   # m² per sample (4 cm spacing) for furniture/articulations
_SKIP_EXPORT    = {"ground", "wall"}   # unbounded planes — never sampled


def get_arm_dof_indices(entity, robot, arm="left"):
    """DOF indices (within entity.get_active_joints()) belonging to one arm.

    Same logic as the eval client: name lookup via robot.{arm}_arm_joints,
    positional heuristic as fallback.
    """
    target_joints = getattr(robot, f"{arm}_arm_joints", None)
    if target_joints is not None:
        try:
            arm_names = {j.get_name() for j in target_joints}
            active = entity.get_active_joints()
            indices = [i for i, j in enumerate(active) if j.get_name() in arm_names]
            if indices:
                return indices
        except Exception:
            pass
    n_dofs = len(entity.get_qpos())
    if arm == "left" or robot.right_entity is not robot.left_entity:
        return list(range(min(6, n_dofs)))
    offset = n_dofs // 2
    return list(range(offset, min(offset + 6, n_dofs)))


def serialize_fk_and_spheres(task_env, path):
    """Serialize both arms' FK models + sphere decompositions into one npz.

    Everything needed to reconstruct all sphere centers offline from qpos
    alone: per arm T_base, joint origins O, screw axes/points, dof indices,
    chain link names, gripper (ee) rigid offsets, validated FK error; per
    sphere its link, slot, link-frame center and padded radius — the exact
    quantities ProximityCost._sphere_setup uses online.
    """
    robot = task_env.robot
    data = {}
    for side, entity in (("left", robot.left_entity), ("right", robot.right_entity)):
        dofs = get_arm_dof_indices(entity, robot, side)
        fk = ArmFK(entity, dofs)
        err = fk.validate(entity)
        if err > _FK_ERR_FAIL:
            raise RuntimeError(f"FK validation failed for {side} arm: {err*1000:.2f} mm")
        prefix = "fl_" if side == "left" else "fr_"
        ee_names = sorted(fk.ee_offsets)
        data[f"{side}_T_base"] = fk.T_base.cpu().numpy().astype(np.float64)
        data[f"{side}_O"]      = np.stack([o.cpu().numpy() for o in fk.O]).astype(np.float64)
        data[f"{side}_axes"]   = np.stack([a.cpu().numpy() for a in fk.axes]).astype(np.float64)
        data[f"{side}_points"] = np.stack([p.cpu().numpy() for p in fk.points]).astype(np.float64)
        data[f"{side}_dofs"]   = np.asarray(dofs, dtype=np.int64)
        data[f"{side}_chain_links"] = np.asarray(fk.chain_link_names)
        data[f"{side}_fk_err"] = np.float64(err)
        data[f"{side}_ee_names"]   = np.asarray(ee_names)
        data[f"{side}_ee_offsets"] = (np.stack([fk.ee_offsets[n] for n in ee_names])
                                      if ee_names else np.zeros((0, 4, 4)))
        link_n, slot_t, slot_i, centers, radii = [], [], [], [], []
        for link, lst in load_collision_spheres(prefix).items():
            if link == f"{prefix}base_link":
                continue
            if link in fk.ee_offsets:
                st, si, pad = "ee", ee_names.index(link), 0.01
            elif link in fk.chain_link_names:
                st, si, pad = "chain", fk.chain_link_names.index(link), 0.0
            else:
                continue
            for c, r in lst:
                link_n.append(link); slot_t.append(st); slot_i.append(si)
                centers.append(c);   radii.append(r + pad)
        data[f"{side}_sph_link"]      = np.asarray(link_n)
        data[f"{side}_sph_slot_type"] = np.asarray(slot_t)
        data[f"{side}_sph_slot_idx"]  = np.asarray(slot_i, dtype=np.int64)
        data[f"{side}_sph_center"]    = np.asarray(centers, dtype=np.float64)
        data[f"{side}_sph_radius"]    = np.asarray(radii, dtype=np.float64)
    np.savez_compressed(path, **data)
    return path


def _write_ply(path, pts, normals, max_points=50_000):
    """Minimal ascii PLY with vertex normals (downsampled for viewing)."""
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, normals = pts[idx], normals[idx]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float nx\nproperty float ny\nproperty float nz\n"
                "end_header\n")
        for p, n in zip(pts, normals):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} "
                    f"{n[0]:.3f} {n[1]:.3f} {n[2]:.3f}\n")


def export_scene(task_env, out_dir):
    """Per-seed scene record for the collision dataset.

    Writes into out_dir:
        scene.npz       samples (N,3) f32, normals (N,3) f32, obj_id (N,) i32
        pointcloud.ply  same samples (downsampled) for visualization
        objects.json    per object: id, name, tags, pose, n_samples
        scene_hash.txt  identity hash over (name, id, rounded pose)

    Everything is exported and TAGGED (target / container / furniture /
    clutter / articulation) — exclusion decisions belong to the relabel
    pass, not to collection.  Post-settle poses (call after setup_demo).
    """
    import hashlib

    os.makedirs(out_dir, exist_ok=True)
    mesh_cache   = getattr(task_env, "_proximity_mesh_cache", {})
    target_names = set(getattr(task_env, "target_object_names", set()))
    furn_names   = set(getattr(task_env, "furniture_names", set()))
    container_names = set()
    for attr in ("box", "des_obj"):
        a = getattr(task_env, attr, None)
        try:
            if a is not None:
                container_names.add(a.get_name())
        except Exception:
            pass

    samples, normals, obj_ids, objects = [], [], [], []

    def _sample(mesh, oid, name, tags, pose_p, pose_q):
        density = _COARSE_DENSITY if ("furniture" in tags or "articulation" in tags) \
                  else _SAMPLE_DENSITY
        n = max(16, int(mesh.area / density))
        try:
            pts, fidx = trimesh.sample.sample_surface(mesh, n)
        except Exception:
            return 0
        samples.append(np.asarray(pts, dtype=np.float32))
        normals.append(np.asarray(mesh.face_normals[fidx], dtype=np.float32))
        obj_ids.append(np.full(len(pts), oid, dtype=np.int32))
        objects.append({"per_scene_id": int(oid), "name": name, "tags": sorted(tags),
                        "pose_p": [round(float(x), 4) for x in pose_p],
                        "pose_q": [round(float(x), 4) for x in pose_q],
                        "n_samples": int(len(pts))})
        return len(pts)

    robot = task_env.robot
    robot_entities = {robot.left_entity, robot.right_entity}

    for actor in task_env.scene.get_all_actors():
        name = actor.get_name()
        tags = set()
        if name in target_names:    tags.add("target")
        if name in container_names: tags.add("container")
        if name in furn_names:      tags.add("furniture")
        if not tags:                tags.add("clutter")
        pose = actor.get_pose()
        if name in _SKIP_EXPORT:
            objects.append({"per_scene_id": int(actor.per_scene_id), "name": name,
                            "tags": sorted(tags | {"skipped"}),
                            "pose_p": [round(float(x), 4) for x in pose.p],
                            "pose_q": [round(float(x), 4) for x in pose.q],
                            "n_samples": 0})
            continue
        aabb = None
        for comp in actor.get_components():
            if hasattr(comp, "get_global_aabb_fast"):
                try:
                    aabb = comp.get_global_aabb_fast()
                except RuntimeError:
                    pass
                break
        if aabb is None:
            continue
        T = _pose_to_mat(pose)
        if name in mesh_cache:
            m = mesh_cache[name].copy()
            m.apply_transform(T)
            try:
                m.fix_normals()
            except Exception:
                pass
        else:
            lo = np.asarray(aabb[0], dtype=np.float64)
            hi = np.asarray(aabb[1], dtype=np.float64)
            if np.any(hi - lo < 1e-6):
                continue
            box_T = np.eye(4); box_T[:3, 3] = (lo + hi) / 2
            m = trimesh.creation.box(extents=hi - lo, transform=box_T)
        _sample(m, actor.per_scene_id, name, tags, pose.p, pose.q)

    # Articulations: AABB box per collision link, synthetic negative ids.
    target_pos = [np.asarray(a.get_pose().p, dtype=np.float64)
                  for a in task_env.scene.get_all_actors()
                  if a.get_name() in target_names]
    for j, art in enumerate(task_env.scene.get_all_articulations()):
        if art in robot_entities:
            continue
        try:
            art_name = art.get_name() or f"articulation_{j}"
        except Exception:
            art_name = f"articulation_{j}"
        boxes = []
        for link in art.get_links():
            if not link.collision_shapes:
                continue
            for comp in link.entity.get_components():
                if hasattr(comp, "get_global_aabb_fast"):
                    try:
                        aabb = comp.get_global_aabb_fast()
                    except RuntimeError:
                        break
                    lo = np.asarray(aabb[0], dtype=np.float64)
                    hi = np.asarray(aabb[1], dtype=np.float64)
                    if np.all(hi - lo > 1e-6):
                        boxes.append((lo, hi))
                    break
        if not boxes:
            continue
        tags = {"articulation"}
        art_lo = np.min([b[0] for b in boxes], axis=0) - 0.02
        art_hi = np.max([b[1] for b in boxes], axis=0) + 0.02
        if any(np.all(p >= art_lo) and np.all(p <= art_hi) for p in target_pos):
            tags.add("container")
        oid = -(1000 + j)
        box_meshes = []
        for lo, hi in boxes:
            box_T = np.eye(4); box_T[:3, 3] = (lo + hi) / 2
            box_meshes.append(trimesh.creation.box(extents=hi - lo, transform=box_T))
        mesh = trimesh.util.concatenate(box_meshes)
        center = (art_lo + art_hi) / 2
        _sample(mesh, oid, art_name, tags, center, [1, 0, 0, 0])

    if not samples:
        raise RuntimeError("export_scene: no geometry exported")
    S  = np.concatenate(samples)
    Nm = np.concatenate(normals)
    I  = np.concatenate(obj_ids)
    np.savez_compressed(os.path.join(out_dir, "scene.npz"),
                        samples=S, normals=Nm, obj_id=I)
    _write_ply(os.path.join(out_dir, "pointcloud.ply"), S, Nm)
    import json as _json
    with open(os.path.join(out_dir, "objects.json"), "w") as f:
        _json.dump(objects, f, indent=1)
    ident = sorted((o["name"], o["per_scene_id"], tuple(o["pose_p"]), tuple(o["pose_q"]))
                   for o in objects)
    h = hashlib.sha1(repr(ident).encode()).hexdigest()
    with open(os.path.join(out_dir, "scene_hash.txt"), "w") as f:
        f.write(h + "\n")
    return h
