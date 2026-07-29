"""IK-branch continuity and warm-start propagation."""

from collections import deque

import numpy as np


def _wrap_linf(a, b):
    """Joint distance the gate uses: max over joints of the wrapped |a-b| (radians, wrap to (-pi,pi])."""
    return float(np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi).max())


def _pick_nearest(candQ_cell, okmask, ref):
    """Among a cell/voxel's CONVERGED candidate branches, the one closest (wrapped-Linf) to ref.
    Vectorised over the candidate axis so the BFS propagations stay fast."""
    valid = candQ_cell[okmask]                                       # (nok, dof)
    d = np.abs((valid - ref + np.pi) % (2.0 * np.pi) - np.pi).max(axis=1)
    return valid[int(np.argmin(d))]


def warm_start_branches(free, cand_q, cand_ok):
    """Continuity propagation over the FREE region: BFS out from a central stable cell; each cell is
    assigned the CANDIDATE branch closest (wrapped-Linf) to the branch already assigned to the
    neighbour it is first reached from. This is the warm-start done offline on the candidate menu.

    Effect: a SPURIOUS branch-hop collapses -- if a nearby branch exists in the cell's candidate set
    it gets chosen, so the jump to the neighbour vanishes. A REAL seam survives -- if the cell simply
    has no candidate near the neighbour, the smallest available jump is still large. So comparing the
    warm field against the raw (best-cost) field separates noise from genuine config-space seams.

    Returns q_warm (ny, nx, dof), NaN on FREE cells with no converged candidate."""
    ny, nx = free.shape
    K, dof = cand_q.shape[1], cand_q.shape[2]
    candQ = cand_q.reshape(ny, nx, K, dof)
    candOK = cand_ok.reshape(ny, nx, K)
    have = free & candOK.any(axis=2)
    q_warm = np.full((ny, nx, dof), np.nan, dtype=np.float32)
    if not have.any():
        return q_warm
    ys, xs = np.nonzero(have)                       # start at the FREE cell nearest the FREE centroid
    start = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    s0 = (int(ys[start]), int(xs[start]))
    q_warm[s0] = candQ[s0][np.flatnonzero(candOK[s0])[0]]      # seed cell: its best-cost branch
    seen = np.zeros_like(have)
    seen[s0] = True
    dq = deque([s0])
    while dq:
        iy, ix = dq.popleft()
        qref = q_warm[iy, ix]
        for dy, dx in _NEIGH8:
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < ny and 0 <= jx < nx and have[jy, jx] and not seen[jy, jx]:
                ks = np.flatnonzero(candOK[jy, jx])
                q_warm[jy, jx] = candQ[jy, jx, ks[int(np.argmin([_wrap_linf(candQ[jy, jx, k], qref)
                                                                 for k in ks]))]]
                seen[jy, jx] = True
                dq.append((jy, jx))
    return q_warm


def warm_start_branches_3d(free_vol, cand_q_vol, cand_ok_vol):
    """3D continuity propagation: like warm_start_branches but BFS over the whole FREE VOLUME,
    26-connected. Because it walks between adjacent z-slices, it enforces VERTICAL branch continuity
    -- the edges a climb-over route actually rides on. The per-slice 2D version never sees vertical
    neighbours, so its columns are branch-inconsistent (a voxel and the one directly above it can be
    on different branches for no reason). Each voxel takes the candidate branch nearest the branch of
    the neighbour it is first reached from. Returns q_warm (nz, ny, nx, dof), NaN where no candidate."""
    nz, ny, nx, _K, dof = cand_q_vol.shape
    have = free_vol & cand_ok_vol.any(axis=-1)
    q_warm = np.full((nz, ny, nx, dof), np.nan, dtype=np.float32)
    if not have.any():
        return q_warm
    zi, yi, xi = np.nonzero(have)                      # start at the FREE voxel nearest the 3D centroid
    start = int(np.argmin((zi - zi.mean()) ** 2 + (yi - yi.mean()) ** 2 + (xi - xi.mean()) ** 2))
    s0 = (int(zi[start]), int(yi[start]), int(xi[start]))
    q_warm[s0] = cand_q_vol[s0][np.flatnonzero(cand_ok_vol[s0])[0]]
    seen = np.zeros_like(have)
    seen[s0] = True
    dq = deque([s0])
    while dq:
        iz, iy, ix = dq.popleft()
        qref = q_warm[iz, iy, ix]
        for dz, dy, dx in _NEIGH26:
            jz, jy, jx = iz + dz, iy + dy, ix + dx
            if (0 <= jz < nz and 0 <= jy < ny and 0 <= jx < nx
                    and have[jz, jy, jx] and not seen[jz, jy, jx]):
                q_warm[jz, jy, jx] = _pick_nearest(cand_q_vol[jz, jy, jx], cand_ok_vol[jz, jy, jx], qref)
                seen[jz, jy, jx] = True
                dq.append((jz, jy, jx))
    return q_warm


_NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


_NEIGH26 = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)]
