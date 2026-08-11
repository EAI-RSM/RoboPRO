"""Two- and three-dimensional widest-path searches."""

from collections import deque
import heapq

import numpy as np

from .continuity import _NEIGH26, _wrap_linf
from .labeling import FREE


def nearest_free_voxel(free_vol, XX, YY, zs, xyz, max_dist):
    """Voxel (iz, iy, ix) of the nearest FREE voxel to world point xyz (m), or None if the closest is
    farther than max_dist. Used to plant the DSU seeds (grasp end / pad end) in the volume."""
    if not free_vol.any():
        return None
    dxy2 = (XX - xyz[0]) ** 2 + (YY - xyz[1]) ** 2               # (ny, nx), same for every slice
    best = None
    for iz in range(len(zs)):
        d2 = np.where(free_vol[iz], dxy2 + (zs[iz] - xyz[2]) ** 2, np.inf)
        iy, ix = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if best is None or d2[iy, ix] < best[0]:
            best = (float(d2[iy, ix]), (iz, int(iy), int(ix)))
    dist = float(np.sqrt(best[0]))
    return None if dist > max_dist else (best[1], dist)


def widest_path_eps_3d(label, edt, qvol, seed_a, seed_b, tau):
    """26-connected widest-path (Kruskal max-min occluder clearance) bottleneck between seed_a and
    seed_b over FREE voxels. If qvol is not None the edges are GATED: two FREE neighbours union only if
    their warm IK configs are within tau (branch-continuous) -- the mandatory 2.5D joint gate. Add
    voxels in DESCENDING clearance; the clearance of the voxel whose addition first connects the seeds
    is eps*. Clearance VALUES come from edt; the gate only decides whether an edge exists.
    Returns (eps_star_m, bottleneck (iz,iy,ix), merged_bool)."""
    nz, ny, nx = label.shape
    free = (label == FREE).ravel()
    edt_flat = edt.ravel()
    q_flat = qvol.reshape(-1, qvol.shape[-1]) if qvol is not None else None
    parent = np.arange(nz * ny * nx)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def flat(iz, iy, ix):
        return (iz * ny + iy) * nx + ix

    sa, sb = flat(*seed_a), flat(*seed_b)
    if not (free[sa] and free[sb]):
        return 0.0, None, False
    idxs = np.flatnonzero(free)
    order = idxs[np.argsort(-edt_flat[idxs], kind="stable")]      # descending clearance (inf first)
    added = np.zeros(nz * ny * nx, dtype=bool)
    for c in order:
        cz, rem = divmod(int(c), ny * nx)
        cy, cx = divmod(rem, nx)
        added[c] = True
        for dz, dy, dx in _NEIGH26:
            z2, y2, x2 = cz + dz, cy + dy, cx + dx
            if 0 <= z2 < nz and 0 <= y2 < ny and 0 <= x2 < nx:
                nb = flat(z2, y2, x2)
                if added[nb] and free[nb] and (q_flat is None or _wrap_linf(q_flat[c], q_flat[nb]) <= tau):
                    union(int(c), int(nb))
        if added[sa] and added[sb] and find(sa) == find(sb):
            return float(edt_flat[c]), (cz, cy, cx), True
    return 0.0, None, False


def reconstruct_widest_path_3d(free_vol, edt, qvol, seed_a, seed_b, eps_star, tau):
    """Recover an actual bottleneck-optimal route to draw: a gated BFS through {FREE and clearance >=
    eps*}, 26-connected. Returns a list of (iz,iy,ix) or None."""
    if eps_star is None:
        return None
    nz, ny, nx = free_vol.shape
    allowed = free_vol & (edt >= eps_star - 1e-9)                 # inf >= inf holds
    q_flat = qvol.reshape(-1, qvol.shape[-1]) if qvol is not None else None

    def flat(iz, iy, ix):
        return (iz * ny + iy) * nx + ix

    sa, sb = tuple(seed_a), tuple(seed_b)
    if not (allowed[sa] and allowed[sb]):
        return None
    prev = {sa: None}
    dq = deque([sa])
    while dq:
        cur = dq.popleft()
        if cur == sb:
            break
        cz, cy, cx = cur
        for dz, dy, dx in _NEIGH26:
            z2, y2, x2 = cz + dz, cy + dy, cx + dx
            if (0 <= z2 < nz and 0 <= y2 < ny and 0 <= x2 < nx and allowed[z2, y2, x2]
                    and (z2, y2, x2) not in prev
                    and (q_flat is None or _wrap_linf(q_flat[flat(cz, cy, cx)], q_flat[flat(z2, y2, x2)]) <= tau)):
                prev[(z2, y2, x2)] = cur
                dq.append((z2, y2, x2))
    if sb not in prev:
        return None
    path, node = [], sb
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


def reconstruct_clearance_preferred_path_3d(
    free_vol, edt, seed_a, seed_b, eps_star, res, zres
):
    """Choose a high-clearance representative from the eps*-optimal component.

    ``widest_path_eps_3d`` has already fixed the bottleneck.  This Dijkstra pass never leaves
    ``FREE & (edt >= eps_star)``; it only breaks the many-way tie between bottleneck-optimal
    routes.  Edge cost is anisotropic physical length divided by the smaller endpoint clearance.
    Unbounded clearance is capped at the largest finite allowed value; an all-unbounded component
    falls back to physical shortest path.

    This is intentionally ungated and is used only by the geometric/reporting path.
    """
    if eps_star is None:
        return None
    allowed = free_vol & (edt >= eps_star - 1e-9)
    sa, sb = tuple(seed_a), tuple(seed_b)
    if not (allowed[sa] and allowed[sb]):
        return None

    finite = edt[allowed & np.isfinite(edt)]
    all_unbounded = finite.size == 0
    finite_cap = float(finite.max()) if finite.size else 1.0

    dist = {sa: 0.0}
    prev = {sa: None}
    heap = [(0.0, sa)]
    nz, ny, nx = allowed.shape
    while heap:
        cost, cur = heapq.heappop(heap)
        if cost > dist.get(cur, np.inf):
            continue
        if cur == sb:
            break
        cz, cy, cx = cur
        for dz, dy, dx in _NEIGH26:
            nxt = (cz + dz, cy + dy, cx + dx)
            z2, y2, x2 = nxt
            if not (0 <= z2 < nz and 0 <= y2 < ny and 0 <= x2 < nx and allowed[nxt]):
                continue
            length = float(np.sqrt((dx * res) ** 2 + (dy * res) ** 2 + (dz * zres) ** 2))
            if all_unbounded:
                edge_cost = length
            else:
                c0 = finite_cap if np.isinf(edt[cur]) else float(edt[cur])
                c1 = finite_cap if np.isinf(edt[nxt]) else float(edt[nxt])
                edge_cost = length / max(min(c0, c1), 1e-12)
            new_cost = cost + edge_cost
            if new_cost + 1e-15 < dist.get(nxt, np.inf):
                dist[nxt] = new_cost
                prev[nxt] = cur
                heapq.heappush(heap, (new_cost, nxt))

    if sb not in prev:
        return None
    path, node = [], sb
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]
