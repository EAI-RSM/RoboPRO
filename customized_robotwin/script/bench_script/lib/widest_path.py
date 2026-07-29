"""Two- and three-dimensional widest-path searches."""

from collections import deque

import numpy as np

from .continuity import _NEIGH8, _NEIGH26, _wrap_linf
from .labeling import FREE


def nearest_free_cell(free, XX, YY, xy, max_dist):
    """Grid cell (iy, ix) of the nearest FREE cell to world point `xy`, or None if the
    closest FREE cell is farther than max_dist (m). Used to plant the DSU seeds: the grasp
    cell (target end) and the pad cell (other end)."""
    if not free.any():
        return None
    d2 = (XX - xy[0]) ** 2 + (YY - xy[1]) ** 2
    d2 = np.where(free, d2, np.inf)
    iy, ix = np.unravel_index(int(np.argmin(d2)), free.shape)
    dist = float(np.sqrt(d2[iy, ix]))
    if dist > max_dist:
        return None
    return (iy, ix), dist


def widest_path_eps(label, edt, seed_a, seed_b):
    """Widest-path (max-min) bottleneck between seed_a and seed_b over FREE cells, 8-connected.

    Kruskal: add FREE cells in DESCENDING clearance, unioning each with already-present FREE
    neighbours; the clearance of the cell whose addition first connects the two seeds is eps* --
    the tightest squeeze on the least-bad route (the max-spanning-tree bottleneck). Clearance
    VALUES come from the Euclidean EDT; 8-connectivity only decides whether a diagonal corner
    can be turned, so it never contaminates the number.

    (2.5D hook: the union step is exactly where the joint-space edge gate goes -- only union
    neighbours whose IK configs are close. In 2D, under the no-mazy-middle assumption, it's off.)

    Returns (eps_star_m, bottleneck_iyix, merged_bool)."""
    ny, nx = label.shape
    free = label == FREE
    free_flat = free.ravel()
    edt_flat = edt.ravel()
    parent = np.arange(ny * nx)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sa = seed_a[0] * nx + seed_a[1]
    sb = seed_b[0] * nx + seed_b[1]
    if not (free_flat[sa] and free_flat[sb]):
        return 0.0, None, False          # a seed isn't FREE -> nothing to connect

    idxs = np.flatnonzero(free_flat)
    order = idxs[np.argsort(-edt_flat[idxs], kind="stable")]     # descending clearance
    added = np.zeros(ny * nx, dtype=bool)

    for c in order:
        cy, cx = divmod(int(c), nx)
        added[c] = True
        for dy, dx in _NEIGH8:
            y2, x2 = cy + dy, cx + dx
            if 0 <= y2 < ny and 0 <= x2 < nx:
                nb = y2 * nx + x2
                if added[nb] and free_flat[nb]:
                    union(int(c), int(nb))
        if added[sa] and added[sb] and find(sa) == find(sb):
            return float(edt_flat[c]), (cy, cx), True

    return 0.0, None, False              # seeds never connect -> inaccessible


def reconstruct_widest_path(free, edt, seed_a, seed_b, eps_star):
    """Recover an actual widest-path route to draw. Every cell on the max-min path has clearance
    >= eps* (the bottleneck), and the DSU guarantees seed_a and seed_b are connected within the
    sub-grid {FREE and clearance >= eps*}. So a BFS through exactly those cells returns a valid
    bottleneck-optimal route (the shortest such, since BFS). Returns a list of (iy, ix) or None."""
    if eps_star is None:
        return None
    ny, nx = free.shape
    allowed = free & (edt >= eps_star - 1e-9)     # inf-clearance case: inf >= inf holds
    sa, sb = tuple(seed_a), tuple(seed_b)
    if not (allowed[sa] and allowed[sb]):
        return None
    prev = {sa: None}
    dq = deque([sa])
    while dq:
        cur = dq.popleft()
        if cur == sb:
            break
        cy, cx = cur
        for dy, dx in _NEIGH8:
            y2, x2 = cy + dy, cx + dx
            if 0 <= y2 < ny and 0 <= x2 < nx and allowed[y2, x2] and (y2, x2) not in prev:
                prev[(y2, x2)] = cur
                dq.append((y2, x2))
    if sb not in prev:
        return None
    path, node = [], sb
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


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
