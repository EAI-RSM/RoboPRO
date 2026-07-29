#!/usr/bin/env python3
"""Checks for the randomized occluder formation (--random-ring-rotation / --num-occluders
menu / --offsets range).

The property that matters most is DETERMINISM per (seed, offset-spec): the formation is
drawn once and must come out identical in the pad-distance pre-check, the Pass-1 measurement
build and the Pass-2 rollout build. If it ever differed, the scene that gets measured would
not be the scene that gets rolled out, and every visibility number would be attached to the
wrong geometry -- silently.

Run:  python test_ring_config.py
"""
import sys

import numpy as np

import lib.occluder_ring as A


def test_parse_offset_specs():
    assert A.parse_offset_specs("0.2") == [(0.2, 0.2, 0.2)]
    lo, hi, nom = A.parse_offset_specs("0.1-0.25")[0]
    assert (lo, hi) == (0.1, 0.25) and abs(nom - 0.175) < 1e-12, (lo, hi, nom)
    mixed = A.parse_offset_specs("0.15, 0.20-0.30 ,0.25")
    assert len(mixed) == 3, mixed
    assert mixed[0] == (0.15, 0.15, 0.15) and mixed[2] == (0.25, 0.25, 0.25)
    assert mixed[1][:2] == (0.20, 0.30)
    # reversed bounds are normalized rather than producing an empty interval
    assert A.parse_offset_specs("0.3-0.1")[0][:2] == (0.1, 0.3)
    # a leading minus is a sign, not a separator
    assert A.parse_offset_specs("-0.2")[0] == (-0.2, -0.2, -0.2)
    print("  [1] parse_offset_specs: fixed / range / mixed / reversed / signed      PASS")


def test_parse_count_choices():
    assert A.parse_count_choices("1") == [1]
    assert A.parse_count_choices("2,3,4,5") == [2, 3, 4, 5]
    assert A.parse_count_choices(" 2 , 4 ") == [2, 4]
    for bad in ("", "  ", "-1"):
        try:
            A.parse_count_choices(bad)
        except SystemExit:
            continue
        raise AssertionError(f"--num-occluders {bad!r} should have been rejected")
    print("  [2] parse_count_choices: single / menu / whitespace / rejects garbage  PASS")


def test_draw_is_deterministic():
    """Same (seed, spec) -> byte-identical formation, every time. This is the invariant the
    measure-build and the rollout-build both rely on."""
    spec = A.parse_offset_specs("0.1-0.25")[0]
    for seed in (0, 1, 7, 42, 12345):
        a = A.draw_ring_config(seed, spec, [2, 3, 4, 5], True)
        for _ in range(5):
            b = A.draw_ring_config(seed, spec, [2, 3, 4, 5], True)
            assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2], (seed, a, b)
    print("  [3] draw_ring_config is deterministic per (seed, spec)                 PASS")


def test_rotation_flag():
    spec = A.parse_offset_specs("0.2")[0]
    for seed in range(30):
        angle0, _, _ = A.draw_ring_config(seed, spec, [3], False)
        assert angle0 == 0.0, f"rotation off must give exactly 0, got {angle0}"
    angles = [A.draw_ring_config(s, spec, [3], True)[0] for s in range(200)]
    assert all(0.0 <= t < 2 * np.pi for t in angles), "theta outside [0, 2pi)"
    assert len(set(angles)) > 150, "theta barely varies across seeds"
    # covers the whole circle, not just one lobe
    hist, _ = np.histogram(angles, bins=8, range=(0, 2 * np.pi))
    assert (hist > 0).all(), f"some quadrants never sampled: {hist}"
    print("  [4] rotation: exactly 0 when off; spans [0,2pi) when on                PASS")


def test_count_menu():
    spec = A.parse_offset_specs("0.2")[0]
    menu = [2, 3, 4, 5]
    drawn = [A.draw_ring_config(s, spec, menu, True)[1] for s in range(300)]
    assert set(drawn) <= set(menu), f"drew a count outside the menu: {set(drawn) - set(menu)}"
    assert set(drawn) == set(menu), f"menu not fully covered in 300 seeds: {sorted(set(drawn))}"
    # a single-value menu is pinned
    assert {A.draw_ring_config(s, spec, [1], True)[1] for s in range(20)} == {1}
    print("  [5] count menu: only menu values, all of them reachable                PASS")


def test_per_occluder_radii():
    fixed = A.parse_offset_specs("0.2")[0]
    rng_spec = A.parse_offset_specs("0.1-0.25")[0]
    # fixed spec -> every occluder at exactly that radius
    _, n, radii = A.draw_ring_config(3, fixed, [4], True)
    assert len(radii) == n == 4 and all(r == 0.2 for r in radii), radii
    # range spec -> in-range, and INDEPENDENT per occluder (not one radius reused)
    varied = 0
    for seed in range(60):
        _, n, radii = A.draw_ring_config(seed, rng_spec, [4], True)
        assert len(radii) == n, (n, radii)
        assert all(0.1 <= r <= 0.25 for r in radii), radii
        if len(set(np.round(radii, 9))) > 1:
            varied += 1
    assert varied > 55, f"radii look shared, not per-occluder ({varied}/60 varied)"
    print("  [6] radii: fixed spec constant; range spec per-occluder and in-range   PASS")


def test_ring_geometry():
    """Per-occluder radii place each bottle at ITS OWN distance, with angles still even."""
    cx, cy = 0.05, 0.10
    radii = [0.10, 0.18, 0.25, 0.13]
    pts = A.occluder_ring_xy(cx, cy, radii, 4, angle0=0.0)
    assert len(pts) == 4, pts
    for k, (x, y) in enumerate(pts):
        d = float(np.hypot(x - cx, y - cy))
        assert abs(d - radii[k]) < 1e-9, (k, d, radii[k])
    # angles evenly spaced by 2pi/n regardless of the radii
    angs = [np.arctan2(x - cx, -(y - cy)) % (2 * np.pi) for x, y in pts]
    for k in range(1, 4):
        step = (angs[k] - angs[k - 1]) % (2 * np.pi)
        assert abs(step - 2 * np.pi / 4) < 1e-9, (k, step)

    # scalar path unchanged: k=0 at angle0=0 sits directly in FRONT (-y)
    p0 = A.occluder_ring_xy(0.0, 0.0, 0.2, 1, angle0=0.0)
    assert len(p0) == 1 and abs(p0[0][0]) < 1e-12 and abs(p0[0][1] + 0.2) < 1e-12, p0

    # rotating by theta rotates the whole formation about the target
    rot = A.occluder_ring_xy(cx, cy, radii, 4, angle0=0.7)
    for k, ((x0, y0), (x1, y1)) in enumerate(zip(pts, rot)):
        assert abs(np.hypot(x1 - cx, y1 - cy) - np.hypot(x0 - cx, y0 - cy)) < 1e-9, k
    a0 = np.arctan2(pts[0][0] - cx, -(pts[0][1] - cy)) % (2 * np.pi)
    a1 = np.arctan2(rot[0][0] - cx, -(rot[0][1] - cy)) % (2 * np.pi)
    assert abs(((a1 - a0) % (2 * np.pi)) - 0.7) < 1e-9, (a0, a1)

    # off-table positions are dropped, the rest keep their places
    tight = A.occluder_ring_xy(0.0, 0.0, [0.2, 0.9, 0.2, 0.9], 4,
                               xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    assert len(tight) == 2, tight
    print("  [7] ring geometry: own radius per k, even angles, rotation, off-table  PASS")


def test_ab_configuration():
    """The exact settings the A/B curated scene uses, end to end."""
    spec = A.parse_offset_specs("0.1-0.25")[0]
    menu = A.parse_count_choices("2,3,4,5")
    counts, thetas = set(), []
    for seed in range(200):
        angle0, n, radii = A.draw_ring_config(seed, spec, menu, True)
        counts.add(n)
        thetas.append(angle0)
        assert 2 <= n <= 5 and len(radii) == n
        assert all(0.1 <= r <= 0.25 for r in radii), radii
    assert counts == {2, 3, 4, 5}, counts
    assert max(thetas) - min(thetas) > 5.5, "rotation not spanning the circle"
    print("  [8] A/B curated config (rot on, n in 2-5, r in 0.10-0.25)              PASS")


def main():
    print("randomized occluder formation -- checks")
    test_parse_offset_specs()
    test_parse_count_choices()
    test_draw_is_deterministic()
    test_rotation_flag()
    test_count_menu()
    test_per_occluder_radii()
    test_ring_geometry()
    test_ab_configuration()
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
