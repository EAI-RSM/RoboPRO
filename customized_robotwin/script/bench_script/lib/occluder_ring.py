"""Deterministic occluder-ring parsing and placement."""

import numpy as np


def occluder_ring_xy(cx, cy, radius, n, angle0=0.0, xlim=None, ylim=None):
    """(x, y) centres of `n` occluders equally spaced IN ANGLE around the target at (cx, cy)
    -- the formation cuts the circle into n equal 2*pi/n slices. Angle is measured from the
    FRONT (-y, the robot/camera side) and rotates toward +x, so:
      * n=1, angle0=0 -> one occluder directly in front (the original single-box layout),
      * the +x quarter-turn position reproduces the old side_occluder_sign=+1 box.

    `radius` is either a scalar (a true circle, the original behaviour) or a per-occluder
    sequence of length >= n, in which case occluder k sits at its OWN radius -- the angles
    stay evenly spaced but the formation is no longer a circle. That is what lets a scene
    mix near and far occluders instead of presenting one uniform wall.

    If xlim/ylim (each a (lo, hi) pair) are given, any position whose CENTRE falls outside
    them is DROPPED -- the formation keeps the same equal spacing, just with fewer bottles
    where it would leave the table, instead of failing the whole scene.
    Returns [] for n<=0. Deterministic; the caller adds any per-actor yaw jitter."""
    pts = []
    n = int(n)
    scalar = np.isscalar(radius)
    for k in range(max(0, n)):
        r = float(radius) if scalar else float(radius[k])
        theta = angle0 + 2.0 * np.pi * k / n
        x = cx + r * np.sin(theta)
        y = cy - r * np.cos(theta)
        if xlim is not None and not (xlim[0] <= x <= xlim[1]):
            continue
        if ylim is not None and not (ylim[0] <= y <= ylim[1]):
            continue
        pts.append((x, y))
    return pts


def parse_offset_specs(text):
    """Parse --offsets into a list of specs. Each comma-separated token is either

      "0.2"        a FIXED ring radius (every occluder at 0.2) -- the original behaviour, or
      "0.1-0.25"   a RANGE: every occluder independently draws its own radius from
                   U[0.1, 0.25], so one scene can hold both near and far occluders.

    Returns [(lo, hi, nominal), ...]; lo == hi for a fixed token. `nominal` is the value used
    to LABEL and group the spec (the midpoint of a range), so --group-by offset still buckets
    a range into one group instead of scattering every scene into its own."""
    specs = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:                      # [1:] so a leading minus is not a separator
            i = tok.index("-", 1)
            lo, hi = float(tok[:i]), float(tok[i + 1:])
            if hi < lo:
                lo, hi = hi, lo
            specs.append((lo, hi, 0.5 * (lo + hi)))
        else:
            v = float(tok)
            specs.append((v, v, v))
    if not specs:
        raise SystemExit("--offsets parsed to nothing")
    return specs


def parse_count_choices(text):
    """Parse --num-occluders into the list of counts a scene may draw from.
    "1" -> always 1 (original behaviour); "2,3,4,5" -> one of those, drawn per scene."""
    vals = [int(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals or any(v < 0 for v in vals):
        raise SystemExit(f"--num-occluders must be one or more non-negative ints, got {text!r}")
    return vals


def draw_ring_config(seed, spec, count_choices, random_rotation):
    """The occluder formation for one (seed, offset-spec), drawn DETERMINISTICALLY.

    Determinism is the whole point: the formation is decided once and must come out identical
    in the pad-distance pre-check, the Pass-1 measurement build and the Pass-2 rollout build,
    or the scene that gets measured is not the scene that gets rolled out. Everything is drawn
    from one RNG keyed on (seed, spec nominal), matching how the existing occluder coin flip
    is keyed.

    Returns (angle0, n, radii): ring rotation in radians (0 unless random_rotation), the
    occluder count drawn from count_choices, and the per-occluder radii list."""
    lo, hi, nominal = spec
    rng = np.random.default_rng(int(seed) * 1000 + int(round(nominal * 100)) + 7919)
    n = int(count_choices[rng.integers(len(count_choices))]) if len(count_choices) > 1 \
        else int(count_choices[0])
    angle0 = float(rng.uniform(0.0, 2.0 * np.pi)) if random_rotation else 0.0
    radii = [float(rng.uniform(lo, hi)) for _ in range(max(0, n))] if hi > lo \
        else [float(lo)] * max(0, n)
    return angle0, n, radii
