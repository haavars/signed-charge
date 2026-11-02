"""Ground-truth minutiae from a clean (noise-free) ridge image.

Extracted per impression from the warped, binarised ridge pattern before
rendering noise is added, so positions are in that impression's pixel
coordinates. Angles follow ANSI/INCITS 378, as NIST's `mindtct -m1` writes
them: degrees counter-clockwise from the x axis (y up), a ridge ending pointing
back along the ridge it ends, a bifurcation pointing away from its stem, between
its two branches. Until ridgegen/0.4 an ending pointed the other way, away from
its ridge (180 degrees off; bifurcations were already right).

A finger's master has minutiae too, with ids (`master`), and each impression
minutia gets the id of the master minutia it is (`identify`): the same id in
two impressions is the same point on the skin, a mated minutia.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def extract(ridge_binary, valid, border=18, min_distance=8, trace=10):
    """Return minutiae as dicts {x, y, angle, type} ('ending' / 'bifurcation')."""
    skeleton = skeletonize(ridge_binary.astype(bool)).astype(np.uint8)
    # Stay away from the contact edge, where the skeleton ends artificially.
    inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=border)

    padded = np.pad(skeleton, 1)
    # Crossing number: half the number of 0/1 transitions around the pixel.
    ring = [padded[1 + dy : padded.shape[0] - 1 + dy, 1 + dx : padded.shape[1] - 1 + dx] for dy, dx in NEIGHBOURS]
    transitions = sum(np.abs(ring[i].astype(int) - ring[(i + 1) % 8].astype(int)) for i in range(8)) // 2

    candidates = []
    for kind, cn in (("ending", 1), ("bifurcation", 3)):
        ys, xs = np.nonzero((skeleton == 1) & (transitions == cn) & (inner > 0))
        candidates += [(int(x), int(y), kind) for x, y in zip(xs, ys)]

    # Drop clusters (short spurs, tiny gaps) that are artefacts, not minutiae.
    kept = []
    for x, y, kind in candidates:
        close = [k for k in kept if (k[0] - x) ** 2 + (k[1] - y) ** 2 < min_distance**2]
        if close:
            for k in close:
                kept.remove(k)
            continue
        kept.append((x, y, kind))

    minutiae = []
    for x, y, kind in kept:
        angle = direction(skeleton, x, y, kind, trace)
        if angle is not None:
            minutiae.append({"x": x, "y": y, "angle": angle, "type": kind})
    return minutiae


def direction(skeleton, x, y, kind, steps):
    """The minutia's angle, from where its branches lead `steps` pixels on, or None if they can't be traced.

    A branch is a run of skeleton pixels round (x, y), as the crossing number counts them: where the skeleton
    is two pixels wide by the minutia, two neighbours are one branch. Until ridgegen/0.8 each neighbour was a
    branch, so such minutiae were dropped, a tenth of them, at random in each impression.
    """
    ring = [(x + dx, y + dy) for dy, dx in NEIGHBOURS]
    on = [on_skeleton(skeleton, px, py) for px, py in ring]
    branches = []
    for start in range(8):
        if on[start] and not on[start - 1]:
            run, i = [], start
            while on[i % 8]:
                run.append(ring[i % 8])
                i += 1
            branches.append(walk(skeleton, x, y, run, set(ring) | {(x, y)}, steps))
    branches = [b for b in branches if b is not None]
    if kind == "ending" and len(branches) == 1:
        bx, by = branches[0]
        # Points back along the ridge it terminates.
        return round(float(np.degrees(np.arctan2(-(by - y), bx - x))) % 360, 1)
    if kind == "bifurcation" and len(branches) == 3:
        angles = [np.arctan2(-(by - y), bx - x) for bx, by in branches]
        # The stem is the branch pointing most away from the other two.
        def spread(i):
            return sum(abs(np.angle(np.exp(1j * (angles[i] - angles[j])))) for j in range(3) if j != i)
        stem = max(range(3), key=spread)
        return round(float(np.degrees(angles[stem] + np.pi)) % 360, 1)
    return None


def on_skeleton(skeleton, x, y):
    h, w = skeleton.shape
    return 0 <= y < h and 0 <= x < w and bool(skeleton[y, x])


def neighbours_on(skeleton, x, y):
    return [(x + dx, y + dy) for dy, dx in NEIGHBOURS if on_skeleton(skeleton, x + dx, y + dy)]


def walk(skeleton, x, y, run, visited, steps):
    """Follow the skeleton from (x, y) out through `run`, its first pixels, for up to `steps` pixels.

    Where the skeleton is two pixels wide the walk takes the step leading furthest from (x, y); it stops where
    the branch ends or splits again. Returns the last pixel, or None for a branch shorter than two pixels.
    """
    visited = set(visited)
    cx, cy = max(run, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    taken = 1
    for _ in range(steps):
        nxt = [p for p in neighbours_on(skeleton, cx, cy) if p not in visited]
        visited.update(nxt)
        far = [p for p in nxt if (p[0] - x) ** 2 + (p[1] - y) ** 2 > (cx - x) ** 2 + (cy - y) ** 2]
        if not far or len(far) > 2:
            break
        cx, cy = max(far, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        taken += 1
    return (cx, cy) if taken > 1 else None


def master(finger, ridges):
    """The minutiae of a finger's master, in master coordinates, each with an id: `f02-m017` is minutia 17 of
    FGP 2, counted in reading order (top to bottom, left to right)."""
    points = extract(ridges > 0, np.ones(ridges.shape, bool))
    points.sort(key=lambda m: (m["y"], m["x"]))
    return [{"id": f"f{finger:02d}-m{i:03d}", **m} for i, m in enumerate(points)]


def identify(points, masters, to_master, radius, max_angle=30, step=3):
    """The id of the master minutia each of `points` is, or None where it's none of them.

    `to_master` maps an (n, 2) array of the points' pixel positions to master coordinates. Each point's
    direction goes along (a point `step` px ahead of it), so its angle is compared in master coordinates.
    Pairs are one to one, of the same type, within `radius` px and `max_angle` degrees. On procedural prints
    98 % of an impression's minutiae land within half a ridge period of their master minutia, all of the same
    type; the rest are where binarising the warped ridges joined or broke a ridge the master doesn't.
    """
    if not points or not masters:
        return [None] * len(points)
    xy = np.array([[p["x"], p["y"]] for p in points], float)
    angle = np.radians([p["angle"] for p in points])
    src = to_master(xy)
    ahead = to_master(xy + step * np.stack([np.cos(angle), -np.sin(angle)], 1))
    angle = np.degrees(np.arctan2(src[:, 1] - ahead[:, 1], ahead[:, 0] - src[:, 0]))

    target = np.array([[m["x"], m["y"], m["angle"]] for m in masters], float)
    distance = np.hypot(src[:, None, 0] - target[None, :, 0], src[:, None, 1] - target[None, :, 1])
    turn = np.abs((angle[:, None] - target[None, :, 2] + 180) % 360 - 180)
    same = np.array([p["type"] for p in points])[:, None] == np.array([m["type"] for m in masters])[None, :]
    allowed = (distance <= radius) & (turn <= max_angle) & same
    ids = [None] * len(points)
    for i, j in zip(*linear_sum_assignment(np.where(allowed, distance, 1e6))):
        if allowed[i, j]:
            ids[i] = masters[j]["id"]
    return ids
