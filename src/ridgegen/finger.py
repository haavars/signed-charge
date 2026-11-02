"""Master fingerprints: the full ridge pattern of one synthetic finger.

A master is generated once per (subject seed, finger) and every impression of
that finger (rolled, plain, slap, later captures) is derived from it, so all of
them share the same ridge pattern and minutiae.

Finger numbers follow ANSI/NIST-ITL FGP codes: 1 right thumb ... 5 right little,
6 left thumb ... 10 left little. Images are "as printed": a right hand's ulnar
(little-finger) side is on the right, so an ulnar loop on a right hand is a
right loop and on a left hand a left loop.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from . import minutiae, synthesis
from .synthesis import grow_ridges, rng_for, smooth_noise, smooth_orientation, zero_pole_orientation

# Master canvas covering the whole unrolled fingertip surface, at 500 ppi.
MASTER_W, MASTER_H = 820, 900

# Pattern class priors per finger (thumb, index, middle, ring, little), roughly
# following published population frequencies: loops dominate, whorls are common
# on thumbs and ring fingers, arches and radial loops mostly on index fingers.
CLASS_PRIORS = {
    "thumb": {"whorl": 0.45, "ulnar_loop": 0.47, "radial_loop": 0.02, "arch": 0.03, "tented_arch": 0.03},
    "index": {"whorl": 0.30, "ulnar_loop": 0.37, "radial_loop": 0.17, "arch": 0.07, "tented_arch": 0.09},
    "middle": {"whorl": 0.22, "ulnar_loop": 0.66, "radial_loop": 0.03, "arch": 0.05, "tented_arch": 0.04},
    "ring": {"whorl": 0.42, "ulnar_loop": 0.52, "radial_loop": 0.01, "arch": 0.03, "tented_arch": 0.02},
    "little": {"whorl": 0.14, "ulnar_loop": 0.82, "radial_loop": 0.01, "arch": 0.02, "tented_arch": 0.01},
}

# Which kind a whorl is, Henry's whorl subclasses, roughly as often as they're
# counted: plain whorls most, double loops and central pocket loops next,
# accidentals rare. A plain whorl is "whorl", as before there were subclasses.
WHORL_TYPES = {"whorl": 0.62, "double_loop": 0.19, "central_pocket_loop": 0.17, "accidental": 0.02}
WHORLS = tuple(WHORL_TYPES)

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]

# Relative fingertip size: ridge spacing and contact area scale with it.
FINGER_SCALE = {"thumb": 1.12, "index": 1.0, "middle": 1.02, "ring": 0.97, "little": 0.88}

MASTER_SALT = 1
# The whorl subclass is drawn from its own randomness, so fingers that aren't
# whorls, and plain whorls, have the same pattern as before subclasses.
WHORL_SALT = 2

# Master models. 1 is ridgegen's to 0.24; 2 since 0.25: each person's fingers drawn
# from published variation, where model 1 gave every loop and whorl one template (core and delta twice as far apart
# as real, in a fixed box at a fixed angle, so the same ridge count on every finger, and ridge spacing half as
# varied), and different people's prints matched far too often. A version's latents are drawn again with its model
# (`model_of`), so stored runs keep their fingers.
MODEL = 2
MODEL_2_SINCE = (0, 25, 0)

# Pattern frequencies, % by hand and finger: plain whorl, lateral pocket and twin loop (double loop), central pocket
# loop, accidental, ulnar loop, radial loop, and arches, plain and tented together (Cummins & Midlo 1943, Finger
# Prints, Palms and Soles, Table 1, p. 67: Scotland Yard's 5,000 people, mainly British; its tented arches are
# under-separated, so arches are split `TENTED_SHARE` tented, as Wijerathne et al. 2013, J Physiol Anthropol 32:27).
PATTERNS = {
    ("right", "thumb"): (31.86, 8.79, 0.74, 0.04, 55.89, 0.22, 2.47),
    ("left", "thumb"): (19.19, 9.84, 0.36, 0.00, 65.90, 0.20, 4.51),
    ("right", "index"): (25.03, 2.94, 2.40, 0.42, 32.30, 26.03, 10.87),
    ("left", "index"): (22.02, 3.50, 2.01, 0.62, 38.10, 23.37, 10.36),
    ("right", "middle"): (13.98, 1.39, 1.15, 0.04, 74.81, 2.53, 6.09),
    ("left", "middle"): (13.21, 2.11, 0.80, 0.06, 73.32, 2.51, 7.98),
    ("right", "ring"): (34.85, 0.64, 5.50, 0.08, 55.61, 1.47, 1.85),
    ("left", "ring"): (22.11, 1.07, 4.64, 0.00, 68.92, 0.50, 2.75),
    ("right", "little"): (11.41, 0.38, 2.01, 0.00, 85.46, 0.20, 0.54),
    ("left", "little"): (6.86, 0.62, 1.53, 0.02, 89.79, 0.02, 1.15),
}
PATTERN_KINDS = ("whorl", "double_loop", "central_pocket_loop", "accidental", "ulnar_loop", "radial_loop", "arches")
TENTED_SHARE = 0.45

# Ridge count between core and delta (ridges crossing the line between them), mean and SD by finger, for an ulnar
# loop: Karmakar & Kobyliansky 2012 (Papers on Anthropology XXI:110-122, Table 3, men and women) and Cummins & Midlo
# 1943 (Table 22). A radial loop's is lower, a whorl's larger count higher (the same tables), for a total finger
# ridge count about 140 (Norovsambuu et al. 2021, PLoS ONE 16:e0252831: men 148, women 132; Karmakar: 161, 156).
LOOP_COUNT = {"thumb": (18.5, 5.0), "index": (11.5, 5.0), "middle": (13.0, 4.7), "ring": (12.5, 5.0),
              "little": (12.5, 4.5)}
RADIAL_LESS = 4.0
WHORL_MORE, WHORL_SD = 4.5, 4.3
COUNT_RANGE = (1, 34)
# Core to delta per ridge counted, in ridge periods (the line crosses ridges at a slant, and they widen round the
# singular points), as the zero-pole field grows them; fitted on masters (signed_charge.exploratory.master_traits).
LOOP_REACH = 1.53
WHORL_REACH = 1.19
# A person's fingers run high or low together: each finger's count shares this correlation with a draw for the
# person, chosen so the total's spread between people is about the 35-41 published (Norovsambuu et al. 2021, Table
# 4; Karmakar & Kobyliansky 2012, Table 5); drawn alone, the fingers gave 24.
COUNT_PERSON = 0.6
# Where the delta lies from the core, degrees off straight down: drawn about the angle model 1 always had.
DELTA_ANGLE = (45, 12, 15, 75)  # mean, SD, clipped to

# Ridge spacing in the rolled print, mm: about 0.43 for men and 0.40 for women in the distal pad, varying between
# people by about a tenth, thumb and index coarsest, ring finest, about a tenth apart (Carvalho et al. 2022, Research,
# Society and Development 11:e7311528057, rolled prints at about 500 ppi, Tables 2 and 8). Model 1 had 0.43 with
# half the spread, thumbs finest.
SPACING_MM = 0.42
SPACING_PERSON_SD = 0.10
SPACING_FINGER = {"thumb": 1.06, "index": 1.05, "middle": 0.99, "ring": 0.96, "little": 0.98}
SPACING_FINGER_SD = 0.04
PERSON_SALT = 4


@dataclass
class Master:
    finger: int
    hand: str
    name: str
    # arch, tented_arch, left_loop, right_loop, or a whorl: whorl (plain),
    # central_pocket_loop, double_loop, accidental
    pattern: str
    cores: list
    deltas: list
    period: float
    scale: float
    ridges: np.ndarray = field(repr=False)  # float32 in [-1, 1], ridges positive
    orientation: np.ndarray = field(repr=False)
    # The finger's true minutiae, in master coordinates, with ids (`minutiae.master`). Every impression's
    # minutiae are matched to them.
    minutiae: list = field(default_factory=list, repr=False)


def hand_of(finger):
    return "right" if finger <= 5 else "left"


def name_of(finger):
    return FINGER_NAMES[(finger - 1) % 5]


def choose_pattern(finger, rng):
    priors = CLASS_PRIORS[name_of(finger)]
    kinds = list(priors)
    kind = rng.choice(kinds, p=np.array([priors[k] for k in kinds]) / sum(priors.values()))
    if kind in ("ulnar_loop", "radial_loop"):
        ulnar_side = "right" if hand_of(finger) == "right" else "left"
        radial_side = "left" if ulnar_side == "right" else "right"
        return f"{ulnar_side if kind == 'ulnar_loop' else radial_side}_loop"
    return str(kind)


def whorl_type(rng):
    return str(rng.choice(WHORLS, p=np.array(list(WHORL_TYPES.values())) / sum(WHORL_TYPES.values())))


def singular_points(pattern, rng):
    """Cores and deltas (master coordinates) for a pattern class."""
    cx = MASTER_W * (0.5 + rng.normal(0, 0.03))
    cy = MASTER_H * (0.40 + rng.normal(0, 0.03))

    if pattern in ("left_loop", "right_loop"):
        # The delta sits below the core, on the side opposite the loop's opening.
        side = -1 if pattern == "right_loop" else 1
        dx = side * rng.uniform(170, 250)
        dy = rng.uniform(200, 280)
        return [(cx - 0.15 * dx, cy)], [(cx + dx, cy + dy)]

    if pattern == "whorl":
        gap = rng.uniform(40, 90)
        tilt = rng.normal(0, 25)
        cores = [(cx - tilt / 2, cy - gap / 2), (cx + tilt / 2, cy + gap / 2)]
        spread = rng.uniform(230, 290)
        drop = rng.uniform(210, 280)
        deltas = [
            (cx - spread, cy + drop + rng.normal(0, 25)),
            (cx + spread, cy + drop + rng.normal(0, 25)),
        ]
        return cores, deltas

    if pattern == "central_pocket_loop":
        # A loop whose innermost ridges curl into a small whorl: two cores close
        # together, and the deltas lopsided, one near under the pocket and one far
        # out on the side the outer loop comes from, so the line between them
        # passes below the pocket.
        side = rng.choice([-1, 1])
        gap = rng.uniform(22, 40)
        shift = side * rng.uniform(4, 14)
        cores = [(cx - shift, cy - gap / 2), (cx + shift, cy + gap / 2)]
        deltas = [
            (cx - side * rng.uniform(120, 170), cy + rng.uniform(130, 170)),
            (cx + side * rng.uniform(300, 370), cy + rng.uniform(280, 340)),
        ]
        return cores, deltas

    if pattern == "double_loop":
        # Two loops wrapped around each other, one opening to each side: two
        # cores well apart, one up and to the side of the other.
        side = rng.choice([-1, 1])
        gap = rng.uniform(110, 170)
        shift = side * rng.uniform(50, 110)
        cores = [(cx - shift / 2, cy - gap / 2), (cx + shift / 2, cy + gap / 2)]
        spread = rng.uniform(240, 300)
        drop = rng.uniform(160, 220)
        deltas = [
            (cx - spread, cy + drop + rng.normal(0, 25)),
            (cx + spread, cy + drop + rng.normal(0, 25)),
        ]
        return cores, deltas

    if pattern == "accidental":
        # Two patterns in one: a whorl with a loop beside and above it, so three
        # deltas, the loop's own off to its side.
        side = rng.choice([-1, 1])
        gap = rng.uniform(40, 70)
        wx = cx - side * rng.uniform(40, 70)
        cores = [(wx, cy - gap / 2 + 30), (wx, cy + gap / 2 + 30)]
        spread = rng.uniform(220, 260)
        drop = rng.uniform(230, 270)
        deltas = [(wx - spread, cy + drop), (wx + spread, cy + drop)]
        lx, ly = cx + side * rng.uniform(150, 190), cy - rng.uniform(90, 130)
        cores.append((lx, ly))
        deltas.append((lx + side * rng.uniform(90, 130), ly + rng.uniform(90, 130)))
        return cores, deltas

    if pattern == "tented_arch":
        return [(cx, cy + 20)], [(cx + rng.normal(0, 8), cy + rng.uniform(90, 130))]

    return [], []  # plain arch


def model_of(version):
    """The master model ridgegen `version` ("0.24.0", None for this one) drew its fingers with."""
    if version is None:
        return MODEL
    return 2 if tuple(int(part) for part in version.split(".")) >= MODEL_2_SINCE else 1


def finger_scale(seed, finger):
    """A finger's scale in its rolled prints, before each capture's own (`fingerprints.rolled_capture`)."""
    return float(rng_for(seed, finger, 1, 9).uniform(0.86, 0.94))


def print_scale(seed, finger, scale):
    """How large a master of `scale` is drawn in its rolled prints: its finger's scale, with large masters drawn
    smaller to fit the frame (`fingerprints.rolled_capture`)."""
    return finger_scale(seed, finger) * min(1.0, 1.02 / scale)


def print_period(seed, finger):
    """The ridge period of `finger`'s rolled prints, px at 500 ppi: its master's at the scale it's drawn there
    (`print_scale`, within each capture's 2.5 %). Its plain and tip impressions are drawn at about the same scale."""
    m = master(seed, finger)
    return m.period * print_scale(seed, finger, m.scale)


def choose_pattern_by_hand(finger, rng):
    """Model 2's pattern for `finger`, by its hand (`PATTERNS`): (pattern, radial), radial for a radial loop."""
    weights = np.array(PATTERNS[(hand_of(finger), name_of(finger))])
    kind = str(rng.choice(PATTERN_KINDS, p=weights / weights.sum()))
    if kind == "arches":
        return ("tented_arch" if rng.random() < TENTED_SHARE else "arch"), False
    if kind in ("ulnar_loop", "radial_loop"):
        ulnar = "right" if hand_of(finger) == "right" else "left"
        side = ulnar if kind == "ulnar_loop" else ("left" if ulnar == "right" else "right")
        return f"{side}_loop", kind == "radial_loop"
    return kind, False


def ridge_period(seed, finger, scale, rng):
    """Model 2's ridge period on the master, px: the spacing its rolled print should have (`SPACING_MM`, by person
    and finger) at the scale it's drawn there (`print_scale`)."""
    person = np.exp(rng_for(seed, 0, PERSON_SALT).normal(0, SPACING_PERSON_SD))
    mm = SPACING_MM * person * SPACING_FINGER[name_of(finger)] * np.exp(rng.normal(0, SPACING_FINGER_SD))
    return float(np.clip(mm * (500 / 25.4) / print_scale(seed, finger, scale), 7.0, 14.0))


def person_count(seed):
    """The person's share of their fingers' ridge counts, in SDs (`COUNT_PERSON`)."""
    return float(rng_for(seed, 0, PERSON_SALT, 1).normal())


def ridge_count(name, pattern, radial, rng, person=0.0):
    """A ridge count drawn for a loop, or a whorl's larger, by finger (`LOOP_COUNT`), `person` SDs of it the
    person's (`person_count`)."""
    mean, sd = LOOP_COUNT[name]
    if pattern in ("left_loop", "right_loop"):
        mean -= RADIAL_LESS if radial else 0.0
    else:
        mean, sd = mean + WHORL_MORE, WHORL_SD
    own = rng.normal(0, np.sqrt(1 - COUNT_PERSON**2))
    return float(np.clip(mean + sd * (COUNT_PERSON * person + own), *COUNT_RANGE))


def delta_angle(rng):
    mean, sd, low, high = DELTA_ANGLE
    return np.deg2rad(np.clip(rng.normal(mean, sd), low, high))


def singular_points_drawn(pattern, name, radial, period, rng, person=0.0):
    """Model 2's cores and deltas: the delta of a loop where its ridge count (`ridge_count`) puts it, in a direction
    drawn about model 1's (`DELTA_ANGLE`), and the core anywhere about the middle of the pad; a whorl's deltas as
    model 1 places them (`singular_points`) but drawn out or in for its larger count, each turned a little."""
    if pattern in ("left_loop", "right_loop"):
        cx = MASTER_W * (0.5 + rng.normal(0, 0.05))
        cy = MASTER_H * (0.40 + rng.normal(0, 0.04))
        side = -1 if pattern == "right_loop" else 1
        reach = ridge_count(name, pattern, radial, rng, person) * period * LOOP_REACH
        angle = delta_angle(rng)
        # The core sits 0.15 of the delta's sideways offset the other way (as model 1), so the reach is to it.
        dx = side * reach * np.sin(angle) / 1.15
        dy = reach * np.cos(angle)
        return [(cx - 0.15 * dx, cy)], [(cx + dx, cy + dy)]
    cores, deltas = singular_points(pattern, rng)
    shift = (MASTER_W * rng.normal(0, 0.04), MASTER_H * rng.normal(0, 0.03))
    cores = [(x + shift[0], y + shift[1]) for x, y in cores]
    deltas = [(x + shift[0], y + shift[1]) for x, y in deltas]
    if pattern not in WHORLS or not deltas:
        return cores, deltas

    def nearest(d):
        return min(cores, key=lambda c: np.hypot(c[0] - d[0], c[1] - d[1]))

    farthest = max(np.hypot(d[0] - nearest(d)[0], d[1] - nearest(d)[1]) for d in deltas)
    factor = ridge_count(name, pattern, False, rng, person) * period * WHORL_REACH / farthest
    moved = []
    for d in deltas:
        (x0, y0), turn = nearest(d), np.deg2rad(rng.normal(0, 10))
        vx, vy = (d[0] - x0) * factor, (d[1] - y0) * factor
        moved.append((x0 + vx * np.cos(turn) - vy * np.sin(turn), y0 + vx * np.sin(turn) + vy * np.cos(turn)))
    return cores, moved


def arch_orientation(rng):
    """Arch: ridges run across the finger and rise into a smooth hump."""
    y, x = np.mgrid[0:MASTER_H, 0:MASTER_W].astype(np.float32)
    height = rng.uniform(1.6, 2.6)
    centre = MASTER_W * (0.5 + rng.normal(0, 0.04))
    width = MASTER_W * rng.uniform(0.22, 0.32)
    # Ridge curves y = y0 - A(y0) * exp(-(x - c)^2 / 2w^2); the hump flattens out
    # towards the tip and the crease.
    amplitude = height * np.clip(1 - np.abs(y / MASTER_H - 0.55) / 0.55, 0, 1)
    slope = amplitude * (x - centre) / width * np.exp(-((x - centre) ** 2) / (2 * width**2))
    return np.mod(np.arctan(slope), np.pi).astype(np.float32)


def twist(theta, cores, rng):
    """Turn the flow around a double loop's two cores, so each loop opens past the other's head into an S.

    Two cores alone give an elongated whorl: their loops face each other. A
    smooth rotation, strongest between the cores and gone a little way out,
    swings each loop's opening to its own side; the singular points stay where
    they are.
    """
    (x0, y0), (x1, y1) = cores
    y, x = np.mgrid[0:MASTER_H, 0:MASTER_W].astype(np.float32)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    radius = np.hypot(x1 - x0, y1 - y0) * rng.uniform(TWIST_RADIUS[0], TWIST_RADIUS[1])
    amount = np.sign(x1 - x0 or 1) * rng.uniform(TWIST_AMOUNT[0], TWIST_AMOUNT[1])
    turn = amount * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2))
    return np.mod(theta + turn, np.pi).astype(np.float32)


TWIST_RADIUS = (0.9, 1.2)
TWIST_AMOUNT = (0.5, 0.8)


def flatten_far_field(theta, cores, deltas):
    """Pull orientation towards horizontal near the tip and the crease.

    The zero-pole model alone leaves ridges near the edges of the fingertip at
    odd angles; real fingers have ridges wrapping over the tip and running
    across the finger near the flexion crease.
    """
    if not deltas:
        return theta
    y = np.arange(MASTER_H, dtype=np.float32)[:, None]
    lowest_delta = max(d[1] for d in deltas)
    top_core = min(c[1] for c in cores) if cores else lowest_delta
    below = np.clip((y - lowest_delta - 60) / 180, 0, 1)
    above = np.clip((top_core - 140 - y) / 200, 0, 1)
    weight = np.maximum(below, above) * np.ones((1, MASTER_W), np.float32)
    c = (1 - weight) * np.cos(2 * theta) + weight * 1.0
    s = (1 - weight) * np.sin(2 * theta)
    return np.mod(0.5 * np.arctan2(s, c), np.pi).astype(np.float32)


@dataclass
class Extended:
    """A master grown on past the fingertip and sides, for latents (`ridgegen.latent`).

    A rolled print shows nearly all of the master, but real latents often come from the very tip and sides,
    which rolled prints miss (NIST TN 2190). The master's own ridges are kept as they are and the ridges grow on
    from them, so every print of the finger is unchanged. `offset` is where the master's origin is on the
    extended canvas. `minutiae` are the master's (moved by `offset`, same ids) and those of the extension, ids
    `f02-e005`: no rolled or plain print shows them, so a latent of the tip alone has no mates in the tenprint.
    """

    master: Master
    ridges: np.ndarray = field(repr=False)
    offset: tuple
    minutiae: list = field(repr=False)


EXTEND_TOP, EXTEND_SIDE = 160, 60  # px at 500 ppi: 8 mm beyond the tip, 3 mm beyond each side
HELD_INSIDE = 16  # px: the master's edge band that grows again; prints' contact starts about 12 px further in
EXTENDED_SALT = 3


@lru_cache(maxsize=24)
def extended(seed, finger, model=MODEL):
    m = master(seed, finger, model)
    top, side = EXTEND_TOP, EXTEND_SIDE
    shape = (MASTER_H + top, MASTER_W + 2 * side)
    # The master's ridges are held, but for a band at its edge, where growing it left ridges meeting their own
    # reflection; no print shows that band (`fingerprints.soft_inside`), and it grows again from the inside.
    band = HELD_INSIDE
    held = np.zeros(shape, bool)
    held[top + band : top + MASTER_H - band, side + band : side + MASTER_W - band] = True
    values = np.zeros(shape, np.float32)
    values[top:, side : side + MASTER_W] = m.ridges

    # Ridges beyond the master follow its flow at its edge, and above the tip turn into arcs round it.
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    arcs = np.arctan2(y - (top + MASTER_H * 0.55), x - (side + MASTER_W / 2)) + np.pi / 2
    edge = cv2.copyMakeBorder(m.orientation, top, 0, side, side, cv2.BORDER_REPLICATE)
    weight = np.exp(-np.clip(top - y, 0, None) / 80)
    c = weight * np.cos(2 * edge) + (1 - weight) * np.cos(2 * arcs)
    s = weight * np.sin(2 * edge) + (1 - weight) * np.sin(2 * arcs)
    theta = smooth_orientation(np.mod(0.5 * np.arctan2(s, c), np.pi), sigma=6)
    ridges = grow_ridges(theta, m.period, rng_for(seed, finger, EXTENDED_SALT), iterations=14, fixed=(held, values))

    # Its minutiae, from the extended ridges: those that are the master's keep its ids. Near the band some of the
    # master's own aren't there any more (they were the reflection's), and the rest are new.
    found = minutiae.extract(ridges > 0, np.ones(shape, bool))
    found = [p for p in found if p["y"] < top + MASTER_H - 18]
    ids = minutiae.identify(found, m.minutiae, lambda xy: xy - (side, top), radius=2, max_angle=20)
    own = [{"id": mid, **p} for p, mid in zip(found, ids) if mid]
    new = sorted((p for p, mid in zip(found, ids) if not mid), key=lambda p: (p["y"], p["x"]))
    extra = [{"id": f"f{finger:02d}-e{i:03d}", **p} for i, p in enumerate(new)]
    return Extended(master=m, ridges=ridges, offset=(side, top), minutiae=own + extra)


@lru_cache(maxsize=48)
def master(seed, finger, model=MODEL):
    """The master of `finger` (FGP 1-10) of the person `seed`, as master model `model` draws it (`MODEL`)."""
    rng = rng_for(seed, finger, MASTER_SALT)
    name = name_of(finger)
    if model == 1:
        pattern = choose_pattern(finger, rng)
        if pattern == "whorl":
            pattern = whorl_type(rng_for(seed, finger, WHORL_SALT))
        cores, deltas = singular_points(pattern, rng)
        wobble_amplitude = 0.12
    else:
        pattern, radial = choose_pattern_by_hand(finger, rng)
        scale = FINGER_SCALE[name] * rng.uniform(0.95, 1.05)
        period = ridge_period(seed, finger, scale, rng)
        cores, deltas = singular_points_drawn(pattern, name, radial, period, rng, person_count(seed))
        wobble_amplitude = rng.uniform(0.08, 0.2)

    if pattern == "arch":
        theta = arch_orientation(rng)
    else:
        theta = zero_pole_orientation((MASTER_H, MASTER_W), cores, deltas)
        theta = flatten_far_field(theta, cores, deltas)
        if pattern == "double_loop":
            theta = twist(theta, cores, rng)

    # Natural wobble in the flow, then smooth away sharp kinks (but keep the
    # singular points, which need a small smoothing radius).
    wobble = smooth_noise(theta.shape, rng, scale=180, amplitude=wobble_amplitude)
    theta = smooth_orientation(np.mod(theta + wobble, np.pi), sigma=6)

    if model == 1:
        scale = FINGER_SCALE[name] * rng.uniform(0.95, 1.05)
        period = rng.uniform(8.8, 10.2) * (0.94 + 0.06 * scale)
    period_map = period * (1 + smooth_noise(theta.shape, rng, scale=300, amplitude=0.05))

    ridges = grow_ridges(theta, period_map, rng, iterations=14)

    return Master(
        finger=finger,
        hand=hand_of(finger),
        name=name,
        pattern=pattern,
        cores=[(float(x), float(y)) for x, y in cores],
        deltas=[(float(x), float(y)) for x, y in deltas],
        period=float(period),
        scale=float(scale),
        ridges=ridges,
        orientation=theta,
        minutiae=minutiae.master(finger, ridges),
    )


# Grown ridges, kept per device (`synthesis.use_device`).
synthesis.keep_per_device(master, extended)
