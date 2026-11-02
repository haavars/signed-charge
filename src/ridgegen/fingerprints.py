"""Rolled and plain finger impressions, and four-finger/thumb slaps.

Output sizes follow the ANSI/NIST-ITL / FBI EBTS maximum capture areas at 500 ppi:
rolled finger 1.6 x 1.5 in (800 x 750), four-finger and two-thumb slaps
3.2 x 3.0 in (1600 x 1500).
"""

from functools import lru_cache

import cv2
import numpy as np

from . import finger as fingers
from . import impression as imp
from . import minutiae, synthesis
from .synthesis import grow_ridges, rng_for, smooth_noise

ROLLED_SIZE = (800, 750)  # width, height
SLAP_SIZE = (1600, 1500)
TIP_SIZE = (800, 500)

ROLLED, PLAIN, TIP = 1, 2, 3  # capture-kind salts

MM = 500 / 25.4  # master pixels per mm of skin
# The skin a plain fingertip presses, mm², for a finger of an index's size (`finger.FINGER_SCALE`, others by the
# square of theirs): a firm press, 10-20 N, by Dzidek et al. 2017's fit for an index finger at 30° (J R Soc Interface
# 14: 20160935, Table 1, A = 91.9 W^0.37; measured to 2 N, 119 mm² there).
PRESSED_MM2 = (200, 280)
# How far short of the distal crease a plain fingertip's contact stops, mm in the print: real flat prints have the
# core at 0.46-0.47 of the contact's length from its top (FVC2000/2002/2004), 0.23's
# slaps 0.40, reaching as far above the core but 1.5-2 mm further below it. Since ridgegen/0.24 (1.5 mm); before, it
# reached the crease. 2 mm since 0.25: model 2's fingers (`finger.MODEL`) put the core at 0.43 at 1.5 mm.
CONTACT_LIFT_MM = 2.0
# The area of an ellipse cut 0.85 of its semi-axis below its centre, over a times b: π/2 + asin(0.85) + 0.85 √(1 − 0.85²).
TIP_SEGMENT = float(np.pi / 2 + np.arcsin(0.85) + 0.85 * np.sqrt(1 - 0.85**2))

# A thumb's flat print is broader than its master's size alone gives (`finger.FINGER_SCALE`, 1.12): about 1.5 times
# an index's area, and rounder. By eye, since ridgegen/0.23; SD 302's thumbs to check.
THUMB_PRESS = 1.2

# Slap layout, as printed: a right hand's index finger is leftmost. (finger, how far its fingertip is below the
# middle finger's in px, splay angle). Since ridgegen/0.23, by eye against NIST's livescan slaps (NISTIR 7209,
# SlapSeg04, figures 1-4): the fingers held together and nearly parallel, the fingertips almost touching, ring
# 5 mm and index 8 mm below the middle finger, the little finger 20 mm. Before, each finger had its own place in the
# frame whatever its size, so 0.22's smaller fingertips stood 4-5 mm apart, fanned out.
RIGHT_FOUR = [(2, 160, -4), (3, 0, -1), (4, 100, 1.5), (5, 400, 5)]
LEFT_FOUR = [(10, 400, -5), (9, 100, -1.5), (8, 0, 1), (7, 160, 4)]
THUMBS = [(6, 100, 12), (1, 100, -12)]
# The paper between neighbouring fingertips, px: four fingers 0.5-3 mm, two thumbs 3-15 mm.
SLAP_GAP = {13: (10, 60), 14: (10, 60), 15: (60, 300)}


def rolled(seed, finger, capture=0):
    """Rolled impression of `finger` (FGP 1-10), rendered procedurally. Returns (image, meta)."""
    c = rolled_capture(seed, finger, capture)
    return imp.render_capture(c), c.meta


def rolled_capture(seed, finger, capture=0):
    """The geometry and ground truth of a rolled impression, as an `impression.Capture`."""
    master = fingers.master(seed, finger)
    key = (seed, finger, ROLLED, capture, 7)
    rng = rng_for(*key)
    w, h = ROLLED_SIZE

    # A finger's size in its prints is its own; captures differ by how hard it was pressed (since ridgegen/0.20,
    # before each capture drew its scale from 0.86-0.94 alone, so two of a finger differed by 7 % at p95).
    scale = finger_scale(seed, finger) * rng.uniform(0.975, 1.025) * min(1.0, 1.02 / master.scale)
    centre_dst = (w / 2 + rng.normal(0, 18), h * 0.54 + rng.normal(0, 14))
    map_x, map_y = imp.warp_maps(
        (h, w), (fingers.MASTER_W / 2, fingers.MASTER_H * 0.47), centre_dst,
        angle_deg=rng.normal(0, 4), scale=scale, distortion=rng.uniform(3, 7), rng=rng,
    )
    ridges = imp.remap(master.ridges, map_x, map_y)
    inside = soft_inside(master, map_x, map_y)

    # Nail-to-nail contact: a tall rounded shape, lighter towards the sides
    # where the finger was rolled on and off.
    rx = w * rng.uniform(0.40, 0.46) * min(master.scale, 1.05)
    ry = h * rng.uniform(0.50, 0.55)
    # How far up the finger the contact reaches, as it was placed and tilted (since ridgegen/0.20; before, two
    # captures' reach towards the tip differed mostly by their scales).
    lift = float(np.clip(rng_for(*key, 1).normal(0, 15), -30, 30))
    contact = imp.ellipse_mask((h, w), (centre_dst[0], centre_dst[1] + ry * 0.2 + lift), (rx, ry),
                               angle_deg=rng.normal(0, 3), bottom=h - rng.uniform(4, 30))
    xs = (np.arange(w, dtype=np.float32) - centre_dst[0]) / rx
    contact *= np.clip(1.15 - 0.45 * xs**2, 0.35, 1)[None, :]
    contact = imp.irregular_edge(contact, rng) * inside

    creases = int(rng.integers(0, 3))
    return imp.Capture(ridges, contact, meta(master, ridges, contact, map_x, map_y, "rolled", capture),
                       rng, key, creases)


def tip(seed, finger, capture=0):
    """Rolled tip impression of `finger` (FGP 1-10), rendered procedurally. Returns (image, meta)."""
    c = tip_capture(seed, finger, capture)
    return imp.render_capture(c), c.meta


def tip_capture(seed, finger, capture=0):
    """A rolled tip (EBTS Appendix P's "Rolled Tips", ANSI/NIST-ITL's TIP): the end joint rolled from one side
    across the tip to the other with the nail on the paper, as on an FD-884a major case card, not the tenprint.

    Drawn from the finger's extended master (`finger.extended`, as latents are), so it shows the extreme tip that
    rolled prints don't, from about 7 mm beyond the master's top down over the upper 10-13 mm of the pad, where it
    overlaps the rolled print. Its own placement, distortion and contact; minutiae ids are the extended master's
    (`f02-m017` on the pad, `f02-e005` beyond it), so a tip latent's mates in it are the ids they share.
    """
    ext = fingers.extended(seed, finger)
    master = ext.master
    key = (seed, finger, TIP, capture, 7)
    rng = rng_for(*key)
    w, h = TIP_SIZE
    side, top = ext.offset

    scale = finger_scale(seed, finger) * rng.uniform(0.975, 1.025) * min(1.0, 1.02 / master.scale)
    # The roll over the end presses the tip's curve flat: the skin is a little compressed along the finger.
    stretch = (rng.uniform(0.97, 1.03), rng.uniform(0.92, 1.0))
    centre_src = (side + fingers.MASTER_W / 2 + rng.normal(0, 10), top + 40 + rng.normal(0, 12))
    centre_dst = (w / 2 + rng.normal(0, 14), h * 0.5 + rng.normal(0, 10))
    map_x, map_y = imp.warp_maps(
        (h, w), centre_src, centre_dst,
        angle_deg=rng.normal(0, 4), scale=scale, distortion=rng.uniform(3, 7), rng=rng, stretch=stretch,
    )
    ridges = imp.remap(ext.ridges, map_x, map_y)
    inside = imp.remap(np.ones_like(ext.ridges), map_x, map_y)
    inside = cv2.GaussianBlur(cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=8), (0, 0), 6)

    # A wide rounded band: the end of the finger at its top, the pad lifting off the paper at its ragged bottom,
    # lighter at the sides where the roll began and ended.
    rx = w * rng.uniform(0.38, 0.45) * min(master.scale, 1.05)
    ry = h * rng.uniform(0.31, 0.36)
    cy = centre_dst[1] + rng.normal(0, 8)
    contact = imp.ellipse_mask((h, w), (centre_dst[0], cy), (rx, ry), angle_deg=rng.normal(0, 3))
    xs = (np.arange(w, dtype=np.float32) - centre_dst[0]) / rx
    contact *= np.clip(1.15 - 0.45 * xs**2, 0.35, 1)[None, :]
    contact = imp.irregular_edge(contact, rng) * inside

    known = ext.minutiae
    return imp.Capture(ridges, contact, meta(master, ridges, contact, map_x, map_y, "tip", capture, known, ext.offset),
                       rng, key, 0)


def finger_scale(seed, finger):
    """A finger's scale in its rolled prints, before each capture's own (`rolled_capture`); `finger.finger_scale`."""
    return fingers.finger_scale(seed, finger)


def plain_finger(seed, finger, capture, rng, canvas=(560, 1300)):
    """A single flat (plain) impression on its own canvas, for slap composition.

    Since ridgegen/0.22 it's drawn at the finger's own scale, as its rolled prints are (`finger_scale`, each capture
    within 2.5 %; before, U(0.97, 1.03), 11 % larger than its rolled prints and thumbs 19 %), with less distortion
    than a rolled print (2-5 px, was 4-9). The fingertip presses `PRESSED_MM2` of skin, sized by the finger and drawn
    per capture as pressure varies, from just above the distal crease up (`CONTACT_LIFT_MM`, 0.24), a little to one
    side or the other by capture (before, the whole pad from crease to tip, an index finger 351 mm², and the same
    skin in every capture). A thumb's presses `THUMB_PRESS` times more (0.23). Below the crease, the middle phalanx
    and most often the proximal one (`press_phalanx`).

    Returns (master, ridges, contact, map_x, map_y, pasted): `pasted` is where the
    ridges are a phalanx's, pasted below the crease, not the master's.
    """
    master = fingers.master(seed, finger)
    w, h = canvas
    scale = finger_scale(seed, finger) * rng.uniform(0.975, 1.025) * min(1.0, 1.02 / master.scale)
    # The finger, on its master: its half width, its distal phalanx's length, where the distal crease is.
    rx_m = rng.uniform(151, 174) * master.scale
    ry_m = rng.uniform(265, 295) * master.scale
    crease_m = fingers.MASTER_H * 0.44 + 0.85 * ry_m
    # The fingertip's contact: an ellipse cut at the crease, its area set in skin.
    thumb = finger in (1, 6)
    area = rng.uniform(*PRESSED_MM2) * master.scale**2 * MM**2 * (THUMB_PRESS if thumb else 1)
    ratio = rng.uniform(*(1.45, 1.65) if thumb else (1.6, 1.85))
    a_m = np.sqrt(area / (TIP_SEGMENT * ratio))
    b_m = a_m * ratio
    side = float(np.clip(rng.normal(0, 18), -40, 40))
    centre_dst = (w / 2, 330)
    lift = CONTACT_LIFT_MM * MM  # canvas px, as the print's
    map_x, map_y = imp.warp_maps(
        (h, w), (fingers.MASTER_W / 2 + side, crease_m - lift / scale - 0.85 * b_m), centre_dst,
        angle_deg=rng.normal(0, 3), scale=scale, distortion=rng.uniform(2, 5), rng=rng,
    )
    ridges = imp.remap(master.ridges, map_x, map_y)
    inside = soft_inside(master, map_x, map_y)
    rx, ry = rx_m * scale, ry_m * scale  # the finger, on the canvas
    cx = centre_dst[0] - side * scale  # its axis
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # The distal crease: a gentle curve across the finger (dipping or rising a
    # little in the middle), not a straight cut.
    crease = centre_dst[1] + 0.85 * b_m * scale + lift
    bow = rng.uniform(-0.04, 0.1) * ry
    crease_line = crease + bow * (1 - np.clip(((xx - cx) / rx) ** 2, 0, 1))
    tip = imp.ellipse_mask((h, w), centre_dst, (a_m * scale, b_m * scale))
    contact = imp.irregular_edge(tip * np.clip((crease_line - lift - yy) / 6, 0, 1), rng) * inside
    pasted = np.zeros((h, w), bool)

    # Below the crease, the middle phalanx and on most fingers the start of the proximal one (a thumb's second
    # phalanx is its last): narrower than the fingertip's contact, less firmly pressed the further from the tip.
    # Since ridgegen/0.23 their width is the fingertip contact's, not the finger's, and the proximal phalanx
    # is new, by eye against NIST's livescan slaps (NISTIR 7209): 0.22's smaller fingertips left the middle
    # phalanx almost as large as the tip, and the lower third of the frame empty.
    half = a_m * scale * rng.uniform(*(0.85, 0.95) if thumb else (0.78, 0.9))
    # The paper between the fingertip and the middle phalanx: the crease, and the contact stopping short of it.
    gap = rng.uniform(4, 14)
    middle = ry * rng.uniform(*(1.1, 1.5) if thumb else (0.9, 1.15))
    length = min(h - crease - gap - 30, middle)
    if length > 60:
        contact = press_phalanx(seed, finger, 0, rng, ridges, contact, pasted, (cx, crease_line + gap, crease + gap),
                                length, half, rng.uniform(0.7, 0.95))
    # The proximal interphalangeal crease, just below the middle phalanx's print.
    proximal = crease + gap + middle + rng.uniform(15, 35)
    length = min(h - proximal - 30, ry * rng.uniform(0.8, 1.4))
    if not thumb and rng.random() < 0.85 and length > 60:
        contact = press_phalanx(seed, finger, 1, rng, ridges, contact, pasted,
                                (cx, crease_line - crease + proximal, proximal), length,
                                half * rng.uniform(1.0, 1.08), rng.uniform(0.45, 0.75))
    return master, ridges, contact, map_x, map_y, pasted


def press_phalanx(seed, finger, part, rng, ridges, contact, pasted, top, length, half, firmness):
    """A phalanx below a crease, pressed into `ridges` and `pasted` in place; returns `contact` with it.

    Ridges run across the finger, on a shape between an ellipse and a rounded rectangle, tapering and fading out
    downwards on a rounded, ragged edge; the corners at the crease round off less than the bottom ones. `top` is
    (the finger's axis, the crease line above it per pixel, its height at the axis); `part` 0 is the middle
    phalanx, 1 the proximal.
    """
    h, w = contact.shape
    cx, line, top_y = top
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    half = half * (1 - rng.uniform(0.04, 0.12) * np.clip((yy - line) / length, 0, 1))
    u = np.abs(xx - cx) / half
    v = (yy - line) / length  # 0 at the crease, 1 at the bottom
    p = rng.uniform(2.0, 2.6)
    w2 = np.abs(2 * v - 1) ** p * np.where(v < 0.5, 0.35, 1.0)
    shape = 1 - (u**p + w2)
    mask = np.clip(shape * 4, 0, 1) * np.clip((yy - line) / 5, 0, 1) * (v <= 1)
    pressure = 1 - rng.uniform(0.35, 0.6) * np.clip(v, 0, 1) ** 1.5
    seg_contact = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 9) * pressure
    seg_contact = imp.irregular_edge(seg_contact, rng) * firmness
    # Ridges pasted whole, larger than the contact area, so its edges aren't the paste's.
    seg = phalanx(seed, finger, part)
    y0 = int(np.clip(top_y + length / 2 - seg.shape[0] / 2, 0, h - seg.shape[0]))
    x0 = int(np.clip(cx - seg.shape[1] / 2, 0, w - seg.shape[1]))
    below = (seg_contact > 0.01)[y0 : y0 + seg.shape[0], x0 : x0 + seg.shape[1]]
    region = ridges[y0 : y0 + seg.shape[0], x0 : x0 + seg.shape[1]]
    region[below] = seg[below]
    pasted[y0 : y0 + seg.shape[0], x0 : x0 + seg.shape[1]] |= below
    return np.maximum(contact, seg_contact * inside_canvas(seg_contact, y0, x0, seg.shape))


def inside_canvas(mask, y0, x0, shape):
    """`mask` where the pasted phalanx ridges cover it, else 0: no contact without ridges."""
    covered = np.zeros(mask.shape, np.float32)
    covered[y0 : y0 + shape[0], x0 : x0 + shape[1]] = 1
    return covered


@lru_cache(maxsize=48)
def phalanx(seed, finger, part=0):
    """Ridge pattern for a phalanx below the fingertip, 0 the middle, 1 the proximal: gently arched ridges across
    the finger."""
    rng = rng_for(seed, finger, 5) if part == 0 else rng_for(seed, finger, 5, part)
    h, w = 420, 520
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    slope = rng.uniform(0.08, 0.2) * np.sin(np.pi * (x / w - 0.5)) + rng.normal(0, 0.05)
    theta = np.mod(np.arctan(slope) + smooth_noise((h, w), rng, 120, 0.1), np.pi)
    return grow_ridges(theta.astype(np.float32), rng.uniform(8.8, 10.0), rng, iterations=12)


def slap(seed, code, capture=0, missing=None):
    """Plain slap: 13 right four fingers, 14 left four fingers, 15 two thumbs.

    Rendered procedurally. Returns (image, meta).
    """
    c = slap_capture(seed, code, capture, missing)
    return imp.render_capture(c), c.meta


def slap_capture(seed, code, capture=0, missing=None):
    """The geometry and ground truth of a slap, as an `impression.Capture`.

    `missing` maps fingers (FGP) that leave no print to why, as ANSI/NIST-ITL's
    AMP codes: "XX" amputated, "UP" unable to print (bandaged, say). They're
    left out, and the others stay where they'd be.
    """
    missing = missing or {}
    layout = {13: RIGHT_FOUR, 14: LEFT_FOUR, 15: THUMBS}[code]
    key = (seed, code, PLAIN, capture, 11)
    rng = rng_for(*key)
    w, h = SLAP_SIZE
    ridges_all = np.zeros((h, w), np.float32)
    contact_all = np.zeros((h, w), np.float32)
    tilt = float(np.clip(rng.normal(0, 6), -15, 15))
    fingers_meta = []
    placed = []
    pieces = []
    geometry = {}
    phalanges, angles = {}, {}

    # Every finger is drawn, a missing one too, so the others stay where they'd be.
    drawn = []
    for finger, *_ in layout:
        finger_rng = rng_for(seed, finger, PLAIN, capture, code)
        drawn.append((finger_rng, plain_finger(seed, finger, capture, finger_rng)))
    centres = hand(layout, [d[1][2] for d in drawn], rng, SLAP_GAP[code], tilt, w)

    for (finger, _below, splay), (finger_rng, drawing), (x, y) in zip(layout, drawn, centres):
        if finger in missing:
            continue
        master, ridges, contact, map_x, map_y, pasted = drawing
        fh, fw = ridges.shape
        angle = splay + tilt + finger_rng.normal(0, 2.5)
        cx = x + finger_rng.normal(0, 8)
        cy = y + finger_rng.normal(0, 25)
        # Rotate the finger canvas about its fingertip centre and place it.
        m = cv2.getRotationMatrix2D((fw / 2, 330), -angle, 1.0)
        m[:, 2] += (cx - fw / 2, cy - 330)
        placed_ridges = cv2.warpAffine(ridges, m, (w, h), flags=cv2.INTER_LINEAR)
        placed_contact = cv2.warpAffine(contact, m, (w, h), flags=cv2.INTER_LINEAR)
        phalanges[finger] = cv2.warpAffine(pasted.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR) > 0.5
        angles[finger] = angle
        pieces.append((finger, placed_ridges, placed_contact))
        geometry[finger] = (master, cv2.invertAffineTransform(m), map_x, map_y, pasted)

        # Singular points: master -> finger canvas -> slap.
        cores = imp.transform_points(master.cores, map_x, map_y)
        deltas = imp.transform_points(master.deltas, map_x, map_y)
        fingers_meta.append({
            "fgp": finger,
            "pattern": master.pattern,
            "cores": [apply_affine(m, p) for p in cores],
            "deltas": [apply_affine(m, p) for p in deltas],
        })

    for i, (finger, placed_ridges, placed_contact) in enumerate(separate(pieces)):
        # Each finger's box in the slap, [left, top, right, bottom], for segment positions (ANSI/NIST 14.021 SEG).
        ys, xs = np.nonzero(placed_contact > 0.5)
        if len(xs):
            fingers_meta[i]["box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        fingers_meta[i].update(fingertip(placed_contact > 0.5, phalanges[finger], angles[finger]))
        placed.append((finger, placed_contact))
        take = placed_contact > contact_all
        ridges_all = np.where(take, placed_ridges, ridges_all)
        contact_all = np.maximum(contact_all, placed_contact)

    creases = int(rng.integers(0, 4))
    points = slap_minutiae(minutiae.extract(ridges_all > 0, contact_all > 0.5), placed, geometry)
    meta = {
        "capture": capture,
        "impression": "plain",
        "fingers": fingers_meta,
        "missing": {str(f): missing[f] for f, *_ in layout if f in missing},
        "minutiae_count": len(points),
        "minutiae": points,
    }
    return imp.Capture(ridges_all, contact_all, meta, rng, key, creases, fingers=placed)


def fingertip(pressed, phalanges, angle):
    """A slap finger's fingertip, as a segmenter cuts it (NIST's `nfseg` and `slapcrop`; SD 300 segments "the
    distal joints"): its box in the slap, [left, top, right, bottom]; its `angle`, degrees the finger is turned
    clockwise from upright, so turning the cut counter-clockwise by it stands it up; and `tip_polygon`, the
    rectangle around the fingertip along the finger, four corners from top left, clockwise. `box` is the whole
    finger's contact, with its phalanges (since ridgegen/0.23.1)."""
    ys, xs = np.nonzero(pressed & ~phalanges)
    out = {"angle": round(float(angle), 1)}
    if len(xs) < 50:
        return out
    a = np.deg2rad(angle)
    cx, cy = xs.mean(), ys.mean()
    # Along the finger: x across it, y down it.
    u = (xs - cx) * np.cos(a) + (ys - cy) * np.sin(a)
    v = -(xs - cx) * np.sin(a) + (ys - cy) * np.cos(a)
    corners = [(u.min(), v.min()), (u.max(), v.min()), (u.max(), v.max()), (u.min(), v.max())]
    out["tip_box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    out["tip_polygon"] = [[round(float(cx + p * np.cos(a) - q * np.sin(a)), 1),
                           round(float(cy + p * np.sin(a) + q * np.cos(a)), 1)] for p, q in corners]
    return out


def hand(layout, contacts, rng, gap, tilt, width):
    """Where each fingertip's centre goes in a slap: side by side by their widths across the tip (each finger's
    canvas has it at row 330), `gap` px of paper between them, the middle finger's 290 px down and the others
    `layout`'s distance below it, centred in the frame give or take 2 mm; then the hand turned by `tilt` as a
    whole, about the fingertips' middle."""
    halves = [np.count_nonzero(c[330] > 0.5) / 2 for c in contacts]
    xs = [0.0]
    for left, right in zip(halves, halves[1:]):
        xs.append(xs[-1] + left + rng.uniform(*gap) + right)
    xs = np.array(xs) + width / 2 - (xs[-1] + halves[-1] - halves[0]) / 2 + rng.normal(0, 40)
    ys = 290 + np.array([below for _finger, below, _splay in layout], float) + rng.normal(0, 30)
    a = np.deg2rad(tilt)
    dx, dy = xs - xs.mean(), ys - ys.mean()
    return list(zip(xs.mean() + dx * np.cos(a) - dy * np.sin(a), ys.mean() + dx * np.sin(a) + dy * np.cos(a)))


def slap_minutiae(points, placed, geometry):
    """A slap's minutiae with the finger each is on (`fgp`, the one pressing hardest there) and its master id:
    back through the finger's placement on the slap, then its warp maps. Those on the pasted middle phalanx
    have no id."""
    if not points:
        return []
    contacts = np.stack([contact for _finger, contact in placed])
    owners = [placed[int(np.argmax(contacts[:, p["y"], p["x"]]))][0] for p in points]
    ids = {}
    for finger in set(owners):
        master, inverse, map_x, map_y, pasted = geometry[finger]
        mine = [i for i, owner in enumerate(owners) if owner == finger]
        def on_canvas(xy, inverse=inverse):
            return xy @ inverse[:, :2].T + inverse[:, 2]

        found = minutiae.identify([points[i] for i in mine], master.minutiae,
                                  lambda xy: imp.to_master(map_x, map_y, on_canvas(xy)), radius=master.period / 2)
        canvas = on_canvas(np.array([[points[i]["x"], points[i]["y"]] for i in mine], float))
        h, w = pasted.shape
        for i, mid, (cx, cy) in zip(mine, found, canvas.round().astype(int)):
            ids[i] = None if 0 <= cy < h and 0 <= cx < w and pasted[cy, cx] else mid
    return [{"id": ids[i], "fgp": owner, **p} for i, (p, owner) in enumerate(zip(points, owners))]


def separate(pieces, gap=(0.6, 0.8)):
    """Fingers pressed side by side don't overlap: where their contact areas meet, each pixel goes to the finger
    with the larger share of the contact there, and where neither has a clear share (`gap`) it's paper, the thin
    white line between neighbouring fingers in a real slap. Without it the fingers' soft edges overlapped into a
    grey haze, and the ridges switched from one finger to the next along a hard seam.

    `pieces` is [(finger, ridges, contact)]; returns them with their contact reduced.
    """
    if not pieces:
        return []
    contacts = np.stack([c for _f, _r, c in pieces]) ** 2
    total = contacts.sum(axis=0) + 1e-6
    out = []
    for i, (finger, ridges, contact) in enumerate(pieces):
        share = contacts[i] / total
        t = np.clip((share - gap[0]) / (gap[1] - gap[0]), 0, 1)
        out.append((finger, ridges, contact * t * t * (3 - 2 * t)))
    return out


def soft_inside(master, map_x, map_y):
    """Where the impression has master pattern, faded out towards the master's edge.

    If the contact area reaches past the master, the print fades (like light
    pressure at the fingertip) instead of ending in a straight cut.
    """
    inside = imp.remap(np.ones_like(master.ridges), map_x, map_y)
    return cv2.GaussianBlur(cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=12), (0, 0), 10)


def apply_affine(m, point):
    x, y = point
    return [round(float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), 1), round(float(m[1, 0] * x + m[1, 1] * y + m[1, 2]), 1)]


def meta(master, ridges, contact, map_x, map_y, kind, capture, known=None, offset=(0, 0)):
    """Ground truth of an impression drawn through `map_x`/`map_y` from the master, or from a canvas around it
    with the master's origin at `offset` and minutiae `known` (an extended master's, for a tip)."""
    points = minutiae.extract(ridges > 0, contact > 0.5)
    ids = minutiae.identify(points, master.minutiae if known is None else known,
                            lambda xy: imp.to_master(map_x, map_y, xy), radius=master.period / 2)
    points = [{"id": mid, **p} for p, mid in zip(points, ids)]
    return {
        "fgp": master.finger,
        "capture": capture,
        "impression": kind,
        "pattern": master.pattern,
        "ridge_period_px": round(master.period, 2),
        "cores": [list(p) for p in imp.transform_points(shifted(master.cores, offset), map_x, map_y)],
        "deltas": [list(p) for p in imp.transform_points(shifted(master.deltas, offset), map_x, map_y)],
        "minutiae_count": len(points),
        "minutiae": points,
    }


def shifted(points, offset):
    return points if offset == (0, 0) else [(x + offset[0], y + offset[1]) for x, y in points]


# Grown ridges, kept per device (`synthesis.use_device`).
synthesis.keep_per_device(phalanx)
