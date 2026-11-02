"""Impressions: what a single capture of a master pattern looks like.

Every capture (rolled or plain, first or later) of the same finger or palm
starts from the same master, then gets its own placement, skin distortion,
contact area, pressure, and sensor noise. That is what makes two captures of one
finger a realistic mated pair rather than identical copies.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from skimage.morphology import skeletonize

from .synthesis import rng_for, smooth_noise

PAPER = 250  # background grey level
RIDGE_DARK = (25, 70)  # darkest ridge grey, sampled per capture
# Sensor noise, standard deviation in grey levels: what a certified scanner may add. FBI EBTS Appendix F asks for
# a signal-to-noise ratio of at least 125 on uniform targets, about 1.8 grey levels over our range. Until
# ridgegen/0.7 it was 4-10, paper grain more than a scanner's.
NOISE = (1.0, 1.8)
# Sweat pores: 88-220 µm across (as surveys of pore detection give them), evenly along the middle of the ridges
# at about 9-18 per cm of ridge (Locard's figure, via Ashbaugh), a few open to the side of the ridge.
PORE_DIAMETER_UM = (88, 220)
PORES_PER_CM = (9, 18)
OPEN_PORES = 0.1
UM_PER_PX = 25400 / 500  # at 500 ppi; divided by the capture's scale


@dataclass
class Capture:
    """One impression before rendering: its geometry and ground truth.

    This is the identity half of an image. Any renderer (procedural or
    diffusion) turns it into pixels, and verification checks the pixels against
    it, so the ground truth holds whatever the renderer does.
    """

    ridges: np.ndarray  # warped clean ridge field, > 0 on ridges
    contact: np.ndarray  # 0..1 contact area
    meta: dict  # ground truth: pattern, singular points, minutiae
    rng: np.random.Generator  # appearance randomness for the first attempt
    appearance_key: tuple  # derives appearance randomness for later attempts
    creases: int = 0
    # Slaps: (fgp, placed contact mask) per finger, for per-finger checks.
    fingers: list = field(default_factory=list)
    # Pixels per 500 ppi pixel: 2 for a 1000 ppi capture (`upscale`). The renderer scales its pixel sizes by it.
    scale: int = 1

    def appearance_rng(self, attempt):
        """Randomness for rendering attempt `attempt` (0, 1, ...).

        Attempt 0 continues the capture's own generator, so first attempts are
        what the procedural renderer has always produced.
        """
        return self.rng if attempt == 0 else rng_for(*self.appearance_key, 101, attempt)

    def ridge_map(self):
        """The clean binarised ridge map: black ridges on white, only in the contact area."""
        return np.where((self.ridges > 0) & (self.contact > 0.5), 0, 255).astype(np.uint8)


def upscale(capture, factor=2):
    """The same impression at `factor` times the resolution (1000 ppi for 2): its clean ridge field and contact
    area resampled, its ground truth scaled. The ridge field is a smooth signal (+ridge / -valley), not pixels, so
    resampling it gives smooth ridges at the new resolution; the renderer then draws ink, pores, edges and noise
    at that resolution. Geometry and ground truth are the 500 ppi capture's, so the two are the same impression.
    """
    h, w = capture.ridges.shape
    size = (w * factor, h * factor)
    return Capture(
        ridges=cv2.resize(capture.ridges, size, interpolation=cv2.INTER_CUBIC),
        contact=np.clip(cv2.resize(capture.contact, size, interpolation=cv2.INTER_LINEAR), 0, 1),
        meta=scale_meta(capture.meta, factor),
        rng=capture.rng,
        appearance_key=capture.appearance_key,
        creases=capture.creases,
        fingers=[(fgp, np.clip(cv2.resize(m, size, interpolation=cv2.INTER_LINEAR), 0, 1)) for fgp, m in capture.fingers],
        scale=capture.scale * factor,
    )


def scale_meta(meta, factor):
    """Ground truth in pixels `factor` times smaller: minutiae, cores and deltas (also per finger of a slap) and
    the ridge period. Angles don't change."""
    def points(ps):
        return [[round(x * factor, 1), round(y * factor, 1)] for x, y in ps]

    out = dict(meta)
    if "minutiae" in meta:
        out["minutiae"] = [{**m, "x": m["x"] * factor, "y": m["y"] * factor} for m in meta["minutiae"]]
    for key in ("cores", "deltas"):
        if key in meta:
            out[key] = points(meta[key])
    if "fingers" in meta:
        out["fingers"] = [
            {**f, "cores": points(f.get("cores", [])), "deltas": points(f.get("deltas", [])),
             **{k: [v * factor for v in f[k]] for k in ("box", "tip_box") if k in f},
             **({"tip_polygon": points(f["tip_polygon"])} if "tip_polygon" in f else {})}
            for f in meta["fingers"]
        ]
    if "ridge_period_px" in meta:
        out["ridge_period_px"] = round(meta["ridge_period_px"] * factor, 2)
    return out


def warp_maps(out_shape, centre_src, centre_dst, angle_deg, scale, distortion, rng, stretch=(1.0, 1.0)):
    """cv2.remap maps sending each output pixel to master coordinates.

    Rigid placement (rotation about the finger centre, translation) plus a
    smooth random displacement field for skin elasticity.
    """
    h, w = out_shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle_deg)
    du, dv = (u - centre_dst[0]) / (scale * stretch[0]), (v - centre_dst[1]) / (scale * stretch[1])
    x = centre_src[0] + du * np.cos(a) - dv * np.sin(a)
    y = centre_src[1] + du * np.sin(a) + dv * np.cos(a)
    if distortion > 0:
        x += smooth_noise(out_shape, rng, scale=160, amplitude=distortion)
        y += smooth_noise(out_shape, rng, scale=160, amplitude=distortion)
    return x.astype(np.float32), y.astype(np.float32)


def to_master(map_x, map_y, points):
    """Master coordinates of output pixel positions `points` ((n, 2), x and y), interpolated in the warp maps:
    exact, since the maps are what the impression was drawn through."""
    p = np.asarray(points, np.float32).reshape(1, -1, 2)
    x = cv2.remap(map_x, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    y = cv2.remap(map_y, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.stack([x[0], y[0]], 1).astype(float)


def remap(array, map_x, map_y, border=0.0):
    return cv2.remap(
        array, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


def irregular_edge(mask, rng, amount=12):
    """Roughen a soft mask's outline so contact areas don't look machine-cut."""
    wobble = smooth_noise(mask.shape, rng, scale=40, amplitude=amount / 60)
    return np.clip(mask + wobble * (mask > 0.02) * (mask < 0.98), 0, 1)


def ellipse_mask(shape, centre, axes, angle_deg=0.0, softness=9, bottom=None):
    """Soft elliptical contact area, optionally cut flat at y = `bottom`."""
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (int(centre[0]), int(centre[1])), (int(axes[0]), int(axes[1])),
                angle_deg, 0, 360, 255, -1)
    if bottom is not None:
        mask[int(bottom):, :] = 0
    return cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), softness)


def render_capture(capture, attempt=0):
    """The procedural renderer: a capture's grey image for rendering `attempt`."""
    rng = capture.appearance_rng(attempt)
    creases = capture.creases if attempt == 0 else int(rng.integers(0, 3))
    return render(capture.ridges, capture.contact, rng, creases=creases, scale=capture.scale)


def render(ridges, contact, rng, pressure=None, noise=None, creases=0, pores=True, scale=1):
    """Turn a warped ridge field (+ridge / -valley) and contact mask into grey pixels.

    `pressure` in [-1, 1]: negative is dry (thin, broken ridges), positive is
    wet or heavy (thick ridges that merge). `scale` is pixels per 500 ppi pixel
    (2 at 1000 ppi): sizes in pixels grow with it, so a print looks the same at
    either resolution, only finer.
    """
    shape = ridges.shape
    pressure = rng.uniform(-0.6, 0.6) if pressure is None else pressure
    noise = rng.uniform(*NOISE) if noise is None else noise

    # Ink/contact varies over the area: lighter towards the edges of the contact
    # and in random patches.
    # Patch size scales with the impression so palms get broad, smooth patches.
    patches = smooth_noise(shape, rng, scale=max(70 * scale, min(shape) / 10), amplitude=0.12)
    threshold = -0.35 * pressure - patches * 0.6
    # Dry prints lose bits of ridge.
    if pressure < -0.3:
        breaks = smooth_noise(shape, rng, scale=6 * scale, amplitude=1.0)
        threshold = threshold + np.where(breaks > 1.5 + pressure, 0.6, 0)

    ink = 1 / (1 + np.exp(-(ridges - threshold) * 7))
    ink *= np.clip(contact * (0.85 + patches), 0, 1)

    if pores:
        ink *= 1 - pore_mask(ridges, contact, rng, scale)

    for _ in range(creases):
        ink *= 1 - crease_line(shape, rng, width=rng.uniform(2, 5) * scale if scale != 1 else None)

    dark = rng.uniform(*RIDGE_DARK)
    grey = PAPER - (PAPER - dark) * ink
    grey = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), rng.uniform(0.5, 0.9) * scale)
    grey += rng.normal(0, noise, shape).astype(np.float32)
    return np.clip(grey, 0, 255).astype(np.uint8)


def pore_mask(ridges, contact, rng, scale=1):
    """Sweat pores as 0..1 (how much each lightens the ridge), placed along the ridges' centre lines.

    Each pore is a soft dot `PORE_DIAMETER_UM` across, lightening the ridge by 50-95 %. Most sit on the ridge's
    centre line, at `PORES_PER_CM` along it; `OPEN_PORES` of them sit at its edge, open to the valley. Pores
    closer than about half a ridge period to another are dropped, so they don't clump. Until ridgegen/0.7 pores
    were scattered at random, about a tenth as many.
    """
    shape = ridges.shape
    pressed = contact > 0.3
    centre = np.flatnonzero(skeletonize((ridges > 0.3) & pressed))
    edge = np.flatnonzero((ridges > 0) & (ridges < 0.25) & pressed)
    if len(centre) == 0:
        return np.zeros(shape, np.float32)
    px_per_cm = 10000 / UM_PER_PX * scale
    # The skeleton's pixel count is about its length; a quarter more are drawn for the thinning below.
    count = min(len(centre), int(len(centre) / px_per_cm * rng.uniform(*PORES_PER_CM) * 1.25))
    picks = rng.choice(centre, count, replace=False)
    opened = rng.random(count) < OPEN_PORES
    if len(edge):
        picks[opened] = rng.choice(edge, int(opened.sum()))

    # Thinning: a pore stays only if it has the highest priority within its radius.
    radius = int(round(4 * scale))
    priority = np.zeros(shape, np.float32)
    priority.flat[picks] = rng.uniform(0.01, 1, count)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    kept = (priority > 0) & (priority >= cv2.dilate(priority, kernel))

    ys, xs = np.nonzero(kept)
    diameters = rng.uniform(*PORE_DIAMETER_UM, len(ys)) / UM_PER_PX * scale
    depths = rng.uniform(0.5, 0.95, len(ys))
    mask = np.zeros(shape, np.float32)
    # Three sizes, each a blurred dot normalised to its depth at the centre.
    for size in np.split(np.argsort(diameters), [int(len(ys) / 3), int(2 * len(ys) / 3)]):
        if len(size) == 0:
            continue
        sigma = float(np.mean(diameters[size])) / 2.5
        dots = np.zeros(shape, np.float32)
        dots[ys[size], xs[size]] = depths[size]
        mask = np.maximum(mask, cv2.GaussianBlur(dots, (0, 0), sigma) * 2 * np.pi * sigma**2)
    return np.clip(mask, 0, 0.95)


def crease_line(shape, rng, width=None, length=None):
    """A soft light line (skin crease or scar) as a 0..1 mask."""
    h, w = shape
    width = width if width is not None else rng.uniform(2, 5)
    length = length if length is not None else rng.uniform(0.2, 0.6) * min(h, w)
    x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
    angle = rng.uniform(0, np.pi)
    x1, y1 = x0 + np.cos(angle) * length, y0 + np.sin(angle) * length
    line = np.zeros(shape, np.uint8)
    cv2.line(line, (int(x0), int(y0)), (int(x1), int(y1)), 255, max(1, int(width)))
    return cv2.GaussianBlur(line.astype(np.float32) / 255, (0, 0), width / 2) * rng.uniform(0.6, 1.0)


def transform_points(points, map_x, map_y):
    """Output-image positions of master points, via the nearest warp sample.

    Returns only the points that land inside the image.
    """
    found = []
    for px, py in points:
        d = (map_x - px) ** 2 + (map_y - py) ** 2
        idx = int(np.argmin(d))
        if d.flat[idx] < 25:
            y, x = divmod(idx, map_x.shape[1])
            found.append((int(x), int(y)))
    return found
