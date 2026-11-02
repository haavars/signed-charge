"""Verification of rendered friction-ridge images against their ground truth.

A renderer (procedural or diffusion) may add, drop or move ridges. Matchers are
built to tolerate exactly that, so a matcher score can't tell us. Instead every
rendered impression is compared with the clean ridge map it was rendered from:

1. NIST `mindtct` extracts minutiae from the rendered image (the detected set D)
   and from the clean binarised ridge map of the same capture (the reference
   set R). Using one extractor on both cancels out its own quirks.
2. D and R are paired one-to-one (Hungarian assignment) within 12 px and 30
   degrees, inside the eroded contact area.
3. recall = paired / |R|, spurious rate = unpaired D / |D|, and the mean
   displacement of the pairs. The unpaired points form the drift map.
4. Each ground-truth minutia is `found` when one of `mindtct`'s detections in the
   image pairs with it, by the same rule: which mated minutiae an extractor can
   actually use.
5. NFIQ 2 scores the image (per finger for slaps).
6. Tone: `blotch`, how much the grey varies over a few ridges (the standard
   deviation of the local mean over 25 px, about 2.5 ridges, less the mean over
   125 px, inside the contact area: blotches and hazes, not the slow pressure
   gradient across a print), and `spray`, how much grey lies in a band just
   outside the contact area. Smudge keeps the minutiae and NFIQ 2 doesn't mind
   it, so neither of the checks above sees it.

An impression is accepted when recall, spurious rate, NFIQ 2 and blotch all meet
the thresholds for its impression type.

Latents (`verify_latent`) are checked differently: they're meant to be hard to read, so nothing is accepted or
rejected on minutiae. What's measured is how much of the ground truth an extractor can still use (which minutiae
`mindtct` finds, and how many it finds that aren't there) and NIST OpenLQM's latent quality, from which the
latent's analysis follows: of value, limited value or no value, as an examiner would say.

The tools are NBIS (`mindtct`, `bozorth3`, `cjpegl`) and NFIQ 2, installed by
`ridgegen install-tools` (`ridgegen.tools`). Without them `available()` is false
and nothing is verified. Override their locations with NBIS_BIN (a directory) and
NFIQ2_BIN / NFIQ2_MODEL. OpenLQM (or OPENLQM_BIN) is optional: without it latents
have no quality score or analysis.
"""

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from . import tools

PPI = 500

MAX_DISTANCE = 12  # px, about 0.6 mm at 500 ppi
MAX_ANGLE = 30  # degrees
BORDER = 24  # px eroded off the contact area: the clean map's edge makes false endings
MIN_QUALITY = 20  # mindtct reliability (0-100); below this, minutiae are mostly noise

# Acceptance thresholds per impression type. Starting values from the plan, to
# be tuned on the procedural baseline. `blotch`: on 18 prints, procedural and
# diffusion ones measure about 7 (median) and at most 9, the smudged 1000 ppi ones
# of ridgegen/0.6 9.1 and up to 10.4. Provisional, until real livescan prints
# (SD302) set it.
THRESHOLDS = {
    "rolled": {"recall": 0.85, "spurious": 0.15, "nfiq2": 35, "blotch": 10},
    "plain": {"recall": 0.85, "spurious": 0.15, "nfiq2": 35, "blotch": 10},
}
TONE_WINDOWS = (25, 125)  # px at 500 ppi: blotch-sized, and the neighbourhood it's measured against


def nbis_tool(name):
    directory = os.environ.get("NBIS_BIN")
    path = Path(directory) / name if directory else tools.path("nbis", "bin", name)
    return str(path) if path.exists() else shutil.which(name)


def nfiq2_tool():
    binary = os.environ.get("NFIQ2_BIN") or tools.path("nfiq2", "bin", "nfiq2")
    model = os.environ.get("NFIQ2_MODEL") or tools.path("nfiq2", "share", "nist_plain_tir-ink.txt")
    return (str(binary), str(model)) if Path(binary).exists() and Path(model).exists() else None


def openlqm_tool():
    binary = os.environ.get("OPENLQM_BIN") or tools.path("openlqm", "bin", "openlqm")
    return str(binary) if Path(binary).exists() else None


def available():
    """True when NBIS and NFIQ 2 are installed."""
    return all(nbis_tool(t) for t in ("mindtct", "bozorth3", "cjpegl")) and nfiq2_tool() is not None


def mindtct(image, workdir=None):
    """Minutiae of an 8-bit grey 500 ppi image, as an (n, 4) array of x, y, angle, quality.

    x and y are pixels from the top left, angle is in degrees counter-clockwise
    from the x axis (ANSI/INCITS 378 convention), quality is 0-100.
    """
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        h, w = image.shape
        raw, jpl, root = f"{tmp}/i.raw", f"{tmp}/i.jpl", f"{tmp}/i"
        np.ascontiguousarray(image, dtype=np.uint8).tofile(raw)
        # Lossless JPEG: mindtct can't read PNG, and WSQ would add its own artefacts.
        run([nbis_tool("cjpegl"), "jpl", raw, "-raw_in", f"{w},{h},8,{PPI}"])
        run([nbis_tool("mindtct"), "-m1", jpl, root])
        points = np.loadtxt(f"{root}.xyt", ndmin=2).reshape(-1, 4)
    # -m1 writes angles in the standard's 2-degree units.
    points[:, 2] = (points[:, 2] * 2) % 360
    return points


def nfiq2(images):
    """NFIQ 2 scores (0-100) for a list of 8-bit grey 500 ppi images, None where it fails."""
    binary, model = nfiq2_tool()
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, image in enumerate(images):
            path = f"{tmp}/{i:03d}.png"
            Image.fromarray(image).save(path, dpi=(PPI, PPI))
            paths.append(path)
        # -a (actionable feedback) makes the output CSV even for a single file.
        output = run([binary, "-F", "-a", "-m", model, *paths])
    scores = {}
    for line in output.splitlines()[1:]:
        fields = line.split(",")
        if len(fields) >= 3:
            name = Path(fields[0].strip('"')).name
            scores[name] = int(fields[2]) if fields[2].isdigit() else None
    return [scores.get(Path(p).name) for p in paths]


def bozorth3(templates, pairs):
    """bozorth3 scores for `pairs` of indices into `templates` (minutiae arrays as from `mindtct`)."""
    if not pairs:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        for i in {i for pair in pairs for i in pair}:
            write_xyt(f"{tmp}/{i}.xyt", templates[i])
        Path(f"{tmp}/pairs.lis").write_text("".join(f"{tmp}/{i}.xyt\n{tmp}/{j}.xyt\n" for i, j in pairs))
        output = run([nbis_tool("bozorth3"), "-m1", "-A", "outfmt=s", "-M", f"{tmp}/pairs.lis"])
    return [int(line.split()[0]) for line in output.splitlines() if line.strip()]


def write_xyt(path, points):
    points = np.asarray(points, dtype=float).reshape(-1, 4)
    with open(path, "w") as f:
        for x, y, angle, quality in points:
            f.write(f"{int(x)} {int(y)} {int(round(angle / 2)) % 180} {int(quality)}\n")


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{Path(command[0]).name} failed ({result.returncode}): {result.stderr.strip()[:300]}")
    return result.stdout


def inner_area(contact, border=BORDER):
    """The contact area eroded by `border` px, where minutiae can be compared fairly."""
    return cv2.erode((contact > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=border) > 0


def keep(points, area, min_quality=MIN_QUALITY):
    """The minutiae inside `area` with at least `min_quality`."""
    if len(points) == 0:
        return points
    x = np.clip(points[:, 0].astype(int), 0, area.shape[1] - 1)
    y = np.clip(points[:, 1].astype(int), 0, area.shape[0] - 1)
    return points[area[y, x] & (points[:, 3] >= min_quality)]


def pair(detected, reference, max_distance=MAX_DISTANCE, max_angle=MAX_ANGLE):
    """One-to-one pairs (i_detected, j_reference, distance) within the tolerances."""
    if len(detected) == 0 or len(reference) == 0:
        return []
    distance = np.hypot(detected[:, None, 0] - reference[None, :, 0], detected[:, None, 1] - reference[None, :, 1])
    turn = np.abs((detected[:, None, 2] - reference[None, :, 2] + 180) % 360 - 180)
    allowed = (distance <= max_distance) & (turn <= max_angle)
    cost = np.where(allowed, distance, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distance[i, j])) for i, j in zip(rows, cols) if allowed[i, j]]


def compare(detected, reference):
    """Recall, spurious rate, displacement and drift points of `detected` against `reference`."""
    pairs = pair(detected, reference)
    paired_d = {i for i, _, _ in pairs}
    paired_r = {j for _, j, _ in pairs}
    missed = [point(reference[j]) for j in range(len(reference)) if j not in paired_r]
    spurious = [point(detected[i]) for i in range(len(detected)) if i not in paired_d]
    return {
        "reference_count": len(reference),
        "detected_count": len(detected),
        "paired_count": len(pairs),
        "minutiae_recall": ratio(len(pairs), len(reference)),
        "minutiae_spurious": ratio(len(spurious), len(detected)),
        "mean_displacement_px": round(float(np.mean([d for *_, d in pairs])), 2) if pairs else None,
        "missed": missed,
        "spurious": spurious,
    }


def ground_truth_agreement(reference, ground_truth):
    """Share of the skeleton ground truth that mindtct also finds on the clean map (positions only)."""
    if not ground_truth:
        return None
    truth = np.array([[m["x"], m["y"], 0, 100] for m in ground_truth], dtype=float)
    ref = reference.copy()
    ref[:, 2] = 0
    return ratio(len(pair(ref, truth)), len(truth))


def found(truth, detected):
    """For each ground-truth minutia (dicts with x, y, angle), whether `detected` (from `mindtct`, at least
    `MIN_QUALITY`) has it, paired one to one within the tolerances."""
    if not truth:
        return []
    detected = detected[detected[:, 3] >= MIN_QUALITY]
    reference = np.array([[m["x"], m["y"], m["angle"], 100] for m in truth], float)
    hits = {j for _i, j, _d in pair(detected, reference)}
    return [j in hits for j in range(len(truth))]


def ratio(a, b):
    return round(float(a / b), 3) if b else None


def point(row):
    return [int(row[0]), int(row[1])]


def verify(image, capture):
    """Verification metrics for `image`, rendered from `capture` (an `impression.Capture`).

    Returns a dict with the metrics, the drift points (`missed`, `spurious`), the
    detected minutiae (`detected`, for batch matching) and, per ground-truth
    minutia of the capture, whether it was found (`found`), or None when the
    tools aren't installed.
    """
    if not available():
        return None
    kind = capture.meta.get("impression", "rolled")
    regions = capture.fingers or [(capture.meta.get("fgp"), capture.contact)]
    # The tools are separate programs, each in its own directory, so they run side by side.
    with ThreadPoolExecutor(3) as pool:
        reference_all = pool.submit(mindtct, capture.ridge_map())
        detected_all = pool.submit(mindtct, image)
        scores = pool.submit(nfiq2, [crop(image, contact) for _fgp, contact in regions])
        reference_all, detected_all, scores = reference_all.result(), detected_all.result(), scores.result()

    fingers = []
    missed, spurious = [], []
    totals = np.zeros(3)
    displacement = []
    for fgp, contact in regions:
        area = inner_area(contact)
        reference = keep(reference_all, area)
        detected = keep(detected_all, area)
        result = compare(detected, reference)
        truth = [m for m in capture.meta.get("minutiae", []) if area[m["y"], m["x"]]]
        fingers.append({
            "fgp": fgp,
            "minutiae_recall": result["minutiae_recall"],
            "minutiae_spurious": result["minutiae_spurious"],
            "mean_displacement_px": result["mean_displacement_px"],
            "reference_count": result["reference_count"],
            "detected_count": result["detected_count"],
            "ground_truth_agreement": ground_truth_agreement(reference, truth),
        })
        totals += (result["paired_count"], result["reference_count"], result["detected_count"])
        if result["mean_displacement_px"] is not None:
            displacement += [result["mean_displacement_px"]] * result["paired_count"]
        missed += result["missed"]
        spurious += result["spurious"]

    for finger, score in zip(fingers, scores):
        finger["nfiq2"] = score

    paired, referenced, detected = totals
    scores = [f["nfiq2"] for f in fingers]
    metrics = {
        "nfiq2": None if None in scores else min(scores),
        "minutiae_recall": ratio(paired, referenced),
        "minutiae_spurious": ratio(detected - paired, detected),
        "mean_displacement_px": round(float(np.mean(displacement)), 2) if displacement else None,
        **tone(image, capture.contact),
    }
    metrics["accepted"] = accepted(metrics, THRESHOLDS.get(kind, THRESHOLDS["rolled"]))
    if capture.fingers:
        metrics["fingers"] = fingers
    else:
        metrics["ground_truth_agreement"] = fingers[0]["ground_truth_agreement"]
    metrics["missed"] = missed
    metrics["spurious"] = spurious
    metrics["detected"] = [[int(x), int(y), round(float(a), 1), int(q)] for x, y, a, q in detected_all]
    metrics["found"] = found(capture.meta.get("minutiae", []), detected_all)
    return metrics


def tone(image, contact):
    """`blotch` and `spray` of a 500 ppi image (see the module doc). The local means are taken over the contact
    area only, so the paper round it doesn't count. `spray` is the darkness in a band 8-40 px outside the contact
    area, beyond the paper's own (measured further out)."""
    pressed = (contact > 0.05).astype(np.uint8)
    small, large = TONE_WINDOWS
    inside = cv2.erode((contact > 0.8).astype(np.uint8), np.ones((small, small), np.uint8)) > 0
    weight = (contact > 0.5).astype(np.float32)
    grey = image.astype(np.float32) * weight
    mean = lambda k: cv2.blur(grey, (k, k)) / np.maximum(cv2.blur(weight, (k, k)), 1e-3)  # noqa: E731
    local = mean(small) - mean(large)
    edge = cv2.dilate(pressed, np.ones((17, 17), np.uint8)) > 0
    near = cv2.dilate(pressed, np.ones((81, 81), np.uint8)) > 0
    far = ~cv2.dilate(pressed, np.ones((161, 161), np.uint8)).astype(bool)
    paper = float(image[far].mean()) if far.any() else 255.0
    band = near & ~edge
    return {
        "blotch": round(float(local[inside].std()), 1) if inside.any() else None,
        "spray": round(max(0.0, paper - float(image[band].mean())), 1) if band.any() else None,
    }


def accepted(metrics, thresholds):
    recall, spurious, quality = metrics["minutiae_recall"], metrics["minutiae_spurious"], metrics["nfiq2"]
    blotch = metrics.get("blotch")
    return bool(
        recall is not None and recall >= thresholds["recall"]
        and spurious is not None and spurious <= thresholds["spurious"]
        and quality is not None and quality >= thresholds["nfiq2"]
        and (blotch is None or blotch <= thresholds["blotch"])
    )


def score(metrics):
    """How good an attempt is, for picking the best of several rejected ones."""
    return (
        metrics["accepted"],
        (metrics["minutiae_recall"] or 0) - (metrics["minutiae_spurious"] or 1),
        metrics["nfiq2"] or 0,
    )


def crop(image, contact, margin=32):
    """The image cut down to one finger's contact area, with paper around the rest."""
    ys, xs = np.nonzero(contact > 0.5)
    if len(ys) == 0:
        return image
    y0 = max(0, ys.min() - margin)
    # NFIQ 2 takes at most 1000 px of height; keep the fingertip end.
    y1 = min(image.shape[0], ys.max() + margin, y0 + 960)
    x0, x1 = max(0, xs.min() - margin), min(image.shape[1], xs.max() + margin)
    region = image[y0:y1, x0:x1].copy()
    # Neighbouring fingers of a slap would confuse NFIQ 2's foreground detection.
    mask = cv2.dilate((contact[y0:y1, x0:x1] > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    region[~mask] = 250
    return region


# OpenLQM's verbose output, after the file name.
LQM_FIELDS = ("oq", "vid", "vcmp", "oc", "area_rygb", "area_ygb", "area_gb", "area_b", "lca_ygb", "lca_grf",
              "am_y", "am_g", "am_b")
# An examiner's analysis from OpenLQM's VID (value for identification, 0-100): of value from 72, limited value from
# 50, below that no value. VID is 0 or 50-100 in practice. The cut at 72 splits SD 302h's latents as its examiners'
# analyses did: OpenLQM 42.7 / 15.9 / 41.4 % against the examiners' 41 / 16 / 43 (NIST TN 2190, and NIST's
# OpenLQM values for SD 302h, original images, masked).
VALUE_VID, LIMITED_VID = 72, 50


def openlqm(image, ppi=1000):
    """NIST OpenLQM's metrics of an 8-bit grey friction-ridge image with dark ridges, as a dict (`LQM_FIELDS`:
    `oq` overall quality 0-99, an estimate of how likely an image-only AFIS search ranks the mate first; `vid` and
    `vcmp` value for identification and comparison 0-100; areas in mm²; `am_*` automated minutiae by quality),
    or None when it isn't installed or fails."""
    tool = openlqm_tool()
    if tool is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/latent.png"
        Image.fromarray(image).save(path, dpi=(ppi, ppi))
        result = subprocess.run([tool, "-v", "-p", str(ppi), path], capture_output=True, text=True, timeout=300)
    fields = result.stdout.strip().split("\t")[1:]
    if result.returncode != 0 or len(fields) < len(LQM_FIELDS):
        return None
    values = [float(v) for v in fields[: len(LQM_FIELDS)]]
    if values[0] < 0:
        return None
    return {k: (int(v) if v.is_integer() else round(v, 1)) for k, v in zip(LQM_FIELDS, values)}


def region_of_interest(contact, shape, margin_mm=0.5):
    """The mark's region of interest, as an examiner draws it round a latent: the hull of where it touched, a
    margin out, at the size of `shape`."""
    mask = cv2.resize((contact > 0.3).astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    points = cv2.findNonZero(mask)
    roi = np.zeros(shape, np.uint8)
    if points is None:
        return roi > 0
    cv2.fillPoly(roi, [cv2.convexHull(points)], 1)
    px = max(1, int(margin_mm / 25.4 * shape[1] / contact.shape[1] * PPI))
    return cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))) > 0


def masked(image, roi):
    """`image` with everything outside `roi` white, as NIST masked SD 302h's latents for OpenLQM."""
    return np.where(roi, image, 255).astype(np.uint8)


def enhance(image, sigma=8):
    """Local contrast normalisation (each pixel against the mean and spread within about two ridges round it), as
    latents are enhanced before features are extracted. Without it `mindtct`, made for tenprints, finds next to
    nothing in a faint latent: on latents of limited value, none of their minutiae, against about ten with it."""
    grey = image.astype(np.float32)
    mean = cv2.GaussianBlur(grey, (0, 0), sigma)
    spread = np.sqrt(cv2.GaussianBlur((grey - mean) ** 2, (0, 0), sigma) + 25)
    return np.clip(128 + (grey - mean) / spread * 40, 0, 255).astype(np.uint8)


def analysis(quality):
    """"value", "limited" or "no_value" from OpenLQM's metrics, or None without them."""
    if not quality:
        return None
    if quality["vid"] >= VALUE_VID:
        return "value"
    return "limited" if quality["vid"] >= LIMITED_VID else "no_value"


def verify_latent(image, latent, full=None, reversed_polarity=False):
    """What can be read from latent `image` (500 ppi, from `latent`, an `ridgegen.latent.Latent` at 500 ppi), and
    its OpenLQM quality from `full`, the image at 1000 ppi, masked to the mark's region of interest (the surface
    round it would count as ridge flow: brushed metal and wood grain do). Ridges lighter than the background
    (`reversed_polarity`) are turned dark first, as an examiner does before a search, and `mindtct` reads the
    image after `enhance`.

    Returns a dict, or None without NBIS: `found` per ground-truth minutia (as in `verify`), `minutiae_found`, the
    `removal` and `addition` error rates (the share of ground-truth minutiae `mindtct` misses, and of its detections
    in the mark that are none of them: GenPrint's validity measure), `background_minutiae` (detections on the
    surface round the mark), `openlqm` and `analysis`.
    """
    if not available():
        return None
    if reversed_polarity:
        image = 255 - image
        full = None if full is None else 255 - full
    truth = latent.meta.get("minutiae", [])
    contact = latent.contact
    # mindtct and OpenLQM side by side, as in `verify`.
    with ThreadPoolExecutor(2) as pool:
        detected_all = pool.submit(mindtct, enhance(image))
        quality = (None if full is None
                   else pool.submit(openlqm, masked(full, region_of_interest(contact, full.shape))))
        detected_all = detected_all.result()
        quality = None if quality is None else quality.result()
    hits = found(truth, detected_all)
    mark = cv2.dilate((contact > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    near = cv2.dilate(mark.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    in_mark = keep(detected_all, mark)
    reference = np.array([[m["x"], m["y"], m["angle"], 100] for m in truth], float).reshape(-1, 4)
    paired = len(pair(in_mark, reference))
    return {
        "minutiae_found": int(sum(hits)),
        "removal": ratio(len(truth) - sum(hits), len(truth)),
        "addition": ratio(len(in_mark) - paired, len(in_mark)),
        "detected_count": len(in_mark),
        "background_minutiae": len(keep(detected_all, ~near)),
        "openlqm": quality,
        "analysis": analysis(quality),
        "found": hits,
        "detected": [[int(x), int(y), round(float(a), 1), int(q)] for x, y, a, q in in_mark],
    }
