"""Fine ridges in the diffusion pass (step 1): fine_ridges_experiment.py OUT [PER_BIN] [ATTEMPTS]

Rolled prints of master model 2's fingers, picked by their ridge period in the print (px at 500 ppi, without
growing a ridge: it's drawn before them), rendered as the service renders them (procedural at 1000 ppi, the
diffusion pass at the model's 500 ppi, `diffusion.at_model_scale`) and verified on a 500 ppi copy as it verifies
them, under these variants:

- `base`: today's pass, strength 0.35.
- `autoencoder`: IMPOSE's autoencoder alone (encode, decode, no noise): what the latent grid itself keeps.
- `strength25`: strength 0.25 (a diagnostic, not a fix: fine-ridged people would get more drawn texture).
- `rescaled`: the pass run on the print resampled to 500·k ppi, k = max(1, TARGET / period), so it sees ridges about
  TARGET px apart, and resampled back (NISTIR 7839's Gaussian low-pass, scaled with the factor).
- `rescaled12`: k = 1.2 on coarse prints (period ≥ 9.2 px), against `base`: how much the texture's size shifts.

One JSON line per print and variant and attempt: the period, k, minutiae recall and spurious share, NFIQ 2, whether
it passed, and texture measures in the contact (the share of power at 0.25-0.5 cycles/px, local contrast). Needs the
GPU (the diffusion model, `ridgegen install-tools --diffusion`) and NBIS and NFIQ 2 (`RIDGEGEN_TOOLS`).

Run: python -m signed_charge.exploratory.fine_ridges_experiment OUT.jsonl 20 3
"""
import json
import sys

import cv2
import numpy as np

from ridgegen import diffusion, fingerprints, jp2, verify
from ridgegen import finger as fingers
from ridgegen import impression as imp
from ridgegen.synthesis import rng_for

TARGET = 8.8
BINS = [(6.4, 7.2), (7.2, 7.6), (7.6, 8.0), (8.0, 8.5), (8.5, 9.2), (9.2, 10.5)]


def print_period(seed, fgp):
    """The ridge period of `fgp`'s rolled print, px at 500 ppi, as master model 2 draws it, without its ridges."""
    rng = rng_for(seed, fgp, fingers.MASTER_SALT)
    fingers.choose_pattern_by_hand(fgp, rng)
    scale = fingers.FINGER_SCALE[fingers.name_of(fgp)] * rng.uniform(0.95, 1.05)
    return fingers.ridge_period(seed, fgp, scale, rng) * fingers.print_scale(seed, fgp, scale)


def pick(per_bin, first=5000):
    chosen = {b: [] for b in BINS}
    seed = first
    while any(len(v) < per_bin for v in chosen.values()):
        for fgp in range(1, 11):
            p = print_period(seed, fgp)
            for b in BINS:
                if b[0] <= p < b[1] and len(chosen[b]) < per_bin:
                    chosen[b].append((seed, fgp, p))
        seed += 1
    return chosen


def resample(image, factor, size=None):
    """`image` scaled by `factor` (< 1 shrinks): a Gaussian low-pass as NISTIR 7839's for 2:1 (sigma 0.8475 at
    factor 0.5), scaled with the factor, then resampled."""
    if factor < 1:
        image = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 0.8475 * (0.5 / factor))
    size = size or (round(image.shape[1] * factor), round(image.shape[0] * factor))
    return cv2.resize(image.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)


def autoencode(grey, seed, strength):
    """IMPOSE's autoencoder alone: encode and decode, no noise."""
    import torch

    model, _sampler = diffusion.load()
    h, w = grey.shape
    padded = np.pad(grey, ((0, -h % 16), (0, -w % 16)), constant_values=255)
    x = torch.from_numpy((padded / 127.5 - 1).astype(np.float32))[None, None].cuda()
    with torch.no_grad(), torch.autocast("cuda"), model.ema_scope():
        out = model.decode_first_stage(model.get_first_stage_encoding(model.encode_first_stage(x)))
    image = ((torch.clamp((out + 1) / 2, 0, 1)[0, 0].float().cpu().numpy()) * 255).round().astype(np.uint8)
    torch.cuda.empty_cache()
    return image[:h, :w]


def rendered(capture, drawn, attempt, variant, period):
    """The 1000 ppi image of `drawn` for `attempt`, as the service renders it, the pass changed by `variant`."""
    image = imp.render_capture(drawn, attempt)
    rng = rng_for(*drawn.appearance_key, 202, attempt)
    seed = int(rng.integers(0, 2**31))
    strength = 0.25 if variant == "strength25" else diffusion.STRENGTH
    run = autoencode if variant == "autoencoder" else diffusion.render
    k = 1.0
    if variant == "rescaled":
        k = max(1.0, TARGET / period)
    elif variant == "rescaled12":
        k = 1.2
    if k == 1.0:
        return diffusion.at_model_scale(image, drawn.contact, lambda low: run(low, seed, strength)), k

    def diffuse(low):
        # `low` is at 500 ppi; the pass runs at 500·k ppi and comes back to 500.
        big = np.clip(resample(low, k), 0, 255).round().astype(np.uint8)
        out = run(big, seed, strength)
        return np.clip(resample(out, 1 / k, (low.shape[1], low.shape[0])), 0, 255).round().astype(np.uint8)

    return diffusion.at_model_scale(image, drawn.contact, diffuse), k


def texture(image500, contact500):
    """Share of the contact's power at 0.25-0.5 cycles/px, and local contrast (std over 9 px)."""
    inside = contact500 > 0.8
    g = image500.astype(np.float32)
    ys, xs = np.nonzero(inside)
    y0, x0 = int(np.median(ys)) - 64, int(np.median(xs)) - 64
    patch = g[max(0, y0):y0 + 128, max(0, x0):x0 + 128]
    if patch.shape != (128, 128):
        return None, None
    patch = (patch - patch.mean()) * np.outer(np.hanning(128), np.hanning(128))
    p = np.abs(np.fft.fftshift(np.fft.fft2(patch))) ** 2
    yy, xx = np.mgrid[-64:64, -64:64] / 128
    r = np.hypot(xx, yy)
    high = float(p[(r >= 0.25) & (r <= 0.5)].sum() / p[r > 0.02].sum())
    local = cv2.GaussianBlur(g, (0, 0), 4.5)
    contrast = float(np.sqrt(cv2.GaussianBlur((g - local) ** 2, (0, 0), 4.5))[inside].mean())
    return high, contrast


def main(out, per_bin=20, attempts=3):
    chosen = pick(per_bin)
    with open(out, "w") as f:
        for b, items in chosen.items():
            for seed, fgp, period in items:
                capture = fingerprints.rolled_capture(seed, fgp, 0)
                drawn = imp.upscale(capture, 2)
                variants = ["base", "autoencoder", "strength25", "rescaled"] + (["rescaled12"] if period >= 9.2 else [])
                for variant in variants:
                    for attempt in range(attempts if variant != "autoencoder" else 1):
                        image, k = rendered(capture, drawn, attempt, variant, period)
                        small = jp2.downsample(image)
                        m = verify.verify(small, capture)
                        high, contrast = texture(small, capture.contact)
                        f.write(json.dumps({
                            "seed": seed, "fgp": fgp, "period": round(period, 2), "bin": b[0], "variant": variant,
                            "attempt": attempt, "k": round(k, 3), "recall": m["minutiae_recall"],
                            "spurious": m["minutiae_spurious"], "nfiq2": m["nfiq2"], "accepted": m["accepted"],
                            "high_frequency": high, "contrast": contrast}) + "\n")
                        f.flush()
            print(b, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
