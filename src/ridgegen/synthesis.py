"""Ridge-pattern synthesis shared by fingers and palms.

The approach follows SFinGe (Cappelli, Maio, Maltoni): build an orientation
field from singular points, pick a ridge frequency, then grow ridges from random
noise by repeatedly filtering with Gabor filters tuned to the local orientation
and frequency. Each pass sharpens the pattern towards clean, continuous ridges;
minutiae appear naturally where the growing ridge systems meet.

Since ridgegen/0.18 the service grows ridges on the GPU when it has one (`use_device`, `RIDGEGEN_RIDGES`), about
four times faster: the same filtering with torch's FFTs. The ridges are the CPU's but for rounding (a field
differing by at most about 3e-4, one pixel in a million or so of another sign, every minutia the same), so an
image is drawn again on the kind of device it was made on, which its ground truth records (`ridges`). Two runs on
one GPU give the same ridges bit for bit; another GPU, CUDA or torch may round otherwise.
"""

import os
from functools import lru_cache

import cv2
import numpy as np
from scipy import fft

ORIENTATION_BINS = 24


# Where ridges grow: "cpu" (scipy) or "cuda" (torch). The patterns grown and kept (`keep_per_device`) are dropped
# when it changes, since each device's are its own.
_device = "cpu"
_kept = []


def use_device(device):
    """Grows ridges on `device` ("cpu" or "cuda") from now on."""
    global _device
    if device not in ("cpu", "cuda"):
        raise ValueError(f"ridges grow on the cpu or cuda, not {device}")
    if device == "cuda" and not cuda_available():
        raise RuntimeError("no CUDA device to grow ridges on")
    if device != _device:
        _device = device
        for cached in _kept:
            cached.cache_clear()


def device():
    """Where ridges grow now, as an image's ground truth records it (`ridges`): "cpu" or "cuda"."""
    return _device


def service_device():
    """Where the service grows ridges: `RIDGEGEN_RIDGES` ("cpu" or "cuda"), by default on the GPU if there is one."""
    asked = os.environ.get("RIDGEGEN_RIDGES")
    return asked if asked else ("cuda" if cuda_available() else "cpu")


@lru_cache(maxsize=1)
def cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def gpu():
    """The GPU ridges grow on and its CUDA, as `{"gpu": name, "cuda": version}`."""
    import torch

    return {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}


def keep_per_device(*functions):
    """Marks `functions` (functools caches) as keeping ridges grown on the device of the time, to be forgotten
    when it changes."""
    _kept.extend(functions)


def rng_for(*parts):
    """A numpy Generator derived deterministically from integer parts."""
    return np.random.default_rng([int(p) & 0xFFFFFFFF for p in parts])


def zero_pole_orientation(shape, cores, deltas, base=0.0):
    """Sherlock-Monro zero-pole orientation model.

    Returns ridge orientation in radians, in image coordinates (x right, y
    down), modulo pi. `base` is added everywhere and may be an array.
    """
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = np.zeros(shape, np.float32) + base
    for cx, cy in cores:
        theta += 0.5 * np.arctan2(y - cy, x - cx)
    for dx, dy in deltas:
        theta -= 0.5 * np.arctan2(y - dy, x - dx)
    return np.mod(theta, np.pi)


def smooth_noise(shape, rng, scale, amplitude=1.0):
    """Smooth random field: white noise upsampled from a coarse grid."""
    h, w = shape
    gh, gw = max(2, int(np.ceil(h / scale)) + 1), max(2, int(np.ceil(w / scale)) + 1)
    coarse = rng.standard_normal((gh, gw)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return field * amplitude


def smooth_orientation(theta, sigma):
    """Smooth an orientation field (mod pi) via its doubled-angle vector field."""
    c = cv2.GaussianBlur(np.cos(2 * theta), (0, 0), sigma)
    s = cv2.GaussianBlur(np.sin(2 * theta), (0, 0), sigma)
    return np.mod(0.5 * np.arctan2(s, c), np.pi)


@lru_cache(maxsize=64)
def gabor_kernel(bin_index, period):
    """Even Gabor kernel for ridges running along orientation bin `bin_index`."""
    theta = np.pi * bin_index / ORIENTATION_BINS
    sigma = 0.55 * period
    radius = int(np.ceil(2.6 * sigma))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
    # Distance across the ridges (perpendicular to the ridge direction).
    across = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * np.cos(2 * np.pi * across / period)
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum()
    return kernel.astype(np.float32)


def grow_ridges(theta, period, rng, iterations=14, mask=None, fixed=None):
    """Grow a ridge pattern following `theta` with ridge spacing `period` (px).

    `period` may be a scalar or a per-pixel array; per-pixel periods are
    handled by blending two period levels. `fixed` is (where, values): ridges
    held as they are while the rest grows on from them, to extend a pattern
    past its edge. Returns float32 in about [-1, 1], ridges positive.
    """
    shape = theta.shape
    # Blend between the two nearest orientation bins so bin edges don't show.
    position = theta / np.pi * ORIENTATION_BINS
    lower = np.floor(position).astype(np.int32) % ORIENTATION_BINS
    upper = (lower + 1) % ORIENTATION_BINS
    frac = (position - np.floor(position)).astype(np.float32)

    if np.isscalar(period):
        periods, period_weights = [float(period)], [np.ones(shape, np.float32)]
    else:
        lo, hi = float(np.min(period)), float(np.max(period))
        if hi - lo < 0.25:
            periods, period_weights = [0.5 * (lo + hi)], [np.ones(shape, np.float32)]
        else:
            t = ((period - lo) / (hi - lo)).astype(np.float32)
            periods, period_weights = [lo, hi], [1 - t, t]

    # Filtering runs in the frequency domain: one forward FFT per pass, shared
    # by every orientation bin, is much cheaper than a spatial filter per bin.
    kernels = [gabor_kernel(b, round(p, 2)) for p in periods for b in range(ORIENTATION_BINS)]
    pad = max(k.shape[0] for k in kernels) // 2 + 1
    padded_shape = (shape[0] + 2 * pad, shape[1] + 2 * pad)

    # Kernel spectra for every orientation bin / period level in use. Per-pixel
    # weights are recomputed each pass rather than stored: for palm-sized images
    # storing one weight map per bin would cost gigabytes.
    used = set(np.unique(lower)) | set(np.unique(upper))
    filters = [
        (b, pi_, kernel_spectrum(kernels[pi_ * ORIENTATION_BINS + b], padded_shape))
        for pi_ in range(len(periods))
        for b in range(ORIENTATION_BINS)
        if b in used
    ]

    def weight_for(b, pi_):
        weight = np.where(lower == b, 1 - frac, 0).astype(np.float32)
        weight += np.where(upper == b, frac, 0)
        return weight * period_weights[pi_] if len(periods) > 1 else weight

    # Small images (fingers): keep the weights, it's faster.
    stored = {(b, pi_): weight_for(b, pi_) for b, pi_, _ in filters} if theta.size <= 2_000_000 else None

    image = rng.standard_normal(shape).astype(np.float32) * 0.2
    if fixed is not None:
        image[fixed[0]] = fixed[1][fixed[0]]
    if _device == "cuda":
        weights = [stored[(b, pi_)] if stored is not None else weight_for(b, pi_) for b, pi_, _ in filters]
        return _grow_cuda(image, [f for _b, _p, f in filters], weights, pad, iterations, mask, fixed)
    for _ in range(iterations):
        spectrum = fft.rfft2(np.pad(image, pad, mode="reflect"), workers=-1)
        out = np.zeros(shape, np.float32)
        for b, pi_, kernel_fft in filters:
            weight = stored[(b, pi_)] if stored is not None else weight_for(b, pi_)
            filtered = fft.irfft2(spectrum * kernel_fft, s=padded_shape, workers=-1)
            out += filtered[pad:-pad, pad:-pad] * weight
        # Normalise and saturate: pushes values towards clean +-1 ridges/valleys.
        std = float(out.std()) or 1.0
        image = np.tanh(3.0 * out / std).astype(np.float32)
        if mask is not None:
            image *= mask
        if fixed is not None:
            image[fixed[0]] = fixed[1][fixed[0]]
    return image


def _grow_cuda(image, spectra, weights, pad, iterations, mask, fixed):
    """`grow_ridges`' passes on the GPU, from its set-up: every orientation bin's filter in one batch (16 at a time),
    each pass as on the CPU."""
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    shape = image.shape
    padded_shape = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    with torch.no_grad():
        k = torch.from_numpy(np.stack(spectra)).to(dev)
        w = torch.from_numpy(np.stack(weights)).to(dev)
        grown = torch.from_numpy(image).to(dev)
        keep = torch.from_numpy(mask.astype(np.float32)).to(dev) if mask is not None else None
        if fixed is not None:
            where = torch.from_numpy(fixed[0]).to(dev)
            values = torch.from_numpy(fixed[1].astype(np.float32)).to(dev)
        for _ in range(iterations):
            padded = F.pad(grown[None, None], (pad, pad, pad, pad), mode="reflect")[0, 0]
            spectrum = torch.fft.rfft2(padded)
            out = torch.zeros(shape, dtype=torch.float32, device=dev)
            for at in range(0, k.shape[0], 16):
                filtered = torch.fft.irfft2(spectrum[None] * k[at:at + 16], s=padded_shape)
                out += (filtered[:, pad:-pad, pad:-pad] * w[at:at + 16]).sum(0)
            std = float(out.std(correction=0)) or 1.0
            grown = torch.tanh(3.0 * out / std)
            if keep is not None:
                grown *= keep
            if fixed is not None:
                grown[where] = values[where]
        return grown.cpu().numpy()


def kernel_spectrum(kernel, shape):
    """FFT of `kernel` centred on the origin of a `shape` array (for convolution)."""
    padded = np.zeros(shape, np.float32)
    r = kernel.shape[0] // 2
    padded[: kernel.shape[0], : kernel.shape[1]] = kernel
    padded = np.roll(padded, (-r, -r), axis=(0, 1))
    return fft.rfft2(padded, workers=-1).astype(np.complex64)
