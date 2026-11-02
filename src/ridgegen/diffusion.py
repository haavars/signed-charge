"""Diffusion renderer: the texture of real prints over a capture's ridges.

The procedural renderer's prints are geometrically right but look drawn:
even ridges, smooth edges, uniform tone. This renderer takes the procedural
image of a capture and runs it part-way through IMPOSE's rolled-fingerprint
latent diffusion model (Pan et al., Tsinghua; Apache 2.0), which was trained on
3,093 optical livescan rolled prints (NIST SD302 and Tsinghua's own) at 500 ppi.
This is SDEdit: noise the image to `STRENGTH` of the way, then denoise it with
the model. The ridge flow and minutiae survive, while the model replaces the
texture with its own: ragged ridge edges, pores, uneven contact, broken contact
at the edges.

Diffusion can still move or invent minutiae, so every image is verified
(ridgegen.verify) and re-rendered with a new seed when it drifts. At strength 0.35
about 97% of attempts pass; higher strengths look more worn but drift more.
Fine ridges, under about 8 px at 500 ppi, break up at 0.35 into fragments the
ground truth doesn't have, so since ridgegen/0.25.1 a print's strength falls with
its ridge period (`strength_for`).

The model is unconditional, so there is one acquisition style: an optical
livescan rolled print at 500 ppi. (Notes until 2026-10-22 called it inked; its
paper says optical.) A 1000 ppi print is diffused at 500 ppi and gets its finer
detail from the drawing (`at_model_scale`).
It needs a CUDA GPU, the `diffusion` extra and `ridgegen install-tools
--diffusion`, which puts IMPOSE, taming-transformers and the checkpoint in the
tools directory (`ridgegen.tools`). The authors' checkpoint is a pickle, which
can run code when loaded; the installer checks its hash and converts it once to
safetensors (`convert`), and only that is loaded.
"""

import os
import sys
import threading
from pathlib import Path

import numpy as np

from . import tools

NAME = "diffusion/impose-rolled-sdedit"

STRENGTH = 0.35  # share of the diffusion trajectory re-run: texture changes, ridges stay
STEPS = 50  # DDIM steps for the full trajectory; STRENGTH * STEPS actually run
PAPER = 255  # the model's paper: white, nearly without noise

# Fine ridges (measured on 1,260 rolled prints): at 0.35 the pass's noise breaks ridges
# 6.4-7.2 px apart (at 500 ppi) into fragments the ground truth doesn't have, 8 % of minutiae spurious and 15 % of
# attempts rejected, against 3 % and about 1 % from 8 px. The autoencoder alone keeps them, and at 0.25 none were
# rejected with texture within seed-to-seed spread of 0.35's. So the strength falls linearly from the request's at
# `FINE_FROM` px to `FINE_STRENGTH` at `FINE_TO` px and below (since ridgegen/0.25.1): about 92 % of fingers are
# rendered as before.
FINE_STRENGTH = 0.25
FINE_FROM, FINE_TO = 8.0, 7.2

_model = None
_sampler = None
_load_lock = threading.Lock()
# In a render worker (ridgegen.workers) the model isn't loaded: `render` hands its image to the service's own
# process, which holds the one copy on the GPU. Set by the worker when it starts.
_remote = None


class GPUOutOfMemory(RuntimeError):
    """A diffusion step a render worker asked for ran out of GPU memory in the service's process."""


def impose():
    return tools.path("impose")


def taming():
    return tools.path("taming-src")


def config():
    return impose() / "configs" / "fingerprint-ldm-vq-128-512-rolled.yaml"


def checkpoint():
    """The converted weights (`convert`), or `IMPOSE_CHECKPOINT`."""
    configured = os.environ.get("IMPOSE_CHECKPOINT")
    if configured:
        return Path(configured)
    return impose() / "models" / "fingerprint_ldm_rolled_512" / "model_ldm_rolled.safetensors"


def available():
    """True when the code, checkpoint, torch and a CUDA GPU are all there."""
    if _remote is not None:
        return True  # the service's process checked before it sent the render here
    if not (config().exists() and checkpoint().exists() and taming().exists()):
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def unavailable_reason():
    if not (config().exists() and checkpoint().exists()):
        return "the diffusion renderer isn't installed (run `ridgegen install-tools --diffusion`)"
    try:
        import torch
    except ImportError:
        return "torch isn't installed in the service's venv (install ridgegen's `diffusion` extra)"
    if not torch.cuda.is_available():
        return "the diffusion renderer needs a CUDA GPU"
    return None


def strength_for(period, strength=STRENGTH):
    """The pass's strength for a print whose ridges are `period` px apart at 500 ppi, `strength` its coarse prints'
    (`FINE_STRENGTH`)."""
    t = float(np.clip((FINE_FROM - period) / (FINE_FROM - FINE_TO), 0, 1))
    return strength - t * (strength - min(strength, FINE_STRENGTH))


def render(procedural, seed, strength=STRENGTH):
    """Re-texture a procedural 8-bit grey image. Deterministic for a given `seed`."""
    if _remote is not None:
        return _remote(procedural, seed, strength)
    import torch

    model, sampler = load()
    h, w = procedural.shape
    padded = np.pad(procedural, ((0, -h % 16), (0, -w % 16)), constant_values=255)
    x = torch.from_numpy((padded / 127.5 - 1).astype(np.float32))[None, None].cuda()
    t_enc = max(1, int(strength * STEPS))
    with torch.no_grad(), torch.autocast("cuda"), model.ema_scope():
        z = model.get_first_stage_encoding(model.encode_first_stage(x))
        sampler.make_schedule(ddim_num_steps=STEPS, ddim_eta=0.0, verbose=False)
        # DDIM with eta 0 is deterministic after the initial noise, which comes from this seed.
        torch.manual_seed(seed)
        noisy = sampler.stochastic_encode(z, torch.tensor([t_enc], device="cuda"))
        z = sampler.decode(noisy, None, t_enc)
        out = model.decode_first_stage(z)
    image = ((torch.clamp((out + 1) / 2, 0, 1)[0, 0].float().cpu().numpy()) * 255).round().astype(np.uint8)
    del x, z, noisy, out
    torch.cuda.empty_cache()  # leave the GPU's memory to whatever else runs on it
    return image[:h, :w]


def out_of_memory(exc):
    """True when `exc` is the GPU running out of memory. The GPU may be shared, so another process can hold it for a
    while: worth trying again later."""
    torch = sys.modules.get("torch")
    return isinstance(exc, GPUOutOfMemory) or (torch is not None and isinstance(exc, torch.OutOfMemoryError))


def at_model_scale(drawn, contact, diffuse):
    """Diffuse a 1000 ppi drawing at the model's own 500 ppi, and keep the drawing's finer detail.

    The model learned at 500 ppi. Run on a 1000 ppi image, every feature is twice the size it knows, and it fills
    the room with large-scale tone: blotches and haze. So `diffuse` (500 ppi grey in,
    500 ppi grey out) runs on the drawing downsampled as NISTIR 7839 says, its output is resampled up, and the
    band the 500 ppi image can't hold (ridge edges, pores) comes from the drawing: only inside the contact area,
    and only where the model kept the drawn ridge or valley, so a ridge it moved doesn't get a second edge.
    """
    import cv2

    from . import jp2

    low = jp2.downsample(drawn)
    size = (drawn.shape[1], drawn.shape[0])
    up = cv2.resize(diffuse(low).astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    drawn = drawn.astype(np.float32)
    fine = drawn - cv2.resize(low.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    # Ridge or valley: which side of its neighbourhood's mean a pixel is on (about half a ridge period).
    side = lambda a: a - cv2.GaussianBlur(a, (0, 0), DETAIL_SIGMA)  # noqa: E731
    agree = cv2.GaussianBlur((side(up) * side(drawn) > 0).astype(np.float32), (0, 0), 1.5)
    keep = agree * np.clip(contact, 0, 1)
    return np.clip(up + DETAIL_GAIN * fine * keep, 0, 255).round().astype(np.uint8)


# px at 1000 ppi: the neighbourhood a ridge or valley is judged against in `at_model_scale`.
DETAIL_SIGMA = 4
# How strongly the drawing's finer band is added back in `at_model_scale`. 1 adds it as drawn; more makes up for
# the edge sharpness the 500 ppi pass smooths away.
DETAIL_GAIN = 2.0


def render_fingers(procedural, fingers, seeds, strength=STRENGTH, margin=32):
    """Re-texture a slap one finger at a time: each finger's region (its contact area and `margin` px around it)
    is cut out, rendered on its own, and pasted back inside its contact area, on white paper as the model
    renders it (the procedural paper is grey and noisy, and would leave a halo round each finger).

    The model was trained on single rolled prints, so a finger on its own is closer to what it knows than a
    whole slap, and it needs a third of the memory. `fingers` is [(fgp, placed contact mask)], `seeds` one per
    finger, `strength` one for all or one per finger (`strength_for`).
    """
    import cv2

    out = np.full_like(procedural, PAPER)
    h, w = procedural.shape
    strengths = strength if isinstance(strength, (list, tuple)) else [strength] * len(fingers)
    for (_fgp, contact), seed, strength in zip(fingers, seeds, strengths):
        ys, xs = np.nonzero(contact > 0.02)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin + 1)
        x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin + 1)
        rendered = render(procedural[y0:y1, x0:x1], seed, strength)
        # Paste within the contact area, grown a little so the ink's own ragged edge comes along, and feathered.
        mask = cv2.dilate((contact[y0:y1, x0:x1] > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8))
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(region * (1 - mask) + rendered * mask, 0, 255).round().astype(np.uint8)
    return out


def load():
    global _model, _sampler
    with _load_lock:
        if _model is None:
            _model, _sampler = _load()
    return _model, _sampler


def _load():
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    for path in (impose(), taming()):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.modules.diffusionmodules import model as blocks
    from ldm.modules.diffusionmodules import openaimodel
    from ldm.util import instantiate_from_config

    blocks.AttnBlock.forward = _attention
    openaimodel.QKVAttentionLegacy.forward = _qkv_attention
    model = instantiate_from_config(OmegaConf.load(config()).model)
    model.load_state_dict(load_file(checkpoint()), strict=False)
    model = model.cuda().eval()
    return model, _Quiet(DDIMSampler(model))


def convert(ckpt, out):
    """Writes the weights of IMPOSE's pickled Lightning checkpoint `ckpt` to safetensors at `out`.

    The pickle is read with torch's weights-only unpickler, which builds only tensors and plain data and runs
    no code. Besides the weights, the file holds training state (epoch, optimizer, scheduler) with NumPy
    scalars, saved by NumPy 1 as `numpy.core.multiarray.scalar`: that name is allowed as a wrapper of NumPy's
    own function. Only `state_dict` is written; the training state isn't needed to render.
    """
    import torch
    from safetensors.torch import save_file

    core = getattr(np, "_core", None) or np.core

    def scalar(*args):
        return core.multiarray.scalar(*args)

    scalar.__module__, scalar.__name__, scalar.__qualname__ = "numpy.core.multiarray", "scalar", "scalar"
    dtypes = [getattr(np.dtypes, name) for name in dir(np.dtypes) if name.endswith("DType")]
    with torch.serialization.safe_globals([scalar, np.dtype, *dtypes]):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    save_file({k: v.contiguous() for k, v in state.items()}, out)


def _attention(self, x):
    """The autoencoder's spatial self-attention, via memory-efficient SDPA.

    The original materialises the full (h*w)^2 attention matrix, about 18 GB for
    a rolled print; SDPA computes the same thing in a few hundred MB.
    """
    import torch.nn.functional as F

    h = self.norm(x)
    q, k, v = self.q(h), self.k(h), self.v(h)
    b, c, height, width = q.shape
    # The fused kernels need contiguous (batch, heads, tokens, channels) inputs;
    # anything else silently falls back to the materialising one.
    q, k, v = (t.reshape(b, 1, c, height * width).transpose(2, 3).contiguous() for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(2, 3).reshape(b, c, height, width)
    return x + self.proj_out(out)


def _qkv_attention(self, qkv):
    """The UNet's multi-head attention, via SDPA for the same reason: 6 GB less on a slap."""
    import torch.nn.functional as F

    bs, width, length = qkv.shape
    ch = width // (3 * self.n_heads)
    q, k, v = qkv.reshape(bs, self.n_heads, 3 * ch, length).split(ch, dim=2)
    q, k, v = (t.transpose(2, 3).contiguous() for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(2, 3).reshape(bs, -1, length)


class _Quiet:
    """DDIMSampler without its per-call progress bars on stdout."""

    def __init__(self, sampler):
        self._sampler = sampler

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def decode(self, *args, **kwargs):
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self._sampler.decode(*args, **kwargs)
