"""JPEG 2000 for friction-ridge images, as NIST SP 500-289 profiles it for 1000 ppi, and NIST's downsampling of
1000 ppi images to 500 ppi (NISTIR 7839).

SP 500-289's codestream (Table 2): Profile-1 (Rsiz 2), one tile, one 8-bit component, COD and QCD in the main
header only, progression RPCL, 6 wavelet decomposition levels, 64 x 64 code blocks without selective arithmetic
coding bypass, and for rolled, flat and slap prints the irreversible 9-7 filter with 6 quality layers at 80, 60,
40, 30, 20 and 10:1, the last the final 10:1. Latents are lossless (the reversible 5-3 filter). The JP2 file
(Table 3) has a capture resolution box, and the codestream a 100-byte comment starting `EncID:` (Table 4).

Pillow's OpenJPEG encoder writes all of it but the resolution box and the profile, which `encode` adds.
"""
import io
import struct

import cv2
import numpy as np
from PIL import Image

# Compression ratios of the quality layers, lowest quality first; the last is the file's.
LAYERS = [80, 60, 40, 30, 20, 10]
DECOMPOSITION_LEVELS = 6
ENCODER_ID = "RIDGEGEN"

# NISTIR 7839's downsampling: a Gaussian low-pass filter, then every other pixel.
DOWNSAMPLE_SIGMA = 0.8475
DOWNSAMPLE_RADIUS = 4


def encode(image, ppi=1000, lossless=False):
    """`image`, an 8-bit grey array or PIL image, as a JP2 file (bytes): lossy at 10:1 with SP 500-289's quality
    layers, or `lossless`."""
    grey = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, np.uint8), mode="L")
    options = {
        "num_resolutions": DECOMPOSITION_LEVELS + 1,
        "codeblock_size": (64, 64),
        "progression": "RPCL",
        "irreversible": not lossless,
        "comment": comment(),
        "no_jp2": False,
    }
    if not lossless:
        options.update(quality_mode="rates", quality_layers=LAYERS)
    buffer = io.BytesIO()
    grey.convert("L").save(buffer, format="JPEG2000", **options)
    return with_capture_resolution(profile_1(buffer.getvalue()), ppi)


def comment():
    """SP 500-289's identification and metadata comment (Table 4): 100 bytes, `EncID:` and the encoder's 20,
    `Resvd:` and 65 reserved."""
    return f"EncID: {ENCODER_ID:<20} Resvd: {'':<65}"


def profile_1(jp2):
    """Rsiz in the SIZ marker set to 2, Profile-1, which the encoder's parameters keep to."""
    soc = jp2.index(b"\xff\x4f\xff\x51")  # SOC, then SIZ: its length (2 bytes), then Rsiz
    at = soc + 6
    return jp2[:at] + struct.pack(">H", 2) + jp2[at + 2 :]


def with_capture_resolution(jp2, ppi):
    """A JP2 file with a resolution superbox ('res ') holding the capture resolution ('resc') in pixels per metre,
    added to the end of its JP2 header box ('jp2h')."""
    per_metre = round(ppi / 0.0254)
    resc = struct.pack(">I4sHHHHbb", 18, b"resc", per_metre, 1, per_metre, 1, 0, 0)
    res = struct.pack(">I4s", 8 + len(resc), b"res ") + resc
    at = jp2.index(b"jp2h") - 4
    length = struct.unpack(">I", jp2[at : at + 4])[0]
    header = struct.pack(">I", length + len(res)) + jp2[at + 4 : at + length] + res
    return jp2[:at] + header + jp2[at + length :]


def boxes(jp2):
    """The JP2 file's top-level boxes, and the boxes inside 'jp2h' and its 'res ': {type: payload}."""
    def walk(data):
        out, i = {}, 0
        while i + 8 <= len(data):
            length, kind = struct.unpack(">I4s", data[i : i + 8])
            length = length or len(data) - i
            out[kind.decode("latin-1")] = data[i + 8 : i + length]
            i += length
        return out

    top = walk(jp2)
    header = walk(top.get("jp2h", b""))
    return {**top, **header, **walk(header.get("res ", b""))}


def downsample(image):
    """A 1000 ppi grey image at 500 ppi, as NISTIR 7839 recommends (SP 500-289 §4.4): a Gaussian low-pass filter
    (sigma 0.8475, radius 4), then every other pixel."""
    taps = np.arange(-DOWNSAMPLE_RADIUS, DOWNSAMPLE_RADIUS + 1, dtype=np.float64)
    kernel = np.exp(-(taps**2) / (2 * DOWNSAMPLE_SIGMA**2))
    kernel /= kernel.sum()
    smooth = cv2.sepFilter2D(np.asarray(image, np.float64), -1, kernel, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(np.rint(smooth[::2, ::2]), 0, 255).astype(np.uint8)
