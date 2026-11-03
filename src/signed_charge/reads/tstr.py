"""Does a generator's charge cost a matcher anything? Train on synthetic, test on real (D-079): tstr.py DATA_DIR ARM SEED OUT.json

A ResNet-18 with an ArcFace head on one arm's ridge maps, all in memory, shifted, turned and cropped on the GPU, then FVC DB1-3 B by the FVC protocol on the same ridge maps: EER by sensor, and FMR100. A separate project's 18 sealed fingers are left out, as its split requires, so 72 fingers. HOLD leaves that share of the arm's identities out of training and reads them the same way (D-082), and `eer_ci` resamples fingers within each sensor.

Run: python -m signed_charge.reads.tstr DATA_DIR ARM SEED OUT.json
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from signed_charge import paths

SIZE, EPOCHS, BATCH, EMBED = 256, int(os.environ.get("EPOCHS", 40)), 128, 256
HOLD, RESAMPLES = float(os.environ.get("HOLD", 0)), 500
MARGIN, SCALE, TURN, SHIFT, HOLE = 0.4, 30.0, 25.0, 0.15, 0.3
SEALED = paths.EXCLUDED_FINGERS  # a separate project's sealed fingers, never read here (DATA.md)


def ridge_map(im):
    """Any print as dark ridges on white at `SIZE`: band-passed, thresholded and centred on the print."""
    g = im.astype(np.float32)
    b = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 6.0)
    var = cv2.GaussianBlur(b * b, (0, 0), 12)
    fg = var > 0.1 * var.max()
    out = np.where((b < 0) & fg, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(fg)
    if not len(ys):
        return np.full((SIZE, SIZE), 255, np.uint8)
    cy, cx = int(ys.mean()), int(xs.mean())
    canvas = np.full((SIZE, SIZE), 255, np.uint8)
    top, left = max(cy - SIZE // 2, 0), max(cx - SIZE // 2, 0)
    crop = out[top : top + SIZE, left : left + SIZE]
    y0, x0 = (SIZE - crop.shape[0]) // 2, (SIZE - crop.shape[1]) // 2
    canvas[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    return canvas


def load(folder):
    paths = sorted(Path(folder).glob("*.png"))
    ids = sorted({p.stem.rsplit("_", 1)[0] for p in paths})
    at = {k: i for i, k in enumerate(ids)}
    images = np.stack([cv2.imread(str(p), 0) for p in paths])
    labels = np.array([at[p.stem.rsplit("_", 1)[0]] for p in paths])
    return images, labels, len(ids)


def fvc():
    from signed_charge.measure import conformality as cf
    from signed_charge.measure import field_similarity as fs
    from signed_charge.reads import signed_false as sf

    sealed = set(json.load(open(SEALED))["identities"])
    bases = [f"FVC{y}/DB{d}_B" for y in (2000, 2002, 2004) for d in (1, 2, 3)]
    out = {}
    for name, label, paths in sf.fvc_fingers():
        if "DB4" not in name and bases.index(name) * 1000 + int(label.rsplit("/", 1)[1]) not in sealed:
            out.setdefault(name, {})[label] = [ridge_map(fs.at_500(cv2.imread(p, 0), cf.PPI.get(name, 500))) for p in sorted(paths)]
    return out


def equal_error(genuine, impostor):
    scores = np.sort(np.concatenate([genuine, impostor]))
    frr = np.searchsorted(np.sort(genuine), scores, side="left") / len(genuine)
    far = 1 - np.searchsorted(np.sort(impostor), scores, side="left") / len(impostor)
    i = int(np.argmin(np.abs(frr - far)))
    fmr100 = float(frr[np.searchsorted(-far, -0.01)]) if (far <= 0.01).any() else 1.0
    return float((frr[i] + far[i]) / 2), fmr100


def scored(z, rng):
    """FVC's protocol on one set: {finger: unit embeddings} to EER, FMR100, the counts and means."""
    keys = sorted(z)

    def rate(picked):
        genuine = np.concatenate([(z[k] @ z[k].T)[np.triu_indices(len(z[k]), 1)] for k in picked])
        first = np.stack([z[k][0] for k in picked])
        return genuine, (first @ first.T)[np.triu_indices(len(picked), 1)]

    genuine, impostor = rate(keys)
    eer, fmr = equal_error(genuine, impostor)
    draws = []
    for _ in range(RESAMPLES):
        picked = list(dict.fromkeys(rng.choice(len(keys), len(keys))))  # a finger twice would score against itself
        g, i = rate([keys[j] for j in picked]) if len(picked) > 1 else (genuine, impostor)
        draws.append(equal_error(g, i)[0])
    return {
        "eer": eer,
        "fmr100": fmr,
        "genuine": len(genuine),
        "impostor": len(impostor),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }, draws


def main(data, arm, seed, out):
    from signed_charge.measure import provenance
    import torch
    import torch.nn.functional as F
    import torchvision

    started = provenance.start()
    torch.manual_seed(seed)
    device = "cuda"
    images, labels, classes = load(Path(data) / arm)
    held = np.random.default_rng(0).permutation(classes)[: int(HOLD * classes)]
    train = ~np.isin(labels, held)
    x = torch.from_numpy(images[train]).to(device)
    y = torch.from_numpy(labels[train]).to(device)
    net = torchvision.models.resnet18(num_classes=EMBED).to(device)
    net.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False).to(device)
    head = torch.nn.Parameter(F.normalize(torch.randn(classes, EMBED, device=device), dim=1))
    steps = EPOCHS * max(len(x) // BATCH, 1)
    opt = torch.optim.SGD([*net.parameters(), head], lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=max(steps, 1))
    scaler = torch.amp.GradScaler("cuda")

    def augment(batch):
        n = len(batch)
        a = torch.deg2rad(torch.empty(n, device=device).uniform_(-TURN, TURN))
        s = torch.empty(n, device=device).uniform_(0.95, 1.05)
        t = torch.empty(n, 2, device=device).uniform_(-SHIFT, SHIFT)
        m = torch.stack([s * a.cos(), -s * a.sin(), t[:, 0], s * a.sin(), s * a.cos(), t[:, 1]], 1).view(n, 2, 3)
        v = 1 - batch[:, None].float() / 255
        v = F.grid_sample(v, F.affine_grid(m, v.shape, align_corners=False), align_corners=False)
        for i in range(n):  # part of a print is often missing
            h, w = (SIZE * torch.empty(2).uniform_(0, HOLE)).int().tolist()
            top, left = torch.randint(0, SIZE - h, (1,)).item(), torch.randint(0, SIZE - w, (1,)).item()
            v[i, :, top : top + h, left : left + w] = 0
        return v

    net.train()
    for step in range(steps):
        idx = torch.randint(0, len(x), (BATCH,), device=device)
        with torch.amp.autocast("cuda"):
            z = F.normalize(net(augment(x[idx])), dim=1)
            cos = z @ F.normalize(head, dim=1).T
            theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            hit = F.one_hot(y[idx], classes).bool()
            logits = SCALE * torch.where(hit, torch.cos(theta + MARGIN), cos)
            loss = F.cross_entropy(logits, y[idx])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0:
            print(arm, seed, step, steps, round(float(loss), 3), flush=True)
    net.eval()

    @torch.no_grad()
    def embed(batch):
        v = 1 - torch.from_numpy(np.stack(batch)).to(device)[:, None].float() / 255
        return F.normalize(net(v), dim=1).cpu().numpy()

    report = {"arm": arm, "seed": seed, "classes": classes, "images": len(x), "epochs": EPOCHS, "sensors": {}}
    rng = np.random.default_rng(seed)
    draws = []
    for sensor, fingers in fvc().items():
        z = {k: embed(fingers[k]) for k in sorted(fingers)}
        report["sensors"][sensor], sensor_draws = scored(z, rng)
        draws.append(sensor_draws)
    report["eer"] = float(np.mean([s["eer"] for s in report["sensors"].values()]))
    report["eer_ci"] = [float(np.percentile(np.mean(draws, 0), q)) for q in (2.5, 97.5)]
    if len(held):
        z = {int(k): embed(list(images[labels == k])) for k in held}
        report["synthetic"], _ = scored({k: v for k, v in z.items() if len(v) > 1}, rng)
    report["_provenance"] = provenance.record(started)
    json.dump(report, open(out, "w"), indent=1)
    print(arm, seed, "EER", round(report["eer"], 4), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
