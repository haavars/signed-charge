"""A print patch made by sweeping a line along the ridges (D-068, D-075): sweep.sheet(seed, b, s, k, m) -> image

A line crosses the ridges and moves along them, carrying a sorted list of ridges. At each step they follow a smooth blueprint, relax towards even spacing with stiffness k, and at rate b a ridge is gained in a gap or lost, a little more often where a gap is wide or tight, by s. A gain's new ridge is lost again soon with chance m. A gain is a fork or a start, a loss a merge or an end, with `FORK` the share of forks. So sign and type are separate draws, and ridges bend round events as well as following the blueprint.
"""
import cv2
import numpy as np

SIZE, LEAD, MARGIN = 320, 120, 80  # the sheet, the sweep's start before it, and what's kept above and below
PERIOD, FORK, TIGHT, WIDE, SCALE, DIRECTION, MEMORY = 10.6, 0.35, 0.7, 1.5, 50, 0.162, 20.0  # px, share, two of the period, px, rad, px
GRACE, JOIN = 40, 28  # px a new ridge has to make room before spacing can take it, and a join's length


def blueprint(rng):
    """A smooth small-angle field with real's spread within a 96 px patch, D-075."""
    w, h = SIZE + LEAD, SIZE + 2 * MARGIN
    field = cv2.GaussianBlur(rng.standard_normal((h + 300, w + 300)).astype(np.float32), (0, 0), SCALE, borderType=cv2.BORDER_REPLICATE)[150:-150, 150:-150]
    patches = [field[i : i + 96, j : j + 96].std() for i in range(0, h - 96, 96) for j in range(0, w - 96, 96)]
    return field / max(np.mean(patches), 1e-9) * DIRECTION


def sheet(seed, b, s, k, m, period=PERIOD):
    rng = np.random.default_rng(seed)
    theta = blueprint(rng)
    height, width = theta.shape
    y = np.arange(0.0, height + period, period)
    lines = [{"x": [0], "y": [v], "end": None, "from": None, "into": None} for v in y]
    live = list(range(len(lines)))

    def across(y, u):
        """Gaps measured across the ridges, not along the sweep line."""
        return np.diff(y) * np.cos(theta[np.clip(np.rint(y[:-1]).astype(int), 0, height - 1), u])

    def new(u, at, parent=None, end=None):
        lines.append({"x": [u], "y": [float(at)], "end": end, "from": parent, "into": None})
        return len(lines) - 1

    for u in range(1, width):
        y = np.array([lines[i]["y"][-1] for i in live])
        cells = np.clip(np.rint(y).astype(int), 0, height - 1)
        y = y + np.tan(theta[cells, u])
        y[1:-1] += k * (y[2:] - 2 * y[1:-1] + y[:-2])
        for i, v in zip(live, y):
            lines[i]["x"].append(u)
            lines[i]["y"].append(float(v))
        # The sheet's edges keep their supply of ridges. These aren't events: they lie outside what's read.
        while lines[live[0]]["y"][-1] > -period:
            live.insert(0, new(u, lines[live[0]]["y"][-1] - period))
        while lines[live[-1]]["y"][-1] < height + period:
            live.append(new(u, lines[live[-1]]["y"][-1] + period))
        y = np.array([lines[i]["y"][-1] for i in live])
        gaps = across(y, u)
        # Losses first: ridges too tight, then free ones, then gains, each read from this step's gaps.
        narrow = np.minimum(gaps[:-1], gaps[1:])
        settled = np.array([u - lines[i]["x"][0] > GRACE for i in live])
        losing = (rng.random(len(narrow)) < b * np.exp(s * (1 - narrow / period))) | (narrow < TIGHT * period)
        losing &= settled[1:-1]
        scheduled = [j for j, i in enumerate(live) if lines[i]["end"] == u and 0 < j < len(live) - 1]
        for j in sorted(set((np.flatnonzero(losing) + 1).tolist()) | set(scheduled), reverse=True):
            if 0 < j < len(live) - 1:
                if rng.random() < FORK:
                    lines[live[j]]["into"] = live[j - 1 if rng.random() < 0.5 else j + 1]
                live.pop(j)
        y = np.array([lines[i]["y"][-1] for i in live])
        gaps = across(y, u)
        settled = np.array([u - lines[i]["x"][0] > GRACE for i in live])
        room = (gaps > period) & settled[:-1] & settled[1:]
        gaining = ((rng.random(len(gaps)) < b * np.exp(s * (gaps / period - 1))) & room) | (gaps > WIDE * period)
        for j in np.flatnonzero(gaining)[::-1]:
            end = u + int(rng.exponential(MEMORY)) + 15 if rng.random() < m else None
            parent = live[j if rng.random() < 0.5 else j + 1] if rng.random() < FORK else None
            live.insert(j + 1, new(u, (y[j] + y[j + 1]) / 2, parent, end))
    return render(lines, period)


def render(lines, period=PERIOD):
    """Forks and merges as smooth joins: a ridge eases out of its parent, or into its neighbour, over `JOIN` px."""
    paths = {i: (line["x"][0], np.array(line["y"])) for i, line in enumerate(lines)}

    def ease(i, other, start, step):
        x0, ys = paths[i]
        ox, oy = paths[other]
        for t in range(JOIN):
            x = start + step * t
            a, c = x - x0, x - ox
            if not (0 <= a < len(ys) and 0 <= c < len(oy)):
                break
            mix = t / JOIN
            mix = mix * mix * (3 - 2 * mix)
            ys[a] = oy[c] + (ys[a] - oy[c]) * mix

    for i, line in enumerate(lines):
        if line["from"] is not None:
            ease(i, line["from"], line["x"][0], 1)
        if line["into"] is not None:
            ease(i, line["into"], line["x"][-1], -1)
    canvas = np.full((2 * SIZE, 2 * SIZE), 255, np.uint8)
    for i, line in enumerate(lines):
        x0, ys = paths[i]
        pts = np.stack([(np.arange(x0, x0 + len(ys), dtype=float) - LEAD) * 2, (ys - MARGIN) * 2], 1).astype(np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 0, thickness=int(round(period)), lineType=cv2.LINE_AA)
    small = cv2.GaussianBlur(cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    return np.where(small < 128, 40, 235).astype(np.uint8)
