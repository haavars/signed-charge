"""Do the stored results say what the paper says? check_claims.py [RESULTS_DIR]

Each line of `claims()` is one printed number: its place in the paper, the result file, the keys to it, and the value as printed.
A claim passes if the stored number rounds to the printed one. Point it at results/verify to check a rerun in the same way.

A file can be a tuple of names, and the first that exists is read. A key can be a function, applied to the value reached so far,
for numbers the paper derives from a stored curve rather than prints as stored.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READERS = ("mindtct", "leader")


def both(place, file, keys, values, digits=2):
    """One printed pair, mindtct / LEADER, with the reader's key at the head of `keys`."""
    return [(place, file, (reader, *keys), value, digits) for reader, value in zip(READERS, values)]


def with_ci(place, file, keys, printed, digits=2):
    """A printed value, or a value with its CI as (mean, (low, high)), under `keys`; the CI sits at `ci`."""
    if not isinstance(printed, tuple):
        return [(place, file, (*keys, "mean"), printed, digits)]
    mean, (low, high) = printed
    return [
        (place, file, (*keys, "mean"), mean, digits),
        (f"{place}, low", file, (*keys, "ci", 0), low, digits),
        (f"{place}, high", file, (*keys, "ci", 1), high, digits),
    ]


def h_band(low, high):
    """h over a band of separations, pooled by pairs, from a stored curve of h by bin centre."""

    def pooled(curve):
        inside = [(h, n) for r, h, n in zip(curve["r"], curve["h"], curve["pairs"]) if low <= r <= high]
        return sum(h * n for h, n in inside) / sum(n for _, n in inside)

    pooled.label = f"h({low}-{high} px)"
    return pooled


def claims():
    out = []
    table1 = {
        "independent": ((-0.02, -0.02), (-0.01, -0.01), (-0.04, -0.10)),
        "pairs": ((-0.38, -0.34), (-0.01, -0.01), (-0.99, -0.95)),
        "one_sign": ((1.00, 1.00), (1.00, 1.00), (3.15, 2.96)),
    }
    for name, (short, long, c60) in table1.items():
        for reader, *values in zip(READERS, short, long, c60):
            curve = ("controls", name, reader)
            out.append((f"Table 1, {name}, h(15-45)", "signed_minutiae", (*curve, h_band(15, 45)), values[0], 2))
            out.append((f"Table 1, {name}, h(60-100)", "signed_minutiae", (*curve, h_band(60, 100)), values[1], 2))
            out.append((f"Table 1, {name}, C(60)", "signed_minutiae", (*curve, "screening", "60", "mean"), values[2], 2))
    table2 = {
        "real": ((31.5, 27.9), (0.126, 0.228), (0.047, 0.096), (0.70, 1.03)),
        "sfinge": ((25.0, 24.8), (0.449, 0.526), (0.146, 0.170), (2.42, 2.76)),
        "model2": ((49.1, 51.2), (0.512, 0.478), (0.205, 0.198), (4.15, 3.88)),
        "d_s1_sp2_f3": ((35.0, 28.3), (0.536, 0.576), (0.234, 0.262), (3.39, 2.88)),
    }
    for name, (n, short, long, c60) in table2.items():
        out += both(f"Table 2, {name}, minutiae", "signed_false", (name, "all", "minutiae"), n, 1)
        out += both(f"Table 2, {name}, h(15-45)", "signed_false", (name, "all", "h_short", "mean"), short, 3)
        out += both(f"Table 2, {name}, h(60-100)", "signed_false", (name, "all", "h_long", "mean"), long, 3)
        out += both(f"Table 2, {name}, C(60)", "signed_false", (name, "all", "c60", "mean"), c60)
    table3 = {
        "a": ((0.81, 1.07), ((0.72, 0.91), (0.97, 1.16))),
        "b": ((0.70, 1.03), ((0.53, 0.87), (0.86, 1.20))),
        "a_sfinge": ((2.38, 2.73), ((2.15, 2.61), (2.49, 2.95))),
        "b_sfinge": ((2.42, 2.76), ((1.79, 3.14), (2.10, 3.50))),
    }
    for name, (mean, cis) in table3.items():
        for reader, m, (low, high) in zip(READERS, mean, cis):
            out.append((f"Table 3, {name}, C(60)", "held_out_look1", (name, reader, "pooled", "mean"), m, 2))
            out.append((f"Table 3, {name}, low", "held_out_look1", (name, reader, "pooled", "ci", 0), low, 2))
            out.append((f"Table 3, {name}, high", "held_out_look1", (name, reader, "pooled", "ci", 1), high, 2))
    table4 = {
        "real": ((0.179, 0.234), (0.044, 0.095), (0.89, 1.00)),
        "sfinge": ((0.562, 0.530), (0.147, 0.144), (2.82, 2.75)),
        "model2": ((0.529, 0.492), (0.208, 0.200), (4.29, 4.01)),
        "d_s1_sp2_f3": ((0.577, 0.592), (0.244, 0.260), (3.57, 2.96)),
    }
    for name, (short, long, c60) in table4.items():
        out += both(f"Table 4, {name}, h(15-45)", "signed_false", (name, "confirmed", "h_short", "mean"), short, 3)
        out += both(f"Table 4, {name}, h(60-100)", "signed_false", (name, "confirmed", "h_long", "mean"), long, 3)
        out += both(f"Table 4, {name}, C(60)", "signed_false", (name, "confirmed", "c60", "mean"), c60)
    columns = ("all", "confirmed", "no_singular", "scale6", "scale14")
    table5 = {
        "real": ((0.70, 1.03), (0.89, 1.00), (0.65, 0.96), (0.69, 1.02), (0.67, 1.02)),
        "sfinge": ((2.42, 2.76), (2.82, 2.75), (2.08, 2.59), (2.39, 2.74), (2.45, 2.80)),
        "model2": ((4.15, 3.88), (4.29, 4.01), (2.78, 2.81), (4.16, 3.89), (4.10, 3.87)),
        "d_s1_sp2_f3": ((3.39, 2.88), (3.57, 2.96), (2.52, 2.35), (3.38, 2.88), (3.37, 2.87)),
        "phantom": ((3.82, 3.49), None, (2.74, 2.74), (3.84, 3.50), (3.80, 3.48)),  # the rendered ZP row
    }
    for name, row in table5.items():
        for column, values in zip(columns, row):
            if values:
                out += both(f"Table 5, {name}, {column}", "signed_curves", (name, column, "c", "60", "mean"), values)
    # Table 6 is the charge part of the extractor-free read. Its stored run is D-061's; D-062 reread only the mean flow,
    # so its stored file has no charge, but scripts/reproduce.py writes both parts under the D-062 name.
    free = ("extractor_free_d061", "extractor_free_d062")
    table6 = {
        "real": ((1.03, (0.88, 1.17)), (1.00, (0.83, 1.17)), (1.22, (1.04, 1.40))),
        "sfinge": ((1.76, (1.37, 2.23)), (2.96, (2.31, 3.67)), (3.11, (2.41, 3.91))),
        "model2": ((4.22, (3.95, 4.47)), (4.26, (4.01, 4.53)), (4.04, (3.78, 4.31))),
    }
    for name, row in table6.items():
        for read, printed in zip(("res15", "mindtct15", "leader15"), row):
            out += with_ci(f"Table 6, {name}, {read}", free, (name, "charge", read, "c60"), printed)
    out += with_ci("Section 5.7, real, winding at 6 px", free, ("real", "charge", "res6", "c60"), (0.26, (0.13, 0.38)))
    out += with_ci("Section 5.7, sfinge, winding at 6 px", free, ("sfinge", "charge", "res6", "c60"), (0.27, (0.03, 0.54)))
    table7 = {
        "real": ((0.71, 1.05), (0.92, 1.20), ((1.31, (1.01, 1.59)), (1.32, (1.07, 1.57)))),
        "sfinge": ((2.42, 2.77), (2.92, 3.09), ((3.57, (2.51, 4.58)), (3.80, (2.74, 4.84)))),
        "model2": ((3.96, 3.69), (4.18, 3.92), (4.32, 4.18)),
    }
    for name, row in table7.items():
        for threshold, values in zip(("0.0", "0.3", "0.5"), row):
            for reader, printed in zip(READERS, values):
                keys = (name, reader, "filter", threshold, "c60_rescaled")
                out += with_ci(f"Table 7, {name}, {reader}, coherence {threshold}", "quality_lift_d063", keys, printed)
    outside7 = {
        "synfing": ((1.66, 2.03), ((1.99, (1.79, 2.18)), (2.29, (2.10, 2.49)))),
        "anguli": ((2.23, 2.22), ((2.88, (2.70, 3.10)), (3.11, (2.94, 3.30)))),
    }
    for name, row in outside7.items():
        for threshold, values in zip(("0.0", "0.5"), row):
            for reader, printed in zip(READERS, values):
                out += with_ci(f"Table 7, {name}, {reader}, coherence {threshold}", "review_controls_d071", (name, reader, threshold), printed)
    table8 = {"anguli": ((2.12, 2.10), (0.47, 0.46)), "synfing": ((1.67, 1.92), (0.33, 0.38))}
    for name, (c60, short) in table8.items():
        out += both(f"Table 8, {name}, C(60)", "outside_d066", ("charge", name, "c60", "mean"), c60)
        out += both(f"Table 8, {name}, h(15-45)", "outside_d066", ("charge", name, "h_short", "mean"), short)
    out += both("Table 8, anguli, classifier", "outside_d066", ("cases", "full", "anguli", "mean"), (0.846, 0.807), 3)
    out += both("Table 8, synfing, classifier", "outside_d066", ("cases", "full", "synfing", "mean"), (0.698, 0.638), 3)
    table9 = {
        "planted": ((4.4, 4.4), (-0.2, -0.4), (0.91, 0.90), (0.68, 0.66)),
        "real": ((4.1, 4.1), (2.8, 2.7), (0.91, 0.90), (0.81, 0.80)),
        "realflow": ((4.2, 4.3), (5.3, 5.2), (0.97, 0.96), (0.89, 0.89)),
        "model2": ((5.3, 5.4), (11.6, 11.7), (0.97, 0.97), (0.91, 0.91)),
        "sfinge": ((5.8, 5.4), (13.1, 13.4), (0.94, 0.94), (0.90, 0.91)),
    }
    for name, (n, x, g9, f50) in table9.items():
        for reader, *values in zip(READERS, n, x, g9, f50):
            out.append((f"Table 9, {name}, N", "flux_scale_d070", (name, reader, "minutiae"), values[0], 1))
            out.append((f"Table 9, {name}, X", "flux_scale_d070", (name, reader, "X", "mean"), values[1], 1))
            out.append((f"Table 9, {name}, r(G(9), Q)", "flux_scale_d070", (name, reader, "G", "9", "r", "mean"), values[2], 2))
            out.append((f"Table 9, {name}, r(F(50), Q)", "flux_scale_d070", (name, reader, "F", "50", "r", "mean"), values[3], 2))
    table10 = {
        "real": ((-0.36, -0.35), (0.01, 0.05), (0.06, 0.17)),
        "sfinge": ((0.40, 0.54), (0.39, 0.47), (0.22, 0.19)),
        "model2": ((0.44, 0.35), (0.53, 0.52), (0.31, 0.26)),
        "realflow": ((0.57, 0.58), (0.49, 0.45), (0.26, 0.27)),
    }
    for name, bands in table10.items():
        for band, values in enumerate(bands, 1):
            for reader, value in zip(READERS, values):
                out.append((f"Table 10, {name}, along bin {band}", "ridge_frame_d072", (name, reader, "h", band, 0), value, 2))
    out.append(("Section 5.6, random waves C(150)", "review_controls_d071", ("waves_12", "C150", "mean"), -0.88, 2))
    return out


def main(results=ROOT / "results" / "expected"):
    results, cache, failed, missing, checked = Path(results), {}, 0, 0, 0
    for place, files, keys, printed, digits in claims():
        shown = tuple(getattr(key, "label", key) for key in keys)
        file = next((f for f in ([files] if isinstance(files, str) else files) if (results / f"{f}.json").exists()), None)
        if file is None:
            missing += 1
            continue
        data = cache.setdefault(file, json.loads((results / f"{file}.json").read_text()))
        checked += 1
        try:
            for key in keys:
                data = key(data) if callable(key) else data[key]
            if round(float(data), digits) != printed:
                failed += 1
                print(f"DIFFERS  {place}: {file} {shown} is {data}, printed {printed}")
        except (KeyError, IndexError, TypeError) as error:
            failed += 1
            print(f"MISSING  {place}: {file} {shown}: {error!r}")
    print(f"{checked} printed numbers checked, {failed} don't match, {missing} skipped for want of a file")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
