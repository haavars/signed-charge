"""Do the stored results say what the paper says? check_claims.py [RESULTS_DIR]

Each line of `claims()` is one printed number: its place in the paper, the result file, the keys to it, and the value as printed.
A claim passes if the stored number rounds to the printed one. Point it at results/verify to check a rerun in the same way.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READERS = ("mindtct", "leader")


def both(place, file, keys, values, digits=2):
    """One printed pair, mindtct / LEADER, with the reader's key at the head of `keys`."""
    return [(place, file, (reader, *keys), value, digits) for reader, value in zip(READERS, values)]


def claims():
    out = []
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
    table3 = {"a": ((0.81, 1.07), ((0.72, 0.91), (0.97, 1.16))), "b": ((0.70, 1.03), ((0.53, 0.87), (0.86, 1.20))), "a_sfinge": ((2.38, 2.73), ((2.15, 2.61), (2.49, 2.95))), "b_sfinge": ((2.42, 2.76), ((1.79, 3.14), (2.10, 3.50)))}
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
    table10 = {"real": ((-0.36, -0.35), (0.01, 0.05), (0.06, 0.17)), "sfinge": ((0.40, 0.54), (0.39, 0.47), (0.22, 0.19)), "model2": ((0.44, 0.35), (0.53, 0.52), (0.31, 0.26)), "realflow": ((0.57, 0.58), (0.49, 0.45), (0.26, 0.27))}
    for name, bands in table10.items():
        for band, values in enumerate(bands, 1):
            for reader, value in zip(READERS, values):
                out.append((f"Table 10, {name}, along bin {band}", "ridge_frame_d072", (name, reader, "h", band, 0), value, 2))
    out.append(("Section 5.6, random waves C(150)", "review_controls_d071", ("waves_12", "C150", "mean"), -0.88, 2))
    return out


def main(results=ROOT / "results" / "expected"):
    results, cache, failed, missing, checked = Path(results), {}, 0, 0, 0
    for place, file, keys, printed, digits in claims():
        path = results / f"{file}.json"
        if not path.exists():
            missing += 1
            continue
        data = cache.setdefault(file, json.load(open(path)))
        checked += 1
        try:
            for key in keys:
                data = data[key]
            if round(float(data), digits) != printed:
                failed += 1
                print(f"DIFFERS  {place}: {file} {keys} is {data}, printed {printed}")
        except (KeyError, IndexError, TypeError) as error:
            failed += 1
            print(f"MISSING  {place}: {file} {keys}: {error!r}")
    print(f"{checked} printed numbers checked, {failed} don't match, {missing} skipped for want of a file")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
