"""Does a rerun equal a stored result? compare_results.py EXPECTED.json ACTUAL.json [TOLERANCE] [--partial]

Every number and string is compared, leaving out `_provenance`. With no tolerance numbers must be equal exactly. With --partial,
the rerun may hold fewer sections than the stored result, where the stored one also read prints the paper doesn't use.
"""
import json
import sys


def walk(x, y, path, out, tolerance):
    if isinstance(x, dict) and isinstance(y, dict):
        for key in sorted(set(x) | set(y)):
            if key == "_provenance":
                continue
            if key not in x or key not in y:
                if not (partial and key in x):
                    out.append((f"{path}/{key}", "present in only one"))
            else:
                walk(x[key], y[key], f"{path}/{key}", out, tolerance)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append((path, f"length {len(x)} against {len(y)}"))
        else:
            for i, (a, b) in enumerate(zip(x, y)):
                walk(a, b, f"{path}[{i}]", out, tolerance)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        if x != y and abs(x - y) > tolerance:
            out.append((path, f"{x} against {y}"))
        counts["n"] += 1
    else:
        if x != y:
            out.append((path, f"{x!r} against {y!r}"))
        counts["n"] += 1


counts = {"n": 0}
partial = False


def main(expected, actual, tolerance=0.0):
    out = []
    walk(json.load(open(expected)), json.load(open(actual)), "", out, float(tolerance))
    print(f"{counts['n']} values, {len(out)} differ")
    for path, what in out[:20]:
        print(f"  {path}: {what}")
    return 1 if out else 0


if __name__ == "__main__":
    partial = "--partial" in sys.argv
    sys.exit(main(*[a for a in sys.argv[1:] if a != "--partial"]))
