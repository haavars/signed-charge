"""Rerun the paper's reads and compare each with its stored result: reproduce.py STAGE [--only NAME] [--list]

Stages, in the order to run them:
    quick      the planted controls and the main read
    generate   the Turing fronts, the outside links, the minutiae cache and the matcher's training prints
    reads      every other read behind the paper
    matcher    the matcher test, on a GPU

Results land in results/verify/, a finished step is skipped, and a summary is appended to results/verify/summary.jsonl.
Held-out reads aren't here, by design: see DATA.md.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from signed_charge import paths

ROOT = paths.ROOT
VERIFY = ROOT / "results" / "verify"
EXPECTED = paths.RESULTS
G = paths.GENERATED
CACHE = str(paths.CACHE)
HOURS = 3600


def runs(pattern):
    return sorted(str(path) for path in G.glob(pattern))


def read(name, module, *args, expected=None, env=None, partial=False, stage="reads"):
    """A step that writes results/verify/NAME.json, compared with results/expected/NAME.json."""
    return {"name": name, "stage": stage, "module": module, "args": args, "expected": expected or name, "env": env or {}, "partial": partial}


def build(name, command, done, env=None):
    """A step that makes data: `command` runs unless `done` already exists."""
    return {"name": name, "stage": "generate", "command": command, "done": done, "env": env or {}}


def out(name):
    return str(VERIFY / f"{name}.json")


def figure(name):
    return str(VERIFY / f"{name}.jpg")


def steps():
    M, R = "signed_charge.measure", "signed_charge.reads"
    field, flows, real_flow, read_flow, lead = (runs(p) for p in ("field_d049/field_*.npz", "flow_d055/*flow_*.npz", "flow_d055/realflow_*.npz", "flow_d055/readflow_*.npz", "lead_d067/lead_*.npz"))
    outside = runs("outside_d066/*.jpg")
    deferred = [
        # quick
        read("signed_minutiae", f"{M}.signed_minutiae", out("signed_minutiae"), partial=True, stage="quick"),
        read("signed_reinked", f"{M}.signed_minutiae", out("signed_reinked"), "--reinked", stage="quick"),
        read("signed_false", f"{R}.signed_false", out("signed_false"), stage="quick"),
        # generate
        build("population", ["scripts/generate/population.sh", str(G / "population_d020"), "120", "60", "20", "4100"], G / "population_d020" / "whorl_a0.7_s0.03_4280.npz"),
        build("fronts on model flows", ["scripts/generate/field_population.sh", str(G / "field_d049")], G / "field_d049" / "field_9124_10.npz"),
        build("fronts on real flows", ["scripts/generate/flow_population.sh", str(G / "flow_d055")], G / "flow_d055" / "readflow_9111_10.npz"),
        build("slower, noisier fronts", ["scripts/generate/lead_run.sh", str(G / "lead_d067"), "population", "0.5"], G / "lead_d067" / "lead_9144_10.npz"),
        build("outside links", ["scripts/generate/outside_links.sh", str(G / "outside_d066")], G / "outside_d066" / "anguli_0090.jpg"),
        build("minutiae cache", [sys.executable, "-m", f"{M}.minutiae_cache", CACHE, *runs("population_d020/*.npz")], paths.CACHE),
        build("matcher prints", [sys.executable, "-m", f"{R}.tstr_data", str(G / "tstr_d079")], G / "tstr_d079" / "synfing"),
        # reads
        read("signed_curves", f"{R}.signed_curves", out("signed_curves")),
        read("classifier_floor", f"{R}.classifier_floor", out("classifier_floor"), "20"),
        read("variant_check", f"{R}.variant_check", out("variant_check"), "e_vary", "e_far"),
        read("fanout", f"{M}.fanout", out("fanout"), partial=True),
        read("minutiae_parts", f"{R}.minutiae_parts", CACHE, out("minutiae_parts")),
        read("minutiae_place", f"{R}.minutiae_place", CACHE, out("minutiae_place")),
        read("minutiae_cored", f"{R}.minutiae_cored", CACHE, out("minutiae_cored")),
        read("population_d020", f"{R}.population_read", out("population_d020"), figure("population"), *runs("population_d020/*.npz")),
        read("field_d049", f"{R}.field_read", CACHE, out("field_d049"), figure("field"), *field),
        read("flow_charge_d054", f"{M}.flow_charge", CACHE, out("flow_charge_d054"), *field),
        read("agreement_total_d057", f"{R}.agreement_total", CACHE, out("agreement_total_d057"), *field),
        read("agreement_null_d057", f"{R}.agreement_null", out("agreement_null_d057")),
        read("flow_read_d055", f"{R}.flow_read", CACHE, out("flow_read_d055"), *flows),
        read("agreement_scales_d062", f"{R}.agreement_scales", CACHE, out("agreement_scales_d062"), *flows),
        read("spacing_pays_d062", f"{R}.spacing_pays", CACHE, out("spacing_pays_d062"), *real_flow),
        read("flow_agree_readflow", f"{R}.flow_agree", CACHE, out("flow_agree_readflow"), *read_flow),
        read("flow_agree_realflow", f"{R}.flow_agree", CACHE, out("flow_agree_realflow"), *real_flow),
        read("extractor_free_d062", f"{R}.extractor_free", out("extractor_free_d062")),
        read("quality_lift_d063", f"{R}.quality_lift", out("quality_lift_d063")),
        read("quality_degrade_d064", f"{R}.quality_degrade", out("quality_degrade_d064")),
        read("cancel_sweep_e102", f"{R}.cancel_sweep", out("cancel_sweep_e102")),
        read("outside_d066", f"{R}.outside_read", CACHE, out("outside_d066"), *outside),
        read("review_controls_d071", f"{R}.review_controls", out("review_controls_d071")),
        read("lead_population_d067", f"{R}.lead_read", CACHE, out("lead_population_d067"), *lead),
        read("lead_population_agree_d067", f"{R}.flow_agree", CACHE, out("lead_population_agree_d067"), *lead),
        read("lead_population_what_d067", f"{R}.neutral_what", out("lead_population_what_d067"), *lead),
        read("flux_scale_d070", f"{R}.flux_scale", out("flux_scale_d070")),
        read("flux_d078_64", f"{R}.flux_scale", out("flux_d078_64"), "real", "sfinge", "model2", "realflow", env={"SIDE": "64"}),
        read("flux_d078_128", f"{R}.flux_scale", out("flux_d078_128"), "real", "sfinge", "model2", "realflow", env={"SIDE": "128"}),
        read("flux_d078_q0", f"{R}.flux_scale", out("flux_d078_q0"), "real", "sfinge", "model2", "realflow", env={"QUALITY": "0"}),
        read("flux_d078_outside", f"{R}.flux_scale", out("flux_d078_outside"), "synfing", "anguli"),
        read("flux_e124_outside_q0", f"{R}.flux_scale", out("flux_e124_outside_q0"), "synfing", "anguli", env={"QUALITY": "0"}),
        read("ridge_frame_d072", f"{M}.ridge_frame", out("ridge_frame_d072")),
        read("ridge_frame_outside_d074", f"{M}.ridge_frame", out("ridge_frame_outside_d074"), "synfing", "anguli"),
        read("ridge_robust_d077", f"{R}.ridge_robust", out("ridge_robust_d077")),
        read("ridge_robust_shared_e117", f"{R}.ridge_robust", out("ridge_robust_shared_e117"), "shared"),
        read("winding_full", f"{M}.winding", out("winding_full")),
        read("repeat_d", f"{R}.repeat_d", out("repeat_d")),
        read("edge_cells", f"{R}.edge_cells", out("edge_cells")),
    ]
    for arm in ("zp", "zpp", "synfing"):
        for seed in (1, 2, 3):
            name = f"tstr_d082_{arm}_{seed}_e200"
            deferred.append(read(name, f"{R}.tstr", str(G / "tstr_d079"), arm, str(seed), out(name), env={"EPOCHS": "200"}, stage="matcher"))
    return deferred


def compare(step):
    """Exact comparison, recorded, and the count of values that differ."""
    expected = EXPECTED / f"{step['expected']}.json"
    command = [sys.executable, str(ROOT / "scripts/compare_results.py"), str(expected), out(step["name"])] + (["--partial"] if step["partial"] else [])
    done = subprocess.run(command, capture_output=True, text=True)
    return done.returncode, done.stdout.strip()


def run(step, jobs):
    env = {**os.environ, "READ_JOBS": str(jobs), "SIGNED_JOBS": str(jobs), "JOBS": str(jobs), "OMP_NUM_THREADS": "1", **step["env"]}
    log = VERIFY / f"{step['name'].replace(' ', '_')}.log"
    started = time.time()
    if "command" in step:
        if Path(step["done"]).exists():
            print(f"have    {step['name']}", flush=True)
            return
        command = step["command"]
    else:
        if Path(out(step["name"])).exists():
            code, text = compare(step)
            print(f"have    {step['name']}: {text.splitlines()[0] if text else ''}", flush=True)
            return
        command = [sys.executable, "-m", step["module"], *step["args"]]
    print(f"start   {step['name']}", flush=True)
    with open(log, "w") as stream:
        try:
            code = subprocess.run(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=12 * HOURS).returncode
        except subprocess.TimeoutExpired:
            code = -1
    minutes = (time.time() - started) / 60
    result = {"step": step["name"], "exit": code, "minutes": round(minutes, 1)}
    if code == 0 and "module" in step:
        result["same"], result["comparison"] = compare(step)
    with open(VERIFY / "summary.jsonl", "a") as stream:
        stream.write(json.dumps(result) + "\n")
    print(f"done    {step['name']}: exit {code}, {minutes:.0f} min, {result.get('comparison', '').splitlines()[0] if result.get('comparison') else ''}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("quick", "generate", "reads", "matcher"))
    parser.add_argument("--only", help="run just the step of this name")
    parser.add_argument("--list", action="store_true", help="list the steps and stop")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    VERIFY.mkdir(parents=True, exist_ok=True)
    chosen = [s for s in steps() if s["stage"] == args.stage and args.only in (None, s["name"])]
    if args.list:
        print("\n".join(s["name"] for s in chosen))
        return
    paths.require_fvc()
    for step in chosen:
        run(step, args.jobs)


if __name__ == "__main__":
    main()
