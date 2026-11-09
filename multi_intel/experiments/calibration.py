"""First load test of the design: do cue-based confidences track real outcomes?

Runs the same task stream under habit-layer OFF and ON, over one or more seeds,
and reports:

  per seed   - routing accuracy, referee interventions, backend calls, choice mix
  pooled     - per-module accuracy with a Wilson 95% interval, Brier, ECE, bins
  spread     - min/mean/max of the headline metrics across seeds

Pooling across seeds is what makes the reasoning module's numbers readable: in a
single 100-task run it only wins a handful of ground-truth cycles, which is too
few for Brier or ECE to mean anything. Seeds are deliberately NOT allowed to share
cached model samples - that would correlate the runs and destroy the very spread
this is measuring.

Run:
  python -m experiments.calibration --n 300
  python -m experiments.calibration --backend ollama --model gemma3:12b-it-qat --n 100 --seeds 0,1,2,3,4
  python -m experiments.calibration --backend ollama --model qwen2.5:1.5b --n 200 --seeds 0,1,2 --out runs.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mi import (CodeModule, HabitLayer, MultiIntelSystem, Outcome, ReasoningModule,  # noqa: E402
                RetrievalModule, make_backend)
from experiments.tasks import CORPUS, make_tasks  # noqa: E402


# ---------------------------------------------------------------- statistics

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Honest about small n where the normal approx is not."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration(records: list[tuple[float, bool]], bins: int = 5):
    if not records:
        return None
    n = len(records)
    hits = sum(ok for _, ok in records)
    brier = sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in records) / n
    buckets = defaultdict(list)
    for c, ok in records:
        buckets[min(bins - 1, int(c * bins))].append((c, ok))
    ece = 0.0
    rows = []
    for i in sorted(buckets):
        xs = buckets[i]
        mc = sum(c for c, _ in xs) / len(xs)
        acc = sum(ok for _, ok in xs) / len(xs)
        ece += abs(mc - acc) * len(xs) / n
        rows.append((f"{i/bins:.1f}-{(i+1)/bins:.1f}", mc, acc, len(xs)))
    lo, hi = wilson(hits, n)
    return {"n": n, "accuracy": hits / n, "ci": (lo, hi), "brier": brier, "ece": ece, "bins": rows}


def spread(vals: list[float]) -> str:
    if not vals:
        return "-"
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{m:.3f}"
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return f"{m:.3f} ± {sd:.3f}  (min {min(vals):.3f}, max {max(vals):.3f})"


# ---------------------------------------------------------------- one run

def build(backend_name: str, habit_on: bool, seed: int, model: str | None):
    if backend_name == "mock":
        backend = make_backend("mock", seed=seed)
    elif backend_name == "ollama":
        # seed the model's sampling too, so a rerun of the same seed reproduces.
        # without this the seed only fixes the task stream and reruns differ.
        kw = {"seed": seed}
        if model:
            kw["model"] = model
        backend = make_backend("ollama", **kw)
    elif model:
        backend = make_backend(backend_name, model=model)
    else:
        backend = make_backend(backend_name)
    mods = [RetrievalModule(CORPUS), CodeModule(), ReasoningModule(backend, k=3)]
    return MultiIntelSystem(mods, habit=HabitLayer(openness=0.1, seed=seed, enabled=habit_on)), backend


def run_once(n: int, backend_name: str, seed: int, habit_on: bool, model: str | None,
             progress: bool = True):
    sys_, backend = build(backend_name, habit_on, seed, model)
    tasks = make_tasks(n, seed=seed)
    combined: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    cue_only: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    t0 = time.time()
    for i, s in enumerate(tasks, 1):
        ep = sys_.cycle(s)
        if progress and (i % 20 == 0 or i == len(tasks)):
            print(f"      ... {i}/{len(tasks)}  ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
        if ep.outcome == Outcome.UNKNOWN:
            continue
        ok = ep.outcome == Outcome.SUCCESS
        bid = next(b for b in ep.bids if b.module == ep.chosen)
        combined[ep.chosen].append((bid.confidence, ok))
        cue_only[ep.chosen].append((bid.cue_confidence, ok))
    stats = sys_.workspace.stats()
    stats["backend_calls"] = getattr(backend, "calls", None)
    stats["seconds"] = round(time.time() - t0, 1)
    return stats, combined, cue_only, sys_


# ---------------------------------------------------------------- reporting

def fmt_cal(c, indent="  "):
    if c is None:
        return indent + "(no ground-truth cycles)"
    lo, hi = c["ci"]
    head = (f"{indent}n={c['n']}  acc={c['accuracy']:.3f} [95% {lo:.2f}-{hi:.2f}]  "
            f"brier={c['brier']:.3f}  ece={c['ece']:.3f}")
    lines = [head]
    for label, mc, acc, k in c["bins"]:
        lines.append(f"{indent}  conf {label}: mean_conf={mc:.2f}  observed_acc={acc:.2f}  n={k}")
    return "\n".join(lines)


def report_condition(label: str, seeds: list[int], n: int, backend_name: str,
                     model: str | None, habit_on: bool, dump: str | None):
    print("=" * 74)
    print(f"habit layer {label}   backend={backend_name}{'/' + model if model else ''}"
          f"   n={n} x {len(seeds)} seed(s)")
    pooled_comb: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    pooled_cue: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    accs, ivs, calls = [], [], []
    last_sys = None
    raw = []
    for seed in seeds:
        print(f"  seed {seed}:", file=sys.stderr, flush=True)
        stats, comb, cue, sys_ = run_once(n, backend_name, seed, habit_on, model)
        last_sys = sys_
        accs.append(stats["accuracy"])
        ivs.append(float(stats["referee_interventions"]))
        if stats["backend_calls"] is not None:
            calls.append(float(stats["backend_calls"]))
        for m, xs in comb.items():
            pooled_comb[m].extend(xs)
        for m, xs in cue.items():
            pooled_cue[m].extend(xs)
        print(f"  seed {seed}: acc={stats['accuracy']:.3f}  interventions={stats['referee_interventions']}"
              f"  calls={stats['backend_calls']}  {stats['seconds']}s  chosen={stats['chosen_counts']}")
        raw.append({"seed": seed, "habit_on": habit_on, "stats": {k: v for k, v in stats.items()},
                    "combined": {m: xs for m, xs in comb.items()},
                    "cue_only": {m: xs for m, xs in cue.items()}})

    print(f"\n  across seeds: routing accuracy {spread(accs)}")
    print(f"                interventions    {spread(ivs)}")
    if calls:
        print(f"                backend calls    {spread(calls)}")

    for mod in sorted(pooled_comb):
        print(f"\n  [{mod}] pooled, combined confidence (cue + habit):")
        print(fmt_cal(calibration(pooled_comb[mod]), indent="    "))
        print(f"  [{mod}] pooled, cue-only confidence:")
        print(fmt_cal(calibration(pooled_cue[mod]), indent="    "))

    if habit_on and last_sys is not None:
        print("\n  learned cue weights (last seed):")
        for mod, w in last_sys.estimator.w.items():
            print(f"    {mod}: " + ", ".join(f"{k}={v:+.2f}" for k, v in w.items())
                  + f", bias={last_sys.estimator.b[mod]:+.2f}, updates={last_sys.estimator.n_updates[mod]}")
    if dump:
        Path(dump).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        print(f"\n  raw records appended -> {dump}")
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="tasks per seed")
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default=None, help="backend model name (e.g. gemma3:12b-it-qat)")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2,3,4")
    ap.add_argument("--out", default=None, help="write raw per-cycle records to this JSON file")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    all_raw = []
    for habit_on in (False, True):
        all_raw += report_condition("ON " if habit_on else "OFF", seeds, args.n,
                                    args.backend, args.model, habit_on, None)
    if args.out:
        Path(args.out).write_text(json.dumps(all_raw, ensure_ascii=False), encoding="utf-8")
        print(f"\nraw records -> {args.out}")


if __name__ == "__main__":
    main()
