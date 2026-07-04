"""Calibration harness: measure the difficulty distribution before claiming it.

Headline metric: greedy-infeasible rate — the fraction of instances where
ramp-blind per-period merit-order dispatch violates a ramp limit. Secondary:
the ramp cost premium (optimal cost vs the ramp-blind cost bound — how much
the ramp constraints cost), and the raw-draw rejection rate (sample-bias
indicator).

    python calibration/measure.py --n 300
"""

import sys, os, json, argparse, random
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multiperiod_dispatch import (
    generate_instance, build_instance, greedy_dispatch, dispatch_cost,
    solve_lp, check_feasibility,
)


def measure_instance(inst):
    lp = solve_lp(inst)
    if lp is None:
        return None
    g = greedy_dispatch(inst)
    ok, _ = check_feasibility(inst, g, tol_mw=1e-6)
    gcost = dispatch_cost(inst, g)
    prem = (lp["cost"] - gcost) / gcost if gcost > 1e-9 else 0.0
    return {"greedy_infeasible": not ok, "premium": max(0.0, prem),
            "T": inst["periods"], "G": len(inst["units"])}


def rejection_rate(cfg, k=400, seed_base=900_000):
    fails = 0
    for i in range(k):
        rng = random.Random(seed_base + i)
        if solve_lp(build_instance(rng, **cfg)) is None:
            fails += 1
    return fails / k


def measure_config(cfg, n):
    rows = [m for m in (measure_instance(generate_instance(s, **cfg))
                        for s in range(n)) if m]
    infeas = [r for r in rows if r["greedy_infeasible"]]
    prem = [r["premium"] for r in infeas]
    return {
        "n": len(rows),
        "greedy_infeasible_rate": len(infeas) / len(rows),
        "median_premium_constrained": median(prem) if prem else 0.0,
        "max_premium": max((r["premium"] for r in rows), default=0.0),
        "rejection_rate": rejection_rate(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=int(os.environ.get("N_INSTANCES", "300")))
    args = ap.parse_args()

    header = f"{'config':<40} | g-infeas | med prem | max prem | reject"
    print("\n  g-infeas = % instances where ramp-blind merit-order dispatch violates a ramp limit")
    print(f"\n{header}\n{'-' * len(header)}")

    results = {}
    base = measure_config({}, args.n)
    results["default"] = {"config": {}, "metrics": base}
    print(f"{'default (shipped)':<40} | {base['greedy_infeasible_rate']*100:6.1f}% "
          f"| {base['median_premium_constrained']*100:6.1f}% "
          f"| {base['max_premium']*100:6.1f}% | {base['rejection_rate']*100:5.1f}%")

    for climb in ((1, 2), (2, 2)):
        for tight in ((0.45, 0.85), (0.55, 0.95), (0.70, 1.00)):
            cfg = {"climb_periods": climb, "climb_tightness": tight}
            r = measure_config(cfg, args.n)
            key = f"climb={climb}_tight={tight}"
            results[key] = {"config": cfg, "metrics": r}
            print(f"{key:<40} | {r['greedy_infeasible_rate']*100:6.1f}% "
                  f"| {r['median_premium_constrained']*100:6.1f}% "
                  f"| {r['max_premium']*100:6.1f}% | {r['rejection_rate']*100:5.1f}%")

    out = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    with open(out, "w") as f:
        json.dump({"n_per_config": args.n, "results": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
