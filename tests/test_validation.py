"""Validation harness: dual-formulation cross-check + red-team gate tests.

1. Cross-check: primary LP (g-major assembly, HiGHS dual simplex) vs an
   independently assembled formulation (t-major, HiGHS interior point) must
   agree on the optimal objective on N instances.
2. Difficulty audit: fraction of instances where ramp-blind per-period
   merit-order dispatch violates a ramp limit (the naive-reasoning failure).
3. Gate tests: every attack class sealed in the sibling environments.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multiperiod_dispatch import (
    generate_instance, solve_lp, solve_lp_crosscheck, greedy_dispatch,
    dispatch_cost, check_feasibility, reward_format, reward_feasibility,
    reward_optimality,
)

N = int(os.environ.get("N_INSTANCES", "200"))
REL_TOL, ABS_TOL = 1e-3, 1.0


def run_crosscheck():
    mismatches, xc_fail, greedy_infeasible, worst = 0, 0, 0, 0.0
    for i in range(N):
        inst = generate_instance(i)
        lp = solve_lp(inst)
        assert lp is not None, f"seed {i}: generator produced infeasible instance"

        ok, _ = check_feasibility(inst, greedy_dispatch(inst), tol_mw=1e-6)
        if not ok:
            greedy_infeasible += 1

        xc = solve_lp_crosscheck(inst)
        if xc is None:
            xc_fail += 1
            continue
        gap = abs(xc["cost"] - lp["cost"])
        rel = gap / max(abs(lp["cost"]), 1e-9)
        worst = max(worst, rel)
        if gap > ABS_TOL and rel > REL_TOL:
            mismatches += 1
            print(f"  MISMATCH seed {i}: {lp['cost']:.2f} vs {xc['cost']:.2f}")

    print(f"cross-check: {N} instances | mismatches={mismatches} | "
          f"crosscheck non-converged={xc_fail} | worst rel gap={worst:.2e}")
    print(f"difficulty:  {greedy_infeasible}/{N} instances make ramp-blind merit-order "
          f"dispatch infeasible ({100*greedy_infeasible/N:.0f}%)")
    assert mismatches == 0, "formulation disagreement — do not ship"
    assert xc_fail <= 0.02 * N, f"crosscheck failed on {xc_fail}/{N} — vacuous pass risk"
    assert greedy_infeasible >= 0.2 * N, "too few ramp-constrained instances — steepen load climb"
    return greedy_infeasible


def run_gate_tests():
    # find an instance where greedy is ramp-infeasible (the interesting case)
    inst, opt = None, None
    for s in range(500):
        cand = generate_instance(seed=s)
        ok, _ = check_feasibility(cand, greedy_dispatch(cand), tol_mw=1e-6)
        if not ok:
            inst, opt = cand, solve_lp(cand)
            break
    assert inst is not None
    T, G = inst["periods"], len(inst["units"])

    def answer(X):
        return json.dumps({"dispatch_mw": [[round(v, 3) for v in row] for row in X]})

    # 1. optimal answer scores full credit
    a = answer(opt["dispatch_mw"])
    assert reward_format(a, inst) == 1.0
    assert reward_feasibility(a, inst) == 1.0
    assert reward_optimality(a, inst, opt["cost"]) > 0.99
    print("gate 1 (optimal): PASS")

    # 2. infeasible-cheap: the ramp-blind greedy dispatch itself
    g = greedy_dispatch(inst)
    a = answer(g)
    assert dispatch_cost(inst, g) <= opt["cost"] + 1e-6, "greedy should be cheap"
    assert reward_optimality(a, inst, opt["cost"]) == 0.0, "ramp-violating cheap answer scored"
    print("gate 2 (ramp-blind greedy, infeasible-cheap): PASS — optimality 0.0")

    # 3. garbage
    assert reward_format("forty-two", inst) == 0.0
    assert reward_optimality("forty-two", inst, opt["cost"]) == 0.0
    print("gate 3 (garbage): PASS")

    # 4. feasible suboptimal: solve with inverted costs (same feasible set)
    inst2 = json.loads(json.dumps(inst))
    mx = max(u["cost"] for u in inst2["units"])
    for u in inst2["units"]:
        u["cost"] = mx + 1.0 - u["cost"]
    sub = solve_lp(inst2)
    assert sub is not None
    a = answer(sub["dispatch_mw"])
    assert reward_feasibility(a, inst) == 1.0
    r = reward_optimality(a, inst, opt["cost"])
    assert 0.0 < r < 1.0, f"expected partial credit, got {r}"
    print(f"gate 4 (feasible suboptimal): PASS — partial credit {r:.3f}")

    # 5. NaN / Infinity / huge-int attacks
    row_of = lambda bad: "[" + ", ".join([bad] * G) + "]"
    for bad in ("NaN", "Infinity", "-Infinity", "9" * 400):
        a = '{"dispatch_mw": [' + ", ".join([row_of(bad)] * T) + "]}"
        assert reward_format(a, inst) == 0.0, f"{bad[:12]}: format gate broken"
        assert reward_feasibility(a, inst) == 0.0, f"{bad[:12]}: feasibility gate broken"
        assert reward_optimality(a, inst, opt["cost"]) == 0.0, f"{bad[:12]}: optimality gate broken"
    print("gate 5 (NaN/Infinity/huge-int attack): PASS — all rewards 0.0")

    # 6. tolerance-rent: shave 0.4 MW off the priciest unit in one period
    shaved = [row[:] for row in opt["dispatch_mw"]]
    victim = None
    for t in range(T):
        for g_i in sorted(range(G), key=lambda k: -inst["units"][k]["cost"]):
            if shaved[t][g_i] >= inst["units"][g_i]["p_min"] + 0.45:
                victim = (t, g_i)
                break
        if victim:
            break
    assert victim is not None
    shaved[victim[0]][victim[1]] -= 0.4
    a = answer(shaved)
    assert reward_feasibility(a, inst) == 1.0  # inside tolerance by design
    r_honest = reward_optimality(answer(opt["dispatch_mw"]), inst, opt["cost"])
    r_shaved = reward_optimality(a, inst, opt["cost"])
    assert r_shaved < 0.999 and r_shaved < r_honest, (
        f"tolerance-rent scored {r_shaved} vs honest {r_honest}")
    print(f"gate 6 (tolerance-rent attack): PASS — {r_shaved:.4f} < honest {r_honest:.4f}")


if __name__ == "__main__":
    run_gate_tests()
    run_crosscheck()
    print("ALL VALIDATION PASSED")
