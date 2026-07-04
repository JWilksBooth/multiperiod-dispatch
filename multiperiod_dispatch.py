"""multiperiod-dispatch: multi-period economic dispatch with ramp limits (UC-lite).

Third environment in the power-systems vertical (after economic-dispatch and
dcopf-grid-verifiers). The physics axis here is TIME: unit output changes
between consecutive periods are limited by ramp rates, and the fleet is drawn
so that cheap units are slow and fast units are expensive (real merit-stack
texture). Per-period merit-order reasoning therefore fails on a measured
fraction of instances: the cheap baseload cannot climb into the peak fast
enough, so cost-optimal dispatch must pre-position it early (out of merit) or
lean on expensive fast peakers.

Validation discipline (same as the sibling environments):
- Ground truth: LP over all periods (scipy linprog, HiGHS dual simplex).
- Cross-check: independently assembled formulation (different variable
  ordering, separately written constraint builder) solved with HiGHS interior
  point. Both must agree on the objective across the dataset.
- Rewards hard-gated on feasibility; violations inside the grading tolerance
  are settled at a 2x imbalance penalty price; cost below the optimum is
  penalized symmetrically (a feasible dispatch cannot beat the LP).
- Red-teamed: NaN/Infinity literals, oversized integers, tolerance-rent,
  infeasible-cheap, and garbage all score 0 (tests/test_validation.py).
"""

from __future__ import annotations

import json
import math
import random
import re

__version__ = "0.1.2"

DEFAULT_NUM_EXAMPLES = 300

# --- Calibration knobs (measured; see calibration/measure.py) ----------------
# Tier tuple: (probability, cost_lo, cost_hi, pmin_frac_lo, pmin_frac_hi,
#              ramp_frac_lo, ramp_frac_hi). Ramp is MW of allowed output change
# per period as a fraction of p_max. Cheap-and-slow vs fast-and-expensive is
# the load-bearing structure: it is what makes per-period merit order fail.
TIERS = (
    (0.40, 15.0, 30.0, 0.35, 0.50, 0.05, 0.15),   # baseload: cheap, inflexible
    (0.40, 35.0, 60.0, 0.25, 0.40, 0.15, 0.35),   # mid-merit
    (0.20, 70.0, 130.0, 0.00, 0.15, 0.50, 1.00),  # peaker: fast, expensive
)
# Load profile is duck-curve shaped in explicit phases: valley plateau ->
# steep 1-2 period climb -> peak plateau -> shoulder decline. The steep climb
# out of a deep valley is what strands cheap slow capacity below the peak
# (measured: a gradual linear climb yields ramp *infeasibility* but a near-zero
# cost premium, because baseload never actually has to move).
DEFAULT_GEN_CONFIG: dict = {
    "periods_range": (5, 8),
    "n_units_range": (3, 5),
    "p_max_range": (50.0, 300.0),
    "valley_frac": (0.32, 0.45),   # deep valley: pushes slow units toward Pmin
    "peak_frac": (0.80, 0.95),     # peak ceiling as fraction of capacity
    "shoulder_frac": (0.55, 0.70), # end-of-horizon load fraction
    "climb_periods": (1, 2),       # steepness: valley->peak in this many steps
    "climb_tightness": (0.55, 0.95),  # climb rate as fraction of FLEET ramp capability
    "mild_day_prob": 0.25,         # P(near-flat profile): the easy tier — ramps barely bind
    "mild_level_frac": (0.50, 0.70),
    "noise_frac": 0.02,            # per-period multiplicative load noise
}
# The peak is capped at valley + climb_periods * fleet_ramp * tightness: the
# system can always physically follow the curve (feasible by construction, so
# the rejection loop does not bias the sample), but at 55-95% of aggregate ramp
# capability the merit-blind ALLOCATION across units frequently cannot — cheap
# slow units get stranded below the peak. Slow fleets get capped peaks, which
# is also true of real systems.


def _draw_unit(rng: random.Random, name: str, p_max_range) -> dict:
    u = rng.random()
    acc = 0.0
    tier = TIERS[-1]
    for t in TIERS:
        acc += t[0]
        if u < acc:
            tier = t
            break
    _, c_lo, c_hi, pm_lo, pm_hi, r_lo, r_hi = tier
    p_max = round(rng.uniform(*p_max_range), 1)
    return {
        "name": name,
        "cost": round(rng.uniform(c_lo, c_hi), 2),
        "p_min": round(rng.uniform(pm_lo, pm_hi) * p_max, 1),
        "p_max": p_max,
        "ramp": round(rng.uniform(r_lo, r_hi) * p_max, 1),
    }


def build_instance(rng: random.Random, **cfg) -> dict:
    """One raw draw (no feasibility check). Exposed for the calibration harness."""
    p = {**DEFAULT_GEN_CONFIG, **cfg}
    T = rng.randint(*p["periods_range"])
    G = rng.randint(*p["n_units_range"])
    units = [_draw_unit(rng, f"G{i+1}", p["p_max_range"]) for i in range(G)]
    # guarantee at least one fast unit, else load swings reject almost everything
    while max(u["ramp"] / u["p_max"] for u in units) < 0.4:
        units[-1] = _draw_unit(rng, units[-1]["name"], p["p_max_range"])

    cap = sum(u["p_max"] for u in units)
    floor = sum(u["p_min"] for u in units)
    fleet_ramp = sum(u["ramp"] for u in units)

    # mild day: near-flat profile, the easy tier of the difficulty mix
    if rng.random() < p["mild_day_prob"]:
        level = rng.uniform(*p["mild_level_frac"]) * cap
        loads = []
        for _ in range(T):
            base = level * (1.0 + rng.uniform(-p["noise_frac"], p["noise_frac"]))
            loads.append(round(min(max(base, floor * 1.05), 0.95 * cap), 1))
        return {"periods": T, "units": units, "loads_mw": loads}

    valley = rng.uniform(*p["valley_frac"]) * cap
    shoulder = rng.uniform(*p["shoulder_frac"]) * cap

    # phase lengths: valley plateau, steep climb, peak plateau, shoulder decline
    climb = rng.randint(*p["climb_periods"])
    tightness = rng.uniform(*p["climb_tightness"])
    peak = min(rng.uniform(*p["peak_frac"]) * cap,
               valley + climb * fleet_ramp * tightness)
    peak = max(peak, valley * 1.1)  # degenerate flat draws: keep some shape
    v_len = rng.randint(1, max(1, T - climb - 2))
    pk_len = rng.randint(1, max(1, min(2, T - v_len - climb)))

    base_profile = []
    for _ in range(v_len):
        base_profile.append(valley)
    for s in range(1, climb + 1):
        base_profile.append(valley + (peak - valley) * s / climb)
    for _ in range(pk_len):
        base_profile.append(peak)
    remaining = T - len(base_profile)
    for s in range(1, remaining + 1):
        base_profile.append(peak + (shoulder - peak) * s / max(1, remaining))
    base_profile = base_profile[:T]

    loads = []
    for base in base_profile:
        base *= 1.0 + rng.uniform(-p["noise_frac"], p["noise_frac"])
        loads.append(round(min(max(base, floor * 1.05), 0.95 * cap), 1))

    return {"periods": T, "units": units, "loads_mw": loads}


def generate_instance(seed: int, max_attempts: int = 200, **cfg) -> dict:
    """Deterministic per seed; infeasible draws rejected and resampled from a
    stream offset by (seed + 1) so retries never replay another seed's draws."""
    rng = random.Random(seed)
    for attempt in range(max_attempts):
        inst = build_instance(rng, **cfg)
        if solve_lp(inst) is not None:
            return inst
        rng = random.Random((seed + 1) * 1_000_003 + attempt)
    raise RuntimeError(f"no feasible instance for seed {seed}")


# ---------------- ground truth: full-horizon LP ------------------------------

def solve_lp(inst: dict) -> dict | None:
    """Primary: variables ordered x[g*T + t], HiGHS dual simplex."""
    from scipy.optimize import linprog
    import numpy as np

    T, units, loads = inst["periods"], inst["units"], inst["loads_mw"]
    G = len(units)
    n = G * T
    c = np.zeros(n)
    for g, u in enumerate(units):
        c[g * T:(g + 1) * T] = u["cost"]

    A_eq = np.zeros((T, n))
    for t in range(T):
        for g in range(G):
            A_eq[t, g * T + t] = 1.0
    b_eq = np.array(loads, float)

    rows = 2 * G * (T - 1)
    A_ub = np.zeros((rows, n))
    b_ub = np.zeros(rows)
    r = 0
    for g, u in enumerate(units):
        for t in range(T - 1):
            A_ub[r, g * T + t + 1] = 1.0; A_ub[r, g * T + t] = -1.0
            b_ub[r] = u["ramp"]; r += 1
            A_ub[r, g * T + t + 1] = -1.0; A_ub[r, g * T + t] = 1.0
            b_ub[r] = u["ramp"]; r += 1

    bounds = []
    for u in units:
        bounds += [(u["p_min"], u["p_max"])] * T

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs-ds")
    if not res.success:
        return None
    x = res.x
    dispatch = [[round(float(x[g * T + t]), 6) for g in range(G)] for t in range(T)]
    return {"dispatch_mw": dispatch, "cost": float(res.fun)}


def solve_lp_crosscheck(inst: dict) -> dict | None:
    """Independently assembled: variables ordered x[t*G + g], built from the
    constraint definitions from scratch, HiGHS interior point."""
    from scipy.optimize import linprog
    import numpy as np

    T, units, loads = inst["periods"], inst["units"], inst["loads_mw"]
    G = len(units)
    n = T * G
    c = np.array([units[g]["cost"] for t in range(T) for g in range(G)])

    A_eq = np.zeros((T, n))
    for t in range(T):
        A_eq[t, t * G:(t + 1) * G] = 1.0
    b_eq = np.array(loads, float)

    A_ub, b_ub = [], []
    for t in range(T - 1):
        for g, u in enumerate(units):
            up = np.zeros(n); up[(t + 1) * G + g] = 1.0; up[t * G + g] = -1.0
            A_ub.append(up); b_ub.append(u["ramp"])
            dn = np.zeros(n); dn[(t + 1) * G + g] = -1.0; dn[t * G + g] = 1.0
            A_ub.append(dn); b_ub.append(u["ramp"])

    bounds = [(units[g]["p_min"], units[g]["p_max"]) for t in range(T) for g in range(G)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ipm")
    if not res.success:
        return None
    return {"cost": float(res.fun)}


def greedy_dispatch(inst: dict) -> list[list[float]]:
    """Per-period merit-order fill, ramp-blind: what naive reasoning produces."""
    out = []
    for t in range(inst["periods"]):
        load = inst["loads_mw"][t]
        row = {u["name"]: u["p_min"] for u in inst["units"]}
        remaining = load - sum(row.values())
        for u in sorted(inst["units"], key=lambda u: u["cost"]):
            if remaining <= 1e-9:
                break
            take = min(remaining, u["p_max"] - u["p_min"])
            row[u["name"]] += take
            remaining -= take
        out.append([row[u["name"]] for u in inst["units"]])
    return out


def dispatch_cost(inst: dict, X: list[list[float]]) -> float:
    return sum(u["cost"] * X[t][g]
               for t in range(inst["periods"])
               for g, u in enumerate(inst["units"]))


# ---------------- parsing and rewards ----------------------------------------

def _is_finite_number(v) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def parse_dispatch(completion: str, inst: dict) -> list[list[float]] | None:
    """Last valid {"dispatch_mw": [[...], ...]} wins. Rejects non-finite values
    and wrong shape (T rows x G columns)."""
    T, G = inst["periods"], len(inst["units"])
    for raw in reversed(re.findall(r'\{[^{}]*"dispatch_mw"[^{}]*\}', completion, re.DOTALL)):
        try:
            obj = json.loads(raw)
        # ValueError covers JSONDecodeError AND CPython's >4300-digit int limit;
        # RecursionError covers deeply nested brackets. Both must skip, not crash.
        except (ValueError, RecursionError):
            continue
        rows = obj.get("dispatch_mw")
        if (isinstance(rows, list) and len(rows) == T
                and all(isinstance(r, list) and len(r) == G
                        and all(_is_finite_number(v) for v in r) for r in rows)):
            return [[float(v) for v in r] for r in rows]
    return None


def check_feasibility(inst: dict, X: list[list[float]],
                      tol_mw: float = 0.5) -> tuple[bool, list[str]]:
    """Positive-proof gate: balance per period, unit bounds, ramp limits."""
    T, units, loads = inst["periods"], inst["units"], inst["loads_mw"]
    v: list[str] = []
    for t in range(T):
        if not abs(sum(X[t]) - loads[t]) <= tol_mw:
            v.append(f"period {t}: balance {sum(X[t]):.1f} vs load {loads[t]:.1f}")
        for g, u in enumerate(units):
            if not (u["p_min"] - tol_mw <= X[t][g] <= u["p_max"] + tol_mw):
                v.append(f"period {t} {u['name']}: {X[t][g]:.1f} outside range")
    for t in range(T - 1):
        for g, u in enumerate(units):
            if not abs(X[t + 1][g] - X[t][g]) <= u["ramp"] + tol_mw:
                v.append(f"{u['name']} t{t}->t{t+1}: ramp {abs(X[t+1][g]-X[t][g]):.1f} > {u['ramp']}")
    return len(v) == 0, v


def reward_format(completion: str, inst: dict) -> float:
    return 1.0 if parse_dispatch(completion, inst) is not None else 0.0


def reward_feasibility(completion: str, inst: dict) -> float:
    X = parse_dispatch(completion, inst)
    if X is None:
        return 0.0
    ok, _ = check_feasibility(inst, X)
    return 1.0 if ok else 0.0


def reward_optimality(completion: str, inst: dict,
                      optimal_cost: float | None = None) -> float:
    """exp(-5|gap|), gated on feasibility. Residual violations inside the
    grading tolerance are settled at a penalty price and below-optimum cost is
    penalized symmetrically, so tolerance abuse strictly loses to honest optimal
    dispatch.

    Two hardenings over a naive per-period penalty: (a) imbalance is priced
    symmetrically (abs), because over-generating one unit within tolerance to
    dodge a downstream ramp is as much a cheat as under-serving; (b) the penalty
    scales with the horizon T. A single within-tolerance violation can buy an
    advantageous position that pays off across all remaining periods, so it must
    cost more than T periods of the most expensive unit — 2*T*c_max gives a
    provable 2x margin over any multi-period gain a 1 MW slack could purchase."""
    X = parse_dispatch(completion, inst)
    if X is None:
        return 0.0
    ok, _ = check_feasibility(inst, X)
    if not ok:
        return 0.0
    if optimal_cost is None:
        sol = solve_lp(inst)
        if sol is None:
            return 0.0
        optimal_cost = sol["cost"]
    T, units = inst["periods"], inst["units"]
    penalty = 2.0 * T * max(u["cost"] for u in units)
    viol = 0.0
    for t in range(T):
        viol += abs(sum(X[t]) - inst["loads_mw"][t])  # symmetric: over- and under-generation
        for g, u in enumerate(units):
            viol += max(0.0, u["p_min"] - X[t][g]) + max(0.0, X[t][g] - u["p_max"])
    for t in range(T - 1):
        for g, u in enumerate(units):
            viol += max(0.0, abs(X[t + 1][g] - X[t][g]) - u["ramp"])
    cost = dispatch_cost(inst, X) + viol * penalty
    if optimal_cost <= 0:
        return 1.0 if cost <= 1e-6 else 0.0
    gap = abs(cost - optimal_cost) / optimal_cost
    return math.exp(-5.0 * gap)


REWARD_WEIGHTS = {"reward_format": 0.10, "reward_feasibility": 0.35,
                  "reward_optimality": 0.55}


# ---------------- prompt + verifiers entry point ------------------------------

def instance_to_prompt(inst: dict) -> str:
    units_txt = "\n".join(
        f"  {u['name']}: cost ${u['cost']}/MWh, output range [{u['p_min']}, {u['p_max']}] MW, "
        f"max output change between consecutive periods {u['ramp']} MW"
        for u in inst["units"])
    loads_txt = "\n".join(f"  Period {t+1}: {mw} MW"
                          for t, mw in enumerate(inst["loads_mw"]))
    G = len(inst["units"])
    example_row = ", ".join(f"<{u['name']} MW>" for u in inst["units"])
    return f"""You are a power system operator scheduling generation across {inst['periods']} consecutive time periods.

Online units (all committed for the whole horizon; note that cheaper units ramp slowly and fast units are expensive):
{units_txt}

System load by period:
{loads_txt}

Find the schedule that minimizes total cost ($) such that, in every period,
total generation equals that period's load and every unit is within its output
range — AND no unit's output changes by more than its ramp limit between one
period and the next. The first period has no prior-period constraint.

Report each MW value to at least one decimal place. Constraints are verified
with a +/-0.5 MW tolerance, but imbalance or limit violations inside that
tolerance are penalized in the cost scoring, so target exact feasibility.

Respond with your final answer as JSON on the last line, exactly in this format
(one inner list per period, {G} values per list, unit order as given):
{{"dispatch_mw": [[{example_row}], ... one list per period ...]}}"""


def build_dataset(num_examples: int = DEFAULT_NUM_EXAMPLES, seed_offset: int = 0):
    rows = []
    for i in range(num_examples):
        inst = generate_instance(seed_offset + i)
        sol = solve_lp(inst)
        rows.append({
            "question": instance_to_prompt(inst),
            "answer": str(round(sol["cost"], 2)),
            "info": {"instance": inst, "optimal_cost": sol["cost"],
                     "optimal_dispatch": sol["dispatch_mw"]},
        })
    return rows


def load_environment(num_examples: int = DEFAULT_NUM_EXAMPLES,
                     seed_offset: int = 0, **kwargs):
    """seed_offset enables disjoint train/eval datasets."""
    import verifiers as vf
    from datasets import Dataset

    dataset = Dataset.from_list(build_dataset(num_examples, seed_offset=seed_offset))

    def _text(completion):
        if isinstance(completion, str):
            return completion
        parts = []
        for m in completion:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for p in content:
                    t = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
        return " ".join(parts)

    def fmt(completion, info, **kw):
        return reward_format(_text(completion), info["instance"])

    def feas(completion, info, **kw):
        return reward_feasibility(_text(completion), info["instance"])

    def opt(completion, info, **kw):
        return reward_optimality(_text(completion), info["instance"],
                                 optimal_cost=info["optimal_cost"])

    rubric = vf.Rubric(funcs=[fmt, feas, opt],
                       weights=list(REWARD_WEIGHTS.values()))
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
