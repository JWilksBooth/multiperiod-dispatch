# multiperiod-dispatch

Multi-period economic dispatch RL environment with ramp limits (unit-commitment-lite). Third environment in a power-systems vertical ([economic-dispatch](https://github.com/JWilksBooth/economic-dispatch) → [dcopf-grid-verifiers](https://github.com/JWilksBooth/dcopf-grid-verifiers) → this). The physics axis here is **time**: unit output changes between consecutive periods are limited by ramp rates, and the fleet is drawn with real merit-stack texture — **cheap units are slow, fast units are expensive**. Per-period merit-order reasoning is therefore provably infeasible on a measured **74% of instances**: the cheap baseload cannot climb out of the valley into the peak fast enough, so cost-optimal scheduling must anticipate the climb periods ahead of time or pay expensive fast units to carry the peak.

## Task

Given 5–8 consecutive periods, a duck-curve-shaped load profile, and 3–5 committed units (linear cost, [Pmin, Pmax], per-period ramp limit), output the full schedule as JSON: `{"dispatch_mw": [[unit MWs period 1], [period 2], ...]}` such that every period balances, every unit respects its range, and no unit's output changes faster than its ramp limit.

## Ground truth validation

Two independently assembled LP formulations must agree (unit-major assembly solved with HiGHS dual simplex vs a separately written time-major assembly solved with HiGHS interior point):

| Metric (200 instances, seeds 0–199) | Result |
|---|---|
| Objective mismatches (>0.1% rel and >$1 abs) | **0 / 200** |
| Worst relative objective gap | 8.8e-16 |
| Ramp-blind merit-order dispatch infeasible | **149 / 200 (74%)** |
| Median cost-of-ramps premium (constrained subset) | 5.4% |

Both formulations share scipy/HiGHS, so this is cross-implementation validation of the constraint assembly (the dominant error source), not independent-solver validation — stated plainly. See [CALIBRATION.md](CALIBRATION.md) for how the load-profile design was iterated against measured difficulty (a naive gradual climb produced ramp *infeasibility* without economic *cost* — the duck-curve construction fixes exactly that).

## Baseline results

50 instances (seeds 0–49), 1 rollout each, July 2026:

| Model | total | format | feasibility | optimality |
|---|---|---|---|---|
| claude-haiku-4-5 (8k tokens) | **0.242** | 0.980 | 0.200 | 0.135 |
| claude-opus-4-8 (16k tokens) | **0.890** | 0.920 | 0.900 | 0.879 |

The weak model writes near-perfectly parseable schedules and only 1 in 5
survives the ramp physics. The frontier model reaches 90% feasibility but its
optimality-when-feasible is 0.977 (vs 0.998 on the congestion sibling) — the
economics of pre-positioning slow units is harder than it looks even when the
constraints are satisfied. Across the vertical the weak-model ladder is
0.861 → 0.520 → 0.242 ([economic-dispatch](https://github.com/JWilksBooth/economic-dispatch)
→ [dcopf-grid-verifiers](https://github.com/JWilksBooth/dcopf-grid-verifiers) → this).

```bash
vf-eval multiperiod-dispatch -p anthropic -m claude-haiku-4-5-20251001 -n 50 -r 1 --max-tokens 8000 --save-results
vf-eval multiperiod-dispatch -p anthropic -m claude-opus-4-8 -n 50 -r 1 --max-tokens 16000 --save-results
```

## Rewards (weighted rubric)

| Reward | Weight | Description |
|---|---|---|
| `reward_format` | 0.10 | Parseable T×G JSON matrix, finite values only |
| `reward_feasibility` | 0.35 | Positive-proof check: per-period balance, unit bounds, all ramp transitions |
| `reward_optimality` | 0.55 | Gated on feasibility; `exp(-5 × |relative cost gap|)` vs LP optimum |

Anti-reward-hacking carried over from the sibling environments and regression-tested by 6 attack gates in `tests/test_validation.py`: the ramp-blind cheap schedule scores 0 (feasibility gate), `NaN`/`Infinity` literals and oversized integers are rejected at parse, and tolerance-rent (under-serving load inside the ±0.5 MW grading tolerance to beat the LP optimum) is settled at a 2× imbalance penalty price with below-optimum cost penalized symmetrically.

## Usage

```bash
pip install -e .

# validation harness (dual-formulation cross-check + attack gates)
N_INSTANCES=200 python tests/test_validation.py

# difficulty distribution / calibration sweep
python calibration/measure.py --n 300

vf-eval multiperiod-dispatch -m <model> -n 50
```

```python
import verifiers as vf
env = vf.load_environment("multiperiod-dispatch", num_examples=300)
# disjoint train/eval: load_environment(1000) + load_environment(200, seed_offset=1000)
```

## Instance generation

Tiered fleet (40% baseload $15–30/MWh with 5–15%/period ramps, 40% mid-merit $35–60 at 15–35%, 20% peakers $70–130 at 50–100%; at least one fast unit guaranteed). Load profiles are duck-curve shaped in explicit phases — valley plateau, steep 1–2 period climb, peak plateau, shoulder decline — with the climb rate constructed at 55–95% of the drawn fleet's aggregate ramp capability, so instances are physically followable by construction (draw rejection ~5%, no survivor bias) while merit-blind allocation frequently is not. 25% of instances are near-flat "mild days" (the easy tier). Fully deterministic per seed; ~10² instances/sec generation.

## Roadmap

- v0.2: startup costs + on/off commitment decisions (true UC, MILP ground truth)
- Sibling roadmap: N-1 contingency screening; LMP/nodal pricing verified against LP duals
