# Calibration

How the instance distribution was designed against measurement, including two
design iterations that the metrics rejected. Reproduce any number:
`python calibration/measure.py --n 300`.

## Metrics

- **greedy-infeasible rate** — fraction of instances where ramp-blind
  per-period merit-order dispatch violates a ramp limit. The headline
  difficulty metric: these instances defeat exactly the reasoning the v1
  sibling environment rewards.
- **cost-of-ramps premium** — `(LP optimum − ramp-blind cost bound) / bound`
  on the constrained subset. If this is ~0, ramps are a feasibility nuisance
  with no economic content and the optimality reward is nearly free.
- **rejection rate** — fraction of raw draws that are LP-infeasible. High
  rejection biases the accepted sample toward easy survivors.

## Design iterations (kept as the decision record)

| Design | g-infeas | med premium | rejection | verdict |
|---|---|---|---|---|
| 1. Gradual linear climb, moderate ramps | 65% | **0.4%** | 7% | rejected: ramps bind but cost nothing — baseload never actually moves |
| 2. Duck curve (deep valley, steep climb), slow cheap tiers | 94% | 2.6% | **64%** | rejected: curve often physically unfollowable — massive survivor bias |
| 3. Climb constructed at 55–95% of the drawn fleet's aggregate ramp | 95% | 5.5% | 9% | feasible by construction, economically sharp — but no easy tier |
| **4. + 25% mild days (shipped)** | **69%** | **5.4%** | **5%** | mixed difficulty, sharp premium, unbiased sample |

The load-bearing insight from iteration 1: a smooth climb makes greedy
*infeasible* (its per-period allocations jump around) without making ramps
*expensive* — baseload just sits at its ceiling. The premium only appears when
a deep valley forces slow cheap units down toward Pmin and the subsequent
climb outruns them, stranding cheap capacity below a sustained peak. That is
the duck-curve / evening-ramp structure grid operators actually manage.

Iteration 3's fix for iteration 2's rejection rate is fleet-relative
construction: peak = valley + climb_periods × fleet_ramp × tightness(0.55–0.95).
The *system* can always follow the curve; the merit-blind *allocation* usually
cannot. Slow fleets get capped peaks — also true of real systems.

## Shipped configuration (v0.1.0)

`DEFAULT_GEN_CONFIG` in `multiperiod_dispatch.py`. Validation on dataset seeds
0–199: 0/200 dual-formulation mismatches (worst rel gap 8.8e-16), 149/200
(74%) greedy-infeasible, 5.4% median premium, all 7 attack gates pass.

## Stylizations stated, not hidden

1. **Tier ramp rates** (baseload 5–15%, mid 15–35%, peaker 50–100% of Pmax per
   period): with 2–4 h periods implied by a 5–8 period day, these bracket real
   per-interval maneuverability (coal/nuclear at the slow end, CTs at the
   fast end). The cheap-slow/fast-expensive correlation is the point; exact
   levels are chosen for signal, not from unit test data.
2. **All units committed for the whole horizon** — no startup/shutdown. True
   unit commitment (binary on/off + startup costs, MILP ground truth) is the
   v0.2 milestone; this environment isolates the continuous ramp-coupling
   skill first.
3. **First period has no prior-state constraint** (no initial condition). Adds
   one degree of freedom the LP and the model share equally.
4. **~74% hard instances vs real grids' mostly-mild days**: deliberate
   oversampling of the skill-relevant region, same rationale as the sibling
   environments.
