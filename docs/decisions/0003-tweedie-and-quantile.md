# 0003: Tweedie for the expected demand, a quantile model for the upper bound

**Status:** accepted

## Context
Daily store x SKU sales are small counts: many zeros, occasional spikes. Inventory decisions
need more than one number: they need to know how *high* demand could plausibly go.

## Decision
Train two LightGBM models per group:

* **`mean`**: Tweedie objective (variance power 1.2). Tweedie is designed for
  non-negative data with many exact zeros and a long right tail, and predicts the expected
  units per day. It is what the M5 winners used.
* **`upper`**: quantile objective at 0.9 (P90): "on 9 days out of 10 demand stays below this".
  Its gap to the mean gives a per-day uncertainty that the replenishment policy turns into safety stock.

Rounds are chosen by early stopping on the last 28 days, then both models are refit on all
data with that many rounds, so the most recent weeks are never thrown away.

## Alternatives considered
* **Squared error (plain regression)**: pulled around by spikes and can predict negative values.
* **Poisson objective**: fine for the mean, but assumes variance = mean; retail demand is more variable.
* **A full predictive distribution** (several quantiles, or a parametric NB head): more
  informative, but more models and harder to explain. P90 is what the policy needs today.

## Consequences
* Mean and quantile models are trained independently, so the P90 can in principle cross the
  mean for very sparse series. The policy clips the per-day uncertainty at zero.
* Backtest P90 coverage on the small preset is 92% (target 90%): well calibrated.
