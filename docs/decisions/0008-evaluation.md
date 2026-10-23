# 0008: Rolling-origin backtest against simple baselines, scored against true demand

**Status:** accepted

## Context
A forecast is only worth deploying if it beats what the business could do without it, on
data it has not seen, measured in a way the business cares about.

## Decision
* **Rolling-origin backtest**: 3 cut-off dates, 14 days apart, at the end of the history.
  For each cut-off the model is trained only on data up to that day and forecasts the next 14
  days. All cut-offs x groups train in parallel.
* **Baselines**, scored the same way: *same weekday last week* and *28-day moving average*.
* **Metrics** (volume-weighted, so best sellers count more): WAPE, bias, MAE, P90 pinball loss
  and coverage, at overall / cut-off / country / aisle / days-ahead level.
* **Censoring-aware scoring**: accuracy is computed on days without a stockout, *and* against
  the simulation's **true demand** on all days. The second is the honest number; the first
  is what a real chain could compute (see 0005 for why it is biased).
* **Business what-if**: each forecast drives the same replenishment policy (0009) and we
  measure cycle service level, fill rate, excess and shortage against true demand.

## Alternatives considered
* **One random train/test split**: leaks the future into training.
* **MAPE**: undefined on zero-sales days, which are a third of the data.
* **Scoring on observed sales only**: rewards models that learn the empty shelf.

## Consequences
* A backtest costs about 3x a training run; acceptable, and it is the evidence the business needs.
* True demand exists only because the data is simulated. In production, use categories with few stockouts
  and lost-sales estimates from store audits as the honest reference.
