# 0004: One model for all horizons: every sales feature lags by at least the horizon

**Status:** accepted

## Context
We forecast H = 14 days ahead from a cut-off day T. If a feature for target day t used sales
from day t-1, the model would need sales we do not have yet when forecasting t = T+5. That is
the classic *leakage* bug: great backtest, useless in production.

## Decision
Every sales-based feature looks back **at least H days**: lags of 14, 21, 28, 35, 42 and 364
days, and rolling averages / standard deviations / zero-shares over windows that end 14 days
before the target day. Information that is genuinely known in advance is used for the target
day itself: calendar and holidays, paydays, planned promotions and discounts, the list price,
the weather *forecast* (never the observed weather), and days since launch.

Because the rule holds for every row, the same feature code serves history and future, and
one model covers every day of the horizon. A test checks `lag_14` against the sales 14 days
earlier, row by row.

## Alternatives considered
* **Recursive forecasting** (predict day 1, feed it back to predict day 2...): uses the
  freshest data, but errors compound and it needs 14 sequential scoring passes.
* **One model per horizon** (14 models, each with its own minimum lag): most accurate for
  day 1, but 14x the training cost and 14x the models to monitor.

## Consequences
* Days 1-7 of the horizon do not use last week's sales. The backtest shows error is nearly flat
  across the 14 days, so the cost is small. Horizon-bucketed models (days 1-7 with a 7-day minimum lag,
  days 8-14 with 14) are the natural next step if the first week needs to be sharper.
* Changing `horizon_days` changes the lag set automatically (`features.lag_days`).
