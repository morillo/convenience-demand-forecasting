# 0009: Order-up-to policy with an empirically calibrated safety stock

**Status:** accepted

## Context
Forecasts create value only when they change orders. Store managers and auditors need a rule
they can follow and check by hand.

## Decision
A periodic-review **order-up-to** policy per store x SKU:

* **Protection period** P = lead time + review period (fresh: review period only, because
  leftovers are written off nightly).
* **Expected demand** over P = sum of the daily mean forecasts.
* **Uncertainty** per day = (P90 - mean) / 1.28; over P = square root of the summed variances.
* **Safety stock** = z(service level) x uncertainty x **aisle safety factor**.
* **Order quantity** = max(0, ceil(expected demand + safety stock - on hand - on order)).

The textbook formula assumes forecast errors on consecutive days are independent. They are not,
so in the backtest it delivered 91% of cycles without stockout against a 95% target. The
**aisle safety factor** corrects that empirically: for each past cycle, take the multiplier that
would have just covered true demand, and use its 95th percentile per aisle (bounded to 0.5-4x). It is
validated **out of sample**: learned on the earlier cut-offs, tested on the last one (95.3% achieved).
The production forecast uses factors learned on all cut-offs (`gold.safety_stock_calibration`).

## Alternatives considered
* **Fixed "days of cover"**: what many stores do today; it ignores uncertainty and seasonality.
* **Newsvendor / simulation-optimised policies per SKU**: better economics with known costs of
  waste and lost sales, but those costs are not in the data yet. That is the natural next step.

## Consequences
* The policy is a one-cycle approximation (it treats the order as available for the whole
  period); it is used to compare forecasts on equal terms, not to promise exact fill rates.
* With real cost data, the service level can be set per category (higher on traffic drivers
  like coffee and water, lower on slow, expensive items).
