# 0005: Treat stockout days as censored demand and unconstrain them with EM

**Status:** accepted (supersedes the first version of this decision, which dropped stockout days)

## Context
When the shelf is empty, sales are lower than demand: the data is **censored**. A model
trained on raw sales learns to forecast the empty shelf and keeps it empty. Our data has no
"lost sales" column; real chains rarely do. We infer stockouts as "closing stock = 0".

## Decision
1. **Detect**: `is_stockout = on_hand_close == 0` (in the silver panel).
2. **Features**: sales on stockout days are blanked before computing lags and rolling statistics.
3. **Training**: unconstrain demand with an **EM algorithm**. Fit the mean model on clean
   days; on stockout days replace sales `s` with `E[Y | Y >= s]` under the model's own
   prediction (Poisson approximation, closed form); refit on all days; repeat
   (`model.censoring_em_iterations = 2`). The P90 model trains on random draws from the same
   truncated distribution, so it still sees realistic spread.
4. **Evaluation**: also score against the simulation's *true* demand, which shows the bias directly.

## Why we changed it
The first version simply **dropped** stockout days from training. The backtest against true
demand showed that version **under-forecast by 15% overall and by 35% on fresh food**, where
41% of days end with an empty shelf. Two reasons. The days that sell out are disproportionately
the high-demand days, so dropping them is selection bias. And for slow sellers, "closing stock = 0" also flags
days that simply sold the last unit. The safety-stock calibration was quietly compensating
(factor 4x on fresh). With EM the bias fell to **-4% overall and -8% on fresh**, WAPE against true
demand improved from 0.607 to 0.593, and the calibration factors dropped to 1.2-1.5x.

## Alternatives considered
* **Drop stockout days**: simple, and biased as described above.
* **Impute with a trailing average of clean days**: the same selection bias, because the clean days are the low ones.
* **A censored-likelihood custom objective** (Tobit-style): exact in principle, but a custom
  LightGBM objective for count data is harder to maintain and to explain. EM gets most of the gain.

## Consequences
* Training the mean model takes three fits instead of one (one clean fit plus two EM passes).
* The Poisson approximation understates the spread of very bursty items; the remaining bias is
  concentrated in slow movers (health & personal care -12%).
* Error measured only on clean days, against *observed* sales, now looks no better than the
  moving average. That metric is itself biased, because it only scores the low-demand days that
  did not sell out. Read accuracy against true demand (simulation) or, in production, on
  categories with few stockouts.
