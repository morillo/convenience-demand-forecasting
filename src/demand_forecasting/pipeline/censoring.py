"""Demand unconstraining: what was demand on days the shelf ran empty?

On a stockout day we only know that demand was *at least* the units sold (``Y >= s``).
Dropping those days biases the model low, because the days that sell out are exactly the
high-demand days. Instead we fill them in with what demand probably was, given the model's
current estimate ``lam`` and the fact that it was at least ``s`` (an EM algorithm,
docs/decisions/0005-stockout-censoring.md):

* for the mean model: the conditional expectation ``E[Y | Y >= s]``,
* for the upper-quantile model: a random draw from the same truncated distribution, so the
  model still sees realistic spread.

Demand is approximated as Poisson(lam) for this step. For a Poisson variable,
``E[Y * 1{Y >= s}] = lam * P(Y >= s - 1)``, which gives the closed form used below.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats


def poisson_sf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """P(Y >= k) for Y ~ Poisson(lam); equals 1 for k <= 0."""
    k = np.asarray(k, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)
    return np.where(k <= 0, 1.0, special.gammainc(np.maximum(k, 1.0), lam))


def conditional_mean(sold: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """E[Y | Y >= sold] for Y ~ Poisson(lam). Never below ``sold``."""
    sold = np.asarray(sold, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)
    tail = poisson_sf(sold, lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = lam * poisson_sf(sold - 1, lam) / tail
    # Far in the tail (sold >> lam) the ratio underflows; demand was then very likely ~sold.
    expected = np.where(tail > 1e-12, expected, sold)
    return np.maximum(expected, sold)


def sample_truncated(sold: np.ndarray, lam: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw of Y ~ Poisson(lam) conditioned on Y >= sold (inverse-CDF sampling)."""
    sold = np.asarray(sold, dtype=float)
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)
    below = 1.0 - poisson_sf(sold, lam)  # P(Y <= sold - 1)
    u = below + rng.random(len(sold)) * (1.0 - below)
    draw = stats.poisson.ppf(np.clip(u, 0.0, 1.0 - 1e-12), lam)
    return np.maximum(np.nan_to_num(draw, nan=0.0, posinf=0.0), sold)
