from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from demand_forecasting.pipeline.censoring import conditional_mean, poisson_sf, sample_truncated


def brute_force_conditional_mean(sold: int, lam: float) -> float:
    ys = np.arange(0, 500)
    p = stats.poisson.pmf(ys, lam)
    keep = ys >= sold
    return float((ys[keep] * p[keep]).sum() / p[keep].sum())


@pytest.mark.parametrize(("sold", "lam"), [(0, 3.0), (3, 3.0), (2, 0.5), (25, 10.0), (1, 0.05)])
def test_conditional_mean_matches_brute_force(sold: int, lam: float) -> None:
    got = conditional_mean(np.array([sold]), np.array([lam]))[0]
    assert got == pytest.approx(brute_force_conditional_mean(sold, lam), rel=1e-6)


def test_conditional_mean_never_below_sales_even_far_in_tail() -> None:
    assert conditional_mean(np.array([200]), np.array([1.0]))[0] >= 200


def test_survival_function_matches_scipy() -> None:
    k, lam = np.array([0, 1, 3, 7]), np.full(4, 3.0)
    np.testing.assert_allclose(poisson_sf(k, lam), stats.poisson.sf(k - 1, lam))


def test_truncated_samples_respect_bound_and_mean() -> None:
    rng = np.random.default_rng(0)
    draws = sample_truncated(np.full(50_000, 3), np.full(50_000, 3.0), rng)
    assert draws.min() >= 3
    assert draws.mean() == pytest.approx(brute_force_conditional_mean(3, 3.0), rel=0.02)
