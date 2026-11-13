"""
Unit tests for statistical trend analysis module.
EarthTrend Explorer - NASA Space Apps 2026
"""

import math
import numpy as np
import pytest

from src.trend_analysis import (
    compute_autocorrelation,
    compute_linear_regression,
    compute_sens_slope,
    mann_kendall_test,
    seasonal_mann_kendall_test,
)


def test_mann_kendall_synthetic_increasing():
    """Known strong upward trend must yield p < 0.05 and trend == 'increasing'."""
    np.random.seed(42)
    x = np.arange(50)
    # y = 0.5 * x + noise
    y = 0.5 * x + np.random.normal(0, 1.0, size=50)

    res = mann_kendall_test(y, alpha=0.05)
    assert res["trend"] == "increasing"
    assert res["is_significant"] is True
    assert res["p_value"] < 0.001
    assert res["z_statistic"] > 3.0


def test_mann_kendall_synthetic_decreasing():
    """Known strong downward trend must yield p < 0.05 and trend == 'decreasing'."""
    np.random.seed(42)
    x = np.arange(50)
    y = -0.4 * x + np.random.normal(0, 1.0, size=50)

    res = mann_kendall_test(y, alpha=0.05)
    assert res["trend"] == "decreasing"
    assert res["is_significant"] is True
    assert res["p_value"] < 0.001
    assert res["z_statistic"] < -3.0


def test_mann_kendall_synthetic_no_trend():
    """Pure Gaussian white noise should yield no significant trend."""
    np.random.seed(12345)
    y = np.random.normal(20.0, 1.0, size=100)

    res = mann_kendall_test(y, alpha=0.05)
    assert res["trend"] == "no trend"
    assert res["is_significant"] is False
    assert res["p_value"] > 0.05


def test_sens_slope_recovery():
    """Verify Sen's slope recovers the true underlying slope under noise and outliers."""
    np.random.seed(42)
    x = np.arange(100)
    true_slope = 0.05
    y = true_slope * x + 10.0 + np.random.normal(0, 0.5, size=100)
    # Add extreme outliers
    y[10] += 50.0
    y[80] -= 40.0

    res = compute_sens_slope(y, x, alpha=0.05)
    estimated_slope = res["slope"]
    # Sen's slope is robust to outliers; should be close to true_slope
    assert abs(estimated_slope - true_slope) < 0.01
    assert res["slope_ci_lower"] <= estimated_slope <= res["slope_ci_upper"]


def test_linear_regression():
    x = np.arange(20, dtype=float)
    y = 2.0 * x + 5.0

    res = compute_linear_regression(y, x)
    assert math.isclose(res["slope"], 2.0, rel_tol=1e-5)
    assert math.isclose(res["intercept"], 5.0, rel_tol=1e-5)
    assert math.isclose(res["r_squared"], 1.0, rel_tol=1e-5)
    assert res["p_value"] < 1e-10


def test_seasonal_mann_kendall():
    """Verify Hirsch-Slack seasonal test removes monthly seasonality."""
    np.random.seed(42)
    # 20 years of monthly data = 240 months
    n_years = 20
    period = 12
    monthly_cycle = np.array([10, 12, 15, 20, 25, 30, 32, 31, 27, 22, 16, 11], dtype=float)
    repeated_cycle = np.tile(monthly_cycle, n_years)

    # Add trend of +0.1 per year (+0.1/12 per month)
    time_steps = np.arange(n_years * period)
    trend_component = 0.05 * time_steps
    noise = np.random.normal(0, 0.5, size=n_years * period)
    synthetic_series = repeated_cycle + trend_component + noise

    res = seasonal_mann_kendall_test(synthetic_series, period=period, alpha=0.05)
    assert res["trend"] == "increasing"
    assert res["is_significant"] is True
    assert res["p_value"] < 0.001


def test_autocorrelation_ar1():
    """Verify ACF computes lag-1 autocorrelation correctly for an AR(1) process."""
    np.random.seed(42)
    n = 200
    phi = 0.6  # target lag-1 autocorrelation
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + np.random.normal(0, 1.0)

    acf_res = compute_autocorrelation(y, max_lag=5)
    assert len(acf_res["acf_values"]) == 6
    assert math.isclose(acf_res["acf_values"][0], 1.0, rel_tol=1e-5)
    # Lag-1 sample ACF should be close to phi (0.6) within statistical error
    assert abs(acf_res["lag_1"] - phi) < 0.15
    assert acf_res["is_lag1_significant"] is True
