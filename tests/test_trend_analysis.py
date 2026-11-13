"""
Unit tests for statistical trend analysis module.
EarthTrend Explorer - NASA Space Apps 2026
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.trend_analysis import (
    compute_autocorrelation,
    compute_calendar_month_breakdown,
    compute_linear_regression,
    compute_sens_slope,
    hamed_rao_modified_mann_kendall,
    mann_kendall_test,
    seasonal_mann_kendall_test,
)


def test_mann_kendall_synthetic_increasing():
    """Known strong upward trend must yield p < 0.05 and trend == 'increasing'."""
    np.random.seed(42)
    x = np.arange(50)
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
    n_years = 20
    period = 12
    monthly_cycle = np.array([10, 12, 15, 20, 25, 30, 32, 31, 27, 22, 16, 11], dtype=float)
    repeated_cycle = np.tile(monthly_cycle, n_years)

    time_steps = np.arange(n_years * period)
    trend_component = 0.05 * time_steps
    noise = np.random.normal(0, 0.5, size=n_years * period)
    synthetic_series = repeated_cycle + trend_component + noise

    res = seasonal_mann_kendall_test(synthetic_series, period=period, alpha=0.05)
    assert res["trend"] == "increasing"
    assert res["is_significant"] is True
    assert res["p_value"] < 0.001


def test_seasonal_sen_slope_units():
    """Verify seasonal Sen slope returns change per year, not change per month."""
    n_years = 15
    period = 12
    # Known trend of +2.0 units per year
    y = np.zeros(n_years * period)
    for yr in range(n_years):
        for m in range(period):
            y[yr * period + m] = 2.0 * yr + m * 0.5

    res = seasonal_mann_kendall_test(y, period=period)
    # Slope must equal 2.0 per year (not 2.0/12)
    assert math.isclose(res["sen_slope"], 2.0, rel_tol=1e-3)


def test_autocorrelation_ar1():
    """Verify ACF computes lag-1 autocorrelation correctly for an AR(1) process."""
    np.random.seed(42)
    n = 200
    phi = 0.6
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + np.random.normal(0, 1.0)

    acf_res = compute_autocorrelation(y, max_lag=5)
    assert len(acf_res["acf_values"]) == 6
    assert math.isclose(acf_res["acf_values"][0], 1.0, rel_tol=1e-5)
    assert abs(acf_res["lag_1"] - phi) < 0.15
    assert acf_res["is_lag1_significant"] is True


def test_detrended_autocorrelation_white_noise():
    """
    Verify that an uncorrelated series with a strong deterministic trend exhibits
    apparent raw autocorrelation, but zero residual autocorrelation after detrending.
    """
    np.random.seed(42)
    n = 50
    x = np.arange(n)
    white_noise = np.random.normal(0, 0.5, size=n)
    # y = trend + white noise (NO real AR persistence)
    y = 0.5 * x + white_noise

    acf_res = compute_autocorrelation(y, x_time=x, max_lag=5)
    # Raw lag-1 is artificially high due to trend
    assert acf_res["lag_1"] > 0.5
    assert acf_res["is_lag1_significant"] is True
    # Residual lag-1 after removing trend is within Bartlett bounds
    assert abs(acf_res["residual_lag_1"]) <= acf_res["ci_bound_95"]
    assert acf_res["is_residual_lag1_significant"] is False


def test_calendar_month_breakdown():
    """Verify month-by-month calendar breakdown correctly computes 12 months."""
    years = list(range(1990, 2020))  # 30 years
    records = []
    for yr in years:
        for m in range(1, 13):
            # Only summer months have trend
            trend_val = 0.1 * (yr - 1990) if m in [6, 7, 8] else 0.0
            records.append({
                "year": yr,
                "month": m,
                "value": 15.0 + m + trend_val + np.random.normal(0, 0.1),
            })
    df = pd.DataFrame(records)

    breakdown = compute_calendar_month_breakdown(df, value_col="value")
    assert len(breakdown) == 12
    assert "JAN" in breakdown["month_name"].values
    assert "JUL" in breakdown["month_name"].values
    # July should have significant trend
    jul_row = breakdown[breakdown["month_name"] == "JUL"].iloc[0]
    assert jul_row["trend"] == "Increasing"
    assert bool(jul_row["is_significant"]) is True


def test_hamed_rao_modified_mann_kendall():
    """Verify Hamed & Rao modified Mann-Kendall handles series properly."""
    np.random.seed(42)
    x = np.arange(40)
    y = 0.1 * x + np.random.normal(0, 0.2, size=40)

    res = hamed_rao_modified_mann_kendall(y, alpha=0.05)
    assert res["trend"] == "increasing"
    assert res["is_significant"] is True
    assert res["p_value"] < 0.01
