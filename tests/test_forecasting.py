"""
Unit and Integration Tests for Climate Anomaly Forecasting Engine
EarthTrend Explorer - NASA Space Apps Challenge 2026
Team: Machine Without Learning
"""

import math
import os
import numpy as np
import pandas as pd
import pytest

from src.data_loader import auto_load_dataset
from src.forecasting import (
    backtest_models,
    classify_future_z_score,
    compute_forecast_metrics,
    forecast_exponential_smoothing,
    forecast_sarima,
    forecast_seasonal_naive,
    generate_12month_forecast,
)


@pytest.fixture
def synthetic_climate_series():
    """Generates 40 years (480 months) of synthetic monthly temperature data with known cycle and trend."""
    np.random.seed(42)
    t = np.arange(480)
    # Annual cycle (period 12) + linear warming trend (+0.03 deg/year = 0.0025 deg/month) + Gaussian noise
    seasonal = 10.0 * np.sin(2.0 * np.pi * t / 12.0)
    trend = 0.0025 * t
    noise = np.random.normal(0.0, 1.0, 480)
    y = 20.0 + seasonal + trend + noise
    return y


@pytest.fixture
def cairo_dataset():
    """Loads authentic Greater Cairo NASA POWER monthly dataset."""
    cairo_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nasa_power_cairo_1981_2023.csv")
    assert os.path.exists(cairo_path), f"Cairo data file not found: {cairo_path}"
    with open(cairo_path, "r", encoding="utf-8") as f:
        content = f.read()
    df, meta = auto_load_dataset(content, default_location="Greater Cairo")
    return df


def test_seasonal_naive_forecast_properties(synthetic_climate_series):
    """Tests 12-month periodicity matching, residual variance, and interval monotonicity for Seasonal Naive."""
    y = synthetic_climate_series
    res = forecast_seasonal_naive(y, horizon=12, alpha_80=0.20, alpha_95=0.05)

    assert res["model_key"] == "naive"
    assert len(res["forecast"]) == 12

    # Point forecast must exactly match the last 12 observed months
    expected_point_forecast = y[-12:]
    np.testing.assert_allclose(res["forecast"], expected_point_forecast, rtol=1e-5)

    # Monotonicity of prediction intervals
    fc = res["forecast"]
    ci80_l = res["ci_80_lower"]
    ci80_u = res["ci_80_upper"]
    ci95_l = res["ci_95_lower"]
    ci95_u = res["ci_95_upper"]

    assert np.all(ci95_l <= ci80_l), "95% lower bound must be <= 80% lower bound"
    assert np.all(ci80_l <= fc), "80% lower bound must be <= point forecast"
    assert np.all(fc <= ci80_u), "Point forecast must be <= 80% upper bound"
    assert np.all(ci80_u <= ci95_u), "80% upper bound must be <= 95% upper bound"


def test_exponential_smoothing_forecast_properties(synthetic_climate_series):
    """Tests Holt-Winters additive ETS model fitting and simulated prediction interval coverage."""
    y = synthetic_climate_series
    res = forecast_exponential_smoothing(y, horizon=12, n_simulations=500, alpha_80=0.20, alpha_95=0.05)

    assert res["model_key"] == "ets"
    assert len(res["forecast"]) == 12
    assert "alpha (level)" in res["params"]

    fc = res["forecast"]
    ci80_l = res["ci_80_lower"]
    ci80_u = res["ci_80_upper"]
    ci95_l = res["ci_95_lower"]
    ci95_u = res["ci_95_upper"]

    # Monotonicity
    assert np.all(ci95_l <= ci80_l)
    assert np.all(ci80_l <= fc)
    assert np.all(fc <= ci80_u)
    assert np.all(ci80_u <= ci95_u)


def test_sarima_forecast_properties(synthetic_climate_series):
    """Tests SARIMAX seasonal model fitting and analytical Kalman covariance intervals."""
    y = synthetic_climate_series
    res = forecast_sarima(y, horizon=12, alpha_80=0.20, alpha_95=0.05)

    assert res["model_key"] == "sarima"
    assert len(res["forecast"]) == 12

    fc = res["forecast"]
    ci80_l = res["ci_80_lower"]
    ci80_u = res["ci_80_upper"]
    ci95_l = res["ci_95_lower"]
    ci95_u = res["ci_95_upper"]

    assert np.all(ci95_l <= ci80_l)
    assert np.all(ci80_l <= fc)
    assert np.all(fc <= ci80_u)
    assert np.all(ci80_u <= ci95_u)


def test_precipitation_nonnegativity_clamping():
    """Verifies that precipitation forecasts and lower confidence intervals are strictly non-negative."""
    # Semi-arid series with near-zero values
    np.random.seed(123)
    y_rain = np.maximum(0.0, np.random.exponential(scale=1.5, size=240) - 1.0)

    for fc_func in [forecast_seasonal_naive, forecast_exponential_smoothing, forecast_sarima]:
        res = fc_func(y_rain, horizon=12, is_precipitation=True)
        assert np.all(res["forecast"] >= 0.0), f"{res['model_name']} predicted negative rainfall!"
        assert np.all(res["ci_80_lower"] >= 0.0), f"{res['model_name']} 80% CI low is negative!"
        assert np.all(res["ci_95_lower"] >= 0.0), f"{res['model_name']} 95% CI low is negative!"


def test_backtesting_tournament_expansion(synthetic_climate_series):
    """Verifies expanding-window chronological cross-validation without temporal data leakage."""
    y = synthetic_climate_series
    res = backtest_models(y, horizon=12, n_splits=4)

    assert res["n_splits"] == 4
    assert len(res["split_details"]) == 4

    # Verify strictly expanding training windows
    train_sizes = [sp["train_size"] for sp in res["split_details"]]
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] == train_sizes[i - 1] + 12, "Train size must expand by horizon each split."

    # Verify summary metrics exist for all 3 models
    summary = res["summary_metrics"]
    for m in ["naive", "ets", "sarima"]:
        assert m in summary
        assert summary[m]["mean_mae"] > 0.0
        assert summary[m]["mean_rmse"] > 0.0
        assert summary[m]["mase"] > 0.0

    assert res["tournament_winner_key"] in ["naive", "ets", "sarima"]


def test_future_anomaly_classification():
    """Verifies standard WMO Z-score severity bracket classification."""
    assert classify_future_z_score(3.5)[0] == "Extreme Warm (+3σ)"
    assert classify_future_z_score(2.5)[0] == "Severe Warm (+2σ)"
    assert classify_future_z_score(1.5)[0] == "Moderate Warm (+1σ)"
    assert classify_future_z_score(0.2)[0] == "Near Normal"
    assert classify_future_z_score(-0.4)[0] == "Near Normal"
    assert classify_future_z_score(-1.5)[0] == "Moderate Cold (-1σ)"
    assert classify_future_z_score(-2.5)[0] == "Severe Cold (-2σ)"
    assert classify_future_z_score(-3.5)[0] == "Extreme Cold (-3σ)"
    assert classify_future_z_score(float("nan"))[0] == "Undefined"


def test_end_to_end_real_data_cairo(cairo_dataset):
    """Executes full end-to-end forecasting pipeline on authentic Cairo 1981–2023 dataset."""
    df_t2m = cairo_dataset[cairo_dataset["variable"] == "T2M"].copy()
    assert len(df_t2m) == 516

    res = generate_12month_forecast(df_t2m, variable="value", model_type="best")

    fc_df = res["forecast_df"]
    assert len(fc_df) == 12

    # Check future dates: 2024-01-01 through 2024-12-01
    assert fc_df["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert fc_df["date"].iloc[-1] == pd.Timestamp("2024-12-01")

    # Check no NaNs in forecast columns
    for col in ["forecast", "ci_80_lower", "ci_80_upper", "ci_95_lower", "ci_95_upper", "expected_anomaly", "expected_z_score"]:
        assert not fc_df[col].isna().any(), f"NaN found in column {col}"

    # Check tournament winner and KPIs
    assert res["tournament_winner"] in ["ets", "sarima", "naive"]
    assert "mean_expected_anomaly" in res["summary_kpis"]
    assert "peak_forecast_value" in res["summary_kpis"]


def test_forecasting_edge_cases():
    """Verifies graceful handling of short series and edge cases."""
    # Series with length < 24 should fall back to Seasonal Naive
    short_series = np.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 21.0])
    res_ets = forecast_exponential_smoothing(short_series, horizon=6)
    assert res_ets["model_key"] == "naive"

    res_sarima = forecast_sarima(short_series, horizon=6)
    assert res_sarima["model_key"] == "naive"
