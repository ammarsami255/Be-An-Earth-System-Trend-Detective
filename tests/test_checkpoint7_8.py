"""
Tests for the Checkpoint 7 / 8 additions:
  * prediction-interval coverage + binomial calibration verdict,
  * exceedance probabilities P(Z >= +1) etc.,
  * reproducible (seeded) Holt-Winters Monte Carlo intervals,
  * regional warming-rate difference test,
  * plain-language narrative builders.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import auto_load_dataset
from src.forecasting import (
    backtest_models,
    exceedance_probability,
    forecast_exponential_smoothing,
    generate_12month_forecast,
    interval_coverage_verdict,
)
from src.narrative import forecast_story, regional_story
from src.trend_analysis import compare_regional_warming_rates

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGIONS = {
    "Greater Cairo": "nasa_power_cairo_1981_2023.csv",
    "Alexandria": "nasa_power_alexandria_1981_2023.csv",
    "Aswan": "nasa_power_aswan_1981_2023.csv",
}


def _load(name):
    with open(os.path.join(ROOT, "data", "raw", REGIONS[name]), encoding="utf-8") as f:
        return auto_load_dataset(f.read(), default_location=name)[0]


def _seasonal_series(n=240, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 20 + 0.01 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)


# ---------------------------------------------------------------- forecasting
def test_exceedance_probability_basic_properties():
    # forecast exactly on the threshold -> 50 %
    assert exceedance_probability(10.0, 8.04, 11.96, 10.0) == pytest.approx(0.5, abs=1e-6)
    # far above the threshold -> ~1, far below -> ~0
    assert exceedance_probability(10.0, 9.0, 11.0, 0.0) > 0.999
    assert exceedance_probability(10.0, 9.0, 11.0, 20.0) < 0.001
    # 95 % interval half-width 1.96 -> sigma 1 -> P(Y >= mu + 1) = 15.87 %
    assert exceedance_probability(0.0, -1.959964, 1.959964, 1.0) == pytest.approx(0.1587, abs=1e-3)


def test_interval_coverage_verdict():
    assert interval_coverage_verdict(95.0, 95, 60) == "well calibrated"
    assert interval_coverage_verdict(60.0, 95, 60).startswith("too narrow")
    assert interval_coverage_verdict(100.0, 80, 60).startswith("too wide")
    assert interval_coverage_verdict(float("nan"), 95, 60) == "not evaluated"


def test_backtest_reports_coverage():
    bt = backtest_models(_seasonal_series(), horizon=12, n_splits=3)
    for m in bt["summary_metrics"].values():
        assert 0.0 <= m["coverage_80"] <= 100.0
        assert 0.0 <= m["coverage_95"] <= 100.0
        assert m["coverage_95"] >= m["coverage_80"]  # nested bands
        assert m["n_test_months"] == 36


def test_ets_intervals_are_reproducible():
    y = _seasonal_series()
    a = forecast_exponential_smoothing(y, horizon=12)
    b = forecast_exponential_smoothing(y, horizon=12)
    np.testing.assert_allclose(a["ci_95_lower"], b["ci_95_lower"])
    np.testing.assert_allclose(a["ci_95_upper"], b["ci_95_upper"])


def test_forecast_probability_columns_and_story():
    df = _load("Greater Cairo")
    res = generate_12month_forecast(df[df["variable"] == "T2M"], variable="value")
    fc = res["forecast_df"]
    for col in ("prob_above_normal", "prob_z_ge_plus1", "prob_z_le_minus1"):
        assert fc[col].between(0, 100).all()
    # being above +1 sigma is never more likely than being above normal
    assert (fc["prob_z_ge_plus1"] <= fc["prob_above_normal"] + 1e-9).all()
    # the leader's verified backtest figures must be unchanged
    assert res["summary_kpis"]["best_model_key"] == "ets"
    assert res["summary_kpis"]["best_model_mase"] == pytest.approx(0.800, abs=1e-3)
    story = forecast_story(res, "T2M", "°C", "Greater Cairo")
    kinds = {k for k, _ in story}
    assert {"projected", "statistical", "caution"} <= kinds


def test_precipitation_forecast_has_no_gaussian_probabilities():
    df = _load("Greater Cairo")
    res = generate_12month_forecast(df[df["variable"] == "PRECTOTCORR"], variable="value")
    assert res["forecast_df"]["prob_above_normal"].isna().all()
    assert any("rain" in text.lower() for _, text in forecast_story(res, "PRECTOTCORR", "mm/day", "Greater Cairo"))


# ---------------------------------------------------------------- regional difference
def test_regional_difference_detects_synthetic_gap():
    years = np.repeat(np.arange(1981, 2024), 12)
    months = np.tile(np.arange(1, 13), 43)
    rng = np.random.default_rng(1)
    common = np.repeat(rng.normal(0, 0.4, 43), 12)  # shared weather

    def make(rate):
        v = 20 + rate * (years - 1981) + common + rng.normal(0, 0.05, len(years))
        return pd.DataFrame({"variable": "T2M", "year": years, "month": months, "value": v})

    res = compare_regional_warming_rates({"Fast": make(0.05), "Slow": make(0.02), "Slow2": make(0.02)})
    fast_rows = res[res["faster_region"] == "Fast"]
    assert fast_rows["is_significant"].all()
    assert fast_rows["difference_per_decade"].between(0.2, 0.4).all()
    same = res[(res["faster_region"] != "Fast") & (res["slower_region"] != "Fast")]
    assert not same["is_significant"].any()


def test_regional_difference_on_real_data_aswan_faster():
    dfs = {name: _load(name) for name in REGIONS}
    res = compare_regional_warming_rates(dfs, variable="T2M")
    assert len(res) == 3
    assert (res["difference_per_decade"] >= 0).all()
    aswan_cairo = res[(res["faster_region"] == "Aswan") & (res["slower_region"] == "Greater Cairo")].iloc[0]
    assert aswan_cairo["is_significant"]
    cairo_alex = res[{"Greater Cairo", "Alexandria"} == res[["faster_region", "slower_region"]].apply(set, axis=1)]
    assert not cairo_alex["is_significant"].any()
    story = regional_story(res, "mean temperature")
    assert any("Aswan is warming faster" in text for _, text in story)


# ---------------------------------------------------------------- 2024 reality check
def test_reality_check_against_2024_observations():
    from src.forecasting import verify_forecast_against_observations
    from src.narrative import verification_story

    with open(os.path.join(ROOT, "data", "verification", "nasa_power_cairo_2024.csv"), encoding="utf-8") as f:
        obs = auto_load_dataset(f.read(), default_location="Greater Cairo")[0]
    df = _load("Greater Cairo")
    res = generate_12month_forecast(df[df["variable"] == "T2M"], variable="value")
    # verification data must lie strictly after the training record (no leakage)
    assert obs["date"].min() > df["date"].max()
    ver = verify_forecast_against_observations(res, obs[obs["variable"] == "T2M"])
    assert ver["available"] and ver["n_months"] == 12
    assert ver["mae"] < ver["naive_mae"]          # beat "same as last year" on unseen 2024
    assert ver["hit_95"] >= 80.0
    assert ver["observed_mean_anomaly"] > 0
    assert any("2024" in text for _, text in verification_story(ver, "°C", "Greater Cairo", False))


def test_reality_check_without_overlap_is_unavailable():
    from src.forecasting import verify_forecast_against_observations
    df = _load("Greater Cairo")
    res = generate_12month_forecast(df[df["variable"] == "T2M"], variable="value")
    old = df[(df["variable"] == "T2M") & (df["year"] == 1990)]
    assert verify_forecast_against_observations(res, old)["available"] is False
