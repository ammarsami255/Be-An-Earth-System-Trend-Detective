"""
Temperature Extremes Tests
EarthTrend Explorer - NASA Space Apps 2026
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data_loader import auto_load_dataset
from src.extremes_analysis import (
    analyze_temperature_extremes,
    annual_extreme_indices,
    build_monthly_extremes,
    compare_regional_extremes,
    extreme_trend_test,
    monthly_percentile_thresholds,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _load(city):
    with open(os.path.join(DATA_DIR, f"nasa_power_{city}_1981_2023.csv"), encoding="utf-8") as f:
        return auto_load_dataset(f.read())[0]


def _synthetic(years, tmax_fn, tmin_fn):
    rows = []
    for y in years:
        for m in range(1, 13):
            rows.append({"year": y, "month": m, "variable": "T2M_MAX", "value": tmax_fn(y, m)})
            rows.append({"year": y, "month": m, "variable": "T2M_MIN", "value": tmin_fn(y, m)})
    return pd.DataFrame(rows)


def test_nasa_power_t2m_max_is_monthly_extreme_not_mean():
    """The NASA POWER 'ANN' value for T2M_MAX equals the max of the 12 months -> absolute extreme."""
    with open(os.path.join(DATA_DIR, "nasa_power_cairo_1981_2023.csv"), encoding="utf-8") as f:
        _, meta = auto_load_dataset(f.read())
    ann = meta["annual_df"]
    df = _load("cairo")
    for year in (1981, 2000, 2023):
        monthly_max = df[(df.variable == "T2M_MAX") & (df.year == year)]["value"].max()
        monthly_min = df[(df.variable == "T2M_MIN") & (df.year == year)]["value"].min()
        ann_max = ann[(ann.variable == "T2M_MAX") & (ann.year == year)]["annual_value"].iloc[0]
        ann_min = ann[(ann.variable == "T2M_MIN") & (ann.year == year)]["annual_value"].iloc[0]
        assert ann_max == pytest.approx(monthly_max)
        assert ann_min == pytest.approx(monthly_min)


def test_annual_indices_pick_hottest_and_coldest_month():
    df = _synthetic([2000], lambda y, m: 30.0 + (5.0 if m == 7 else 0.0), lambda y, m: 10.0 - (6.0 if m == 1 else 0.0))
    monthly = build_monthly_extremes(df)
    assert (monthly["range"] == monthly["tmax"] - monthly["tmin"]).all()
    thr = pd.DataFrame({"month": range(1, 13), "hot_threshold": 34.0, "cold_threshold": 5.0})
    annual = annual_extreme_indices(monthly, thr).iloc[0]
    assert annual["annual_hottest"] == 35.0 and annual["hottest_month"] == "JUL"
    assert annual["annual_coldest"] == 4.0 and annual["coldest_month"] == "JAN"
    assert annual["hot_months"] == 1 and annual["cold_months"] == 1


def test_thresholds_are_per_calendar_month_and_use_baseline_only():
    years = range(1981, 2021)
    # 2011-2020 are much hotter; they must not influence the 1981-2010 thresholds
    df = _synthetic(years, lambda y, m: (50.0 if y > 2010 else 20.0 + m), lambda y, m: 5.0 + m)
    thr = monthly_percentile_thresholds(build_monthly_extremes(df)).set_index("month")
    assert thr.loc[1, "hot_threshold"] == pytest.approx(21.0)
    assert thr.loc[12, "hot_threshold"] == pytest.approx(32.0)
    assert thr.loc[6, "cold_threshold"] == pytest.approx(11.0)


def test_count_index_uses_ols_magnitude():
    rng = np.random.default_rng(3)
    years = np.arange(1981, 2024)
    counts = np.clip(np.round((years - 1981) / 15 + rng.normal(0, 0.4, len(years))), 0, None)
    res = extreme_trend_test(counts, years, is_count=True)
    assert res["slope_method"] == "OLS (counts)"
    assert res["is_significant"]
    assert res["sen_per_decade"] > 0.3  # Sen's median slope would collapse to ~0 on these ties


def test_real_nasa_extremes():
    cairo = analyze_temperature_extremes(_load("cairo"), "Greater Cairo")
    assert len(cairo["annual"]) == 43
    t = cairo["trends"]
    assert t["annual_hottest"]["is_significant"] and t["annual_hottest"]["sen_per_decade"] > 0
    assert t["hot_months"]["is_significant"] and t["hot_months"]["sen_per_decade"] > 0
    assert t["cold_months"]["is_significant"] and t["cold_months"]["sen_per_decade"] < 0
    assert not t["extreme_range"]["is_significant"]
    # Hot months by construction ~10% of baseline months -> ~1.2 per year on average
    base = cairo["annual"][cairo["annual"].year <= 2010]
    assert base["hot_months"].mean() == pytest.approx(1.2, abs=0.4)
    assert "Hottest value of the year" in cairo["summary"]["q3_how_much"]

    table = compare_regional_extremes({"Greater Cairo": _load("cairo"), "Aswan": _load("aswan")})
    assert len(table) == 10
    aswan_hot = table[(table.Region == "Aswan") & table.Index.str.startswith("Unusually hot")].iloc[0]
    assert aswan_hot["Trend"] == "Significant increase"


def test_metric_info_contains_txx_tnn_proxies():
    from src.extremes_analysis import METRIC_INFO
    assert "TXx proxy" in METRIC_INFO["annual_hottest"][0]
    assert "TNn proxy" in METRIC_INFO["annual_coldest"][0]
    assert "T2M_MAX" in METRIC_INFO["annual_hottest"][0]
    assert "T2M_MIN" in METRIC_INFO["annual_coldest"][0]
