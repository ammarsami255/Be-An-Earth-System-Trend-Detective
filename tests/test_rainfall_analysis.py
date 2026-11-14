"""
Rainfall Analysis Tests
EarthTrend Explorer - NASA Space Apps 2026
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data_loader import auto_load_dataset
from src.rainfall_analysis import (
    analyze_rainfall,
    annual_rainfall_totals,
    compare_regional_rainfall,
    monthly_rainfall_climatology,
    monthly_rainfall_totals,
    pettitt_change_point_test,
    rainfall_percent_of_normal,
    rainfall_trend_test,
    wet_season_totals,
)
from src.visualization import (
    plot_rainfall_annual_totals,
    plot_rainfall_monthly_climatology,
    plot_regional_rainfall,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _synthetic_rain(years, rate_fn):
    """Tidy PRECTOTCORR frame where rate_fn(year, month) returns mm/day."""
    rows = []
    for y in years:
        for m in range(1, 13):
            rows.append({"year": y, "month": m, "variable": "PRECTOTCORR", "value": rate_fn(y, m)})
    return pd.DataFrame(rows)


def _load(city):
    with open(os.path.join(DATA_DIR, f"nasa_power_{city}_1981_2023.csv"), encoding="utf-8") as f:
        return auto_load_dataset(f.read())[0]


def test_monthly_totals_convert_rate_to_mm_using_days_in_month():
    df = _synthetic_rain([2020, 2021], lambda y, m: 1.0)  # 1 mm/day every month
    monthly = monthly_rainfall_totals(df)
    feb_2020 = monthly[(monthly.year == 2020) & (monthly.month == 2)].total_mm.iloc[0]
    feb_2021 = monthly[(monthly.year == 2021) & (monthly.month == 2)].total_mm.iloc[0]
    assert feb_2020 == pytest.approx(29.0)  # leap year
    assert feb_2021 == pytest.approx(28.0)
    annual = annual_rainfall_totals(monthly)
    assert annual.set_index("year").total_mm.to_dict() == pytest.approx({2020: 366.0, 2021: 365.0})


def test_incomplete_year_is_dropped():
    df = _synthetic_rain([2020, 2021], lambda y, m: 1.0)
    df = df[~((df.year == 2021) & (df.month == 12))]
    annual = annual_rainfall_totals(monthly_rainfall_totals(df))
    assert annual.year.tolist() == [2020]


def test_wet_season_spans_new_year_and_drops_partial_first_season():
    # Rain only in January (1 mm/day) and October (2 mm/day)
    df = _synthetic_rain([2001, 2002, 2003], lambda y, m: {1: 1.0, 10: 2.0}.get(m, 0.0))
    seasons = wet_season_totals(monthly_rainfall_totals(df))
    # Season ending 2001 needs Oct-Dec 2000 -> dropped. Season ending 2004 needs Jan-Mar 2004 -> dropped.
    assert seasons.season_end_year.tolist() == [2002, 2003]
    assert seasons.season_label.tolist() == ["2001/02", "2002/03"]
    # Oct (31 days x 2) + Jan (31 days x 1) = 93 mm
    assert seasons.total_mm.tolist() == pytest.approx([93.0, 93.0])


def test_climatology_and_percent_of_normal():
    df = _synthetic_rain(range(1981, 2011), lambda y, m: 1.0 if m == 1 else 0.0)
    monthly = monthly_rainfall_totals(df)
    clim = monthly_rainfall_climatology(monthly).set_index("month")
    assert clim.loc[1, "mean_mm"] == pytest.approx(31.0)
    assert clim.loc[1, "wet_fraction"] == 1.0
    assert clim.loc[7, "wet_fraction"] == 0.0
    annual, normal = rainfall_percent_of_normal(annual_rainfall_totals(monthly))
    assert normal == pytest.approx(31.0)
    assert np.allclose(annual.pct_of_normal, 100.0)


def test_trend_test_detects_synthetic_wetting():
    rng = np.random.default_rng(0)
    years = np.arange(1981, 2024)
    values = 50.0 + 2.0 * (years - 1981) + rng.normal(0, 5, len(years))
    res = rainfall_trend_test(values, years)
    assert res["is_significant"]
    assert res["verdict"] == "Significant trend: getting wetter"
    assert res["sen_slope_per_decade"] == pytest.approx(20.0, rel=0.15)
    assert not res["low_reliability"]


def test_trend_test_flags_zero_inflated_series():
    years = np.arange(1981, 2024)
    values = np.zeros(len(years))
    values[[3, 17, 30]] = [0.5, 1.2, 0.8]
    res = rainfall_trend_test(values, years)
    assert not res["is_significant"]
    assert res["low_reliability"]
    assert res["zero_fraction"] > 0.9


def test_real_nasa_rainfall_results():
    dfs = {"Greater Cairo": _load("cairo"), "Alexandria": _load("alexandria"), "Aswan": _load("aswan")}
    cairo = analyze_rainfall(dfs["Greater Cairo"], "Greater Cairo")
    assert (cairo["start_year"], cairo["end_year"]) == (1981, 2023)
    assert len(cairo["annual"]) == 43
    assert len(cairo["seasons"]) == 42
    t = cairo["trends"]["annual_total"]
    assert t["mean"] == pytest.approx(62.1, abs=0.1)
    assert not t["is_significant"]
    lo, hi = t["sen_ci_per_decade"]
    assert lo < 0 < hi

    table = compare_regional_rainfall(dfs).set_index("Region")
    assert table.loc["Aswan", "Reliability"] == "Low"
    assert table.loc["Alexandria", "Mean (mm/yr)"] > table.loc["Greater Cairo", "Mean (mm/yr)"] > table.loc["Aswan", "Mean (mm/yr)"]
    assert (table["Annual trend"] == "No significant trend").all()

    # Abrupt upward shift in the mid-2010s for Cairo and Alexandria, none for Aswan
    cp = cairo["change_point"]
    assert cp["is_significant"] and cp["direction"] == "upward"
    assert cp["new_regime_start"] == 2015
    assert table.loc["Alexandria", "Abrupt shift (Pettitt)"] == "Yes, from 2015"
    assert table.loc["Aswan", "Abrupt shift (Pettitt)"] == "No"
    assert "Pettitt" in cairo["summary"]["q4_is_significant"]

    figs = [
        plot_rainfall_annual_totals(cairo["annual"], t, cairo["normal_mm"], "Greater Cairo", change_point=cp),
        plot_rainfall_monthly_climatology(cairo["climatology"], "Greater Cairo"),
        plot_regional_rainfall({"Greater Cairo": cairo["annual"]}),
    ]
    assert all(len(f.data) >= 1 for f in figs)


def test_pettitt_finds_step_change_without_gradual_trend():
    rng = np.random.default_rng(1)
    years = np.arange(1981, 2024)
    values = np.where(years >= 2010, 100.0, 40.0) + rng.normal(0, 8, len(years))
    cp = pettitt_change_point_test(values, years)
    assert cp["is_significant"]
    assert cp["new_regime_start"] == 2010
    assert cp["change_year"] == 2009
    assert cp["direction"] == "upward"
    assert cp["mean_before"] == pytest.approx(40.0, abs=4)
    assert cp["mean_after"] == pytest.approx(100.0, abs=5)


def test_pettitt_no_shift_in_pure_noise():
    rng = np.random.default_rng(2)
    years = np.arange(1981, 2024)
    cp = pettitt_change_point_test(rng.normal(50, 10, len(years)), years)
    assert not cp["is_significant"]
    assert cp["p_value"] > 0.05


def test_pettitt_statistic_matches_hand_calculation():
    # x = [1, 2, 3, 10, 11, 12]: best split after index 2, U_2 = -9 (all 9 pairs have x_i < x_j)
    cp = pettitt_change_point_test(np.array([1, 2, 3, 10, 11, 12, 13, 14, 15, 16.0]), np.arange(10))
    assert cp["k_statistic"] == pytest.approx(25.0)  # split 5|5 -> 25 pairs all negative
    assert cp["u_series"][0] == -9.0


def test_temperature_extremes_comparison_on_real_data():
    from src.trend_analysis import compare_temperature_extremes
    from src.visualization import plot_temperature_extremes_comparison

    dfs = {"Greater Cairo": _load("cairo"), "Aswan": _load("aswan")}
    comp = compare_temperature_extremes(dfs)
    assert len(comp) == 6
    assert set(comp["variable"]) == {"T2M", "T2M_MAX", "T2M_MIN"}
    assert comp["is_significant"].all()
    assert (comp["sen_per_decade"] > 0).all()
    cairo_mean = comp[(comp.region == "Greater Cairo") & (comp.variable == "T2M")].iloc[0]
    assert cairo_mean["sen_per_decade"] == pytest.approx(0.293, abs=0.005)  # README benchmark
    # Extremes really are extremes
    cairo = comp[comp.region == "Greater Cairo"].set_index("variable")
    assert cairo.loc["T2M_MIN", "mean_c"] < cairo.loc["T2M", "mean_c"] < cairo.loc["T2M_MAX", "mean_c"]
    assert len(plot_temperature_extremes_comparison(comp).data) == 3
