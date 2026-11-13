"""
Unit and Integration Tests for Climate Anomaly Detection Engine
EarthTrend Explorer - NASA Space Apps Challenge 2026
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.anomaly_detection import (
    analyze_compound_anomalies,
    compare_regional_anomalies,
    compute_anomalies_and_zscores,
    compute_precipitation_anomalies,
    compute_spi,
    compute_temperature_anomalies,
    cross_variable_anomaly_correlation,
    decadal_extreme_frequency,
    detect_persistent_anomalies,
    get_top_ranked_extremes,
)
from src.data_loader import auto_load_dataset


# ==============================================================================
# 1. TEMPERATURE ANOMALY TESTS
# ==============================================================================

def test_compute_temperature_anomalies_synthetic_math() -> None:
    """Verifies that monthly departures and Z-scores exactly match closed-form formulas."""
    records = []
    # 30 baseline years (1981-2010) + 10 evaluation years (2011-2020)
    for y in range(1981, 2021):
        for m in range(1, 13):
            # Deterministic pattern: Jan has mean 10, Feb 12, etc. + small perturbation
            val = 10.0 + m * 2.0 + (y - 1981) * 0.1
            records.append({
                "date": f"{y}-{m:02d}-01",
                "year": y,
                "month": m,
                "variable": "T2M",
                "value": val,
            })
    df = pd.DataFrame(records)

    anom_df, summary = compute_temperature_anomalies(df, variable="T2M", baseline_start_year=1981, baseline_end_year=2010)
    assert len(anom_df) == 40 * 12
    assert summary["baseline_years_count"] == 30

    # For Jan: baseline values are 10 + 2 + (0..29)*0.1 = 12.0 + (0..29)*0.1
    jan_base = [12.0 + i * 0.1 for i in range(30)]
    expected_jan_mean = np.mean(jan_base)
    expected_jan_std = np.std(jan_base, ddof=1)

    jan_row_1981 = anom_df[(anom_df["year"] == 1981) & (anom_df["month"] == 1)].iloc[0]
    assert np.isclose(jan_row_1981["baseline_mean"], expected_jan_mean, atol=1e-5)
    assert np.isclose(jan_row_1981["baseline_std"], expected_jan_std, atol=1e-5)
    assert np.isclose(jan_row_1981["anomaly"], 12.0 - expected_jan_mean, atol=1e-5)
    assert np.isclose(jan_row_1981["z_score"], (12.0 - expected_jan_mean) / expected_jan_std, atol=1e-5)


def test_temperature_anomaly_constant_series_edge_case() -> None:
    """Verifies that a constant series (std = 0) does not cause division-by-zero or crashes."""
    records = []
    for y in range(1981, 2011):
        for m in range(1, 13):
            records.append({"year": y, "month": m, "variable": "T2M", "value": 25.0, "date": f"{y}-{m:02d}-01"})
    df = pd.DataFrame(records)

    anom_df, summary = compute_temperature_anomalies(df, variable="T2M")
    assert (anom_df["anomaly"] == 0.0).all()
    assert (anom_df["z_score"] == 0.0).all()
    assert (anom_df["classification"] == "Near Normal").all()


def test_distinct_variables_handling() -> None:
    """Verifies that T2M, T2M_MAX, and T2M_MIN are handled independently without mixing."""
    records = []
    for y in range(1981, 2011):
        for m in range(1, 13):
            records.append({"year": y, "month": m, "variable": "T2M", "value": 20.0, "date": f"{y}-{m:02d}-01"})
            records.append({"year": y, "month": m, "variable": "T2M_MAX", "value": 35.0, "date": f"{y}-{m:02d}-01"})
            records.append({"year": y, "month": m, "variable": "T2M_MIN", "value": 10.0, "date": f"{y}-{m:02d}-01"})
    df = pd.DataFrame(records)

    anom_t2m, _ = compute_temperature_anomalies(df, variable="T2M")
    anom_max, _ = compute_temperature_anomalies(df, variable="T2M_MAX")
    anom_min, _ = compute_temperature_anomalies(df, variable="T2M_MIN")

    assert anom_t2m["baseline_mean"].iloc[0] == 20.0
    assert anom_max["baseline_mean"].iloc[0] == 35.0
    assert anom_min["baseline_mean"].iloc[0] == 10.0


# ==============================================================================
# 2. PERSISTENT ANOMALIES & SPELLS TESTS
# ==============================================================================

def test_detect_persistent_anomalies_warm_and_cold_spells() -> None:
    """Verifies detection of persistent runs of anomalous months."""
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    # Inject a 5-month warm spell (months 3-7, 0-indexed: April to August) with Z >= 1.5
    z_scores = [0.2] * 24
    for i in range(3, 8):
        z_scores[i] = 1.8 + (i - 3) * 0.1  # Peak at i=7 (Z=2.2)
    # Inject a 3-month cold spell (months 15-17) with Z <= -1.2
    for i in range(15, 18):
        z_scores[i] = -1.5

    df = pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "z_score": z_scores,
    })

    warm_spells = detect_persistent_anomalies(df, z_threshold=1.0, min_duration=3, spell_type="warm")
    assert len(warm_spells) == 1
    assert warm_spells.iloc[0]["duration_months"] == 5
    assert np.isclose(warm_spells.iloc[0]["peak_z_score"], 2.2, atol=1e-5)

    cold_spells = detect_persistent_anomalies(df, z_threshold=1.0, min_duration=3, spell_type="cold")
    assert len(cold_spells) == 1
    assert cold_spells.iloc[0]["duration_months"] == 3
    assert np.isclose(cold_spells.iloc[0]["peak_z_score"], -1.5, atol=1e-5)


# ==============================================================================
# 3. PRECIPITATION ANOMALY TESTS
# ==============================================================================

def test_precipitation_anomalies_leap_years_and_percent_normal() -> None:
    """Verifies monthly accumulation rate conversion, leap year days, and Percent of Normal."""
    records = []
    # 1983 (common, 28 Feb), 1984 (leap, 29 Feb)
    for y in [1983, 1984]:
        for m in range(1, 13):
            records.append({
                "year": y,
                "month": m,
                "variable": "PRECTOTCORR",
                "value": 1.0,  # 1.0 mm/day
            })
    df = pd.DataFrame(records)

    p_anom, summary = compute_precipitation_anomalies(df, baseline_start_year=1983, baseline_end_year=1984)

    # In 1983, Feb has 28 days -> total_mm = 28.0
    feb_1983 = p_anom[(p_anom["year"] == 1983) & (p_anom["month"] == 2)].iloc[0]
    assert feb_1983["total_mm"] == 28.0

    # In 1984, Feb has 29 days -> total_mm = 29.0
    feb_1984 = p_anom[(p_anom["year"] == 1984) & (p_anom["month"] == 2)].iloc[0]
    assert feb_1984["total_mm"] == 29.0

    # Baseline mean for Feb = (28 + 29) / 2 = 28.5
    assert feb_1983["baseline_mean"] == 28.5
    assert np.isclose(feb_1983["anomaly_mm"], -0.5)
    assert np.isclose(feb_1984["anomaly_mm"], +0.5)


# ==============================================================================
# 4. STANDARDIZED PRECIPITATION INDEX (SPI) TESTS
# ==============================================================================

def test_spi_exact_gamma_transform() -> None:
    """Verifies that SPI correctly fits a Gamma distribution and maps quantiles to standard normal."""
    np.random.seed(42)
    # Generate 30 baseline years of known Gamma-distributed rainfall for Dec (alpha=3, beta=10, mean=30)
    true_alpha = 3.0
    true_scale = 10.0
    dec_base_vals = stats.gamma.rvs(true_alpha, scale=true_scale, size=30)

    records = []
    for idx, y in enumerate(range(1981, 2011)):
        for m in range(1, 13):
            val = dec_base_vals[idx] if m == 12 else 5.0
            records.append({"year": y, "month": m, "total_mm": val})
    df = pd.DataFrame(records)

    spi_df = compute_spi(df, scale=1, baseline_start_year=1981, baseline_end_year=2010)

    # For Dec, SPI distribution should have mean near 0 and std near 1
    dec_spi = spi_df[spi_df["month"] == 12]["SPI_1"].dropna()
    assert len(dec_spi) == 30
    assert abs(dec_spi.mean()) < 0.2
    assert abs(dec_spi.std() - 1.0) < 0.2


def test_spi_high_zero_aridity_guardrail() -> None:
    """Verifies that high zero-fraction regimes (q >= 0.50) are flagged and do not assign positive SPI to zero rain."""
    records = []
    for y in range(1981, 2011):
        for m in range(1, 13):
            # 80% of years have 0 mm rain in July
            val = 0.0 if y % 5 != 0 else 10.0
            records.append({"year": y, "month": m, "total_mm": val})
    df = pd.DataFrame(records)

    spi_df = compute_spi(df, scale=1, baseline_start_year=1981, baseline_end_year=2010, max_zero_fraction=0.50)

    # Month 7 should have SPI NaN because q = 24/30 = 0.80 >= 0.50
    jul_rows = spi_df[spi_df["month"] == 7]
    assert jul_rows["SPI_1"].isna().all()
    assert "Undefined / Low reliability" in jul_rows["SPI_1_status"].iloc[0]


# ==============================================================================
# 5. COMPOUND ANOMALIES & CROSS-VARIABLE TESTS
# ==============================================================================

def test_compound_anomalies_classification() -> None:
    """Verifies compound hot-dry, hot-wet, cold-dry, cold-wet classification."""
    t_df = pd.DataFrame({
        "year": [2000, 2001, 2002, 2003, 2004],
        "month": [1, 1, 1, 1, 1],
        "anomaly": [1.5, 1.8, -1.5, -2.0, 0.1],
        "z_score": [1.5, 1.8, -1.5, -2.0, 0.1],
    })
    p_df = pd.DataFrame({
        "year": [2000, 2001, 2002, 2003, 2004],
        "month": [1, 1, 1, 1, 1],
        "total_mm": [5.0, 150.0, 10.0, 180.0, 100.0],
        "anomaly_mm": [-50.0, 50.0, -40.0, 80.0, 0.0],
        "percent_of_normal": [50.0, 150.0, 60.0, 180.0, 100.0],
    })

    compound_df, summary = analyze_compound_anomalies(t_df, p_df)
    cats = compound_df["compound_category"].tolist()
    assert cats[0] == "Compound Hot & Dry"
    assert cats[1] == "Compound Hot & Wet"
    assert cats[2] == "Compound Cold & Dry"
    assert cats[3] == "Compound Cold & Wet"
    assert cats[4] == "Near Normal"


def test_cross_variable_correlation_and_n_eff() -> None:
    """Verifies that cross-variable correlation adjusts degrees of freedom for serial correlation."""
    n = 100
    np.random.seed(42)
    # Create negatively correlated AR(1) series
    t_series = [np.random.normal()]
    p_series = [-0.5 * t_series[0] + np.random.normal()]
    for _ in range(1, n):
        t_series.append(0.6 * t_series[-1] + np.random.normal())
        p_series.append(0.6 * p_series[-1] - 0.4 * t_series[-1] + np.random.normal())

    years = [2000 + i // 12 for i in range(n)]
    months = [1 + (i % 12) for i in range(n)]

    t_df = pd.DataFrame({"year": years, "month": months, "anomaly": t_series})
    p_df = pd.DataFrame({"year": years, "month": months, "anomaly_mm": p_series})

    res = cross_variable_anomaly_correlation(t_df, p_df, max_lag=2)
    assert res["sample_size"] == n
    assert res["effective_sample_size"] < n  # Effective N must be reduced by positive AR(1)
    assert res["pearson_r"] < 0  # Should be negative
    assert len(res["lagged_cross_correlations"]) == 5  # -2, -1, 0, 1, 2


# ==============================================================================
# 6. INTEGRATION TESTS ON REAL EGYPT NASA DATA
# ==============================================================================

def test_real_egypt_datasets_anomaly_detection() -> None:
    """Verifies anomaly detection engine on actual NASA POWER Cairo, Alexandria, and Aswan files."""
    regional_anoms = {}
    for loc in ["cairo", "alexandria", "aswan"]:
        with open(f"data/raw/nasa_power_{loc}_1981_2023.csv", "r") as f:
            content = f.read()
        df, _ = auto_load_dataset(content)

        # 1. Temperature Anomalies
        t_anom, t_sum = compute_temperature_anomalies(df, variable="T2M")
        assert len(t_anom) == 516
        assert t_sum["baseline_years_count"] == 30
        assert t_sum["complete_years_count"] == 43
        regional_anoms[loc] = t_anom

        # 2. Precipitation Anomalies
        p_anom, p_sum = compute_precipitation_anomalies(df)
        assert len(p_anom) == 516
        assert p_sum["baseline_years_count"] == 30

        # Aswan should be flagged as zero-inflated (> 50% zeros)
        if loc == "aswan":
            assert p_sum["is_zero_inflated"] is True
            assert p_sum["zero_fraction"] > 0.80

        # 3. SPI Calculation
        spi_df = compute_spi(p_anom, scale=12)
        assert len(spi_df) == 516
        if loc in ["cairo", "alexandria"]:
            # SPI-12 in December has 0 zeros in Cairo and Alexandria, so must be valid
            dec_valid = spi_df[spi_df["month"] == 12]["SPI_12"].dropna()
            assert len(dec_valid) >= 42

    # 4. Regional Comparison
    cmp_res = compare_regional_anomalies(regional_anoms)
    assert cmp_res["total_overlapping_months"] == 516
    assert cmp_res["spatial_concordance_rate"] > 0.70  # Over 70% of months share the same anomaly sign across Egypt


# ==============================================================================
# 7. REGRESSION TESTS FOR AUDIT CORRECTIONS
# ==============================================================================

def test_compound_anomalies_decadal_all_four_categories() -> None:
    """
    Regression test verifying independent aggregation of all four compound categories
    across decades, guarding specifically against the cold_dry_count copy-paste bug.
    """
    t_df = pd.DataFrame({
        "year":  [1980, 1981, 1982, 1983, 1990, 1991, 1992, 1993],
        "month": [1,    1,    1,    1,    1,    1,    1,    1],
        "anomaly": [1.5, 1.2, -1.5, -1.8, 2.0, 1.5, -0.2, -1.2],
        "z_score": [1.5, 1.2, -1.5, -1.8, 2.0, 1.5, -0.2, -1.2],
    })
    p_df = pd.DataFrame({
        "year":  [1980, 1981, 1982, 1983, 1990, 1991, 1992, 1993],
        "month": [1,    1,    1,    1,    1,    1,    1,    1],
        "total_mm": [5.0, 100.0, 10.0, 150.0, 5.0, 160.0, 90.0, 140.0],
        "anomaly_mm": [-50.0, 0.0, -40.0, 50.0, -50.0, 60.0, -10.0, 40.0],
        "percent_of_normal": [50.0, 100.0, 50.0, 150.0, 50.0, 160.0, 90.0, 140.0],
    })

    compound_df, summary = analyze_compound_anomalies(t_df, p_df)
    dec_df = summary["decadal_shift"]
    row_80s = dec_df[dec_df["decade_label"] == "1980s"].iloc[0]
    row_90s = dec_df[dec_df["decade_label"] == "1990s"].iloc[0]

    # In 1980s: 1 Hot-Dry (1980), 0 Hot-Wet (1981 is Near Normal), 1 Cold-Dry (1982), 1 Cold-Wet (1983)
    assert row_80s["total_months"] == 4
    assert row_80s["hot_dry_count"] == 1
    assert row_80s["hot_wet_count"] == 0
    assert row_80s["cold_dry_count"] == 1
    assert row_80s["cold_wet_count"] == 1
    assert np.isclose(row_80s["hot_dry_pct"], 25.0)
    assert np.isclose(row_80s["cold_dry_pct"], 25.0)

    # In 1990s: 1 Hot-Dry (1990), 1 Hot-Wet (1991), 0 Cold-Dry (1992 is Near Normal), 1 Cold-Wet (1993)
    assert row_90s["total_months"] == 4
    assert row_90s["hot_dry_count"] == 1
    assert row_90s["hot_wet_count"] == 1
    assert row_90s["cold_dry_count"] == 0
    assert row_90s["cold_wet_count"] == 1
    assert np.isclose(row_90s["hot_dry_pct"], 25.0)
    assert np.isclose(row_90s["cold_dry_pct"], 0.0)
    assert np.isclose(row_90s["cold_wet_pct"], 25.0)


def test_spi_theoretical_lower_bound_and_drought_limitation() -> None:
    """Verifies exposure of Phi^-1(q) lower bound and drought limitation flag when q > 0.1587."""
    records = []
    # 30 baseline years (1981-2010)
    # In month 1: 6 out of 30 years have 0 mm -> q = 6/30 = 0.20 > 0.1587
    # Theoretical lower bound = Phi^-1(0.20) = -0.8416 > -1.0
    for y in range(1981, 2011):
        for m in range(1, 13):
            val = 0.0 if (m == 1 and (y - 1981) < 6) else (20.0 + (y - 1981) * 0.5)
            records.append({"year": y, "month": m, "total_mm": val})
    df = pd.DataFrame(records)

    spi_df = compute_spi(df, scale=1, baseline_start_year=1981, baseline_end_year=2010)
    m1_rows = spi_df[spi_df["month"] == 1]

    expected_lb = float(stats.norm.ppf(0.20))
    assert np.isclose(m1_rows["SPI_1_lower_bound"].iloc[0], expected_lb, atol=1e-4)
    assert bool(m1_rows["SPI_1_drought_limited"].iloc[0]) is True
    assert "Drought Detection Structurally Truncated" in m1_rows["SPI_1_status"].iloc[0]

    # For zero-precipitation years in month 1, category must indicate truncated bound
    zero_rows = m1_rows[m1_rows["total_mm"] == 0.0]
    assert len(zero_rows) == 6
    assert np.isclose(zero_rows["SPI_1"].iloc[0], expected_lb, atol=1e-4)
    assert "Drought-limited bound" in zero_rows["SPI_1_category"].iloc[0]


def test_precipitation_percent_of_normal_reliability_flag() -> None:
    """Verifies that Percent of Normal is flagged as Low Reliability when baseline mean < 1.0 mm."""
    records = []
    for y in range(1981, 2011):
        for m in range(1, 13):
            # Month 7 has trace mean 0.2 mm/month, Month 1 has 30 mm/month
            val = 0.2 if m == 7 else 30.0
            records.append({"year": y, "month": m, "variable": "PRECTOTCORR", "value": val / 30.0})
    df = pd.DataFrame(records)

    p_anom, summary = compute_precipitation_anomalies(df)
    m7_rows = p_anom[p_anom["month"] == 7]
    m1_rows = p_anom[p_anom["month"] == 1]

    assert m7_rows["baseline_mean"].iloc[0] < 1.0
    assert (m7_rows["percent_of_normal_reliable"] == False).all()
    assert (m7_rows["percent_of_normal_flag"] == "Low Reliability (Baseline < 1.0 mm)").all()

    assert m1_rows["baseline_mean"].iloc[0] >= 1.0
    assert (m1_rows["percent_of_normal_reliable"] == True).all()
    assert (m1_rows["percent_of_normal_flag"] == "Reliable").all()


def test_detect_persistent_anomalies_calendar_continuity_with_gaps() -> None:
    """Verifies that temporal/calendar gaps terminate spells instead of bridging across missing months."""
    # Jan 2000 (Z=2), Feb 2000 (Z=2) [duration 2] -> GAP of 2 months -> May, Jun, Jul 2000 (Z=2) [duration 3]
    df = pd.DataFrame({
        "year": [2000, 2000, 2000, 2000, 2000],
        "month": [1, 2, 5, 6, 7],
        "z_score": [2.0, 2.0, 2.0, 2.0, 2.0]
    })
    spells = detect_persistent_anomalies(df, z_threshold=1.0, min_duration=3, spell_type="warm")
    # Only the 3-month run (May-Jul) should be detected as a spell; Jan-Feb was only 2 months and broken by gap
    assert len(spells) == 1
    assert spells.iloc[0]["start_date"] == "2000-05"
    assert spells.iloc[0]["end_date"] == "2000-07"
    assert spells.iloc[0]["duration_months"] == 3


def test_spi_boundary_precision_and_unreliability_distinction() -> None:
    """
    Verifies that the drought-limited SPI condition evaluates exact Phi(-1.0) boundary,
    preventing misclassification from rounded 0.1587 constant, and ensures that
    Undefined SPI remains strictly distinct from Limited Drought Detection.
    """
    records = []
    # 30 baseline years (1981-2010)
    # Month 1: q = 0.20 > Phi(-1.0) -> Drought-limited (valid Gamma fit, but lower bound > -1.0)
    # Month 2: q = 0.00 -> Normal drought detection (lower bound <= -1.0)
    # Month 3: q = 0.60 >= 0.50 -> Aridity Guardrail triggers (Undefined SPI, NaN)
    for y in range(1981, 2011):
        for m in range(1, 13):
            if m == 1:
                val = 0.0 if (y - 1981) < 6 else (25.0 + (y - 1981) * 0.5)
            elif m == 2:
                val = 30.0 + (y - 1981) * 0.5
            elif m == 3:
                val = 0.0 if (y - 1981) < 18 else (15.0 + (y - 1981) * 0.5)
            else:
                val = 20.0
            records.append({"year": y, "month": m, "total_mm": val})
    df = pd.DataFrame(records)

    spi_df = compute_spi(df, scale=1, baseline_start_year=1981, baseline_end_year=2010)

    # 1. Month 1: Drought Limited (q = 6/30 = 0.20)
    m1 = spi_df[spi_df["month"] == 1]
    assert (m1["SPI_1_drought_limited"] == True).all()
    assert (m1["SPI_1_lower_bound"] > -1.0).all()
    assert m1["SPI_1"].notna().all()  # SPI is defined and finite!
    assert "Drought Detection Structurally Truncated" in m1["SPI_1_status"].iloc[0]
    # Zero rain months are labeled with drought-limited bound, NOT "Near Normal"
    zero_rain_m1 = m1[m1["total_mm"] == 0.0]
    assert len(zero_rain_m1) == 6
    assert (zero_rain_m1["SPI_1_category"].str.startswith("Zero rain (Drought-limited bound")).all()

    # 2. Month 2: Normal (q = 0.00)
    m2 = spi_df[spi_df["month"] == 2]
    assert (m2["SPI_1_drought_limited"] == False).all()
    assert (m2["SPI_1_lower_bound"] <= -1.0).all()
    assert m2["SPI_1"].notna().all()
    assert (m2["SPI_1_status"] == "Valid").all()

    # 3. Month 3: Undefined SPI due to hyper-aridity (q = 18/30 = 0.60 >= 0.50)
    m3 = spi_df[spi_df["month"] == 3]
    assert m3["SPI_1"].isna().all()  # Must be NaN!
    assert (m3["SPI_1_category"] == "Aridity Bound / Unreliable").all()
    assert "Undefined / Low reliability" in m3["SPI_1_status"].iloc[0]


def test_detect_persistent_anomalies_missing_month_restart_scenario() -> None:
    """
    Verifies the exact audit scenario:
    - January: anomaly
    - February: anomaly
    - March: missing
    - April: anomaly
    - May: anomaly

    Asserts that:
    1. January-February terminates cleanly as a 2-month spell.
    2. April-May begins fresh as a separate 2-month spell.
    3. Neither spell bridges across the missing March.
    """
    df = pd.DataFrame({
        "year": [2000, 2000, 2000, 2000],
        "month": [1, 2, 4, 5],
        "z_score": [2.0, 2.1, 2.2, 2.3]
    })

    # With min_duration=2, both should be detected as distinct spells
    spells = detect_persistent_anomalies(df, z_threshold=1.0, min_duration=2, spell_type="warm")
    assert len(spells) == 2

    # Spell 1: Jan to Feb
    s1 = spells[spells["start_date"] == "2000-01"].iloc[0]
    assert s1["end_date"] == "2000-02"
    assert s1["duration_months"] == 2

    # Spell 2: Apr to May
    s2 = spells[spells["start_date"] == "2000-04"].iloc[0]
    assert s2["end_date"] == "2000-05"
    assert s2["duration_months"] == 2

    # With min_duration=3, neither qualifies (both duration 2 < 3)
    spells3 = detect_persistent_anomalies(df, z_threshold=1.0, min_duration=3, spell_type="warm")
    assert len(spells3) == 0


