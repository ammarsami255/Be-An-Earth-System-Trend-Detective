"""
End-to-End Integration Tests with Real NASA Earth Science Data
EarthTrend Explorer - NASA Space Apps 2026
"""

import os
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import compute_anomalies_and_zscores, decadal_extreme_frequency, get_top_ranked_extremes
from src.data_loader import auto_load_dataset, validate_variable_data
from src.preprocessing import aggregate_to_annual, check_and_deduplicate_timestamps, handle_missing_values
from src.trend_analysis import (
    compute_autocorrelation,
    compute_linear_regression,
    compute_sens_slope,
    mann_kendall_test,
    seasonal_mann_kendall_test,
    synthesize_scientific_answers,
)
from src.visualization import (
    plot_anomaly_bars,
    plot_autocorrelation_acf,
    plot_climatology_heatmap,
    plot_decadal_shift_bars,
    plot_regional_comparison,
    plot_timeseries_with_trend,
)


@pytest.fixture
def real_cairo_dataset():
    file_path = os.path.join("data", "raw", "nasa_power_cairo_1981_2023.csv")
    assert os.path.exists(file_path), "Cairo NASA raw dataset must exist on disk."
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    df, meta = auto_load_dataset(content, "Greater Cairo")
    return df, meta


def test_full_pipeline_cairo_t2m(real_cairo_dataset):
    df, meta = real_cairo_dataset
    assert meta["location"] == "Greater Cairo"
    assert "T2M" in meta["available_variables"]

    # 1. Variable extraction & validation
    t2m_df = df[df["variable"] == "T2M"].copy()
    val_info = validate_variable_data(df, "T2M")
    assert val_info["valid"] is True
    assert val_info["valid_records"] == 516
    assert val_info["missing_records"] == 0

    # 2. Preprocessing
    clean_df, dupe_rep = check_and_deduplicate_timestamps(t2m_df)
    assert dupe_rep["has_duplicates"] is False

    clean_df, miss_rep = handle_missing_values(clean_df)
    assert miss_rep["missing_count"] == 0

    # 3. Annual aggregation
    ann_df = aggregate_to_annual(clean_df)
    assert len(ann_df) == 43  # 1981 to 2023

    # 4. Statistical Trend Testing (Annual)
    mk_ann = mann_kendall_test(ann_df["value"])
    assert mk_ann["trend"] == "increasing"
    assert mk_ann["is_significant"] is True
    assert mk_ann["p_value"] < 0.001

    sen_ann = compute_sens_slope(ann_df["value"], ann_df["year"])
    assert sen_ann["slope"] > 0.02  # Positive warming slope (~ +0.029 °C/year)
    assert sen_ann["slope_ci_lower"] < sen_ann["slope"] < sen_ann["slope_ci_upper"]

    ols_ann = compute_linear_regression(ann_df["value"], ann_df["year"])
    assert ols_ann["slope"] > 0.02
    assert ols_ann["r_squared"] > 0.40

    acf_ann = compute_autocorrelation(ann_df["value"])
    assert len(acf_ann["acf_values"]) == 13

    # 5. Seasonal Mann-Kendall (Monthly)
    smk_res = seasonal_mann_kendall_test(clean_df["value"], period=12)
    assert smk_res["trend"] == "increasing"
    assert smk_res["is_significant"] is True

    # 5b. Hamed & Rao Modified Mann-Kendall and Month-by-Month breakdown
    from src.trend_analysis import compute_calendar_month_breakdown, hamed_rao_modified_mann_kendall
    hr_res = hamed_rao_modified_mann_kendall(ann_df["value"])
    assert hr_res["trend"] == "increasing"
    assert hr_res["is_significant"] is True

    cal_df = compute_calendar_month_breakdown(clean_df)
    assert len(cal_df) == 12
    # Verify nominal significance vs multiple-testing adjusted significance
    assert cal_df["is_significant"].sum() == 8
    assert cal_df["sig_nominal"].sum() == 8
    assert cal_df["sig_bh"].sum() == 6
    assert cal_df["sig_holm"].sum() == 5

    # 6. Synthesis of 4 questions
    answers = synthesize_scientific_answers(
        mk_res=mk_ann,
        sen_res=sen_ann,
        ols_res=ols_ann,
        acf_res=acf_ann,
        variable_name="Temperature at 2 Meters",
        units="°C",
        location_name="Greater Cairo",
        start_year=1981,
        end_year=2023,
        time_resolution="annual",
    )
    assert "Temperature at 2 Meters" in answers["q1_what"]
    assert "Greater Cairo" in answers["q2_where"]
    assert "+0.029" in answers["q3_how_much"] or "+0.0" in answers["q3_how_much"]
    assert "YES" in answers["q4_is_significant"]

    # 7. Anomaly Detection
    anom_df, anom_summary = compute_anomalies_and_zscores(clean_df, baseline_start_year=1981, baseline_end_year=2010)
    assert len(anom_df) == 516
    assert "anomaly" in anom_df.columns
    assert "z_score" in anom_df.columns

    dec_df = decadal_extreme_frequency(anom_df, z_threshold=1.5)
    assert len(dec_df) >= 4  # 1980s, 1990s, 2000s, 2010s, 2020s

    top_ranks = get_top_ranked_extremes(anom_df, top_n=5)
    assert len(top_ranks["hottest_anomalies"]) == 5

    # 8. Visualization figures smoke test
    fig_ts = plot_timeseries_with_trend(ann_df, "T2M", "°C", "Greater Cairo", time_col="year", sen_slope_info=sen_ann, ols_info=ols_ann)
    assert fig_ts is not None

    fig_heat = plot_climatology_heatmap(clean_df, "T2M", "°C", "Greater Cairo")
    assert fig_heat is not None

    fig_anom = plot_anomaly_bars(anom_df, "T2M", "°C", "Greater Cairo")
    assert fig_anom is not None

    fig_acf = plot_autocorrelation_acf(acf_ann)
    assert fig_acf is not None

    fig_dec = plot_decadal_shift_bars(dec_df)
    assert fig_dec is not None
