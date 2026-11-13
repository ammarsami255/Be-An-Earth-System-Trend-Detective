"""
EarthTrend Explorer - Climate Anomaly Detection Engine
NASA Space Apps Challenge 2026: "Be An Earth System Trend Detective!"
Team: Machine Without Learning

Scientifically sound climate anomaly detection based on standard WMO climatological baselines:
1. WMO Standard Climatological Baseline: 1981–2010 (30-year normal period).
2. Temperature Anomaly Engine:
   - Monthly absolute departures: A_t = T_t - μ_m
   - Standardized Z-scores: Z_t = (T_t - μ_m) / σ_m
   - Distinct treatment of T2M (mean), T2M_MAX (monthly extreme max), and T2M_MIN (monthly extreme min)
   - Persistent anomaly detection (persistent warm and cold spells / run lengths ≥ 3 months)
   - Annual aggregation and extreme classification (|Z| ≥ 1.5, 2.0, 3.0)
3. Precipitation Anomaly Engine:
   - Rate-to-accumulation conversion: PRECTOTCORR mm/day -> mm/month (leap-year correct)
   - Monthly, wet-season (Oct–Mar), and annual accumulation departures
   - Percent of Normal: 100 * P_t / μ_P,m
   - Zero-inflation diagnostics: Flags hyper-arid distributions where Gaussian Z-scores collapse
4. Standardized Precipitation Index (SPI) Engine:
   - McKee et al. (1993) / Edwards & McKee (1997) multi-scalar drought index (SPI-3, SPI-6, SPI-12)
   - 2-parameter Gamma distribution fitting with zero-precipitation probability correction:
     H(x) = q + (1 - q) * G(x),  SPI = Φ^(-1)(H(x))
   - Scientific aridity guardrail: Flags SPI as undefined/unreliable when baseline q ≥ 0.50
     or non-zero count < 5 (e.g. Aswan or hyper-arid summer months)
5. Compound Extremes & Cross-Variable Analysis:
   - Compound Hot-Dry and Cold-Wet classification
   - Pearson (r) and Spearman (rho) cross-correlation with effective degrees of freedom (N_eff)
   - Lagged cross-correlation (precipitation leading temperature by k months)
   - Cross-regional anomaly coherence across Greater Cairo, Alexandria, and Aswan
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.rainfall_analysis import (
    NORMAL_PERIOD as WMO_BASELINE_PERIOD,
    RAIN_VARIABLE,
    WET_SEASON_MONTHS,
    monthly_rainfall_totals,
)

# WMO 30-Year Standard Baseline
DEFAULT_BASELINE_START: int = 1981
DEFAULT_BASELINE_END: int = 2010
MONTH_ABBR: List[str] = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Standard WMO / IPCC Anomaly Classification Brackets
Z_CLASSIFICATION_BRACKETS = [
    (3.0, float("inf"), "Extreme Warm (+3σ)", "#8B0000"),
    (2.0, 3.0, "Severe Warm (+2σ)", "#DC143C"),
    (1.0, 2.0, "Moderate Warm (+1σ)", "#FF7F50"),
    (-1.0, 1.0, "Near Normal", "#708090"),
    (-2.0, -1.0, "Moderate Cold (-1σ)", "#6495ED"),
    (-3.0, -2.0, "Severe Cold (-2σ)", "#4169E1"),
    (float("-inf"), -3.0, "Extreme Cold (-3σ)", "#00008B"),
]

# WMO Reference Drought Index Categories (WMO-No. 1090)
SPI_CLASSIFICATION_BRACKETS = [
    (2.0, float("inf"), "Extremely Wet (SPI ≥ +2.0)", "#08306b"),
    (1.5, 2.0, "Very Wet (+1.5 ≤ SPI < +2.0)", "#2171b5"),
    (1.0, 1.5, "Moderately Wet (+1.0 ≤ SPI < +1.5)", "#6baed6"),
    (-1.0, 1.0, "Near Normal (-1.0 < SPI < +1.0)", "#708090"),
    (-1.5, -1.0, "Moderately Dry (-1.5 < SPI ≤ -1.0)", "#fd8d3c"),
    (-2.0, -1.5, "Severely Dry (-2.0 < SPI ≤ -1.5)", "#e6550d"),
    (float("-inf"), -2.0, "Extremely Dry (SPI ≤ -2.0)", "#a63603"),
]


# ==============================================================================
# SECTION 1: TEMPERATURE ANOMALY ENGINE
# ==============================================================================

def compute_temperature_anomalies(
    df: pd.DataFrame,
    variable: str = "T2M",
    value_col: str = "value",
    time_col: str = "date",
    baseline_start_year: int = DEFAULT_BASELINE_START,
    baseline_end_year: int = DEFAULT_BASELINE_END,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes monthly climatological temperature departures and standardized Z-scores.
    Evaluates each observation against the calendar-month baseline mean and standard deviation:
        A_{y, m} = T_{y, m} - μ_m
        Z_{y, m} = (T_{y, m} - μ_m) / σ_m

    Supports:
        - T2M: Monthly mean air temperature at 2 meters.
        - T2M_MAX: Monthly absolute hottest temperature (extreme max).
        - T2M_MIN: Monthly absolute coldest temperature (extreme min).
    """
    work_df = df.copy()
    if "variable" in work_df.columns:
        sub = work_df[work_df["variable"] == variable].dropna(subset=[value_col]).copy()
    else:
        sub = work_df.dropna(subset=[value_col]).copy()

    if sub.empty:
        raise ValueError(f"No records found for variable '{variable}'.")

    if "year" not in sub.columns or "month" not in sub.columns:
        dt = pd.to_datetime(sub[time_col])
        sub["year"] = dt.dt.year
        sub["month"] = dt.dt.month

    if "month_name" not in sub.columns:
        sub["month_name"] = sub["month"].apply(lambda m: MONTH_ABBR[int(m) - 1])

    # 1. Extract Baseline (1981–2010 by default)
    base_mask = (sub["year"] >= baseline_start_year) & (sub["year"] <= baseline_end_year)
    base_df = sub[base_mask]
    baseline_years_count = base_df["year"].nunique()

    if baseline_years_count < 10:
        # Fallback if filtered dataset has very few baseline years
        base_df = sub
        baseline_notes = f"Warning: Dataset contains only {baseline_years_count} baseline years. Full period used."
    else:
        baseline_notes = f"Standard WMO 30-year normal ({baseline_start_year}–{baseline_end_year}, n={baseline_years_count} yrs)."

    # 2. Monthly Climatology: Mean μ_m and Standard Deviation σ_m per calendar month
    monthly_stats = base_df.groupby("month")[value_col].agg(
        baseline_mean="mean",
        baseline_std=lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        baseline_count="count",
    ).reset_index()

    merged = pd.merge(sub, monthly_stats, on="month", how="left")
    merged["anomaly"] = merged[value_col] - merged["baseline_mean"]
    merged["z_score"] = np.where(
        merged["baseline_std"] > 1e-9,
        merged["anomaly"] / merged["baseline_std"],
        0.0,
    )

    # 3. Classify event severity into WMO sigma brackets
    conditions = [
        merged["z_score"] >= 3.0,
        (merged["z_score"] >= 2.0) & (merged["z_score"] < 3.0),
        (merged["z_score"] >= 1.0) & (merged["z_score"] < 2.0),
        (merged["z_score"] > -1.0) & (merged["z_score"] < 1.0),
        (merged["z_score"] <= -1.0) & (merged["z_score"] > -2.0),
        (merged["z_score"] <= -2.0) & (merged["z_score"] > -3.0),
        merged["z_score"] <= -3.0,
    ]
    labels = [b[2] for b in Z_CLASSIFICATION_BRACKETS]
    merged["classification"] = np.select(conditions, labels, default="Near Normal")

    # 4. Sort and reset
    merged.sort_values(by=["year", "month"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # 5. Annual Aggregation
    # Complete years only (12 months)
    year_counts = merged.groupby("year")["month"].count()
    complete_years = year_counts[year_counts == 12].index
    ann_df = merged[merged["year"].isin(complete_years)].groupby("year").agg(
        annual_mean=(value_col, "mean"),
        annual_mean_anomaly=("anomaly", "mean"),
    ).reset_index()

    ann_base_mask = (ann_df["year"] >= baseline_start_year) & (ann_df["year"] <= baseline_end_year)
    ann_base_mean = float(ann_df.loc[ann_base_mask, "annual_mean"].mean()) if ann_base_mask.sum() > 0 else float(ann_df["annual_mean"].mean())
    ann_base_std = float(ann_df.loc[ann_base_mask, "annual_mean"].std(ddof=1)) if ann_base_mask.sum() > 1 else 1.0
    ann_df["annual_z_score"] = (ann_df["annual_mean"] - ann_base_mean) / (ann_base_std if ann_base_std > 1e-9 else 1.0)

    summary: Dict[str, Any] = {
        "variable": variable,
        "baseline_period": f"{baseline_start_year}–{baseline_end_year}",
        "baseline_years_count": int(baseline_years_count),
        "baseline_notes": baseline_notes,
        "total_observations": len(merged),
        "complete_years_count": len(complete_years),
        "mean_anomaly": float(merged["anomaly"].mean()),
        "std_anomaly": float(merged["anomaly"].std(ddof=1)) if len(merged) > 1 else 0.0,
        "max_positive_anomaly": float(merged["anomaly"].max()),
        "max_negative_anomaly": float(merged["anomaly"].min()),
        "extreme_warm_count_3sigma": int((merged["z_score"] >= 3.0).sum()),
        "severe_warm_count_2sigma": int(((merged["z_score"] >= 2.0) & (merged["z_score"] < 3.0)).sum()),
        "moderate_warm_count_1sigma": int(((merged["z_score"] >= 1.0) & (merged["z_score"] < 2.0)).sum()),
        "extreme_cold_count_3sigma": int((merged["z_score"] <= -3.0).sum()),
        "severe_cold_count_2sigma": int(((merged["z_score"] <= -2.0) & (merged["z_score"] > -3.0)).sum()),
        "moderate_cold_count_1sigma": int(((merged["z_score"] <= -1.0) & (merged["z_score"] > -2.0)).sum()),
        "monthly_baseline_table": monthly_stats.to_dict(orient="records"),
        "annual_anomalies": ann_df,
    }
    return merged, summary


def compute_anomalies_and_zscores(
    df: pd.DataFrame,
    value_col: str = "value",
    time_col: str = "date",
    baseline_start_year: int = DEFAULT_BASELINE_START,
    baseline_end_year: int = DEFAULT_BASELINE_END,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backwards-compatible wrapper around compute_temperature_anomalies.
    Preserves exact function signature for existing callers in app.py and tests.
    """
    var = "T2M"
    if "variable" in df.columns:
        unique_vars = df["variable"].dropna().unique()
        if len(unique_vars) == 1:
            var = str(unique_vars[0])
    return compute_temperature_anomalies(
        df=df,
        variable=var,
        value_col=value_col,
        time_col=time_col,
        baseline_start_year=baseline_start_year,
        baseline_end_year=baseline_end_year,
    )


# ==============================================================================
# SECTION 2: PERSISTENT ANOMALIES & CLIMATE SPELL DETECTION
# ==============================================================================

def detect_persistent_anomalies(
    anomaly_df: pd.DataFrame,
    z_threshold: float = 1.0,
    min_duration: int = 3,
    spell_type: str = "warm",
) -> pd.DataFrame:
    """
    Detects persistent climatic spells: runs of consecutive months with anomalous Z-scores.
    
    Parameters:
        anomaly_df: Output from compute_temperature_anomalies() containing 'year', 'month', 'z_score'.
        z_threshold: Critical Z-score threshold (e.g. 1.0 for moderate spell, 1.5 for severe spell).
        min_duration: Minimum consecutive months required to qualify as a persistent spell (default: 3).
        spell_type: 'warm' (Z >= +z_threshold) or 'cold' (Z <= -z_threshold).
    
    Returns:
        pd.DataFrame: Table of detected spells with start date, end date, duration, peak Z, mean Z.
    """
    df = anomaly_df.sort_values(by=["year", "month"]).reset_index(drop=True).copy()
    if "z_score" not in df.columns:
        raise ValueError("DataFrame must contain 'z_score' column.")

    if spell_type == "warm":
        mask = df["z_score"] >= z_threshold
    elif spell_type == "cold":
        mask = df["z_score"] <= -z_threshold
    else:
        raise ValueError(f"spell_type must be 'warm' or 'cold', got '{spell_type}'.")

    spells: List[Dict[str, Any]] = []
    in_spell = False
    start_idx = 0

    for i in range(len(df)):
        flag = bool(mask.iloc[i])
        if in_spell:
            # Check calendar-month continuity from previous row
            prev_y = int(df.loc[i - 1, "year"])
            prev_m = int(df.loc[i - 1, "month"])
            curr_y = int(df.loc[i, "year"])
            curr_m = int(df.loc[i, "month"])
            month_step = (curr_y - prev_y) * 12 + (curr_m - prev_m)

            if month_step != 1:
                # Temporal gap detected: terminate previous spell at i - 1
                duration = i - start_idx
                if duration >= min_duration:
                    sub_spell = df.iloc[start_idx:i]
                    peak_idx = sub_spell["z_score"].idxmax() if spell_type == "warm" else sub_spell["z_score"].idxmin()
                    spells.append({
                        "spell_type": spell_type.capitalize() + " Spell",
                        "start_year": int(sub_spell.iloc[0]["year"]),
                        "start_month": int(sub_spell.iloc[0]["month"]),
                        "start_date": f"{int(sub_spell.iloc[0]['year'])}-{int(sub_spell.iloc[0]['month']):02d}",
                        "end_year": int(sub_spell.iloc[-1]["year"]),
                        "end_month": int(sub_spell.iloc[-1]["month"]),
                        "end_date": f"{int(sub_spell.iloc[-1]['year'])}-{int(sub_spell.iloc[-1]['month']):02d}",
                        "duration_months": int(duration),
                        "mean_z_score": float(sub_spell["z_score"].mean()),
                        "peak_z_score": float(df.loc[peak_idx, "z_score"]),
                        "peak_date": f"{int(df.loc[peak_idx, 'year'])}-{int(df.loc[peak_idx, 'month']):02d}",
                        "cumulative_z_score": float(sub_spell["z_score"].sum()),
                    })
                # Check whether current row begins a new spell
                if flag:
                    in_spell = True
                    start_idx = i
                else:
                    in_spell = False
            elif not flag:
                # Contiguous, but threshold no longer satisfied: terminate spell at i - 1
                in_spell = False
                duration = i - start_idx
                if duration >= min_duration:
                    sub_spell = df.iloc[start_idx:i]
                    peak_idx = sub_spell["z_score"].idxmax() if spell_type == "warm" else sub_spell["z_score"].idxmin()
                    spells.append({
                        "spell_type": spell_type.capitalize() + " Spell",
                        "start_year": int(sub_spell.iloc[0]["year"]),
                        "start_month": int(sub_spell.iloc[0]["month"]),
                        "start_date": f"{int(sub_spell.iloc[0]['year'])}-{int(sub_spell.iloc[0]['month']):02d}",
                        "end_year": int(sub_spell.iloc[-1]["year"]),
                        "end_month": int(sub_spell.iloc[-1]["month"]),
                        "end_date": f"{int(sub_spell.iloc[-1]['year'])}-{int(sub_spell.iloc[-1]['month']):02d}",
                        "duration_months": int(duration),
                        "mean_z_score": float(sub_spell["z_score"].mean()),
                        "peak_z_score": float(df.loc[peak_idx, "z_score"]),
                        "peak_date": f"{int(df.loc[peak_idx, 'year'])}-{int(df.loc[peak_idx, 'month']):02d}",
                        "cumulative_z_score": float(sub_spell["z_score"].sum()),
                    })
        else:
            if flag:
                in_spell = True
                start_idx = i

    # Check active spell at end of series
    if in_spell:
        duration = len(df) - start_idx
        if duration >= min_duration:
            sub_spell = df.iloc[start_idx:]
            peak_idx = sub_spell["z_score"].idxmax() if spell_type == "warm" else sub_spell["z_score"].idxmin()
            spells.append({
                "spell_type": spell_type.capitalize() + " Spell (Active at End)",
                "start_year": int(sub_spell.iloc[0]["year"]),
                "start_month": int(sub_spell.iloc[0]["month"]),
                "start_date": f"{int(sub_spell.iloc[0]['year'])}-{int(sub_spell.iloc[0]['month']):02d}",
                "end_year": int(sub_spell.iloc[-1]["year"]),
                "end_month": int(sub_spell.iloc[-1]["month"]),
                "end_date": f"{int(sub_spell.iloc[-1]['year'])}-{int(sub_spell.iloc[-1]['month']):02d}",
                "duration_months": int(duration),
                "mean_z_score": float(sub_spell["z_score"].mean()),
                "peak_z_score": float(df.loc[peak_idx, "z_score"]),
                "peak_date": f"{int(df.loc[peak_idx, 'year'])}-{int(df.loc[peak_idx, 'month']):02d}",
                "cumulative_z_score": float(sub_spell["z_score"].sum()),
            })

    if not spells:
        return pd.DataFrame(columns=[
            "spell_type", "start_date", "end_date", "duration_months",
            "mean_z_score", "peak_z_score", "peak_date", "cumulative_z_score"
        ])

    res = pd.DataFrame(spells).sort_values(by="duration_months", ascending=False).reset_index(drop=True)
    return res


# ==============================================================================
# SECTION 3: PRECIPITATION ANOMALY & SPI ENGINE
# ==============================================================================

def compute_precipitation_anomalies(
    df: pd.DataFrame,
    variable: str = RAIN_VARIABLE,
    baseline_start_year: int = DEFAULT_BASELINE_START,
    baseline_end_year: int = DEFAULT_BASELINE_END,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes precipitation anomalies:
    1. Converts daily rates (mm/day) to monthly accumulated totals (mm/month, leap-year correct).
    2. Calculates monthly baseline totals and absolute departures: A_P = P - μ_P,m.
    3. Calculates Percent of Normal: 100 * P / μ_P,m.
    4. Evaluates zero-inflation diagnostics and flags hyper-arid regimes.
    """
    monthly_precip = monthly_rainfall_totals(df, variable=variable)
    monthly_precip.sort_values(by=["year", "month"], inplace=True)
    monthly_precip.reset_index(drop=True, inplace=True)

    # Climatological Baseline
    base_mask = (monthly_precip["year"] >= baseline_start_year) & (monthly_precip["year"] <= baseline_end_year)
    base_df = monthly_precip[base_mask]
    n_base_years = base_df["year"].nunique()

    # Monthly baseline statistics: mean total, zero frequency q, std dev
    base_stats = base_df.groupby("month").agg(
        baseline_mean=("total_mm", "mean"),
        baseline_std=("total_mm", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        baseline_zeros=("total_mm", lambda s: int((s == 0.0).sum())),
        baseline_count=("total_mm", "count"),
    ).reset_index()

    base_stats["baseline_zero_fraction"] = base_stats["baseline_zeros"] / base_stats["baseline_count"]

    merged = pd.merge(monthly_precip, base_stats, on="month", how="left")
    merged["anomaly_mm"] = merged["total_mm"] - merged["baseline_mean"]

    # Percent of Normal
    merged["percent_of_normal"] = np.where(
        merged["baseline_mean"] > 1e-4,
        (merged["total_mm"] / merged["baseline_mean"]) * 100.0,
        np.nan,
    )

    # Reliability flag: Percent of Normal is unstable when baseline climatological mean < 1.0 mm
    merged["percent_of_normal_reliable"] = merged["baseline_mean"] >= 1.0
    merged["percent_of_normal_flag"] = np.where(
        merged["baseline_mean"] < 1.0,
        "Low Reliability (Baseline < 1.0 mm)",
        "Reliable",
    )

    # Date string
    merged["date_str"] = merged.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)

    # Zero-inflation diagnostics
    total_months = len(merged)
    total_zeros = int((merged["total_mm"] == 0.0).sum())
    zero_fraction = total_zeros / total_months if total_months > 0 else 0.0

    # Wet-season aggregation (Oct–Mar)
    ws_records: List[Dict[str, Any]] = []
    unique_years = sorted(merged["year"].unique())
    for end_yr in unique_years:
        start_yr = end_yr - 1
        oct_dec = merged[(merged["year"] == start_yr) & (merged["month"].isin([10, 11, 12]))]["total_mm"]
        jan_mar = merged[(merged["year"] == end_yr) & (merged["month"].isin([1, 2, 3]))]["total_mm"]
        if len(oct_dec) == 3 and len(jan_mar) == 3:
            ws_records.append({
                "season_label": f"{start_yr}/{str(end_yr)[2:]}",
                "season_end_year": int(end_yr),
                "wet_season_total_mm": float(oct_dec.sum() + jan_mar.sum()),
            })
    ws_df = pd.DataFrame(ws_records)
    if not ws_df.empty:
        ws_base_mask = (ws_df["season_end_year"] >= baseline_start_year) & (ws_df["season_end_year"] <= baseline_end_year)
        ws_base_mean = float(ws_df.loc[ws_base_mask, "wet_season_total_mm"].mean()) if ws_base_mask.sum() > 0 else float(ws_df["wet_season_total_mm"].mean())
        ws_df["wet_season_anomaly_mm"] = ws_df["wet_season_total_mm"] - ws_base_mean
        ws_df["percent_of_normal"] = (ws_df["wet_season_total_mm"] / (ws_base_mean if ws_base_mean > 1e-4 else 1.0)) * 100.0

    # Annual totals
    ann_df = merged.groupby("year")["total_mm"].agg(["sum", "count"]).reset_index()
    ann_df = ann_df[ann_df["count"] == 12].rename(columns={"sum": "annual_total_mm"}).drop(columns=["count"])
    ann_base_mask = (ann_df["year"] >= baseline_start_year) & (ann_df["year"] <= baseline_end_year)
    ann_base_mean = float(ann_df.loc[ann_base_mask, "annual_total_mm"].mean()) if ann_base_mask.sum() > 0 else float(ann_df["annual_total_mm"].mean())
    ann_df["annual_anomaly_mm"] = ann_df["annual_total_mm"] - ann_base_mean
    ann_df["percent_of_normal"] = (ann_df["annual_total_mm"] / (ann_base_mean if ann_base_mean > 1e-4 else 1.0)) * 100.0

    summary: Dict[str, Any] = {
        "variable": variable,
        "baseline_period": f"{baseline_start_year}–{baseline_end_year}",
        "baseline_years_count": int(n_base_years),
        "total_months": int(total_months),
        "zero_months_count": int(total_zeros),
        "zero_fraction": float(zero_fraction),
        "is_zero_inflated": bool(zero_fraction >= 0.50),
        "mean_annual_baseline_mm": float(ann_base_mean),
        "monthly_baseline_table": base_stats.to_dict(orient="records"),
        "wet_season_anomalies": ws_df,
        "annual_anomalies": ann_df,
    }
    return merged, summary


def compute_spi(
    precip_df: pd.DataFrame,
    scale: int = 12,
    baseline_start_year: int = DEFAULT_BASELINE_START,
    baseline_end_year: int = DEFAULT_BASELINE_END,
    max_zero_fraction: float = 0.50,
    min_nonzero_count: int = 5,
) -> pd.DataFrame:
    """
    Computes the Standardized Precipitation Index (SPI) at rolling scale k (e.g. k=3, 6, 12):
    1. Computes rolling accumulated precipitation: x_t = sum_{i=0}^{k-1} P_{t-i}
    2. Stratifies by calendar month m.
    3. Calculates baseline empirical probability of zero precipitation: q = m_0 / N.
    4. Evaluates aridity reliability: If q >= max_zero_fraction or non-zero count < min_nonzero_count,
       flags SPI as statistically unreliable / undefined due to hyper-aridity.
    5. For valid series, fits 2-parameter Gamma distribution to positive values (x > 0) via MLE:
           g(x) = (1 / (β^α Γ(α))) * x^(α-1) * e^(-x/β)
    6. Computes mixed cumulative probability: H(x) = q + (1 - q) * G(x)
    7. Inverts H(x) to standard normal quantile: SPI = Φ^(-1)(H(x))
    """
    df = precip_df.sort_values(by=["year", "month"]).reset_index(drop=True).copy()
    if "total_mm" not in df.columns:
        raise ValueError("DataFrame must contain 'total_mm' column. Run compute_precipitation_anomalies first.")

    # Rolling sum of scale k
    roll_col = f"roll_{scale}_mm"
    df[roll_col] = df["total_mm"].rolling(scale).sum()

    spi_col = f"SPI_{scale}"
    status_col = f"SPI_{scale}_status"
    lb_col = f"SPI_{scale}_lower_bound"
    dl_col = f"SPI_{scale}_drought_limited"
    df[spi_col] = np.nan
    df[status_col] = "Insufficient history"
    df[lb_col] = np.nan
    df[dl_col] = False

    base_mask = (df["year"] >= baseline_start_year) & (df["year"] <= baseline_end_year)

    for m in range(1, 13):
        m_mask = df["month"] == m
        m_indices = df[m_mask].index

        # Baseline observations for month m
        base_m_series = df.loc[m_mask & base_mask, roll_col].dropna()
        n_base = len(base_m_series)

        if n_base < 10:
            df.loc[m_indices, status_col] = f"Insufficient baseline years (n={n_base})"
            continue

        zeros_count = int((base_m_series == 0.0).sum())
        q = zeros_count / n_base
        pos_base = base_m_series[base_m_series > 0.0]
        n_pos = len(pos_base)

        # Theoretical lower bound of SPI when precipitation is zero: SPI_min = Φ^(-1)(q)
        if q > 0.0:
            lower_bound = float(stats.norm.ppf(np.clip(q, 1e-7, 1.0 - 1e-7)))
        else:
            lower_bound = float(stats.norm.ppf(1e-7))
        df.loc[m_indices, lb_col] = lower_bound

        # Flag when drought detection is structurally limited (Phi^(-1)(q) > -1.0 <=> q > Phi(-1.0))
        is_drought_limited = bool(lower_bound > -1.0)
        df.loc[m_indices, dl_col] = is_drought_limited

        # Aridity check
        if q >= max_zero_fraction or n_pos < min_nonzero_count:
            df.loc[m_indices, status_col] = f"Undefined / Low reliability (q={q:.2f} >= {max_zero_fraction:.2f} or n_pos={n_pos} < {min_nonzero_count})"
            continue

        # Fit 2-parameter Gamma distribution to positive baseline values via MLE
        try:
            # Using scipy.stats.gamma.fit with floc=0 fixes location to zero
            alpha_hat, loc_hat, beta_hat = stats.gamma.fit(pos_base, floc=0)
            if alpha_hat <= 0 or beta_hat <= 0 or np.isnan(alpha_hat) or np.isnan(beta_hat):
                df.loc[m_indices, status_col] = "Gamma parameter estimation failed"
                continue
        except Exception as e:
            df.loc[m_indices, status_col] = f"Gamma fit error: {str(e)[:30]}"
            continue

        # Calculate SPI for all observations in month m
        for idx in m_indices:
            val = df.loc[idx, roll_col]
            if np.isnan(val):
                continue
            if val <= 0.0:
                H = q
            else:
                G = stats.gamma.cdf(val, alpha_hat, scale=beta_hat)
                H = q + (1.0 - q) * G

            # Clamp H to prevent numerical overflow in inverse normal CDF
            H_clamped = np.clip(H, 1e-7, 1.0 - 1e-7)
            spi_val = float(stats.norm.ppf(H_clamped))
            df.loc[idx, spi_col] = spi_val
            if is_drought_limited:
                df.loc[idx, status_col] = f"Valid (Drought Detection Structurally Truncated at {lower_bound:+.2f})"
            else:
                df.loc[idx, status_col] = "Valid"

    # Assign WMO classification labels
    cat_col = f"SPI_{scale}_category"
    df[cat_col] = "Near Normal"

    for low, high, label, _ in SPI_CLASSIFICATION_BRACKETS:
        cond = (df[spi_col] >= low) & (df[spi_col] < high)
        df.loc[cond, cat_col] = label

    # For zero-precipitation observations in drought-limited months, annotate category
    drought_lim_zero = (df[dl_col] == True) & (df[roll_col] <= 1e-5) & df[spi_col].notna()
    if drought_lim_zero.any():
        df.loc[drought_lim_zero, cat_col] = df.loc[drought_lim_zero].apply(
            lambda r: f"Zero rain (Drought-limited bound: {r[lb_col]:+.2f})", axis=1
        )

    df.loc[df[spi_col].isna(), cat_col] = np.where(
        df.loc[df[spi_col].isna(), status_col] == "Insufficient history",
        "Insufficient history",
        "Aridity Bound / Unreliable",
    )
    return df


# ==============================================================================
# SECTION 4: COMPOUND EXTREMES & CLIMATE SHIFTS
# ==============================================================================

def analyze_compound_anomalies(
    temp_anomaly_df: pd.DataFrame,
    precip_anomaly_df: pd.DataFrame,
    temp_z_col: str = "z_score",
    precip_col: str = "total_mm",
    pn_col: str = "percent_of_normal",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates compound climate anomalies:
    - Compound Hot & Dry: Z_temp >= +1.0 AND Percent of Normal <= 75%
    - Compound Hot & Wet: Z_temp >= +1.0 AND Percent of Normal >= 125%
    - Compound Cold & Dry: Z_temp <= -1.0 AND Percent of Normal <= 75%
    - Compound Cold & Wet: Z_temp <= -1.0 AND Percent of Normal >= 125%
    - Near Normal: All other months
    """
    t_sub = temp_anomaly_df[["year", "month", "anomaly", temp_z_col]].rename(
        columns={"anomaly": "t_anomaly", temp_z_col: "t_z_score"}
    )
    p_sub = precip_anomaly_df[["year", "month", "total_mm", "anomaly_mm", pn_col]].rename(
        columns={"total_mm": "p_total_mm", "anomaly_mm": "p_anomaly_mm", pn_col: "p_percent_normal"}
    )

    merged = pd.merge(t_sub, p_sub, on=["year", "month"], how="inner")
    merged["date_str"] = merged.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)

    cond_hot_dry = (merged["t_z_score"] >= 1.0) & (merged["p_percent_normal"] <= 75.0)
    cond_hot_wet = (merged["t_z_score"] >= 1.0) & (merged["p_percent_normal"] >= 125.0)
    cond_cold_dry = (merged["t_z_score"] <= -1.0) & (merged["p_percent_normal"] <= 75.0)
    cond_cold_wet = (merged["t_z_score"] <= -1.0) & (merged["p_percent_normal"] >= 125.0)

    conditions = [cond_hot_dry, cond_hot_wet, cond_cold_dry, cond_cold_wet]
    choices = [
        "Compound Hot & Dry",
        "Compound Hot & Wet",
        "Compound Cold & Dry",
        "Compound Cold & Wet",
    ]
    merged["compound_category"] = np.select(conditions, choices, default="Near Normal")

    # Decadal aggregation
    merged["decade"] = (merged["year"] // 10) * 10
    merged["decade_label"] = merged["decade"].astype(str) + "s"

    dec_grouped = merged.groupby("decade_label").agg(
        total_months=("year", "count"),
        hot_dry_count=("compound_category", lambda s: int((s == "Compound Hot & Dry").sum())),
        hot_wet_count=("compound_category", lambda s: int((s == "Compound Hot & Wet").sum())),
        cold_dry_count=("compound_category", lambda s: int((s == "Compound Cold & Dry").sum())),
        cold_wet_count=("compound_category", lambda s: int((s == "Compound Cold & Wet").sum())),
    ).reset_index()

    dec_grouped["hot_dry_pct"] = (dec_grouped["hot_dry_count"] / dec_grouped["total_months"]) * 100.0
    dec_grouped["hot_wet_pct"] = (dec_grouped["hot_wet_count"] / dec_grouped["total_months"]) * 100.0
    dec_grouped["cold_dry_pct"] = (dec_grouped["cold_dry_count"] / dec_grouped["total_months"]) * 100.0
    dec_grouped["cold_wet_pct"] = (dec_grouped["cold_wet_count"] / dec_grouped["total_months"]) * 100.0

    summary: Dict[str, Any] = {
        "total_months": len(merged),
        "total_hot_dry": int(cond_hot_dry.sum()),
        "total_hot_wet": int(cond_hot_wet.sum()),
        "total_cold_dry": int(cond_cold_dry.sum()),
        "total_cold_wet": int(cond_cold_wet.sum()),
        "decadal_shift": dec_grouped,
    }
    return merged, summary


# ==============================================================================
# SECTION 5: CROSS-VARIABLE & REGIONAL CORRELATION
# ==============================================================================

def cross_variable_anomaly_correlation(
    temp_anomaly_df: pd.DataFrame,
    precip_anomaly_df: pd.DataFrame,
    max_lag: int = 3,
) -> Dict[str, Any]:
    """
    Computes cross-correlation and lagged relationships between temperature anomalies
    and precipitation anomalies.
    
    Includes:
    - Overall Pearson (r) and Spearman rank (rho) correlation.
    - Effective sample size adjustment for temporal serial autocorrelation:
        N_eff = N * (1 - r1_x * r1_y) / (1 + r1_x * r1_y)
    - Seasonal stratification: Winter (Oct–Mar) vs. Summer (Apr–Sep).
    - Lagged cross-correlation: r(A_P(t), A_T(t + k)) for k in [-max_lag, +max_lag].
    """
    t_sub = temp_anomaly_df[["year", "month", "anomaly"]].rename(columns={"anomaly": "t_anom"})
    p_sub = precip_anomaly_df[["year", "month", "anomaly_mm"]].rename(columns={"anomaly_mm": "p_anom"})
    merged = pd.merge(t_sub, p_sub, on=["year", "month"]).dropna().sort_values(by=["year", "month"]).reset_index(drop=True)

    if len(merged) < 10:
        return {"error": "Insufficient overlapping observations for cross-variable correlation."}

    t_vals = merged["t_anom"].values
    p_vals = merged["p_anom"].values
    n = len(merged)

    # 1. Pearson and Spearman Correlation
    r_pearson, p_pearson = stats.pearsonr(t_vals, p_vals)
    rho_spearman, p_spearman = stats.spearmanr(t_vals, p_vals)

    # 2. Effective sample size adjustment (Quenouille 1947 / Pyper & Peterman 1998)
    r1_t = float(np.corrcoef(t_vals[:-1], t_vals[1:])[0, 1]) if n > 2 else 0.0
    r1_p = float(np.corrcoef(p_vals[:-1], p_vals[1:])[0, 1]) if n > 2 else 0.0
    denom = 1.0 + r1_t * r1_p
    n_eff = float(n * (1.0 - r1_t * r1_p) / denom) if denom > 1e-4 else float(n)
    n_eff = max(3.0, min(float(n), n_eff))

    # Adjusted p-value using t-distribution with n_eff - 2 df
    t_stat = r_pearson * np.sqrt((n_eff - 2.0) / max(1e-9, 1.0 - r_pearson**2))
    p_adjusted = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n_eff - 2.0)))

    # 3. Seasonal Stratification
    winter_mask = merged["month"].isin([10, 11, 12, 1, 2, 3])
    summer_mask = merged["month"].isin([4, 5, 6, 7, 8, 9])

    r_winter, p_winter = stats.pearsonr(merged.loc[winter_mask, "t_anom"], merged.loc[winter_mask, "p_anom"]) if winter_mask.sum() > 5 else (0.0, 1.0)
    r_summer, p_summer = stats.pearsonr(merged.loc[summer_mask, "t_anom"], merged.loc[summer_mask, "p_anom"]) if summer_mask.sum() > 5 else (0.0, 1.0)

    # 4. Lagged Cross-Correlation: Corr(P(t), T(t + k))
    # If k > 0: Precipitation leads Temperature
    # If k < 0: Temperature leads Precipitation
    lag_records: List[Dict[str, Any]] = []
    p_series = merged["p_anom"]
    t_series = merged["t_anom"]

    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            p_slice = p_series.iloc[:-k]
            t_slice = t_series.iloc[k:]
        elif k < 0:
            p_slice = p_series.iloc[-k:]
            t_slice = t_series.iloc[:k]
        else:
            p_slice = p_series
            t_slice = t_series

        r_lag, p_lag = stats.pearsonr(p_slice, t_slice)
        lag_records.append({
            "lag": int(k),
            "correlation": float(r_lag),
            "p_value": float(p_lag),
            "relationship": "P leads T" if k > 0 else ("T leads P" if k < 0 else "Concurrent"),
        })

    return {
        "sample_size": int(n),
        "effective_sample_size": float(n_eff),
        "pearson_r": float(r_pearson),
        "pearson_p_nominal": float(p_pearson),
        "pearson_p_adjusted": float(p_adjusted),
        "spearman_rho": float(rho_spearman),
        "spearman_p": float(p_spearman),
        "winter_pearson_r": float(r_winter),
        "winter_p": float(p_winter),
        "summer_pearson_r": float(r_summer),
        "summer_p": float(p_summer),
        "lagged_cross_correlations": lag_records,
        "bartlett_critical_bound": float(1.959964 / np.sqrt(n)),
    }


def compare_regional_anomalies(
    regional_dict: Dict[str, pd.DataFrame],
    anomaly_col: str = "anomaly",
) -> Dict[str, Any]:
    """
    Computes spatial coherence of anomalies across regions (e.g. Cairo, Alexandria, Aswan).
    Returns pairwise correlation matrix and the fraction of months where all regions
    share the identical anomaly sign (spatial concordance).
    """
    merged: Optional[pd.DataFrame] = None
    regions = list(regional_dict.keys())

    for reg in regions:
        df = regional_dict[reg][["year", "month", anomaly_col]].rename(columns={anomaly_col: reg})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=["year", "month"])

    if merged is None or len(merged) < 5:
        return {"error": "Insufficient overlapping regional records."}

    corr_df = merged[regions].corr(method="pearson")
    spearman_df = merged[regions].corr(method="spearman")

    # Spatial concordance: fraction of months where all regions have sgn(A) >= 0 or all sgn(A) <= 0
    signs = np.sign(merged[regions].values)
    all_positive = (signs >= 0).all(axis=1)
    all_negative = (signs <= 0).all(axis=1)
    concordant_months = int((all_positive | all_negative).sum())
    total_months = len(merged)
    concordance_rate = concordant_months / total_months if total_months > 0 else 0.0

    return {
        "regions": regions,
        "total_overlapping_months": total_months,
        "concordant_months": concordant_months,
        "spatial_concordance_rate": float(concordance_rate),
        "pearson_correlation_matrix": corr_df.to_dict(),
        "spearman_correlation_matrix": spearman_df.to_dict(),
    }


# ==============================================================================
# SECTION 6: HISTORICAL RANKINGS & DECADAL FREQUENCIES
# ==============================================================================

def decadal_extreme_frequency(
    anomaly_df: pd.DataFrame,
    z_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Computes the frequency and proportion of warm/cold extremes across decades:
    1980s, 1990s, 2000s, 2010s, 2020s.
    Demonstrates how the probability density of warm extremes has shifted over time.
    """
    df = anomaly_df.copy()
    if "z_score" not in df.columns:
        raise ValueError("DataFrame must contain 'z_score' column.")

    df["decade"] = (df["year"] // 10) * 10
    df["decade_label"] = df["decade"].astype(str) + "s"

    grouped = df.groupby("decade_label")
    res = grouped.agg(
        total_months=("year", "count"),
        warm_extremes=("z_score", lambda x: int((x >= z_threshold).sum())),
        cold_extremes=("z_score", lambda x: int((x <= -z_threshold).sum())),
        mean_anomaly=("anomaly", "mean"),
    ).reset_index()

    res["warm_extreme_pct"] = (res["warm_extremes"] / res["total_months"]) * 100.0
    res["cold_extreme_pct"] = (res["cold_extremes"] / res["total_months"]) * 100.0
    res["decade_start"] = res["decade_label"].str[:4].astype(int)
    res.sort_values(by="decade_start", inplace=True)
    res.reset_index(drop=True, inplace=True)
    return res


def get_top_ranked_extremes(
    df: pd.DataFrame,
    value_col: str = "value",
    anomaly_col: str = "anomaly",
    top_n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Returns ranked lists of warmest and coldest events.
    """
    work = df.copy()
    if "date" in work.columns:
        work["date_str"] = pd.to_datetime(work["date"]).dt.strftime("%Y-%m")
    elif "year" in work.columns and "month" in work.columns:
        work["date_str"] = work.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
    else:
        work["date_str"] = work["year"].astype(str)

    cols_to_keep = [c for c in ["date_str", "year", "month_name", value_col, anomaly_col, "z_score", "location"] if c in work.columns]

    hottest_anomalies = work.sort_values(by=anomaly_col, ascending=False).head(top_n)[cols_to_keep].reset_index(drop=True)
    coldest_anomalies = work.sort_values(by=anomaly_col, ascending=True).head(top_n)[cols_to_keep].reset_index(drop=True)

    hottest_absolute = work.sort_values(by=value_col, ascending=False).head(top_n)[cols_to_keep].reset_index(drop=True)
    coldest_absolute = work.sort_values(by=value_col, ascending=True).head(top_n)[cols_to_keep].reset_index(drop=True)

    return {
        "hottest_anomalies": hottest_anomalies,
        "coldest_anomalies": coldest_anomalies,
        "hottest_absolute": hottest_absolute,
        "coldest_absolute": coldest_absolute,
    }
