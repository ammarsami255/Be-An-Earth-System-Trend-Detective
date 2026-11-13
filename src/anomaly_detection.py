"""
EarthTrend Explorer - Anomaly Detection Module
NASA Space Apps Challenge 2026

Scientifically sound anomaly detection based on standard WMO climatological baselines:
- Monthly climatological anomalies (relative to 1981-2010 WMO normal)
- Standardized Z-score deviations (Z = (x - μ_m) / σ_m)
- Extreme heat/cold event threshold classification (|Z| ≥ 2.0 and |Z| ≥ 3.0)
- Decadal frequency shifts showing changes in the probability distribution of extremes
- Historical ranking of hottest and coldest months/years
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_anomalies_and_zscores(
    df: pd.DataFrame,
    value_col: str = "value",
    time_col: str = "date",
    baseline_start_year: int = 1981,
    baseline_end_year: int = 2010,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes monthly climatological anomalies and standardized Z-scores.
    Baseline defaults to the standard WMO 30-year climatological normal (1981-2010).
    """
    work_df = df.copy()
    if "year" not in work_df.columns or "month" not in work_df.columns:
        dt = pd.to_datetime(work_df[time_col])
        work_df["year"] = dt.dt.year
        work_df["month"] = dt.dt.month

    # Baseline selection
    mask_base = (work_df["year"] >= baseline_start_year) & (work_df["year"] <= baseline_end_year)
    base_df = work_df[mask_base] if mask_base.sum() > 0 else work_df

    clim_monthly = base_df.groupby("month")[value_col].agg(
        baseline_mean="mean",
        baseline_std="std",
    ).reset_index()

    merged = pd.merge(work_df, clim_monthly, on="month", how="left")
    merged["anomaly"] = merged[value_col] - merged["baseline_mean"]
    merged["z_score"] = np.where(merged["baseline_std"] > 0, merged["anomaly"] / merged["baseline_std"], 0.0)

    # Classify event severity
    # Severe Warm: Z >= 2.0; Extreme Warm: Z >= 3.0
    # Severe Cold: Z <= -2.0; Extreme Cold: Z <= -3.0
    conditions = [
        merged["z_score"] >= 3.0,
        (merged["z_score"] >= 2.0) & (merged["z_score"] < 3.0),
        (merged["z_score"] > -2.0) & (merged["z_score"] < 2.0),
        (merged["z_score"] <= -2.0) & (merged["z_score"] > -3.0),
        merged["z_score"] <= -3.0,
    ]
    choices = [
        "Extreme Warm (+3σ)",
        "Severe Warm (+2σ)",
        "Near Normal",
        "Severe Cold (-2σ)",
        "Extreme Cold (-3σ)",
    ]
    merged["classification"] = np.select(conditions, choices, default="Near Normal")

    merged.sort_values(by=time_col, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    summary: Dict[str, Any] = {
        "baseline_period": f"{baseline_start_year}–{baseline_end_year}",
        "total_observations": len(merged),
        "mean_anomaly": float(merged["anomaly"].mean()),
        "max_positive_anomaly": float(merged["anomaly"].max()),
        "max_negative_anomaly": float(merged["anomaly"].min()),
        "extreme_warm_count": int((merged["z_score"] >= 2.0).sum()),
        "extreme_cold_count": int((merged["z_score"] <= -2.0).sum()),
        "monthly_baseline_table": clim_monthly.to_dict(orient="records"),
    }
    return merged, summary


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
