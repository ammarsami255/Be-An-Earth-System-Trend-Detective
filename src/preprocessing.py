"""
EarthTrend Explorer - Preprocessing Module
NASA Space Apps Challenge 2026

Handles data cleaning, missing value reporting, timestamp deduplication,
temporal filtering, seasonal deseasonalization, and temporal aggregations (monthly, annual).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def check_and_deduplicate_timestamps(
    df: pd.DataFrame,
    time_col: str = "date",
    strategy: str = "first",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detects and handles duplicate timestamps transparently.
    Preserves original data in audit report.
    """
    report: Dict[str, Any] = {
        "has_duplicates": False,
        "duplicate_count": 0,
        "duplicate_timestamps": [],
        "strategy_applied": strategy,
    }

    if time_col not in df.columns:
        return df, report

    dupes_mask = df.duplicated(subset=[time_col], keep=False)
    dupe_count = int(dupes_mask.sum())

    if dupe_count > 0:
        report["has_duplicates"] = True
        report["duplicate_count"] = dupe_count
        report["duplicate_timestamps"] = df.loc[dupes_mask, time_col].astype(str).tolist()

        if strategy == "first":
            clean_df = df.drop_duplicates(subset=[time_col], keep="first").copy()
        elif strategy == "last":
            clean_df = df.drop_duplicates(subset=[time_col], keep="last").copy()
        elif strategy == "mean":
            # Group by time_col and average numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            clean_df = df.groupby(time_col, as_index=False)[num_cols].mean()
            # Restore non-numeric columns from first
            non_num_cols = [c for c in df.columns if c not in num_cols and c != time_col]
            if non_num_cols:
                first_non_num = df.groupby(time_col, as_index=False)[non_num_cols].first()
                clean_df = pd.merge(clean_df, first_non_num, on=time_col)
        else:
            clean_df = df.copy()
    else:
        clean_df = df.copy()

    clean_df.sort_values(by=time_col, inplace=True)
    clean_df.reset_index(drop=True, inplace=True)
    return clean_df, report


def handle_missing_values(
    df: pd.DataFrame,
    value_col: str = "value",
    method: str = "drop",  # "drop", "interpolate", "forward_fill"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transparently identifies and treats missing values.
    Never alters data silently without logging.
    """
    total_len = len(df)
    missing_mask = df[value_col].isna() | (df[value_col] == -999.0)
    missing_count = int(missing_mask.sum())
    pct_missing = (missing_count / total_len * 100.0) if total_len > 0 else 0.0

    report: Dict[str, Any] = {
        "total_records": total_len,
        "missing_count": missing_count,
        "pct_missing": round(pct_missing, 2),
        "method_applied": method,
        "missing_dates": df.loc[missing_mask, "date"].astype(str).tolist() if "date" in df.columns else [],
    }

    clean_df = df.copy()
    clean_df.loc[missing_mask, value_col] = np.nan

    if method == "drop":
        clean_df = clean_df.dropna(subset=[value_col]).copy()
    elif method == "interpolate":
        # Linear interpolation for small gaps
        clean_df[value_col] = clean_df[value_col].interpolate(method="linear", limit=3)
        # Drop any remaining unfillable edge NaNs
        clean_df = clean_df.dropna(subset=[value_col]).copy()
    elif method == "forward_fill":
        clean_df[value_col] = clean_df[value_col].ffill().bfill()
        clean_df = clean_df.dropna(subset=[value_col]).copy()

    clean_df.reset_index(drop=True, inplace=True)
    report["clean_records"] = len(clean_df)
    return clean_df, report


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Union[str, pd.Timestamp, datetime],
    end_date: Union[str, pd.Timestamp, datetime],
    time_col: str = "date",
) -> pd.DataFrame:
    """
    Slices the time series within a selected date range.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    filtered = df[(df[time_col] >= start_ts) & (df[time_col] <= end_ts)].copy()
    filtered.sort_values(by=time_col, inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def aggregate_to_annual(
    df: pd.DataFrame,
    time_col: str = "date",
    value_col: str = "value",
    min_months_required: int = 10,
) -> pd.DataFrame:
    """
    Aggregates monthly data to annual mean/total.
    Requires at least `min_months_required` valid observations to produce an annual point
    to avoid incomplete year bias.
    """
    work_df = df.copy()
    if "year" not in work_df.columns:
        work_df["year"] = pd.to_datetime(work_df[time_col]).dt.year

    # Count valid months per year
    grp = work_df.groupby("year")
    annual_stats = grp.agg(
        valid_months=(value_col, "count"),
        annual_mean=(value_col, "mean"),
        annual_min=(value_col, "min"),
        annual_max=(value_col, "max"),
        annual_std=(value_col, "std"),
    ).reset_index()

    # Filter out years with insufficient data
    valid_years = annual_stats[annual_stats["valid_months"] >= min_months_required].copy()
    valid_years["date"] = pd.to_datetime(valid_years["year"].astype(str) + "-07-01")  # mid-year timestamp
    valid_years["value"] = valid_years["annual_mean"]

    # Inherit location and variable if present
    if "location" in df.columns and not df.empty:
        valid_years["location"] = df["location"].iloc[0]
    if "variable" in df.columns and not df.empty:
        valid_years["variable"] = df["variable"].iloc[0]
    if "units" in df.columns and not df.empty:
        valid_years["units"] = df["units"].iloc[0]

    valid_years.sort_values(by="year", inplace=True)
    valid_years.reset_index(drop=True, inplace=True)
    return valid_years


def calculate_climatology(
    df: pd.DataFrame,
    value_col: str = "value",
    time_col: str = "date",
    baseline_start_year: int = 1981,
    baseline_end_year: int = 2010,
) -> pd.DataFrame:
    """
    Computes standard 30-year WMO climatological baseline (1981-2010) or all-year baseline.
    Returns 12-month table with monthly mean, std, min, max.
    """
    work_df = df.copy()
    if "year" not in work_df.columns or "month" not in work_df.columns:
        dt = pd.to_datetime(work_df[time_col])
        work_df["year"] = dt.dt.year
        work_df["month"] = dt.dt.month

    # Filter baseline period if available
    base_mask = (work_df["year"] >= baseline_start_year) & (work_df["year"] <= baseline_end_year)
    base_df = work_df[base_mask] if base_mask.sum() > 0 else work_df

    climatology = base_df.groupby("month")[value_col].agg(
        clim_mean="mean",
        clim_std="std",
        clim_min="min",
        clim_max="max",
        clim_count="count",
    ).reset_index()

    month_names_map = {i: name for i, name in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
    climatology["month_name"] = climatology["month"].map(month_names_map)
    return climatology


def compute_monthly_anomalies(
    df: pd.DataFrame,
    value_col: str = "value",
    time_col: str = "date",
    baseline_start_year: int = 1981,
    baseline_end_year: int = 2010,
) -> pd.DataFrame:
    """
    Computes deseasonalized monthly anomalies:
    Anomaly = Actual_Value - Climatological_Monthly_Mean
    Z-Score = Anomaly / Climatological_Monthly_Std
    """
    work_df = df.copy()
    if "year" not in work_df.columns or "month" not in work_df.columns:
        dt = pd.to_datetime(work_df[time_col])
        work_df["year"] = dt.dt.year
        work_df["month"] = dt.dt.month

    clim = calculate_climatology(
        work_df,
        value_col=value_col,
        time_col=time_col,
        baseline_start_year=baseline_start_year,
        baseline_end_year=baseline_end_year,
    )

    merged = pd.merge(work_df, clim[["month", "clim_mean", "clim_std"]], on="month", how="left")
    merged["anomaly"] = merged[value_col] - merged["clim_mean"]
    merged["z_score"] = np.where(merged["clim_std"] > 0, merged["anomaly"] / merged["clim_std"], 0.0)

    merged.sort_values(by=time_col, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged
