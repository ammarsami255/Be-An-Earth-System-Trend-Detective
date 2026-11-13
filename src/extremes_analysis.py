"""
EarthTrend Explorer - Temperature Extremes Module
NASA Space Apps Challenge 2026

What the NASA POWER monthly variables really are:
- T2M_MAX = the HOTTEST 2-meter temperature reached within the month (absolute monthly extreme)
- T2M_MIN = the COLDEST 2-meter temperature reached within the month (absolute monthly extreme)
(Check: the NASA POWER 'ANN' column for T2M_MAX equals the maximum of the 12 monthly values.)

Questions answered for each region:
  1. Is the hottest value of the year getting hotter?           (annual max of T2M_MAX)
  2. Is the coldest value of the year getting warmer?           (annual min of T2M_MIN)
  3. Are unusually HOT months becoming more frequent?           (T2M_MAX above the 1981-2010
                                                                 90th percentile of that calendar month)
  4. Are unusually COLD months becoming less frequent?          (T2M_MIN below the 1981-2010
                                                                 10th percentile of that calendar month)
  5. Is the gap between monthly hottest and coldest shrinking?  (monthly extreme range T2M_MAX - T2M_MIN)

Percentile thresholds are computed per calendar month, so a "hot July" is judged against
other Julys, not against January. This mirrors the percentile approach of the WMO/ETCCDI
climate extremes indices (e.g. TX90p, TN10p), adapted to monthly data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.rainfall_analysis import pettitt_change_point_test
from src.trend_analysis import (
    compute_calendar_month_breakdown,
    compute_linear_regression,
    compute_sens_slope,
    hamed_rao_modified_mann_kendall,
    mann_kendall_test,
)

BASELINE = (1981, 2010)
HOT_PERCENTILE = 90.0
COLD_PERCENTILE = 10.0
MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

METRIC_INFO = {
    "annual_hottest": ("Hottest value of the year (max of T2M_MAX)", "°C"),
    "annual_coldest": ("Coldest value of the year (min of T2M_MIN)", "°C"),
    "hot_months": ("Unusually hot months per year (T2M_MAX > 1981–2010 P90)", "months"),
    "cold_months": ("Unusually cold months per year (T2M_MIN < 1981–2010 P10)", "months"),
    "extreme_range": ("Mean monthly extreme range (T2M_MAX − T2M_MIN)", "°C"),
}


def build_monthly_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins T2M_MAX and T2M_MIN into one row per month.
    Returns columns: ['year', 'month', 'month_name', 'tmax', 'tmin', 'range']
    """
    piv = (
        df[df["variable"].isin(["T2M_MAX", "T2M_MIN"])]
        .dropna(subset=["value"])
        .pivot_table(index=["year", "month"], columns="variable", values="value")
        .dropna()
        .reset_index()
    )
    if piv.empty or "T2M_MAX" not in piv or "T2M_MIN" not in piv:
        raise ValueError("Dataset must contain both T2M_MAX and T2M_MIN.")
    out = pd.DataFrame({
        "year": piv["year"].astype(int),
        "month": piv["month"].astype(int),
        "tmax": piv["T2M_MAX"].astype(float),
        "tmin": piv["T2M_MIN"].astype(float),
    })
    out["range"] = out["tmax"] - out["tmin"]
    out["month_name"] = [MONTH_ABBR[m - 1] for m in out["month"]]
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def monthly_percentile_thresholds(
    monthly: pd.DataFrame,
    baseline: Tuple[int, int] = BASELINE,
    hot_pct: float = HOT_PERCENTILE,
    cold_pct: float = COLD_PERCENTILE,
) -> pd.DataFrame:
    """
    Per-calendar-month thresholds from the baseline period:
    hot_threshold = P90 of T2M_MAX for that month, cold_threshold = P10 of T2M_MIN.
    Returns columns: ['month', 'month_name', 'hot_threshold', 'cold_threshold']
    """
    base = monthly[(monthly["year"] >= baseline[0]) & (monthly["year"] <= baseline[1])]
    if base.empty:
        raise ValueError(f"No data inside the baseline period {baseline}.")
    thr = base.groupby("month").agg(
        hot_threshold=("tmax", lambda s: float(np.percentile(s, hot_pct))),
        cold_threshold=("tmin", lambda s: float(np.percentile(s, cold_pct))),
    ).reset_index()
    thr["month_name"] = [MONTH_ABBR[m - 1] for m in thr["month"]]
    return thr[["month", "month_name", "hot_threshold", "cold_threshold"]]


def annual_extreme_indices(monthly: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    One row per complete year (12 months) with all five extreme indices.
    Returns columns: ['year', 'annual_hottest', 'hottest_month', 'annual_coldest',
                      'coldest_month', 'hot_months', 'cold_months', 'extreme_range']
    """
    m = monthly.merge(thresholds[["month", "hot_threshold", "cold_threshold"]], on="month", how="left")
    m["is_hot"] = m["tmax"] > m["hot_threshold"]
    m["is_cold"] = m["tmin"] < m["cold_threshold"]

    rows = []
    for year, g in m.groupby("year"):
        if len(g) != 12:
            continue
        hot_row = g.loc[g["tmax"].idxmax()]
        cold_row = g.loc[g["tmin"].idxmin()]
        rows.append({
            "year": int(year),
            "annual_hottest": float(hot_row["tmax"]),
            "hottest_month": hot_row["month_name"],
            "annual_coldest": float(cold_row["tmin"]),
            "coldest_month": cold_row["month_name"],
            "hot_months": int(g["is_hot"].sum()),
            "cold_months": int(g["is_cold"].sum()),
            "extreme_range": float(g["range"].mean()),
        })
    return pd.DataFrame(rows)


COUNT_METRICS = {"hot_months", "cold_months"}


def extreme_trend_test(
    values: np.ndarray,
    years: np.ndarray,
    alpha: float = 0.05,
    is_count: bool = False,
) -> Dict[str, Any]:
    """
    Mann-Kendall + Hamed-Rao (significance), slope per decade with 95% CI (magnitude) and
    Pettitt (abrupt shift) for one yearly index.

    Magnitude: Sen's slope for continuous temperatures. For small integer COUNTS (0, 1, 2 ...)
    most pairwise slopes are exactly zero, so Sen's median slope collapses to 0 even when the
    trend is significant; for counts the OLS slope (with t-based 95% CI) is reported instead.
    Significance always comes from the rank-based Mann-Kendall / Hamed-Rao tests.
    """
    y = np.asarray(values, dtype=float)
    x = np.asarray(years, dtype=float)
    mk = mann_kendall_test(y, alpha=alpha)
    hr = hamed_rao_modified_mann_kendall(y, alpha=alpha)
    if is_count:
        ols = compute_linear_regression(y, x)
        sen = {"slope": ols["slope"], "intercept": ols["intercept"],
               "slope_ci_lower": ols["slope_ci_lower"], "slope_ci_upper": ols["slope_ci_upper"]}
        slope_method = "OLS (counts)"
    else:
        sen = compute_sens_slope(y, x, alpha=alpha)
        slope_method = "Sen's slope"
    cp = pettitt_change_point_test(y, x.astype(int), alpha=alpha)
    significant = bool(mk["is_significant"] and hr["is_significant"])
    if significant:
        verdict = "Significant increase" if mk["z_statistic"] > 0 else "Significant decrease"
    else:
        verdict = "No significant trend"
    return {
        "n_years": len(y),
        "mean": float(np.mean(y)),
        "sen_slope_per_year": float(sen["slope"]),
        "sen_intercept": float(sen["intercept"]),
        "sen_per_decade": float(sen["slope"]) * 10.0,
        "ci_per_decade": (float(sen["slope_ci_lower"]) * 10.0, float(sen["slope_ci_upper"]) * 10.0),
        "mk_z": float(mk["z_statistic"]),
        "mk_p": float(mk["p_value"]),
        "hr_p": float(hr["p_value"]),
        "is_significant": significant,
        "verdict": verdict,
        "slope_method": slope_method,
        "pettitt": cp,
    }


def decadal_extreme_counts(annual: pd.DataFrame) -> pd.DataFrame:
    """Total unusually hot / cold months per decade (only complete decades of 10 years are flagged)."""
    d = annual.copy()
    d["decade"] = (d["year"] // 10) * 10
    out = d.groupby("decade").agg(
        years=("year", "count"),
        hot_months=("hot_months", "sum"),
        cold_months=("cold_months", "sum"),
    ).reset_index()
    out["label"] = [f"{dec}s" + ("" if n == 10 else f" ({n} yrs)") for dec, n in zip(out["decade"], out["years"])]
    # Per-year rate so partial decades are comparable
    out["hot_per_year"] = out["hot_months"] / out["years"]
    out["cold_per_year"] = out["cold_months"] / out["years"]
    return out


def analyze_temperature_extremes(df: pd.DataFrame, location_name: str, alpha: float = 0.05) -> Dict[str, Any]:
    """Full temperature-extremes analysis for one region."""
    monthly = build_monthly_extremes(df)
    thresholds = monthly_percentile_thresholds(monthly)
    annual = annual_extreme_indices(monthly, thresholds)

    trends = {
        key: extreme_trend_test(annual[key].values, annual["year"].values, alpha, is_count=key in COUNT_METRICS)
        for key in METRIC_INFO
    }

    tmax_df = monthly.rename(columns={"tmax": "value"})[["year", "month", "value"]]
    tmin_df = monthly.rename(columns={"tmin": "value"})[["year", "month", "value"]]
    by_month_hot = compute_calendar_month_breakdown(tmax_df, alpha=alpha)
    by_month_cold = compute_calendar_month_breakdown(tmin_df, alpha=alpha)

    return {
        "location": location_name,
        "start_year": int(annual["year"].min()),
        "end_year": int(annual["year"].max()),
        "monthly": monthly,
        "thresholds": thresholds,
        "annual": annual,
        "trends": trends,
        "decades": decadal_extreme_counts(annual),
        "by_month_tmax": by_month_hot,
        "by_month_tmin": by_month_cold,
        "summary": summarize_extremes(location_name, annual, trends, by_month_hot, by_month_cold),
    }


def _fmt_trend(t: Dict[str, Any], unit: str) -> str:
    lo, hi = t["ci_per_decade"]
    return f"{t['sen_per_decade']:+.2f} {unit}/decade (95% CI [{lo:+.2f}, {hi:+.2f}], MK p = {t['mk_p']:.1e})"


def summarize_extremes(
    location_name: str,
    annual: pd.DataFrame,
    trends: Dict[str, Dict[str, Any]],
    by_month_hot: pd.DataFrame,
    by_month_cold: pd.DataFrame,
) -> Dict[str, str]:
    """Plain-language answers to the 4 challenge questions for temperature extremes."""
    start, end = int(annual["year"].min()), int(annual["year"].max())
    th, tc = trends["annual_hottest"], trends["annual_coldest"]
    hm, cm, rg = trends["hot_months"], trends["cold_months"], trends["extreme_range"]

    q1 = (
        "Temperature EXTREMES: the hottest (T2M_MAX) and coldest (T2M_MIN) values reached in each month, "
        f"NASA POWER / MERRA-2, {start}–{end}."
    )
    q2 = f"{location_name} (single MERRA-2 0.5° × 0.625° grid cell)."
    q3 = (
        f"Hottest value of the year: {_fmt_trend(th, '°C')}. "
        f"Coldest value of the year: {_fmt_trend(tc, '°C')}. "
        f"Unusually hot months: {hm['sen_per_decade']:+.2f} months/decade; "
        f"unusually cold months: {cm['sen_per_decade']:+.2f} months/decade. "
        f"Monthly extreme range: {rg['sen_per_decade']:+.2f} °C/decade."
    )

    sig_items = []
    for key, label in [
        ("annual_hottest", "hottest value of the year"),
        ("annual_coldest", "coldest value of the year"),
        ("hot_months", "number of unusually hot months"),
        ("cold_months", "number of unusually cold months"),
        ("extreme_range", "monthly extreme range"),
    ]:
        t = trends[key]
        word = "rising" if t["mk_z"] > 0 else "falling"
        sig_items.append(f"{label}: {'YES, ' + word if t['is_significant'] else 'no'} (MK p = {t['mk_p']:.3f}, Hamed-Rao p = {t['hr_p']:.3f})")
    hot_months_sig = by_month_hot.loc[by_month_hot["sig_bh"], "month_name"].tolist()
    cold_months_sig = by_month_cold.loc[by_month_cold["sig_bh"], "month_name"].tolist()
    q4 = (
        "Significant (both Mann-Kendall and Hamed-Rao, α = 0.05) — " + "; ".join(sig_items) + ". "
        f"Calendar months whose hottest value warms significantly after FDR correction: "
        f"{', '.join(hot_months_sig) if hot_months_sig else 'none'}. "
        f"Calendar months whose coldest value warms significantly after FDR correction: "
        f"{', '.join(cold_months_sig) if cold_months_sig else 'none'}."
    )
    return {"q1_what": q1, "q2_where": q2, "q3_how_much": q3, "q4_is_significant": q4}


def compare_regional_extremes(dfs: Dict[str, pd.DataFrame], alpha: float = 0.05) -> pd.DataFrame:
    """One row per region and extreme index."""
    rows = []
    for name, df in dfs.items():
        res = analyze_temperature_extremes(df, name, alpha)
        for key, (label, unit) in METRIC_INFO.items():
            t = res["trends"][key]
            lo, hi = t["ci_per_decade"]
            cp = t["pettitt"]
            rows.append({
                "Region": name,
                "Index": label,
                "Mean": round(t["mean"], 2),
                "Slope / decade": f"{t['sen_per_decade']:+.2f} {unit}",
                "Slope method": t["slope_method"],
                "95% CI": f"[{lo:+.2f}, {hi:+.2f}]",
                "MK p": float(f"{t['mk_p']:.2e}"),
                "Hamed-Rao p": float(f"{t['hr_p']:.2e}"),
                "Trend": t["verdict"],
                "Abrupt shift (Pettitt)": f"Yes, from {cp['new_regime_start']}" if cp["is_significant"] else "No",
            })
    return pd.DataFrame(rows)
