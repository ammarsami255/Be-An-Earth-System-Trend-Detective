"""
EarthTrend Explorer - Rainfall Analysis Module
NASA Space Apps Challenge 2026

Precipitation needs different handling from temperature:
- NASA POWER reports PRECTOTCORR as a monthly MEAN RATE (mm/day). Rainfall is
  physically meaningful as an accumulated TOTAL, so we convert to mm/month by
  multiplying by the number of days in each month, then sum to mm/year.
- Egyptian rainfall is zero-inflated (many completely dry months) and heavily
  right-skewed (one storm can dominate a year). Non-parametric methods
  (Mann-Kendall, Sen's slope) are therefore used, and results are flagged as
  low-reliability when most values are zero or tied.
- Rain falls almost entirely in the cool season, so a wet-season total
  (October -> March, labelled by the year in which the season ends) is analysed
  alongside the calendar-year total.

Answers the 4 challenge questions for rainfall:
  1. What is changing?     -> annual total, wet-season total, number of wet months
  2. Where?                -> per MERRA-2 grid cell (Cairo, Alexandria, Aswan)
  3. How much?             -> Sen's slope in mm/decade and % of mean per decade
  4. Is it significant?    -> Mann-Kendall + Hamed-Rao (autocorrelation-corrected)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.trend_analysis import (
    compute_sens_slope,
    hamed_rao_modified_mann_kendall,
    mann_kendall_test,
)

RAIN_VARIABLE = "PRECTOTCORR"
WET_MONTH_THRESHOLD_MM = 1.0  # a month with >= 1 mm total counts as "wet"
WET_SEASON_MONTHS = (10, 11, 12, 1, 2, 3)  # October -> March
NORMAL_PERIOD = (1981, 2010)  # WMO standard climate normal
MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Above these fractions the trend test is reported but flagged as low reliability
MAX_ZERO_FRACTION = 0.5
MAX_TIED_FRACTION = 0.5


def monthly_rainfall_totals(df: pd.DataFrame, variable: str = RAIN_VARIABLE) -> pd.DataFrame:
    """
    Converts the tidy NASA POWER rainfall rate (mm/day) into monthly totals (mm/month).

    Returns columns: ['year', 'month', 'month_name', 'rate_mm_day', 'days', 'total_mm']
    """
    sub = df[df["variable"] == variable].dropna(subset=["value"]).copy()
    if sub.empty:
        raise ValueError(f"No '{variable}' rainfall records found in the dataset.")

    sub["days"] = [
        pd.Timestamp(year=int(y), month=int(m), day=1).days_in_month
        for y, m in zip(sub["year"], sub["month"])
    ]
    sub["rate_mm_day"] = sub["value"].astype(float)
    sub["total_mm"] = sub["rate_mm_day"] * sub["days"]
    sub["month_name"] = [MONTH_ABBR[int(m) - 1] for m in sub["month"]]

    out = sub[["year", "month", "month_name", "rate_mm_day", "days", "total_mm"]]
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def annual_rainfall_totals(monthly: pd.DataFrame, min_months: int = 12) -> pd.DataFrame:
    """
    Sums monthly totals into calendar-year totals. Years with fewer than `min_months`
    valid months are dropped (a partial year would look artificially dry).

    Returns columns: ['year', 'total_mm', 'wet_months', 'max_month_mm', 'n_months']
    """
    grouped = monthly.groupby("year")
    annual = pd.DataFrame({
        "total_mm": grouped["total_mm"].sum(),
        "wet_months": grouped["total_mm"].apply(lambda s: int((s >= WET_MONTH_THRESHOLD_MM).sum())),
        "max_month_mm": grouped["total_mm"].max(),
        "n_months": grouped["total_mm"].count(),
    }).reset_index()
    annual = annual[annual["n_months"] >= min_months].reset_index(drop=True)
    return annual


def wet_season_totals(monthly: pd.DataFrame, season_months: Tuple[int, ...] = WET_SEASON_MONTHS) -> pd.DataFrame:
    """
    Sums rainfall over the wet season (default October -> March). A season that starts
    in October of year Y-1 and ends in March of year Y is labelled Y. Only complete
    seasons (all months present) are kept, so the first season (which would need
    Oct-Dec 1980) is dropped.

    Returns columns: ['season_end_year', 'season_label', 'total_mm', 'n_months']
    """
    sub = monthly[monthly["month"].isin(season_months)].copy()
    start_month = season_months[0]
    crosses_year = any(m < start_month for m in season_months)
    sub["season_end_year"] = np.where(
        crosses_year & (sub["month"] >= start_month), sub["year"] + 1, sub["year"]
    )

    grouped = sub.groupby("season_end_year")
    seasons = pd.DataFrame({
        "total_mm": grouped["total_mm"].sum(),
        "n_months": grouped["total_mm"].count(),
    }).reset_index()
    seasons = seasons[seasons["n_months"] == len(season_months)].reset_index(drop=True)
    seasons["season_label"] = [
        f"{y - 1}/{str(y)[-2:]}" if crosses_year else str(y) for y in seasons["season_end_year"]
    ]
    return seasons[["season_end_year", "season_label", "total_mm", "n_months"]]


def monthly_rainfall_climatology(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Average total rainfall for each calendar month, plus how often that month is wet.
    Shows WHEN in the year it rains.

    Returns columns: ['month', 'month_name', 'mean_mm', 'median_mm', 'wet_fraction']
    """
    grouped = monthly.groupby("month")
    clim = pd.DataFrame({
        "mean_mm": grouped["total_mm"].mean(),
        "median_mm": grouped["total_mm"].median(),
        "wet_fraction": grouped["total_mm"].apply(lambda s: float((s >= WET_MONTH_THRESHOLD_MM).mean())),
    }).reset_index()
    clim["month_name"] = [MONTH_ABBR[int(m) - 1] for m in clim["month"]]
    return clim[["month", "month_name", "mean_mm", "median_mm", "wet_fraction"]]


def rainfall_percent_of_normal(
    annual: pd.DataFrame,
    normal_period: Tuple[int, int] = NORMAL_PERIOD,
) -> Tuple[pd.DataFrame, float]:
    """
    Expresses each year's total as a percentage of the 1981-2010 normal.
    Percent-of-normal is the standard rainfall anomaly measure for arid regions;
    Z-scores are avoided because rainfall is strongly skewed.

    Returns (DataFrame with added 'pct_of_normal' column, normal_mm).
    """
    base = annual[(annual["year"] >= normal_period[0]) & (annual["year"] <= normal_period[1])]
    normal_mm = float(base["total_mm"].mean()) if not base.empty else float("nan")
    out = annual.copy()
    if normal_mm and normal_mm > 0:
        out["pct_of_normal"] = out["total_mm"] / normal_mm * 100.0
    else:
        out["pct_of_normal"] = np.nan
    return out, normal_mm


def rainfall_trend_test(
    values: np.ndarray,
    years: np.ndarray,
    alpha: float = 0.05,
    is_amount: bool = True,
) -> Dict[str, Any]:
    """
    Runs Mann-Kendall, Hamed-Rao and Sen's slope on one yearly rainfall series and
    adds rainfall-specific reliability diagnostics.

    A trend is called significant only if BOTH Mann-Kendall and Hamed-Rao reject H0.
    `is_amount` is True for rainfall totals in mm and False for counts (wet months).
    """
    y = np.asarray(values, dtype=float)
    x = np.asarray(years, dtype=float)
    mask = ~np.isnan(y)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 10:
        raise ValueError(f"Rainfall trend test needs at least 10 years, got {n}.")

    mk = mann_kendall_test(y, alpha=alpha)
    hr = hamed_rao_modified_mann_kendall(y, alpha=alpha)
    sen = compute_sens_slope(y, x, alpha=alpha)

    mean_val = float(np.mean(y))
    zero_fraction = float(np.mean(y == 0.0))
    _, counts = np.unique(np.round(y, 6), return_counts=True)
    tied_fraction = float(counts[counts > 1].sum() / n)

    reliability_notes: List[str] = []
    if zero_fraction > MAX_ZERO_FRACTION:
        reliability_notes.append(f"{zero_fraction:.0%} of values are exactly zero")
    if tied_fraction > MAX_TIED_FRACTION:
        reliability_notes.append(f"{tied_fraction:.0%} of values are tied")
    if is_amount and mean_val < 5.0:
        reliability_notes.append("totals are extremely small (hyper-arid climate)")
    low_reliability = len(reliability_notes) > 0

    significant = bool(mk["is_significant"] and hr["is_significant"])
    slope_dec = sen["slope"] * 10.0
    pct_per_decade = (slope_dec / mean_val * 100.0) if mean_val > 0 else float("nan")

    if significant:
        direction = "wetter" if mk["z_statistic"] > 0 else "drier"
        verdict = f"Significant trend: getting {direction}"
    else:
        verdict = "No significant trend"

    return {
        "n_years": n,
        "mean": mean_val,
        "sen_slope_per_year": float(sen["slope"]),
        "sen_slope_per_decade": slope_dec,
        "sen_ci_per_decade": (float(sen["slope_ci_lower"]) * 10.0, float(sen["slope_ci_upper"]) * 10.0),
        "sen_intercept": float(sen["intercept"]),
        "pct_of_mean_per_decade": pct_per_decade,
        "mk_z": float(mk["z_statistic"]),
        "mk_p": float(mk["p_value"]),
        "mk_tau": float(mk["tau"]),
        "hr_z": float(hr["z_statistic"]),
        "hr_p": float(hr["p_value"]),
        "hr_factor": float(hr["variance_correction_factor"]),
        "is_significant": significant,
        "verdict": verdict,
        "zero_fraction": zero_fraction,
        "tied_fraction": tied_fraction,
        "low_reliability": low_reliability,
        "reliability_notes": reliability_notes,
    }


def pettitt_change_point_test(
    values: np.ndarray,
    years: np.ndarray,
    alpha: float = 0.05,
    n_perms: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Pettitt (1979) non-parametric test for a single abrupt shift (change point).

    Mann-Kendall asks "is there a GRADUAL monotonic trend?". Pettitt asks a different
    question: "did the series JUMP to a new level at some year?". A series can show no
    gradual trend yet a significant step change (e.g. flat, then suddenly higher).

        U_t = sum_{i<=t} sum_{j>t} sgn(x_i - x_j),   t = 1..n-1
        K   = max_t |U_t|
        p  ~= 2 * exp(-6 K^2 / (n^3 + n^2))          (asymptotic approximation, capped at 1)

    Standardized terminology:
    - change_year: final year of the pre-change regime (index t_idx).
    - new_regime_start: first year of the post-change regime (index t_idx + 1).

    Scientific attribution note:
    The test detects WHEN a statistical break occurred in the time series, NOT WHY.
    A detected statistical change in the MERRA-2 precipitation series must NOT be
    attributed conclusively to climate change, GPM/TRMM observing-system transitions,
    or satellite assimilation artifacts without independent ground-station validation.

    Permutation qualification note:
    Monte Carlo permutations evaluate significance under the null hypothesis of exchangeable
    observations. If serial correlation is present, exchangeability is violated and block
    permutation or pre-whitening is required.
    """
    y = np.asarray(values, dtype=float)
    x = np.asarray(years)
    mask = ~np.isnan(y)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 10:
        raise ValueError(f"Pettitt test needs at least 10 values, got {n}.")

    sgn = np.sign(y[:, None] - y[None, :])  # sgn[i, j] = sgn(x_i - x_j)
    u = np.array([sgn[: t + 1, t + 1 :].sum() for t in range(n - 1)], dtype=float)
    t_idx = int(np.argmax(np.abs(u)))
    k_stat = float(np.abs(u[t_idx]))
    p_val = float(min(1.0, 2.0 * np.exp(-6.0 * k_stat ** 2 / (n ** 3 + n ** 2))))

    # Optional Monte Carlo permutation test
    p_perm: Optional[float] = None
    if n_perms is not None and n_perms > 0:
        rng = np.random.default_rng(seed)
        ranks = pd.Series(y).rank().values
        t_range = np.arange(1, n)
        count_exceed = 0
        for _ in range(n_perms):
            p_ranks = rng.permutation(ranks)
            w = np.cumsum(p_ranks)[:-1]
            u_perm = 2.0 * w - t_range * (n + 1)
            if np.max(np.abs(u_perm)) >= k_stat:
                count_exceed += 1
        p_perm = float(count_exceed / n_perms)

    before, after = y[: t_idx + 1], y[t_idx + 1 :]
    mean_before, mean_after = float(np.mean(before)), float(np.mean(after))
    is_sig = bool(p_val < alpha)
    direction = "upward" if mean_after > mean_before else "downward"
    pct_change = (mean_after - mean_before) / mean_before * 100.0 if mean_before > 0 else float("nan")

    if is_sig:
        interpretation = (
            f"Significant {direction} shift starting in {int(x[t_idx + 1])} (p = {p_val:.4f}): "
            f"mean {mean_before:.1f} before vs {mean_after:.1f} after ({pct_change:+.0f}%)."
        )
    else:
        interpretation = f"No significant abrupt shift detected (most likely break {int(x[t_idx + 1])}, p = {p_val:.3f})."

    out_dict: Dict[str, Any] = {
        "test_name": "Pettitt Change-Point Test",
        "change_year": int(x[t_idx]),
        "new_regime_start": int(x[t_idx + 1]),
        "k_statistic": k_stat,
        "p_value": p_val,
        "p_permutation": p_perm,
        "is_significant": is_sig,
        "direction": direction,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "pct_change": pct_change,
        "n_before": len(before),
        "n_after": len(after),
        "u_series": u,
        "interpretation": interpretation,
    }
    return out_dict


def analyze_rainfall(df: pd.DataFrame, location_name: str, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Full rainfall analysis for one location. Returns all intermediate tables and the
    trend results for three metrics: annual total, wet-season total, wet-month count.
    """
    monthly = monthly_rainfall_totals(df)
    annual = annual_rainfall_totals(monthly)
    seasons = wet_season_totals(monthly)
    clim = monthly_rainfall_climatology(monthly)
    annual_pon, normal_mm = rainfall_percent_of_normal(annual)

    trends = {
        "annual_total": rainfall_trend_test(annual["total_mm"].values, annual["year"].values, alpha),
        "wet_season_total": rainfall_trend_test(seasons["total_mm"].values, seasons["season_end_year"].values, alpha),
        "wet_months": rainfall_trend_test(
            annual["wet_months"].values.astype(float), annual["year"].values, alpha, is_amount=False
        ),
    }
    change_point = pettitt_change_point_test(annual["total_mm"].values, annual["year"].values, alpha)

    return {
        "location": location_name,
        "start_year": int(annual["year"].min()),
        "end_year": int(annual["year"].max()),
        "monthly": monthly,
        "annual": annual_pon,
        "seasons": seasons,
        "climatology": clim,
        "normal_mm": normal_mm,
        "dry_month_fraction": float((monthly["total_mm"] < WET_MONTH_THRESHOLD_MM).mean()),
        "trends": trends,
        "change_point": change_point,
        "summary": summarize_rainfall_findings(location_name, annual_pon, trends, change_point),
    }


def summarize_rainfall_findings(
    location_name: str,
    annual: pd.DataFrame,
    trends: Dict[str, Dict[str, Any]],
    change_point: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Plain-language answers to the 4 challenge questions for rainfall."""
    t = trends["annual_total"]
    start, end = int(annual["year"].min()), int(annual["year"].max())

    q1 = (
        f"Total rainfall (mm/year), wet-season rainfall (Oct–Mar) and the number of wet months "
        f"(≥ {WET_MONTH_THRESHOLD_MM:.0f} mm) from NASA POWER / MERRA-2 PRECTOTCORR, {start}–{end}."
    )
    q2 = f"{location_name} (single MERRA-2 0.5° × 0.625° grid cell)."
    lo, hi = t["sen_ci_per_decade"]
    q3 = (
        f"Average rainfall is {t['mean']:.1f} mm/year. Sen's slope is {t['sen_slope_per_decade']:+.2f} mm/decade "
        f"({t['pct_of_mean_per_decade']:+.1f}% of the mean per decade), 95% CI [{lo:+.2f}, {hi:+.2f}] mm/decade."
    )
    if t["is_significant"]:
        q4 = (
            f"YES. Both Mann-Kendall (p = {t['mk_p']:.4f}) and Hamed-Rao (p = {t['hr_p']:.4f}) reject "
            "the null hypothesis of no trend."
        )
    else:
        crosses_zero = lo <= 0.0 <= hi
        q4 = (
            f"NO. Mann-Kendall p = {t['mk_p']:.3f}, Hamed-Rao p = {t['hr_p']:.3f} (both must be < 0.05)"
            + (", and the 95% confidence interval of the slope includes zero" if crosses_zero else "")
            + ". Year-to-year rainfall swings are too large to distinguish a long-term trend from natural variability."
        )
    if change_point is not None and change_point["is_significant"]:
        cp = change_point
        q4 += (
            f" HOWEVER, the Pettitt test finds a significant ABRUPT {cp['direction']} shift starting in "
            f"{cp['new_regime_start']} (p = {cp['p_value']:.4f}): average {cp['mean_before']:.1f} mm/year before "
            f"vs {cp['mean_after']:.1f} mm/year after. Rainfall did not change gradually — it jumped. "
            "The test cannot tell whether this is a real climate shift or a change in the reanalysis input data."
        )
    if t["low_reliability"]:
        q4 += " ⚠️ Low reliability: " + "; ".join(t["reliability_notes"]) + "."

    return {"q1_what": q1, "q2_where": q2, "q3_how_much": q3, "q4_is_significant": q4}


def compare_regional_rainfall(dfs: Dict[str, pd.DataFrame], alpha: float = 0.05) -> pd.DataFrame:
    """One summary row per region for the annual-total rainfall trend."""
    rows = []
    for name, df in dfs.items():
        res = analyze_rainfall(df, name, alpha)
        t = res["trends"]["annual_total"]
        s = res["trends"]["wet_season_total"]
        cp = res["change_point"]
        lo, hi = t["sen_ci_per_decade"]
        rows.append({
            "Region": name,
            "Mean (mm/yr)": round(t["mean"], 1),
            "Dry months (%)": round(res["dry_month_fraction"] * 100.0, 1),
            "Sen slope (mm/decade)": round(t["sen_slope_per_decade"], 2),
            "95% CI (mm/decade)": f"[{lo:+.2f}, {hi:+.2f}]",
            "% of mean / decade": round(t["pct_of_mean_per_decade"], 1),
            "MK p-value": round(t["mk_p"], 4),
            "Hamed-Rao p-value": round(t["hr_p"], 4),
            "Wet-season MK p-value": round(s["mk_p"], 4),
            "Annual trend": t["verdict"],
            "Abrupt shift (Pettitt)": f"Yes, from {cp['new_regime_start']}" if cp["is_significant"] else "No",
            "Pettitt p-value": round(cp["p_value"], 4),
            "Mean before → after (mm/yr)": f"{cp['mean_before']:.1f} → {cp['mean_after']:.1f}",
            "Reliability": "Low" if t["low_reliability"] else "OK",
        })
    return pd.DataFrame(rows)
