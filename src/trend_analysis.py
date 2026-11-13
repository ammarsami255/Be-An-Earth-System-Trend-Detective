"""
EarthTrend Explorer - Trend Analysis Module
NASA Space Apps Challenge 2026

Scientifically rigorous trend analysis engine:
- Non-parametric Mann-Kendall trend test
- Seasonal Mann-Kendall test (Hirsch-Slack)
- Theil-Sen Robust Slope Estimator with 95% Confidence Intervals
- Ordinary Least Squares (OLS) Linear Regression with R² and p-value
- Serial Autocorrelation (ACF) diagnostic
- Transparent scientific reporting answering the 4 core questions:
  1. What is changing?
  2. Where is it changing?
  3. How much is it changing?
  4. Is the observed trend statistically significant?
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

try:
    import pymannkendall as pmk
    HAS_PYMK = True
except ImportError:
    HAS_PYMK = False


def mann_kendall_test(
    data: Union[np.ndarray, List[float], pd.Series],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes the non-parametric Mann-Kendall monotonic trend test.
    H0: No monotonic trend in the time series (data is i.i.d.).
    H1: A monotonic upward or downward trend exists.
    """
    y = np.asarray(data, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 4:
        raise ValueError(f"Mann-Kendall test requires at least 4 observations, got {n}.")

    # Prefer pymannkendall if installed
    if HAS_PYMK:
        res = pmk.original_test(y, alpha=alpha)
        trend = str(res.trend)
        h = bool(res.h)
        p_val = float(res.p)
        z_val = float(res.z)
        tau_val = float(res.Tau)
        s_val = float(res.s)
        var_s = float(res.var_s)
        slope = float(res.slope)
    else:
        # High-precision SciPy/NumPy mathematical implementation
        # S = sum(sgn(yj - yk))
        diff = y[:, None] - y[None, :]
        sgn = np.sign(diff)
        # Sum upper triangle
        s_val = float(np.sum(np.triu(sgn, k=1)))

        # Unique values and tie correction
        unique_vals, counts = np.unique(y, return_counts=True)
        ties = counts[counts > 1]
        tie_term = np.sum(ties * (ties - 1) * (2 * ties + 5)) if len(ties) > 0 else 0.0

        var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

        if s_val > 0:
            z_val = (s_val - 1.0) / math.sqrt(var_s)
        elif s_val < 0:
            z_val = (s_val + 1.0) / math.sqrt(var_s)
        else:
            z_val = 0.0

        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_val)))
        h = bool(p_val < alpha)

        if h and z_val > 0:
            trend = "increasing"
        elif h and z_val < 0:
            trend = "decreasing"
        else:
            trend = "no trend"

        # Kendall's Tau
        d_val = 0.5 * n * (n - 1)
        tau_val = s_val / d_val if d_val > 0 else 0.0

        # Sen's slope
        indices = np.triu_indices(n, k=1)
        pairwise_slopes = (y[indices[1]] - y[indices[0]]) / (indices[1] - indices[0])
        slope = float(np.median(pairwise_slopes))

    # Significance interpretation string
    if h:
        sig_text = f"Statistically significant {trend} trend (p = {p_val:.4e} < {alpha})"
    else:
        sig_text = f"No statistically significant monotonic trend detected (p = {p_val:.4f} ≥ {alpha})"

    return {
        "test_name": "Mann-Kendall Monotonic Trend Test",
        "trend": trend,
        "is_significant": h,
        "p_value": p_val,
        "z_statistic": z_val,
        "tau": tau_val,
        "s_statistic": s_val,
        "var_s": var_s,
        "sen_slope": slope,
        "alpha": alpha,
        "sample_size": n,
        "interpretation": sig_text,
    }


def seasonal_mann_kendall_test(
    data: Union[np.ndarray, List[float], pd.Series],
    period: int = 12,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes Hirsch-Slack Seasonal Mann-Kendall test.
    Eliminates annual 12-month seasonality by comparing identical months across years.
    """
    y = np.asarray(data, dtype=float)
    y_clean = y[~np.isnan(y)]
    n = len(y_clean)

    if n < period * 2:
        return {
            "test_name": "Seasonal Mann-Kendall Test",
            "error": f"Insufficient data for Seasonal Mann-Kendall (requires at least {period * 2} points, got {n}).",
            "trend": "insufficient data",
            "is_significant": False,
            "p_value": 1.0,
            "sen_slope": 0.0,
        }

    if HAS_PYMK:
        res = pmk.seasonal_test(y_clean, period=period, alpha=alpha)
        trend = str(res.trend)
        h = bool(res.h)
        p_val = float(res.p)
        z_val = float(res.z)
        tau_val = float(res.Tau)
        s_val = float(res.s)
        var_s = float(res.var_s)
        slope = float(res.slope)
    else:
        # Fallback: compute seasonal subseries
        s_tot = 0.0
        var_tot = 0.0
        all_slopes = []
        for m in range(period):
            season_data = y[m::period]
            season_data = season_data[~np.isnan(season_data)]
            ns = len(season_data)
            if ns >= 2:
                # Subseries MK
                diff = season_data[:, None] - season_data[None, :]
                s_m = float(np.sum(np.triu(np.sign(diff), k=1)))
                var_m = (ns * (ns - 1) * (2 * ns + 5)) / 18.0
                s_tot += s_m
                var_tot += var_m

                idx = np.triu_indices(ns, k=1)
                pw_slopes = (season_data[idx[1]] - season_data[idx[0]]) / (idx[1] - idx[0])
                all_slopes.extend(pw_slopes.tolist())

        if s_tot > 0:
            z_val = (s_tot - 1.0) / math.sqrt(var_tot) if var_tot > 0 else 0.0
        elif s_tot < 0:
            z_val = (s_tot + 1.0) / math.sqrt(var_tot) if var_tot > 0 else 0.0
        else:
            z_val = 0.0

        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_val)))
        h = bool(p_val < alpha)
        trend = "increasing" if (h and z_val > 0) else ("decreasing" if (h and z_val < 0) else "no trend")
        tau_val = s_tot / (0.5 * n * (n - 1)) if n > 1 else 0.0
        s_val = s_tot
        var_s = var_tot
        slope = float(np.median(all_slopes)) if all_slopes else 0.0

    return {
        "test_name": "Hirsch-Slack Seasonal Mann-Kendall Test",
        "period": period,
        "trend": trend,
        "is_significant": h,
        "p_value": p_val,
        "z_statistic": z_val,
        "tau": tau_val,
        "s_statistic": s_val,
        "var_s": var_s,
        "sen_slope": slope,
        "alpha": alpha,
        "sample_size": n,
        "interpretation": f"{'Significant' if h else 'No significant'} seasonal trend (p = {p_val:.4e})",
    }


def compute_sens_slope(
    y: Union[np.ndarray, List[float], pd.Series],
    x: Optional[Union[np.ndarray, List[float], pd.Series]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes Sen's Slope (Theil-Sen estimator) with analytical 95% Confidence Intervals.
    """
    y_arr = np.asarray(y, dtype=float)
    if x is None:
        x_arr = np.arange(len(y_arr), dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)

    mask = ~np.isnan(y_arr) & ~np.isnan(x_arr)
    y_clean = y_arr[mask]
    x_clean = x_arr[mask]
    n = len(y_clean)

    if n < 2:
        raise ValueError("Sen's slope requires at least 2 valid observations.")

    # Pairwise slopes
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x_clean[idx_j] - x_clean[idx_i]
    dy = y_clean[idx_j] - y_clean[idx_i]

    # Filter out identical x
    valid_dx = dx != 0
    slopes = dy[valid_dx] / dx[valid_dx]
    slopes.sort()

    median_slope = float(np.median(slopes))
    intercept = float(np.median(y_clean - median_slope * x_clean))

    # Analytical confidence intervals via normal rank approximation
    n_slopes = len(slopes)
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    c_alpha = z_crit * math.sqrt(var_s)

    rank_lower = int(math.floor((n_slopes - c_alpha) / 2.0))
    rank_upper = int(math.ceil((n_slopes + c_alpha) / 2.0))

    rank_lower = max(0, min(rank_lower, n_slopes - 1))
    rank_upper = max(0, min(rank_upper, n_slopes - 1))

    slope_ci_lower = float(slopes[rank_lower])
    slope_ci_upper = float(slopes[rank_upper])

    return {
        "slope": median_slope,
        "intercept": intercept,
        "slope_ci_lower": slope_ci_lower,
        "slope_ci_upper": slope_ci_upper,
        "alpha": alpha,
        "n_pairs": n_slopes,
        "n_points": n,
    }


def compute_linear_regression(
    y: Union[np.ndarray, List[float], pd.Series],
    x: Optional[Union[np.ndarray, List[float], pd.Series]] = None,
) -> Dict[str, Any]:
    """
    Computes standard Ordinary Least Squares (OLS) linear regression as a supplementary benchmark.
    """
    y_arr = np.asarray(y, dtype=float)
    if x is None:
        x_arr = np.arange(len(y_arr), dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)

    mask = ~np.isnan(y_arr) & ~np.isnan(x_arr)
    y_clean = y_arr[mask]
    x_clean = x_arr[mask]

    res = stats.linregress(x_clean, y_clean)

    # 95% CI for slope
    t_crit = stats.t.ppf(0.975, df=len(x_clean) - 2)
    slope_ci = t_crit * res.stderr

    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r_value": float(res.rvalue),
        "r_squared": float(res.rvalue ** 2),
        "p_value": float(res.pvalue),
        "std_err": float(res.stderr),
        "slope_ci_lower": float(res.slope - slope_ci),
        "slope_ci_upper": float(res.slope + slope_ci),
        "n_points": len(x_clean),
    }


def compute_autocorrelation(
    data: Union[np.ndarray, List[float], pd.Series],
    max_lag: int = 12,
) -> Dict[str, Any]:
    """
    Computes sample Autocorrelation Function (ACF) and lag-1 autocorrelation coefficient.
    Reports Bartlett's 95% confidence interval for white noise [±1.96 / sqrt(N)].
    """
    y = np.asarray(data, dtype=float)
    y_clean = y[~np.isnan(y)]
    n = len(y_clean)

    if n < max_lag + 2:
        max_lag = max(1, n - 2)

    mean_y = np.mean(y_clean)
    var_y = np.var(y_clean)

    acf_values: List[float] = []
    lags: List[int] = list(range(0, max_lag + 1))

    for lag in lags:
        if lag == 0:
            acf_values.append(1.0)
        elif var_y > 0:
            c_k = np.sum((y_clean[lag:] - mean_y) * (y_clean[:-lag] - mean_y)) / n
            acf_values.append(float(c_k / var_y))
        else:
            acf_values.append(0.0)

    lag_1 = acf_values[1] if len(acf_values) > 1 else 0.0
    ci_bound = 1.96 / math.sqrt(n) if n > 0 else 0.0
    is_lag1_significant = abs(lag_1) > ci_bound

    if is_lag1_significant and lag_1 > 0:
        implication = (
            f"Moderate/Strong positive autocorrelation (r₁ = {lag_1:.3f} > {ci_bound:.3f}). "
            "Standard Mann-Kendall test may underestimate p-values (inflated Type I error). "
            "Seasonal Mann-Kendall or annual aggregation is strongly recommended."
        )
    elif is_lag1_significant and lag_1 < 0:
        implication = (
            f"Negative serial correlation (r₁ = {lag_1:.3f}). "
            "Time series exhibits high-frequency oscillations."
        )
    else:
        implication = (
            f"Low serial correlation (r₁ = {lag_1:.3f} within ±{ci_bound:.3f}). "
            "Standard independent-observation assumptions hold reasonably well."
        )

    return {
        "lags": lags,
        "acf_values": acf_values,
        "lag_1": lag_1,
        "ci_bound_95": ci_bound,
        "is_lag1_significant": is_lag1_significant,
        "sample_size": n,
        "implication": implication,
    }


def synthesize_scientific_answers(
    mk_res: Dict[str, Any],
    sen_res: Dict[str, Any],
    ols_res: Dict[str, Any],
    acf_res: Dict[str, Any],
    variable_name: str,
    units: str,
    location_name: str,
    time_span_years: float,
    start_year: int,
    end_year: int,
    time_resolution: str = "monthly",
) -> Dict[str, Any]:
    """
    Synthesizes rigorous, transparent answers to the 4 central NASA Space Apps challenge questions:
    1. What is changing?
    2. Where is it changing?
    3. How much is it changing?
    4. Is the observed trend statistically significant?
    """
    trend = mk_res["trend"]
    p_val = mk_res["p_value"]
    is_sig = mk_res["is_significant"]
    slope = sen_res["slope"]

    # Express rate in intuitive scientific units
    if time_resolution == "annual":
        annual_rate = slope
    else:
        annual_rate = slope * 12.0  # monthly slope scaled to per-year

    decadal_rate = annual_rate * 10.0
    total_change = annual_rate * time_span_years

    # Q1: What is changing?
    q1 = f"Environmental Variable: {variable_name} ({units}) over the {time_resolution} observation record."

    # Q2: Where is it changing?
    q2 = f"Geographic Target: {location_name} (Temporal coverage: {start_year} – {end_year}, spanning {time_span_years:.1f} years)."

    # Q3: How much is it changing?
    if trend == "increasing":
        direction_word = "warming / increasing"
        sign_str = "+"
    elif trend == "decreasing":
        direction_word = "cooling / decreasing"
        sign_str = ""
    else:
        direction_word = "stable / no monotonic trend"
        sign_str = ""

    q3 = (
        f"The non-parametric Sen's median rate of change is {sign_str}{annual_rate:.4f} {units}/year "
        f"({sign_str}{decadal_rate:.3f} {units}/decade). "
        f"Over the full {time_span_years:.0f}-year period, the estimated cumulative change is {sign_str}{total_change:.2f} {units} "
        f"(OLS linear regression rate: {sign_str}{ols_res['slope'] * (12.0 if time_resolution == 'monthly' else 1.0):.4f} {units}/year, R² = {ols_res['r_squared']:.3f})."
    )

    # Q4: Is the observed trend statistically significant?
    if is_sig:
        q4 = (
            f"YES. The Mann-Kendall test rejects the null hypothesis of no trend at the 95% confidence level "
            f"(p = {p_val:.4e} < 0.05, Standardized Z = {mk_res['z_statistic']:.2f}, Kendall's Tau = {mk_res['tau']:.3f}). "
            f"The trend is statistically significant."
        )
    else:
        q4 = (
            f"NO. The Mann-Kendall test fails to reject the null hypothesis of no monotonic trend "
            f"(p = {p_val:.4f} ≥ 0.05, Standardized Z = {mk_res['z_statistic']:.2f}). "
            f"Any observed change cannot be distinguished from natural internal variability at the 95% confidence level."
        )

    # Statistical vs practical significance distinction
    if is_sig and abs(total_change) >= 1.0:
        practical_meaning = (
            f"Both statistically significant and practically meaningful: A cumulative change of {sign_str}{total_change:.2f} {units} "
            f"represents substantial environmental shift with potential impacts on cooling degree days, water demand, and urban heat exposure."
        )
    elif is_sig:
        practical_meaning = (
            f"Statistically significant, but modest in absolute magnitude: Cumulative change is {sign_str}{total_change:.2f} {units}."
        )
    else:
        practical_meaning = (
            "Not statistically significant: Observed fluctuations remain within the bounds of historical multi-decadal variability."
        )

    return {
        "q1_what": q1,
        "q2_where": q2,
        "q3_how_much": q3,
        "q4_is_significant": q4,
        "annual_rate": annual_rate,
        "decadal_rate": decadal_rate,
        "total_change": total_change,
        "practical_significance": practical_meaning,
        "autocorrelation_limitation": acf_res["implication"],
    }
