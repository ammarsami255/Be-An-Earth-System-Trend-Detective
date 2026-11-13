"""
EarthTrend Explorer - Trend Analysis Module
NASA Space Apps Challenge 2026

Scientifically rigorous trend analysis engine:
- Non-parametric Mann-Kendall trend test (original)
- Hamed & Rao (1998) Modified Mann-Kendall test for serially correlated data
- Hirsch-Slack Seasonal Mann-Kendall test for monthly series
- Month-by-month calendar breakdown (12 individual months)
- Theil-Sen Robust Slope Estimator with 95% Confidence Intervals
- Ordinary Least Squares (OLS) Linear Regression with R² and p-value
- Serial Autocorrelation (ACF) diagnostic on raw and detrended residuals
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

MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


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
        # Fallback SciPy/NumPy implementation
        diff = y[:, None] - y[None, :]
        sgn = np.sign(diff)
        s_val = float(np.sum(np.triu(sgn, k=1)))

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

        d_val = 0.5 * n * (n - 1)
        tau_val = s_val / d_val if d_val > 0 else 0.0

        indices = np.triu_indices(n, k=1)
        pairwise_slopes = (y[indices[1]] - y[indices[0]]) / (indices[1] - indices[0])
        slope = float(np.median(pairwise_slopes))

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


def hamed_rao_modified_mann_kendall(
    data: Union[np.ndarray, List[float], pd.Series],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes the Hamed & Rao (1998) Modified Mann-Kendall test.
    Corrects the variance of S (var_s) for statistically significant serial autocorrelation
    in the detrended ranks of the series.
    Provides complete intermediate calculation diagnostics.
    """
    y = np.asarray(data, dtype=float)
    y_clean = y[~np.isnan(y)]
    n = len(y_clean)

    if n < 4:
        raise ValueError(f"Hamed-Rao test requires at least 4 observations, got {n}.")

    # 1. Base test statistic S and original variance
    base_res = mann_kendall_test(y_clean, alpha=alpha)
    s_val = base_res["s_statistic"]
    original_var_s = base_res["var_s"]
    slope = base_res["sen_slope"]

    # 2. Hamed & Rao (1998) variance correction calculation
    x_detrend = y_clean - np.arange(1, n + 1) * slope
    ranks = stats.rankdata(x_detrend)
    mean_r = np.mean(ranks)
    var_r = np.var(ranks)

    # Detrended rank autocorrelations
    crit_bound = float(stats.norm.ppf(1.0 - alpha / 2.0) / math.sqrt(n))
    sig_lags: List[Dict[str, Any]] = []
    sni = 0.0

    for i in range(1, n):
        if var_r > 0:
            c_i = float(np.sum((ranks[i:] - mean_r) * (ranks[:-i] - mean_r)) / n)
            rho_i = c_i / var_r
        else:
            rho_i = 0.0

        if abs(rho_i) > crit_bound:
            sig_lags.append({"lag": i, "rho": round(rho_i, 5)})
            term = (n - i) * (n - i - 1) * (n - i - 2) * rho_i
            sni += term

    corr_factor = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * sni
    corrected_var_s = float(original_var_s * corr_factor)

    # Standardized Z with corrected variance
    if s_val > 0:
        z_val = (s_val - 1.0) / math.sqrt(corrected_var_s)
    elif s_val < 0:
        z_val = (s_val + 1.0) / math.sqrt(corrected_var_s)
    else:
        z_val = 0.0

    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_val))))
    h = bool(p_val < alpha)
    trend = "increasing" if (h and z_val > 0) else ("decreasing" if (h and z_val < 0) else "no trend")
    tau_val = float(s_val / (0.5 * n * (n - 1)))

    if len(sig_lags) == 0:
        diag_explanation = (
            f"All {n-1} detrended rank autocorrelations fall within the two-tailed 95% critical threshold "
            f"[±{crit_bound:.4f}]. Thus sni = 0.0 and the variance correction factor n/n* = 1.000000. "
            f"The equality Var*(S) == Var(S) = {original_var_s:.2f} indicates that rank serial correlation "
            f"does not reach statistical significance at alpha={alpha}. Note that with n={n}, statistical "
            f"power to detect subtle autocorrelation is limited; failure to reject the null hypothesis of "
            "zero autocorrelation is not proof of absolute independence."
        )
    else:
        diag_explanation = (
            f"Found {len(sig_lags)} significant detrended rank autocorrelation lags exceeding ±{crit_bound:.4f}. "
            f"Variance correction factor n/n* = {corr_factor:.4f}, adjusting Var(S) from {original_var_s:.2f} to {corrected_var_s:.2f}."
        )

    return {
        "test_name": "Hamed & Rao (1998) Modified Mann-Kendall Test",
        "trend": trend,
        "is_significant": h,
        "p_value": p_val,
        "z_statistic": z_val,
        "tau": tau_val,
        "s_statistic": s_val,
        "original_var_s": original_var_s,
        "corrected_var_s": corrected_var_s,
        "var_s": corrected_var_s,
        "variance_correction_factor": float(corr_factor),
        "significant_lags_count": len(sig_lags),
        "significant_lags": sig_lags,
        "critical_bound": crit_bound,
        "sen_slope": slope,
        "alpha": alpha,
        "sample_size": n,
        "explanation": diag_explanation,
        "interpretation": f"{'Significant' if h else 'No significant'} trend under Hamed-Rao correction (p = {p_val:.4e})",
    }


def seasonal_mann_kendall_test(
    data: Union[np.ndarray, List[float], pd.Series],
    period: int = 12,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes Hirsch-Slack Seasonal Mann-Kendall test.
    Eliminates annual 12-month seasonality by comparing identical months across years.
    
    NOTE ON UNITS:
    The returned `sen_slope` represents change per year (per cycle of length `period`),
    NOT change per month. To obtain decadal rate, multiply by 10 (not 120).
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
            "slope_units_explanation": "units/year",
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
        "slope_units_explanation": "units/year (per 12-month cycle)",
        "alpha": alpha,
        "sample_size": n,
        "interpretation": f"{'Significant' if h else 'No significant'} seasonal trend across cycles (p = {p_val:.4e})",
    }


def compute_calendar_month_breakdown(
    df: pd.DataFrame,
    value_col: str = "value",
    month_col: str = "month",
    year_col: str = "year",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Rigorously tests each of the 12 individual calendar months separately.
    Includes both nominal p-values and multiple testing adjustments:
    - Holm-Bonferroni (Family-Wise Error Rate control)
    - Benjamini-Hochberg (False Discovery Rate control)
    Prevents overgeneralized claims like 'warming is present in all 12 individual months'.
    """
    rows = []
    month_names_list = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    for m_num in range(1, 13):
        m_name = month_names_list[m_num - 1]
        sub = df[df[month_col] == m_num].dropna(subset=[value_col]).sort_values(by=year_col)
        y = sub[value_col].values
        x = sub[year_col].values
        n_obs = len(y)

        if n_obs >= 4:
            mk = mann_kendall_test(y, alpha=alpha)
            ols = stats.linregress(x, y)
            rows.append({
                "month_num": m_num,
                "month_name": m_name,
                "observations": n_obs,
                "mean_val": round(float(np.mean(y)), 2),
                "trend": mk["trend"].capitalize(),
                "is_significant": mk["is_significant"],
                "p_value": float(f"{mk['p_value']:.5f}"),
                "z_statistic": round(float(mk["z_statistic"]), 2),
                "kendall_tau": round(float(mk["tau"]), 3),
                "sen_slope_per_year": float(f"{mk['sen_slope']:.5f}"),
                "sen_slope_per_decade": round(float(mk["sen_slope"] * 10.0), 3),
                "ols_slope_per_decade": round(float(ols.slope * 10.0), 3),
                "ols_p_value": float(f"{ols.pvalue:.5f}"),
            })
        else:
            rows.append({
                "month_num": m_num,
                "month_name": m_name,
                "observations": n_obs,
                "mean_val": round(float(np.mean(y)), 2) if n_obs > 0 else np.nan,
                "trend": "Insufficient Data",
                "is_significant": False,
                "p_value": 1.0,
                "z_statistic": 0.0,
                "kendall_tau": 0.0,
                "sen_slope_per_year": 0.0,
                "sen_slope_per_decade": 0.0,
                "ols_slope_per_decade": 0.0,
                "ols_p_value": 1.0,
            })

    res_df = pd.DataFrame(rows)
    m = len(res_df)
    p_vals = res_df["p_value"].values
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]

    # 1. Holm-Bonferroni (FWER)
    holm_p = np.zeros(m)
    for k in range(m):
        holm_p[k] = sorted_p[k] * (m - k)
    for k in range(1, m):
        holm_p[k] = max(holm_p[k], holm_p[k - 1])
    holm_p = np.clip(holm_p, 0.0, 1.0)
    holm_adj = np.zeros(m)
    holm_adj[sorted_idx] = holm_p

    # 2. Benjamini-Hochberg (FDR)
    bh_p = np.zeros(m)
    for k in range(m):
        bh_p[k] = sorted_p[k] * m / (k + 1)
    for k in range(m - 2, -1, -1):
        bh_p[k] = min(bh_p[k], bh_p[k + 1])
    bh_p = np.clip(bh_p, 0.0, 1.0)
    bh_adj = np.zeros(m)
    bh_adj[sorted_idx] = bh_p

    res_df["p_holm"] = np.round(holm_adj, 5)
    res_df["p_bh"] = np.round(bh_adj, 5)
    res_df["sig_nominal"] = res_df["p_value"] < alpha
    res_df["sig_holm"] = res_df["p_holm"] < alpha
    res_df["sig_bh"] = res_df["p_bh"] < alpha

    return res_df


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

    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x_clean[idx_j] - x_clean[idx_i]
    dy = y_clean[idx_j] - y_clean[idx_i]

    valid_dx = dx != 0
    slopes = dy[valid_dx] / dx[valid_dx]
    slopes.sort()

    median_slope = float(np.median(slopes))
    intercept = float(np.median(y_clean - median_slope * x_clean))

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
    x_time: Optional[Union[np.ndarray, List[float], pd.Series]] = None,
    max_lag: int = 12,
) -> Dict[str, Any]:
    """
    Computes sample Autocorrelation Function (ACF) on:
    1. Raw observations.
    2. Detrended OLS residuals (to verify whether apparent correlation is an artifact of the trend).
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

    # Compute detrended residual autocorrelation
    if x_time is not None:
        x_clean = np.asarray(x_time)[~np.isnan(y)]
    else:
        x_clean = np.arange(n)

    ols_fit = stats.linregress(x_clean, y_clean)
    residuals = y_clean - (ols_fit.intercept + ols_fit.slope * x_clean)
    mean_res = np.mean(residuals)
    var_res = np.var(residuals)
    
    residual_acf_values = [1.0]
    for lag in range(1, max_lag + 1):
        if var_res > 0:
            c_res = np.sum((residuals[lag:] - mean_res) * (residuals[:-lag] - mean_res)) / n
            residual_acf_values.append(float(c_res / var_res))
        else:
            residual_acf_values.append(0.0)

    res_lag_1 = residual_acf_values[1] if len(residual_acf_values) > 1 else 0.0
    is_res_lag1_significant = abs(res_lag_1) > ci_bound

    if is_lag1_significant and not is_res_lag1_significant:
        implication = (
            f"Raw lag-1 autocorrelation is positive (r₁ = {lag_1:.3f} > {ci_bound:.3f}), "
            f"but after removing the monotonic trend, detrended residual autocorrelation drops to {res_lag_1:.3f} "
            f"(within Bartlett's 95% bound ±{ci_bound:.3f}). "
            "This confirms the apparent serial persistence is predominantly driven by the monotonic warming trend "
            "itself, rather than red noise persistence."
        )
    elif is_res_lag1_significant:
        implication = (
            f"Residual serial correlation remains significant after detrending (residual r₁ = {res_lag_1:.3f} > {ci_bound:.3f}). "
            "Hamed & Rao modified Mann-Kendall or pre-whitening is required to prevent inflated Type I errors."
        )
    else:
        implication = (
            f"Serial autocorrelation is low (raw r₁ = {lag_1:.3f}, residual r₁ = {res_lag_1:.3f} within ±{ci_bound:.3f}). "
            "Standard independent-observation assumptions hold well."
        )

    return {
        "lags": lags,
        "acf_values": acf_values,
        "residual_acf_values": residual_acf_values,
        "lag_1": lag_1,
        "residual_lag_1": res_lag_1,
        "ci_bound_95": ci_bound,
        "is_lag1_significant": is_lag1_significant,
        "is_residual_lag1_significant": is_res_lag1_significant,
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
    start_year: int,
    end_year: int,
    time_resolution: str = "annual",
) -> Dict[str, Any]:
    """
    Synthesizes rigorous, transparent answers to the 4 central NASA Space Apps challenge questions.
    Uses exact elapsed years (end_year - start_year) for cumulative change calculations.
    """
    trend = mk_res["trend"]
    p_val = mk_res["p_value"]
    is_sig = mk_res["is_significant"]
    slope = sen_res["slope"]

    # Rate scaling
    if time_resolution == "annual":
        annual_rate = slope
    else:
        # If monthly, slope is per month -> scale to per year
        annual_rate = slope * 12.0

    decadal_rate = annual_rate * 10.0
    elapsed_years = float(end_year - start_year)
    total_change = annual_rate * elapsed_years

    # Q1: What is changing?
    q1 = f"Environmental Variable: {variable_name} ({units}) over the {time_resolution} observation record."

    # Q2: Where is it changing?
    q2 = f"Geographic Target: {location_name} (Temporal coverage: {start_year} – {end_year}, spanning {elapsed_years:.0f} elapsed years / {mk_res['sample_size']} observations)."

    # Q3: How much is it changing?
    sign_str = "+" if annual_rate > 0 else ""
    ols_rate = ols_res["slope"] * (12.0 if time_resolution == "monthly" else 1.0)
    ols_cum = ols_rate * elapsed_years

    q3 = (
        f"The non-parametric Sen's median rate of change is {sign_str}{annual_rate:.4f} {units}/year "
        f"({sign_str}{decadal_rate:.3f} {units}/decade). "
        f"Over the {elapsed_years:.0f}-year elapsed period ({start_year}–{end_year}), the estimated cumulative change is {sign_str}{total_change:.2f} {units} "
        f"(OLS linear regression rate: {sign_str}{ols_rate:.4f} {units}/year, cumulative: {sign_str}{ols_cum:.2f} {units}, R² = {ols_res['r_squared']:.3f})."
    )

    # Q4: Is the observed trend statistically significant?
    if is_sig:
        q4 = (
            f"YES. The Mann-Kendall test rejects the null hypothesis of no trend at the 95% confidence level "
            f"(p = {p_val:.4e} < 0.05, Standardized Z = {mk_res['z_statistic']:.2f}, Kendall's Tau = {mk_res['tau']:.3f}, n = {mk_res['sample_size']}). "
            f"The monotonic trend is statistically significant."
        )
    else:
        q4 = (
            f"NO. The Mann-Kendall test fails to reject the null hypothesis of no monotonic trend "
            f"(p = {p_val:.4f} ≥ 0.05, Standardized Z = {mk_res['z_statistic']:.2f}, n = {mk_res['sample_size']}). "
            f"Observed fluctuations cannot be distinguished from natural internal variability at the 95% confidence level."
        )

    # Statistical vs practical significance
    if is_sig and abs(total_change) >= 1.0:
        practical_meaning = (
            f"Both statistically significant and practically meaningful: A cumulative change of {sign_str}{total_change:.2f} {units} "
            f"over {elapsed_years:.0f} years represents a substantial environmental shift with direct consequences for cooling degree days, "
            f"evaporative demand, and urban heat exposure."
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
        "elapsed_years": elapsed_years,
        "practical_significance": practical_meaning,
        "autocorrelation_limitation": acf_res["implication"],
    }
