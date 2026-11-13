"""
EarthTrend Explorer - Climate Anomaly Forecasting Engine
NASA Space Apps Challenge 2026: "Be An Earth System Trend Detective!"
Team: Machine Without Learning

Provides scientifically sound, forward-looking predictive modeling and anomaly projections:
1. Candidate Models:
   - Seasonal Naive Baseline (benchmark / reference null model)
   - Holt-Winters Additive Exponential Smoothing (state-space ETS)
   - Seasonal Autoregressive Integrated Moving Average (SARIMAX)
2. Rigorous Chronological Backtesting:
   - Expanding-window multi-origin cross-validation (no future data leakage)
   - Model tournament evaluation on out-of-sample MAE, RMSE, and MASE
3. Probabilistic Prediction Intervals:
   - Analytical Kalman covariance (SARIMA)
   - Monte Carlo residual simulation (Holt-Winters)
   - Empirical seasonal residual variance (Seasonal Naive)
   - Guardrails: Non-negativity clamping for precipitation and aridity flags
4. Future Anomaly Detection:
   - Projections evaluated against WMO 1981–2010 monthly climatological normal (μ_m, σ_m)
   - Uncertainty propagation into anomaly and Z-score domain
   - Standard IPCC / WMO severity classification mapping
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.anomaly_detection import (
    DEFAULT_BASELINE_END,
    DEFAULT_BASELINE_START,
    MONTH_ABBR,
    Z_CLASSIFICATION_BRACKETS,
)
from src.preprocessing import calculate_climatology


def _ensure_1d_array(series: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """Helper to convert input series to a clean 1D float numpy array."""
    if isinstance(series, pd.Series):
        arr = series.dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
    return arr


def classify_future_z_score(z: float) -> Tuple[str, str]:
    """
    Classifies a forecasted standardized anomaly (Z-score) into standard WMO severity brackets.
    Returns (bracket_label, hex_color).
    """
    if math.isnan(z):
        return "Undefined", "#708090"

    for lower, upper, label, color in Z_CLASSIFICATION_BRACKETS:
        # Match right-open or closed bracket
        if lower <= z < upper or (upper == float("inf") and z >= lower) or (lower == float("-inf") and z <= upper):
            return label, color

    return "Near Normal", "#708090"


# ==============================================================================
# 1. CANDIDATE FORECASTING MODELS
# ==============================================================================

def forecast_seasonal_naive(
    series: Union[np.ndarray, pd.Series],
    horizon: int = 12,
    alpha_80: float = 0.20,
    alpha_95: float = 0.05,
    is_precipitation: bool = False,
) -> Dict[str, Any]:
    """
    Seasonal Naive Benchmark Model:
    y_hat_{T+h} = y_{T+h-12} (for h in 1..12).
    
    Prediction intervals are derived from in-sample seasonal difference error variance:
    e_t = y_t - y_{t-12}.
    """
    y = _ensure_1d_array(series)
    n = len(y)
    m = 12  # annual monthly seasonality

    if n < m:
        raise ValueError(f"Series length ({n}) must be at least {m} for seasonal forecasting.")

    # Generate point forecast: repeat the last 12 values
    forecast = np.zeros(horizon, dtype=float)
    for h in range(1, horizon + 1):
        k = ((h - 1) % m) + 1
        forecast[h - 1] = y[n - m + (k - 1)]

    # Compute in-sample 1-step seasonal difference errors
    if n > m:
        seasonal_diffs = y[m:] - y[:-m]
        sigma_naive = float(np.std(seasonal_diffs, ddof=1)) if len(seasonal_diffs) > 1 else float(np.std(seasonal_diffs))
        if sigma_naive == 0.0:
            sigma_naive = 1e-4
    else:
        sigma_naive = float(np.std(y)) if np.std(y) > 0 else 1.0

    # Critical values for standard normal
    z_80 = float(stats.norm.ppf(1.0 - alpha_80 / 2.0))
    z_95 = float(stats.norm.ppf(1.0 - alpha_95 / 2.0))

    # Horizon scaling: for seasonal random walk, Var(e_{T+h}) = k * sigma^2 where k = ceil(h/m)
    h_multipliers = np.array([math.sqrt(math.ceil(h / m)) for h in range(1, horizon + 1)])

    ci_80_lower = forecast - z_80 * sigma_naive * h_multipliers
    ci_80_upper = forecast + z_80 * sigma_naive * h_multipliers
    ci_95_lower = forecast - z_95 * sigma_naive * h_multipliers
    ci_95_upper = forecast + z_95 * sigma_naive * h_multipliers

    # Non-negativity guardrail for precipitation
    if is_precipitation:
        forecast = np.maximum(0.0, forecast)
        ci_80_lower = np.maximum(0.0, ci_80_lower)
        ci_95_lower = np.maximum(0.0, ci_95_lower)

    # In-sample fitted values and residuals
    fitted = np.full(n, np.nan)
    fitted[m:] = y[:-m]
    residuals = y[m:] - fitted[m:]

    return {
        "model_name": "Seasonal Naive Baseline",
        "model_key": "naive",
        "forecast": forecast,
        "ci_80_lower": ci_80_lower,
        "ci_80_upper": ci_80_upper,
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "fitted_values": fitted,
        "residuals": residuals,
        "sigma": sigma_naive,
        "params": {"method": "Seasonal persistence (12-month lag)", "sigma_naive": sigma_naive},
    }


def forecast_exponential_smoothing(
    series: Union[np.ndarray, pd.Series],
    horizon: int = 12,
    seasonal_periods: int = 12,
    trend: str = "add",
    seasonal: str = "add",
    n_simulations: int = 1000,
    alpha_80: float = 0.20,
    alpha_95: float = 0.05,
    is_precipitation: bool = False,
) -> Dict[str, Any]:
    """
    Holt-Winters Additive Exponential Smoothing (state-space ETS).
    
    Prediction intervals are generated via Monte Carlo simulation using the
    estimated state innovations (1,000 repetitions).
    """
    y = _ensure_1d_array(series)
    n = len(y)

    if n < seasonal_periods * 2:
        # Fall back to Seasonal Naive if series is too short for stable Holt-Winters
        return forecast_seasonal_naive(
            series, horizon=horizon, alpha_80=alpha_80, alpha_95=alpha_95, is_precipitation=is_precipitation
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                seasonal_periods=seasonal_periods,
                trend=trend,
                seasonal=seasonal,
                initialization_method="estimated",
            )
            fit_res = model.fit()
            forecast = np.asarray(fit_res.forecast(horizon), dtype=float)

        # Monte Carlo simulated prediction intervals
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sims = fit_res.simulate(nsimulations=horizon, repetitions=n_simulations, error="add")
            # sims shape: (horizon, n_simulations)
            ci_80_lower = np.percentile(sims, (alpha_80 / 2.0) * 100.0, axis=1)
            ci_80_upper = np.percentile(sims, (1.0 - alpha_80 / 2.0) * 100.0, axis=1)
            ci_95_lower = np.percentile(sims, (alpha_95 / 2.0) * 100.0, axis=1)
            ci_95_upper = np.percentile(sims, (1.0 - alpha_95 / 2.0) * 100.0, axis=1)
        except Exception:
            # Analytical fallback if simulation fails
            resid_std = float(np.std(fit_res.resid, ddof=1))
            z_80 = float(stats.norm.ppf(1.0 - alpha_80 / 2.0))
            z_95 = float(stats.norm.ppf(1.0 - alpha_95 / 2.0))
            scale = np.array([math.sqrt(1.0 + 0.05 * h) for h in range(horizon)])
            ci_80_lower = forecast - z_80 * resid_std * scale
            ci_80_upper = forecast + z_80 * resid_std * scale
            ci_95_lower = forecast - z_95 * resid_std * scale
            ci_95_upper = forecast + z_95 * resid_std * scale

        # Guarantee monotonicity: ci95_low <= ci80_low <= forecast <= ci80_high <= ci95_high
        ci_80_lower = np.minimum(ci_80_lower, forecast)
        ci_80_upper = np.maximum(ci_80_upper, forecast)
        ci_95_lower = np.minimum(ci_95_lower, ci_80_lower)
        ci_95_upper = np.maximum(ci_95_upper, ci_80_upper)

        if is_precipitation:
            forecast = np.maximum(0.0, forecast)
            ci_80_lower = np.maximum(0.0, ci_80_lower)
            ci_95_lower = np.maximum(0.0, ci_95_lower)

        residuals = np.asarray(fit_res.resid, dtype=float)
        fitted = np.asarray(fit_res.fittedvalues, dtype=float)

        params = {
            "alpha (level)": round(float(fit_res.params.get("smoothing_level", 0.0)), 4),
            "beta (trend)": round(float(fit_res.params.get("smoothing_trend", 0.0)), 4),
            "gamma (seasonal)": round(float(fit_res.params.get("smoothing_seasonal", 0.0)), 4),
            "aic": round(float(getattr(fit_res, "aic", np.nan)), 2),
            "bic": round(float(getattr(fit_res, "bic", np.nan)), 2),
        }

        return {
            "model_name": "Holt-Winters Exponential Smoothing (ETS)",
            "model_key": "ets",
            "forecast": forecast,
            "ci_80_lower": ci_80_lower,
            "ci_80_upper": ci_80_upper,
            "ci_95_lower": ci_95_lower,
            "ci_95_upper": ci_95_upper,
            "fitted_values": fitted,
            "residuals": residuals,
            "params": params,
        }

    except Exception:
        # Fallback to Seasonal Naive on optimization failure
        return forecast_seasonal_naive(
            series, horizon=horizon, alpha_80=alpha_80, alpha_95=alpha_95, is_precipitation=is_precipitation
        )


def forecast_sarima(
    series: Union[np.ndarray, pd.Series],
    horizon: int = 12,
    order: Tuple[int, int, int] = (1, 0, 0),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 12),
    alpha_80: float = 0.20,
    alpha_95: float = 0.05,
    is_precipitation: bool = False,
) -> Dict[str, Any]:
    """
    SARIMAX Seasonal Autoregressive Integrated Moving Average Model.
    Default order: SARIMA(1, 0, 0) x (1, 1, 0, 12).
    
    Prediction intervals are computed using exact analytical state-space Kalman filter covariance.
    """
    y = _ensure_1d_array(series)
    n = len(y)

    if n < seasonal_order[3] * 2:
        return forecast_seasonal_naive(
            series, horizon=horizon, alpha_80=alpha_80, alpha_95=alpha_95, is_precipitation=is_precipitation
        )

    fit_success = False
    fit_res = None

    # Primary configuration
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit_res = model.fit(disp=False)
            fit_success = True
    except Exception:
        fit_success = False

    # Secondary parsimonious fallback if primary fails
    if not fit_success:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y,
                    order=(0, 1, 1),
                    seasonal_order=(0, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_res = model.fit(disp=False)
                fit_success = True
        except Exception:
            return forecast_seasonal_naive(
                series, horizon=horizon, alpha_80=alpha_80, alpha_95=alpha_95, is_precipitation=is_precipitation
            )

    forecast_res = fit_res.get_forecast(steps=horizon)
    forecast = np.asarray(forecast_res.predicted_mean, dtype=float)

    ci_80 = np.asarray(forecast_res.conf_int(alpha=alpha_80), dtype=float)
    ci_95 = np.asarray(forecast_res.conf_int(alpha=alpha_95), dtype=float)

    ci_80_lower = ci_80[:, 0]
    ci_80_upper = ci_80[:, 1]
    ci_95_lower = ci_95[:, 0]
    ci_95_upper = ci_95[:, 1]

    # Monotonicity check
    ci_80_lower = np.minimum(ci_80_lower, forecast)
    ci_80_upper = np.maximum(ci_80_upper, forecast)
    ci_95_lower = np.minimum(ci_95_lower, ci_80_lower)
    ci_95_upper = np.maximum(ci_95_upper, ci_80_upper)

    if is_precipitation:
        forecast = np.maximum(0.0, forecast)
        ci_80_lower = np.maximum(0.0, ci_80_lower)
        ci_95_lower = np.maximum(0.0, ci_95_lower)

    residuals = np.asarray(fit_res.resid, dtype=float)
    fitted = np.asarray(fit_res.fittedvalues, dtype=float)

    params = {
        "order": str(fit_res.model.order),
        "seasonal_order": str(fit_res.model.seasonal_order),
        "aic": round(float(fit_res.aic), 2),
        "bic": round(float(fit_res.bic), 2),
        "ar_params": [round(float(p), 4) for p in fit_res.arparams] if hasattr(fit_res, "arparams") else [],
        "ma_params": [round(float(p), 4) for p in fit_res.maparams] if hasattr(fit_res, "maparams") else [],
    }

    return {
        "model_name": f"SARIMA{fit_res.model.order}x{fit_res.model.seasonal_order}",
        "model_key": "sarima",
        "forecast": forecast,
        "ci_80_lower": ci_80_lower,
        "ci_80_upper": ci_80_upper,
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "fitted_values": fitted,
        "residuals": residuals,
        "params": params,
    }


# ==============================================================================
# 2. CHRONOLOGICAL EXPANDING-WINDOW BACKTESTING HARNESS
# ==============================================================================

def compute_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Computes standard point forecast accuracy metrics: MAE and RMSE."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"mae": mae, "rmse": rmse}


def backtest_models(
    series: Union[np.ndarray, pd.Series],
    horizon: int = 12,
    n_splits: int = 5,
    is_precipitation: bool = False,
) -> Dict[str, Any]:
    """
    Executes expanding-window chronological cross-validation across n_splits forecast origins.
    
    Guarantees strict temporal causality: at origin k, models train ONLY on data up to origin k,
    and are evaluated strictly on the subsequent unseen `horizon` months.
    
    Computes origin-level and aggregate MAE, RMSE, and MASE (Hyndman & Koehler 2006).
    The model with lowest overall out-of-sample MAE is designated tournament winner.
    """
    y = _ensure_1d_array(series)
    n = len(y)
    m = 12

    # Verification of data adequacy
    min_train = m * 3  # at least 3 years
    total_required = min_train + n_splits * horizon
    if n < total_required:
        n_splits = max(1, (n - min_train) // horizon)

    # In-sample Seasonal Naive MAE for MASE denominator calculation
    if n > m:
        naive_in_sample_mae = float(np.mean(np.abs(y[m:] - y[:-m])))
        if naive_in_sample_mae == 0.0:
            naive_in_sample_mae = 1e-4
    else:
        naive_in_sample_mae = 1.0

    models_to_test = ["naive", "ets", "sarima"]
    split_details: List[Dict[str, Any]] = []
    model_errors: Dict[str, Dict[str, List[float]]] = {
        m_key: {"mae": [], "rmse": []} for m_key in models_to_test
    }

    # Split origins: expanding training set, fixed 12-month test horizon
    for split_idx in range(n_splits):
        # origin index in reverse: split 0 is furthest back, split n_splits-1 is most recent
        test_end = n - (n_splits - 1 - split_idx) * horizon
        test_start = test_end - horizon
        train_end = test_start

        y_train = y[:train_end]
        y_test = y[test_start:test_end]

        origin_record: Dict[str, Any] = {
            "split_idx": split_idx + 1,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "test_start_idx": test_start,
            "test_end_idx": test_end,
            "actuals": y_test.tolist(),
            "models": {},
        }

        # 1. Seasonal Naive
        fc_naive = forecast_seasonal_naive(y_train, horizon=horizon, is_precipitation=is_precipitation)
        met_naive = compute_forecast_metrics(y_test, fc_naive["forecast"])
        model_errors["naive"]["mae"].append(met_naive["mae"])
        model_errors["naive"]["rmse"].append(met_naive["rmse"])
        origin_record["models"]["naive"] = {
            "forecast": fc_naive["forecast"].tolist(),
            "metrics": met_naive,
        }

        # 2. Holt-Winters ETS
        fc_ets = forecast_exponential_smoothing(y_train, horizon=horizon, is_precipitation=is_precipitation)
        met_ets = compute_forecast_metrics(y_test, fc_ets["forecast"])
        model_errors["ets"]["mae"].append(met_ets["mae"])
        model_errors["ets"]["rmse"].append(met_ets["rmse"])
        origin_record["models"]["ets"] = {
            "forecast": fc_ets["forecast"].tolist(),
            "metrics": met_ets,
        }

        # 3. SARIMA
        fc_sarima = forecast_sarima(y_train, horizon=horizon, is_precipitation=is_precipitation)
        met_sarima = compute_forecast_metrics(y_test, fc_sarima["forecast"])
        model_errors["sarima"]["mae"].append(met_sarima["mae"])
        model_errors["sarima"]["rmse"].append(met_sarima["rmse"])
        origin_record["models"]["sarima"] = {
            "forecast": fc_sarima["forecast"].tolist(),
            "metrics": met_sarima,
        }

        split_details.append(origin_record)

    # Compute aggregate performance table
    summary_metrics: Dict[str, Dict[str, float]] = {}
    model_display_names = {
        "naive": "Seasonal Naive Baseline",
        "ets": "Holt-Winters ETS",
        "sarima": "SARIMA(1,0,0)(1,1,0,12)",
    }

    best_model_key = "naive"
    lowest_mae = float("inf")

    for m_key in models_to_test:
        avg_mae = float(np.mean(model_errors[m_key]["mae"]))
        avg_rmse = float(np.mean(model_errors[m_key]["rmse"]))
        mase = avg_mae / naive_in_sample_mae

        summary_metrics[m_key] = {
            "model_name": model_display_names[m_key],
            "mean_mae": round(avg_mae, 3),
            "mean_rmse": round(avg_rmse, 3),
            "mase": round(mase, 3),
        }

        if avg_mae < lowest_mae:
            lowest_mae = avg_mae
            best_model_key = m_key

    return {
        "n_splits": n_splits,
        "horizon": horizon,
        "naive_in_sample_mae": round(naive_in_sample_mae, 4),
        "split_details": split_details,
        "summary_metrics": summary_metrics,
        "tournament_winner_key": best_model_key,
        "tournament_winner_name": model_display_names[best_model_key],
    }


# ==============================================================================
# 3. HIGH-LEVEL ORCHESTRATOR & ANOMALY PROJECTION ENGINE
# ==============================================================================

def generate_12month_forecast(
    df: pd.DataFrame,
    variable: str = "T2M",
    model_type: str = "best",
    horizon: int = 12,
    n_splits_backtest: int = 5,
    baseline_start_year: int = DEFAULT_BASELINE_START,
    baseline_end_year: int = DEFAULT_BASELINE_END,
) -> Dict[str, Any]:
    """
    End-to-End Climate Anomaly Forecasting Pipeline:
    1. Extracts time series for target variable.
    2. Runs expanding-window backtesting tournament to benchmark models and validate generalization.
    3. Fits chosen or best model on full series and projects forward `horizon` months.
    4. Computes 80% and 95% uncertainty envelopes.
    5. Evaluates forecasts against the WMO 1981–2010 monthly climatological normal.
    6. Produces expected future departures, standardized Z-scores, and severity classifications.
    7. Computes residual diagnostic metrics and returns complete analytics dictionary.
    """
    clean_df = df.copy()

    # Identify date column
    date_col = "date" if "date" in clean_df.columns else None
    if date_col is None:
        raise ValueError("Input dataframe must have a 'date' column.")

    clean_df["date"] = pd.to_datetime(clean_df[date_col])
    clean_df.sort_values(by="date", inplace=True)
    clean_df.reset_index(drop=True, inplace=True)

    # Determine if variable represents precipitation
    is_precipitation = "PREC" in variable.upper() or "RAIN" in variable.upper()
    if not is_precipitation and "variable" in clean_df.columns:
        var_sample = str(clean_df["variable"].iloc[0]).upper()
        if "PREC" in var_sample or "RAIN" in var_sample:
            is_precipitation = True

    # Identify value column
    val_col = variable if variable in clean_df.columns else "value"
    if val_col not in clean_df.columns:
        raise ValueError(f"Variable '{variable}' not found in DataFrame columns: {list(clean_df.columns)}")

    y = clean_df[val_col].to_numpy(dtype=float)
    dates = clean_df["date"]
    last_hist_date = dates.iloc[-1]

    # Run backtesting validation tournament
    backtest_res = backtest_models(
        series=y,
        horizon=horizon,
        n_splits=n_splits_backtest,
        is_precipitation=is_precipitation,
    )

    # Determine active model
    winner_key = backtest_res["tournament_winner_key"]
    if model_type == "best":
        chosen_key = winner_key
    else:
        chosen_key = model_type.lower()
        if chosen_key not in ["naive", "ets", "sarima"]:
            chosen_key = winner_key

    # Fit selected model on full historical record
    if chosen_key == "ets":
        fc_dict = forecast_exponential_smoothing(y, horizon=horizon, is_precipitation=is_precipitation)
    elif chosen_key == "sarima":
        fc_dict = forecast_sarima(y, horizon=horizon, is_precipitation=is_precipitation)
    else:
        fc_dict = forecast_seasonal_naive(y, horizon=horizon, is_precipitation=is_precipitation)

    # Generate future monthly timestamps (Month Start)
    future_dates = pd.date_range(
        start=last_hist_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )

    # Calculate WMO 1981-2010 Monthly Climatology
    clim_df = calculate_climatology(
        clean_df,
        value_col=val_col,
        time_col="date",
        baseline_start_year=baseline_start_year,
        baseline_end_year=baseline_end_year,
    )
    # clim_df columns: month, clim_mean, clim_std, clim_min, clim_max, clim_count, month_name
    clim_lookup = clim_df.set_index("month")[["clim_mean", "clim_std"]].to_dict(orient="index")

    # Construct Future Projections DataFrame
    forecast_records: List[Dict[str, Any]] = []
    for i, f_date in enumerate(future_dates):
        m = f_date.month
        m_name = MONTH_ABBR[m - 1]
        fc_val = float(fc_dict["forecast"][i])
        ci80_l = float(fc_dict["ci_80_lower"][i])
        ci80_u = float(fc_dict["ci_80_upper"][i])
        ci95_l = float(fc_dict["ci_95_lower"][i])
        ci95_u = float(fc_dict["ci_95_upper"][i])

        if is_precipitation:
            fc_val = max(0.0, fc_val)
            ci80_l = max(0.0, ci80_l)
            ci95_l = max(0.0, ci95_l)

        m_stats = clim_lookup.get(m, {"clim_mean": np.nan, "clim_std": np.nan})
        c_mean = float(m_stats["clim_mean"])
        c_std = float(m_stats["clim_std"]) if float(m_stats["clim_std"]) > 0 else 1.0

        # Expected anomaly and Z-score
        exp_anom = fc_val - c_mean
        exp_anom_ci95_l = ci95_l - c_mean
        exp_anom_ci95_u = ci95_u - c_mean

        exp_z = exp_anom / c_std if c_std > 0 else 0.0
        exp_z_ci95_l = exp_anom_ci95_l / c_std if c_std > 0 else 0.0
        exp_z_ci95_u = exp_anom_ci95_u / c_std if c_std > 0 else 0.0

        bracket_label, bracket_color = classify_future_z_score(exp_z)

        # Precipitation Percent of Normal
        pct_norm = (fc_val / c_mean * 100.0) if (is_precipitation and c_mean > 0.1) else np.nan

        forecast_records.append({
            "date": f_date,
            "month": m,
            "month_name": m_name,
            "forecast": round(fc_val, 2),
            "ci_80_lower": round(ci80_l, 2),
            "ci_80_upper": round(ci80_u, 2),
            "ci_95_lower": round(ci95_l, 2),
            "ci_95_upper": round(ci95_u, 2),
            "clim_mean": round(c_mean, 2),
            "clim_std": round(c_std, 2),
            "expected_anomaly": round(exp_anom, 2),
            "expected_anomaly_ci95_lower": round(exp_anom_ci95_l, 2),
            "expected_anomaly_ci95_upper": round(exp_anom_ci95_u, 2),
            "expected_z_score": round(exp_z, 2),
            "expected_z_ci95_lower": round(exp_z_ci95_l, 2),
            "expected_z_ci95_upper": round(exp_z_ci95_u, 2),
            "severity_bracket": bracket_label,
            "severity_color": bracket_color,
            "pct_of_normal": round(pct_norm, 1) if not math.isnan(pct_norm) else np.nan,
        })

    forecast_df = pd.DataFrame(forecast_records)

    # Historical observations reference (last 60 months)
    hist_tail = clean_df.tail(60).copy()
    hist_tail["month"] = hist_tail["date"].dt.month
    hist_merged = pd.merge(hist_tail, clim_df[["month", "clim_mean", "clim_std"]], on="month", how="left")
    hist_merged["anomaly"] = hist_merged[val_col] - hist_merged["clim_mean"]
    hist_merged["z_score"] = np.where(
        hist_merged["clim_std"] > 0,
        hist_merged["anomaly"] / hist_merged["clim_std"],
        0.0,
    )
    hist_merged.sort_values(by="date", inplace=True)
    hist_merged.reset_index(drop=True, inplace=True)

    # Aridity diagnostic
    is_hyper_arid = False
    if is_precipitation:
        mean_rain = float(np.mean(y))
        zero_pct = float(np.mean(y <= 0.01)) * 100.0
        if mean_rain < 1.0 or zero_pct > 80.0:
            is_hyper_arid = True

    # Summary Executive KPIs
    mean_exp_anom = float(forecast_df["expected_anomaly"].mean())
    peak_idx = int(forecast_df["forecast"].argmax())
    peak_row = forecast_df.iloc[peak_idx]
    mean_spread = float(((forecast_df["ci_95_upper"] - forecast_df["ci_95_lower"]) / 2.0).mean())

    warm_count = int((forecast_df["expected_z_score"] >= 1.0).sum())
    cold_count = int((forecast_df["expected_z_score"] <= -1.0).sum())
    near_normal_count = int(((forecast_df["expected_z_score"] > -1.0) & (forecast_df["expected_z_score"] < 1.0)).sum())

    summary_kpis = {
        "best_model_name": backtest_res["tournament_winner_name"],
        "best_model_key": winner_key,
        "best_model_mase": backtest_res["summary_metrics"][winner_key]["mase"],
        "mean_expected_anomaly": round(mean_exp_anom, 2),
        "peak_forecast_value": round(float(peak_row["forecast"]), 2),
        "peak_forecast_date": peak_row["date"].strftime("%B %Y"),
        "peak_forecast_z": round(float(peak_row["expected_z_score"]), 2),
        "mean_ci95_spread": round(mean_spread, 2),
        "warm_months_count": warm_count,
        "cold_months_count": cold_count,
        "near_normal_count": near_normal_count,
    }

    return {
        "variable": variable,
        "is_precipitation": is_precipitation,
        "is_hyper_arid": is_hyper_arid,
        "selected_model_name": fc_dict["model_name"],
        "selected_model_key": chosen_key,
        "tournament_winner": winner_key,
        "model_params": fc_dict.get("params", {}),
        "forecast_df": forecast_df,
        "historical_df": hist_merged,
        "backtest_results": backtest_res,
        "residuals": fc_dict.get("residuals", np.array([])),
        "fitted_values": fc_dict.get("fitted_values", np.array([])),
        "summary_kpis": summary_kpis,
    }
