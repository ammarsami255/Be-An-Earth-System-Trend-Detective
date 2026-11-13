"""
EarthTrend Explorer - Visualization Module
NASA Space Apps Challenge 2026

Scientific Plotly visualizations adhering to NASA Earth Science design standards:
- Dark navy accents (#0B3D91, #105BD8, #FC3D21) with clean, high-contrast scientific layouts
- Time series with Sen's slope and OLS trendlines
- Climatological heatmaps (Year vs Month matrix)
- Monthly anomaly red/blue bar charts
- Regional comparison charts (Cairo, Alexandria, Aswan, etc.)
- Autocorrelation (ACF) diagnostic plots
- Decadal extreme frequency shift charts
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# NASA color palette standards
NASA_BLUE = "#0B3D91"
NASA_RED = "#FC3D21"
NASA_LIGHT_BLUE = "#4895EF"
NASA_DARK_NAVY = "#031B4D"
NEUTRAL_BG = "#F8F9FA"
CARD_BG = "#FFFFFF"
TEXT_COLOR = "#1E293B"
GRID_COLOR = "#E2E8F0"
WARM_COLOR = "#E63946"
COLD_COLOR = "#1D3557"


def _apply_scientific_theme(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """Applies clean NASA scientific aesthetic to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "font": {"family": "Inter, Segoe UI, sans-serif", "size": 17, "color": NASA_DARK_NAVY},
        },
        xaxis={
            "title": {"text": x_title, "font": {"size": 13, "color": TEXT_COLOR}},
            "gridcolor": GRID_COLOR,
            "showline": True,
            "linecolor": "#CBD5E1",
            "zeroline": False,
            "tickfont": {"color": TEXT_COLOR},
        },
        yaxis={
            "title": {"text": y_title, "font": {"size": 13, "color": TEXT_COLOR}},
            "gridcolor": GRID_COLOR,
            "showline": True,
            "linecolor": "#CBD5E1",
            "zeroline": False,
            "tickfont": {"color": TEXT_COLOR},
        },
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11, "color": TEXT_COLOR},
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        margin={"l": 55, "r": 35, "t": 60, "b": 50},
        hovermode="x unified",
    )
    return fig


def plot_timeseries_with_trend(
    df: pd.DataFrame,
    variable_name: str,
    units: str,
    location_name: str,
    time_col: str = "date",
    value_col: str = "value",
    sen_slope_info: Optional[Dict[str, Any]] = None,
    ols_info: Optional[Dict[str, Any]] = None,
    show_sen: bool = True,
    show_ols: bool = True,
    show_points: bool = True,
) -> go.Figure:
    """
    Renders the primary time-series chart with observed data points and fitted trendlines.
    """
    fig = go.Figure()
    clean_df = df.dropna(subset=[value_col, time_col]).sort_values(by=time_col)

    # 1. Observed time-series trace
    mode_str = "lines+markers" if show_points else "lines"
    fig.add_trace(
        go.Scatter(
            x=clean_df[time_col],
            y=clean_df[value_col],
            mode=mode_str,
            name=f"Observed ({variable_name})",
            line={"color": "#3A86FF", "width": 1.6},
            marker={"size": 4, "color": "#0B3D91", "opacity": 0.7},
            hovertemplate="<b>Date</b>: %{x|%Y-%m}<br><b>Value</b>: %{y:.2f} " + units + "<extra></extra>",
        )
    )

    # Determine x values for trend calculation
    # If time_col is datetime, convert to year fractions or index
    is_dt = pd.api.types.is_datetime64_any_dtype(clean_df[time_col])
    if is_dt:
        start_year = clean_df[time_col].dt.year.iloc[0]
        x_numeric = (clean_df[time_col] - clean_df[time_col].iloc[0]).dt.total_seconds() / (365.25 * 86400.0)
    else:
        x_numeric = clean_df[time_col].values.astype(float) - clean_df[time_col].iloc[0]

    # 2. Sen's Slope Trendline
    if show_sen and sen_slope_info is not None:
        slope = sen_slope_info["slope"]
        intercept = sen_slope_info["intercept"]
        # If sen_slope was computed on years or indices, compute line
        if is_dt:
            # Sen's slope is per-month or per-year
            # Let's compute line directly over dates
            y_first = intercept + slope * 0
            y_last = intercept + slope * (len(clean_df) - 1 if "n_pairs" in sen_slope_info else x_numeric[-1])
            sen_y = np.linspace(clean_df[value_col].median() - slope * (len(clean_df)/2), clean_df[value_col].median() + slope * (len(clean_df)/2), len(clean_df))
            # More accurately, evaluate median slope through data centroid:
            x_idx = np.arange(len(clean_df))
            med_x = np.median(x_idx)
            med_y = np.median(clean_df[value_col])
            sen_line = med_y + slope * (x_idx - med_x)
        else:
            med_x = np.median(clean_df[time_col])
            med_y = np.median(clean_df[value_col])
            sen_line = med_y + slope * (clean_df[time_col] - med_x)

        fig.add_trace(
            go.Scatter(
                x=clean_df[time_col],
                y=sen_line,
                mode="lines",
                name=f"Theil-Sen Trend ({slope:+.4f} {units}/step)",
                line={"color": NASA_RED, "width": 2.5},
                hoverinfo="skip",
            )
        )

    # 3. OLS Linear Trendline
    if show_ols and ols_info is not None:
        ols_slope = ols_info["slope"]
        ols_intercept = ols_info["intercept"]
        r2 = ols_info.get("r_squared", 0.0)
        p_val = ols_info.get("p_value", 1.0)

        x_coords = np.arange(len(clean_df)) if is_dt else clean_df[time_col].values
        ols_line = ols_intercept + ols_slope * x_coords

        fig.add_trace(
            go.Scatter(
                x=clean_df[time_col],
                y=ols_line,
                mode="lines",
                name=f"OLS Linear (R²={r2:.3f}, p={p_val:.3e})",
                line={"color": "#10B981", "width": 2.0, "dash": "dash"},
                hoverinfo="skip",
            )
        )

    title = f"{variable_name} Time Series — {location_name}"
    return _apply_scientific_theme(fig, title, "Time", f"{variable_name} ({units})")


def plot_climatology_heatmap(
    df: pd.DataFrame,
    variable_name: str,
    units: str,
    location_name: str,
    value_col: str = "value",
) -> go.Figure:
    """
    Renders a climatological raster matrix (Heatmap):
    Y-axis = Year, X-axis = Month (Jan-Dec), Color = Value.
    """
    work = df.dropna(subset=[value_col]).copy()
    if "year" not in work.columns:
        work["year"] = pd.to_datetime(work["date"]).dt.year
    if "month" not in work.columns:
        work["month"] = pd.to_datetime(work["date"]).dt.month

    pivot = work.pivot_table(index="year", columns="month", values=value_col, aggfunc="mean")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Reindex columns to 1..12
    pivot = pivot.reindex(columns=list(range(1, 13)))

    colorscale = "Thermal" if "temp" in variable_name.lower() or "t2m" in variable_name.lower() else "Blues"

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=pivot.index.tolist(),
            colorscale=colorscale,
            colorbar={"title": {"text": f"{units}"}},
            hovertemplate="<b>Year</b>: %{y}<br><b>Month</b>: %{x}<br><b>Value</b>: %{z:.2f} " + units + "<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"<b>Climatology Matrix (Year vs Month) — {location_name}</b>",
            "x": 0.02,
            "xanchor": "left",
            "font": {"family": "Inter, Segoe UI, sans-serif", "size": 17, "color": NASA_DARK_NAVY},
        },
        xaxis={"title": "Month", "tickmode": "array", "tickvals": month_labels},
        yaxis={"title": "Year", "autorange": "reversed"},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin={"l": 55, "r": 35, "t": 60, "b": 50},
    )
    return fig


def plot_anomaly_bars(
    anomaly_df: pd.DataFrame,
    variable_name: str,
    units: str,
    location_name: str,
    baseline_str: str = "1981–2010",
) -> go.Figure:
    """
    Renders standardized red/blue anomaly bar chart (Red = Positive / Warmer, Blue = Negative / Cooler).
    """
    df = anomaly_df.dropna(subset=["anomaly", "date"]).sort_values(by="date").copy()
    colors = np.where(df["anomaly"] >= 0, NASA_RED, NASA_BLUE)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["anomaly"],
            marker_color=colors,
            name="Anomaly",
            hovertemplate="<b>Date</b>: %{x|%Y-%m}<br><b>Anomaly</b>: %{y:+.2f} " + units + "<extra></extra>",
        )
    )

    # Add zero reference line
    fig.add_hline(y=0.0, line_width=1.5, line_color="#334155")

    title = f"Monthly {variable_name} Anomalies (Rel. to {baseline_str} Normal) — {location_name}"
    return _apply_scientific_theme(fig, title, "Date", f"Anomaly ({units})")


def plot_regional_comparison(
    dfs_dict: Dict[str, pd.DataFrame],
    variable_name: str,
    units: str,
    value_col: str = "value",
    time_col: str = "date",
) -> go.Figure:
    """
    Renders multi-region overlay comparison (e.g. Greater Cairo vs Alexandria vs Aswan).
    """
    fig = go.Figure()
    palette = ["#0B3D91", "#E63946", "#2A9D8F", "#F4A261", "#9B5DE5"]

    for idx, (loc_name, df_loc) in enumerate(dfs_dict.items()):
        clean_loc = df_loc.dropna(subset=[value_col, time_col]).sort_values(by=time_col)
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=clean_loc[time_col],
                y=clean_loc[value_col],
                mode="lines",
                name=loc_name,
                line={"color": color, "width": 2.0},
                hovertemplate=f"<b>{loc_name}</b>: %{{y:.2f}} {units}<extra></extra>",
            )
        )

    title = f"Regional Environmental Comparison — {variable_name}"
    return _apply_scientific_theme(fig, title, "Time", f"{variable_name} ({units})")


def plot_autocorrelation_acf(acf_info: Dict[str, Any]) -> go.Figure:
    """
    Renders Autocorrelation Function (ACF) chart with Bartlett's 95% confidence bounds.
    Displays both Raw ACF and Detrended Residual ACF when available.
    """
    fig = go.Figure()
    lags = acf_info["lags"]
    acf_vals = acf_info["acf_values"]
    ci = acf_info["ci_bound_95"]

    # Shaded confidence band for white noise
    fig.add_hline(y=ci, line_dash="dash", line_color="#94A3B8", annotation_text="+95% White Noise Bound")
    fig.add_hline(y=-ci, line_dash="dash", line_color="#94A3B8", annotation_text="-95% White Noise Bound")
    fig.add_hline(y=0.0, line_color="#475569")

    # Raw ACF Lollipops and Markers
    for lag, val in zip(lags, acf_vals):
        fig.add_trace(
            go.Scatter(
                x=[lag, lag],
                y=[0, val],
                mode="lines",
                line={"color": NASA_BLUE, "width": 2.0},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=acf_vals,
            mode="markers",
            marker={"size": 8, "color": NASA_BLUE},
            name="Raw Series ACF",
            hovertemplate="<b>Lag</b>: %{x}<br><b>Raw ACF</b>: %{y:.3f}<extra></extra>",
        )
    )

    # Detrended Residual ACF
    if "residual_acf_values" in acf_info:
        res_acf = acf_info["residual_acf_values"]
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=res_acf,
                mode="lines+markers",
                marker={"size": 7, "color": "#10B981", "symbol": "diamond"},
                line={"color": "#10B981", "dash": "dot", "width": 1.5},
                name="Detrended Residual ACF",
                hovertemplate="<b>Lag</b>: %{x}<br><b>Residual ACF</b>: %{y:.3f}<extra></extra>",
            )
        )

    title = "Serial Autocorrelation Diagnostic (Raw vs. Detrended Residuals)"
    return _apply_scientific_theme(fig, title, "Lag (Steps)", "Autocorrelation Coefficient (r)")


def plot_decadal_shift_bars(decadal_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing changes in warm extreme frequency across decades.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=decadal_df["decade_label"],
            y=decadal_df["warm_extreme_pct"],
            name="Warm Extremes (≥ +1.5σ)",
            marker_color=NASA_RED,
            hovertemplate="<b>%{x}</b>: %{y:.1f}% of months<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=decadal_df["decade_label"],
            y=decadal_df["cold_extreme_pct"],
            name="Cold Extremes (≤ -1.5σ)",
            marker_color=NASA_BLUE,
            hovertemplate="<b>%{x}</b>: %{y:.1f}% of months<extra></extra>",
        )
    )

    fig.update_layout(barmode="group")
    title = "Decadal Frequency Shift of Extreme Months"
    return _apply_scientific_theme(fig, title, "Decade", "Percentage of Months (%)")


def plot_rainfall_annual_totals(
    annual_df: pd.DataFrame,
    trend: Dict[str, Any],
    normal_mm: float,
    location_name: str,
    x_col: str = "year",
    value_col: str = "total_mm",
    title_prefix: str = "Annual Rainfall Totals",
    change_point: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Bar chart of yearly rainfall totals (blue = above the 1981–2010 normal, tan = below),
    with the 1981–2010 normal as a dashed line and Sen's slope trend line.
    If a significant Pettitt `change_point` is given, the mean level before and after
    the shift is drawn as a step line.
    """
    df = annual_df.dropna(subset=[value_col]).sort_values(by=x_col)
    above = df[value_col] >= normal_mm
    colors = np.where(above, NASA_LIGHT_BLUE, "#D4A373")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[value_col],
            marker_color=colors,
            name="Rainfall total",
            hovertemplate="<b>%{x}</b>: %{y:.1f} mm<extra></extra>",
        )
    )
    fig.add_hline(
        y=normal_mm,
        line_dash="dash",
        line_color="#334155",
        annotation_text=f"1981–2010 normal: {normal_mm:.1f} mm",
        annotation_position="top left",
    )
    x_vals = df[x_col].astype(float).values
    trend_y = trend["sen_intercept"] + trend["sen_slope_per_year"] * x_vals
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=trend_y,
            mode="lines",
            name=f"Sen's slope ({trend['sen_slope_per_decade']:+.2f} mm/decade)",
            line={"color": NASA_RED, "width": 2.5},
            hoverinfo="skip",
        )
    )
    if change_point is not None and change_point.get("is_significant"):
        cp_year = change_point["new_regime_start"]
        first, last = float(df[x_col].min()), float(df[x_col].max())
        fig.add_trace(
            go.Scatter(
                x=[first, cp_year - 0.5, cp_year - 0.5, last],
                y=[change_point["mean_before"], change_point["mean_before"], change_point["mean_after"], change_point["mean_after"]],
                mode="lines",
                name=f"Pettitt shift in {cp_year} (mean before → after)",
                line={"color": "#7B2CBF", "width": 2.5, "dash": "dot"},
                hoverinfo="skip",
            )
        )
    title = f"{title_prefix} — {location_name}"
    fig = _apply_scientific_theme(fig, title, "Year", "Rainfall (mm)")
    # Several long legend entries: place the legend below the plot so it does not cover the title
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0.0}, margin={"b": 90})
    return fig


def plot_rainfall_monthly_climatology(clim_df: pd.DataFrame, location_name: str) -> go.Figure:
    """Average rainfall per calendar month (bars) and how often each month is wet (line)."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=clim_df["month_name"],
            y=clim_df["mean_mm"],
            marker_color=NASA_BLUE,
            name="Mean rainfall (mm/month)",
            hovertemplate="<b>%{x}</b>: %{y:.1f} mm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=clim_df["month_name"],
            y=clim_df["wet_fraction"] * 100.0,
            mode="lines+markers",
            name="Years with ≥ 1 mm (%)",
            yaxis="y2",
            line={"color": "#2A9D8F", "width": 2.0},
            hovertemplate="<b>%{x}</b>: wet in %{y:.0f}% of years<extra></extra>",
        )
    )
    fig = _apply_scientific_theme(fig, f"When Does It Rain? Monthly Rainfall Climatology — {location_name}", "Month", "Mean rainfall (mm/month)")
    fig.update_layout(
        yaxis2={
            "title": {"text": "Years with rain (%)"},
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
            "showgrid": False,
        }
    )
    return fig


def plot_regional_rainfall(annual_by_region: Dict[str, pd.DataFrame]) -> go.Figure:
    """Overlay of yearly rainfall totals for several regions."""
    fig = go.Figure()
    palette = ["#0B3D91", "#E63946", "#2A9D8F", "#F4A261", "#9B5DE5"]
    for idx, (name, df) in enumerate(annual_by_region.items()):
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df["total_mm"],
                mode="lines+markers",
                name=name,
                line={"color": palette[idx % len(palette)], "width": 2.0},
                marker={"size": 4},
                hovertemplate=f"<b>{name}</b> %{{x}}: %{{y:.1f}} mm<extra></extra>",
            )
        )
    return _apply_scientific_theme(fig, "Annual Rainfall Totals by Region", "Year", "Rainfall (mm/year)")


def plot_temperature_extremes_comparison(comp_df: pd.DataFrame) -> go.Figure:
    """Grouped bars: warming rate (°C/decade, Sen's slope with 95% CI) of mean, hottest and coldest temperature per region."""
    colors = {"T2M": NASA_BLUE, "T2M_MAX": NASA_RED, "T2M_MIN": NASA_LIGHT_BLUE}
    fig = go.Figure()
    for var, sub in comp_df.groupby("variable", sort=False):
        fig.add_trace(
            go.Bar(
                x=sub["region"],
                y=sub["sen_per_decade"],
                name=f"{sub['label'].iloc[0]} ({var})",
                marker_color=colors.get(var, "#64748B"),
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": sub["ci_upper_per_decade"] - sub["sen_per_decade"],
                    "arrayminus": sub["sen_per_decade"] - sub["ci_lower_per_decade"],
                    "color": "#334155",
                },
                hovertemplate="<b>%{x}</b>: %{y:+.2f} °C/decade<extra></extra>",
            )
        )
    fig = _apply_scientific_theme(fig, "Warming Rate: Mean vs Hottest vs Coldest (Sen's slope, 95% CI)", "Region", "°C per decade")
    fig.update_layout(
        barmode="group",
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "left", "x": 0.0},
        margin={"b": 90},
    )
    return fig


def plot_extreme_index(
    annual_df: pd.DataFrame,
    value_col: str,
    trend: Dict[str, Any],
    title: str,
    y_title: str,
    color: str = NASA_RED,
    as_bars: bool = False,
) -> go.Figure:
    """One yearly extreme index (line or bars) with its trend line (Sen's slope or OLS for counts)."""
    df = annual_df.sort_values("year")
    fig = go.Figure()
    if as_bars:
        fig.add_trace(go.Bar(x=df["year"], y=df[value_col], marker_color=color, name=y_title,
                             hovertemplate="<b>%{x}</b>: %{y}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=df["year"], y=df[value_col], mode="lines+markers", name="Yearly value",
                                 line={"color": color, "width": 2.0}, marker={"size": 5},
                                 hovertemplate="<b>%{x}</b>: %{y:.2f}<extra></extra>"))
    x_vals = df["year"].astype(float).values
    fig.add_trace(go.Scatter(
        x=df["year"], y=trend["sen_intercept"] + trend["sen_slope_per_year"] * x_vals, mode="lines",
        name=f"{trend.get('slope_method', 'Trend')}: {trend['sen_per_decade']:+.2f} / decade",
        line={"color": NASA_DARK_NAVY, "width": 2.5, "dash": "dash"}, hoverinfo="skip",
    ))
    fig = _apply_scientific_theme(fig, title, "Year", y_title)
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0.0}, margin={"b": 80})
    return fig


def plot_decadal_extreme_months(decades_df: pd.DataFrame) -> go.Figure:
    """Unusually hot vs cold months per year, averaged by decade."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=decades_df["label"], y=decades_df["hot_per_year"], name="Unusually hot months / year",
                         marker_color=NASA_RED, hovertemplate="%{x}: %{y:.2f} hot months/yr<extra></extra>"))
    fig.add_trace(go.Bar(x=decades_df["label"], y=decades_df["cold_per_year"], name="Unusually cold months / year",
                         marker_color=NASA_LIGHT_BLUE, hovertemplate="%{x}: %{y:.2f} cold months/yr<extra></extra>"))
    fig.add_hline(y=1.2, line_dash="dot", line_color="#334155",
                  annotation_text="Expected if climate were like 1981–2010 (≈1.2/yr)", annotation_position="top right", annotation_font_color=TEXT_COLOR)
    fig = _apply_scientific_theme(fig, "Unusually Hot vs Cold Months per Year, by Decade", "Decade", "Months per year")
    fig.update_layout(barmode="group", hovermode="closest", legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0.0}, margin={"b": 90})
    return fig


def plot_monthly_extreme_trends(by_month_tmax: pd.DataFrame, by_month_tmin: pd.DataFrame) -> go.Figure:
    """Warming rate of each calendar month's hottest and coldest value; solid bars = significant after FDR."""
    fig = go.Figure()
    for df, name, color in [(by_month_tmax, "Hottest value (T2M_MAX)", NASA_RED),
                            (by_month_tmin, "Coldest value (T2M_MIN)", NASA_LIGHT_BLUE)]:
        opacity = np.where(df["sig_bh"], 1.0, 0.35)
        fig.add_trace(go.Bar(
            x=df["month_name"], y=df["sen_slope_per_decade"], name=name, marker_color=color,
            marker_opacity=opacity, customdata=df["p_bh"],
            hovertemplate="<b>%{x}</b>: %{y:+.2f} °C/decade (FDR p = %{customdata:.3f})<extra></extra>",
        ))
    fig.add_hline(y=0.0, line_width=1.2, line_color="#334155")
    fig = _apply_scientific_theme(fig, "Which Months Are Warming?", "Month", "°C per decade (Sen's slope)")
    fig.update_layout(barmode="group", hovermode="closest", legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0.0}, margin={"b": 90})
    return fig
