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
from plotly.subplots import make_subplots

# NASA-inspired palette tuned for the dark "space" theme (.streamlit/config.toml).
# Colors are brightened so they keep contrast on the #070D1C page / #0F1B33 card background.
NASA_BLUE = "#4C8DFF"
NASA_RED = "#FF5A3C"
NASA_LIGHT_BLUE = "#7CC4FF"
NASA_DARK_NAVY = "#F1F5F9"  # used for titles and trend lines -> near-white on dark
NEUTRAL_BG = "#070D1C"
CARD_BG = "#0F1B33"
TEXT_COLOR = "#CBD5E1"
GRID_COLOR = "rgba(148, 163, 184, 0.14)"
AXIS_LINE_COLOR = "rgba(148, 163, 184, 0.35)"
REF_LINE_COLOR = "#94A3B8"
WARM_COLOR = "#FF5A3C"
COLD_COLOR = "#5DA9E9"
FONT_FAMILY = "Inter, Segoe UI, sans-serif"


def _apply_scientific_theme(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """Applies clean NASA scientific aesthetic to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "font": {"family": FONT_FAMILY, "size": 17, "color": NASA_DARK_NAVY},
        },
        xaxis={
            "title": {"text": x_title, "font": {"size": 13, "color": TEXT_COLOR}},
            "gridcolor": GRID_COLOR,
            "showline": True,
            "linecolor": AXIS_LINE_COLOR,
            "zeroline": False,
            "tickfont": {"color": TEXT_COLOR},
        },
        yaxis={
            "title": {"text": y_title, "font": {"size": 13, "color": TEXT_COLOR}},
            "gridcolor": GRID_COLOR,
            "showline": True,
            "linecolor": AXIS_LINE_COLOR,
            "zeroline": False,
            "tickfont": {"color": TEXT_COLOR},
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor=CARD_BG,
        font={"family": FONT_FAMILY, "color": TEXT_COLOR},
        hoverlabel={"bgcolor": "#13213D", "bordercolor": "#2B3B5C", "font": {"color": "#F1F5F9", "family": FONT_FAMILY}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11, "color": TEXT_COLOR},
            "bgcolor": "rgba(0,0,0,0)",
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
            line={"color": "#5B9BFF", "width": 1.6},
            marker={"size": 4, "color": NASA_BLUE, "opacity": 0.8},
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
            "font": {"family": FONT_FAMILY, "size": 17, "color": NASA_DARK_NAVY},
        },
        xaxis={"title": "Month", "tickmode": "array", "tickvals": month_labels},
        yaxis={"title": "Year", "autorange": "reversed"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor=CARD_BG,
        font={"family": FONT_FAMILY, "color": TEXT_COLOR},
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
    fig.add_hline(y=0.0, line_width=1.5, line_color=REF_LINE_COLOR)

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
    palette = ["#4C8DFF", "#FF5A3C", "#2EC4B6", "#F4A261", "#B388FF"]

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
    fig.add_hline(y=0.0, line_color=REF_LINE_COLOR)

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
    colors = np.where(above, NASA_LIGHT_BLUE, "#C9A26B")

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
        line_color=REF_LINE_COLOR,
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
                line={"color": "#C084FC", "width": 2.5, "dash": "dot"},
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
            line={"color": "#2EC4B6", "width": 2.0},
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
    palette = ["#4C8DFF", "#FF5A3C", "#2EC4B6", "#F4A261", "#B388FF"]
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
                    "color": REF_LINE_COLOR,
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
    fig.add_hline(y=1.2, line_dash="dot", line_color=REF_LINE_COLOR,
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
    fig.add_hline(y=0.0, line_width=1.2, line_color=REF_LINE_COLOR)
    fig = _apply_scientific_theme(fig, "Which Months Are Warming?", "Month", "°C per decade (Sen's slope)")
    fig.update_layout(barmode="group", hovermode="closest", legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0.0}, margin={"b": 90})
    return fig


# ==============================================================================
# SECTION: CLIMATE ANOMALY & SPI VISUALIZATIONS
# ==============================================================================

def plot_standardized_anomalies_zscore(
    anomaly_df: pd.DataFrame,
    variable_name: str,
    location_name: str,
) -> go.Figure:
    """
    Renders standardized Z-score anomalies with WMO sigma threshold bands:
    ±1.0σ (Moderate), ±2.0σ (Severe), ±3.0σ (Extreme).
    """
    df = anomaly_df.dropna(subset=["z_score"]).copy()
    if "date" in df.columns:
        x_col = df["date"]
    elif "date_str" in df.columns:
        x_col = df["date_str"]
    else:
        x_col = df.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)

    colors = []
    for z in df["z_score"]:
        if z >= 3.0:
            colors.append("#8B0000")  # Extreme Warm
        elif z >= 2.0:
            colors.append("#DC143C")  # Severe Warm
        elif z >= 1.0:
            colors.append("#FF7F50")  # Moderate Warm
        elif z <= -3.0:
            colors.append("#00008B")  # Extreme Cold
        elif z <= -2.0:
            colors.append("#4169E1")  # Severe Cold
        elif z <= -1.0:
            colors.append("#6495ED")  # Moderate Cold
        else:
            colors.append("#708090")  # Near Normal

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_col,
            y=df["z_score"],
            marker_color=colors,
            name="Z-Score Anomaly",
            hovertemplate="<b>Date</b>: %{x}<br><b>Z-Score</b>: %{y:+.2f}σ<br><b>Classification</b>: %{customdata}<extra></extra>",
            customdata=df.get("classification", [""] * len(df)),
        )
    )

    # Add reference threshold lines
    fig.add_hline(y=0.0, line_width=1.2, line_color=REF_LINE_COLOR)
    fig.add_hline(y=2.0, line_dash="dash", line_color="#FF5A3C", annotation_text="+2.0σ Severe Warm", annotation_position="top right", annotation_font_color="#FF5A3C")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="#5DA9E9", annotation_text="-2.0σ Severe Cold", annotation_position="bottom right", annotation_font_color="#5DA9E9")
    fig.add_hline(y=3.0, line_dash="dot", line_color="#8B0000", annotation_text="+3.0σ Extreme Warm", annotation_position="top right", annotation_font_color="#8B0000")
    fig.add_hline(y=-3.0, line_dash="dot", line_color="#00008B", annotation_text="-3.0σ Extreme Cold", annotation_position="bottom right", annotation_font_color="#00008B")

    title = f"Standardized Monthly {variable_name} Anomalies (Z-Score) — {location_name}"
    return _apply_scientific_theme(fig, title, "Date", "Standardized Anomaly (Z-Score σ)")


def plot_spi_series(
    spi_df: pd.DataFrame,
    scale: int,
    location_name: str,
) -> go.Figure:
    """
    Renders the Standardized Precipitation Index (SPI) timeseries with WMO drought threshold bands.
    """
    spi_col = f"SPI_{scale}"
    sub = spi_df.dropna(subset=[spi_col]).copy()
    if sub.empty:
        fig = go.Figure()
        return _apply_scientific_theme(fig, f"SPI-{scale} Undefined (Hyper-Arid Regime) — {location_name}", "Date", f"SPI-{scale}")

    if "date" in sub.columns:
        x_col = sub["date"]
    elif "date_str" in sub.columns:
        x_col = sub["date_str"]
    else:
        x_col = sub.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)

    colors = []
    for val in sub[spi_col]:
        if val >= 2.0:
            colors.append("#08306b")  # Extremely Wet
        elif val >= 1.5:
            colors.append("#2171b5")  # Very Wet
        elif val >= 1.0:
            colors.append("#6baed6")  # Moderately Wet
        elif val <= -2.0:
            colors.append("#a63603")  # Extremely Dry
        elif val <= -1.5:
            colors.append("#e6550d")  # Severely Dry
        elif val <= -1.0:
            colors.append("#fd8d3c")  # Moderately Dry
        else:
            colors.append("#708090")  # Near Normal

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_col,
            y=sub[spi_col],
            marker_color=colors,
            name=f"SPI-{scale}",
            hovertemplate="<b>Date</b>: %{x}<br><b>SPI-" + str(scale) + "</b>: %{y:+.2f}<br><b>Category</b>: %{customdata}<extra></extra>",
            customdata=sub.get(f"SPI_{scale}_category", [""] * len(sub)),
        )
    )

    fig.add_hline(y=0.0, line_width=1.2, line_color=REF_LINE_COLOR)
    fig.add_hline(y=-1.5, line_dash="dash", line_color="#e6550d", annotation_text="-1.5 Severely Dry", annotation_position="bottom right", annotation_font_color="#e6550d")
    fig.add_hline(y=-2.0, line_dash="dot", line_color="#a63603", annotation_text="-2.0 Extremely Dry", annotation_position="bottom right", annotation_font_color="#a63603")
    fig.add_hline(y=+1.5, line_dash="dash", line_color="#2171b5", annotation_text="+1.5 Very Wet", annotation_position="top right", annotation_font_color="#2171b5")
    fig.add_hline(y=+2.0, line_dash="dot", line_color="#08306b", annotation_text="+2.0 Extremely Wet", annotation_position="top right", annotation_font_color="#08306b")

    title = f"Standardized Precipitation Index (SPI-{scale}) — {location_name}"
    return _apply_scientific_theme(fig, title, "Date", f"SPI-{scale} Value")


def plot_compound_scatter(
    compound_df: pd.DataFrame,
    location_name: str,
) -> go.Figure:
    """
    Renders 2D quadrant scatter plot of Temperature Anomaly (Z-Score) vs Precipitation (Percent of Normal).
    """
    df = compound_df.dropna(subset=["t_z_score", "p_percent_normal"]).copy()
    color_map = {
        "Compound Hot & Dry": "#FF5A3C",
        "Compound Hot & Wet": "#FFB703",
        "Compound Cold & Dry": "#7CC4FF",
        "Compound Cold & Wet": "#2EC4B6",
        "Near Normal": "#708090",
    }
    colors = [color_map.get(cat, "#708090") for cat in df["compound_category"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["t_z_score"],
            y=df["p_percent_normal"],
            mode="markers",
            marker={"size": 7, "color": colors, "opacity": 0.8},
            text=df["date_str"],
            customdata=df["compound_category"],
            hovertemplate="<b>Date</b>: %{text}<br><b>Temp Z</b>: %{x:+.2f}σ<br><b>Precip Norm</b>: %{y:.1f}%<br><b>Category</b>: %{customdata}<extra></extra>",
        )
    )

    # Quadrant lines
    fig.add_vline(x=0.0, line_width=1.2, line_dash="dot", line_color=REF_LINE_COLOR)
    fig.add_hline(y=100.0, line_width=1.2, line_dash="dot", line_color=REF_LINE_COLOR)

    title = f"Compound Climate Anomalies: Temperature Z vs. Precipitation Normal — {location_name}"
    fig = _apply_scientific_theme(fig, title, "Temperature Standardized Anomaly (Z-Score σ)", "Precipitation (% of 1981–2010 Normal)")
    return fig


def plot_lagged_cross_correlation(
    cross_corr_dict: Dict[str, Any],
    location_name: str,
) -> go.Figure:
    """
    Renders lagged cross-correlation between precipitation and temperature anomalies
    with Bartlett 95% white noise bounds.
    """
    lags_data = cross_corr_dict.get("lagged_cross_correlations", [])
    if not lags_data:
        fig = go.Figure()
        return _apply_scientific_theme(fig, f"Lagged Cross-Correlation — {location_name}", "Lag (months)", "Correlation (r)")

    lags = [item["lag"] for item in lags_data]
    corrs = [item["correlation"] for item in lags_data]
    descriptions = [item["relationship"] for item in lags_data]
    crit = cross_corr_dict.get("bartlett_critical_bound", 0.086)

    colors = np.where(np.abs(corrs) >= crit, "#FF5A3C", "#708090")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=lags,
            y=corrs,
            marker_color=colors,
            name="Cross-Correlation r",
            hovertemplate="<b>Lag %{x} months</b> (%{customdata})<br>Correlation: %{y:+.3f}<extra></extra>",
            customdata=descriptions,
        )
    )

    fig.add_hline(y=0.0, line_width=1.0, line_color=REF_LINE_COLOR)
    fig.add_hline(y=crit, line_dash="dash", line_color="#FFB703", annotation_text="+95% Critical Bound", annotation_position="top right", annotation_font_color="#FFB703")
    fig.add_hline(y=-crit, line_dash="dash", line_color="#FFB703", annotation_text="-95% Critical Bound", annotation_position="bottom right", annotation_font_color="#FFB703")

    title = f"Lagged Cross-Correlation: Precip Anomaly leads Temp Anomaly — {location_name}"
    fig = _apply_scientific_theme(fig, title, "Lag k (Months: k > 0 means Precip leads Temp)", "Pearson Correlation (r)")
    return fig


# ==============================================================================
# FORECASTING & PREDICTIVE VISUALIZATIONS
# ==============================================================================

def plot_forecast_with_intervals(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    variable_name: str,
    units: str,
    location_name: str,
    show_history_months: int = 60,
    model_name: str = "Forecast",
) -> go.Figure:
    """
    Renders 12-month forward predictive forecast with dual shaded prediction
    envelopes (80% and 95%), WMO 1981-2010 normal curve, and historical context.
    """
    fig = go.Figure()

    # Sliced history for context
    hist_sub = historical_df.tail(show_history_months).copy()
    val_col = "value" if "value" in hist_sub.columns else variable_name

    # 1. Historical Observed Line
    fig.add_trace(
        go.Scatter(
            x=hist_sub["date"],
            y=hist_sub[val_col],
            mode="lines+markers",
            name="Observed History",
            line={"color": "#94A3B8", "width": 2},
            marker={"size": 4, "color": "#94A3B8"},
            hovertemplate="<b>Observed %{x|%b %Y}</b><br>Value: %{y:.2f} " + units + "<extra></extra>",
        )
    )

    # Historical climatological normal (if available)
    if "clim_mean" in hist_sub.columns:
        fig.add_trace(
            go.Scatter(
                x=hist_sub["date"],
                y=hist_sub["clim_mean"],
                mode="lines",
                name="1981–2010 Baseline Mean",
                line={"color": "rgba(148, 163, 184, 0.45)", "width": 1.2, "dash": "dot"},
                hoverinfo="skip",
            )
        )

    # 2. Climatological Normal for Forecast Period
    if "clim_mean" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["clim_mean"],
                mode="lines",
                name="1981–2010 Baseline Normal",
                line={"color": "rgba(255, 255, 255, 0.5)", "width": 1.5, "dash": "dot"},
                hovertemplate="<b>WMO Normal %{x|%b %Y}</b><br>Baseline Mean: %{y:.2f} " + units + "<extra></extra>",
            )
        )

    # Bridge connection between last historical point and first forecast point
    last_hist_dt = hist_sub["date"].iloc[-1]
    last_hist_val = hist_sub[val_col].iloc[-1]

    # 3. 95% Prediction Interval Envelope (Upper then Lower reversed)
    x_ci95 = [last_hist_dt] + list(forecast_df["date"]) + list(forecast_df["date"])[::-1] + [last_hist_dt]
    y_ci95 = [last_hist_val] + list(forecast_df["ci_95_upper"]) + list(forecast_df["ci_95_lower"])[::-1] + [last_hist_val]
    fig.add_trace(
        go.Scatter(
            x=x_ci95,
            y=y_ci95,
            fill="toself",
            fillcolor="rgba(255, 215, 0, 0.10)",
            line={"color": "rgba(255, 215, 0, 0.25)", "width": 1},
            name="95% Prediction Interval",
            hoverinfo="skip",
        )
    )

    # 4. 80% Prediction Interval Envelope
    x_ci80 = [last_hist_dt] + list(forecast_df["date"]) + list(forecast_df["date"])[::-1] + [last_hist_dt]
    y_ci80 = [last_hist_val] + list(forecast_df["ci_80_upper"]) + list(forecast_df["ci_80_lower"])[::-1] + [last_hist_val]
    fig.add_trace(
        go.Scatter(
            x=x_ci80,
            y=y_ci80,
            fill="toself",
            fillcolor="rgba(255, 215, 0, 0.22)",
            line={"color": "rgba(255, 215, 0, 0.45)", "width": 1},
            name="80% Prediction Interval",
            hoverinfo="skip",
        )
    )

    # 5. Point Forecast Line
    x_fc = [last_hist_dt] + list(forecast_df["date"])
    y_fc = [last_hist_val] + list(forecast_df["forecast"])
    fig.add_trace(
        go.Scatter(
            x=x_fc,
            y=y_fc,
            mode="lines+markers",
            name=f"{model_name}",
            line={"color": "#FFD700", "width": 2.8},
            marker={"size": 6, "color": "#FFD700"},
            hovertemplate="<b>Forecast %{x|%b %Y}</b><br>Projection: %{y:.2f} " + units + "<extra></extra>",
        )
    )

    # Vertical dividing line between history and forecast
    dt_str = last_hist_dt.strftime("%Y-%m-%d") if hasattr(last_hist_dt, "strftime") else str(last_hist_dt)
    fig.add_vline(
        x=dt_str,
        line_width=1.5,
        line_dash="dash",
        line_color="#FC3D21",
    )
    fig.add_annotation(
        x=dt_str,
        yref="paper",
        y=0.98,
        text="Forecast Origin",
        showarrow=False,
        xanchor="right",
        font={"color": "#FC3D21", "size": 11, "family": FONT_FAMILY},
    )

    title = f"12-Month Climate Projection & Prediction Intervals ({variable_name}) — {location_name}"
    fig = _apply_scientific_theme(fig, title, "Date", f"{variable_name} ({units})")
    return fig


def plot_future_anomaly_bars(
    future_df: pd.DataFrame,
    variable_name: str,
    units: str,
    location_name: str,
) -> go.Figure:
    """
    Renders bar chart of expected future departures relative to WMO 1981-2010 normal,
    colored by standard IPCC/WMO severity bracket with 95% uncertainty whiskers.
    """
    fig = go.Figure()

    # Calculate asymmetric error whisker lengths relative to point expected anomaly
    err_plus = future_df["expected_anomaly_ci95_upper"] - future_df["expected_anomaly"]
    err_minus = future_df["expected_anomaly"] - future_df["expected_anomaly_ci95_lower"]

    fig.add_trace(
        go.Bar(
            x=future_df["date"].dt.strftime("%b %Y"),
            y=future_df["expected_anomaly"],
            marker_color=future_df["severity_color"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": err_plus,
                "arrayminus": err_minus,
                "color": "#F1F5F9",
                "thickness": 1.2,
                "width": 4,
            },
            name="Expected Anomaly",
            customdata=np.stack((
                future_df["expected_z_score"],
                future_df["severity_bracket"],
                future_df["expected_anomaly_ci95_lower"],
                future_df["expected_anomaly_ci95_upper"],
            ), axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Expected Departure: %{y:+.2f} " + units + "<br>"
                "Standardized Z: %{customdata[0]:+.2f}σ<br>"
                "Severity: %{customdata[1]}<br>"
                "95% CI Range: [%{customdata[2]:+.2f}, %{customdata[3]:+.2f}] " + units +
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0.0, line_width=1.2, line_color=REF_LINE_COLOR)

    title = f"Expected Future Climate Anomalies vs. WMO 1981–2010 Normal — {location_name}"
    fig = _apply_scientific_theme(fig, title, "Forecast Month", f"Departure ({units})")
    return fig


def plot_model_backtest_comparison(
    backtest_results: Dict[str, Any],
    location_name: str = "Regional",
) -> go.Figure:
    """
    Compares out-of-sample forecast accuracy (MAE, RMSE, MASE) across candidate models.
    Highlights the tournament winner.
    """
    summary = backtest_results.get("summary_metrics", {})
    if not summary:
        fig = go.Figure()
        return _apply_scientific_theme(fig, "Model Comparison", "Model", "Score")

    model_names = [v["model_name"] for v in summary.values()]
    maes = [v["mean_mae"] for v in summary.values()]
    rmses = [v["mean_rmse"] for v in summary.values()]
    mases = [v["mase"] for v in summary.values()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="MAE (Mean Absolute Error)",
            x=model_names,
            y=maes,
            marker_color="#4C8DFF",
            hovertemplate="<b>%{x}</b><br>MAE: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="RMSE (Root Mean Squared Error)",
            x=model_names,
            y=rmses,
            marker_color="#7CC4FF",
            hovertemplate="<b>%{x}</b><br>RMSE: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="MASE (vs Seasonal Naive)",
            x=model_names,
            y=mases,
            marker_color="#FFB703",
            hovertemplate="<b>%{x}</b><br>MASE: %{y:.3f} (Values < 1.0 beat naive)<extra></extra>",
        )
    )

    # Reference line at MASE = 1.0 (benchmark equivalence)
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="#FC3D21",
        annotation_text="MASE = 1.0 (Seasonal Naive Parity)",
        annotation_position="top right",
        annotation_font_color="#FC3D21",
    )

    fig.update_layout(barmode="group")
    title = f"Candidate Model Cross-Validation Tournament (Out-of-Sample Expanding Window) — {location_name}"
    fig = _apply_scientific_theme(fig, title, "Forecasting Model", "Error Metric Value")
    return fig


def plot_forecast_residuals_diagnostic(
    residuals: np.ndarray,
    dates: Optional[pd.Series] = None,
    model_name: str = "Selected Model",
    location_name: str = "",
) -> go.Figure:
    """
    Renders 2-panel diagnostic check of model in-sample residuals:
    1. Residual time series (checking constant zero mean and absence of heteroscedasticity).
    2. Residual histogram with Gaussian distribution overlay.
    """
    resid = np.asarray(residuals, dtype=float)
    resid = resid[~np.isnan(resid)]

    if len(resid) == 0:
        fig = go.Figure()
        return _apply_scientific_theme(fig, f"Residual Diagnostics — {model_name}", "Step", "Residual")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Residual Time Series (Mean={np.mean(resid):+.3f})",
            f"Residual Distribution (Std={np.std(resid):.3f})",
        ),
        column_widths=[0.62, 0.38],
    )

    # Panel 1: Residual time series
    x_steps = dates.iloc[-len(resid):] if dates is not None and len(dates) >= len(resid) else list(range(len(resid)))
    fig.add_trace(
        go.Scatter(
            x=x_steps,
            y=resid,
            mode="lines",
            line={"color": "#4C8DFF", "width": 1.2},
            name="Residuals",
            hovertemplate="Residual: %{y:+.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color=REF_LINE_COLOR, row=1, col=1)

    # Panel 2: Residual histogram
    fig.add_trace(
        go.Histogram(
            y=resid,
            nbinsy=24,
            marker_color="rgba(124, 196, 255, 0.6)",
            name="Residual Frequency",
            orientation="h",
            showlegend=False,
            hovertemplate="Count: %{x}<br>Residual: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    title = f"Forecasting Residual Diagnostics: {model_name} — {location_name}"
    fig = _apply_scientific_theme(fig, title, "", "")
    return fig


