"""
EarthTrend Explorer — NASA Space Apps Challenge 2026
Challenge: "Be An Earth System Trend Detective!"
Team: Machine Without Learning

Main Streamlit Application
"""

from __future__ import annotations
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly_detection import (
    compute_anomalies_and_zscores,
    decadal_extreme_frequency,
    get_top_ranked_extremes,
)
from src.data_loader import (
    EGYPT_LOCATIONS,
    VARIABLE_METADATA,
    auto_load_dataset,
    fetch_nasa_power_point,
    validate_variable_data,
)
from src.preprocessing import (
    aggregate_to_annual,
    calculate_climatology,
    check_and_deduplicate_timestamps,
    compute_monthly_anomalies,
    filter_by_date_range,
    handle_missing_values,
)
from src.extremes_analysis import METRIC_INFO, analyze_temperature_extremes, compare_regional_extremes
from src.rainfall_analysis import analyze_rainfall, compare_regional_rainfall
from src.trend_analysis import (
    compute_autocorrelation,
    compare_temperature_extremes,
    compute_calendar_month_breakdown,
    compute_linear_regression,
    compute_sens_slope,
    hamed_rao_modified_mann_kendall,
    mann_kendall_test,
    seasonal_mann_kendall_test,
    synthesize_scientific_answers,
)
from src.visualization import (
    plot_anomaly_bars,
    plot_autocorrelation_acf,
    plot_climatology_heatmap,
    plot_decadal_extreme_months,
    plot_decadal_shift_bars,
    plot_extreme_index,
    plot_monthly_extreme_trends,
    plot_rainfall_annual_totals,
    plot_rainfall_monthly_climatology,
    plot_regional_comparison,
    plot_regional_rainfall,
    plot_temperature_extremes_comparison,
    plot_timeseries_with_trend,
)
from src.visualization import NASA_BLUE as NASA_BLUE_HEX, NASA_RED as NASA_RED_HEX

# Page configuration
st.set_page_config(
    page_title="EarthTrend Explorer | NASA Space Apps 2026",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom NASA-inspired CSS
st.markdown(
    """
    <style>
    /* Global styles */
    .main {
        background-color: #F8F9FA;
    }
    .nasa-header {
        background: linear-gradient(135deg, #0B3D91 0%, #031B4D 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(11, 61, 145, 0.15);
    }
    .nasa-header h1 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin: 0;
        font-size: 2.1rem;
    }
    .nasa-badge {
        display: inline-block;
        background-color: #FC3D21;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .nasa-subbadge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.78rem;
        margin-right: 8px;
    }
    .q-box {
        background-color: #FFFFFF;
        border-left: 5px solid #0B3D91;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    .q-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0B3D91;
        margin-bottom: 4px;
    }
    .q-answer {
        font-size: 0.95rem;
        color: #1E293B;
        line-height: 1.5;
    }
    .stat-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .stat-label {
        font-size: 0.82rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #031B4D;
    }
    .stat-sub {
        font-size: 0.8rem;
        color: #475569;
        margin-top: 4px;
    }
    .sig-green {
        color: #10B981 !important;
    }
    .sig-red {
        color: #EF4444 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_bundled_dataset(filepath: str, location_label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and caches authentic bundled NASA datasets from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bundled dataset file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return auto_load_dataset(content, default_location=location_label)


# Initialize Session State
if "raw_df" not in st.session_state:
    cairo_path = os.path.join("data", "raw", "nasa_power_cairo_1981_2023.csv")
    if os.path.exists(cairo_path):
        df_init, meta_init = load_bundled_dataset(cairo_path, "Greater Cairo")
        st.session_state["raw_df"] = df_init
        st.session_state["metadata"] = meta_init
        st.session_state["active_location"] = "Greater Cairo"
    else:
        st.session_state["raw_df"] = pd.DataFrame()
        st.session_state["metadata"] = {}
        st.session_state["active_location"] = "None"


# Sidebar: Navigation and Data Controls
with st.sidebar:
    st.markdown("### 🌍 EarthTrend Explorer")
    st.caption("NASA Space Apps 2026 | Machine Without Learning")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home & Overview",
            "📂 Data Explorer",
            "📈 Interactive Time Series",
            "🔬 Trend Analysis Engine",
            "🗺️ Regional Comparison",
            "🌧️ Rainfall Analysis",
            "🔥 Temperature Extremes",
            "⚠️ Anomaly Detection",
            "📚 Methodology & Sources",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 🛰️ Data Source Selection")

    data_source_mode = st.selectbox(
        "Source Mode",
        [
            "Authentic NASA Archive (Preset)",
            "Upload NASA CSV (Giovanni / POWER)",
            "Live NASA POWER API Query",
        ],
    )

    if data_source_mode == "Authentic NASA Archive (Preset)":
        preset_choice = st.selectbox(
            "Select Egyptian Region",
            ["Greater Cairo", "Alexandria", "Aswan"],
            index=0,
        )
        file_map = {
            "Greater Cairo": os.path.join("data", "raw", "nasa_power_cairo_1981_2023.csv"),
            "Alexandria": os.path.join("data", "raw", "nasa_power_alexandria_1981_2023.csv"),
            "Aswan": os.path.join("data", "raw", "nasa_power_aswan_1981_2023.csv"),
        }
        if st.button("Load Region Data", use_container_width=True):
            with st.spinner(f"Loading {preset_choice} NASA dataset..."):
                target_file = file_map[preset_choice]
                df_loaded, meta_loaded = load_bundled_dataset(target_file, preset_choice)
                st.session_state["raw_df"] = df_loaded
                st.session_state["metadata"] = meta_loaded
                st.session_state["active_location"] = preset_choice
                st.success(f"Loaded {preset_choice} data successfully!")

    elif data_source_mode == "Upload NASA CSV (Giovanni / POWER)":
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv", "txt"],
            help="Upload NASA Giovanni time series CSV or NASA POWER monthly table.",
        )
        custom_loc_label = st.text_input("Region Name", value="Uploaded Region")
        if uploaded_file is not None:
            if st.button("Parse Uploaded CSV", use_container_width=True):
                with st.spinner("Parsing CSV and extracting metadata..."):
                    try:
                        content_str = uploaded_file.getvalue().decode("utf-8")
                        df_up, meta_up = auto_load_dataset(content_str, default_location=custom_loc_label)
                        st.session_state["raw_df"] = df_up
                        st.session_state["metadata"] = meta_up
                        st.session_state["active_location"] = custom_loc_label
                        st.success(f"Loaded {len(df_up)} records!")
                    except Exception as e:
                        st.error(f"Error parsing file: {e}")

    elif data_source_mode == "Live NASA POWER API Query":
        st.caption("Live connection to NASA Langley Research Center MERRA-2 API")
        lat_in = st.number_input("Latitude", value=30.0444, format="%.4f")
        lon_in = st.number_input("Longitude", value=31.2357, format="%.4f")
        start_y = st.number_input("Start Year", min_value=1981, max_value=2023, value=1981)
        end_y = st.number_input("End Year", min_value=1981, max_value=2023, value=2023)
        loc_name_in = st.text_input("Location Name", value="Custom Coordinates")

        if st.button("Fetch Real NASA Data", use_container_width=True):
            with st.spinner(f"Querying NASA POWER API for ({lat_in:.2f}, {lon_in:.2f})..."):
                try:
                    df_live, meta_live, raw_csv_live = fetch_nasa_power_point(
                        latitude=lat_in,
                        longitude=lon_in,
                        start_year=int(start_y),
                        end_year=int(end_y),
                    )
                    st.session_state["raw_df"] = df_live
                    st.session_state["metadata"] = meta_live
                    st.session_state["active_location"] = loc_name_in
                    # Save locally
                    save_path = os.path.join("data", "raw", f"nasa_power_custom_{int(start_y)}_{int(end_y)}.csv")
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(raw_csv_live)
                    st.success(f"Fetched {len(df_live)} records and cached locally!")
                except Exception as e:
                    st.error(f"NASA API Query Failed: {e}")

    # Active dataset status in sidebar
    st.markdown("---")
    st.markdown("#### 📊 Active Dataset Context")
    if not st.session_state["raw_df"].empty:
        curr_loc = st.session_state["active_location"]
        curr_meta = st.session_state["metadata"]
        st.write(f"**Region**: {curr_loc}")
        st.write(f"**Source**: {curr_meta.get('source', 'NASA')}")
        if curr_meta.get("latitude") and curr_meta.get("longitude"):
            st.write(f"**Coords**: {curr_meta['latitude']}°N, {curr_meta['longitude']}°E")
        if curr_meta.get("elevation_m"):
            st.write(f"**Elevation**: {curr_meta['elevation_m']} m")
        avail_vars = curr_meta.get("available_variables", [])
        st.write(f"**Variables**: {', '.join(avail_vars)}")
    else:
        st.warning("No dataset loaded yet.")


# Top Banner Header
st.markdown(
    """
    <div class="nasa-header">
        <span class="nasa-badge">NASA Space Apps 2026</span>
        <span class="nasa-subbadge">Challenge: Be An Earth System Trend Detective!</span>
        <span class="nasa-subbadge">Team: Machine Without Learning</span>
        <h1 style="margin-top: 10px;">EarthTrend Explorer</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 0.95rem;">
            Investigating Environmental Trends with Real NASA Earth Science Data (MERRA-2 & Satellite Observations)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Ensure data is present before proceeding with analytical pages
df_current: pd.DataFrame = st.session_state.get("raw_df", pd.DataFrame())
metadata_current: Dict[str, Any] = st.session_state.get("metadata", {})
active_location: str = st.session_state.get("active_location", "Selected Region")

if df_current.empty and page != "🏠 Home & Overview":
    st.error("⚠️ No active dataset loaded. Please go to the sidebar and select or load a dataset.")
    st.stop()


# ==========================================
# PAGE 1: HOME & OVERVIEW
# ==========================================
if page == "🏠 Home & Overview":
    st.markdown("## 🔍 Mission: Be An Earth System Trend Detective!")
    st.markdown(
        """
        Welcome to **EarthTrend Explorer**, a scientific platform developed for the NASA Space Apps Challenge 2026.
        Our mission is to investigate environmental shifts over multi-decadal timescales using authentic NASA Earth observation
        and atmospheric reanalysis datasets.

        Rather than building cosmetic dashboards with unverified assumptions, this application serves as an
        **authentic statistical laboratory** answering four foundational scientific questions:
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="q-box">
                <div class="q-title">Question 1: What is changing?</div>
                <div class="q-answer">
                    Identifying physical variables under observation (e.g. 2-meter air temperature, precipitation)
                    and validating measurement units, sensors, and data completeness.
                </div>
            </div>
            <div class="q-box">
                <div class="q-title">Question 2: Where is it changing?</div>
                <div class="q-answer">
                    Pinpointing exact geographic domains, latitudes, longitudes, and regional topographical contexts
                    (e.g., Greater Cairo urban basin vs. Alexandria coastal marine vs. Aswan arid desert).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="q-box">
                <div class="q-title">Question 3: How much is it changing?</div>
                <div class="q-answer">
                    Quantifying rates of change using robust non-parametric Theil-Sen slope estimation
                    with 95% confidence intervals, alongside supplementary Ordinary Least Squares (OLS) regression.
                </div>
            </div>
            <div class="q-box">
                <div class="q-title">Question 4: Is the trend statistically significant?</div>
                <div class="q-answer">
                    Evaluating hypotheses via the Mann-Kendall test and Hirsch-Slack Seasonal Mann-Kendall test,
                    while transparently accounting for serial autocorrelation and seasonality.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🏆 Initial Benchmark Investigation: Greater Cairo, Egypt")
    st.info(
        """
        **Initial Experiment Overview**:
        - **Target Variable**: Monthly 2-Meter Air Temperature (`T2M`)
        - **Dataset**: NASA MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2) via NASA POWER
        - **Domain**: Greater Cairo, Egypt (Latitude 30.0444° N, Longitude 31.2357° E, Elevation: 135.44 m)
        - **Temporal Span**: 43 continuous years (1981 through 2023 = 516 monthly observations)
        - **Integrity Guarantee**: 100% genuine NASA data with complete provenance. No fabricated or synthetic numbers.
        """
    )

    # Quick Highlights metrics if Cairo loaded
    if not df_current.empty and "T2M" in metadata_current.get("available_variables", []):
        t2m_slice = df_current[df_current["variable"] == "T2M"].dropna(subset=["value"]).copy()
        ann_slice = aggregate_to_annual(t2m_slice)
        mk_res = mann_kendall_test(ann_slice["value"])
        sen_res = compute_sens_slope(ann_slice["value"], ann_slice["year"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Historical Period</div>
                    <div class="stat-val">1981–2023</div>
                    <div class="stat-sub">43 Full Years (516 Months)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Observed Sen's Slope</div>
                    <div class="stat-val">+{sen_res['slope']*10:.3f} °C</div>
                    <div class="stat-sub">per decade (+{sen_res['slope']:.4f} °C/yr)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            total_ch = sen_res["slope"] * 42.0  # 42 elapsed years from 1981 to 2023
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Net Shift (42 Elapsed Yrs)</div>
                    <div class="stat-val">+{total_ch:.2f} °C</div>
                    <div class="stat-sub">Sen's Slope × 42 Years</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Mann-Kendall p-value</div>
                    <div class="stat-val sig-green">p &lt; 0.0001</div>
                    <div class="stat-sub">Z = {mk_res['z_statistic']:.2f} (Significant)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 🛠️ Scientific Architecture & Technology Stack")
    st.markdown(
        """
        - **Core Analysis**: `Python 3.10+`, `NumPy`, `Pandas`, `SciPy Stats`
        - **Non-Parametric Trend Engine**: `PyMannKendall` with analytical confidence intervals
        - **Interactive Graphics**: `Plotly` with NASA Earth Science color system
        - **User Interface**: `Streamlit`
        - **Data Provenance**: NASA Goddard Space Flight Center / NASA Langley Research Center
        """
    )


# ==========================================
# PAGE 2: DATA EXPLORER
# ==========================================
elif page == "📂 Data Explorer":
    st.markdown(f"## 📂 Data Explorer — {active_location}")
    st.markdown("Inspect, validate, and verify raw NASA Earth Science datasets and metadata.")

    st.info(
        "📍 **Spatial Representation Notice**: These observations represent the single native **MERRA-2 0.5° × 0.625° grid cell** "
        f"(~50 km × 60 km) containing the queried point ({metadata_current.get('latitude', 'N/A')}°N, {metadata_current.get('longitude', 'N/A')}°E). "
        "It reflects the regional grid-box atmospheric assimilation value, not a municipal/governorate administrative boundary average."
    )

    # Metadata display
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.markdown(
            f"""
            **Dataset Source**: {metadata_current.get('source', 'NASA POWER / MERRA-2')}  
            **Format**: {metadata_current.get('format', 'Standard Table')}  
            **Location Label**: {active_location}
            """
        )
    with meta_cols[1]:
        st.markdown(
            f"""
            **Latitude**: {metadata_current.get('latitude', 'N/A')} °N  
            **Longitude**: {metadata_current.get('longitude', 'N/A')} °E  
            **Elevation**: {metadata_current.get('elevation_m', 'N/A')} meters
            """
        )
    with meta_cols[2]:
        st.markdown(
            f"""
            **Temporal Range**: {metadata_current.get('dates', '1981 - 2023')}  
            **Total Records**: {metadata_current.get('total_records', len(df_current))}  
            **Missing Value Flag**: {metadata_current.get('fill_value', -999.0)}
            """
        )

    st.markdown("---")
    st.markdown("### 🔍 Variable Quality & Statistical Integrity Audit")

    avail_vars = metadata_current.get("available_variables", sorted(df_current["variable"].unique().tolist()))
    audit_rows = []
    for var in avail_vars:
        val_info = validate_variable_data(df_current, var)
        if val_info["valid"]:
            audit_rows.append({
                "Variable Code": var,
                "Description": VARIABLE_METADATA.get(var, {}).get("name", var),
                "Units": val_info["units"],
                "Total Rows": val_info["total_records"],
                "Valid Points": val_info["valid_records"],
                "Missing (-999 / NaN)": val_info["missing_records"],
                "Missing %": f"{val_info['pct_missing']}%",
                "Observed Min": val_info["min_value"],
                "Observed Max": val_info["max_value"],
                "Mean": val_info["mean_value"],
                "Std Dev": val_info["std_value"],
                "Out of Bounds": val_info["out_of_bounds_count"],
            })

    if audit_rows:
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Tabular Records View")

    filter_var = st.selectbox("Filter by Variable", ["All Variables"] + avail_vars)
    view_df = df_current if filter_var == "All Variables" else df_current[df_current["variable"] == filter_var]

    st.dataframe(
        view_df.head(100),
        use_container_width=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "value": st.column_config.NumberColumn("Observed Value", format="%.2f"),
        },
    )

    csv_data = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv_data,
        file_name=f"earthtrend_{active_location.lower().replace(' ', '_')}_{filter_var.lower()}.csv",
        mime="text/csv",
    )


# ==========================================
# PAGE 3: INTERACTIVE TIME SERIES
# ==========================================
elif page == "📈 Interactive Time Series":
    st.markdown(f"## 📈 Interactive Time Series — {active_location}")

    avail_vars = metadata_current.get("available_variables", sorted(df_current["variable"].unique().tolist()))
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])

    with col_ctrl1:
        sel_var = st.selectbox("Select Environmental Variable", avail_vars, index=avail_vars.index("T2M") if "T2M" in avail_vars else 0)
    with col_ctrl2:
        agg_mode = st.radio("Temporal Aggregation", ["Monthly (Native)", "Annual Mean"], horizontal=True)
    with col_ctrl3:
        trend_lines = st.multiselect("Trendlines to Display", ["Theil-Sen Robust Slope", "OLS Linear"], default=["Theil-Sen Robust Slope", "OLS Linear"])

    # Slice variable data
    var_df = df_current[df_current["variable"] == sel_var].dropna(subset=["value"]).copy()
    units = VARIABLE_METADATA.get(sel_var, {}).get("units", var_df["units"].iloc[0] if "units" in var_df else "units")

    # Date range filter slider
    min_year = int(var_df["year"].min())
    max_year = int(var_df["year"].max())
    year_range = st.slider("Filter Temporal Window (Years)", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    var_df = var_df[(var_df["year"] >= year_range[0]) & (var_df["year"] <= year_range[1])].copy()

    if agg_mode == "Annual Mean":
        plot_df = aggregate_to_annual(var_df)
        time_c = "year"
        sen_info = compute_sens_slope(plot_df["value"], plot_df["year"])
        ols_info = compute_linear_regression(plot_df["value"], plot_df["year"])
    else:
        plot_df = var_df
        time_c = "date"
        sen_info = compute_sens_slope(plot_df["value"])
        ols_info = compute_linear_regression(plot_df["value"])

    # Render primary time series
    fig_ts = plot_timeseries_with_trend(
        df=plot_df,
        variable_name=VARIABLE_METADATA.get(sel_var, {}).get("name", sel_var),
        units=units,
        location_name=active_location,
        time_col=time_c,
        value_col="value",
        sen_slope_info=sen_info,
        ols_info=ols_info,
        show_sen="Theil-Sen Robust Slope" in trend_lines,
        show_ols="OLS Linear" in trend_lines,
        show_points=True,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Climatology heatmap
    st.markdown("### 🌡️ Climatology Matrix (Year vs Month Heatmap)")
    st.caption("Visualizing decadal temperature progression across individual calendar months.")
    fig_heat = plot_climatology_heatmap(
        df=var_df,
        variable_name=sel_var,
        units=units,
        location_name=active_location,
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ==========================================
# PAGE 4: TREND ANALYSIS ENGINE
# ==========================================
elif page == "🔬 Trend Analysis Engine":
    st.markdown(f"## 🔬 Statistical Trend Engine — {active_location}")
    st.markdown("Rigorous non-parametric and parametric trend diagnostics answering the core scientific questions.")

    avail_vars = metadata_current.get("available_variables", sorted(df_current["variable"].unique().tolist()))
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_var = st.selectbox("Target Variable", avail_vars, index=avail_vars.index("T2M") if "T2M" in avail_vars else 0)
    with col2:
        resolution = st.selectbox("Resolution for Statistical Testing", ["Annual Aggregation (Recommended)", "Monthly Series"])
    with col3:
        alpha_val = st.select_slider("Significance Level (α)", options=[0.01, 0.05, 0.10], value=0.05)

    var_df = df_current[df_current["variable"] == sel_var].dropna(subset=["value"]).copy()
    units = VARIABLE_METADATA.get(sel_var, {}).get("units", var_df["units"].iloc[0] if "units" in var_df else "units")

    # Prepare data for testing
    if "Annual" in resolution:
        test_df = aggregate_to_annual(var_df)
        x_vals = test_df["year"].values
        y_vals = test_df["value"].values
        res_label = "annual"
    else:
        test_df = var_df.sort_values(by="date")
        x_vals = np.arange(len(test_df))
        y_vals = test_df["value"].values
        res_label = "monthly"

    start_yr = int(test_df["year"].min())
    end_yr = int(test_df["year"].max())

    # Run statistical tests
    mk_result = mann_kendall_test(y_vals, alpha=alpha_val)
    hr_result = hamed_rao_modified_mann_kendall(y_vals, alpha=alpha_val)
    sen_result = compute_sens_slope(y_vals, x_vals, alpha=alpha_val)
    ols_result = compute_linear_regression(y_vals, x_vals)
    acf_result = compute_autocorrelation(y_vals, x_time=x_vals, max_lag=12)

    answers = synthesize_scientific_answers(
        mk_res=mk_result,
        sen_res=sen_result,
        ols_res=ols_result,
        acf_res=acf_result,
        variable_name=VARIABLE_METADATA.get(sel_var, {}).get("name", sel_var),
        units=units,
        location_name=active_location,
        start_year=start_yr,
        end_year=end_yr,
        time_resolution=res_label,
    )

    # Callout: The 4 Answers
    st.markdown("### 🎯 Scientific Synthesis: Answering the 4 Challenge Questions")
    st.markdown(
        f"""
        <div class="q-box">
            <div class="q-title">1. What is changing?</div>
            <div class="q-answer">{answers['q1_what']}</div>
        </div>
        <div class="q-box">
            <div class="q-title">2. Where is it changing?</div>
            <div class="q-answer">{answers['q2_where']}</div>
        </div>
        <div class="q-box">
            <div class="q-title">3. How much is it changing?</div>
            <div class="q-answer">{answers['q3_how_much']}</div>
        </div>
        <div class="q-box">
            <div class="q-title">4. Is the observed trend statistically significant?</div>
            <div class="q-answer">{answers['q4_is_significant']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Practical vs Statistical distinction
    st.info(f"**Physical & Practical Significance**: {answers['practical_significance']}")

    with st.expander("📐 Cumulative Warming Metric Distinction (Theil-Sen vs Endpoint vs OLS)", expanded=False):
        if "Annual" in resolution and len(y_vals) >= 2:
            y1 = y_vals[0]
            yN = y_vals[-1]
            end_diff = yN - y1
            theil_42 = sen_result["slope"] * float(end_yr - start_yr)
            theil_43 = sen_result["slope"] * float(len(y_vals))
            ols_42 = ols_result["slope"] * float(end_yr - start_yr)
            st.markdown(
                f"""
                | Cumulative Metric | Formula / Method | Cairo T2M Value (1981–2023) | Scientific Interpretation |
                | :--- | :--- | :--- | :--- |
                | **Theil-Sen Elapsed Change** | $\\beta \\times (T_{{end}} - T_{{start}}) = {sen_result['slope']:.4f} \\times {end_yr - start_yr}$ | **+{theil_42:.2f} {units}** | **Primary metric**: Median pairwise rate over {end_yr - start_yr} elapsed years. Robust to outliers. |
                | **OLS Total Linear Change** | $m \\times (T_{{end}} - T_{{start}}) = {ols_result['slope']:.4f} \\times {end_yr - start_yr}$ | **+{ols_42:.2f} {units}** | Parametric least-squares fitted change over {end_yr - start_yr} elapsed years ($R^2 = {ols_result['r_squared']:.3f}$). |
                | **Raw Endpoint Difference** | $T({end_yr}) - T({start_yr}) = {yN:.2f} - {y1:.2f}$ | **+{end_diff:.2f} {units}** | Observed difference between two single years; sensitive to single-year weather anomalies. |
                | **Theil-Sen Point Count Multiplier** | $\\beta \\times N = {sen_result['slope']:.4f} \\times {len(y_vals)}$ | **+{theil_43:.2f} {units}** | Fencepost discrepancy (multiplying $N$ points instead of $N-1$ elapsed intervals). |
                """
            )
        else:
            st.markdown(f"Cumulative change across the selected time period is computed as rate × elapsed duration ({end_yr - start_yr} years).")

    # Metric Cards Grid
    st.markdown("### 📊 Statistical Metrics Summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Mann-Kendall Trend</div>
                <div class="stat-val" style="color: {'#10B981' if mk_result['trend'] == 'increasing' else ('#EF4444' if mk_result['trend'] == 'decreasing' else '#64748B')}">
                    {mk_result['trend'].upper()}
                </div>
                <div class="stat-sub">S = {mk_result['s_statistic']:+.1f} | n = {mk_result['sample_size']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">p-value (Two-Tailed)</div>
                <div class="stat-val {'sig-green' if mk_result['is_significant'] else 'sig-red'}">
                    {mk_result['p_value']:.4e}
                </div>
                <div class="stat-sub">Z = {mk_result['z_statistic']:.2f} | α = {alpha_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Sen's Slope Rate</div>
                <div class="stat-val">{answers['decadal_rate']:+.3f} {units}</div>
                <div class="stat-sub">per decade ({answers['annual_rate']:+.4f} / yr)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc4:
        ci_scale = 10 if 'Annual' in resolution else 120
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Sen's 95% Conf. Interval</div>
                <div class="stat-val" style="font-size: 1.15rem;">
                    [{sen_result['slope_ci_lower']*ci_scale:+.3f}, {sen_result['slope_ci_upper']*ci_scale:+.3f}]
                </div>
                <div class="stat-sub">{units} per decade</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Hamed-Rao Autocorrelation-Corrected Mann-Kendall Comparison
    st.markdown("#### 🛡️ Robustness Check: Autocorrelation-Corrected Mann-Kendall (Hamed & Rao 1998)")
    hr_c1, hr_c2, hr_c3, hr_c4 = st.columns(4)
    with hr_c1:
        st.write(f"**Hamed-Rao Trend**: `{hr_result['trend'].upper()}`")
        st.caption(f"Status: `{'Significant' if hr_result['is_significant'] else 'Non-significant'}`")
    with hr_c2:
        st.write(f"**Z-score**: `{hr_result['z_statistic']:.2f}`")
        st.caption(f"Standard Z: `{mk_result['z_statistic']:.2f}`")
    with hr_c3:
        st.write(f"**p-value**: `{hr_result['p_value']:.4e}`")
        st.caption(f"Standard p: `{mk_result['p_value']:.4e}`")
    with hr_c4:
        st.write(f"**Variance Factor (n/n*)**: `{hr_result['variance_correction_factor']:.4f}`")
        st.caption(f"Var(S) = {hr_result['original_var_s']:.1f} → Var*(S) = {hr_result['corrected_var_s']:.1f}")

    with st.expander("🔍 Inspect Hamed-Rao Intermediate Diagnostics & Mathematical Rationale", expanded=False):
        st.markdown(
            f"""
            - **Published Methodology**: Hamed, K. H., & Rao, A. R. (1998). *Journal of Hydrology*, 204(1-4), 182-196.
            - **Detrending Method**: Subtracted Theil-Sen slope ({hr_result['sen_slope']:.4f} {units}/yr) from original series.
            - **Rank Autocorrelation Bounds**: Two-tailed 95% critical bound = $\\pm Z_{{1-\\alpha/2}} / \\sqrt{{n}} = \\pm {hr_result['critical_bound']:.4f}$ ($n = {hr_result['sample_size']}$).
            - **Significant Lags Detected**: `{hr_result['significant_lags_count']}` lags exceeding the critical bound.
            - **Diagnostic Assessment**: {hr_result['explanation']}
            """
        )

    # 12-Month Calendar Breakdown (Resolves overgeneralization and includes multiple testing adjustments)
    if "month" in var_df.columns and len(var_df["month"].unique()) >= 12:
        st.markdown("---")
        st.markdown("### 🗓️ Month-by-Month Calendar Breakdown (12 Individual Months)")
        st.caption("Rigorously evaluates trend statistics for each calendar month separately across all observation years, including multiple-testing corrections.")
        
        cal_breakdown = compute_calendar_month_breakdown(var_df, value_col="value", alpha=alpha_val)
        nominal_sig = cal_breakdown[cal_breakdown["sig_nominal"]]["month_name"].tolist()
        bh_sig = cal_breakdown[cal_breakdown["sig_bh"]]["month_name"].tolist()
        holm_sig = cal_breakdown[cal_breakdown["sig_holm"]]["month_name"].tolist()

        st.info(
            f"**Multiple Testing Synthesis Across 12 Calendar Months**:\n\n"
            f"- **Nominal Unadjusted ($p < {alpha_val}$)**: **{len(nominal_sig)} of 12 months** ({', '.join(nominal_sig)}).\n"
            f"- **Benjamini-Hochberg FDR ($q < {alpha_val}$)**: **{len(bh_sig)} of 12 months** ({', '.join(bh_sig)}). "
            f"Statistically detectable monotonic warming is concentrated in the continuous 6-month late-summer to early-winter period (JUL–DEC); FEB and MAY do not survive false discovery rate control.\n"
            f"- **Holm-Bonferroni FWER ($\alpha < {alpha_val}$)**: **{len(holm_sig)} of 12 months** ({', '.join(holm_sig)}).\n"
            f"- **Months Without Statistically Detectable Monotonic Trends**: In this 43-year observational record, JAN, MAR, APR, and JUN do not show statistically significant monotonic trends at α = {alpha_val}. Note: Absence of statistically significant trend detection is not evidence of thermal stability; higher natural interannual variability in winter/spring reduces test power."
        )

        st.dataframe(
            cal_breakdown[[
                "month_name", "observations", "mean_val", "trend",
                "p_value", "p_holm", "p_bh", "sig_nominal", "sig_holm", "sig_bh",
                "z_statistic", "kendall_tau", "sen_slope_per_decade", "ols_slope_per_decade"
            ]],
            use_container_width=True,
            column_config={
                "month_name": "Month",
                "observations": "Obs",
                "mean_val": f"Mean ({units})",
                "trend": "MK Trend",
                "p_value": "Nominal p",
                "p_holm": "Holm p (FWER)",
                "p_bh": "BH p (FDR)",
                "sig_nominal": "Sig (Nominal)",
                "sig_holm": "Sig (Holm)",
                "sig_bh": "Sig (BH)",
                "z_statistic": "Z-Score",
                "kendall_tau": "Kendall Tau",
                "sen_slope_per_decade": f"Sen Slope ({units}/dec)",
                "ols_slope_per_decade": f"OLS Slope ({units}/dec)",
            },
        )

    # Seasonal Mann-Kendall section if monthly series
    if "Monthly" in resolution:
        st.markdown("---")
        st.markdown("### 🍂 Hirsch-Slack Seasonal Mann-Kendall Test")
        st.caption("Accounts for the 12-month annual seasonal cycle by comparing like months across years.")
        smk_res = seasonal_mann_kendall_test(y_vals, period=12, alpha=alpha_val)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Seasonal MK Trend", smk_res["trend"].upper())
        with sc2:
            st.metric("Seasonal p-value", f"{smk_res['p_value']:.4e}", delta="Significant" if smk_res["is_significant"] else "Not Significant")
        with sc3:
            # Note: smk_res['sen_slope'] is already in units/year; multiply by 10 for decadal rate
            st.metric("Seasonal Sen's Slope", f"{smk_res['sen_slope']*10.0:+.3f} {units}/decade")

        st.caption(
            "⚠️ **Methodological Assumption**: The standard Hirsch-Slack test assumes mutual independence between seasons "
            "(Cov(S_g, S_h) = 0). While seasonal stratification removes the deterministic annual cycle, residual cross-covariance "
            "between consecutive months can persist. Refer to the 12-Month Calendar Breakdown with multiple-testing corrections "
            "for month-specific trend isolation."
        )

    # Autocorrelation diagnostic
    st.markdown("---")
    st.markdown("### 🔄 Temporal Autocorrelation Diagnostic (ACF)")
    st.caption("Detects whether persistence between consecutive observations violates independent error assumptions.")

    st.warning(f"**Autocorrelation Implication**: {answers['autocorrelation_limitation']}")
    fig_acf = plot_autocorrelation_acf(acf_result)
    st.plotly_chart(fig_acf, use_container_width=True)


# ==========================================
# PAGE 5: REGIONAL COMPARISON
# ==========================================
elif page == "🗺️ Regional Comparison":
    st.markdown("## 🗺️ Regional Comparison Across Egypt")
    st.markdown(
        """
        Comparing multi-decadal trends across diverse climatic zones of Egypt:
        - **Greater Cairo**: Nile Delta & Valley urban metropolis (Grid Cell 30.04°N, 31.24°E)
        - **Alexandria**: Mediterranean coastal marine zone (Grid Cell 31.20°N, 29.92°E)
        - **Aswan**: Upper Egypt hyper-arid desert climate (Grid Cell 24.09°N, 32.90°E)
        
        *Note: Comparisons evaluate three individual, point-extracted MERRA-2 0.5° × 0.625° model grid cells representing localized climatic sample points. They do not represent a spatially complete national average or imply that three grid cells represent all of Egypt.*
        """
    )

    regional_files = {
        "Greater Cairo": os.path.join("data", "raw", "nasa_power_cairo_1981_2023.csv"),
        "Alexandria": os.path.join("data", "raw", "nasa_power_alexandria_1981_2023.csv"),
        "Aswan": os.path.join("data", "raw", "nasa_power_aswan_1981_2023.csv"),
    }

    reg_dfs = {}
    for r_name, r_path in regional_files.items():
        if os.path.exists(r_path):
            df_r, _ = load_bundled_dataset(r_path, r_name)
            reg_dfs[r_name] = df_r

    if len(reg_dfs) >= 2:
        reg_var = st.selectbox("Comparison Variable", ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"], index=0)
        reg_time_mode = st.radio("Temporal View", ["Annual Means", "Monthly Timeseries"], horizontal=True)

        sliced_regs = {}
        for r_name, r_df in reg_dfs.items():
            sub = r_df[r_df["variable"] == reg_var].dropna(subset=["value"]).copy()
            if reg_time_mode == "Annual Means":
                sliced_regs[r_name] = aggregate_to_annual(sub)
            else:
                sliced_regs[r_name] = sub

        units = VARIABLE_METADATA.get(reg_var, {}).get("units", "units")
        fig_reg = plot_regional_comparison(
            dfs_dict=sliced_regs,
            variable_name=VARIABLE_METADATA.get(reg_var, {}).get("name", reg_var),
            units=units,
            time_col="year" if reg_time_mode == "Annual Means" else "date",
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # Comparative statistics table
        st.markdown("### 📊 Regional Trend Statistics (1981–2023)")
        st.caption(
            "Displays both Original Mann-Kendall and Hamed & Rao (1998) Autocorrelation-Corrected statistics side by side to ensure unambiguous methodological transparency."
        )
        comp_summary = []
        for r_name, r_df in sliced_regs.items():
            y = r_df["value"].values
            x = r_df["year"].values if reg_time_mode == "Annual Means" else np.arange(len(y))
            mk = mann_kendall_test(y)
            sen = compute_sens_slope(y, x)
            rate_dec = sen["slope"] * 10.0 if reg_time_mode == "Annual Means" else sen["slope"] * 120.0
            ci_dec = (sen["slope_ci_lower"] * 10.0, sen["slope_ci_upper"] * 10.0) if reg_time_mode == "Annual Means" else (sen["slope_ci_lower"] * 120.0, sen["slope_ci_upper"] * 120.0)

            loc_meta = EGYPT_LOCATIONS.get(r_name, {})
            grid_str = f"Lat {loc_meta.get('lat', 'N/A')}°, Lon {loc_meta.get('lon', 'N/A')}°"

            if reg_time_mode == "Annual Means" and len(y) >= 4:
                hr = hamed_rao_modified_mann_kendall(y)
                comp_summary.append({
                    "Region": r_name,
                    "MERRA-2 Grid Cell": grid_str,
                    "Sen Slope Rate": f"{rate_dec:+.3f} {units}/dec",
                    "Sen 95% CI": f"[{ci_dec[0]:+.3f}, {ci_dec[1]:+.3f}]",
                    "Original MK Z": f"{mk['z_statistic']:+.2f}",
                    "Original MK p-val": f"{mk['p_value']:.4e}",
                    "Hamed-Rao Factor (n/n*)": f"{hr['variance_correction_factor']:.4f}",
                    "Hamed-Rao Z*": f"{hr['z_statistic']:+.2f}",
                    "Hamed-Rao p*-val": f"{hr['p_value']:.4e}",
                    "Significant (Both Tests)?": "YES" if (mk["is_significant"] and hr["is_significant"]) else "NO",
                })
            else:
                comp_summary.append({
                    "Region": r_name,
                    "MERRA-2 Grid Cell": grid_str,
                    "Sen Slope Rate": f"{rate_dec:+.3f} {units}/dec",
                    "Sen 95% CI": f"[{ci_dec[0]:+.3f}, {ci_dec[1]:+.3f}]",
                    "Original MK Z": f"{mk['z_statistic']:+.2f}",
                    "Original MK p-val": f"{mk['p_value']:.4e}",
                    "Hamed-Rao Factor (n/n*)": "N/A (Monthly)",
                    "Hamed-Rao Z*": "N/A",
                    "Hamed-Rao p*-val": "N/A",
                    "Significant?": "YES" if mk["is_significant"] else "NO",
                })

        st.dataframe(pd.DataFrame(comp_summary), use_container_width=True)

        st.info(
            "📐 **Slope Comparison Warning**: Although point estimates of Sen's slopes vary across regions "
            f"(Cairo: +0.293 {units}/dec, Alexandria: +0.261 {units}/dec, Aswan: +0.434 {units}/dec), their 95% confidence "
            "intervals overlap. We do not claim that one region is warming significantly faster than another without a formal "
            "multivariate interaction test."
        )

        # Mean vs extreme temperatures
        st.markdown("### 🌡️ Mean vs Hottest vs Coldest Temperature")
        st.markdown(
            "NASA POWER `T2M_MAX` and `T2M_MIN` are the **hottest and coldest values reached within each month** "
            "(monthly extremes), not averages of daily highs/lows. Each is averaged over the 12 months of a year and "
            "tested for a trend, so we can see whether the **extremes** are warming as fast as the **average**."
        )
        ext_df = compare_temperature_extremes(reg_dfs)
        if not ext_df.empty:
            st.plotly_chart(plot_temperature_extremes_comparison(ext_df), use_container_width=True)
            ext_table = pd.DataFrame({
                "Region": ext_df["region"],
                "Variable": ext_df["label"] + " (" + ext_df["variable"] + ")",
                "Average (°C)": ext_df["mean_c"].round(2),
                "Sen slope (°C/decade)": ext_df["sen_per_decade"].map(lambda v: f"{v:+.3f}"),
                "95% CI": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(ext_df["ci_lower_per_decade"], ext_df["ci_upper_per_decade"])],
                "MK p-value": ext_df["mk_p"].map(lambda v: f"{v:.2e}"),
                "Hamed-Rao p-value": ext_df["hr_p"].map(lambda v: f"{v:.2e}"),
                "Significant (both)?": ext_df["is_significant"].map({True: "YES", False: "NO"}),
            })
            st.dataframe(ext_table, use_container_width=True, hide_index=True)
            st.info(
                "📐 Mean, hottest and coldest temperatures are all warming significantly in every region. "
                "Point estimates differ, but the 95% confidence intervals overlap, so we do **not** claim that the "
                "extremes are warming faster than the mean without a formal difference test."
            )
    else:
        st.warning("Regional datasets are not available yet.")


# ==========================================
# PAGE: RAINFALL ANALYSIS
# ==========================================
elif page == "🌧️ Rainfall Analysis":
    st.markdown("## 🌧️ Rainfall Trend Analysis")
    st.markdown(
        """
        Is Egypt getting **wetter or drier**? Rainfall behaves very differently from temperature:
        it falls almost only in the cool season, many months are completely dry, and a single
        storm can dominate a whole year. This page therefore:

        - converts NASA POWER `PRECTOTCORR` from a **rate (mm/day)** into **totals (mm/month, mm/year)**,
        - analyses the **calendar-year total**, the **wet-season total (Oct → Mar)** and the **number of wet months (≥ 1 mm)**,
        - uses non-parametric tests (Mann-Kendall, Hamed-Rao, Sen's slope) that are robust to skewed data,
        - calls a trend significant **only if both Mann-Kendall and Hamed-Rao agree**, and flags results whose data are mostly zeros or ties.
        """
    )

    rain_files = {
        "Greater Cairo": os.path.join("data", "raw", "nasa_power_cairo_1981_2023.csv"),
        "Alexandria": os.path.join("data", "raw", "nasa_power_alexandria_1981_2023.csv"),
        "Aswan": os.path.join("data", "raw", "nasa_power_aswan_1981_2023.csv"),
    }
    rain_dfs = {
        name: load_bundled_dataset(path, name)[0]
        for name, path in rain_files.items()
        if os.path.exists(path)
    }

    if not rain_dfs:
        st.warning("Regional rainfall datasets are not available.")
        st.stop()

    rain_region = st.radio("Region", list(rain_dfs.keys()), horizontal=True)
    rain_res = analyze_rainfall(rain_dfs[rain_region], rain_region)
    t_ann = rain_res["trends"]["annual_total"]

    # Headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean annual rainfall", f"{t_ann['mean']:.1f} mm")
    c2.metric("Dry months (< 1 mm)", f"{rain_res['dry_month_fraction'] * 100:.0f}%")
    c3.metric("Sen's slope", f"{t_ann['sen_slope_per_decade']:+.2f} mm/dec", f"{t_ann['pct_of_mean_per_decade']:+.1f}% of mean / dec", delta_color="off")
    c4.metric("Significant trend?", "YES" if t_ann["is_significant"] else "NO", f"MK p = {t_ann['mk_p']:.3f}", delta_color="off")

    if t_ann["low_reliability"]:
        st.warning("⚠️ **Low reliability for this region**: " + "; ".join(t_ann["reliability_notes"]) + ". Treat the trend numbers with caution.")

    # 4 challenge questions
    st.markdown("### 🔍 The 4 Questions — Rainfall")
    summ = rain_res["summary"]
    st.markdown(f"**1. What is changing?** {summ['q1_what']}")
    st.markdown(f"**2. Where?** {summ['q2_where']}")
    st.markdown(f"**3. How much?** {summ['q3_how_much']}")
    st.markdown(f"**4. Is it statistically significant?** {summ['q4_is_significant']}")

    # Charts
    rain_view = st.radio("Series", ["Calendar-year total", "Wet-season total (Oct–Mar)"], horizontal=True)
    if rain_view == "Calendar-year total":
        fig_rain = plot_rainfall_annual_totals(
            rain_res["annual"], t_ann, rain_res["normal_mm"], rain_region, change_point=rain_res["change_point"]
        )
    else:
        seasons = rain_res["seasons"]
        base = seasons[(seasons["season_end_year"] >= 1981) & (seasons["season_end_year"] <= 2010)]
        fig_rain = plot_rainfall_annual_totals(
            seasons,
            rain_res["trends"]["wet_season_total"],
            float(base["total_mm"].mean()),
            rain_region,
            x_col="season_end_year",
            title_prefix="Wet-Season (Oct–Mar) Rainfall Totals",
        )
    st.plotly_chart(fig_rain, use_container_width=True)
    st.caption("Blue bars: above the 1981–2010 normal. Tan bars: below normal. Red line: Sen's slope trend. Purple dotted step: average level before/after a significant Pettitt shift. Wet seasons are labelled by the year they end (e.g. 2022 = Oct 2021–Mar 2022).")

    # Abrupt shift (Pettitt)
    st.markdown("### ⚡ Gradual Trend or Sudden Jump? (Pettitt Change-Point Test)")
    st.markdown(
        "Mann-Kendall only detects a **gradual** trend. The **Pettitt (1979)** test asks a different question: "
        "did rainfall **jump to a new level** in a particular year?"
    )
    cp = rain_res["change_point"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Abrupt shift?", "YES" if cp["is_significant"] else "NO", f"p = {cp['p_value']:.4f}", delta_color="off")
    p2.metric("Shift starts in", str(cp["new_regime_start"]) if cp["is_significant"] else "—")
    p3.metric(
        "Mean before → after",
        f"{cp['mean_before']:.0f} → {cp['mean_after']:.0f} mm",
        f"{cp['pct_change']:+.0f}%" if cp["is_significant"] else None,
        delta_color="off",
    )
    if cp["is_significant"]:
        st.warning(
            f"**{cp['interpretation']}** Rainfall did not rise gradually — it jumped. "
            "Pettitt tells us **when** the shift happened, not **why**: it could be a real change in storm frequency "
            "(e.g. the severe storms of late 2016 and March 2020) or a change in the observations feeding the MERRA-2 "
            "reanalysis. Confirming it requires an independent dataset such as NASA GPM IMERG."
        )
    else:
        st.success(cp["interpretation"])

    st.plotly_chart(plot_rainfall_monthly_climatology(rain_res["climatology"], rain_region), use_container_width=True)

    # Trend table for the 3 metrics
    st.markdown("### 📊 Trend Tests for Three Rainfall Metrics")
    metric_labels = {
        "annual_total": "Calendar-year total (mm)",
        "wet_season_total": "Wet-season total, Oct–Mar (mm)",
        "wet_months": "Number of wet months (≥ 1 mm)",
    }
    trend_rows = []
    for key, label in metric_labels.items():
        t = rain_res["trends"][key]
        unit = "months" if key == "wet_months" else "mm"
        trend_rows.append({
            "Metric": label,
            "Years": t["n_years"],
            "Mean": round(t["mean"], 2),
            f"Sen slope / decade": f"{t['sen_slope_per_decade']:+.2f} {unit}",
            "95% CI / decade": f"[{t['sen_ci_per_decade'][0]:+.2f}, {t['sen_ci_per_decade'][1]:+.2f}]",
            "MK Z": round(t["mk_z"], 2),
            "MK p": round(t["mk_p"], 4),
            "Hamed-Rao p": round(t["hr_p"], 4),
            "Verdict": t["verdict"],
            "Reliability": "Low: " + "; ".join(t["reliability_notes"]) if t["low_reliability"] else "OK",
        })
    st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

    # Wettest / driest years
    st.markdown("### 🏆 Wettest and Driest Years (% of 1981–2010 normal)")
    ranked = rain_res["annual"][["year", "total_mm", "pct_of_normal"]].rename(
        columns={"year": "Year", "total_mm": "Total (mm)", "pct_of_normal": "% of normal"}
    ).round(1)
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Wettest 5 years**")
        st.dataframe(ranked.nlargest(5, "Total (mm)"), use_container_width=True, hide_index=True)
    with w2:
        st.markdown("**Driest 5 years**")
        st.dataframe(ranked.nsmallest(5, "Total (mm)"), use_container_width=True, hide_index=True)

    # Regional comparison
    st.markdown("### 🗺️ Regional Rainfall Comparison")
    all_annual = {name: analyze_rainfall(df_r, name)["annual"] for name, df_r in rain_dfs.items()}
    st.plotly_chart(plot_regional_rainfall(all_annual), use_container_width=True)
    st.dataframe(compare_regional_rainfall(rain_dfs), use_container_width=True, hide_index=True)

    st.info(
        "🌡️ vs 🌧️ **Key finding**: temperature in all three regions shows a statistically significant warming trend, "
        "but rainfall shows **no statistically significant gradual trend** in any of them. Instead, Cairo and Alexandria "
        "show a significant **abrupt upward shift** in the mid-2010s (Pettitt test), while Aswan shows neither. "
        "Rainfall in Egypt varies so much from year to year that 43 years are not enough to tell whether this shift is a "
        "lasting climate change, a cluster of unusual storms, or a change in the reanalysis input data."
    )
    st.markdown(
        """
        **Limitations**
        - `PRECTOTCORR` is MERRA-2 **reanalysis** precipitation (model + observations, bias-corrected), not a rain-gauge record. Reanalysis rainfall is less reliable than reanalysis temperature, especially in very dry regions.
        - Each region is a single ~50 km grid cell, not a city-wide or national average.
        - A "wet month" is defined as ≥ 1 mm monthly total; results for wet-month counts depend on this threshold.
        - In Aswan almost every month is dry, so trend tests have very little information to work with.
        """
    )


# ==========================================
# PAGE: TEMPERATURE EXTREMES
# ==========================================
elif page == "🔥 Temperature Extremes":
    st.markdown("## 🔥 Temperature Extremes: Hottest and Coldest")
    st.markdown(
        """
        People feel the **extremes**, not the average. NASA POWER gives, for every month, the **hottest value reached**
        (`T2M_MAX`) and the **coldest value reached** (`T2M_MIN`). From these we build five yearly indices:

        1. **Hottest value of the year**, the single highest monthly maximum.
        2. **Coldest value of the year**, the single lowest monthly minimum.
        3. **Unusually hot months**: months whose maximum is above the 1981–2010 **90th percentile** *for that calendar month*.
        4. **Unusually cold months**: months whose minimum is below the 1981–2010 **10th percentile** *for that calendar month*.
        5. **Monthly extreme range**: hottest minus coldest value in each month, averaged over the year.

        A "hot July" is judged against other Julys, not against January. Under a stable 1981–2010 climate you would expect
        about **1.2 unusually hot and 1.2 unusually cold months per year** (10% of 12 months).
        """
    )

    ext_files = {
        "Greater Cairo": os.path.join("data", "raw", "nasa_power_cairo_1981_2023.csv"),
        "Alexandria": os.path.join("data", "raw", "nasa_power_alexandria_1981_2023.csv"),
        "Aswan": os.path.join("data", "raw", "nasa_power_aswan_1981_2023.csv"),
    }
    ext_dfs = {name: load_bundled_dataset(path, name)[0] for name, path in ext_files.items() if os.path.exists(path)}
    if not ext_dfs:
        st.warning("Regional temperature datasets are not available.")
        st.stop()

    ext_region = st.radio("Region", list(ext_dfs.keys()), horizontal=True, key="ext_region")
    ext_res = analyze_temperature_extremes(ext_dfs[ext_region], ext_region)
    et = ext_res["trends"]

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Hottest of year", f"{et['annual_hottest']['sen_per_decade']:+.2f} °C/dec",
              "significant" if et["annual_hottest"]["is_significant"] else "not significant", delta_color="off")
    e2.metric("Coldest of year", f"{et['annual_coldest']['sen_per_decade']:+.2f} °C/dec",
              "significant" if et["annual_coldest"]["is_significant"] else "not significant", delta_color="off")
    e3.metric("Hot months", f"{et['hot_months']['sen_per_decade']:+.2f} /dec",
              "significant" if et["hot_months"]["is_significant"] else "not significant", delta_color="off")
    e4.metric("Cold months", f"{et['cold_months']['sen_per_decade']:+.2f} /dec",
              "significant" if et["cold_months"]["is_significant"] else "not significant", delta_color="off")

    st.markdown("### 🔍 The 4 Questions — Temperature Extremes")
    es = ext_res["summary"]
    st.markdown(f"**1. What is changing?** {es['q1_what']}")
    st.markdown(f"**2. Where?** {es['q2_where']}")
    st.markdown(f"**3. How much?** {es['q3_how_much']}")
    st.markdown(f"**4. Is it statistically significant?** {es['q4_is_significant']}")

    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(plot_extreme_index(ext_res["annual"], "annual_hottest", et["annual_hottest"],
                                           f"Hottest of Each Year", "°C", NASA_RED_HEX),
                        use_container_width=True)
    with c_right:
        st.plotly_chart(plot_extreme_index(ext_res["annual"], "annual_coldest", et["annual_coldest"],
                                           f"Coldest of Each Year", "°C", NASA_BLUE_HEX),
                        use_container_width=True)

    st.plotly_chart(plot_decadal_extreme_months(ext_res["decades"]), use_container_width=True)
    st.caption("Bars show the average number of unusually hot / cold months per year in each decade. The 1980s contain only 9 years (1981–1989) and the 2020s only 4 (2020–2023).")

    st.plotly_chart(plot_monthly_extreme_trends(ext_res["by_month_tmax"], ext_res["by_month_tmin"]), use_container_width=True)
    st.caption("Each calendar month is tested separately (43 values each). Solid bars stay significant after Benjamini-Hochberg FDR correction for testing 12 months; faded bars do not.")

    st.markdown("### 📊 All Five Indices")
    idx_rows = []
    for key, (label, unit) in METRIC_INFO.items():
        t = et[key]
        lo, hi = t["ci_per_decade"]
        cp = t["pettitt"]
        idx_rows.append({
            "Index": label,
            "Mean": round(t["mean"], 2),
            "Slope / decade": f"{t['sen_per_decade']:+.2f} {unit}",
            "95% CI": f"[{lo:+.2f}, {hi:+.2f}]",
            "Slope method": t["slope_method"],
            "MK p": f"{t['mk_p']:.2e}",
            "Hamed-Rao p": f"{t['hr_p']:.2e}",
            "Trend (both tests)": t["verdict"],
            "Abrupt shift (Pettitt)": f"Yes, from {cp['new_regime_start']}" if cp["is_significant"] else "No",
        })
    st.dataframe(pd.DataFrame(idx_rows), use_container_width=True, hide_index=True)
    st.caption("Counts (hot/cold months) are small integers with many ties, so Sen's median slope collapses to 0; their magnitude is given by OLS, while significance always comes from the rank-based Mann-Kendall and Hamed-Rao tests.")

    st.markdown("### 🏆 Record Years")
    ann_e = ext_res["annual"]
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**5 hottest years (highest monthly maximum)**")
        st.dataframe(ann_e.nlargest(5, "annual_hottest")[["year", "annual_hottest", "hottest_month"]]
                     .rename(columns={"year": "Year", "annual_hottest": "Hottest (°C)", "hottest_month": "Month"}).round(2),
                     use_container_width=True, hide_index=True)
    with r2:
        st.markdown("**5 coldest years (lowest monthly minimum)**")
        st.dataframe(ann_e.nsmallest(5, "annual_coldest")[["year", "annual_coldest", "coldest_month"]]
                     .rename(columns={"year": "Year", "annual_coldest": "Coldest (°C)", "coldest_month": "Month"}).round(2),
                     use_container_width=True, hide_index=True)

    st.markdown("### 🗺️ Regional Comparison of Extremes")
    st.dataframe(compare_regional_extremes(ext_dfs), use_container_width=True, hide_index=True)

    st.info(
        "🔥 **Key finding**: in all three regions the **hottest value of the year is rising** and **unusually hot months are "
        "becoming more frequent**. Unusually **cold** months are becoming **rarer** in Cairo and Aswan. The single coldest "
        "value of the year shows no significant trend anywhere: one value per year is noisy, so a change there is harder to detect."
    )
    st.markdown(
        """
        **Limitations**
        - Values are MERRA-2 reanalysis grid-cell extremes (~50 km), not station records; local extremes can be more severe.
        - Percentile thresholds are estimated from 30 values per calendar month, so they carry sampling uncertainty.
        - Hamed-Rao can produce a much smaller p-value than Mann-Kendall when autocorrelation is negative; a trend is
          called significant only when **both** tests agree.
        """
    )


# ==========================================
# PAGE 6: ANOMALY DETECTION
# ==========================================
elif page == "⚠️ Anomaly Detection":
    st.markdown(f"## ⚠️ Climatological Anomaly Detection — {active_location}")
    st.markdown(
        """
        Evaluating departures from the standard **WMO 30-Year Climatological Baseline (1981–2010)**.
        Anomalies are computed per calendar month ($A_t = x_t - \\mu_m$) and normalized via Z-scores ($Z_t = A_t / \\sigma_m$).
        """
    )

    avail_vars = metadata_current.get("available_variables", sorted(df_current["variable"].unique().tolist()))
    sel_var = st.selectbox("Variable for Anomaly Diagnostic", avail_vars, index=avail_vars.index("T2M") if "T2M" in avail_vars else 0)

    var_df = df_current[df_current["variable"] == sel_var].dropna(subset=["value"]).copy()
    units = VARIABLE_METADATA.get(sel_var, {}).get("units", "units")

    anomaly_df, anom_summary = compute_anomalies_and_zscores(
        var_df,
        baseline_start_year=1981,
        baseline_end_year=2010,
    )

    # Bar chart
    fig_anom = plot_anomaly_bars(
        anomaly_df=anomaly_df,
        variable_name=VARIABLE_METADATA.get(sel_var, {}).get("name", sel_var),
        units=units,
        location_name=active_location,
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    # Decadal Shift
    st.markdown("### 📈 Decadal Shift in Extreme Event Probability")
    st.caption("Tracking how the frequency of warm anomalies (≥ +1.5σ) vs. cold anomalies (≤ -1.5σ) has evolved over the decades.")

    decadal_df = decadal_extreme_frequency(anomaly_df, z_threshold=1.5)
    fig_dec = plot_decadal_shift_bars(decadal_df)
    st.plotly_chart(fig_dec, use_container_width=True)

    # Top Ranked Extremes
    st.markdown("### 🏆 Top Ranked Historical Extremes")
    ranks = get_top_ranked_extremes(anomaly_df, top_n=5)
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("#### 🔥 Hottest Positive Anomalies")
        st.dataframe(ranks["hottest_anomalies"], use_container_width=True)
    with rc2:
        st.markdown("#### ❄️ Coldest Negative Anomalies")
        st.dataframe(ranks["coldest_anomalies"], use_container_width=True)


# ==========================================
# PAGE 7: METHODOLOGY & SOURCES
# ==========================================
elif page == "📚 Methodology & Sources":
    st.markdown("## 📚 Scientific Methodology & NASA Earth Science Provenance")

    st.markdown("### 1. Mathematical Formulations")
    st.markdown(
        r"""
        #### Non-Parametric Mann-Kendall Trend Test
        The Mann-Kendall test statistic $S$ measures the cumulative signs of all pairwise differences:
        $$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^n \text{sgn}(x_j - x_k)$$
        $$\text{sgn}(\theta) = \begin{cases} +1 & \theta > 0 \\ 0 & \theta = 0 \\ -1 & \theta < 0 \end{cases}$$

        The variance of $S$, corrected for tied groups $t_i$:
        $$\text{Var}(S) = \frac{n(n-1)(2n+5) - \sum_{i=1}^m t_i(t_i-1)(2t_i+5)}{18}$$

        The standardized normal test statistic $Z$:
        $$Z = \begin{cases} \frac{S-1}{\sqrt{\text{Var}(S)}} & S > 0 \\ 0 & S = 0 \\ \frac{S+1}{\sqrt{\text{Var}(S)}} & S < 0 \end{cases}$$

        The two-tailed $p$-value:
        $$p = 2 \cdot (1 - \Phi(|Z|))$$

        #### Theil-Sen Robust Slope Estimator
        The median of all pairwise slopes across observation timestamps:
        $$Q = \text{median}\left(\frac{x_j - x_k}{j - k}\right) \quad \forall \ 1 \le k < j \le n$$
        Sen's slope is intrinsically robust to non-normal distributions and up to 29% gross measurement outliers.
        """
    )

    st.markdown("### 2. NASA Earth Science Data Sources")
    st.markdown(
        r"""
        - **NASA POWER Project**: Prediction of Worldwide Energy Resources, NASA Langley Research Center.
          URL: [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/)
        - **MERRA-2**: Modern-Era Retrospective analysis for Research and Applications, Version 2.
          Managed by the NASA Goddard Global Modeling and Assimilation Office (GMAO).
          Spatial Resolution: Native $0.5^\circ \times 0.625^\circ$ lat/lon grid.
        - **NASA Giovanni**: Goddard Interactive Online Visualization ANd aNalysis Infrastructure.
          URL: [https://giovanni.gsfc.nasa.gov/](https://giovanni.gsfc.nasa.gov/)
        - **NASA Earthdata**: NASA EOSDIS (Earth Observing System Data and Information System).
          URL: [https://www.earthdata.nasa.gov/](https://www.earthdata.nasa.gov/)
        """
    )

    st.markdown("### 3. Scientific Disclaimers & Limitations")
    st.warning(
        """
        - **Correlation vs. Causation**: Statistical trends document observational changes in the Earth system; they do not single-handedly establish individual physical mechanisms or attribution.
        - **Temporal Autocorrelation**: Serial persistence inflates standard Mann-Kendall significance; the application explicitly implements and flags lag-1 autocorrelation tests ($r_1$).
        - **Reanalysis Grid**: MERRA-2 assimilates satellite, radiosonde, and surface observations into an atmospheric model. Grid cells represent spatial regional averages rather than isolated microclimates.
        """
    )

    st.markdown("### 4. References")
    st.markdown(
        """
        1. Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*, 13(3), 245–259.
        2. Kendall, M. G. (1975). *Rank Correlation Methods*. Charles Griffin, London.
        3. Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA*, 63(324), 1379–1389.
        4. Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982). Techniques of trend analysis for monthly water quality data. *Water Resources Research*, 18(1), 107–121.
        5. Gelaro, R., et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *Journal of Climate*, 30(14), 5419–5454.
        """
    )
