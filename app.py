"""
EarthTrend Explorer — NASA Space Apps Challenge 2026
Challenge: "Be An Earth System Trend Detective!"
Team: Machine Without Learning

Main Streamlit Application
"""

from __future__ import annotations
import io
import html
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly_detection import (
    analyze_compound_anomalies,
    compare_regional_anomalies,
    compute_anomalies_and_zscores,
    compute_precipitation_anomalies,
    compute_spi,
    compute_temperature_anomalies,
    cross_variable_anomaly_correlation,
    decadal_extreme_frequency,
    detect_persistent_anomalies,
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
from src.forecasting import (
    generate_12month_forecast,
    interval_coverage_verdict,
    verify_forecast_against_observations,
)
from src.extremes_analysis import METRIC_INFO, analyze_temperature_extremes, compare_regional_extremes
from src.rainfall_analysis import analyze_rainfall, compare_regional_rainfall
from src.narrative import extremes_story, forecast_story, rainfall_story, regional_story, verification_story
from src.trend_analysis import (
    compare_regional_warming_rates,
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
    plot_forecast_verification,
    plot_anomaly_bars,
    plot_autocorrelation_acf,
    plot_climatology_heatmap,
    plot_compound_scatter,
    plot_decadal_extreme_months,
    plot_decadal_shift_bars,
    plot_extreme_index,
    plot_forecast_residuals_diagnostic,
    plot_forecast_with_intervals,
    plot_future_anomaly_bars,
    plot_lagged_cross_correlation,
    plot_model_backtest_comparison,
    plot_monthly_extreme_trends,
    plot_rainfall_annual_totals,
    plot_rainfall_monthly_climatology,
    plot_regional_comparison,
    plot_regional_rainfall,
    plot_spi_series,
    plot_standardized_anomalies_zscore,
    plot_temperature_extremes_comparison,
    plot_timeseries_with_trend,
)
from src.visualization import NASA_BLUE as NASA_BLUE_HEX, NASA_RED as NASA_RED_HEX

# Page configuration
st.set_page_config(
    page_title="EarthTrend Explorer | NASA Space Apps 2026",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom NASA-inspired CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ---------- Global: dark "space" theme ---------- */
    html, body, [class*="css"], .stMarkdown, .stText, p, li, label {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 85% -10%, rgba(76, 141, 255, 0.14), transparent 60%),
            radial-gradient(900px 500px at -10% 110%, rgba(252, 61, 33, 0.08), transparent 60%),
            #070D1C;
    }
    .block-container {
        padding-top: 3.5rem;
        max-width: 1280px;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
        letter-spacing: -0.01em;
        color: #F1F5F9 !important;
    }
    h2 {
        font-weight: 800 !important;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    h3 {
        font-weight: 700 !important;
        margin-top: 1.4rem !important;
    }
    p, li { color: #CBD5E1; line-height: 1.65; }
    strong, b { color: #F1F5F9; }
    code {
        background: rgba(76, 141, 255, 0.12) !important;
        color: #9CC2FF !important;
        border-radius: 4px;
    }
    a { color: #7CC4FF !important; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1630 0%, #070D1C 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 6px 10px;
        border-radius: 8px;
        transition: background 0.15s ease;
        width: 100%;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(76, 141, 255, 0.12);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(252, 61, 33, 0.14);
        box-shadow: inset 3px 0 0 #FC3D21;
    }

    /* ---------- Header banner ---------- */
    .nasa-header {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(600px 200px at 90% 0%, rgba(124, 196, 255, 0.35), transparent 70%),
            linear-gradient(135deg, #0B3D91 0%, #06215A 55%, #031033 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 28px;
        border: 1px solid rgba(124, 196, 255, 0.25);
        box-shadow: 0 10px 40px rgba(11, 61, 145, 0.35);
    }
    .nasa-header::after {
        /* faint "star field" */
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 70% 20%, rgba(255,255,255,0.6), transparent),
            radial-gradient(1.5px 1.5px at 85% 70%, rgba(255,255,255,0.5), transparent),
            radial-gradient(1px 1px at 40% 80%, rgba(255,255,255,0.5), transparent),
            radial-gradient(1px 1px at 55% 50%, rgba(255,255,255,0.4), transparent);
        pointer-events: none;
    }
    .nasa-header h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        margin: 0;
        font-size: 2.4rem;
        letter-spacing: -0.02em;
        border: none !important;
    }
    .nasa-badge {
        display: inline-block;
        background-color: #FC3D21;
        color: white;
        padding: 4px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .nasa-subbadge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #E6EDF7;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        margin-right: 8px;
    }

    /* ---------- Question boxes (Home / Trend pages) ---------- */
    .q-box {
        background: linear-gradient(180deg, rgba(15, 27, 51, 0.95), rgba(11, 20, 40, 0.95));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-left: 4px solid #4C8DFF;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 16px;
    }
    .q-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #7CC4FF;
        margin-bottom: 6px;
    }
    .q-answer {
        font-size: 0.95rem;
        color: #CBD5E1;
        line-height: 1.6;
    }

    /* ---------- Stat cards (custom HTML) and st.metric ---------- */
    .stat-card, div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #11203D 0%, #0C1830 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .stat-card { text-align: center; }
    .stat-card:hover, div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(76, 141, 255, 0.45);
    }
    .stat-label, div[data-testid="stMetricLabel"] p {
        font-size: 0.76rem !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F1F5F9;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 800;
        color: #F1F5F9;
        font-size: clamp(1.15rem, 2.1vw, 1.75rem) !important;
    }
    /* never truncate metric text with "..." - wrap it instead */
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] div,
    div[data-testid="stMetricDelta"] div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    .stat-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 4px;
    }
    .sig-green { color: #34D399 !important; }
    .sig-red { color: #F87171 !important; }

    /* ---------- Charts, tables, alerts ---------- */
    div[data-testid="stPlotlyChart"] {
        background: #0F1B33;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        padding: 6px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(252, 61, 33, 0.6);
    }
    .stButton > button:hover {
        border-color: #FC3D21;
        color: #FFFFFF;
        background: rgba(252, 61, 33, 0.18);
    }

    /* =====================================================
       DASHBOARD LAYOUT (top navbar, no sidebar)
       ===================================================== */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebarCollapsedControl"],
    button[data-testid="stExpandSidebarButton"],
    .stAppDeployButton,
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    /* ---------- Top bar ---------- */
    .st-key-topbar {
        position: sticky;
        top: 0.5rem;
        z-index: 999;
        background: rgba(11, 20, 40, 0.82);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 8px 14px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 10px;
    }
    .st-key-topbar div[data-testid="stHorizontalBlock"] { gap: 0.6rem; }
    .brand { display: flex; align-items: center; gap: 10px; min-height: 40px; }
    .st-key-topbar div[data-testid="stColumn"]:nth-child(1) { min-height: 42px; }
    .brand-logo {
        font-size: 1.55rem;
        width: 40px; height: 40px;
        display: inline-flex; align-items: center; justify-content: center;
        border-radius: 12px;
        background: radial-gradient(circle at 30% 30%, #2B6BFF, #0B3D91);
        box-shadow: 0 0 18px rgba(76, 141, 255, 0.45);
    }
    .brand-name { font-weight: 800; font-size: 1.12rem; color: #FFFFFF; letter-spacing: -0.01em; line-height: 1.1; }
    .brand-tag { font-size: 0.7rem; color: #94A3B8; letter-spacing: 0.04em; }

    /* navigation radio -> pill tabs */
    .st-key-topbar div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 4px;
        flex-wrap: wrap;
        justify-content: center;
    }
    .st-key-topbar label[data-testid="stRadioOption"] {
        margin: 0 !important;
        padding: 6px 12px !important;
        border-radius: 999px;
        border: 1px solid transparent;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    .st-key-topbar label[data-testid="stRadioOption"] > div > div:first-child { display: none !important; } /* hide radio dot */
    .st-key-topbar label[data-testid="stRadioOption"] p {
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        color: #A8B6CC !important;
        white-space: nowrap;
    }
    .st-key-topbar label[data-testid="stRadioOption"]:hover {
        background: rgba(76, 141, 255, 0.12);
    }
    .st-key-topbar label[data-testid="stRadioOption"]:hover p { color: #FFFFFF !important; }
    .st-key-topbar label[data-testid="stRadioOption"]:has(input:checked) {
        background: linear-gradient(135deg, rgba(252, 61, 33, 0.95), rgba(255, 110, 70, 0.9));
        border-color: rgba(255, 140, 110, 0.6);
        box-shadow: 0 4px 16px rgba(252, 61, 33, 0.35);
    }
    .st-key-topbar label[data-testid="stRadioOption"]:has(input:checked) p { color: #FFFFFF !important; }
    .st-key-topbar div[data-testid="stSelectbox"] > div { margin: 0; }
    .st-key-topbar div[data-baseweb="select"] > div {
        border-radius: 999px !important;
        background: rgba(76, 141, 255, 0.10) !important;
        border-color: rgba(76, 141, 255, 0.35) !important;
        font-weight: 600;
    }
    .st-key-topbar div[data-testid="stPopover"] button {
        border-radius: 999px !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        background: rgba(148, 163, 184, 0.08) !important;
    }

    /* ---------- Context strip ---------- */
    .context-strip {
        display: flex; flex-wrap: wrap; gap: 8px;
        margin: 2px 4px 18px 4px;
    }
    .context-strip span {
        font-size: 0.76rem;
        color: #94A3B8;
        background: rgba(148, 163, 184, 0.07);
        border: 1px solid rgba(148, 163, 184, 0.14);
        padding: 3px 10px;
        border-radius: 999px;
    }
    .context-strip span:first-child { color: #E6EDF7; border-color: rgba(252, 61, 33, 0.45); }

    /* ---------- Hero (Home) ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 44px 40px 40px 40px;
        border-radius: 22px;
        margin-bottom: 22px;
        border: 1px solid rgba(124, 196, 255, 0.22);
        background:
            radial-gradient(520px 260px at 88% 10%, rgba(255, 90, 60, 0.30), transparent 70%),
            radial-gradient(700px 360px at 10% 120%, rgba(76, 141, 255, 0.35), transparent 70%),
            linear-gradient(135deg, #0A1A3F 0%, #071026 100%);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
    }
    .hero::after {
        content: "";
        position: absolute; inset: 0; pointer-events: none;
        background-image:
            radial-gradient(1px 1px at 12% 22%, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 32% 68%, rgba(255,255,255,0.5), transparent),
            radial-gradient(1.5px 1.5px at 58% 18%, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 76% 58%, rgba(255,255,255,0.5), transparent),
            radial-gradient(1px 1px at 90% 82%, rgba(255,255,255,0.6), transparent),
            radial-gradient(1px 1px at 46% 40%, rgba(255,255,255,0.4), transparent);
    }
    .hero-eyebrow {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
        color: #FF8A70; margin-bottom: 10px;
    }
    .hero-title {
        font-size: clamp(2.2rem, 4.6vw, 3.6rem);
        font-weight: 800; line-height: 1.05; letter-spacing: -0.03em;
        color: #FFFFFF;
    }
    .hero-lead { margin-top: 14px; font-size: 1.15rem; color: #C7D2E3; max-width: 760px; line-height: 1.55; }
    .hero-answer {
        display: inline-block;
        padding: 0 10px; margin: 0 2px;
        border-radius: 8px;
        background: linear-gradient(135deg, #FC3D21, #FF7A45);
        color: #FFFFFF; font-weight: 800;
    }

    /* ---------- KPI cards ---------- */
    .kpi-card {
        position: relative;
        height: 100%;
        min-height: 128px;
        padding: 18px 18px 16px 18px;
        border-radius: 16px;
        background: linear-gradient(180deg, #11203D 0%, #0B162D 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.30);
        overflow: hidden;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-card::before {
        content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px;
        background: var(--kpi-accent, #4C8DFF);
    }
    .kpi-card::after {
        content: ""; position: absolute; right: -40px; top: -40px; width: 120px; height: 120px;
        border-radius: 50%;
        background: radial-gradient(circle, var(--kpi-glow, rgba(76,141,255,0.25)), transparent 70%);
    }
    .kpi-red    { --kpi-accent: #FF5A3C; --kpi-glow: rgba(255, 90, 60, 0.28); }
    .kpi-orange { --kpi-accent: #F59E0B; --kpi-glow: rgba(245, 158, 11, 0.25); }
    .kpi-blue   { --kpi-accent: #4C8DFF; --kpi-glow: rgba(76, 141, 255, 0.28); }
    .kpi-green  { --kpi-accent: #34D399; --kpi-glow: rgba(52, 211, 153, 0.22); }
    .kpi-slate  { --kpi-accent: #64748B; --kpi-glow: rgba(100, 116, 139, 0.20); }
    .kpi-card:hover { border-color: var(--kpi-accent); }
    .kpi-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: #94A3B8; margin-bottom: 8px;
    }
    .kpi-value {
        font-size: clamp(1.5rem, 2.6vw, 2.1rem); font-weight: 800; color: #FFFFFF;
        letter-spacing: -0.02em; line-height: 1.1;
    }
    .kpi-sub { margin-top: 8px; font-size: 0.8rem; color: #94A3B8; line-height: 1.4; }

    /* ---------- Page header ---------- */
    .page-head { display: flex; align-items: center; gap: 16px; margin: 6px 0 20px 0; }
    .page-icon {
        font-size: 1.7rem; width: 56px; height: 56px; flex: none;
        display: flex; align-items: center; justify-content: center;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(76,141,255,0.25), rgba(252,61,33,0.18));
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    .page-title { font-size: 1.9rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1.15; }
    .page-sub { font-size: 0.95rem; color: #94A3B8; margin-top: 2px; }

    /* ---------- Tabs ---------- */
    div[data-testid="stTabs"] div[role="tablist"] {
        gap: 6px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    }
    div[data-testid="stTabs"] button[role="tab"] {
        padding: 8px 16px;
        border-radius: 10px 10px 0 0;
    }
    div[data-testid="stTabs"] button[role="tab"] p { font-weight: 600; font-size: 0.95rem; color: #A8B6CC; }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: rgba(76, 141, 255, 0.10); }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #FFFFFF; }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { background-color: #FC3D21; }

    /* ---------- Findings list ---------- */
    .finding {
        display: flex; gap: 14px; align-items: flex-start;
        padding: 14px 16px; margin-bottom: 10px;
        border-radius: 14px;
        background: rgba(15, 27, 51, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #CBD5E1; font-size: 0.93rem; line-height: 1.55;
    }
    .finding-num {
        flex: none; font-weight: 800; font-size: 0.85rem; color: #FF8A70;
        background: rgba(252, 61, 33, 0.12); border: 1px solid rgba(252, 61, 33, 0.35);
        border-radius: 8px; padding: 2px 8px;
    }

    .section-gap { height: 18px; }
    .plain-words {
        margin: 6px 0 18px 0; padding: 16px 18px 6px 18px; border-radius: 16px;
        background: linear-gradient(180deg, rgba(52, 211, 153, 0.08), rgba(15, 27, 51, 0.85));
        border: 1px solid rgba(52, 211, 153, 0.30);
    }
    .plain-words .pw-title { font-weight: 800; color: #FFFFFF; font-size: 1.02rem; margin-bottom: 8px; }
    .plain-words ul { list-style: none; padding: 0; margin: 0; }
    .plain-words li {
        display: flex; gap: 12px; align-items: flex-start;
        color: #D7E0EE; font-size: 0.95rem; line-height: 1.55; margin-bottom: 10px;
    }
    .plain-words .pw-tag {
        flex: none; min-width: 118px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; padding: 3px 8px; border-radius: 8px; text-align: center;
        background: rgba(148, 163, 184, 0.12); color: #CBD5E1;
    }
    .plain-words .pw-observed .pw-tag { background: rgba(76, 141, 255, 0.16); color: #9CC2FF; }
    .plain-words .pw-statistical .pw-tag { background: rgba(52, 211, 153, 0.14); color: #6EE7B7; }
    .plain-words .pw-projected .pw-tag { background: rgba(245, 158, 11, 0.14); color: #FCD34D; }
    .plain-words .pw-caution .pw-tag { background: rgba(255, 90, 60, 0.14); color: #FF9F8A; }
    @media (max-width: 640px) {
        .plain-words li { flex-direction: column; gap: 4px; font-size: 0.88rem; }
        .plain-words .pw-tag { min-width: 0; }
    }
    /* headings of the original pages shown inside tabs become sub-headings */
    div[data-testid="stTabs"] h2 {
        font-size: 1.35rem !important;
        border-bottom: none !important;
        padding-bottom: 0 !important;
        color: #E6EDF7 !important;
    }
    .footer {
        margin-top: 48px; padding-top: 18px;
        border-top: 1px solid rgba(148, 163, 184, 0.14);
        text-align: center; font-size: 0.8rem; color: #64748B;
    }
    div[data-testid="stExpander"] details {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 27, 51, 0.6);
    }

    /* =====================================================
       RESPONSIVE LAYOUT
       laptop > 1180px · small laptop 960–1180 · tablet 640–960 · phone < 640
       ===================================================== */
    /* KPI rows: 4 → 2 → 1 cards per row */
    :is(.st-key-kpis, .st-key-fc_kpis, .st-key-rc_kpis) div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 1rem; }
    .st-key-topbar label[data-testid="stRadioOption"] p { transition: font-size 0.15s ease; }

    @media (max-width: 1180px) {
        .brand-tag { display: none; }
        .st-key-topbar label[data-testid="stRadioOption"] { padding: 5px 9px !important; }
        .st-key-topbar label[data-testid="stRadioOption"] p { font-size: 0.8rem !important; }
        .kpi-value { font-size: 1.6rem; }
    }

    @media (max-width: 1024px) {
        :is(.st-key-kpis, .st-key-fc_kpis, .st-key-rc_kpis) div[data-testid="stColumn"] {
            flex: 1 1 calc(50% - 0.5rem) !important;
            min-width: calc(50% - 0.5rem) !important;
            width: calc(50% - 0.5rem) !important;
        }
    }

    /* tablet and below: navbar becomes two rows - [brand · region · gear] / [scrollable pills] */
    @media (max-width: 960px) {
        .st-key-topbar { position: relative; top: 0; padding: 10px 12px; }
        .st-key-topbar div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            row-gap: 8px;
        }
        .st-key-topbar div[data-testid="stColumn"] { min-width: 0 !important; }
        .st-key-topbar div[data-testid="stColumn"]:nth-child(1) { flex: 1 1 auto !important; width: auto !important; order: 1; }
        .st-key-topbar div[data-testid="stColumn"]:nth-child(3) { flex: 0 0 180px !important; width: 180px !important; order: 2; }
        .st-key-topbar div[data-testid="stColumn"]:nth-child(4) { flex: 0 0 64px !important; width: 64px !important; order: 3; }
        .st-key-topbar div[data-testid="stColumn"]:nth-child(2) { flex: 1 1 100% !important; width: 100% !important; order: 4; }
        .st-key-topbar div[role="radiogroup"] {
            flex-wrap: nowrap !important;
            justify-content: flex-start !important;
            overflow-x: auto;
            scrollbar-width: none;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 2px;
        }
        .st-key-topbar div[role="radiogroup"]::-webkit-scrollbar { display: none; }
        .st-key-topbar div[role="radiogroup"] > div { flex: none; }
        .page-title { font-size: 1.55rem; }
        .hero { padding: 34px 28px 30px 28px; }
    }

    /* phone */
    @media (max-width: 640px) {
        .block-container { padding-left: 0.85rem !important; padding-right: 0.85rem !important; padding-top: 0.6rem !important; }
        .st-key-topbar { border-radius: 14px; margin-bottom: 6px; }
        .st-key-topbar div[data-testid="stColumn"]:nth-child(1) { flex: 1 1 100% !important; width: 100% !important; }
        .st-key-topbar div[data-testid="stColumn"]:nth-child(3) { flex: 1 1 auto !important; width: auto !important; }
        .st-key-topbar div[data-testid="stColumn"]:nth-child(4) { flex: 0 0 56px !important; width: 56px !important; }
        .brand-logo { width: 34px; height: 34px; font-size: 1.25rem; border-radius: 10px; }
        .brand-name { font-size: 1rem; }

        .context-strip { gap: 6px; margin-bottom: 12px; }
        .context-strip span { font-size: 0.7rem; padding: 2px 8px; }
        .context-strip span:nth-child(n+3) { display: none; } /* keep region + coordinates only */

        .hero { padding: 24px 18px 22px 18px; border-radius: 16px; margin-bottom: 14px; }
        .hero-eyebrow { font-size: 0.66rem; letter-spacing: 0.1em; }
        .hero-title { font-size: 1.95rem; }
        .hero-lead { font-size: 0.98rem; }

        :is(.st-key-kpis, .st-key-fc_kpis, .st-key-rc_kpis) div[data-testid="stHorizontalBlock"] { gap: 0.6rem; }
        :is(.st-key-kpis, .st-key-fc_kpis, .st-key-rc_kpis) div[data-testid="stColumn"] {
            flex: 1 1 calc(50% - 0.3rem) !important;
            min-width: calc(50% - 0.3rem) !important;
            width: calc(50% - 0.3rem) !important;
        }
        .kpi-card { min-height: 112px; padding: 14px 12px 12px 12px; border-radius: 14px; }
        .kpi-label { font-size: 0.62rem; letter-spacing: 0.05em; margin-bottom: 6px; }
        .kpi-value { font-size: 1.3rem; }
        .kpi-sub { font-size: 0.7rem; margin-top: 6px; }

        .page-head { gap: 12px; margin-bottom: 14px; }
        .page-icon { width: 44px; height: 44px; font-size: 1.3rem; border-radius: 12px; }
        .page-title { font-size: 1.25rem; }
        .page-sub { font-size: 0.82rem; }

        h2 { font-size: 1.35rem !important; }
        h3 { font-size: 1.12rem !important; }
        .q-box { padding: 12px 14px; }
        .q-answer { font-size: 0.88rem; }
        .finding { padding: 12px; font-size: 0.86rem; }

        /* tabs scroll sideways instead of squashing */
        div[data-testid="stTabs"] div[role="tablist"] { overflow-x: auto; scrollbar-width: none; flex-wrap: nowrap; }
        div[data-testid="stTabs"] button[role="tab"] { padding: 6px 10px; flex: none; }
        div[data-testid="stTabs"] button[role="tab"] p { font-size: 0.85rem; }

        /* charts: smaller titles, no floating toolbar over the title */
        div[data-testid="stPlotlyChart"] { padding: 2px; border-radius: 12px; }
        .js-plotly-plot .gtitle { font-size: 13px !important; }
        .js-plotly-plot .modebar-container { display: none !important; }
        .js-plotly-plot .legendtext { font-size: 10px !important; }
    }

    @media (max-width: 360px) {
        :is(.st-key-kpis, .st-key-fc_kpis, .st-key-rc_kpis) div[data-testid="stColumn"] { flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }
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


def render_four_questions(summary: Dict[str, str], topic: str) -> None:
    """Shows the 4 challenge questions as styled cards (2 x 2 grid)."""
    import html as _html
    st.markdown(f"### 🔍 The 4 Questions — {topic}")
    items = [
        ("1. What is changing?", summary["q1_what"]),
        ("2. Where is it changing?", summary["q2_where"]),
        ("3. How much is it changing?", summary["q3_how_much"]),
        ("4. Is it statistically significant?", summary["q4_is_significant"]),
    ]
    for row in (items[:2], items[2:]):
        cols = st.columns(2)
        for col, (title, text) in zip(cols, row):
            col.markdown(
                f'<div class="q-box"><div class="q-title">{title}</div>'
                f'<div class="q-answer">{_html.escape(text)}</div></div>',
                unsafe_allow_html=True,
            )


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




# ==========================================
# PAGE 1: HOME & OVERVIEW
# ==========================================
def page_about() -> None:
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
def page_data_explorer() -> None:
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
def page_time_series() -> None:
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
def page_trend_engine() -> None:
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
                <div class="stat-val" style="color: {'#34D399' if mk_result['trend'] == 'increasing' else ('#F87171' if mk_result['trend'] == 'decreasing' else '#94A3B8')}">
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
            - **Significant Lags Detected**: `{hr_result['significant_lags_count']}` lag(s) exceeding the critical bound: {hr_result.get('significant_lags', [])}.
            - **Variance Correction Factor ($\\\\eta = n/n^*$)**: `{hr_result['variance_correction_factor']:.6f}` adjusting $\\text{{Var}}(S) = {hr_result['original_var_s']:.2f} \\to \\text{{Var}}^*(S) = {hr_result['corrected_var_s']:.2f}$.
            - **Variance Inflation vs Deflation & Sensitivity**:
              - Positive autocorrelation ($\\\\rho_i > 0$) yields $\\\\eta > 1$ (variance inflation), reducing $|Z^*|$ to prevent false positive trends from persistent runs.
              - Negative autocorrelation ($\\\\rho_i < 0$) yields $\\\\eta < 1$ (variance deflation), increasing $|Z^*|$. For example, in Alexandria annual mean temperature, a significant negative rank autocorrelation at lag 12 ($\\\\rho_{{12}} = -0.301$ vs critical bound $\\pm 0.299$) produces $\\\\eta = 0.780520$, deflating variance from $9130.3$ to $7126.4$ and raising $Z$ from $+5.50$ ($p = 3.70 \\times 10^{{-8}}$) to $Z^* = +6.23$ ($p^* = 4.64 \\times 10^{{-10}}$).
              - Because all lags up to $n-1$ are tested, the correction factor is sensitive to lag selection and empirical high-order correlations.
              - Original Mann-Kendall and Hamed-Rao modified tests address related but distinct inferential hypotheses; neither is universally conservative across all autocorrelation structures.
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
        non_sig = cal_breakdown[~cal_breakdown["sig_nominal"]]["month_name"].tolist()
        dropped_bh = cal_breakdown[cal_breakdown["sig_nominal"] & ~cal_breakdown["sig_bh"]]["month_name"].tolist()
        dropped_holm = cal_breakdown[cal_breakdown["sig_bh"] & ~cal_breakdown["sig_holm"]]["month_name"].tolist()

        fdr_detail = (
            f"Months nominally significant that do not survive FDR: {', '.join(dropped_bh)}."
            if dropped_bh else "All nominally significant months survive FDR."
        )
        holm_detail = (
            f"Months surviving FDR that do not survive conservative FWER: {', '.join(dropped_holm)}."
            if dropped_holm else "All FDR-significant months survive FWER."
        )
        non_sig_detail = (
            f"Months without statistically detectable monotonic trends: {', '.join(non_sig)}."
            if non_sig else "All 12 calendar months exhibit statistically significant trends."
        )

        st.info(
            f"**Multiple Testing Synthesis Across 12 Calendar Months**:\n\n"
            f"- **Nominal Unadjusted ($p < {alpha_val}$)**: **{len(nominal_sig)} of 12 months** ({', '.join(nominal_sig) if nominal_sig else 'None'}).\n"
            f"- **Benjamini-Hochberg FDR ($q < {alpha_val}$)**: **{len(bh_sig)} of 12 months** ({', '.join(bh_sig) if bh_sig else 'None'}). {fdr_detail}\n"
            f"- **Holm-Bonferroni FWER ($\alpha < {alpha_val}$)**: **{len(holm_sig)} of 12 months** ({', '.join(holm_sig) if holm_sig else 'None'}). {holm_detail}\n"
            f"- **Non-Significant Calendar Months**: {non_sig_detail} Note: In Greater Cairo T2M, exactly 5 months survive Holm FWER (Jul, Aug, Sep, Nov, Dec) and 6 months survive BH FDR (Jul, Aug, Sep, Oct, Nov, Dec). June is not statistically significant under any test (raw p = 0.094, BH p = 0.113). Absence of statistically significant trend detection is not evidence of thermal stability; higher natural interannual variability reduces test power."
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
def page_regional() -> None:
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
            "📐 **Regional Warming Comparison**: Although point estimates of Sen's slopes vary across regions "
            f"(Cairo: +0.293 {units}/dec, Alexandria: +0.261 {units}/dec, Aswan: +0.434 {units}/dec), Aswan exhibits a higher "
            "nominal warming slope, but the available confidence intervals overlap and the current interaction analysis does not "
            "establish statistically significant regional divergence after multiple-comparison considerations."
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

        # Formal test: do two regions warm at different speeds?
        st.markdown("### ⚖️ Is one region warming faster than another?")
        st.markdown(
            "Seeing +0.43 °C/decade in Aswan and +0.29 °C/decade in Cairo is not proof that Aswan warms faster — "
            "both numbers carry uncertainty. We test the **difference** directly on the paired series "
            "*annual(A) − annual(B)*, which cancels the weather the regions share. A pair counts as different only if "
            "OLS with Newey-West errors **and** Mann-Kendall/Hamed-Rao both reject equal rates, **and** the result "
            "survives Holm correction for testing three pairs."
        )
        diff_labels = {"T2M": "mean temperature", "T2M_MAX": "monthly hottest temperature", "T2M_MIN": "monthly coldest temperature"}
        diff_var = st.radio(
            "Temperature measure", list(diff_labels.keys()), horizontal=True, key="diff_var",
            format_func=lambda v: f"{v} · {diff_labels[v]}",
        )
        diff_df = cached_warming_diff(diff_var)
        if not diff_df.empty:
            render_plain_words(regional_story(diff_df, diff_labels[diff_var]))
            st.dataframe(pd.DataFrame({
                "Pair": diff_df["faster_region"] + " vs " + diff_df["slower_region"],
                "Rates (°C/decade)": [f"{a:+.2f} vs {b:+.2f}" for a, b in zip(diff_df["faster_rate_per_decade"], diff_df["slower_rate_per_decade"])],
                "Difference (°C/decade)": diff_df["difference_per_decade"].map(lambda v: f"{v:+.3f}"),
                "95% CI (HAC)": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(diff_df["diff_ci_lower_per_decade"], diff_df["diff_ci_upper_per_decade"])],
                "OLS-HAC p": diff_df["ols_hac_p"].map(lambda v: f"{v:.2e}"),
                "Hamed-Rao p": diff_df["hr_p"].map(lambda v: f"{v:.2e}"),
                "Holm p": diff_df["holm_p"].map(lambda v: f"{v:.2e}"),
                "Different speed?": diff_df["is_significant"].map({True: "YES", False: "NO"}),
            }), use_container_width=True, hide_index=True)
    else:
        st.warning("Regional datasets are not available yet.")


# ==========================================
# PAGE 6: ANOMALY DETECTION ENGINE
# ==========================================
def page_anomalies() -> None:
    page_header(
        "⚠️",
        f"Climate Anomalies & Drought — {active_location}",
        "Departures from WMO 1981–2010 normal, persistent heat/cold spells, multi-scalar SPI drought analysis, and compound extremes.",
    )

    tab_t, tab_p, tab_c, tab_r = st.tabs([
        "🌡️ Temperature Anomalies",
        "🌧️ Rainfall & SPI Drought",
        "🔥 Compound Extremes",
        "🔗 Cross-Variable & Regional",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: TEMPERATURE ANOMALIES & PERSISTENT SPELLS
    # --------------------------------------------------------------------------
    with tab_t:
        st.markdown("### 🌡️ Temperature Departures from WMO 1981–2010 Normal")
        avail_vars = [v for v in ["T2M", "T2M_MAX", "T2M_MIN"] if v in df_current["variable"].unique()]
        if not avail_vars:
            avail_vars = sorted(df_current["variable"].unique().tolist())

        var_col1, var_col2 = st.columns([2, 3])
        with var_col1:
            sel_var = st.selectbox(
                "Temperature Variable",
                avail_vars,
                index=0,
                key="anom_temp_var_select",
            )
        with var_col2:
            if sel_var == "T2M":
                st.caption("ℹ️ **T2M**: Monthly mean 2-meter air temperature departure ($A_t = T_t - \\mu_m$).")
            elif sel_var == "T2M_MAX":
                st.caption("ℹ️ **T2M_MAX**: Monthly **absolute hottest temperature** reached within each month.")
            elif sel_var == "T2M_MIN":
                st.caption("ℹ️ **T2M_MIN**: Monthly **absolute coldest temperature** reached within each month.")

        t_anom_df, t_summary = compute_temperature_anomalies(
            df_current,
            variable=sel_var,
            baseline_start_year=1981,
            baseline_end_year=2010,
        )

        # KPI row
        k1, k2, k3, k4 = st.container(key="temp_anom_kpis").columns(4)
        kpi_card(k1, "Mean Anomaly", f"{t_summary['mean_anomaly']:+.2f} °C", "Full 1981–2023 record", "blue")
        kpi_card(k2, "Hottest Month Anomaly", f"{t_summary['max_positive_anomaly']:+.2f} °C", f"+{t_anom_df['z_score'].max():.1f}σ above normal", "red")
        kpi_card(k3, "Coldest Month Anomaly", f"{t_summary['max_negative_anomaly']:+.2f} °C", f"{t_anom_df['z_score'].min():.1f}σ below normal", "blue")
        warm_3s = t_summary.get("extreme_warm_count_3sigma", 0)
        kpi_card(k4, "Extreme Events (≥ +3σ)", f"{warm_3s} months", "Probability < 0.15% under baseline normal", "orange" if warm_3s > 0 else "slate")

        # Standardized Z-Score Bar Chart
        fig_z = plot_standardized_anomalies_zscore(
            anomaly_df=t_anom_df,
            variable_name=VARIABLE_METADATA.get(sel_var, {}).get("name", sel_var),
            location_name=active_location,
        )
        st.plotly_chart(fig_z, use_container_width=True)

        # Persistent Spells Expander
        with st.expander("🔥 Persistent Abnormal Periods (Warm Spells & Cold Spells)", expanded=False):
            st.markdown(
                """
                **Persistent Spell Detection**: Identifies runs of consecutive months with persistent standardized departures ($|Z| \\ge 1.0\\sigma$).
                Long warm spells indicate multi-month synoptic warm anomalies or persistent positive land-atmosphere feedback.
                """
            )
            col_sp1, col_sp2 = st.columns(2)
            with col_sp1:
                st.markdown("#### 🔥 Longest Warm Spells (Z ≥ +1.0σ, duration ≥ 3 months)")
                warm_spells = detect_persistent_anomalies(t_anom_df, z_threshold=1.0, min_duration=3, spell_type="warm")
                if not warm_spells.empty:
                    st.dataframe(
                        warm_spells[["start_date", "end_date", "duration_months", "peak_z_score", "mean_z_score"]].rename(
                            columns={
                                "start_date": "Start",
                                "end_date": "End",
                                "duration_months": "Duration (mo)",
                                "peak_z_score": "Peak Z (σ)",
                                "mean_z_score": "Mean Z (σ)",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No warm spells of duration ≥ 3 months detected.")

            with col_sp2:
                st.markdown("#### ❄️ Longest Cold Spells (Z ≤ -1.0σ, duration ≥ 3 months)")
                cold_spells = detect_persistent_anomalies(t_anom_df, z_threshold=1.0, min_duration=3, spell_type="cold")
                if not cold_spells.empty:
                    st.dataframe(
                        cold_spells[["start_date", "end_date", "duration_months", "peak_z_score", "mean_z_score"]].rename(
                            columns={
                                "start_date": "Start",
                                "end_date": "End",
                                "duration_months": "Duration (mo)",
                                "peak_z_score": "Peak Z (σ)",
                                "mean_z_score": "Mean Z (σ)",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No cold spells of duration ≥ 3 months detected.")

        # Top Ranked Extremes
        with st.expander("🏆 Historical Monthly Extremes Ranking", expanded=False):
            ranks = get_top_ranked_extremes(t_anom_df, top_n=5)
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("#### 🔥 Top 5 Warmest Departures")
                st.dataframe(ranks["hottest_anomalies"], use_container_width=True, hide_index=True)
            with rc2:
                st.markdown("#### ❄️ Top 5 Coldest Departures")
                st.dataframe(ranks["coldest_anomalies"], use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: PRECIPITATION ANOMALIES & SPI DROUGHT ENGINE
    # --------------------------------------------------------------------------
    with tab_p:
        st.markdown("### 🌧️ Precipitation Anomalies & Standardized Precipitation Index (SPI)")
        st.info(
            "📐 **Statistical Aridity Caveat**: Ordinary Gaussian Z-scores ($Z = (P - \\mu)/\\sigma$) are **statistically invalid** "
            "for zero-inflated precipitation. When precipitation is bounded at zero and the baseline mean is small, negative Z-scores "
            "are bounded above $-1.0\\sigma$, making severe drought mathematically impossible to diagnose under a normal distribution. "
            "We therefore implement the WMO-standard **Standardized Precipitation Index (SPI)** using 2-parameter Gamma fitting "
            "with zero-probability adjustment."
        )

        has_precip = "PRECTOTCORR" in df_current["variable"].unique()
        if has_precip:
            p_anom_df, p_summary = compute_precipitation_anomalies(
                df_current,
                baseline_start_year=1981,
                baseline_end_year=2010,
            )

            kp1, kp2, kp3, kp4 = st.container(key="precip_anom_kpis").columns(4)
            kpi_card(kp1, "Baseline Annual Rain", f"{p_summary['mean_annual_baseline_mm']:.1f} mm", "WMO 1981–2010 Normal", "blue")
            kpi_card(kp2, "Dry Months", f"{p_summary['zero_fraction']*100:.1f}%", f"{p_summary['zero_months_count']} / {p_summary['total_months']} months with 0 mm", "orange" if p_summary['is_zero_inflated'] else "slate")
            ann_p_df = p_summary["annual_anomalies"]
            latest_pn = ann_p_df["percent_of_normal"].iloc[-1] if not ann_p_df.empty else 100.0
            latest_yr = int(ann_p_df["year"].iloc[-1]) if not ann_p_df.empty else 2023
            kpi_card(kp3, f"{latest_yr} % of Normal", f"{latest_pn:.0f}%", f"{ann_p_df['annual_total_mm'].iloc[-1]:.1f} mm total", "green" if latest_pn >= 100 else "orange")
            wet_season_df = p_summary.get("wet_season_anomalies", pd.DataFrame())
            kpi_card(kp4, "Wet Seasons Tracked", f"{len(wet_season_df)}", "Oct → Mar Hydrological Window", "blue")

            if p_summary["is_zero_inflated"]:
                st.warning(
                    f"⚠️ **Hyper-Arid Regime Warning**: {active_location} has **{p_summary['zero_fraction']*100:.1f}% completely dry months**. "
                    "When zero-precipitation frequency exceeds 50%, SPI is mathematically bounded below by a positive or near-zero threshold, "
                    "preventing reliable statistical estimation of drought. In hyper-arid months, SPI is flagged as undefined."
                )

            if (p_anom_df["percent_of_normal_reliable"] == False).any():
                unreliable_months = p_anom_df[p_anom_df["percent_of_normal_reliable"] == False]["month_name"].unique().tolist()
                st.caption(
                    f"⚠️ **Percent of Normal Stability Alert**: In dry months ({', '.join(unreliable_months)}) where "
                    f"climatological normal rainfall is $< 1.0\\text{{ mm}}$, Percent of Normal fluctuates erratically on trace showers "
                    f"and is flagged as **Low Reliability**. Use absolute departures (mm) for physical evaluation."
                )

            # SPI Scale selector
            spi_scale_choice = st.radio(
                "SPI Accumulation Scale",
                [3, 6, 12],
                index=2,
                format_func=lambda s: {
                    3: "SPI-3 (3-Month: Seasonal & Agricultural Moisture)",
                    6: "SPI-6 (6-Month: Medium Hydrological Storage)",
                    12: "SPI-12 (12-Month: Long-Term Annual Water Balance)",
                }[s],
                horizontal=True,
                key="spi_scale_radio",
            )

            spi_df = compute_spi(p_anom_df, scale=spi_scale_choice, baseline_start_year=1981, baseline_end_year=2010)

            if (spi_df.get(f"SPI_{spi_scale_choice}_drought_limited", pd.Series(dtype=bool)) == True).any():
                st.info(
                    f"ℹ️ **Drought Detection Truncation**: For accumulation windows with zero-rain baseline frequency "
                    f"$q > \\Phi(-1.0) \\approx 0.1587$, the theoretical minimum SPI satisfies $\\Phi^{{-1}}(q) > -1.0$. Even under complete absence of precipitation "
                    f"(0.0 mm), SPI cannot reach the standard moderate drought threshold ($\\le -1.0$). In such months, drought detection "
                    f"is structurally truncated by hyper-aridity."
                )

            with st.expander("ℹ️ Scientific Guardrails on SPI in Arid Climates: Undefined SPI vs Limited Drought Detection", expanded=False):
                st.markdown(
                    """
                    In arid and hyper-arid climates, zero-precipitation periods occur frequently, leading to structural mathematical boundaries in the SPI:

                    1. **Theoretical Lower Bound ($\\\\text{SPI}_{\\\\min} = \\\\Phi^{-1}(q)$)**:
                       When precipitation accumulation is zero ($x = 0$), the cumulative probability is $H(0) = q$, where $q$ is the empirical probability of zero precipitation during the 1981–2010 baseline. The transformed SPI value is therefore strictly bounded below by $\\\\text{SPI}_{\\\\min} = \\\\Phi^{-1}(q)$.

                    2. **Limited Drought Detection ($0.1587 < q < 0.50$)**:
                       Because moderate meteorological drought is defined as $\\\\text{SPI} \\\\le -1.0$, whenever $q > \\\\Phi(-1.0) \\\\approx 0.1587$, the theoretical lower bound satisfies $\\\\Phi^{-1}(q) > -1.0$. In these windows, even a total absence of rainfall ($0.0\\\\text{ mm}$) cannot register as moderate drought. The index is mathematically defined, but its ability to detect drought is structurally truncated.

                    3. **Undefined SPI ($q \\\\ge 0.50$ or $n_{\\\\text{pos}} < 5$)**:
                       When zero-rain frequency $q \\\\ge 0.50$, the cumulative probability for zero rainfall satisfies $H(0) \\\\ge 0.50$, mapping zero rainfall to a non-negative (normal or moist) SPI score ($\\\\Phi^{-1}(q) \\\\ge 0.0$). Reporting positive moisture indices during complete absence of rain is physically misleading. Following WMO guidelines and Wu et al. (2007), SPI is designated as **Undefined / NaN** under these hyper-arid conditions.
                    """
                )

            # Plot SPI
            fig_spi = plot_spi_series(spi_df, scale=spi_scale_choice, location_name=active_location)
            st.plotly_chart(fig_spi, use_container_width=True)

            # Annual Accumulation Table
            with st.expander("📊 Annual Accumulations & Percent of Normal Table", expanded=False):
                st.dataframe(
                    ann_p_df.rename(
                        columns={
                            "year": "Year",
                            "annual_total_mm": "Accumulated Rain (mm)",
                            "annual_anomaly_mm": "Departure (mm)",
                            "percent_of_normal": "% of 1981–2010 Normal",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("Precipitation records (PRECTOTCORR) not found in the current dataset.")

    # --------------------------------------------------------------------------
    # TAB 3: COMPOUND CLIMATE EXTREMES
    # --------------------------------------------------------------------------
    with tab_c:
        st.markdown("### 🔥 Compound Climate Extremes & Decadal Shifts")
        st.caption(
            "Compound Hot & Dry months intensify evaporative stress and drought, while Compound Hot & Wet months indicate "
            "tropical pluvial surges. Here we evaluate how the joint distribution has shifted across decades."
        )

        if has_precip:
            t_anom_t2m, _ = compute_temperature_anomalies(df_current, variable="T2M")
            compound_df, compound_sum = analyze_compound_anomalies(t_anom_t2m, p_anom_df)

            c1, c2 = st.columns([3, 2])
            with c1:
                fig_comp = plot_compound_scatter(compound_df, location_name=active_location)
                st.plotly_chart(fig_comp, use_container_width=True)
            with c2:
                st.markdown("#### 📈 Decadal Compound Event Shifts")
                dec_comp_table = compound_sum["decadal_shift"]
                st.dataframe(
                    dec_comp_table[[
                        "decade_label", "total_months",
                        "hot_dry_count", "hot_dry_pct",
                        "hot_wet_count", "hot_wet_pct",
                        "cold_dry_count", "cold_dry_pct",
                        "cold_wet_count", "cold_wet_pct"
                    ]].rename(
                        columns={
                            "decade_label": "Decade",
                            "total_months": "Months",
                            "hot_dry_count": "Hot & Dry (n)",
                            "hot_dry_pct": "Hot & Dry (%)",
                            "hot_wet_count": "Hot & Wet (n)",
                            "hot_wet_pct": "Hot & Wet (%)",
                            "cold_dry_count": "Cold & Dry (n)",
                            "cold_dry_pct": "Cold & Dry (%)",
                            "cold_wet_count": "Cold & Wet (n)",
                            "cold_wet_pct": "Cold & Wet (%)",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.info(
                    f"Over the full record: **{compound_sum['total_hot_dry']} months** Compound Hot & Dry, "
                    f"**{compound_sum['total_hot_wet']} months** Compound Hot & Wet, "
                    f"**{compound_sum['total_cold_dry']} months** Compound Cold & Dry, "
                    f"and **{compound_sum['total_cold_wet']} months** Compound Cold & Wet."
                )
                st.caption(
                    "⚠️ **Decade Time Coverage**: The 1980s covers 9 elapsed years (1981–1989, 108 calendar months) due to "
                    "record start; the 2020s covers 4 elapsed years (2020–2023, 48 calendar months). Intermediate decades "
                    "(1990s, 2000s, 2010s) each contain 10 complete years (120 calendar months). Decadal percentages are "
                    "normalized to total elapsed months in each period."
                )

            # Decadal shift in general temperature extremes
            st.markdown("#### 📊 Decadal Shift in Extreme Temperature Months")
            decadal_t = decadal_extreme_frequency(t_anom_t2m, z_threshold=1.5)
            fig_dec_t = plot_decadal_shift_bars(decadal_t)
            st.plotly_chart(fig_dec_t, use_container_width=True)
        else:
            st.warning("Precipitation records required for compound extremes analysis.")

    # --------------------------------------------------------------------------
    # TAB 4: CROSS-VARIABLE & REGIONAL CORRELATION
    # --------------------------------------------------------------------------
    with tab_r:
        st.markdown("### 🔗 Cross-Variable Anomaly Association & Regional Concordance")

        if has_precip:
            t_anom_t2m, _ = compute_temperature_anomalies(df_current, variable="T2M")
            cross_corr = cross_variable_anomaly_correlation(t_anom_t2m, p_anom_df, max_lag=3)

            cor1, cor2 = st.columns(2)
            with cor1:
                st.markdown("#### 🔬 Temperature vs. Precipitation Correlation")
                st.write(
                    f"- **Overall Pearson $r$**: `{cross_corr['pearson_r']:+.3f}` (nominal $p = {cross_corr['pearson_p_nominal']:.4f}$)\n"
                    f"- **Effective Sample Size ($N_{{\\text{{eff}}}}$)**: `{cross_corr['effective_sample_size']:.1f}` (accounting for serial persistence)\n"
                    f"- **Autocorrelation-Adjusted $p$-value**: `{cross_corr['pearson_p_adjusted']:.4f}`\n"
                    f"- **Spearman Rank $\\rho_s$**: `{cross_corr['spearman_rho']:+.3f}` ($p = {cross_corr['spearman_p']:.4f}$)\n"
                    f"- **Winter (Rainy Season, Oct–Mar) $r$**: `{cross_corr['winter_pearson_r']:+.3f}` ($p = {cross_corr['winter_p']:.4f}$)\n"
                    f"- **Summer (Dry Season, Apr–Sep) $r$**: `{cross_corr['summer_pearson_r']:+.3f}` ($p = {cross_corr['summer_p']:.4f}$)\n"
                )
                st.caption(
                    "⚠️ **Physical Interpretation**: In northern and coastal Egypt, rainy months show significant negative correlation "
                    "with temperature anomalies. This is consistent with cloud cover reflection, winter Mediterranean depression "
                    "cold-air advection, and potential land-atmosphere feedbacks; however, statistical correlation does not establish "
                    "physical causation. In hyper-arid Upper Egypt, precipitation is decoupled from temperature anomalies."
                )
            with cor2:
                fig_lag = plot_lagged_cross_correlation(cross_corr, location_name=active_location)
                st.plotly_chart(fig_lag, use_container_width=True)

        # Cross-Regional Anomaly Comparison
        st.markdown("---")
        st.markdown("### 🗺️ Cross-Regional Anomaly Concordance across Egypt")
        reg_anom_dict = {}
        for reg_name, path in PRESET_FILES.items():
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        df_reg, _ = auto_load_dataset(f.read(), default_location=reg_name)
                    anom_reg, _ = compute_temperature_anomalies(df_reg, variable="T2M")
                    reg_anom_dict[reg_name] = anom_reg
                except Exception:
                    pass

        if len(reg_anom_dict) >= 2:
            reg_cmp = compare_regional_anomalies(reg_anom_dict)
            st.write(
                f"Comparing **{len(reg_anom_dict)} Egyptian regions** over {reg_cmp['total_overlapping_months']} overlapping months:\n"
                f"- **Spatial Sign Concordance Rate**: **`{reg_cmp['spatial_concordance_rate']*100:.1f}%`** "
                f"({reg_cmp['concordant_months']} months shared the exact same anomaly sign simultaneously across all regions).\n"
                "- Demonstrates that broad synoptic airmass shifts drive coherent temperature departures across Egypt."
            )
            st.dataframe(pd.DataFrame(reg_cmp["pearson_correlation_matrix"]), use_container_width=True)


# ==========================================
# PAGE: CLIMATE ANOMALY FORECASTING & VALIDATION
# ==========================================
# Observations NASA POWER released AFTER the bundled 1981–2023 record. They are never used to fit
# or select a model — only to score the forecast afterwards (true out-of-sample verification).
VERIFY_FILES: Dict[str, str] = {
    "Greater Cairo": os.path.join("data", "verification", "nasa_power_cairo_2024.csv"),
    "Alexandria": os.path.join("data", "verification", "nasa_power_alexandria_2024.csv"),
    "Aswan": os.path.join("data", "verification", "nasa_power_aswan_2024.csv"),
}


@st.cache_data(show_spinner=False)
def load_verification_observations(region: str) -> pd.DataFrame:
    path = VERIFY_FILES.get(region)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        return auto_load_dataset(f.read(), default_location=region)[0]


@st.cache_data(show_spinner=False, max_entries=48)
def cached_forecast(var_df: pd.DataFrame, value_col: str, model_key: str) -> Dict[str, Any]:
    """Backtest + 12-month forecast are deterministic (seeded), so cache them per input."""
    return generate_12month_forecast(var_df, variable=value_col, model_type=model_key, horizon=12, n_splits_backtest=5)


def page_forecasting() -> None:
    page_header(
        "🔮",
        f"Climate Anomaly Forecasting & Chronological Validation — {active_location}",
        "12-month forward predictive projections, probabilistic prediction intervals (80% & 95%), and out-of-sample backtesting.",
    )

    if df_current.empty:
        st.warning("No data loaded. Please select a preset region or upload data in the sidebar.")
        return

    # User controls row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 3])
    with ctrl_col1:
        avail_vars = [v for v in ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"] if v in df_current["variable"].unique()]
        if not avail_vars:
            avail_vars = sorted(df_current["variable"].unique().tolist())
        target_var = st.selectbox(
            "Predictive Variable",
            avail_vars,
            index=0,
            key="fc_var_select",
        )
    with ctrl_col2:
        model_choice = st.selectbox(
            "Forecasting Model",
            [
                "Tournament Winner (Best Out-of-Sample MAE)",
                "Holt-Winters ETS (Additive Trend & Season)",
                "SARIMA(1,0,0)(1,1,0,12)",
                "Seasonal Naive Baseline",
            ],
            index=0,
            key="fc_model_select",
        )
        model_key_map = {
            "Tournament Winner (Best Out-of-Sample MAE)": "best",
            "Holt-Winters ETS (Additive Trend & Season)": "ets",
            "SARIMA(1,0,0)(1,1,0,12)": "sarima",
            "Seasonal Naive Baseline": "naive",
        }
        chosen_model_key = model_key_map[model_choice]

    with ctrl_col3:
        if target_var == "T2M":
            st.caption("ℹ️ **T2M**: Monthly mean 2m temperature. Deseasonalized departures exhibit Gaussian distribution.")
        elif target_var == "T2M_MAX":
            st.caption("ℹ️ **T2M_MAX**: Monthly maximum 2m temperature peak (TXx proxy).")
        elif target_var == "T2M_MIN":
            st.caption("ℹ️ **T2M_MIN**: Monthly minimum 2m temperature low (TNn proxy).")
        elif target_var == "PRECTOTCORR":
            st.caption("ℹ️ **PRECTOTCORR**: Monthly precipitation accumulation. Lower interval bound clamped at 0.0.")

    # Slice variable dataframe
    var_df = df_current[df_current["variable"] == target_var].copy()
    if var_df.empty:
        st.error(f"No records found for variable {target_var}.")
        return

    units = var_df["units"].iloc[0] if "units" in var_df.columns else "units"

    # Run forecasting and backtesting engine
    with st.spinner(f"Computing chronological backtest and 12-month {target_var} projections..."):
        fc_res = cached_forecast(
            var_df,
            "value" if "value" in var_df.columns else target_var,
            chosen_model_key,
        )

    fc_df = fc_res["forecast_df"]
    hist_df = fc_res["historical_df"]
    backtest = fc_res["backtest_results"]
    kpis = fc_res["summary_kpis"]

    # Hyper-Arid Alert Banner if applicable
    if fc_res["is_hyper_arid"]:
        st.warning(
            f"🏜️ **Hyper-Arid Regime Advisory ({active_location})**: Over 80% of historical months record zero precipitation "
            f"(historical mean = {var_df['value'].mean():.2f} {units}). Statistical linear/state-space models project baseline "
            "trace background rainfall with wide confidence intervals. Rare convective storm spikes are episodic and cannot be "
            "extrapolated deterministically."
        )

    # 4 TABS LAYOUT
    tab1, tab_rc, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Overview",
        "✅ 2024 Reality Check",
        "📈 Projections & Intervals",
        "⚠️ Future Anomaly Classification",
        "🧪 Chronological Validation & Backtesting",
    ])

    # --------------------------------------------------------------------------
    # TAB: REALITY CHECK AGAINST NEWLY RELEASED NASA OBSERVATIONS
    # --------------------------------------------------------------------------
    with tab_rc:
        st.markdown("### ✅ Did the forecast come true?")
        obs_all = load_verification_observations(active_location) if active_location in VERIFY_FILES else pd.DataFrame()
        obs_var = obs_all[obs_all["variable"] == target_var] if not obs_all.empty else obs_all
        ver = verify_forecast_against_observations(fc_res, obs_var) if not obs_var.empty else {"available": False}
        if not ver.get("available"):
            st.info(
                "A reality check needs NASA observations released after the forecast was made. They are bundled for "
                "the three preset regions (Jan–Dec 2024). Pick Greater Cairo, Alexandria or Aswan in the top bar."
            )
        else:
            r1, r2, r3, r4 = st.container(key="rc_kpis").columns(4)
            beat = ver["mae"] < ver["naive_mae"]
            kpi_card(r1, "Average miss", f"{ver['mae']:.2f} {units}",
                     f"'Same as last year' missed by {ver['naive_mae']:.2f}", "green" if beat else "slate")
            kpi_card(r2, "Inside 95% band", f"{ver['hit_95']:.0f}%",
                     f"{int(ver['table']['in_95'].sum())} of {ver['n_months']} real months", "blue")
            kpi_card(r3, "Predicted anomaly", f"{ver['predicted_mean_anomaly']:+.2f} {units}",
                     "vs 1981–2010 normal (forecast)", "orange")
            kpi_card(r4, "Observed anomaly", f"{ver['observed_mean_anomaly']:+.2f} {units}",
                     f"what NASA recorded, {ver['period']}", "red")
            render_plain_words(verification_story(ver, units, active_location, fc_res["is_precipitation"]))
            st.plotly_chart(
                plot_forecast_verification(ver["table"], target_var, units, active_location),
                use_container_width=True,
            )
            st.caption(
                "🔒 **No peeking.** The models are fitted and selected only on 1981–2023. The 2024 values were downloaded "
                "afterwards from the NASA POWER API (same MERRA-2 grid cell) and are used only for this score. "
                "Diamonds: green = inside the 95% band, red = outside. NASA may revise recent months slightly."
            )
            with st.expander("Month-by-month table"):
                vt = ver["table"][["date", "forecast", "ci_95_lower", "ci_95_upper", "observed", "error", "in_95"]].copy()
                vt["date"] = vt["date"].dt.strftime("%b %Y")
                st.dataframe(vt.rename(columns={
                    "date": "Month", "forecast": f"Forecast ({units})", "ci_95_lower": "95% low",
                    "ci_95_upper": "95% high", "observed": f"Observed ({units})", "error": "Observed − forecast",
                    "in_95": "Inside 95% band?",
                }), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("### 📊 Executive Forecast Summary & Predictive Trajectory")

        # 4 KPI Cards
        k1, k2, k3, k4 = st.container(key="fc_kpis").columns(4)
        winner_name = kpis["best_model_name"]
        winner_mase = kpis["best_model_mase"]
        kpi_card(k1, "Validation Winner", winner_name, f"MASE = {winner_mase:.3f} (< 1.0 beats naive)", "blue")

        mean_anom = kpis["mean_expected_anomaly"]
        anom_color = "red" if mean_anom > 0 else "blue"
        kpi_card(k2, "12-Mo Expected Anomaly", f"{mean_anom:+.2f} {units}", "Relative to 1981–2010 normal", anom_color)

        peak_val = kpis["peak_forecast_value"]
        peak_date = kpis["peak_forecast_date"]
        peak_z = kpis["peak_forecast_z"]
        kpi_card(k3, "Peak Forecast Month", f"{peak_val:.1f} {units}", f"{peak_date} ({peak_z:+.1f}σ)", "orange")

        ci_spread = kpis["mean_ci95_spread"]
        kpi_card(k4, "Uncertainty Spread", f"±{ci_spread:.2f} {units}", "Mean 95% Confidence Half-Width", "slate")

        render_plain_words(forecast_story(fc_res, target_var, units, active_location))

        # Primary Forecast Figure
        fig_fc = plot_forecast_with_intervals(
            historical_df=hist_df,
            forecast_df=fc_df,
            variable_name=target_var,
            units=units,
            location_name=active_location,
            show_history_months=60,
            model_name=fc_res["selected_model_name"],
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.caption(
            "📌 **Reading the Projection**: Gold line represents model point expectation. Dual shaded yellow envelopes "
            "indicate 80% (inner) and 95% (outer) probabilistic prediction intervals. Dotted white line denotes the "
            "WMO 1981–2010 monthly climatological normal. Vertical dashed red line marks the transition from historical "
            "observations to future projections."
        )

    # --------------------------------------------------------------------------
    # TAB 2: PROJECTIONS & INTERVALS TABLE
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("### 📈 Detailed 12-Month Projected Trajectory & Uncertainty Bands")

        display_cols = [
            "date", "month_name", "forecast", "ci_80_lower", "ci_80_upper",
            "ci_95_lower", "ci_95_upper", "clim_mean", "expected_anomaly"
        ]
        if fc_res["is_precipitation"]:
            display_cols.append("pct_of_normal")

        rename_map = {
            "date": "Date",
            "month_name": "Month",
            "forecast": f"Point Forecast ({units})",
            "ci_80_lower": f"80% CI Low ({units})",
            "ci_80_upper": f"80% CI High ({units})",
            "ci_95_lower": f"95% CI Low ({units})",
            "ci_95_upper": f"95% CI High ({units})",
            "clim_mean": f"WMO Normal ({units})",
            "expected_anomaly": f"Expected Departure ({units})",
            "pct_of_normal": "Percent of Normal (%)",
        }

        table_df = fc_df[display_cols].copy()
        table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
        table_df = table_df.rename(columns=rename_map)
        st.dataframe(table_df, use_container_width=True)

        col_dl, col_exp = st.columns([1, 3])
        with col_dl:
            csv_data = fc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Projections CSV",
                data=csv_data,
                file_name=f"earthtrend_{active_location.lower().replace(' ', '_')}_{target_var.lower()}_forecast_2024.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("🔬 Model Specification & Fitted Parameters", expanded=False):
            st.write(f"- **Selected Architecture**: `{fc_res['selected_model_name']}`")
            st.write(f"- **Model Key**: `{fc_res['selected_model_key']}`")
            st.json(fc_res["model_params"])

    # --------------------------------------------------------------------------
    # TAB 3: FUTURE ANOMALY CLASSIFICATION
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("### ⚠️ Future Anomaly Classification & IPCC/WMO Severity Mapping")

        # Future Anomaly Departure Bars
        fig_anom = plot_future_anomaly_bars(
            future_df=fc_df,
            variable_name=target_var,
            units=units,
            location_name=active_location,
        )
        st.plotly_chart(fig_anom, use_container_width=True)

        st.markdown("#### 📋 Month-by-Month Projected Anomaly Severity")
        anom_display_cols = [
            "month_name", "forecast", "clim_mean", "expected_anomaly",
            "expected_z_score", "expected_z_ci95_lower", "expected_z_ci95_upper", "severity_bracket"
        ]
        if not fc_res["is_precipitation"]:
            anom_display_cols += ["prob_above_normal", "prob_z_ge_plus1", "prob_z_le_minus1"]
        anom_table = fc_df[anom_display_cols].copy()
        anom_table = anom_table.rename(columns={
            "prob_above_normal": "P(above normal) %",
            "prob_z_ge_plus1": "P(unusually warm, Z≥+1) %",
            "prob_z_le_minus1": "P(unusually cold, Z≤−1) %",
            "month_name": "Month",
            "forecast": f"Forecast ({units})",
            "clim_mean": f"Normal ({units})",
            "expected_anomaly": f"Departure ({units})",
            "expected_z_score": "Z-Score (σ)",
            "expected_z_ci95_lower": "Z 95% Low (σ)",
            "expected_z_ci95_upper": "Z 95% High (σ)",
            "severity_bracket": "Severity Classification",
        })
        st.dataframe(anom_table, use_container_width=True)

        if not fc_res["is_precipitation"]:
            st.caption(
                "🎲 **Expected value ≠ risk.** The probability columns turn each forecast into an actual chance, "
                "using a normal predictive distribution whose spread comes from the 95% prediction interval "
                "(σ = width / 3.92). A month forecast exactly on the +1σ line still has only ~50% chance of exceeding it."
            )

        st.info(
            "💡 **Scientific Distinction — Observed vs. Expected Anomalies**:\n\n"
            "• **Historical Anomalies ($1981–2023$)**: Deterministic observational ground truth derived from NASA MERRA-2 "
            "reanalysis observations relative to the 1981–2010 baseline.\n"
            "• **Forecasted Anomalies ($2024$)**: Model-derived probabilistic expectations subject to structural assumptions, "
            "parameter estimation uncertainty, and stochastic innovations. An expected +1.2σ anomaly communicates that the "
            "expected state lies in the warm upper quartile, but must be interpreted in conjunction with the 95% prediction interval."
        )

    # --------------------------------------------------------------------------
    # TAB 4: CHRONOLOGICAL VALIDATION & BACKTESTING
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("### 🧪 Chronological Expanding-Window Cross-Validation Tournament")
        st.markdown(
            "To prevent future data leakage, models are evaluated using expanding chronological windows across **5 historical origins** "
            "(2018–2022). Each model is trained strictly on observations prior to each origin and evaluated against the subsequent 12 unseen months."
        )

        # Backtest comparison chart
        fig_cmp = plot_model_backtest_comparison(backtest, location_name=active_location)
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Performance metrics table
        st.markdown("#### 🏆 Out-of-Sample Tournament Scoreboard")
        score_rows = []
        for m_k, m_v in backtest["summary_metrics"].items():
            score_rows.append({
                "Model Key": m_k,
                "Model Architecture": m_v["model_name"],
                "Average Test MAE": f"{m_v['mean_mae']:.3f} {units}",
                "Average Test RMSE": f"{m_v['mean_rmse']:.3f} {units}",
                "MASE (vs Seasonal Naive)": f"{m_v['mase']:.3f}",
                "80% band hit-rate": f"{m_v['coverage_80']:.0f}% · {interval_coverage_verdict(m_v['coverage_80'], 80, m_v['n_test_months'])}",
                "95% band hit-rate": f"{m_v['coverage_95']:.0f}% · {interval_coverage_verdict(m_v['coverage_95'], 95, m_v['n_test_months'])}",
                "Tournament Status": "🏅 Best Model (Winner)" if m_k == backtest["tournament_winner_key"] else "Evaluated Candidate",
            })
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True)
        st.caption(
            "🎯 **Are the uncertainty bands honest?** Hit-rate = share of the held-out months that actually fell inside "
            "each model's band. A well-calibrated 95% band should catch ≈95% of them; verdicts use an exact binomial test (α = 0.05)."
        )

        # Split-by-Split Details Expander
        with st.expander("🔍 Detailed Multi-Origin Backtest Results (5 Splits)", expanded=False):
            split_records = []
            for sp in backtest["split_details"]:
                split_records.append({
                    "Split Origin": f"Origin #{sp['split_idx']} (Train N={sp['train_size']})",
                    "Naive MAE": f"{sp['models']['naive']['metrics']['mae']:.3f}",
                    "ETS MAE": f"{sp['models']['ets']['metrics']['mae']:.3f}",
                    "SARIMA MAE": f"{sp['models']['sarima']['metrics']['mae']:.3f}",
                })
            st.dataframe(pd.DataFrame(split_records), use_container_width=True)

        # Residual diagnostics
        st.markdown("#### 🔬 Residual Diagnostics Check")
        fig_res = plot_forecast_residuals_diagnostic(
            residuals=fc_res["residuals"],
            dates=hist_df["date"],
            model_name=fc_res["selected_model_name"],
            location_name=active_location,
        )
        st.plotly_chart(fig_res, use_container_width=True)


# ==========================================
# PAGE 7: METHODOLOGY & SOURCES
# ==========================================
def page_methodology() -> None:
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


# ==========================================
# SHARED HELPERS FOR THE DASHBOARD LAYOUT
# ==========================================
PRESET_FILES: Dict[str, str] = {
    "Greater Cairo": os.path.join("data", "raw", "nasa_power_cairo_1981_2023.csv"),
    "Alexandria": os.path.join("data", "raw", "nasa_power_alexandria_1981_2023.csv"),
    "Aswan": os.path.join("data", "raw", "nasa_power_aswan_1981_2023.csv"),
}


def preset_frames() -> Dict[str, pd.DataFrame]:
    """The three bundled NASA POWER regions (disk reads are cached)."""
    return {name: load_bundled_dataset(path, name)[0] for name, path in PRESET_FILES.items() if os.path.exists(path)}


def analysis_region() -> str:
    """Region used by the preset-based pages: the active region if it is a preset, else Greater Cairo."""
    loc = st.session_state.get("active_location", "Greater Cairo")
    return loc if loc in PRESET_FILES else "Greater Cairo"


@st.cache_data(show_spinner=False)
def cached_rainfall(region: str) -> Dict[str, Any]:
    return analyze_rainfall(load_bundled_dataset(PRESET_FILES[region], region)[0], region)


@st.cache_data(show_spinner=False)
def cached_rainfall_table() -> pd.DataFrame:
    return compare_regional_rainfall(preset_frames())


@st.cache_data(show_spinner=False)
def cached_extremes(region: str) -> Dict[str, Any]:
    return analyze_temperature_extremes(load_bundled_dataset(PRESET_FILES[region], region)[0], region)


@st.cache_data(show_spinner=False)
def cached_extremes_table() -> pd.DataFrame:
    return compare_regional_extremes(preset_frames())


@st.cache_data(show_spinner=False)
def cached_temperature_summary(region: str) -> Dict[str, Any]:
    """Annual mean T2M trend for the Home hero."""
    df = load_bundled_dataset(PRESET_FILES[region], region)[0]
    ann = aggregate_to_annual(df[df["variable"] == "T2M"].dropna(subset=["value"]))
    y, x = ann["value"].values, ann["year"].values
    mk = mann_kendall_test(y)
    sen = compute_sens_slope(y, x)
    ols = compute_linear_regression(y, x)
    acf = compute_autocorrelation(y, x)
    answers = synthesize_scientific_answers(
        mk, sen, ols, acf, "2-meter air temperature (T2M)", "°C", region, int(x.min()), int(x.max()), "annual"
    )
    return {"annual": ann[["year", "value"]].copy(), "mk": mk, "sen": sen, "answers": answers,
            "start": int(x.min()), "end": int(x.max())}


@st.cache_data(show_spinner=False)
def cached_warming_diff(variable: str) -> pd.DataFrame:
    return compare_regional_warming_rates(preset_frames(), variable=variable)


STORY_TAGS = {
    "observed": ("📡", "Observed"),
    "statistical": ("📐", "Tested"),
    "projected": ("🔮", "Projected"),
    "caution": ("⚠️", "Keep in mind"),
}


def render_plain_words(story, title: str = "In plain words") -> None:
    """Checkpoint 8: plain-language summary built only from verified outputs (src/narrative.py)."""
    items = "".join(
        f'<li class="pw-{kind}"><span class="pw-tag">{STORY_TAGS[kind][0]} {STORY_TAGS[kind][1]}</span>'
        f'<span>{html.escape(text)}</span></li>'
        for kind, text in story
    )
    st.markdown(
        f'<div class="plain-words"><div class="pw-title">🗣️ {title}</div><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def kpi_card(col, label: str, value: str, sub: str, accent: str = "blue") -> None:
    col.markdown(
        f'<div class="kpi-card kpi-{accent}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-head"><div class="page-icon">{icon}</div><div>'
        f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def go_to(page_name: str) -> None:
    """Button callback: switch the top navigation to another page."""
    st.session_state["nav"] = page_name


# ==========================================
# PAGE: HOME (dashboard hero)
# ==========================================
def page_home() -> None:
    region = analysis_region()
    temp = cached_temperature_summary(region)
    ext = cached_extremes(region)
    rain = cached_rainfall(region)

    sen_year = temp["sen"]["slope"]
    elapsed = temp["end"] - temp["start"]
    net = sen_year * elapsed
    warming = temp["mk"]["is_significant"] and sen_year > 0

    ann_e = ext["annual"]
    early = ann_e[ann_e["year"] <= 1990]["hot_months"].mean()
    late = ann_e[ann_e["year"] >= 2011]["hot_months"].mean()
    cp = rain["change_point"]

    lead_tail = "— and the extremes are shifting too." if warming else "."
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">NASA Space Apps 2026 · Be An Earth System Trend Detective</div>
            <div class="hero-title">Is {region} heating up?</div>
            <div class="hero-lead">
                {temp['end'] - temp['start'] + 1} years of NASA MERRA-2 data ({temp['start']}–{temp['end']}) say
                <span class="hero-answer">{'YES' if warming else 'NOT CLEARLY'}</span> {lead_tail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.container(key="kpis").columns(4)
    kpi_card(k1, "Net warming", f"{net:+.2f} °C", f"since {temp['start']} · {sen_year * 10:+.2f} °C per decade", "red")
    if early > 0:
        kpi_card(k2, "Unusually hot months", f"×{late / early:.1f}",
                 f"{late:.1f}/yr in 2011–{temp['end']} vs {early:.1f}/yr in 1981–90", "orange")
    else:
        kpi_card(k2, "Unusually hot months", f"{late:.1f}/yr", f"in 2011–{temp['end']} (none in 1981–90)", "orange")
    if cp["is_significant"]:
        kpi_card(k3, "Rainfall jump (Pettitt)", f"{cp['change_year']} → {cp['new_regime_start']}",
                 f"{cp['mean_before']:.0f} → {cp['mean_after']:.0f} mm/yr (p = {cp['p_value']:.3f})", "blue")
    else:
        kpi_card(k3, "Rainfall", "No shift", f"no gradual trend or abrupt jump (p = {cp['p_value']:.2f})", "blue")
    p = temp["mk"]["p_value"]
    kpi_card(k4, "Is it real?", "p < 0.001" if p < 0.001 else f"p = {p:.3f}",
             "Mann-Kendall: warming is not chance" if temp["mk"]["is_significant"] else "not statistically significant",
             "green")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    tab_hi, tab_about = st.tabs(["✨ Highlights", "🎯 About the mission"])

    with tab_hi:
        chart_col, find_col = st.columns([3, 2], gap="large")
        with chart_col:
            mean_trend = {
                "sen_intercept": temp["sen"]["intercept"],
                "sen_slope_per_year": sen_year,
                "sen_per_decade": sen_year * 10.0,
                "slope_method": "Sen's slope",
            }
            ann_plot = temp["annual"].rename(columns={"value": "t2m"})
            st.plotly_chart(
                plot_extreme_index(ann_plot, "t2m", mean_trend, f"Annual Mean Temperature — {region}", "°C", NASA_RED_HEX),
                use_container_width=True,
            )
        with find_col:
            hot_word = "more frequent" if ext["trends"]["hot_months"]["is_significant"] else "not clearly more frequent"
            if cp["is_significant"]:
                rain_title = "Rain did not trend — it jumped."
                rain_text = (f"In MERRA-2, average rainfall shifted from {cp['mean_before']:.0f} to {cp['mean_after']:.0f} mm/yr "
                             f"starting in {cp['new_regime_start']} (pre-change regime ended {cp['change_year']}). "
                             "Ground-station gauge validation is required before attributing this to climate change.")
            else:
                rain_title = "Rain stayed noisy."
                rain_text = "No gradual or abrupt change is detectable."
            # 04 — forecast made from 1981–2023, scored against NASA's real 2024 months
            ver_html = ""
            obs_home = load_verification_observations(region)
            if not obs_home.empty:
                df_home = load_bundled_dataset(PRESET_FILES[region], region)[0]
                fc_home = cached_forecast(df_home[df_home["variable"] == "T2M"].copy(), "value", "best")
                ver_home = verify_forecast_against_observations(fc_home, obs_home[obs_home["variable"] == "T2M"])
                if ver_home.get("available"):
                    ver_html = (
                        f'<div class="finding"><span class="finding-num">04</span><div>'
                        f'<b>We predicted {ver_home["period"][-4:]} before seeing it.</b> Trained only on data to 2023, the '
                        f'model expected {ver_home["predicted_mean_anomaly"]:+.1f} °C above normal; NASA then recorded '
                        f'{ver_home["observed_mean_anomaly"]:+.1f} °C, with {ver_home["hit_95"]:.0f}% of months inside '
                        f'our 95% band.</div></div>'
                    )
            st.markdown("#### Key findings")
            st.markdown(
                f"""
                <div class="finding"><span class="finding-num">01</span><div>
                <b>The average is rising.</b> Mean temperature rose {sen_year * 10:+.2f} °C per decade,
                about {net:+.1f} °C since {temp['start']}.</div></div>
                <div class="finding"><span class="finding-num">02</span><div>
                <b>The extremes moved.</b> The hottest value of the year rises
                {ext['trends']['annual_hottest']['sen_per_decade']:+.2f} °C/decade and unusually hot months are
                {hot_word}.</div></div>
                <div class="finding"><span class="finding-num">03</span><div>
                <b>{rain_title}</b> {rain_text}</div></div>
                {ver_html}
                """,
                unsafe_allow_html=True,
            )
            b1, b2 = st.columns(2)
            b1.button("Temperature →", on_click=go_to, args=("🌡️ Temperature",), use_container_width=True)
            b2.button("Extremes →", on_click=go_to, args=("🔥 Extremes",), use_container_width=True)
            b3, b4 = st.columns(2)
            b3.button("Rainfall →", on_click=go_to, args=("🌧️ Rainfall",), use_container_width=True)
            b4.button("Forecast →", on_click=go_to, args=("🔮 Forecasting",), use_container_width=True)
        render_four_questions(temp["answers"], f"Temperature in {region}")

    with tab_about:
        page_about()


# ==========================================
# PAGE: TEMPERATURE (tabs over the original pages)
# ==========================================
def page_temperature() -> None:
    page_header("🌡️", f"Temperature — {active_location}",
                "Mean 2-meter air temperature: time series, statistical trend tests and anomalies.")
    t1, t2, t3 = st.tabs(["📈 Time series", "🔬 Trend statistics", "⚠️ Anomalies"])
    with t1:
        page_time_series()
    with t2:
        page_trend_engine()
    with t3:
        page_anomalies()


# ==========================================
# PAGE: RAINFALL
# ==========================================
def page_rainfall() -> None:
    region = analysis_region()
    res = cached_rainfall(region)
    t_ann = res["trends"]["annual_total"]
    cp = res["change_point"]

    page_header("🌧️", f"Rainfall — {region}",
                "NASA POWER PRECTOTCORR converted from mm/day to monthly and yearly totals.")
    if active_location not in PRESET_FILES:
        st.info(f"Rainfall analysis uses the bundled regions; showing {region}. Pick a region in the top bar to switch.")

    k1, k2, k3, k4 = st.container(key="kpis").columns(4)
    kpi_card(k1, "Mean annual rainfall", f"{t_ann['mean']:.0f} mm",
             f"{res['dry_month_fraction'] * 100:.0f}% of months are dry (< 1 mm)", "blue")
    kpi_card(k2, "Gradual trend", "YES" if t_ann["is_significant"] else "NO",
             f"Sen {t_ann['sen_slope_per_decade']:+.1f} mm/decade · MK p = {t_ann['mk_p']:.2f}",
             "green" if t_ann["is_significant"] else "slate")
    kpi_card(k3, "Abrupt shift (Pettitt)", f"{cp['change_year']} → {cp['new_regime_start']}" if cp["is_significant"] else "None",
             f"Pettitt p = {cp['p_value']:.4f}", "orange" if cp["is_significant"] else "slate")
    kpi_card(k4, "Before → after", f"{cp['mean_before']:.0f} → {cp['mean_after']:.0f}",
             f"mm/yr ({cp['pct_change']:+.0f}%)" if cp["is_significant"] else "mm/yr (not significant)",
             "red" if cp["is_significant"] else "slate")
    if t_ann["low_reliability"]:
        st.warning("⚠️ **Low reliability for this region**: " + "; ".join(t_ann["reliability_notes"]) + ".")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    tab_o, tab_c, tab_s, tab_r = st.tabs(["📋 Overview", "📊 Charts", "🔬 Statistics", "🗺️ Regions"])

    with tab_o:
        render_plain_words(rainfall_story(res, region))
        render_four_questions(res["summary"], "Rainfall")
        if cp["is_significant"]:
            st.warning(
                f"**{cp['interpretation']}**\n\n"
                f"Statistical break in MERRA-2 reanalysis: Pre-change regime: 1981–{cp['change_year']} ({cp['mean_before']:.1f} mm/yr); "
                f"Post-change regime: {cp['new_regime_start']}–2023 ({cp['mean_after']:.1f} mm/yr). "
                "Pettitt's test detects **when** a statistical change occurred, not **why**. This empirical jump must NOT be "
                "conclusively attributed to climate change, TRMM/GPM observing-system transitions, or reanalysis artifacts "
                "without independent validation against Egyptian Meteorological Authority (EMA) ground-station rain gauges."
            )
        with st.expander("How this analysis works"):
            st.markdown(
                """
                - `PRECTOTCORR` is a **rate (mm/day)**; it is multiplied by the days in each month to get **totals**.
                - Three yearly series are tested: the **calendar-year total**, the **wet-season total (Oct → Mar)** and the
                  **number of wet months (≥ 1 mm)**.
                - **Mann-Kendall + Hamed-Rao** test for a *gradual* trend (significant only if both agree);
                  **Sen's slope** gives its size; the **Pettitt test** looks for a *sudden jump*.
                - Series that are mostly zeros or ties are flagged as low reliability.
                """
            )

    with tab_c:
        view = st.radio("Series", ["Calendar-year total", "Wet-season total (Oct–Mar)"], horizontal=True, key="rain_series")
        if view == "Calendar-year total":
            fig = plot_rainfall_annual_totals(res["annual"], t_ann, res["normal_mm"], region, change_point=cp)
        else:
            seasons = res["seasons"]
            base = seasons[(seasons["season_end_year"] >= 1981) & (seasons["season_end_year"] <= 2010)]
            fig = plot_rainfall_annual_totals(
                seasons, res["trends"]["wet_season_total"], float(base["total_mm"].mean()), region,
                x_col="season_end_year", title_prefix="Wet-Season (Oct–Mar) Rainfall Totals",
            )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Blue: above the 1981–2010 normal · tan: below · red: Sen's slope · purple dotted: level before/after a "
                   "significant Pettitt shift. Wet seasons are labelled by the year they end.")
        st.plotly_chart(plot_rainfall_monthly_climatology(res["climatology"], region), use_container_width=True)

    with tab_s:
        labels = {
            "annual_total": "Calendar-year total (mm)",
            "wet_season_total": "Wet-season total, Oct–Mar (mm)",
            "wet_months": "Number of wet months (≥ 1 mm)",
        }
        rows = []
        for key, label in labels.items():
            t = res["trends"][key]
            unit = "months" if key == "wet_months" else "mm"
            rows.append({
                "Metric": label, "Years": t["n_years"], "Mean": round(t["mean"], 2),
                "Sen slope / decade": f"{t['sen_slope_per_decade']:+.2f} {unit}",
                "95% CI / decade": f"[{t['sen_ci_per_decade'][0]:+.2f}, {t['sen_ci_per_decade'][1]:+.2f}]",
                "MK Z": round(t["mk_z"], 2), "MK p": round(t["mk_p"], 4), "Hamed-Rao p": round(t["hr_p"], 4),
                "Verdict": t["verdict"],
                "Reliability": ("Low: " + "; ".join(t["reliability_notes"])) if t["low_reliability"] else "OK",
            })
        st.markdown("#### Trend tests")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        ranked = res["annual"][["year", "total_mm", "pct_of_normal"]].rename(
            columns={"year": "Year", "total_mm": "Total (mm)", "pct_of_normal": "% of normal"}).round(1)
        w1, w2 = st.columns(2)
        with w1:
            st.markdown("#### Wettest 5 years")
            st.dataframe(ranked.nlargest(5, "Total (mm)"), use_container_width=True, hide_index=True)
        with w2:
            st.markdown("#### Driest 5 years")
            st.dataframe(ranked.nsmallest(5, "Total (mm)"), use_container_width=True, hide_index=True)
        with st.expander("Limitations"):
            st.markdown(
                """
                - `PRECTOTCORR` is MERRA-2 **reanalysis** precipitation, not a rain-gauge record; reanalysis rainfall is less
                  reliable than temperature, especially in very dry regions.
                - Each region is a single ~50 km grid cell.
                - "Wet month" = ≥ 1 mm monthly total; wet-month results depend on this threshold.
                - In Aswan almost every month is dry, so trend tests have very little information.
                """
            )

    with tab_r:
        all_annual = {name: cached_rainfall(name)["annual"] for name in preset_frames()}
        st.plotly_chart(plot_regional_rainfall(all_annual), use_container_width=True)
        st.dataframe(cached_rainfall_table(), use_container_width=True, hide_index=True)
        st.info(
            "🌡️ vs 🌧️ Temperature shows a significant warming trend in all three regions, but rainfall shows **no "
            "significant gradual trend** in any. Cairo and Alexandria instead show a significant **abrupt upward shift** in "
            "MERRA-2 rainfall around 2014/2015 (pre-change regime ends in 2014; post-change regime begins in 2015); Aswan shows neither. "
            "43 years of reanalysis data are not enough to establish whether this reflects physical climate shifts, natural decadal "
            "variability, or observing-system artifacts without independent validation against Egyptian Meteorological Authority (EMA) ground gauges."
        )


# ==========================================
# PAGE: TEMPERATURE EXTREMES
# ==========================================
def page_extremes() -> None:
    region = analysis_region()
    res = cached_extremes(region)
    et = res["trends"]

    page_header("🔥", f"Temperature Extremes — {region}",
                "Hottest (T2M_MAX) and coldest (T2M_MIN) values reached in each month, 1981–2023.")
    if active_location not in PRESET_FILES:
        st.info(f"Extremes analysis uses the bundled regions; showing {region}. Pick a region in the top bar to switch.")

    def sig(t: Dict[str, Any]) -> str:
        return "significant" if t["is_significant"] else "not significant"

    k1, k2, k3, k4 = st.container(key="kpis").columns(4)
    kpi_card(k1, "Hottest of the year (TXx proxy)", f"{et['annual_hottest']['sen_per_decade']:+.2f} °C",
             f"per decade · {sig(et['annual_hottest'])}", "red")
    kpi_card(k2, "Coldest of the year (TNn proxy)", f"{et['annual_coldest']['sen_per_decade']:+.2f} °C",
             f"per decade · {sig(et['annual_coldest'])}", "blue")
    kpi_card(k3, "Unusually hot months", f"{et['hot_months']['sen_per_decade']:+.2f}",
             f"months/yr per decade · {sig(et['hot_months'])}", "orange")
    kpi_card(k4, "Unusually cold months", f"{et['cold_months']['sen_per_decade']:+.2f}",
             f"months/yr per decade · {sig(et['cold_months'])}", "slate")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    tab_o, tab_c, tab_s, tab_r = st.tabs(["📋 Overview", "📊 Charts", "🔬 Statistics", "🗺️ Regions"])

    with tab_o:
        render_plain_words(extremes_story(res, region))
        render_four_questions(res["summary"], "Temperature Extremes")
        with st.expander("What the five indices mean"):
            st.markdown(
                """
                1. **Hottest value of the year (TXx proxy)**: the single highest monthly peak (absolute monthly maximum).
                2. **Coldest value of the year (TNn proxy)**: the single lowest monthly peak (absolute monthly minimum).
                3. **Unusually hot months**: monthly maximum above the 1981–2010 **90th percentile** *for that calendar month*.
                4. **Unusually cold months**: monthly minimum below the 1981–2010 **10th percentile** *for that calendar month*.
                5. **Monthly extreme range proxy**: hottest minus coldest value in each month, averaged over the year.

                A "hot July" is judged against other Julys. Under a stable 1981–2010 climate you would expect about
                **1.2 unusually hot and 1.2 unusually cold months per year**.

                *Note on proxy status*: Because NASA POWER provides monthly aggregated peak values rather than daily time series,
                these indices represent monthly-resolution extreme temperature proxies rather than standard daily ETCCDI indices.
                Short-duration daily heatwaves or diurnal extremes cannot be resolved from monthly data alone.
                """
            )

    with tab_c:
        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(plot_extreme_index(res["annual"], "annual_hottest", et["annual_hottest"],
                                               "Hottest of Each Year", "°C", NASA_RED_HEX), use_container_width=True)
        with c_right:
            st.plotly_chart(plot_extreme_index(res["annual"], "annual_coldest", et["annual_coldest"],
                                               "Coldest of Each Year", "°C", NASA_BLUE_HEX), use_container_width=True)
        st.plotly_chart(plot_decadal_extreme_months(res["decades"]), use_container_width=True)
        st.caption("Average number of unusually hot / cold months per year in each decade (the 1980s have 9 years, the 2020s 4).")
        st.plotly_chart(plot_monthly_extreme_trends(res["by_month_tmax"], res["by_month_tmin"]), use_container_width=True)
        st.caption("Each calendar month tested separately; solid bars stay significant after Benjamini-Hochberg FDR "
                   "correction, faded bars do not.")

    with tab_s:
        rows = []
        for key, (label, unit) in METRIC_INFO.items():
            t = et[key]
            lo, hi = t["ci_per_decade"]
            cpe = t["pettitt"]
            rows.append({
                "Index": label, "Mean": round(t["mean"], 2),
                "Slope / decade": f"{t['sen_per_decade']:+.2f} {unit}", "95% CI": f"[{lo:+.2f}, {hi:+.2f}]",
                "Slope method": t["slope_method"], "MK p": f"{t['mk_p']:.2e}", "Hamed-Rao p": f"{t['hr_p']:.2e}",
                "Trend (both tests)": t["verdict"],
                "Abrupt shift (Pettitt)": f"Yes ({cpe['change_year']} → {cpe['new_regime_start']})" if cpe["is_significant"] else "No",
            })
        st.markdown("#### All five indices")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Counts are small integers with many ties, so Sen's median slope collapses to 0; their magnitude is "
                   "given by OLS. Significance always comes from Mann-Kendall and Hamed-Rao.")
        ann_e = res["annual"]
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### 5 hottest years")
            st.dataframe(ann_e.nlargest(5, "annual_hottest")[["year", "annual_hottest", "hottest_month"]]
                         .rename(columns={"year": "Year", "annual_hottest": "Hottest (°C)", "hottest_month": "Month"}).round(2),
                         use_container_width=True, hide_index=True)
        with r2:
            st.markdown("#### 5 coldest years")
            st.dataframe(ann_e.nsmallest(5, "annual_coldest")[["year", "annual_coldest", "coldest_month"]]
                         .rename(columns={"year": "Year", "annual_coldest": "Coldest (°C)", "coldest_month": "Month"}).round(2),
                         use_container_width=True, hide_index=True)
        with st.expander("Limitations"):
            st.markdown(
                """
                - MERRA-2 reanalysis grid-cell extremes (~50 km), not station records; local extremes can be more severe.
                - Percentile thresholds come from 30 values per calendar month and carry sampling uncertainty.
                - Hamed-Rao can give a much smaller p-value than Mann-Kendall when autocorrelation is negative; a trend is
                called significant only when **both** tests agree.
                """
            )

    with tab_r:
        st.dataframe(cached_extremes_table(), use_container_width=True, hide_index=True)
        st.info(
            "🔥 In all three regions the **hottest value of the year (TXx proxy) is rising** and **unusually hot months are becoming "
            "more frequent**. Unusually **cold** months are becoming **rarer** in Cairo and Aswan. The single coldest value "
            "of the year shows no significant trend anywhere."
        )


# ==========================================
# TOP BAR (brand · navigation · region · data source)
# ==========================================
NAV_PAGES = {
    "🏠 Home": page_home,
    "🌡️ Temperature": page_temperature,
    "🔥 Extremes": page_extremes,
    "🌧️ Rainfall": page_rainfall,
    "⚠️ Anomalies": page_anomalies,
    "🔮 Forecasting": page_forecasting,
    "🗺️ Regions": page_regional,
    "📂 Data": page_data_explorer,
    "📚 Method": page_methodology,
}


def _load_preset(region: str) -> None:
    df_loaded, meta_loaded = load_bundled_dataset(PRESET_FILES[region], region)
    st.session_state["raw_df"] = df_loaded
    st.session_state["metadata"] = meta_loaded
    st.session_state["active_location"] = region


def _on_region_change() -> None:
    _load_preset(st.session_state.get("region_select", "Greater Cairo"))


if "raw_df" not in st.session_state:
    if os.path.exists(PRESET_FILES["Greater Cairo"]):
        _load_preset("Greater Cairo")
    else:
        st.session_state["raw_df"] = pd.DataFrame()
        st.session_state["metadata"] = {}
        st.session_state["active_location"] = "None"

with st.container(key="topbar"):
    brand_col, nav_col, region_col, src_col = st.columns([1.5, 8.4, 1.5, 0.7], vertical_alignment="center")
    with brand_col:
        st.markdown(
            '<div class="brand"><span class="brand-logo">🌍</span><div>'
            '<div class="brand-name">EarthTrend</div><div class="brand-tag">NASA Space Apps 2026</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
    with nav_col:
        page = st.radio("Navigation", list(NAV_PAGES), horizontal=True, label_visibility="collapsed", key="nav")
    with region_col:
        st.selectbox("Region", list(PRESET_FILES), key="region_select", on_change=_on_region_change,
                     label_visibility="collapsed")
    with src_col:
        with st.popover("⚙️", help="Other data sources: upload a NASA CSV or query the live NASA POWER API",
                        use_container_width=True):
            st.markdown("**🛰️ Other data sources**")
            data_source_mode = st.selectbox(
                "Source Mode", ["Upload NASA CSV (Giovanni / POWER)", "Live NASA POWER API Query"]
            )
            if data_source_mode == "Upload NASA CSV (Giovanni / POWER)":
                uploaded_file = st.file_uploader("Upload CSV File", type=["csv", "txt"],
                                                 help="NASA Giovanni time series CSV or NASA POWER monthly table.")
                custom_loc_label = st.text_input("Region Name", value="Uploaded Region")
                if uploaded_file is not None and st.button("Parse Uploaded CSV", use_container_width=True):
                    try:
                        df_up, meta_up = auto_load_dataset(uploaded_file.getvalue().decode("utf-8"),
                                                           default_location=custom_loc_label)
                        st.session_state["raw_df"] = df_up
                        st.session_state["metadata"] = meta_up
                        st.session_state["active_location"] = custom_loc_label
                        st.success(f"Loaded {len(df_up)} records!")
                    except Exception as e:
                        st.error(f"Error parsing file: {e}")
            else:
                st.caption("Live connection to the NASA Langley Research Center POWER API (MERRA-2)")
                lat_in = st.number_input("Latitude", value=30.0444, format="%.4f")
                lon_in = st.number_input("Longitude", value=31.2357, format="%.4f")
                start_y = st.number_input("Start Year", min_value=1981, max_value=2023, value=1981)
                end_y = st.number_input("End Year", min_value=1981, max_value=2023, value=2023)
                loc_name_in = st.text_input("Location Name", value="Custom Coordinates")
                if st.button("Fetch Real NASA Data", use_container_width=True):
                    with st.spinner(f"Querying NASA POWER for ({lat_in:.2f}, {lon_in:.2f})..."):
                        try:
                            df_live, meta_live, raw_csv_live = fetch_nasa_power_point(
                                latitude=lat_in, longitude=lon_in, start_year=int(start_y), end_year=int(end_y)
                            )
                            st.session_state["raw_df"] = df_live
                            st.session_state["metadata"] = meta_live
                            st.session_state["active_location"] = loc_name_in
                            save_path = os.path.join("data", "raw", f"nasa_power_custom_{int(start_y)}_{int(end_y)}.csv")
                            with open(save_path, "w", encoding="utf-8") as f:
                                f.write(raw_csv_live)
                            st.success(f"Fetched {len(df_live)} records and cached locally!")
                        except Exception as e:
                            st.error(f"NASA API Query Failed: {e}")
            if st.session_state.get("active_location") not in PRESET_FILES:
                st.button("↩ Back to bundled regions", on_click=_on_region_change, use_container_width=True)

# Context strip under the navbar
df_current: pd.DataFrame = st.session_state.get("raw_df", pd.DataFrame())
metadata_current: Dict[str, Any] = st.session_state.get("metadata", {})
active_location: str = st.session_state.get("active_location", "Selected Region")

if not df_current.empty:
    chips = [f"📍 {active_location}"]
    if metadata_current.get("latitude") and metadata_current.get("longitude"):
        chips.append(f"{metadata_current['latitude']}°N, {metadata_current['longitude']}°E")
    if metadata_current.get("elevation_m"):
        chips.append(f"{metadata_current['elevation_m']} m elevation")
    if "year" in df_current.columns and not df_current["year"].dropna().empty:
        chips.append(f"{int(df_current['year'].min())}–{int(df_current['year'].max())}")
    chips.append(str(metadata_current.get("source", "NASA POWER · MERRA-2")))
    st.markdown('<div class="context-strip">' + "".join(f"<span>{c}</span>" for c in chips) + "</div>",
                unsafe_allow_html=True)

# ==========================================
# ROUTER
# ==========================================
if df_current.empty and page != "🏠 Home":
    st.error("⚠️ No active dataset loaded. Pick a region in the top bar or load a dataset with ⚙️.")
    st.stop()

NAV_PAGES[page]()

st.markdown(
    '<div class="footer">EarthTrend Explorer · Team <b>Machine Without Learning</b> · NASA Space Apps Challenge 2026 · '
    "Data: NASA POWER (MERRA-2), NASA Langley Research Center</div>",
    unsafe_allow_html=True,
)
