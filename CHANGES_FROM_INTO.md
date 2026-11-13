# Changes: From → Into

**Branch**: `feature/rainfall-analysis`  
**Base**: `main` @ `b4b8db4` (already merged into the branch, no conflicts)  
**Scope**: rainfall analysis, temperature-extremes analysis, full UI redesign (dashboard, dark theme, responsive)  
**Status**: 36/36 tests passing · `scripts/reproducible_scientific_audit.py` passes · every page checked in the browser at 375 px, 768 px and 1366 px

> This file lists every change as **From** (how it was on `main`) → **Into** (how it is on this branch), so you can review and merge with confidence.
> All the original statistical code (`mann_kendall_test`, `hamed_rao_modified_mann_kendall`, `compute_sens_slope`, …) is **unchanged**. Everything new is additive, except the UI layout and two corrected variable descriptions.

---

## 0. TL;DR

| # | Area | From | Into |
|---|---|---|---|
| 1 | Rainfall | Only selectable in generic pages; treated like temperature (mm/day averages) | Dedicated analysis + page: totals in mm, wet season, wet months, Pettitt jump test |
| 2 | `T2M_MAX` / `T2M_MIN` | Described as "monthly mean of daily max/min" (**wrong**) | Correctly described as the **absolute monthly extreme** (hottest/coldest value in the month) |
| 3 | Temperature extremes | No dedicated analysis | New module + page with 5 extreme indices |
| 4 | Mean vs extremes | Not compared | Warming rate of T2M, T2M_MAX, T2M_MIN compared per region |
| 5 | Layout | Sidebar with 9 pages | Top navbar with 7 pages, tabs inside pages |
| 6 | Home | Long text page | Dashboard hero with live KPI cards and key findings |
| 7 | Theme | Default Streamlit + white charts (clashed in dark mode) | Consistent dark "space" theme for app **and** charts |
| 8 | Mobile | Desktop only | Responsive: phone, tablet, laptop |
| 9 | Tests | 20 | 36 |

---

## 1. Pages: From → Into (where did my page go?)

| From (old sidebar page) | Into (new navbar page → tab) |
|---|---|
| 🏠 Home & Overview | 🏠 **Home** → tab *About the mission* (old content kept as-is) |
| 📂 Data Explorer | 📂 **Data** |
| 📈 Interactive Time Series | 🌡️ **Temperature** → tab *Time series* |
| 🔬 Trend Analysis Engine | 🌡️ **Temperature** → tab *Trend statistics* |
| ⚠️ Anomaly Detection | 🌡️ **Temperature** → tab *Anomalies* |
| 🗺️ Regional Comparison | 🗺️ **Regions** (+ new "Mean vs Hottest vs Coldest" section at the bottom) |
| 📚 Methodology & Sources | 📚 **Method** |
| — | 🔥 **Extremes** (new) → tabs *Overview · Charts · Statistics · Regions* |
| — | 🌧️ **Rainfall** (new) → tabs *Overview · Charts · Statistics · Regions* |
| — | 🏠 **Home** → tab *Highlights* (new dashboard) |

---

## 2. Data source controls

| From | Into |
|---|---|
| Sidebar: "Source Mode" select → "Select Egyptian Region" → **Load Region Data** button | Top bar: **region dropdown** (Greater Cairo / Alexandria / Aswan) that switches **instantly**, no button |
| Upload CSV and Live NASA POWER API in the sidebar | Same functionality, moved into the **⚙️ popover** in the top bar (+ a "Back to bundled regions" button) |
| "Active Dataset Context" block in the sidebar | **Context strip** under the navbar (region · coordinates · elevation · years · source) |
| Top banner `nasa-header` on every page | Removed (replaced by the navbar + page headers) |

---

## 3. Rainfall (`PRECTOTCORR`)

| From | Into |
|---|---|
| Analysed like temperature: monthly/annual **mean of mm/day** | Converted to **totals**: mm/day × days in month → mm/month → mm/year (`monthly_rainfall_totals`, `annual_rainfall_totals`) |
| Calendar year only | + **Wet season Oct → Mar** (labelled by end year, incomplete first season dropped) |
| — | + **Number of wet months** (≥ 1 mm) per year |
| — | + Monthly climatology ("when does it rain?") and **% of 1981–2010 normal** |
| Trend = Mann-Kendall only | Trend significant **only if Mann-Kendall AND Hamed-Rao agree**; Sen's slope in mm/decade and % of mean/decade |
| No check for zero-inflated data | **Reliability flags** when > 50 % of values are zero or tied, or totals are tiny (Aswan) |
| No break detection | **Pettitt (1979) change-point test** (`pettitt_change_point_test`) — detects a sudden jump that Mann-Kendall cannot see |

**Results (1981–2023)**

| Region | Mean | Gradual trend (MK + HR) | Abrupt shift (Pettitt) |
|---|---|---|---|
| Greater Cairo | 62 mm/yr | **No** (MK p = 0.77) | **Yes, from 2015** (p = 0.0067): 40 → 146 mm/yr |
| Alexandria | 108 mm/yr | **No** (MK p = 0.08, HR p = 0.24) | **Yes, from 2015** (p ≈ 0.006) |
| Aswan | 1.6 mm/yr | **No** (low reliability) | **No** |

> ⚠️ Pettitt says **when**, not **why**. The 2015 jump may be real (e.g. storms of late 2016 and March 2020) or an artefact of changing MERRA-2 inputs. The app states this explicitly. Recommended next step: confirm with an independent dataset (NASA GPM IMERG).

---

## 4. `T2M_MAX` / `T2M_MIN` meaning (correction)

| From | Into |
|---|---|
| `src/data_loader.py`: *"MERRA-2 monthly mean of daily maximum temperature"* | *"maximum 2-meter temperature reached within the month (absolute monthly extreme, NOT a mean of daily maxima)"* — same for `T2M_MIN` |
| README: "daily maximum/minimum temperatures" | "monthly extreme maximum/minimum temperatures (hottest/coldest value reached within each month)" |

**Proof**: in the NASA POWER CSV the `ANN` value of `T2M_MAX` equals the **maximum** of the 12 monthly values (e.g. Cairo 2000: Jan = 24.7 °C, ANN = 44.87 °C = July). A mean of daily maxima for Cairo in January would be ~19–20 °C. Covered by `test_nasa_power_t2m_max_is_monthly_extreme_not_mean`.

👉 **When presenting**: say *"the hottest value reached in the month"*, not *"average daily maximum"*.

---

## 5. Temperature extremes (new)

| From | Into |
|---|---|
| `T2M_MAX` / `T2M_MIN` only selectable in generic pages | New module `src/extremes_analysis.py` + 🔥 **Extremes** page |

Five yearly indices per region, each with Mann-Kendall + Hamed-Rao, slope/decade with 95 % CI and Pettitt:

1. **Hottest value of the year** (max of `T2M_MAX`)
2. **Coldest value of the year** (min of `T2M_MIN`)
3. **Unusually hot months**: `T2M_MAX` above the 1981–2010 **90th percentile of that calendar month**
4. **Unusually cold months**: `T2M_MIN` below the 1981–2010 **10th percentile of that calendar month**
5. **Monthly extreme range** (`T2M_MAX − T2M_MIN`)

Plus a per-calendar-month breakdown with Benjamini-Hochberg FDR, decade counts, record years and a regional table.

> Method note: for indices 3–4 (small integer counts with many ties) Sen's median slope collapses to 0, so the **magnitude** is reported with OLS; **significance** still comes from Mann-Kendall + Hamed-Rao.

**Results, Greater Cairo**: hottest value of the year **+0.33 °C/decade** (significant) · unusually hot months **↑ significant** (0.4/yr in 1981–90 → 1.3/yr in 2011–23, ×3.3) · unusually cold months **↓ significant** · coldest value of the year: not significant · extreme range: not significant.  
All three regions: hottest value ↑ and hot months ↑ significant; Aswan has the fastest rise in hot months (+0.81 months/yr per decade).

---

## 6. Mean vs hottest vs coldest (new, Regions page)

| From | Into |
|---|---|
| Regional page compared only one variable at a time | New section comparing warming of `T2M`, `T2M_MAX`, `T2M_MIN` side by side with 95 % CI error bars (`compare_temperature_extremes` in `src/trend_analysis.py`) |

| Region | T2M | T2M_MAX | T2M_MIN | (°C/decade, all significant) |
|---|---|---|---|---|
| Greater Cairo | +0.29 | +0.34 | +0.37 | |
| Alexandria | +0.26 | +0.44 | +0.39 | |
| Aswan | +0.43 | +0.50 | +0.43 | |

Confidence intervals overlap, so the app does **not** claim extremes warm faster than the mean.

---

## 7. Home page

| From | Into |
|---|---|
| Long text: mission, 4 questions, Cairo benchmark | **Hero**: "Is ⟨region⟩ heating up?" + live YES/NO answer |
| Hard-coded benchmark numbers | **4 KPI cards computed live** from the analysis for the selected region: net warming (+1.23 °C), unusually-hot-month multiplier (×3.3), rainfall jump (2015), significance (p < 0.001) |
| — | Annual temperature chart with Sen's slope, **"Three findings"** with buttons to Temperature / Extremes / Rainfall, and the 4 challenge questions as cards |
| — | Old Home content preserved in the *About the mission* tab |

---

## 8. Visual design

| From | Into |
|---|---|
| Default Streamlit theme (followed OS light/dark) | `.streamlit/config.toml`: fixed **dark theme** (navy background, NASA red primary), minimal toolbar |
| White Plotly charts on a dark page | One dark chart style in `_apply_scientific_theme` (card background, light text, subtle grid, dark tooltips); brighter palette for contrast |
| White `q-box` / `stat-card` | Dark cards with accent borders; `st.metric` styled the same |
| Sidebar navigation | Sticky glass **top navbar** with pill tabs; sidebar hidden |
| Plain section text for the 4 questions | `render_four_questions()` card grid |
| Metric values cut off with "…" | Values wrap |

---

## 9. Responsive

| Width | Layout |
|---|---|
| ≥ 1180 px (laptop) | Navbar in one row, 4 KPI cards per row |
| 960–1180 px | Smaller pills, brand tagline hidden |
| 640–960 px (tablet) | Navbar on two rows (brand · region · ⚙️ / scrollable pills), KPI cards 2 × 2, navbar not sticky |
| < 640 px (phone) | Compact hero/headers, KPI cards 2 × 2, tabs scroll sideways, chart titles smaller, Plotly toolbar hidden |
| < 360 px | KPI cards one per row |

---

## 10. Code structure (for developers)

| From | Into |
|---|---|
| `app.py`: one long `if page == … elif …` chain | Each original page is a function: `page_about`, `page_data_explorer`, `page_time_series`, `page_trend_engine`, `page_regional`, `page_anomalies`, `page_methodology` (**bodies unchanged**) |
| — | New page functions: `page_home`, `page_temperature`, `page_rainfall`, `page_extremes` |
| Page chosen by sidebar radio | `NAV_PAGES` dict + top-bar radio (`key="nav"`) → router at the bottom of `app.py` |
| Analysis recomputed on every rerun | `cached_rainfall`, `cached_extremes`, `cached_temperature_summary`, … (`st.cache_data`, per region) |
| — | Helpers: `kpi_card`, `page_header`, `render_four_questions`, `go_to` |

**To add a new page**: write `def page_xxx() -> None:` and add `"🔤 Name": page_xxx` to `NAV_PAGES`.

### Files

| Status | File | What |
|---|---|---|
| **new** | `src/rainfall_analysis.py` | Rainfall totals, wet season, wet months, climatology, % of normal, trend tests with reliability flags, **Pettitt test**, regional table |
| **new** | `src/extremes_analysis.py` | 5 temperature-extreme indices, percentile thresholds, trends, decade counts, regional table |
| **new** | `tests/test_rainfall_analysis.py` | 11 tests (unit conversion, leap years, wet season, Pettitt on synthetic + real data, mean-vs-extremes comparison, …) |
| **new** | `tests/test_extremes_analysis.py` | 5 tests (incl. proof that `T2M_MAX` is a monthly extreme) |
| **new** | `.streamlit/config.toml` | Dark theme, minimal toolbar |
| **new** | `CHANGES_FROM_INTO.md` | This file |
| changed | `app.py` | Layout, navbar, new pages, CSS (analysis calls unchanged) |
| changed | `src/visualization.py` | Dark chart theme + 7 new chart functions (rainfall, extremes, comparison) |
| changed | `src/trend_analysis.py` | **Added** `compare_temperature_extremes` (nothing existing modified) |
| changed | `src/data_loader.py` | `T2M_MAX` / `T2M_MIN` descriptions corrected (text only) |
| changed | `README.md` | Same description fix (1 line) |
| changed | `.gitignore` | `.streamlit/*` ignored except `config.toml` (secrets stay ignored) |

---

## 11. How to review / run

```bash
git fetch origin
git checkout feature/rainfall-analysis
pip install -r requirements.txt      # no new dependencies
pytest -q                            # expect 36 passed
streamlit run app.py                 # http://localhost:8501
```

---

## 12. Known notes / not done

- **README** still describes the old 7 sidebar pages; it does not yet mention the Rainfall / Extremes pages or the findings above.
- Streamlit prints deprecation warnings for `use_container_width` (Streamlit wants `width="stretch"`); it still works in the pinned version, but should be replaced before upgrading Streamlit.
- `scripts/reproducible_scientific_audit.py` crashes on the Windows console because of the `✓` character; run it with `PYTHONIOENCODING=utf-8` (it passes).
- The rainfall jump in 2015 is **not yet validated** against an independent dataset (GPM IMERG).
- Completed: Climate Anomaly Detection Engine (WMO 1981–2010 normal, persistent spells, multi-scalar SPI, compound extremes).
- Completed: Climate Anomaly Forecasting System (Seasonal Naive baseline, Holt-Winters ETS, SARIMAX, expanding-window multi-origin backtesting, prediction intervals, and dedicated 4-tab Streamlit forecasting interface).

---

## 13. Phase 3 & 4 Update: Climate Anomaly Detection & Anomaly Forecasting Engine (2026-11-13)

### 13.1 Anomaly Forecasting Engine (`src/forecasting.py`)
- **Three Candidate Architectures Implemented**:
  1. **Seasonal Naive Baseline Benchmark**: Point projection $\hat{y}_{T+h} = y_{T+h-12}$ repeating the final historical year's seasonal cycle. Residual variance $\hat{\sigma}_{\text{naive}}^2$ computed from in-sample seasonal differences $y_t - y_{t-12}$ with multi-step horizon variance scaling $\text{Var}(\hat{y}_{T+h}) = k \cdot \hat{\sigma}^2$ ($k = \lceil h/12 \rceil$).
  2. **Holt-Winters Additive Exponential Smoothing (ETS)**: State-space additive level, linear trend, and 12-month seasonality fit via MLE/SSE. Generates 80% and 95% prediction intervals via 1,000 Monte Carlo innovations simulations.
  3. **SARIMAX State-Space Model**: Parsimonious $\text{SARIMA}(1,0,0) \times (1,1,0)_{12}$ formulation with exact analytical Kalman filter covariance prediction intervals.
- **Strict Chronological Expanding-Window Cross-Validation**:
  - Evaluated across **5 historical forecast origins** (2018–2022) with a 12-month test horizon per origin.
  - Zero future data leakage: each split trains strictly on observations prior to the forecast origin.
  - Out-of-sample metrics computed per split and aggregate: MAE, RMSE, and **Mean Absolute Scaled Error (MASE)** relative to the in-sample Seasonal Naive benchmark (Hyndman & Koehler, 2006).
  - Model tournament automatically designates the model with lowest out-of-sample MAE as winner.
  - Empirical Cairo T2M tournament result: Holt-Winters ETS wins with $\text{MAE} = 0.981^\circ\text{C}$, $\text{RMSE} = 1.201^\circ\text{C}$, and $\text{MASE} = 0.800$ (outperforming Seasonal Naive $\text{MAE} = 1.287^\circ\text{C}$).
- **Probabilistic Prediction Intervals (80% & 95%)**:
  - Fully monotonic uncertainty bounds ($\text{CI}_{95,\text{low}} \le \text{CI}_{80,\text{low}} \le \hat{y} \le \text{CI}_{80,\text{high}} \le \text{CI}_{95,\text{high}}$).
- **WMO 1981–2010 Climatological Anomaly Evaluation**:
  - Projects expected departures $\hat{A}_{T+h} = \hat{y}_{T+h} - \mu_{m(h)}$ and standardized departures $\hat{Z}_{T+h} = \hat{A}_{T+h} / \sigma_{m(h)}$.
  - Maps departures into standard IPCC/WMO severity brackets: Extreme Warm ($\ge +3\sigma$), Severe Warm ($+2\sigma$ to $+3\sigma$), Moderate Warm ($+1\sigma$ to $+2\sigma$), Near Normal ($-1\sigma$ to $+1\sigma$), Moderate Cold ($-2\sigma$ to $-1\sigma$), Severe Cold ($-3\sigma$ to $-2\sigma$), and Extreme Cold ($\le -3\sigma$).
- **Physical Guardrails**:
  - Lower bound clamping at $0.0\text{ mm}$ for precipitation predictions and prediction intervals.
  - Automatic **Hyper-Arid Regime Advisory** banner for regions with $>80\%$ zero-rain months (e.g., Aswan), alerting users that background models reflect trace baseline precipitation and cannot deterministically extrapolate episodic convective pulses.

### 13.2 Dashboard Integration (`app.py` & `src/visualization.py`)
- **Top Navigation Integration**:
  - Added `"🔮 Forecasting": page_forecasting` to `NAV_PAGES` as a primary navigation item.
  - Adjusted topbar layout ratios to `[1.5, 8.4, 1.5, 0.7]` for responsive rendering across desktop and mobile.
- **Four Analytical Forecasting Tabs**:
  1. **📊 Executive Overview**:
     - Model selector, variable selector, 4 KPI cards (Validation Winner with MASE, 12-Month Expected Anomaly, Peak Forecast Month, 95% Uncertainty Spread).
     - Primary forecast plot (`plot_forecast_with_intervals`): 5-year historical line, 12-month future projection, 80% & 95% dual shaded yellow envelopes, dotted WMO normal curve, and vertical red dashed dividing line at forecast origin.
     - Hyper-arid advisory banner for Aswan precipitation.
  2. **📈 Projections & Intervals**:
     - Complete 12-month projection table with point forecasts, 80%/95% CI bounds, WMO normal, expected departures, and Percent of Normal.
     - One-click CSV export button (`earthtrend_<region>_<var>_forecast_2024.csv`).
     - Expandable model specification and fitted parameters panel.
  3. **⚠️ Future Anomaly Classification**:
     - Departure bar chart (`plot_future_anomaly_bars`) with asymmetric 95% uncertainty whiskers.
     - Month-by-month severity classification table.
     - Conceptual callout box articulating the distinction between historical observed ground truth vs. forward probabilistic projections.
  4. **🧪 Chronological Validation & Backtesting**:
     - Grouped bar chart (`plot_model_backtest_comparison`) comparing MAE, RMSE, and MASE across all 3 candidate models with MASE = 1.0 benchmark parity line.
     - Multi-origin tournament scoreboard across all 5 test splits.
     - 2-panel residual diagnostic charts (`plot_forecast_residuals_diagnostic`): Residual time series and distribution histogram.

### 13.3 Test Suite & Quality Assurance
- **Total Tests**: **61 tests passed out of 61 (100% pass rate in ~21s)**.
  - `tests/test_forecasting.py`: 8 new unit and integration tests covering Seasonal Naive, Holt-Winters ETS, SARIMAX, prediction interval monotonicity, precipitation non-negativity clamping, expanding-window backtesting tournament, future anomaly classification, Cairo end-to-end real data pipeline, and short-series fallbacks.
  - `tests/test_streamlit_app.py`: Headless AppTest verified navigation across all 9 pages including `"🔮 Forecasting"`.
  - `tests/test_anomaly_detection.py`: 16 anomaly detection tests passing.
  - `tests/test_rainfall_analysis.py`: 11 rainfall and Pettitt tests passing.
  - `tests/test_extremes_analysis.py`: 6 temperature extreme tests passing.
  - `tests/test_trend_analysis.py`: 11 trend and Hamed-Rao tests passing.
  - `tests/test_preprocessing.py`: 6 data cleaning tests passing.
  - `tests/test_integration.py`: 1 full pipeline integration test passing.
- **Reproducible Scientific Audit**: `python scripts/reproducible_scientific_audit.py` executed with zero numerical discrepancies across all regional trends, Hamed-Rao corrections, and Cairo calendar month multiple testing adjustments.
- **Warnings Forensics**: All 21 warnings confirmed as pre-existing synthetic zero-variance test fixtures (20 SciPy log warnings in `test_anomaly_detection.py`) and 1 external Dash plugin hook deprecation. Zero warnings originated from forecasting code.

### 13.4 Dependencies & Documentation Updates
- `requirements.txt`: Added `statsmodels>=0.14.0`.
- `README.md`: Updated test badge to `61/61_Passed`, added Section 2.6 documenting the Anomaly Forecasting Engine, and updated technology stack.
- `MATHEMATICAL_AND_TECHNICAL_DECISIONS.md`: Added Section 2.6 detailing mathematical derivations for Seasonal Naive, Holt-Winters ETS, SARIMAX, MASE, prediction intervals, and the technical decisions comparison table.
- `SOURCES_AND_REFERENCES.md`: Added citations for Hyndman & Athanasopoulos (2018), Hyndman & Koehler (2006), Box, Jenkins, Reinsel, & Ljung (2015), and Holt (1957) / Winters (1960).
- `SESSION_HANDOVER.md`: Updated Section 6 recording the implementation, architecture, and verification outcomes.

### 13.5 Git Provenance & Branch Records
- **New Branch**: `feature/anomaly-forecasting` (branched from local working tree on `feature/rainfall-analysis`).
- **Previous Branch**: `feature/rainfall-analysis` (HEAD at `e448d70bbba427d1aa61531d92f94f2646ed546a`).
- **Safety Checkpoint**: `refs/checkpoints/pre-forecasting` (`60f22e23c4b22baaff3dc6cd88cd32e734454854`).
- **New Commit**: `c5c9c95ad43f870fd554bf78860cf5ab784d1b6e` (`feat: implement climate anomaly forecasting engine and project roadmap`).
