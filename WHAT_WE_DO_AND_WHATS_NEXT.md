# EarthTrend Explorer — What We Do & What's Next

**Project**: EarthTrend Explorer  
**Challenge**: NASA Space Apps Challenge 2026 — *"Be An Earth System Trend Detective!"*  
**Team**: Machine Without Learning  
**Workspace**: `D:\Space-Apps`  
**Current Active Branch**: [`feature/anomaly-forecasting`](file:///D:/Space-Apps)  
**Previous Working Branch**: `feature/rainfall-analysis`  
**Safety Checkpoint**: `refs/checkpoints/pre-forecasting` (`60f22e23c4b22baaff3dc6cd88cd32e734454854`)  
**Test Suite Status**: **71/71 tests passing (100% pass rate)**  
**Local Dashboard**: [http://localhost:8501](http://localhost:8501) (`HTTP 200 OK`)

---

## 1. Project Identity & Objective

EarthTrend Explorer is a scientifically sound statistical laboratory and interactive web platform built for the NASA Space Apps Challenge 2026. The platform investigates multi-decadal environmental trends, abrupt climate regime shifts, historical climatological anomalies, and forward predictive climate projections across Egyptian climate regimes using authentic **NASA Earth Science** observations and atmospheric reanalysis data.

Rather than presenting cosmetic dashboards or unverified assumptions, EarthTrend Explorer answers four fundamental scientific questions:
1. **What is changing?** (Mean temperatures, monthly extreme proxies, precipitation totals, drought indices, and compound extremes).
2. **Where is it changing?** (Greater Cairo urban basin, Alexandria Mediterranean coastline, and Aswan hyper-arid desert).
3. **How much is it changing?** (Robust non-parametric Theil-Sen slopes, decadal percentages, and change-point magnitudes).
4. **Is it statistically significant?** (Mann-Kendall, Hamed-Rao serial correlation corrections, Pettitt change-point test, multiple testing adjustments, and chronological out-of-sample backtesting).

---

## 2. Dataset & Geographic Scope

- **Data Source**: NASA POWER Project (Modern-Era Retrospective analysis for Research and Applications, Version 2 — MERRA-2, NASA GMAO).
- **Temporal Coverage**: 1981–2023 (43 contiguous years, exactly 516 monthly observations per variable, 0 missing records).
- **Spatial Resolution**: Native $0.5^\circ \times 0.625^\circ$ lat/lon reanalysis grid cells.
- **Monitored Variables**:
  - `T2M`: Monthly mean 2-meter air temperature (°C).
  - `T2M_MAX`: Monthly maximum 2-meter air temperature peak hour (°C, absolute monthly extreme).
  - `T2M_MIN`: Monthly minimum 2-meter air temperature low hour (°C, absolute monthly extreme).
  - `PRECTOTCORR`: Bias-corrected total precipitation flux (converted from mm/day to physically accumulated mm/month and mm/year).
- **Regions Audited**:
  - **Greater Cairo** ($30.0444^\circ\text{N}, 31.2357^\circ\text{E}$, $135.44\text{ m}$ elevation): Nile Delta/Valley transition.
  - **Alexandria** ($31.2001^\circ\text{N}, 29.9187^\circ\text{E}$, $7.0\text{ m}$ elevation): Mediterranean coastal marine climate.
  - **Aswan** ($24.0889^\circ\text{N}, 32.8998^\circ\text{E}$, $108.0\text{ m}$ elevation): Upper Egypt hyper-arid desert.

---

## 3. Technical Architecture

- **Core Analytics**: Python 3.10+, NumPy, Pandas.
- **Statistical Engine**: SciPy (`scipy.stats`), Statsmodels (`statsmodels.tsa`), PyMannKendall (`pymannkendall`).
- **Interactive Visualization**: Plotly (`plotly.graph_objects`, `plotly.express`, `plotly.subplots`) styled with a cohesive NASA dark space aesthetic.
- **Web Interface**: Streamlit with responsive top navigation bar, 9 analytical pages, and multi-tab workflows.
- **Verification Suite**: Pytest (61 unit, integration, and headless Streamlit AppTest tests).
- **Key Documentation**:
  - [`CHANGES_FROM_INTO.md`](CHANGES_FROM_INTO.md): Exhaustive change log from baseline to current.
  - [`MATHEMATICAL_AND_TECHNICAL_DECISIONS.md`](MATHEMATICAL_AND_TECHNICAL_DECISIONS.md): Mathematical formulations, derivations, and decision matrices.
  - [`SOURCES_AND_REFERENCES.md`](SOURCES_AND_REFERENCES.md): Academic bibliography and data provenance.
  - [`SESSION_HANDOVER.md`](SESSION_HANDOVER.md): Developer onboarding and session history.

---

## 4. Project Roadmap & Checkpoints

### Legend
- `[x]` **Completed & Verified** (Implemented in code, passing tests, and confirmed in UI).
- `[~]` **In Progress** (Active research or intermediate implementation).
- `[ ]` **Not Started / Planned** (Scheduled future work package).
- `[!]` **Blocked / Needs Scientific Decision** (Requires external validation or ground-truth data).

---

### CHECKPOINT 1 — Project Foundation & Data Engineering
**Status**: `[x] Completed & Verified`

- [x] Repository initialization, modular directory structure (`src/`, `tests/`, `data/`, `.streamlit/`).
- [x] NASA POWER API ingestion client (`src/data_loader.py::fetch_nasa_power_point`).
- [x] NASA POWER monthly header parser, coordinate extractor, and tidy long-format converter (`parse_nasa_power_header`, `load_nasa_power_csv`).
- [x] Bundled datasets for Greater Cairo, Alexandria, and Aswan (1981–2023, 516 months each, 0 missing values).
- [x] Timestamp deduplication and gap detection engine (`src/preprocessing.py::check_and_deduplicate_timestamps`).
- [x] Automated missing value reporter and imputation options (`handle_missing_values`).
- [x] Physical precipitation unit conversion: rate ($\text{mm/day}$) $\times$ days in calendar month (with leap-year handling) $\to$ physical accumulation ($\text{mm/month}$, $\text{mm/year}$).

---

### CHECKPOINT 2 — Historical Non-Parametric Trend Analysis
**Status**: `[x] Completed & Verified`

- [x] Non-parametric Mann-Kendall monotonic trend test ($S$, $\text{Var}(S)$, $Z$, $p$, $\tau$) (`src/trend_analysis.py::mann_kendall_test`).
- [x] Theil-Sen median slope estimator with Gilbert (1987) analytical 95% confidence intervals (`compute_sens_slope`).
- [x] Hirsch-Slack Seasonal Mann-Kendall omnibus test (`seasonal_mann_kendall_test`).
- [x] Hamed-Rao (1998) variance correction for detrended serial autocorrelation ($\eta = n/n^*$) (`hamed_rao_modified_mann_kendall`).
- [x] Discovery & verification of lag-12 rank autocorrelation in Alexandria ($\rho_{12} = -0.301$), deflating variance and raising $Z^*$ from $+5.50$ to $+6.23$.
- [x] 12 calendar month breakdown with multiple-testing error control (`compute_calendar_month_breakdown`):
  - False Discovery Rate (FDR): Benjamini-Hochberg ($q < 0.05$, 6 significant months in Cairo).
  - Family-Wise Error Rate (FWER): Holm-Bonferroni ($\alpha < 0.05$, 5 significant months in Cairo).
- [x] Regional warming quantification: Greater Cairo ($+0.29^\circ\text{C/decade}$), Alexandria ($+0.26^\circ\text{C/decade}$), and Aswan ($+0.43^\circ\text{C/decade}$).

---

### CHECKPOINT 3 — Climate Anomalies, Drought (SPI) & Regime Shifts
**Status**: `[x] Completed & Verified`

- [x] WMO 1981–2010 30-year climatological normal baseline calculation ($\mu_m, \sigma_m$) (`src/preprocessing.py::calculate_climatology`).
- [x] Monthly temperature departures and standardized Z-scores across `T2M`, `T2M_MAX`, and `T2M_MIN` (`src/anomaly_detection.py::compute_temperature_anomalies`).
- [x] Calendar-continuous persistent spell detection ($\ge 3$ months, $|Z| \ge 1.0\sigma$) with strict gap termination (`detect_persistent_anomalies`).
- [x] Multi-scalar Standardized Precipitation Index (SPI-3, SPI-6, SPI-12) via 2-parameter Gamma MLE fitting with zero-precipitation probability adjustment $H(x) = q + (1-q)G(x)$ (`compute_spi`).
- [x] Theoretical lower bound exposure: $\text{SPI}_{\text{lower}} = \Phi^{-1}(q)$, proving structural drought truncation when zero-rain frequency $q > 0.1587$.
- [x] Aridity guardrail: Automatically flags SPI as undefined/unreliable when baseline zero frequency $q \ge 0.50$ or non-zero count $< 5$.
- [x] Percent of Normal with low-reliability flag when baseline monthly rainfall $\mu_m < 1.0\text{ mm/month}$.
- [x] Pettitt (1979) non-parametric change-point test identifying the abrupt rainfall regime shift between 2014 and 2015 in Cairo and Alexandria ($+266\%$ and $+181\%$ jumps, $p < 0.01$).
- [x] Decadal compound hot-dry anomaly surge analysis ($Z_{\text{T}} \ge +1.0\sigma$ and $P \le 75\%$ of normal).
- [x] Cross-variable lagged correlation with Quenouille/Pyper-Peterman effective sample size correction ($N_{\text{eff}}$).

---

### CHECKPOINT 4 — Climate Anomaly Forecasting Engine
**Status**: `[x] Completed & Verified`

- [x] Seasonal Naive baseline benchmark model ($\hat{y}_{T+h} = y_{T+h-12}$) with in-sample seasonal difference error variance and seasonal random walk scaling (`src/forecasting.py::forecast_seasonal_naive`).
- [x] Holt-Winters Additive Exponential Smoothing (state-space ETS) with 1,000 Monte Carlo innovations simulations (`forecast_exponential_smoothing`).
- [x] Parsimonious $\text{SARIMA}(1,0,0) \times (1,1,0)_{12}$ state-space model with analytical Kalman filter covariance prediction intervals (`forecast_sarima`).
- [x] Chronological expanding-window cross-validation tournament across 5 historical forecast origins (2018–2022) with zero future leakage (`backtest_models`).
- [x] Out-of-sample performance evaluation via MAE, RMSE, and Mean Absolute Scaled Error (MASE, Hyndman & Koehler 2006). Winner on Cairo T2M: Holt-Winters ETS ($\text{MASE} = 0.800$, $\text{MAE} = 0.981^\circ\text{C}$).
- [x] Monotonic 80% and 95% probabilistic prediction intervals.
- [x] Future anomaly projection and standardized Z-score derivation relative to WMO 1981–2010 monthly normals (`generate_12month_forecast`).
- [x] Standard IPCC/WMO anomaly severity classification mapping (Extreme Warm $\ge +3\sigma$ down to Extreme Cold $\le -3\sigma$).
- [x] Precipitation non-negativity clamping at $0.0\text{ mm}$ for forecasts and lower uncertainty bounds.
- [x] Automatic Hyper-Arid Regime Advisory banner for zero-inflated distributions ($>80\%$ dry months).

---

### CHECKPOINT 5 — Streamlit Web Dashboard Integration
**Status**: `[x] Completed & Verified`

- [x] Full top navbar with responsive layout and glass aesthetic (`app.py`).
- [x] Dedicated `"🔮 Forecasting"` page in `NAV_PAGES` featuring 4 analytical tabs:
  1. *Executive Overview*: Dynamic model and variable selectors, 4 KPI cards (Validation Winner, 12-Mo Expected Anomaly, Peak Month, Uncertainty Spread), primary forecast chart with historical context, prediction envelopes, WMO normal, and origin dividing marker.
  2. *Projections & Intervals*: 12-month projection table, 80%/95% CI bounds, WMO normal, Percent of Normal, model parameters expander, and CSV export.
  3. *Future Anomaly Classification*: Departure bar chart with asymmetric 95% uncertainty whiskers, severity table, and conceptual callout distinguishing observed vs expected anomalies.
  4. *Chronological Validation & Backtesting*: Multi-origin tournament scoreboard, candidate comparison grouped bar chart (MAE, RMSE, MASE), and 2-panel residual diagnostic charts (time series and distribution histogram).
- [x] Cohesive Plotly dark space theme across all visualizations (`src/visualization.py`).

---

### CHECKPOINT 6 — Scientific Reliability & Test Suite
**Status**: `[x] Completed & Verified`

- [x] 61 tests passing across 8 test suites (100% pass rate in ~21s).
- [x] Dedicated forecasting test suite (`tests/test_forecasting.py`, 8 tests).
- [x] Dedicated anomaly detection test suite (`tests/test_anomaly_detection.py`, 16 tests).
- [x] Streamlit headless AppTest verifying clean rendering and navigation across all 9 pages (`tests/test_streamlit_app.py`).
- [x] Master reproducible scientific audit script (`scripts/reproducible_scientific_audit.py`) verified with zero numerical discrepancies.
- [x] Warning forensics: confirmed 0 warnings originate from forecasting code.
- [x] Academic references and API documentation compiled in [`SOURCES_AND_REFERENCES.md`](SOURCES_AND_REFERENCES.md).
- [x] Mathematical justifications documented in [`MATHEMATICAL_AND_TECHNICAL_DECISIONS.md`](MATHEMATICAL_AND_TECHNICAL_DECISIONS.md).

---

### CHECKPOINT 7 — Forecasting Improvements & Methodological Advancements
**Status**: `[x] Completed & Verified (except the GPM item, which needs external data)` — details in `CHANGES_FROM_INTO.md` §14

- [x] Verify availability of official NASA POWER reanalysis observations beyond December 2023 (available through 2025; 2024 bundled in `data/verification/`).
- [x] Use the genuine 2024 observations as a **true out-of-sample test**: forecasts made from 1981–2023 beat the naive benchmark for all 9 temperature series; 92–100% of 2024 months fall inside the 95% band (`verify_forecast_against_observations`, "✅ 2024 Reality Check" tab). The 1981–2023 analysis record is deliberately kept frozen so all documented numbers stay valid.
- [x] Evaluate candidate models across all permutations of Egyptian locations and variables (3 regions × 4 variables; table in `CHANGES_FROM_INTO.md` §14.10).
- [x] Evaluate empirical coverage rates of 80% and 95% prediction intervals across expanding-window splits, with an exact binomial calibration verdict.
- [x] Distinguish expected standardized departure ($\hat{Z}$) from actual probabilistic exceedance risk ($P(Z \ge z)$): P(above normal), P(Z ≥ +1), P(Z ≤ −1) per month.
- [!] **Scientific Decision Required**: Cross-validate the 2014 $\to$ 2015 MERRA-2 rainfall jump against independent satellite observations (NASA GPM IMERG 2000–present) and ground-station records from the Egyptian Meteorological Authority (EMA) to distinguish physical climate change from observing system transitions.
- [x] Formally test regional warming rate divergence ($H_0: \beta_{\text{Aswan}} - \beta_{\text{Cairo}} = 0$) on the paired difference series (OLS + Newey-West and MK/Hamed-Rao, Holm-corrected): Aswan warms **+0.16 °C/decade faster than Cairo** and **+0.18 faster than Alexandria** (significant); Cairo vs Alexandria not significant (`compare_regional_warming_rates`).

---

### CHECKPOINT 8 — Grounded Scientific Storytelling & Interpretability
**Status**: `[x] Completed & Verified` (`src/narrative.py`, "🗣️ In plain words" boxes)

- [x] Implement rule-based and data-grounded scientific natural language summaries based strictly on verified model outputs.
- [x] Plain-language explanations of observed historical trends, Pettitt changepoints, and temperature extremes (Rainfall, Extremes, Regions pages).
- [x] Plain-language communication of forecast uncertainty, prediction intervals, and expected departures for non-technical stakeholders (Forecasting + Reality Check tabs).
- [x] Strict adherence to scientific honesty: every sentence is tagged Observed / Tested / Projected / Keep in mind; no causal claims.

---

### CHECKPOINT 9 — NASA Space Apps Presentation & Submission Delivery
**Status**: `[ ] Scheduled Future Work Package`

- [ ] Review official NASA Space Apps Challenge 2026 submission rubrics and judging criteria.
- [ ] Structure the project narrative around the four foundational scientific questions.
- [ ] Prepare an executive slide deck and demonstration script highlighting:
  - Multi-decadal non-parametric warming across Egypt.
  - Pettitt abrupt rainfall shift and GPM satellite cross-validation context.
  - Decadal surge in compound Hot & Dry extremes.
  - Chronologically backtested 12-month predictive anomaly projections.
- [ ] Produce high-resolution visual assets and walkthrough demonstration video.
- [ ] Finalize public GitHub repository presentation, badges, license, and submission deliverables.

---

## 5. Current Development Priority

> **Next Priority Task**: **WP1 — NASA GPM IMERG Satellite Cross-Validation & 2024 Observation Ingestion**  
> Cross-validate the 2014 $\to$ 2015 MERRA-2 precipitation step-change in Cairo and Alexandria against independent satellite observations from the NASA Global Precipitation Measurement (GPM) IMERG mission, and check for official NASA POWER updates extending the historical monthly record into 2024.
