# EarthTrend Explorer 🌍

**NASA Space Apps Challenge 2026**  
**Challenge**: *"Be An Earth System Trend Detective!"*  
**Team**: Machine Without Learning  
**Participation**: Virtual  
**Project**: EarthTrend Explorer  
**Branch**: `feature/rainfall-analysis`  

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyMannKendall](https://img.shields.io/badge/PyMannKendall-1.4.3-green.svg)](https://github.com/mmhs013/pymannkendall)
[![NASA Data](https://img.shields.io/badge/NASA_Earth_Science-MERRA--2-0B3D91.svg)](https://power.larc.nasa.gov/)
[![Tests Passing](https://img.shields.io/badge/pytest-61%2F61_Passed-success.svg)](https://docs.pytest.org/)
[![Theme](https://img.shields.io/badge/Theme-Space_Dark-0F1B33.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Key Documentation & Handover Guides

For researchers, evaluators, and incoming developers:

- 📋 [**CHANGES_FROM_INTO.md**](CHANGES_FROM_INTO.md): **From → Into Change Log** detailing all feature additions (rainfall, temperature extremes, dashboard redesign, dark space theme, responsive layout, Pettitt test).
- 📋 [**TEAM_HANDOVER.md**](TEAM_HANDOVER.md): Full modular architecture (Mermaid), module responsibilities, updated work packages, and developer onboarding.
- 📐 [**MATHEMATICAL_AND_TECHNICAL_DECISIONS.md**](MATHEMATICAL_AND_TECHNICAL_DECISIONS.md): Complete mathematical derivations, formulas (LaTeX), test statistics ($S$, $\tau$, $Z^*$, Theil-Sen, Pettitt change-point $U_t$/$K$, Holm FWER, BH FDR), and design rationales.
- 📚 [**SOURCES_AND_REFERENCES.md**](SOURCES_AND_REFERENCES.md): Academic bibliography (Mann 1945, Sen 1968, Pettitt 1979, Hamed & Rao 1998, Hirsch-Slack 1982, ETCCDI), NASA APIs, and WMO standards.
- 🛠️ [**CONTRIBUTING.md**](CONTRIBUTING.md): Git branch policy, conventional commits, code quality standards, and verification checklists.
- 📝 [**SESSION_HANDOVER.md**](SESSION_HANDOVER.md): Chronological development history, audit investigations, bug resolutions, and verification logs.
- 🔬 [**FINAL_METHODOLOGY_REVIEW.md**](FINAL_METHODOLOGY_REVIEW.md): Independent scientific audit, side-by-side Hamed-Rao continuity verification, and regional comparisons.

---

## 1. Project Overview

EarthTrend Explorer is a scientifically rigorous interactive data-analysis web application designed to investigate multi-decadal environmental trends using authentic **NASA Earth Science** observations and reanalysis data.

Rather than acting as a cosmetic dashboard with hardcoded or fabricated values, EarthTrend Explorer is a true **statistical laboratory** built to answer four fundamental scientific questions:

1. **What is changing?**  
   - **Temperature**: Mean air temperature (`T2M`) and absolute monthly temperature extremes (`T2M_MAX` hottest value reached in month, `T2M_MIN` coldest value reached in month).
   - **Temperature Extremes**: 5 climate indices (annual hottest value, annual coldest value, unusually hot months >P90, unusually cold months <P10, extreme range).
   - **Precipitation**: Bias-corrected precipitation (`PRECTOTCORR`) converted from mm/day rates to physical accumulated totals (mm/month, mm/year), wet-season accumulation (October–March), and number of wet months ($\ge 1\text{ mm}$).
2. **Where is it changing?**  
   Targeted across Egyptian climate regimes (Greater Cairo urban basin, Alexandria Mediterranean marine coast, and Aswan hyper-arid desert) mapped to exact NASA MERRA-2 $0.5^\circ \times 0.625^\circ$ grid cells.
3. **How much is it changing?**  
   Quantified via the robust non-parametric **Theil-Sen median slope estimator** with analytical 95% confidence intervals (Gilbert 1987), decadal rates, percentage of mean per decade, and supplementary OLS regression.
4. **Is the observed trend statistically significant?**  
   Evaluates null hypotheses ($H_0$: no monotonic trend) using the **Mann-Kendall test**, **Hamed-Rao Modified Mann-Kendall test** (accounting for serial autocorrelation), **Hirsch-Slack Seasonal Mann-Kendall test**, **Benjamini-Hochberg FDR / Holm FWER corrections** across 12 calendar months, and the **Pettitt (1979) change-point test** for abrupt regime shifts.

---

## 2. Key Scientific Findings (1981–2023)

### 2.1 Temperature: Greater Cairo Benchmark
Continuous 43-year monthly 2-meter air temperature (`T2M`) for **Greater Cairo** ($30.0444^\circ\text{N}, 31.2357^\circ\text{E}$, elevation: $135.44\text{ m}$):

| Metric | Result | Scientific Interpretation |
| :--- | :--- | :--- |
| **Observation Record** | 1981–2023 (43 years, 516 months) | Complete historical record, zero missing values, single MERRA-2 grid cell |
| **Annual MK Trend ($n=43$)** | **Increasing (Upward)** | Null hypothesis rejected ($H_1$ accepted, $S = +512.0$) |
| **Annual p-value** | $8.89 \times 10^{-8}$ | Highly statistically significant ($p < 0.001 \ll \alpha = 0.05$) |
| **Annual Standardized Z** | $+5.35$ | Well above the $99.9\%$ critical threshold ($Z_{0.999} = 3.29$) |
| **Kendall's $\tau$** | $+0.567$ | Substantial positive monotonic rank correlation |
| **Sen's Slope Rate** | **$+0.0293^\circ\text{C / year}$** | **$+0.293^\circ\text{C / decade}$** warming rate |
| **Sen's 95% CI** | $[+0.202, +0.389]^\circ\text{C / decade}$ | Entire confidence interval is strictly positive |
| **OLS Linear Rate** | $+0.0305^\circ\text{C / year}$ ($R^2 = 0.503$) | Strong linear agreement with robust median slope ($p = 9.99 \times 10^{-8}$) |
| **Net Cumulative Shift** | **$+1.23^\circ\text{C}$** | Net warming over 42 elapsed years ($+1.26^\circ\text{C}$ over 43 steps) |
| **Seasonal MK ($n=516$)** | $Z = +8.05$, $p < 10^{-15}$ | Multi-decadal upward trajectory ($S = +2665$, Sen's slope $+0.303^\circ\text{C/decade}$) |
| **Monthly Breakdown** | 5 (Holm FWER) / 6 (BH FDR) of 12 | 8 nominally significant ($p < 0.05$: Feb, May, Jul–Dec); 6 survive Benjamini-Hochberg FDR ($q < 0.05$: Jul–Dec); 5 survive Holm FWER ($\alpha < 0.05$: Jul, Aug, Sep, Nov, Dec); Jan, Mar, Apr, Jun show no detectable trend |
| **Hamed-Rao Modified MK** | $Z^* = +5.35$, $p = 8.89 \times 10^{-8}$ | Detrended rank autocorrelations within critical bound ($\pm 0.299$); correction factor $n/n^* = 1.000$ |

### 2.2 Regional Temperature Comparison Across Egypt
All three locations represent single, authentic NASA MERRA-2 grid cells ($0.5^\circ \times 0.625^\circ$), verified side by side:

| Region | Coordinates | Climate Regime | Sen Slope (°C/dec) | Sen 95% CI (°C/dec) | Theil-Sen Net (42 yrs) | MK Z | MK p-value | Hamed-Rao Factor | Corrected Z* | Corrected p* | Significant? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Greater Cairo** | $30.04^\circ\text{N}, 31.24^\circ\text{E}$ | Urban / Nile Valley | **+0.293** | $[+0.202, +0.389]$ | **+1.23 °C** | +5.35 | $8.89 \times 10^{-8}$ | 1.000000 | **+5.35** | **$8.89 \times 10^{-8}$** | **YES** |
| **Alexandria** | $31.20^\circ\text{N}, 29.92^\circ\text{E}$ | Mediterranean Marine | **+0.261** | $[+0.186, +0.344]$ | **+1.10 °C** | +5.50 | $3.70 \times 10^{-8}$ | 0.780520 | **+6.23** | **$4.64 \times 10^{-10}$** | **YES** |
| **Aswan** | $24.09^\circ\text{N}, 32.90^\circ\text{E}$ | Hyper-Arid Desert | **+0.434** | $[+0.312, +0.553]$ | **+1.82 °C** | +5.76 | $8.61 \times 10^{-9}$ | 1.000000 | **+5.76** | **$8.61 \times 10^{-9}$** | **YES** |

> [!NOTE]
> In Alexandria, a significant negative rank autocorrelation at lag 12 ($\rho_{12} = -0.30129$ vs critical bound $\pm 0.29889$) deflates variance, increasing $Z^*$ from $+5.50$ to $+6.23$. In Aswan and Cairo, detrended serial autocorrelations fall within the 95% critical bound ($\pm 0.2989$). Aswan exhibits a higher nominal warming slope ($+0.434^\circ\text{C/dec}$ vs Cairo's $+0.293^\circ\text{C/dec}$), but the available confidence intervals overlap and the current interaction analysis does not establish statistically significant regional divergence after multiple-comparison considerations. Original and modified MK statistics answer related but distinct inferential hypotheses, and neither is universally conservative.

### 2.3 Rainfall Analysis & Abrupt Regime Shifts (Pettitt Test)
Precipitation totals in mm/year, wet season (Oct–Mar), and Pettitt change-point detection:

| Region | Mean Annual Total | Dry Months (%) | Gradual Trend (MK + HR) | Pettitt Change-Point | Regime Shift Magnitude | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Greater Cairo** | $62.1\text{ mm/yr}$ | $52.7\%$ | **No significant trend** (MK $p = 0.77$) | **Regime Shift: 2014 → 2015** ($p = 0.0067$) | $39.9 \to 146.0\text{ mm/yr}$ ($+266\%$) | OK |
| **Alexandria** | $107.5\text{ mm/yr}$ | $44.6\%$ | **No significant trend** (HR $p = 0.24$) | **Regime Shift: 2014 → 2015** ($p = 0.0060$) | $78.0 \to 219.0\text{ mm/yr}$ ($+181\%$) | OK |
| **Aswan** | $1.6\text{ mm/yr}$ | $95.5\%$ | **No significant trend** | **No abrupt shift** ($p = 0.999$) | Stable hyper-arid zero-inflated baseline | Low ($>50\%$ zero) |

> [!IMPORTANT]
> The Pettitt change-point test detects **when** a statistical break occurred, not **why**. The abrupt jump between 2014 (final pre-change year) and 2015 (first post-change year) in Cairo and Alexandria reflects an empirical break in MERRA-2 precipitation totals. Conclusive attribution to physical climate change versus reanalysis observing-system transitions (such as TRMM to GPM constellation inputs) cannot be confirmed without validation against independent Egyptian Meteorological Authority (EMA) ground-station rain gauges.

### 2.4 Temperature Extremes & Mean vs Extremes
- **Hottest Value of the Year (TXx proxy)**: Rising significantly in all three regions (Cairo: $+0.33^\circ\text{C/decade}$, Alexandria: $+0.44^\circ\text{C/decade}$, Aswan: $+0.50^\circ\text{C/decade}$).
- **Coldest Value of the Year (TNn proxy)**: Showing positive slopes in Cairo and Aswan, but lower significance.
- **Unusually Hot Months (>90th percentile)**: Significantly increasing (Cairo: $0.4\text{ months/yr}$ in 1980s $\to 1.3\text{ months/yr}$ in 2010s–2020s, a $\times 3.3$ surge).
- **Unusually Cold Months (<10th percentile)**: Significantly declining across all regions.
- **Mean vs Extremes Comparison**:
  - Cairo: Mean T2M ($+0.29^\circ\text{C/dec}$), Hottest T2M_MAX proxy ($+0.34^\circ\text{C/dec}$), Coldest T2M_MIN proxy ($+0.37^\circ\text{C/dec}$).
  - Alexandria: Mean T2M ($+0.26^\circ\text{C/dec}$), Hottest T2M_MAX proxy ($+0.44^\circ\text{C/dec}$), Coldest T2M_MIN proxy ($+0.39^\circ\text{C/dec}$).
  - Aswan: Mean T2M ($+0.43^\circ\text{C/dec}$), Hottest T2M_MAX proxy ($+0.50^\circ\text{C/dec}$), Coldest T2M_MIN proxy ($+0.43^\circ\text{C/dec}$).
  - Because 95% confidence intervals overlap, we conservatively state that extremes and means are warming in unison without claiming differential rates.
  - *Terminology note*: Because NASA POWER provides monthly aggregated peak values rather than daily series, these metrics represent monthly-resolution extreme proxies rather than standard daily ETCCDI indices.

### 2.5 Climate Anomalies, Drought (SPI) & Decadal Compound Shifts
- **WMO 1981–2010 Climatological Baseline**: Evaluates departures ($A_t = T_t - \mu_m$) and standardized Z-scores ($Z_t = A_t / \sigma_m$) across `T2M`, `T2M_MAX`, and `T2M_MIN`.
- **Persistent Climatic Spells**: Calendar-continuous tracking of $|Z| \ge 1.0\sigma$ runs ($\ge 3$ months) identifies multi-month persistent warm spells and cold spells; temporal data gaps strictly terminate spells.
- **Multi-Scalar SPI Drought & Aridity Limitations**:
  - Multi-scale SPI (SPI-3, SPI-6, SPI-12) via 2-parameter Gamma MLE fitting and zero-adjusted CDF $H(x) = q + (1-q)G(x)$.
  - **Theoretical Lower Bound Exposure**: Exposes $\Phi^{-1}(q)$ as the structural minimum SPI. When $q > 0.1587$, $\Phi^{-1}(q) > -1.0$, structurally truncating moderate drought detection for zero-precipitation months.
  - **Aridity Guardrail**: Whenever $q \ge 0.50$ or non-zero count $< 5$, SPI is flagged as `Undefined / Low Reliability` to prevent false positive (wet) indices in desert dry seasons.
  - **Percent of Normal Stability**: Flags `percent_of_normal_reliable = False` ("Low Reliability") when baseline monthly rainfall $\mu_m < 1.0\text{ mm}$ to prevent runaway percentages on trace showers.
- **Decadal Surge in Compound Hot & Dry Extremes**:
  - Concurrent Hot & Dry months ($Z_{\text{T}} \ge +1.0\sigma$ AND $P \le 75\%$ of normal) have surged dramatically across all Egyptian climate regimes:
    - **Greater Cairo**: **6.5%** in the 1980s (7/108) $\to$ **10.0%** in the 1990s (12/120) $\to$ **13.3%** in the 2000s (16/120) $\to$ **24.2%** in the 2010s (29/120) $\to$ **27.1%** in the 2020s (13/48).
    - **Alexandria**: **7.4%** in the 1980s (8/108) $\to$ **6.7%** in the 1990s (8/120) $\to$ **15.8%** in the 2000s (19/120) $\to$ **28.3%** in the 2010s (34/120) $\to$ **33.3%** in the 2020s (16/48).
    - **Aswan**: **4.6%** in the 1980s (5/108) $\to$ **9.2%** in the 1990s (11/120) $\to$ **15.0%** in the 2000s (18/120) $\to$ **18.3%** in the 2010s (22/120) $\to$ **25.0%** in the 2020s (12/48).
    *(Temporal Coverage Note: The 1980s covers 9 elapsed years: 1981–1989, 108 calendar months due to record start; the 2020s covers 4 elapsed years: 2020–2023, 48 calendar months. Full intermediate decades 1990s, 2000s, 2010s each contain 10 complete years, 120 calendar months. Percentages are strictly normalized to elapsed months)*.
  - Meanwhile, Compound Cold & Dry months have collapsed (Cairo: 13 months in 1980s $\to$ 1 month in 2020s).
- **Cross-Variable Association & Regional Concordance**:
  - Winter rainy months in coastal and northern Egypt correlate negatively with temperature ($r \approx -0.19$, $p < 0.001$), consistent with winter Mediterranean depression cold-air advection, cloud reflection, and land-atmosphere feedbacks (serial persistence adjusted via Quenouille $N_{\text{eff}}$). Statistical correlation is distinguished from physical causation. Over 70% of months share the identical anomaly sign across Egypt.

### 2.6 Climate Anomaly Forecasting Engine & Chronological Validation
- **12-Month Horizon Projections**: Forward-looking predictive modeling for `T2M`, `T2M_MAX`, `T2M_MIN`, and `PRECTOTCORR` across Cairo, Alexandria, and Aswan.
- **Candidate Architectures**:
  1. **Seasonal Naive Baseline**: Benchmark persistence model repeating the corresponding calendar month from the final historical year ($\hat{y}_{T+h} = y_{T+h-12}$).
  2. **Holt-Winters Exponential Smoothing (ETS)**: State-space additive level, additive trend, and additive 12-month seasonality with 1,000 Monte Carlo innovations simulations.
  3. **SARIMAX**: Seasonal autoregressive state-space formulation ($\text{SARIMA}(1,0,0) \times (1,1,0)_{12}$) with exact analytical Kalman covariance bounds.
- **Rigorous Expanding-Window Backtesting Tournament**:
  - Validates models across 5 historical forecast origins (2018–2022) with strict temporal causality (no future data leakage).
  - Evaluates out-of-sample MAE, RMSE, and Mean Absolute Scaled Error (MASE, Hyndman & Koehler 2006). A model is crowned tournament winner only if it genuinely outperforms the Seasonal Naive benchmark ($\text{MASE} < 1.0$).
  - For Cairo T2M: Holt-Winters ETS wins the tournament with $\text{MAE} = 0.981^\circ\text{C}$, $\text{RMSE} = 1.201^\circ\text{C}$, and $\text{MASE} = 0.800$ (outperforming Seasonal Naive $\text{MAE} = 1.287^\circ\text{C}$).
- **Probabilistic Prediction Intervals**:
  - Dual shaded envelopes at 80% and 95% confidence levels, strictly monotonic and communicating true predictive uncertainty.
- **Expected Future Anomaly Classification**:
  - Projections evaluated against the WMO 1981–2010 monthly normal ($\mu_m, \sigma_m$).
  - Maps expected departures ($\hat{A}_{T+h}$) and standardized departures ($\hat{Z}_{T+h}$) into standard IPCC / WMO severity brackets: Extreme Warm ($\ge +3\sigma$), Severe Warm ($+2\sigma$ to $+3\sigma$), Moderate Warm ($+1\sigma$ to $+2\sigma$), Near Normal ($-1\sigma$ to $+1\sigma$), and Cold brackets.
- **Physical Guardrails**:
  - Non-negativity clamping ($\max(0.0, \cdot)$) for precipitation point forecasts and lower confidence bounds.
  - Automatic **Hyper-Arid Regime Advisory** banner for regions with $>80\%$ dry months (e.g., Aswan), explicitly warning users that background trace projections cannot extrapolate rare episodic convective storm pulses deterministically.
- **Interactive Dashboard Integration**:
  - Dedicated `"🔮 Forecasting"` page in `app.py` featuring 4 comprehensive analytical tabs:
    1. *Executive Overview*: Model selector, 4 KPI cards, primary forecast chart with prediction envelopes, baseline normal curve, and origin dividing marker.
    2. *Forecasts & Prediction Intervals*: Complete month-by-month table, 80%/95% CI spreads, fitted parameters, and CSV download.
    3. *Future Anomaly Classification*: Departure bar chart with uncertainty whiskers, Z-score severity table, and conceptual distinction between historical observed truth vs. forward probabilistic projections.
    4. *Chronological Validation & Backtesting*: Multi-origin cross-validation diagram, candidate comparison grouped bar chart, 5-split error breakdown table, and 2-panel residual diagnostic charts.

---

## 3. Technology Stack

- **Language**: Python 3.10+
- **Web Interface**: Streamlit (multi-page responsive UI with glass top navbar, custom space dark theme)
- **Data Engineering**: Pandas, NumPy
- **Statistical & Forecasting Engine**: SciPy (`scipy.stats`), Statsmodels (`statsmodels.tsa`), PyMannKendall (`pymannkendall`)
- **Interactive Visualizations**: Plotly (`plotly.graph_objects`, `plotly.express`, `plotly.subplots`) with custom dark space palette
- **Testing & Verification**: Pytest (61 unit, integration, statistical, and headless Streamlit AppTest tests)
- **Version Control**: Git & GitHub

---

## 4. NASA Data Sources & Provenance

1. **NASA POWER (Prediction Of Worldwide Energy Resources)**:
   - Managed by NASA Langley Research Center (LaRC).
   - Primary model: **MERRA-2** (Modern-Era Retrospective analysis for Research and Applications, Version 2) produced by the NASA Goddard GMAO.
   - Resolution: Native $0.5^\circ \text{ latitude} \times 0.625^\circ \text{ longitude}$ grid (~50 km).
   - Parameters: `T2M` (monthly mean), `T2M_MAX` (absolute monthly maximum), `T2M_MIN` (absolute monthly minimum), `PRECTOTCORR` (corrected precipitation).
   - API: `https://power.larc.nasa.gov/api/temporal/monthly/point`
   - Fill Value: `-999.0` (automatically parsed, logged, and converted to `NaN`).

2. **NASA Giovanni**:
   - Goddard Interactive Online Visualization ANd aNalysis Infrastructure (`https://giovanni.gsfc.nasa.gov/`).
   - Supports importing Giovanni exported time-series CSVs with header comments (`#`).

3. **In-App Live NASA Fetcher**:
   - Integrated live REST client querying NASA POWER for any coordinate pair worldwide.

---

## 5. Repository Structure

```
Be-An-Earth-System-Trend-Detective/
│
├── .streamlit/
│   └── config.toml                          # Fixed dark space theme and minimal toolbar
│
├── app.py                                   # Streamlit dashboard (top navbar, 7 views, sub-tabs)
├── requirements.txt                         # Pinned Python dependencies
├── README.md                                # Project overview, scientific results, setup guide
├── CHANGES_FROM_INTO.md                     # From -> Into complete changelog
├── CONTRIBUTING.md                          # Contribution guidelines, branching policy, PR process
├── TEAM_HANDOVER.md                         # Onboarding guide, architecture, work packages, roadmap
├── MATHEMATICAL_AND_TECHNICAL_DECISIONS.md  # Mathematical formulations, LaTeX proofs, design rationale
├── SOURCES_AND_REFERENCES.md                # Academic bibliography and data source documentation
├── SESSION_HANDOVER.md                      # Chronological development and session verification log
├── FINAL_METHODOLOGY_REVIEW.md              # Scientific audit & Hamed-Rao continuity verification
├── FINAL_STATISTICAL_REVIEW.md              # Prior statistical review & multiple testing analysis
├── SCIENTIFIC_AUDIT.md                      # Initial scientific audit and validation report
│
├── data/
│   ├── raw/                                 # Authentic NASA POWER CSV datasets (43-year records)
│   │   ├── nasa_power_cairo_1981_2023.csv
│   │   ├── nasa_power_alexandria_1981_2023.csv
│   │   └── nasa_power_aswan_1981_2023.csv
│   └── processed/                           # Processed and aggregated outputs
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # NASA POWER / Giovanni loaders with corrected metadata
│   ├── preprocessing.py                     # Data cleaning, deduplication, annual aggregation
│   ├── trend_analysis.py                    # MK, Hamed-Rao, Hirsch-Slack, Sen slope, ACF, FDR, extreme comparison
│   ├── rainfall_analysis.py                 # Rainfall totals, wet season, Pettitt test, reliability flags
│   ├── extremes_analysis.py                 # 5 temperature-extreme indices, percentiles, decadal counts
│   ├── anomaly_detection.py                 # WMO 1981-2010 climatology, Z-scores, spells, multi-scale SPI, compound extremes
│   └── visualization.py                     # Plotly dark theme charts (time series, rainfall, extremes, SPI, compound)
│
├── tests/
│   ├── __init__.py
│   ├── test_anomaly_detection.py            # Unit tests for temperature/precip anomalies, SPI Gamma fit, spells, N_eff
│   ├── test_preprocessing.py                # Synthetic cleaning and aggregation test cases
│   ├── test_trend_analysis.py               # Statistical validation against known baselines (MK, HR, etc.)
│   ├── test_rainfall_analysis.py            # Unit conversion, leap years, wet season, Pettitt, real data
│   ├── test_extremes_analysis.py            # Extreme indices, percentile thresholds, monthly extremes proof
│   ├── test_integration.py                  # End-to-end integration tests using real NASA datasets
│   └── test_streamlit_app.py                # Headless UI rendering tests across all 8 navbar pages
│
├── scripts/
│   ├── reproducible_scientific_audit.py     # Master reproducible audit script for regional and monthly tests
│   ├── test_aswan_methods.py                # Verification script comparing monthly mean vs native ANN
│   ├── verify_regional_hamed_rao.py         # Verification script for side-by-side Hamed-Rao statistics
│   ├── audit_regional_comparison.py         # Regional comparison audit
│   ├── verify_multiple_testing.py           # Multiple testing verification script
│   └── verify_cumulative_warming.py         # Cumulative warming metric verification
│
└── assets/                                  # Graphical assets and branding
```

---

## 6. Installation & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/ammarsami255/Be-An-Earth-System-Trend-Detective.git
cd Be-An-Earth-System-Trend-Detective
git checkout feature/rainfall-analysis
```

### 2. Set up virtual environment & install dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the comprehensive test suite
```bash
pytest -v
```
*All 52 tests pass, verifying data loading, unit conversions, Pettitt change-point test, temperature extremes, Hamed-Rao correction, multi-scalar SPI Gamma drought fitting, persistent spells, compound extremes, and headless UI rendering.*

### 4. Run the master scientific audit script
```bash
# On Windows PowerShell:
$env:PYTHONIOENCODING="utf-8"; python scripts/reproducible_scientific_audit.py
# On Linux/macOS:
python scripts/reproducible_scientific_audit.py
```

### 5. Launch the web application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 7. Application Sections & Capabilities

1. **🏠 Home**: Executive hero dashboard with live "Is the region heating up?" verdict, 4 dynamic KPI summary cards, key findings, and challenge question cards.
2. **🌡️ Temperature**:
   - *Time series*: Interactive multi-decadal time series with toggleable Theil-Sen slope, OLS regression, and Climatology Heatmap matrix.
   - *Trend statistics*: Mann-Kendall $p$-value, standardized $Z$, Kendall's $\tau$, Sen's slope 95% CI, Hamed-Rao autocorrelation correction, Hirsch-Slack seasonal test, and 12-month calendar breakdown with Benjamini-Hochberg FDR / Holm FWER control.
   - *Anomalies*: Monthly departures from the 1981–2010 WMO normal, standardized Z-scores, decadal extreme frequency shift, and top-5 ranked historical extremes.
3. **🔥 Extremes**: 5 yearly temperature extreme indices, calendar-month percentile thresholds (P90 hot / P10 cold), decade event counts, and hottest/coldest record years.
4. **🌧️ Rainfall**: Bias-corrected precipitation converted from mm/day rates to accumulated totals (mm/month, mm/year), wet-season analysis (October–March), zero-inflation diagnostics, and Pettitt (1979) change-point test.
5. **⚠️ Anomalies**: Full-featured Climate Anomaly & Drought engine:
   - *Temperature Anomalies & Spells*: Standardized Z-scores ($Z = A/\sigma$) across `T2M`, `T2M_MAX`, and `T2M_MIN`, with persistent warm/cold spell detection (duration $\ge 3$ months).
   - *Precipitation Anomalies & SPI*: Percent of Normal, wet-season departures, multi-scalar Standardized Precipitation Index (SPI-3, SPI-6, SPI-12) via Gamma MLE fitting with zero-probability correction, aridity guardrails ($q \ge 0.50$), and drought truncation bounds ($\Phi^{-1}(q)$).
   - *Compound Extremes*: Joint Compound Hot & Dry, Hot & Wet, Cold & Dry, Cold & Wet classification and decadal probability shifts.
   - *Cross-Variable & Regional*: Quenouille $N_{\text{eff}}$ serial-correlation-adjusted Pearson/Spearman correlation, lagged cross-correlation ($k \in \{-3, \dots, +3\}$), and Egypt-wide spatial sign concordance.
6. **🗺️ Regions**: Side-by-side comparative analysis of Greater Cairo, Alexandria, and Aswan across temperature, rainfall, and extreme metrics.
7. **📂 Data**: Interactive dataset inspector with filter controls, CSV download, and physical validation diagnostics.
8. **📚 Method**: Comprehensive scientific methodology guide detailing non-parametric mathematical formulas, Hamed-Rao variance derivations, Pettitt equations, SPI Gamma integrations, and NASA Earth Science citations.

---

## 8. Selected Scientific References

1. **Mann, H. B. (1945)**. Nonparametric tests against trend. *Econometrica*, 13(3), 245–259.
2. **Kendall, M. G. (1975)**. *Rank Correlation Methods*. Charles Griffin, London.
3. **Sen, P. K. (1968)**. Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63(324), 1379–1389.
4. **Hamed, K. H., & Rao, A. R. (1998)**. A modified Mann-Kendall trend test for autocorrelated data. *Journal of Hydrology*, 204(1–4), 182–196.
5. **Pettitt, A. N. (1979)**. A non-parametric approach to the change-point problem. *Applied Statistics*, 28(2), 126–135.
6. **Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982)**. Techniques of trend analysis for monthly water quality data. *Water Resources Research*, 18(1), 107–121.
7. **Benjamini, Y., & Hochberg, Y. (1995)**. Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
8. **Gelaro, R., et al. (2017)**. The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *Journal of Climate*, 30(14), 5419–5454.
9. **Karl, T. R., Nicholls, N., & Ghazi, A. (1999)**. CLIVAR/GCOS/WMO workshop on indices and indicators for climate extremes. *Climatic Change*, 42(1), 3–7.
10. **World Meteorological Organization (2017)**. *WMO Guidelines on the Calculation of Climate Normals* (WMO-No. 1203). Geneva.

---

## 9. Team & Space Apps Challenge Attribution

- **Challenge**: NASA International Space Apps Challenge 2026 — *"Be An Earth System Trend Detective!"*
- **Team**: Machine Without Learning
- **Participation**: Virtual
- **License**: MIT Open Source License
