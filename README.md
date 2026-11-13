# EarthTrend Explorer 🌍

**NASA Space Apps Challenge 2026**  
**Challenge**: *"Be An Earth System Trend Detective!"*  
**Team**: Machine Without Learning  
**Participation**: Virtual  
**Project**: EarthTrend Explorer  

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyMannKendall](https://img.shields.io/badge/PyMannKendall-1.4.3-green.svg)](https://github.com/mmhs013/pymannkendall)
[![NASA Data](https://img.shields.io/badge/NASA_Earth_Science-MERRA--2-0B3D91.svg)](https://power.larc.nasa.gov/)
[![Tests Passing](https://img.shields.io/badge/pytest-20%2F20_Passed-success.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Key Documentation & Technical Handover Guides

For researchers, contributors, and evaluators, the repository includes comprehensive technical, mathematical, and onboarding documentation:

- 📋 [**TEAM_HANDOVER.md**](TEAM_HANDOVER.md): **Start Here!** Complete team onboarding, modular architecture overview, independent work packages, integration strategy, and development roadmap.
- 📐 [**MATHEMATICAL_AND_TECHNICAL_DECISIONS.md**](MATHEMATICAL_AND_TECHNICAL_DECISIONS.md): Full mathematical derivations, formulas (LaTeX), test statistics ($S$, $\tau$, $Z$, $\text{Var}^*(S)$, Theil-Sen, Benjamini-Hochberg FDR, Holm FWER), and architectural design rationales.
- 📚 [**SOURCES_AND_REFERENCES.md**](SOURCES_AND_REFERENCES.md): Complete scientific bibliography, academic papers, official NASA APIs, and WMO climatological guidelines.
- 🛠️ [**CONTRIBUTING.md**](CONTRIBUTING.md): Git branch policy, conventional commits, PR guidelines, code quality standards, and testing checklist.
- 📝 [**SESSION_HANDOVER.md**](SESSION_HANDOVER.md): Chronological development log, verified calculations, bug resolutions, and session audit history.
- 🔬 [**FINAL_METHODOLOGY_REVIEW.md**](FINAL_METHODOLOGY_REVIEW.md): Independent scientific audit, side-by-side Hamed-Rao continuity verification, and regional comparisons.

---

## 1. Project Overview

EarthTrend Explorer is a scientifically rigorous interactive data-analysis web application designed to investigate multi-decadal environmental trends using authentic **NASA Earth Science** observations and reanalysis data.

Rather than acting as a cosmetic dashboard with hardcoded or fabricated values, EarthTrend Explorer is a true **statistical laboratory** built to answer four fundamental scientific questions:

1. **What is changing?**  
   Identifies and validates physical environmental variables (e.g., 2-meter air temperature `T2M`, daily maximum/minimum temperatures `T2M_MAX`/`T2M_MIN`, and corrected precipitation `PRECTOTCORR`), ensuring unit consistency, boundary checks, and transparent data quality auditing.

2. **Where is it changing?**  
   Locates the exact geographic domain, coordinates, and regional elevation context (e.g., Greater Cairo urban basin, Alexandria Mediterranean coastline, and Aswan hyper-arid desert).

3. **How much is it changing?**  
   Quantifies the rate of change using the robust non-parametric **Theil-Sen median slope estimator** with analytical 95% confidence intervals, alongside supplementary Ordinary Least Squares (OLS) linear regression.

4. **Is the observed trend statistically significant?**  
   Evaluates null hypotheses ($H_0$: no monotonic trend) using the **Mann-Kendall test**, **Hamed-Rao Modified Mann-Kendall test** (accounting for serial autocorrelation), and **Hirsch-Slack Seasonal Mann-Kendall test**, while explicitly diagnosing serial autocorrelation and seasonality.

---

## 2. Key Findings: Greater Cairo Benchmark Experiment (1981–2023)

We investigated continuous 43-year monthly 2-meter air temperature data (`T2M`) for **Greater Cairo, Egypt** ($30.0444^\circ\text{N}, 31.2357^\circ\text{E}$, elevation: $135.44\text{ m}$) from NASA MERRA-2 via NASA POWER:

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Observation Record** | 1981–2023 (43 years, 516 months) | Complete historical record, zero missing values, single MERRA-2 grid cell |
| **Annual MK Trend ($n=43$)** | **Increasing (Upward)** | Null hypothesis rejected ($H_1$ accepted, $S = +512.0$) |
| **Annual p-value** | $8.89 \times 10^{-8}$ | Highly statistically significant ($p < 0.001 \ll \alpha = 0.05$) |
| **Annual Standardized Z** | $+5.35$ | Well above the $99.9\%$ critical threshold ($Z_{0.999} = 3.29$) |
| **Kendall's $\tau$** | $+0.567$ | Substantial positive monotonic rank correlation |
| **Sen's Slope Rate** | **$+0.0293^\circ\text{C / year}$** | **$+0.293^\circ\text{C / decade}$** warming rate |
| **Sen's 95% CI** | $[+0.0202, +0.0389]^\circ\text{C / year}$ | Entire confidence interval is strictly positive ($[+0.202, +0.389]^\circ\text{C/dec}$) |
| **OLS Linear Rate** | $+0.0305^\circ\text{C / year}$ ($R^2 = 0.503$) | Strong linear agreement with robust median slope ($p = 9.99 \times 10^{-8}$) |
| **Net Cumulative Shift** | **$+1.23^\circ\text{C}$** | Net warming over 42 elapsed years ($+1.26^\circ\text{C}$ over 43 steps) |
| **Seasonal MK ($n=516$)** | $Z = +8.05$, $p < 10^{-15}$ | Multi-decadal upward trajectory ($S = +2665$, Sen's slope $+0.303^\circ\text{C/decade}$) |
| **Monthly Breakdown** | 6 of 12 months significant (FDR) | Under Benjamini-Hochberg FDR control ($q < 0.05$), 6 months (Jul–Dec) are significant; Feb & May are nominally significant only ($p < 0.05$); Jan, Mar, Apr, Jun show no detectable trend |
| **Hamed-Rao Modified MK** | $Z^* = +5.35$, $p = 8.89 \times 10^{-8}$ | Detrended rank autocorrelations fall within 95% critical bound ($\pm 0.299$); correction factor $n/n^* = 1.000$ (insufficient evidence to detect red noise at $n=43$) |

---

## 3. Technology Stack

- **Language**: Python 3.10+
- **Web Interface**: Streamlit (multi-page responsive UI)
- **Data Engineering**: Pandas, NumPy
- **Statistical Engine**: SciPy (`scipy.stats`), PyMannKendall (`pymannkendall`)
- **Interactive Visualizations**: Plotly (`plotly.graph_objects`, `plotly.express`)
- **Testing & Verification**: Pytest (unit, statistical, integration, and UI testing)
- **Version Control**: Git & GitHub

---

## 4. NASA Data Sources & Provenance

1. **NASA POWER (Prediction Of Worldwide Energy Resources)**:
   - Managed by NASA Langley Research Center (LaRC).
   - Primary model: **MERRA-2** (Modern-Era Retrospective analysis for Research and Applications, Version 2) produced by the NASA Goddard Global Modeling and Assimilation Office (GMAO).
   - Resolution: Native $0.5^\circ \text{ latitude} \times 0.625^\circ \text{ longitude}$ grid (~50 km).
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
├── app.py                                   # Streamlit interactive application (7 navigation views)
├── requirements.txt                         # Pinned Python dependencies
├── README.md                                # Project overview, scientific results, and setup guide
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
│   ├── data_loader.py                       # NASA POWER, Giovanni, and generic CSV loaders
│   ├── preprocessing.py                     # Data cleaning, validation, annual aggregation
│   ├── trend_analysis.py                    # MK, Hamed-Rao, Hirsch-Slack, Sen's slope, ACF, FDR
│   ├── anomaly_detection.py                 # WMO 1981-2010 climatology, Z-scores, extreme analysis
│   └── visualization.py                     # Plotly interactive charts with NASA design palette
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py                # Synthetic data cleaning and aggregation test cases
│   ├── test_trend_analysis.py               # Statistical validation against known baselines (MK, HR, etc.)
│   ├── test_integration.py                  # End-to-end integration tests using real NASA datasets
│   └── test_streamlit_app.py                # Headless UI rendering tests across all 7 views
│
├── scripts/
│   ├── verify_regional_hamed_rao.py         # Verification script for side-by-side Hamed-Rao statistics
│   ├── test_regional_hamed_rao.py           # Verification script for regional continuity checks
│   └── download_nasa_data.py                # Automated data download utility from NASA POWER API
│
└── assets/                                  # Graphical assets and branding
```

---

## 6. Installation & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/ammarsami255/Be-An-Earth-System-Trend-Detective.git
cd Be-An-Earth-System-Trend-Detective
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
*All 20 tests verify data cleaning, statistical formulations (including Hamed-Rao red-noise correction), end-to-end integrations, and headless Streamlit UI rendering.*

### 4. Launch the web application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 7. Application Sections & Capabilities

1. **🏠 Home & Project Overview**: Executive summary, mission statement, the 4 scientific questions, and Cairo 43-year benchmark highlights.
2. **📂 Data Explorer**: Complete transparency into raw dataset headers, geographic coordinates, elevation, missing value counts, and variable validation audit.
3. **📈 Interactive Time Series**: Interactive time series with toggleable Theil-Sen robust slope, OLS regression line, and Year-vs-Month Climatology Heatmap.
4. **🔬 Trend Analysis Engine**: Statistical hypothesis testing, Mann-Kendall $p$-value, standardized $Z$, Kendall's $\tau$, Sen's slope 95% CI, Hamed-Rao autocorrelation correction, Hirsch-Slack seasonal test, Benjamini-Hochberg FDR / Holm FWER correction, and Autocorrelation Function (ACF) diagnostic.
5. **🗺️ Regional Comparison**: Comparative analysis across Egyptian climate regimes (Cairo urban basin, Alexandria Mediterranean coast, and Aswan hyper-arid desert) featuring side-by-side original Mann-Kendall and serial-correlation corrected Hamed-Rao statistics.
6. **⚠️ Anomaly Detection**: Monthly departures from the 1981–2010 WMO normal, standardized Z-scores ($Z = A_t / \sigma_m$), decadal extreme frequency shift, and top-5 ranked historical extremes.
7. **📚 Methodology & Sources**: Detailed mathematical formulations, academic references, data limitations, and distinction between statistical and practical significance.

---

## 8. Selected Scientific References

1. **Mann, H. B. (1945)**. Nonparametric tests against trend. *Econometrica*, 13(3), 245–259.
2. **Kendall, M. G. (1975)**. *Rank Correlation Methods*. Charles Griffin, London.
3. **Sen, P. K. (1968)**. Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63(324), 1379–1389.
4. **Hamed, K. H., & Rao, A. R. (1998)**. A modified Mann-Kendall trend test for autocorrelated data. *Journal of Hydrology*, 204(1–4), 182–196.
5. **Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982)**. Techniques of trend analysis for monthly water quality data. *Water Resources Research*, 18(1), 107–121.
6. **Benjamini, Y., & Hochberg, Y. (1995)**. Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
7. **Gelaro, R., et al. (2017)**. The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *Journal of Climate*, 30(14), 5419–5454.
8. **World Meteorological Organization (2017)**. *WMO Guidelines on the Calculation of Climate Normals* (WMO-No. 1203). Geneva.

---

## 9. Team & Space Apps Challenge Attribution

- **Challenge**: NASA International Space Apps Challenge 2026 — *"Be An Earth System Trend Detective!"*
- **Team**: Machine Without Learning
- **Participation**: Virtual
- **License**: MIT Open Source License
