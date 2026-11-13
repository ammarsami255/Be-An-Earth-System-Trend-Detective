# EarthTrend Explorer 🌍

**NASA Space Apps Challenge 2026**  
**Challenge**: *"Be An Earth System Trend Detective!"*  
**Team**: Machine Without Learning  
**Participation**: Virtual  
**Project**: EarthTrend Explorer  

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-FF4B4B.svg)](https://streamlit.io/)
[![PyMannKendall](https://img.shields.io/badge/PyMannKendall-1.4.3-green.svg)](https://github.com/mmhs013/pymannkendall)
[![NASA Data](https://img.shields.io/badge/NASA_Earth_Science-MERRA--2-0B3D91.svg)](https://power.larc.nasa.gov/)
[![Tests Passing](https://img.shields.io/badge/pytest-14%2F14_Passed-success.svg)](https://docs.pytest.org/)

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
   Evaluates null hypotheses ($H_0$: no monotonic trend) using the **Mann-Kendall test** and **Hirsch-Slack Seasonal Mann-Kendall test**, while explicitly diagnosing serial autocorrelation and seasonality.

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
| **Monthly Breakdown** | 8 of 12 months significant | Significant in Feb, May, Jul–Dec (Nov fastest: $+0.504^\circ\text{C/dec}$); Jan, Mar, Apr, Jun non-sig |
| **Hamed-Rao Modified MK** | $Z^* = +5.35$, $p = 8.89 \times 10^{-8}$ | Confirmed robust against red noise: detrended residual $r_1 = -0.248$ (within white noise bound) |

---

## 3. Technology Stack

- **Language**: Python 3.10+
- **Web Interface**: Streamlit
- **Data Engineering**: Pandas, NumPy
- **Statistical Engine**: SciPy (`scipy.stats`), PyMannKendall (`pymannkendall`)
- **Interactive Visualizations**: Plotly (`plotly.graph_objects`, `plotly.express`)
- **Testing & Verification**: Pytest, Streamlit AppTest framework
- **Version Control**: Git

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
EarthTrend-Explorer/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Scientific documentation and user guide
│
├── data/
│   ├── raw/                    # Authentic NASA CSV datasets
│   │   ├── nasa_power_cairo_1981_2023.csv
│   │   ├── nasa_power_alexandria_1981_2023.csv
│   │   └── nasa_power_aswan_1981_2023.csv
│   └── processed/              # Processed / aggregated exports
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # NASA POWER / Giovanni / generic CSV loaders
│   ├── preprocessing.py        # Data cleaning, deduplication, annual aggregation
│   ├── trend_analysis.py       # Mann-Kendall, Sen's slope, Hirsch-Slack, ACF
│   ├── anomaly_detection.py    # 1981-2010 WMO baseline climatology & Z-scores
│   └── visualization.py        # Plotly charts with NASA design system
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Controlled synthetic test cases & cleaning tests
│   ├── test_trend_analysis.py  # Statistical validation (monotonic, noise, CIs)
│   └── test_integration.py     # End-to-end integration test with real Cairo data
│
└── assets/                     # Logos, diagrams, and project assets
```

---

## 6. Installation & Execution

### 1. Clone the repository
```bash
git clone https://github.com/machine-without-learning/EarthTrend-Explorer.git
cd EarthTrend-Explorer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the automated test suite
```bash
pytest tests -v
```
*All 14 tests verify mathematical algorithms against controlled synthetic baselines and real NASA data.*

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
4. **🔬 Trend Analysis Engine**: Statistical hypothesis testing, Mann-Kendall $p$-value, standardized $Z$, Kendall's $\tau$, Sen's slope 95% CI, Hirsch-Slack seasonal test, and Autocorrelation Function (ACF) diagnostic.
5. **🗺️ Regional Comparison**: Comparative analysis across Egyptian climate regimes (Cairo urban basin, Alexandria Mediterranean coast, and Aswan hyper-arid desert).
6. **⚠️ Anomaly Detection**: Monthly departures from the 1981–2010 WMO normal, standardized Z-scores ($Z = A_t / \sigma_m$), decadal extreme frequency shift, and top-5 ranked historical extremes.
7. **📚 Methodology & Sources**: Detailed mathematical formulations, academic references, data limitations, and distinction between statistical and practical significance.

---

## 8. Scientific References

1. **Mann, H. B. (1945)**. Nonparametric tests against trend. *Econometrica*, 13(3), 245–259.
2. **Kendall, M. G. (1975)**. *Rank Correlation Methods*. Charles Griffin, London.
3. **Sen, P. K. (1968)**. Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63(324), 1379–1389.
4. **Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982)**. Techniques of trend analysis for monthly water quality data. *Water Resources Research*, 18(1), 107–121.
5. **Gelaro, R., et al. (2017)**. The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). *Journal of Climate*, 30(14), 5419–5454.

---

## 9. Team & Space Apps Challenge Attribution

- **Challenge**: NASA International Space Apps Challenge 2026 — *"Be An Earth System Trend Detective!"*
- **Team**: Machine Without Learning
- **License**: MIT Open Source License
