# Sources and References Collection
**Project**: EarthTrend Explorer (NASA Space Apps Challenge 2026: *"Be An Earth System Trend Detective!"*)  
**Team**: Machine Without Learning  
**Workspace**: `D:\Space-Apps`  

---

## 1. Directly Inspected Academic Literature & Research Papers

### 1.1 Mann (1945) & Kendall (1975) — Non-Parametric Monotonic Trend Detection
- **Titles**:
  - *Nonparametric Tests Against Trend* (Mann, 1945)
  - *Rank Correlation Methods* (Kendall, 1975, 4th ed.)
- **Authors**: Henry B. Mann (1945); Maurice G. Kendall (1975)
- **Citations**:
  - Mann, H. B. (1945). *Econometrica*, 13(3), 245–259. DOI: [https://doi.org/10.2307/1907187](https://doi.org/10.2307/1907187)
  - Kendall, M. G. (1975). *Rank Correlation Methods*, 4th ed., Charles Griffin & Co. Ltd., London. ISBN: 0-85264-199-0.
- **Relevant Theory**: Derivation of the rank sum test statistic $S = \sum_{i < j} \text{sgn}(x_j - x_i)$, exact variance formulation under tied ranks $\text{Var}(S) = \frac{1}{18} [n(n-1)(2n+5) - \sum_t t(t-1)(2t+5)]$, standardized normal test statistic $Z$, and Kendall's rank correlation coefficient $\tau = \frac{S}{\frac{1}{2} n(n-1)}$.
- **Influence on Project**: Implemented directly in `src/trend_analysis.py::mann_kendall_test()` as the foundational monotonic trend test for multi-decadal temperature and rainfall series.

### 1.2 Pettitt (1979) — Non-Parametric Change-Point Problem
- **Title**: *A Non-Parametric Approach to the Change-Point Problem*
- **Author**: A. N. Pettitt
- **Journal**: *Applied Statistics* (Journal of the Royal Statistical Society: Series C), Volume 28, Issue 2, Pages 126–135
- **Year**: 1979
- **DOI / URL**: [https://doi.org/10.2307/2346729](https://doi.org/10.2307/2346729)
- **Relevant Section**: Section 2 (Theory and Distribution of $U_{t,T}$ and $K_T$), equations 2.1–2.4.
- **Information Extracted**: Formula for rank sum statistic $U_t = \sum_{i=1}^t \sum_{j=t+1}^n \text{sgn}(x_i - x_j)$, maximum test statistic $K = \max |U_t|$, and the limiting significance probability $p \approx 2 \exp(-6K^2 / (n^3 + n^2))$.
- **Why Needed**: Mann-Kendall detects monotonic trends, but cannot identify abrupt step jumps or regime shifts in hydrology.
- **Influence on Project**: Implemented in `src/rainfall_analysis.py::pettitt_change_point_test()`, identifying the abrupt statistical shift between 2014 (final year of pre-change regime) and 2015 (first year of post-change regime) in MERRA-2 rainfall over Cairo and Alexandria.

### 1.3 Karl, Nicholls, & Ghazi (1999) & Zhang et al. (2011) — Climate Extremes Indices (ETCCDI)
- **Title**: *CLIVAR/GCOS/WMO Workshop on Indices and Indicators for Climate Extremes: Workshop Summary*
- **Authors**: Thomas R. Karl, Neville Nicholls, Anver Ghazi (1999); follow-up by Zhang, X., et al. (2011), *WIREs Climate Change*, 2(6), 851–870.
- **DOI / URL**: [https://doi.org/10.1023/A:1005491526870](https://doi.org/10.1023/A:1005491526870) ; [https://doi.org/10.1002/wcc.147](https://doi.org/10.1002/wcc.147)
- **Information Extracted**: Percentile-based thresholds (TX90p, TN10p) calculated per calendar month relative to 1981–2010 normal baselines.
- **Influence on Project**: Implemented in `src/extremes_analysis.py` for monthly-resolution extreme temperature proxies ("TXx proxy", "TNn proxy", "unusually hot months", "unusually cold months"). Clarified that NASA POWER monthly maximums and minimums represent monthly-resolution extreme proxies rather than standard daily ETCCDI indices.

### 1.4 Hamed & Rao (1998) — Variance Correction for Autocorrelated Data
- **Title**: *A modified Mann-Kendall trend test for autocorrelated data*
- **Authors**: Khaled H. Hamed, A. Ramachandra Rao
- **Journal**: *Journal of Hydrology*, Volume 204, Issues 1–4, Pages 182–196
- **Year**: 1998
- **DOI / URL**: [https://doi.org/10.1016/S0022-1694(97)00125-X](https://doi.org/10.1016/S0022-1694(97)00125-X)
- **Relevant Section**: Section 3 (Variance of the Mann-Kendall test statistic), Eq. 10–13.
- **Information Extracted**: Effective sample size ratio $\eta = n/n^*$, rank autocorrelation of detrended series $\rho_i$, and critical threshold condition $|\rho_i| > \frac{Z_{1-\alpha/2}}{\sqrt{n}}$.
- **Variance Inflation vs Deflation**: When positive autocorrelation exists ($\rho_i > 0$), $\eta > 1$ (variance inflation, reducing false trends). When negative autocorrelation exists ($\rho_i < 0$), $\eta < 1$ (variance deflation, increasing $|Z^*|$). In Alexandria annual mean $T_{\text{2m}}$, lag-12 negative autocorrelation ($\rho_{12} = -0.30129$) yields $\eta = 0.780520$, deflating variance from $9130.3$ to $7126.4$ and raising $Z$ from $+5.50$ to $Z^* = +6.23$.
- **Influence on Project**: Implemented in `src/trend_analysis.py::hamed_rao_modified_mann_kendall()`. Original and modified statistics are presented side by side in the UI.

### 1.5 Hirsch, Slack, & Smith (1982) & Hirsch & Slack (1984) — Seasonal Mann-Kendall Test
- **Titles**:
  - *Techniques of Trend Analysis for Monthly Water Quality Data* (Hirsch, Slack, & Smith, 1982)
  - *A Nonparametric Test for Trend in Hydrological Data With Serial Correlation* (Hirsch & Slack, 1984)
- **Authors**: Robert M. Hirsch, James R. Slack, Richard A. Smith
- **Citations**:
  - Hirsch et al. (1982). *Water Resources Research*, 18(1), 107–121. DOI: [https://doi.org/10.1029/WR018i001p00107](https://doi.org/10.1029/WR018i001p00107)
  - Hirsch & Slack (1984). *Water Resources Research*, 20(6), 727–732. DOI: [https://doi.org/10.1029/WR020i006p00727](https://doi.org/10.1029/WR020i006p00727)
- **Relevant Section**: Seasonal Kendall test for trends, seasonal stratification, and inter-seasonal covariance formulation.
- **Influence on Project**: Implemented in `src/trend_analysis.py::seasonal_mann_kendall_test()`.

### 1.6 Sen (1968) & Theil (1950) — Non-Parametric Median Slope Estimator
- **Title**: *Estimates of the Regression Coefficient Based on Kendall's Tau*
- **Author**: Pranab Kumar Sen
- **Journal**: *Journal of the American Statistical Association*, Volume 63, Issue 324, Pages 1379–1389
- **Year**: 1968
- **DOI / URL**: [https://doi.org/10.1080/01621459.1968.10480934](https://doi.org/10.1080/01621459.1968.10480934)
- **Influence on Project**: Implemented in `src/trend_analysis.py::compute_sens_slope()`.

### 1.7 Gilbert (1987) — Analytical Confidence Intervals for Sen's Slope
- **Title**: *Statistical Methods for Environmental Pollution Monitoring*
- **Author**: Richard O. Gilbert
- **Publisher**: Van Nostrand Reinhold, New York
- **Year**: 1987
- **Information Extracted**: Analytical rank index bounds $M_1 = \lfloor(N - C_\alpha)/2\rfloor$ and $M_2 = \lceil(N + C_\alpha)/2\rceil + 1$ using $C_\alpha = Z_{1-\alpha/2}\sqrt{\text{Var}(S)}$.
- **Influence on Project**: Analytical confidence intervals for Sen's slope in `src/trend_analysis.py`.

### 1.8 Benjamini & Hochberg (1995) — False Discovery Rate Control
- **Title**: *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*
- **Authors**: Yoav Benjamini, Yosef Hochberg
- **Journal**: *Journal of the Royal Statistical Society: Series B*, Volume 57, Issue 1, Pages 289–300
- **Year**: 1995
- **DOI / URL**: [https://doi.org/10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- **Influence on Project**: Controls FDR across 12 individual calendar month trend tests in `src/trend_analysis.py::compute_calendar_month_breakdown()`.

### 1.9 Holm (1979) — Sequentially Rejective Bonferroni Test
- **Title**: *A Simple Sequentially Rejective Multiple Test Procedure*
- **Author**: Sture Holm
- **Journal**: *Scandinavian Journal of Statistics*, Volume 6, Issue 2, Pages 65–70
- **Year**: 1979
- **URL**: [https://www.jstor.org/stable/4615733](https://www.jstor.org/stable/4615733)
- **Influence on Project**: Conservative FWER control in `compute_calendar_month_breakdown()`.

### 1.10 McKee, Doesken, & Kleist (1993) & Edwards & McKee (1997) — Standardized Precipitation Index (SPI)
- **Title**: *The Relationship of Drought Frequency and Duration to Time Scales*
- **Authors**: Thomas B. McKee, Nolan J. Doesken, John Kleist
- **Conference / Report**: *Proceedings of the 8th Conference on Applied Climatology*, Anaheim, CA, Pages 179–184 (1993); followed by Edwards, D. C. and McKee, T. B. (1997), *Characteristics of 20th Century Drought in the United States at Multiple Time Scales*, Atmospheric Science Paper No. 634.
- **Information Extracted**: Multi-scalar accumulation ($k = 3, 6, 12$ months), 2-parameter Gamma fitting, mixed cumulative probability $H(x) = q + (1-q)G(x)$, and standard normal inversion $\text{SPI} = \Phi^{-1}(H(x))$.
- **Influence on Project**: Implemented in `src/anomaly_detection.py::compute_spi()`.

### 1.11 Thom (1958) — Gamma Distribution Parameter Estimation for Precipitation
- **Title**: *A Note on the Gamma Distribution*
- **Author**: H. C. S. Thom
- **Journal**: *Monthly Weather Review*, Volume 86, Issue 4, Pages 117–122
- **Year**: 1958
- **DOI / URL**: [https://doi.org/10.1175/1520-0493(1958)086<0117:ANOTGD>2.0.CO;2](https://doi.org/10.1175/1520-0493(1958)086<0117:ANOTGD>2.0.CO;2)
- **Information Extracted**: Maximum likelihood approximations for Gamma shape ($\alpha$) and scale ($\beta$) parameters from positive precipitation observations: $\hat{\alpha} \approx \frac{1 + \sqrt{1 + 4A/3}}{4A}$ where $A = \ln(\bar{x}) - \frac{1}{n} \sum \ln(x)$.
- **Influence on Project**: Foundation of Gamma distribution fitting in `src/anomaly_detection.py::compute_spi()`.

### 1.12 Gelaro et al. (2017) — NASA MERRA-2 Reanalysis System
- **Title**: *The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2)*
- **Authors**: Ronald Gelaro, Will McCarty, Max J. Suárez, Ricardo Todling, et al.
- **Journal**: *Journal of Climate*, Volume 30, Issue 14, Pages 5419–5454
- **Year**: 2017
- **DOI / URL**: [https://doi.org/10.1175/JCLI-D-16-0758.1](https://doi.org/10.1175/JCLI-D-16-0758.1)
- **Information Extracted**: System architecture of NASA GMAO MERRA-2 reanalysis, 0.5° × 0.625° atmospheric model grid, aerosol and satellite radiance assimilation, and precipitation observation-correction algorithms underlying `PRECTOTCORR`.
- **Influence on Project**: Scientific validation of data lineage, spatial resolution, and variable semantics in `src/data_loader.py`.

### 1.13 Quenouille (1947) & Pyper & Peterman (1998) — Effective Degrees of Freedom for Autocorrelated Cross-Correlation
- **Title**: *Comparison of methods to account for autocorrelation in correlation analyses of fish data*
- **Authors**: Brian J. Pyper, Randall M. Peterman (based on Quenouille, M. H., 1947)
- **Journal**: *Canadian Journal of Fisheries and Aquatic Sciences*, Volume 55, Issue 9, Pages 2127–2140
- **Year**: 1998
- **DOI / URL**: [https://doi.org/10.1139/f98-104](https://doi.org/10.1139/f98-104)
- **Information Extracted**: Effective degrees of freedom / sample size: $N_{\text{eff}} \approx N \frac{1 - r_{1, x} r_{1, y}}{1 + r_{1, x} r_{1, y}}$, correcting $t$-statistics when cross-correlating autocorrelated time series.
- **Influence on Project**: Implemented in `src/anomaly_detection.py::cross_variable_anomaly_correlation()`.

### 1.11 Hyndman & Athanasopoulos (2018) & Hyndman & Koehler (2006) — Time Series Forecasting & MASE
- **Titles**:
  - *Forecasting: Principles and Practice* (2nd ed., 2018)
  - *Another look at measures of forecast accuracy* (2006)
- **Authors**: Rob J. Hyndman, George Athanasopoulos; Anne B. Koehler
- **Citations**:
  - Hyndman, R.J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*, 2nd edition, OTexts: Melbourne, Australia. URL: [https://otexts.com/fpp2/](https://otexts.com/fpp2/)
  - Hyndman, R.J., & Koehler, A.B. (2006). *International Journal of Forecasting*, 22(4), 679–688. DOI: [https://doi.org/10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
- **Information Extracted**: Seasonal naive formulation, scaling of seasonal multi-step error variances, and Mean Absolute Scaled Error (MASE):
  $$\text{MASE} = \frac{\text{MAE}}{\frac{1}{T-m}\sum_{t=m+1}^T |y_t - y_{t-m}|}$$
- **Influence on Project**: Implemented in `src/forecasting.py` for Seasonal Naive baseline benchmarking, expanding-window tournament selection, and prediction interval generation.

### 1.12 Box, Jenkins, Reinsel, & Ljung (2015) — Seasonal Autoregressive Models (SARIMA)
- **Title**: *Time Series Analysis: Forecasting and Control* (5th ed.)
- **Authors**: George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung
- **Publisher**: John Wiley & Sons, Hoboken, NJ. ISBN: 978-1-118-67502-1.
- **Information Extracted**: Multiplicative seasonal autoregressive integrated moving average $\text{SARIMA}(p,d,q) \times (P,D,Q)_s$ model specification and Kalman filter analytical prediction covariance bounds.
- **Influence on Project**: Implemented in `src/forecasting.py::forecast_sarima()`.

### 1.13 Holt (1957) & Winters (1960) — Exponential Smoothing with Seasonality (ETS)
- **Titles**:
  - *Forecasting seasonals and trends by exponentially weighted moving averages* (Holt, 1957)
  - *Forecasting sales by exponentially weighted moving averages* (Winters, 1960)
- **Citations**:
  - Holt, C.C. (1957). ONR Memorandum No. 52. Reprinted in *International Journal of Forecasting*, 20(1), 5–10 (2004). DOI: [https://doi.org/10.1016/j.ijforecast.2003.09.015](https://doi.org/10.1016/j.ijforecast.2003.09.015)
  - Winters, P.R. (1960). *Management Science*, 6(3), 324–342. DOI: [https://doi.org/10.1287/mnsc.6.3.324](https://doi.org/10.1287/mnsc.6.3.324)
- **Information Extracted**: Additive level, trend, and seasonal smoothing equations for series with constant seasonal amplitude.
- **Influence on Project**: Implemented in `src/forecasting.py::forecast_exponential_smoothing()` with 1,000 Monte Carlo innovations simulations.

---

## 2. Directly Inspected Source Code Repositories

### 2.1 PyMannKendall
- **Repository**: `https://github.com/mmhs013/pyMannKendall`
- **Version Inspected**: `1.4.3`
- **Functions Verified**: `hamed_rao_modification_test()`, `__acf()`, `seasonal_test()`, `sens_slope()`.

### 2.2 SciPy (`scipy.stats`)
- **Version**: `scipy >= 1.11.0`
- **Functions Used**: `scipy.stats.rankdata`, `scipy.stats.norm`, `scipy.stats.linregress`, `scipy.stats.gamma`.

---

## 3. NASA Earth Science Data Platforms & API References

### 3.1 NASA POWER Project
- **Organization**: NASA Langley Research Center (LaRC)
- **Platform URL**: [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/)
- **Underlying Model**: MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2) produced by NASA GMAO.
- **Parameters Queried**:
  - `T2M`: Temperature at 2 Meters (°C) — monthly mean
  - `T2M_MAX`: Maximum Temperature at 2 Meters (°C) — absolute monthly extreme (peak hour of month)
  - `T2M_MIN`: Minimum Temperature at 2 Meters (°C) — absolute monthly extreme (peak minimum hour of month)
  - `PRECTOTCORR`: Precipitation Corrected (mm/day) — bias-corrected precipitation flux
- **Role in Project**: Primary empirical data source for multi-decadal analysis (1981–2023) across Greater Cairo, Alexandria, and Aswan.

### 3.2 NASA Giovanni
- **Organization**: NASA GES DISC
- **Platform URL**: [https://giovanni.gsfc.nasa.gov/](https://giovanni.gsfc.nasa.gov/)
- **Role in Project**: Supported format in `src/data_loader.py::load_giovanni_csv()`.

---

## 4. Climatological Standards & Reference Manuals

### 4.1 World Meteorological Organization (WMO) Climate Normals (WMO-No. 49 & WMO-No. 1203)
- **Titles**:
  - *Technical Regulations, Volume I: General Meteorological Standards and Recommended Practices* (WMO-No. 49)
  - *WMO Guidelines on the Calculation of Climate Normals* (WMO-No. 1203, 2017)
- **Standard vs. Provisional Normals**:
  - **Standard Climate Normal**: Defined as 30 consecutive calendar years over internationally agreed periods (e.g. 1981–2010). Used as the primary baseline across EarthTrend Explorer.
  - **Provisional Climate Normal**: WMO regulations explicitly allow the calculation and use of Provisional Climate Normals when at least 10 years of homogeneous data are available (WMO-No. 49 Section B.1 Climatology; WMO-No. 1203 Section 2.2). This justifies retaining the $\ge 10$-year threshold in `src/anomaly_detection.py` for provisional baselines.

### 4.2 World Meteorological Organization (2012) — Standardized Precipitation Index User Guide
- **Title**: *Standardized Precipitation Index User Guide* (M. Svoboda, M. Hayes, D. Wood)
- **Publication**: WMO-No. 1090, World Meteorological Organization, Geneva, Switzerland
- **Year**: 2012
- **Information Extracted**: Standard SPI classification brackets ($\ge +2.0$ Extremely Wet, $+1.5$ to $+2.0$ Very Wet, $+1.0$ to $+1.5$ Moderately Wet, $-1.0$ to $+1.0$ Near Normal, $-1.5$ to $-1.0$ Moderately Dry, $-2.0$ to $-1.5$ Severely Dry, $\le -2.0$ Extremely Dry).
- **Influence on Project**: Color coding and classification categories in `src/anomaly_detection.py` and `src/visualization.py`.

---

## 5. Summary Table of Reference Influence

| Reference | Inspected Section | Implementation Impact | Project File |
| :--- | :--- | :--- | :--- |
| **Mann (1945) & Kendall (1975)** | Rank sum $S$ & tie correction | Foundational non-parametric monotonic trend test | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Pettitt (1979)** | Section 2, Eq. 2.1–2.4 | Non-parametric abrupt change-point test | [`src/rainfall_analysis.py`](file:///D:/Space-Apps/src/rainfall_analysis.py) |
| **ETCCDI / Karl (1999)** | TX90p / TN10p | Percentile-based monthly temperature extreme proxies | [`src/extremes_analysis.py`](file:///D:/Space-Apps/src/extremes_analysis.py) |
| **Hamed & Rao (1998)** | Section 3.2, Eq. 10–13 | Autocorrelation-corrected variance $\text{Var}^*(S)$ | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Hirsch et al. (1982, 1984)** | Pages 108–110; Eq. 1–8 | Seasonal Mann-Kendall omnibus test & covariance | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Sen (1968)** | Eq. 2.1–2.8 | Non-parametric median slope estimator | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Gilbert (1987)** | Chapter 16, pp. 217–219 | Analytical rank-based 95% confidence intervals for Sen's slope | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Benjamini & Hochberg (1995)** | Section 1–3 | False Discovery Rate ($q < 0.05$) across 12 calendar months | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Holm (1979)** | Section 1–2 | Family-Wise Error Rate control across 12 calendar months | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **McKee et al. (1993)** | Multi-scale Gamma PDF | Standardized Precipitation Index (SPI-3, SPI-6, SPI-12) | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **Thom (1958)** | Eq. 11–14 | Maximum likelihood approximations for Gamma shape/scale | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **Gelaro et al. (2017)** | MERRA-2 System Description | Data lineage, reanalysis grids, parameter definitions | [`src/data_loader.py`](file:///D:/Space-Apps/src/data_loader.py) |
| **NASA POWER API** | Monthly Point Service | Data ingestion, metadata header parsing, coordinate mapping | [`src/data_loader.py`](file:///D:/Space-Apps/src/data_loader.py) |
| **WMO-No. 49 & 1203** | Standard & Provisional Normals | 1981–2010 baseline & $\ge 10$-year provisional normal rule | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **WMO-No. 1090 (2012)** | Drought categories | Standard SPI threshold classification brackets | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **Pyper & Peterman (1998)** | Effective Sample Size | Autocorrelation-adjusted degrees of freedom $N_{\text{eff}}$ for cross-correlation | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **Hyndman & Athanasopoulos (2018)** | Chapters 5, 8, 9 | Seasonal Naive benchmark, Exponential Smoothing (ETS), and prediction intervals | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
| **Hyndman & Koehler (2006)** | Section 3.2 | Mean Absolute Scaled Error (MASE) for out-of-sample backtesting evaluation | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
| **Box, Jenkins et al. (2015)** | Chapter 9 | Seasonal ARIMA (SARIMAX) state-space formulation and Kalman filter covariance | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
| **Holt (1957) & Winters (1960)** | Additive Holt-Winters | Linear trend and additive seasonal smoothing with Monte Carlo simulation bounds | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
