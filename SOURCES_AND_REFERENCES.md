# Sources and References Collection
**Project**: EarthTrend Explorer (NASA Space Apps Challenge 2026: *"Be An Earth System Trend Detective!"*)  
**Team**: Machine Without Learning  
**Workspace**: `D:\Space-Apps`  

---

## 1. Directly Inspected Academic Literature & Research Papers

### 1.1 Hamed & Rao (1998) — Variance Correction for Autocorrelated Data
- **Title**: *A modified Mann-Kendall trend test for autocorrelated data*
- **Authors**: Khaled H. Hamed, A. Ramachandra Rao
- **Journal**: *Journal of Hydrology*, Volume 204, Issues 1–4, Pages 182–196
- **Year**: 1998
- **DOI / URL**: [https://doi.org/10.1016/S0022-1694(97)00125-X](https://doi.org/10.1016/S0022-1694(97)00125-X)
- **Relevant Section**: Section 3 (Variance of the Mann-Kendall test statistic), Eq. 10–13.
- **Information Extracted**: Formula for effective sample size ratio $n/n^*$, rank autocorrelation of detrended series $\rho_i$, and the critical threshold condition $|\rho_i| > \frac{Z_{1-\alpha/2}}{\sqrt{n}}$.
- **Why Needed**: Annual and monthly climate time series frequently exhibit temporal persistence, violating the independence assumption of the standard Mann-Kendall test.
- **Influence on Project**: Implemented directly in `src/trend_analysis.py::hamed_rao_modified_mann_kendall()`.
- **Related Project Files**: [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py), [`tests/test_trend_analysis.py`](file:///D:/Space-Apps/tests/test_trend_analysis.py), [`FINAL_METHODOLOGY_REVIEW.md`](file:///D:/Space-Apps/FINAL_METHODOLOGY_REVIEW.md).
- **Inspection Type**: Directly inspected and verified against analytical formulas.

### 1.2 Hirsch, Slack, & Smith (1982) — Seasonal Mann-Kendall Test
- **Title**: *Techniques of Trend Analysis for Monthly Water Quality Data*
- **Authors**: Robert M. Hirsch, James R. Slack, Richard A. Smith
- **Journal**: *Water Resources Research*, Volume 18, Issue 1, Pages 107–121
- **Year**: 1982
- **DOI / URL**: [https://doi.org/10.1029/WR018i001p00107](https://doi.org/10.1029/WR018i001p00107)
- **Relevant Section**: Seasonal Kendall test for trends, pages 108–110.
- **Information Extracted**: Calculation of seasonal score $S_g$, additive variance under inter-seasonal independence $\text{Var}(S_{\text{tot}}) = \sum \text{Var}(S_g)$, and seasonal Sen's slope.
- **Why Needed**: Monthly environmental time series exhibit large 12-month cyclic temperature swings that confound standard linear models.
- **Influence on Project**: Implemented in `src/trend_analysis.py::seasonal_mann_kendall_test()` and qualified for inter-seasonal covariance assumptions.
- **Related Project Files**: [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py), [`app.py`](file:///D:/Space-Apps/app.py).
- **Inspection Type**: Directly inspected.

### 1.3 Sen (1968) & Theil (1950) — Non-Parametric Median Slope Estimator
- **Title**: *Estimates of the Regression Coefficient Based on Kendall's Tau*
- **Author**: Pranab Kumar Sen
- **Journal**: *Journal of the American Statistical Association*, Volume 63, Issue 324, Pages 1379–1389
- **Year**: 1968
- **DOI / URL**: [https://doi.org/10.1080/01621459.1968.10480934](https://doi.org/10.1080/01621459.1968.10480934)
- **Relevant Section**: Section 2 (The Estimator), pages 1380–1383.
- **Information Extracted**: Median of pairwise slope combinations $Q = \text{median}((x_j - x_k)/(j - k))$ and robust properties against non-normal errors.
- **Influence on Project**: Implemented in `src/trend_analysis.py::compute_sens_slope()`.
- **Inspection Type**: Directly inspected.

### 1.4 Gilbert (1987) — Analytical Confidence Intervals for Sen's Slope
- **Title**: *Statistical Methods for Environmental Pollution Monitoring*
- **Author**: Richard O. Gilbert
- **Publisher**: Van Nostrand Reinhold, New York
- **Year**: 1987
- **ISBN**: 978-0442230500
- **Relevant Section**: Chapter 16 (Nonparametric Tests for Monotonic Trend: Sen's Estimator of Slope, pages 217–219).
- **Information Extracted**: Analytical rank index bounds $M_1 = \lfloor(N - C_\alpha)/2\rfloor$ and $M_2 = \lceil(N + C_\alpha)/2\rceil + 1$ using $C_\alpha = Z_{1-\alpha/2}\sqrt{\text{Var}(S)}$.
- **Influence on Project**: Eliminates the need for expensive bootstrap approximations by computing exact rank-based analytical confidence intervals for Sen's slope.
- **Inspection Type**: Directly inspected.

### 1.5 Benjamini & Hochberg (1995) — False Discovery Rate Control
- **Title**: *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*
- **Authors**: Yoav Benjamini, Yosef Hochberg
- **Journal**: *Journal of the Royal Statistical Society: Series B (Methodological)*, Volume 57, Issue 1, Pages 289–300
- **Year**: 1995
- **DOI / URL**: [https://doi.org/10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- **Relevant Section**: Section 1–3 (The FDR Procedure).
- **Information Extracted**: Step-up adjustment $p_{(k)}^{\text{BH}} = \min(1, \min_{j \ge k}\frac{m}{j}p_{(j)})$.
- **Influence on Project**: Implemented in `src/trend_analysis.py::compute_calendar_month_breakdown()` to resolve false discovery inflation across the 12 calendar month tests.
- **Inspection Type**: Directly inspected.

### 1.6 Holm (1979) — Sequentially Rejective Bonferroni Test
- **Title**: *A Simple Sequentially Rejective Multiple Test Procedure*
- **Author**: Sture Holm
- **Journal**: *Scandinavian Journal of Statistics*, Volume 6, Issue 2, Pages 65–70
- **Year**: 1979
- **URL**: [https://www.jstor.org/stable/4615733](https://www.jstor.org/stable/4615733)
- **Information Extracted**: Step-down Family-Wise Error Rate control $p_{(k)}^{\text{Holm}} = \min(1, \max_{j \le k}(m - j + 1)p_{(j)})$.
- **Influence on Project**: Implemented as conservative upper-bound FWER control in `compute_calendar_month_breakdown()`.
- **Inspection Type**: Directly inspected.

---

## 2. Directly Inspected Source Code Repositories

### 2.1 PyMannKendall
- **Repository**: `https://github.com/mmhs013/pyMannKendall`
- **Authors / Maintainers**: Md. Manjurul Hussain Shourov
- **Version Inspected**: `1.4.3`
- **Inspected Files & Functions**:
  - `pymannkendall/pymannkendall.py`:
    - `hamed_rao_modification_test()`
    - `__acf()`
    - `seasonal_test()`
    - `sens_slope()`
- **Information Extracted**:
  - Verified how `pymannkendall` computes detrended rank autocovariance and rank variance via `np.correlate`.
  - Confirmed the exact formula for $n/n^*$ and continuity correction $Z^* = (S - 1)/\sqrt{\text{Var}^*(S)}$.
  - Confirmed that `seasonal_test` returns slopes in units/year (not units/month), resolving a $12\times$ scaling bug.

### 2.2 SciPy (`scipy.stats`)
- **Repository**: `https://github.com/scipy/scipy`
- **Version**: `scipy >= 1.11.0`
- **Functions Used**: `scipy.stats.rankdata`, `scipy.stats.norm`, `scipy.stats.linregress`.

---

## 3. NASA Earth Science Data Platforms & API References

### 3.1 NASA POWER Project
- **Organization**: NASA Langley Research Center (LaRC)
- **Platform URL**: [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/)
- **API Documentation**: [https://power.larc.nasa.gov/docs/services/api/temporal/monthly/](https://power.larc.nasa.gov/docs/services/api/temporal/monthly/)
- **Underlying Atmospheric Model**: MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2), produced by the NASA Goddard Global Modeling and Assimilation Office (GMAO).
- **Native Spatial Resolution**: $0.5^\circ \text{ latitude} \times 0.625^\circ \text{ longitude}$ grid.
- **Variables Queried**:
  - `T2M`: Temperature at 2 Meters (°C)
  - `T2M_MAX`: Maximum Temperature at 2 Meters (°C)
  - `T2M_MIN`: Minimum Temperature at 2 Meters (°C)
  - `PRECTOTCORR`: Precipitation Corrected (mm/day)
- **Role in Project**: Primary empirical data source for multi-decadal analysis (1981–2023) across Greater Cairo, Alexandria, and Aswan.

### 3.2 NASA Giovanni
- **Organization**: NASA Goddard Earth Sciences Data and Information Services Center (GES DISC)
- **Platform URL**: [https://giovanni.gsfc.nasa.gov/](https://giovanni.gsfc.nasa.gov/)
- **Role in Project**: Formats supported in `src/data_loader.py::load_giovanni_csv()`.

### 3.3 NASA Earthdata
- **Organization**: NASA Earth Science Data and Information System (EOSDIS)
- **Platform URL**: [https://www.earthdata.nasa.gov/](https://www.earthdata.nasa.gov/)
- **Role in Project**: Scientific provenance framework for EOSDIS satellite observations.

---

## 4. Climatological Standards & Reference Manuals

### 4.1 World Meteorological Organization (WMO) Climate Normals
- **Title**: *WMO Guidelines on the Calculation of Climate Normals*
- **Publisher**: World Meteorological Organization, Geneva, Switzerland
- **Publication**: WMO-No. 1203 (2017)
- **Information Extracted**: Standard 30-year reference normal definition (1981–2010), calculation of monthly climatological baselines, and standardized temperature anomaly definitions.
- **Influence on Project**: Implemented in `src/preprocessing.py::compute_monthly_anomalies()` and `src/anomaly_detection.py`.

---

## 5. Summary Table of Reference Influence

| Reference | Inspected Section | Implementation Impact | Project File |
| :--- | :--- | :--- | :--- |
| **Hamed & Rao (1998)** | Section 3.2, Eq. 10–13 | Autocorrelation-corrected variance $\text{Var}^*(S)$ | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Hirsch et al. (1982)** | Pages 108–110 | Hirsch-Slack Seasonal Mann-Kendall omnibus test | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Gilbert (1987)** | Chapter 16, pp. 217–219 | Analytical rank-based 95% confidence intervals for Sen's slope | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Benjamini & Hochberg (1995)** | Section 1–3 | False Discovery Rate ($q < 0.05$) across 12 calendar months | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Holm (1979)** | Section 1–2 | Family-Wise Error Rate control across 12 calendar months | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **PyMannKendall Source** | `__acf`, `seasonal_test` | Code verification of numerical acf and yearly slope units | [`tests/test_trend_analysis.py`](file:///D:/Space-Apps/tests/test_trend_analysis.py) |
| **NASA POWER API** | Monthly Point Service | Data ingestion, metadata header parsing, coordinate mapping | [`src/data_loader.py`](file:///D:/Space-Apps/src/data_loader.py) |
| **WMO No. 1203** | Standard Normals | 1981–2010 baseline climatology for anomaly detection | [`src/preprocessing.py`](file:///D:/Space-Apps/src/preprocessing.py) |
