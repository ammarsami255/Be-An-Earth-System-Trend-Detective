# Mathematical Foundations and Technical Decisions
**Project**: EarthTrend Explorer (NASA Space Apps Challenge 2026: *"Be An Earth System Trend Detective!"*)  
**Team**: Machine Without Learning  
**Status**: Comprehensive Mathematical Documentation & Methodology Log  
**Workspace**: `D:\Space-Apps`  

---

## 1. Mathematical Architecture Overview

`EarthTrend Explorer` is designed to answer four core scientific questions regarding multi-decadal environmental trends:
1. **What is changing?** (Variable identification, extremes, and rainfall accumulation)
2. **Where is it changing?** (Geographic frame & MERRA-2 $0.5^\circ \times 0.625^\circ$ spatial resolution)
3. **How much is it changing?** (Rate of change per decade, percentage of mean, and cumulative shift)
4. **Is the observed trend statistically significant?** (Hypothesis testing, serial correlation correction, multiple-testing control, and abrupt break detection)

To ensure scientific rigor, the application prioritizes **non-parametric methods** (which make no assumptions of normal distribution and are robust to extreme outliers), while utilizing parametric models (Ordinary Least Squares) as supplementary diagnostics.

---

## 2. Complete Inventory of Mathematical Equations & Formulas

### 2.1 Mann-Kendall Trend Test Score ($S$)

#### Formula
$$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^n \text{sgn}(x_j - x_k)$$

$$\text{sgn}(x_j - x_k) = \begin{cases} +1 & \text{if } x_j - x_k > 0 \\ 0 & \text{if } x_j - x_k = 0 \\ -1 & \text{if } x_j - x_k < 0 \end{cases}$$

#### Definitions
- $n$: Total number of observations in the time series ($n = 43$ for annual series, $n = 516$ for monthly series).
- $x_j, x_k$: Sequential data values at time indices $j$ and $k$ where $j > k$.
- $\text{sgn}$: The signum function evaluating directional pairwise changes.

#### Intuition & Purpose
$S$ evaluates every possible distinct pair of observations $(x_k, x_j)$ with $k < j$. If the series is consistently increasing, most pairs will have $x_j > x_k$, resulting in a large positive $S$. If decreasing, $S$ is negative. Under the null hypothesis of no trend, $E[S] = 0$.

#### Implementation
- **File**: [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py)
- **Function**: `mann_kendall_test()`
- **Assumptions**: Observations are independent and identically distributed (i.i.d.) under $H_0$. Continuous measurement scale.
- **Limitations**: In the presence of positive serial autocorrelation, the variance of $S$ is underestimated, leading to elevated Type I error rates.

---

### 2.2 Theoretical Variance of Mann-Kendall Score ($\text{Var}(S)$)

#### Formula
$$\text{Var}(S) = \frac{n(n - 1)(2n + 5) - \sum_{p=1}^g t_p(t_p - 1)(2t_p + 5)}{18}$$

#### Definitions
- $n$: Sample size.
- $g$: Number of tied groups (groups of identical values).
- $t_p$: Number of data points tied in the $p$-th tied group.
- If there are no ties ($g = 0$):
  $$\text{Var}(S) = \frac{n(n - 1)(2n + 5)}{18}$$

#### Intuition & Purpose
Calculates the exact theoretical variance of $S$ under the null hypothesis of independence. For Cairo annual temperature ($n = 43$, 0 ties):
$$\text{Var}(S) = \frac{43 \times 42 \times (2 \times 43 + 5)}{18} = \frac{43 \times 42 \times 91}{18} = \frac{164346}{18} = \mathbf{9129.3333}$$

---

### 2.3 Standardized Mann-Kendall Test Statistic ($Z$)

#### Formula
$$Z = \begin{cases} \frac{S - 1}{\sqrt{\text{Var}(S)}} & \text{if } S > 0 \\ 0 & \text{if } S = 0 \\ \frac{S + 1}{\sqrt{\text{Var}(S)}} & \text{if } S < 0 \end{cases}$$

$$p = 2 \cdot \left[1 - \Phi(|Z|)\right] = 2 \cdot \left[1 - \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{|Z|} e^{-u^2/2} du\right]$$

#### Definitions
- $Z$: Standard normal test statistic ($Z \sim \mathcal{N}(0, 1)$ asymptotically for $n \ge 10$).
- $\pm 1$: Continuity correction accounting for the discrete integer distribution of $S$ approximated by a continuous normal distribution.
- $\Phi(|Z|)$: Cumulative distribution function of the standard normal distribution.

---

### 2.4 Kendall's Rank Correlation Coefficient ($\tau$)

#### Formula
$$\tau = \frac{S}{D}$$
$$D = \sqrt{\left[\frac{n(n - 1)}{2} - \frac{1}{2}\sum_{p=1}^g t_p(t_p - 1)\right] \left[\frac{n(n - 1)}{2}\right]}$$
Without ties:
$$\tau = \frac{S}{\frac{1}{2} n (n - 1)} = \frac{2S}{n(n - 1)}$$

---

### 2.5 Theil-Sen Robust Median Slope Estimator ($\beta$)

#### Formula
$$Q_{j,k} = \frac{x_j - x_k}{t_j - t_k} \quad \forall \ 1 \le k < j \le n$$

$$\beta = \text{median}\left(\{Q_{j,k} : 1 \le k < j \le n\}\right)$$

$$\text{Intercept } \alpha = \text{median}(x) - \beta \cdot \text{median}(t)$$

Has a high **breakdown point of 29.3%**, making it robust against gross anomalies.

---

### 2.6 Analytical 95% Confidence Interval for Sen's Slope (Gilbert 1987)

#### Formula
$$C_\alpha = Z_{1 - \alpha / 2} \cdot \sqrt{\text{Var}(S)}$$
$$M_1 = \left\lfloor \frac{N - C_\alpha}{2} \right\rfloor$$
$$M_2 = \left\lceil \frac{N + C_\alpha}{2} \right\rceil + 1$$
$$\beta_{\text{lower}} = Q_{(M_1)}, \quad \beta_{\text{upper}} = Q_{(M_2)}$$

---

### 2.7 Hamed & Rao (1998) Modified Mann-Kendall Variance Correction

#### Formula
$$\text{Var}^*(S) = \text{Var}(S) \cdot \frac{n}{n^*}$$
$$\frac{n}{n^*} = 1 + \frac{2}{n(n - 1)(n - 2)} \sum_{i=1}^{n-1} (n - i)(n - i - 1)(n - i - 2) \rho_i$$

Where $\rho_i$ is the rank autocorrelation of the detrended series $x^*_t = x_t - \beta \cdot t$. The summation includes only lags $i$ where:
$$|\rho_i| > \frac{Z_{1 - \alpha/2}}{\sqrt{n}} = \frac{1.959964}{\sqrt{n}}$$

$$Z^* = \frac{S - \text{sign}(S)}{\sqrt{\text{Var}^*(S)}} \quad (S \ne 0)$$

---

### 2.8 Hirsch-Slack Seasonal Mann-Kendall Omnibus Test

#### Formula
$$S_g = \sum_{k=1}^{n_g - 1} \sum_{j=k+1}^{n_g} \text{sgn}(x_{g,j} - x_{g,k}) \quad \text{for season } g \in \{1, \dots, m\}$$
$$S_{\text{tot}} = \sum_{g=1}^{m} S_g$$
$$\text{Var}(S_{\text{tot}}) = \sum_{g=1}^{m} \text{Var}(S_g)$$
$$Z_{\text{seasonal}} = \frac{S_{\text{tot}} - \text{sign}(S_{\text{tot}})}{\sqrt{\text{Var}(S_{\text{tot}})}}$$
$$\beta_{\text{seasonal}} = \text{median}\left(\bigcup_{g=1}^m \{Q_{j,k}^{(g)}\}\right)$$

---

### 2.9 Multiple Testing Corrections for 12 Calendar Months

When testing $m = 12$ calendar months simultaneously:
$$\alpha_{\text{family}} \le 1 - (1 - \alpha)^m = 1 - (0.95)^{12} \approx \mathbf{45.96\%}$$

1. **Holm-Bonferroni (FWER Control)**:
   $$p_{(k)}^{\text{Holm}} = \min\left(1, \max_{j \le k} \left[(m - j + 1) \cdot p_{(j)}\right]\right)$$
   Reject $H_{0,(k)}$ if $p_{(k)}^{\text{Holm}} < \alpha$. (5 months significant in Cairo: JUL, AUG, SEP, NOV, DEC).

2. **Benjamini-Hochberg (FDR Control)**:
   $$p_{(k)}^{\text{BH}} = \min\left(1, \min_{j \ge k} \left[\frac{m}{j} \cdot p_{(j)}\right]\right)$$
   Reject $H_{0,(k)}$ if $p_{(k)}^{\text{BH}} < q$. (6 months significant in Cairo: JUL, AUG, SEP, OCT, NOV, DEC).

---

### 2.10 Cumulative Warming Formulations

1. **Theil-Sen Elapsed Change (Primary)**: $\Delta T_{\text{Sen}} = \beta \cdot (T_{\text{end}} - T_{\text{start}}) = +1.23^\circ\text{C}$
2. **Raw Endpoint Difference**: $\Delta T_{\text{endpoint}} = T(t_n) - T(t_1) = +1.63^\circ\text{C}$
3. **Parametric OLS Linear Fitted Change**: $\Delta T_{\text{OLS}} = m \cdot (T_{\text{end}} - T_{\text{start}}) = +1.28^\circ\text{C}$
4. **Theil-Sen Sample Count Multiplier (Fencepost Discrepancy)**: $\Delta T_{\text{fencepost}} = \beta \cdot n = +1.26^\circ\text{C}$
5. **Decadal Baseline Climatological Shift**: $\Delta T_{\text{decadal}} = \bar{T}_{2014\text{–}2023} - \bar{T}_{1981\text{–}1990} = +0.95^\circ\text{C}$

---

### 2.11 Rainfall Rate-to-Accumulated Total Conversion

NASA POWER exports precipitation (`PRECTOTCORR`) as a daily average rate in $\text{mm/day}$ for each month. While temperature is an intensive state variable properly averaged across time, rainfall is an extensive cumulative flux:

1. **Monthly Accumulation**:
   $$P_{\text{month}} = R_{\text{day}} \times D_{\text{month}}$$
   where $R_{\text{day}}$ is the monthly mean precipitation rate ($\text{mm/day}$) and $D_{\text{month}}$ is the exact number of days in the specific month (accounting for leap years in February, e.g. 29 days in 2020 vs 28 in 2021).
2. **Annual Accumulation**:
   $$P_{\text{annual}} = \sum_{m=1}^{12} P_{m} \quad (\text{mm/year})$$
   Any year with fewer than 12 valid months is dropped to prevent artificially dry years from biasing the multi-decadal trend.

---

### 2.12 Wet-Season Dynamic Accumulation (October → March)

In Mediterranean and North African semi-arid/arid regimes, rainfall is concentrated almost exclusively in the cool winter season. Analyzing solely calendar years artificially splits single meteorological wet seasons across December 31 and January 1:

$$P_{\text{wet season}, Y} = \sum_{m=10}^{12} P_{m, Y-1} + \sum_{m=1}^{3} P_{m, Y}$$
- The season is labeled by its concluding year $Y$ (e.g. `2001/02`).
- The incomplete first season (requiring Oct–Dec 1980) is dropped to prevent boundary distortion.

---

### 2.13 Percent of Normal for Arid Precipitation Anomaly Analysis

Standardized Gaussian $Z$-scores ($Z = (x - \mu)/\sigma$) are mathematically invalid for arid precipitation because rainfall distributions are strongly right-skewed and zero-inflated (e.g., Cairo has 52.7% completely dry months, Aswan has 95.5%). 

Following WMO arid climatology standards, precipitation anomalies are expressed as **Percent of Normal**:
$$\text{PoN}_Y = \frac{P_Y}{\bar{P}_{1981\text{–}2010}} \times 100\%$$
where $\bar{P}_{1981\text{–}2010}$ is the 30-year WMO baseline normal total ($\text{mm/year}$).

---

### 2.14 Pettitt (1979) Non-Parametric Change-Point Test

Mann-Kendall evaluates monotonic, gradual drift. It has low statistical power to detect abrupt step-changes or regime shifts. In arid hydrology, climate shifts often manifest as sudden jumps (e.g. atmospheric circulation reorganizations or reanalysis sensor assimilation changes).

Pettitt's test detects a single abrupt change-point at time $t$ in a series of length $n$:

1. **Rank Sum Statistic**:
   $$U_t = \sum_{i=1}^t \sum_{j=t+1}^n \text{sgn}(x_i - x_j), \quad 1 \le t < n$$
2. **Test Statistic $K$**:
   $$K = \max_{1 \le t < n} |U_t|$$
3. **Approximate Two-Sided $p$-Value**:
   $$p \approx 2 \exp\left(-\frac{6 K^2}{n^3 + n^2}\right)$$
   (bounded at $p \le 1.0$).
4. **Change-Point Identification & Standardized Terminology**:
   $$t^* = \arg\max_{1 \le t < n} |U_t|$$
   The break divides the series into two regimes:
   - Pre-change regime: observations $x_1, \dots, x_{t^*}$ (ending at `change_year`, e.g. 2014 for Cairo rainfall).
   - Post-change regime: observations $x_{t^*+1}, \dots, x_n$ (starting at `new_regime_start`, e.g. 2015 for Cairo rainfall).
5. **Monte Carlo Permutation Diagnostics**:
   In addition to the asymptotic formula, an optional two-sided permutation test is supported ($B = 100{,}000$ permutations, seed 42) to evaluate empirical significance under the null hypothesis of exchangeable observations ($p_{\text{MC}} = 0.0023$ for Cairo, $p_{\text{MC}} = 0.0021$ for Alexandria). If serial correlation is present, exchangeability is violated and block permutation or pre-whitening is required.
6. **Scientific Attribution Boundary**:
   The test detects *when* an empirical break occurred in the MERRA-2 time series, *not why*. Conclusive attribution to physical climate change, GPM/TRMM constellation observing-system transitions, or satellite assimilation artifacts cannot be confirmed without validation against independent Egyptian Meteorological Authority (EMA) ground-station rain gauges.

---

### 2.15 Temperature Extremes: Absolute Monthly Extremes vs Daily Means

In NASA POWER MERRA-2 products, `T2M_MAX` and `T2M_MIN` represent the **absolute monthly extremes** reached within the month, **not** the monthly mean of daily maxima/minima.
- **Proof**: The annual summary value (`ANN`) in the raw CSV equals the exact mathematical maximum of the 12 monthly `T2M_MAX` values (e.g. Cairo year 2000: July $44.87^\circ\text{C} = \text{ANN}$). A mean of daily maxima for Cairo in winter would be $\sim 19^\circ\text{C}$, whereas the monthly record reached $24.7^\circ\text{C}$.
- **Implication**: `T2M_MAX` is the peak heat event of the month, and `T2M_MIN` is the coldest night of the month.

---

### 2.16 Five Temperature Extreme Indices (Monthly-Resolution ETCCDI Proxies)

Mirroring the World Meteorological Organization / Expert Team on Climate Change Detection and Indices (WMO/ETCCDI) framework adapted to monthly-resolution proxy data:

1. **Annual Hottest Value (TXx proxy)**: $T_{\text{hot}, Y} = \max_{m=1\dots 12} (\text{T2M\_MAX}_{Y, m})$
2. **Annual Coldest Value (TNn proxy)**: $T_{\text{cold}, Y} = \min_{m=1\dots 12} (\text{T2M\_MIN}_{Y, m})$
3. **Unusually Hot Months Count**: $N_{\text{hot}, Y} = \sum_{m=1}^{12} \mathbb{I}(\text{T2M\_MAX}_{Y, m} > P90_{m, 1981\text{–}2010})$
4. **Unusually Cold Months Count**: $N_{\text{cold}, Y} = \sum_{m=1}^{12} \mathbb{I}(\text{T2M\_MIN}_{Y, m} < P10_{m, 1981\text{–}2010})$
5. **Monthly Extreme Range Proxy**: $\Delta T_{\text{range}, Y} = \frac{1}{12}\sum_{m=1}^{12} (\text{T2M\_MAX}_{Y, m} - \text{T2M\_MIN}_{Y, m})$

Thresholds $P90_m$ and $P10_m$ are computed on each calendar month separately over the 1981–2010 WMO normal period to preserve seasonal relativity (e.g. a hot July is compared against historical Julys, not January).

*Terminology Qualification*: Because NASA POWER supplies monthly peak values rather than daily time series, these metrics represent monthly-resolution extreme proxies rather than standard daily ETCCDI indices. High-frequency daily heatwaves or diurnal ranges cannot be directly evaluated without sub-monthly data.

---

### 2.17 Count Data Trend Estimation (Why OLS for Magnitude + MK for Significance)

For discrete integer count metrics (e.g., $N_{\text{hot}, Y} \in \{0, 1, 2, 3\}$), a high proportion of pairwise differences $(y_j - y_i)$ are zero (tied ranks). Consequently:
- **Sen's Median Slope Collapse**: The median of pairwise slopes collapses to exactly $0.000$, even when the upward trend is statistically significant under Mann-Kendall and Hamed-Rao.
- **Methodological Solution**:
  - **Significance**: Derived strictly from non-parametric rank tests (Mann-Kendall + Hamed-Rao), which naturally incorporate tie-correction formulas.
  - **Magnitude**: Quantified using Ordinary Least Squares (OLS) slope with analytical $t$-distribution 95% confidence intervals, providing an uncollapsed rate of change (e.g., $+0.26\text{ hot months/decade}$).

---

### 2.18 Multi-Variable Regional Comparison (`compare_temperature_extremes`)

To answer whether extreme temperatures are warming faster than mean temperatures:
- Each series (`T2M`, `T2M_MAX`, `T2M_MIN`) is aggregated to annual means.
- Theil-Sen slope and analytical 95% confidence intervals are computed side by side.
- **Scientific Caveat**: Because the 95% confidence intervals overlap across `T2M`, `T2M_MAX`, and `T2M_MIN` in all three regions, the application explicitly warns users that differential warming rates cannot be claimed without a formal multivariate interaction test.

---

### 2.19 WMO Climatological Baseline & Calendar-Month Normalization

To evaluate departures from historical expectations without distorting seasonal cycles, anomalies are computed relative to the standard World Meteorological Organization (WMO) 30-year normal period (1981–2010):
- **Calendar-Month Stratification**: Each monthly observation is evaluated against its corresponding calendar month across the 30-year normal ($N_m = 30$):
  $$\mu_m = \frac{1}{30} \sum_{y=1981}^{2010} x_{y, m}, \quad \sigma_m = \sqrt{\frac{1}{29} \sum_{y=1981}^{2010} (x_{y, m} - \mu_m)^2}$$
- **Absolute Anomaly**:
  $$A_{y, m} = x_{y, m} - \mu_m$$
- **Standardized Anomaly ($Z$-Score)**:
  $$Z_{y, m} = \begin{cases} \frac{x_{y, m} - \mu_m}{\sigma_m}, & \sigma_m > 0 \\ 0.0, & \sigma_m = 0 \end{cases}$$
- **WMO Severity Brackets**:
  - Extreme Warm: $Z \ge +3.0\sigma$ ($p < 0.15\%$ under normality)
  - Severe Warm: $+2.0\sigma \le Z < +3.0\sigma$
  - Moderate Warm: $+1.0\sigma \le Z < +2.0\sigma$
  - Near Normal: $-1.0\sigma < Z < +1.0\sigma$
  - Moderate Cold: $-2.0\sigma < Z \le -1.0\sigma$
  - Severe Cold: $-3.0\sigma < Z \le -2.0\sigma$
  - Extreme Cold: $Z \le -3.0\sigma$
- **Standard vs. Provisional Climate Normals (WMO-No. 49 & WMO-No. 1203)**:
  - *Standard Normal*: 30 consecutive calendar years (1981–2010), used as the primary baseline across all modules.
  - *Provisional Normal*: WMO Technical Regulations (WMO-No. 49 Section B.1) and WMO Guidelines on Climate Normals (WMO-No. 1203 Section 2.2) explicitly permit the calculation and provisional use of normals when at least 10 years of homogeneous data are available. The anomaly detection engine (`src/anomaly_detection.py`) strictly enforces a minimum threshold of $\ge 10$ years for user-selected or custom baselines, fully compliant with WMO provisional normal standards.

---

### 2.20 Distinct Temperature Anomaly Semantics (`T2M`, `T2M_MAX`, `T2M_MIN`)

Our audit established that NASA POWER monthly variables have distinct physical definitions:
- $A(\text{T2M})$ measures the departure of the **monthly mean temperature**.
- $A(\text{T2M\_MAX})$ measures the departure of the **month's absolute hottest single peak temperature** relative to historical monthly peaks.
- $A(\text{T2M\_MIN})$ measures the departure of the **month's absolute coldest single trough temperature** relative to historical monthly troughs.
- **Scientific Rule**: $A(\text{T2M})$, $A(\text{T2M\_MAX})$, and $A(\text{T2M\_MIN})$ are strictly calculated and displayed independently. They are never mixed, averaged, or subtracted across incompatible baseline references.

---

### 2.21 Persistent Anomaly & Climate Spell Detection (Run-Length Analysis)

Individual anomalous months may represent fleeting weather noise. Persistent climatic spells (heat waves, prolonged cold spells) represent multi-month synoptic blocking or land-atmosphere coupling:
- **Definition**: A contiguous sequence of consecutive months $t, t+1, \dots, t+k-1$ where $|Z_t| \ge Z_{\text{crit}}$ with minimum duration $k \ge 3$ months.
- **Metrics Tracked**:
  - Duration in months ($k$)
  - Peak standardized departure ($\max |Z|$)
  - Mean standardized departure ($\frac{1}{k} \sum Z_t$)
  - Cumulative anomaly intensity ($\sum Z_t$)

---

### 2.22 Rejection of Gaussian $Z$-Scores for Arid Precipitation

A common mistake in climate dashboards is applying ordinary Gaussian $Z$-scores ($Z = (P - \mu)/\sigma$) to precipitation. In Egypt's arid and hyper-arid climate, this is **statistically invalid**:
1. **Physical Non-Negativity**: Precipitation is bounded below by zero ($P \ge 0$). Normal distributions admit negative values.
2. **Skewness & Zero-Inflation**: In Aswan, 86.4% of months are strictly zero rain; in Cairo, 40.5% are zero. The distribution consists of a discrete point mass at zero and a highly skewed positive tail.
3. **Severe Drought Paradox**: When baseline mean is small and variance is relatively large ($\mu = 1.0\text{ mm}, \sigma = 3.0\text{ mm}$), completely zero rain ($P = 0$) receives:
   $$Z = \frac{0 - 1.0}{3.0} = -0.33\sigma$$
   Under a Gaussian assumption, complete absence of rain cannot even reach the $-1.0\sigma$ moderate drought threshold, making severe drought ($Z \le -1.5\sigma$) mathematically impossible to detect!
4. **Conclusion**: Gaussian $Z$-scores are rejected for precipitation in favor of Percent of Normal and the non-parametric Standardized Precipitation Index (SPI).

---

### 2.23 Standardized Precipitation Index (SPI) Engine with Mixed Gamma Distribution

The Standardized Precipitation Index (McKee et al. 1993, Edwards & McKee 1997, WMO-No. 1090) resolves the skewness and zero-inflation problem by fitting a mixed Gamma distribution:
1. **Multi-Scalar Rolling Accumulation**: For scale $k \in \{3, 6, 12\}$ months:
   $$x_t = \sum_{i=0}^{k-1} P_{t-i}$$
2. **Stratification**: Evaluated per calendar month $m$ to preserve seasonal accumulation cycles.
3. **Zero-Probability Adjustment**: The empirical baseline probability of zero accumulation is:
   $$q = \frac{m_0}{N}$$
   where $m_0$ is the number of zero values in the 30-year baseline.
4. **Gamma PDF Fitting**: For non-zero values ($x > 0$), the 2-parameter Gamma distribution is fitted via Maximum Likelihood Estimation (MLE):
   $$g(x) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\beta}$$
5. **Mixed Cumulative Distribution Function**:
   $$H(x) = q + (1 - q) G(x)$$
   where $G(x) = \frac{1}{\Gamma(\alpha)} \int_0^{x/\beta} t^{\alpha-1} e^{-t} dt$.
6. **Standard Normal Inversion**:
   $$\text{SPI} = \Phi^{-1}(H(x))$$
   where $\Phi^{-1}$ is the inverse standard normal cumulative distribution function (`scipy.stats.norm.ppf`), clamped to $[10^{-7}, 1 - 10^{-7}]$.
7. **WMO Drought Categories**:
   - $\text{SPI} \ge +2.0$: Extremely Wet
   - $+1.5 \le \text{SPI} < +2.0$: Very Wet
   - $+1.0 \le \text{SPI} < +1.5$: Moderately Wet
   - $-1.0 < \text{SPI} < +1.0$: Near Normal
   - $-1.5 < \text{SPI} \le -1.0$: Moderately Dry
   - $-2.0 < \text{SPI} \le -1.5$: Severely Dry
   - $\text{SPI} \le -2.0$: Extremely Dry

---

### 2.24 Aridity Guardrails for SPI & Drought Truncation Boundary

A critical limitation in SPI literature (Wu et al. 2007, Stagge et al. 2015) occurs when baseline zero probability is non-negligible:
- When precipitation is zero ($x = 0$), $H(0) = q$.
- Therefore, the theoretical minimum attainable SPI for that calendar month is bounded below:
  $$\text{SPI}_{\text{min}} = \Phi^{-1}(q)$$
- **Structural Drought Detection Truncation**: When $q > \Phi(-1.0) \approx 0.1587$, then $\text{SPI}_{\text{min}} > -1.0$. Even under complete absence of precipitation ($0.0\text{ mm}$), SPI cannot reach the standard WMO moderate drought threshold ($\le -1.0$). In such cases, the engine exposes `SPI_{scale}_lower_bound` and flags `SPI_{scale}_drought_limited = True`.
- **Hyper-Aridity Guardrail ($q \ge 0.50$)**: If $q = 0.50$, $\Phi^{-1}(0.50) = 0.00$. If $q = 0.80$ (typical of summer months or Upper Egypt), $H(0) = 0.80 \implies \text{SPI} = \Phi^{-1}(0.80) = +0.84$! Completely dry weather would be falsely classified as "moderately wet".
- **Scientific Guardrail Implemented**: Whenever $q \ge 0.50$ or the number of non-zero baseline observations is $< 5$, the engine flags SPI as **`Undefined / Unreliable due to Hyper-Aridity`** and outputs `NaN`. This prevents spurious positive drought indices during hyper-arid desert dry seasons.
- **Percent of Normal Stability Guardrail**: When climatological baseline monthly precipitation $\mu_m < 1.0\text{ mm}$, Percent of Normal ($100 \cdot P / \mu_m$) fluctuates erratically on trace showers (e.g., $0.5\text{ mm}$ produces $1000\%$). The engine flags these months as `percent_of_normal_reliable = False` with `percent_of_normal_flag = "Low Reliability (Baseline < 1.0 mm)"`, preserving numerical values while alerting researchers.

---

### 2.25 Compound Extremes & Cross-Variable Lagged Correlation

To investigate compound hazards and land-atmosphere feedbacks:
1. **Compound Quadrants**:
   - Compound Hot & Dry: $Z_T \ge +1.0\sigma$ AND $P \le 75\%$ of normal
   - Compound Hot & Wet: $Z_T \ge +1.0\sigma$ AND $P \ge 125\%$ of normal
   - Compound Cold & Dry: $Z_T \le -1.0\sigma$ AND $P \le 75\%$ of normal
   - Compound Cold & Wet: $Z_T \le -1.0\sigma$ AND $P \ge 125\%$ of normal
2. **Serial Correlation Adjustment for Degrees of Freedom ($N_{\text{eff}}$)**:
   Because temperature and precipitation series exhibit temporal persistence, raw Pearson $p$-values underestimate Type I error. We compute the Quenouille (1947) / Pyper & Peterman (1998) effective sample size:
   $$N_{\text{eff}} \approx N \frac{1 - r_{1, T} r_{1, P}}{1 + r_{1, T} r_{1, P}}$$
   and test significance using $t = r \sqrt{\frac{N_{\text{eff}} - 2}{1 - r^2}}$.
3. **Lagged Cross-Correlation**:
   Computes $r(A_P(t), A_T(t + k))$ for $k \in \{-3, \dots, +3\}$ months. In Alexandria and Cairo, $k=+1$ exhibits statistically significant negative correlation ($r \approx -0.19$, $p < 0.001$). This is consistent with persistent synoptic cold-air advection from winter Mediterranean depressions and potential land-atmosphere feedbacks, but statistical correlation does not prove physical causation.
4. **Decadal Aggregation & Temporal Coverage Normalization**:
   Decadal shifts are evaluated across distinct 10-year label groups. Time coverage across the 1981–2023 record is explicitly documented to prevent readers from confusing partial boundary decades with complete 10-year spans:
   - **1980s**: 1981–1989 (9 elapsed years, 108 calendar months).
   - **1990s**: 1990–1999 (10 complete years, 120 calendar months).
   - **2000s**: 2000–2009 (10 complete years, 120 calendar months).
   - **2010s**: 2010–2019 (10 complete years, 120 calendar months).
   - **2020s**: 2020–2023 (4 elapsed years, 48 calendar months).
   Decadal percentages are normalized to total elapsed months in each period ($\text{Count} / N_{\text{decade}} \times 100\%$):
   - **Greater Cairo Hot & Dry**: 7/108 (6.5%) in 1980s $\to$ 12/120 (10.0%) in 1990s $\to$ 16/120 (13.3%) in 2000s $\to$ 29/120 (24.2%) in 2010s $\to$ 13/48 (27.1%) in 2020s.
   - **Alexandria Hot & Dry**: 8/108 (7.4%) in 1980s $\to$ 8/120 (6.7%) in 1990s $\to$ 19/120 (15.8%) in 2000s $\to$ 34/120 (28.3%) in 2010s $\to$ 16/48 (33.3%) in 2020s.
   - **Aswan Hot & Dry**: 5/108 (4.6%) in 1980s $\to$ 11/120 (9.2%) in 1990s $\to$ 18/120 (15.0%) in 2000s $\to$ 22/120 (18.3%) in 2010s $\to$ 12/48 (25.0%) in 2020s.

---

## 2.6 Climate Anomaly Forecasting Engine & Chronological Validation

### 1. Mathematical Formulations of Candidate Forecasting Models

#### A. Seasonal Naive Benchmark Model
As established by Hyndman & Athanasopoulos (2018), any credible predictive system must be benchmarked against a formal seasonal persistence null model:
$$\hat{y}_{T+h} = y_{T+h-12 \cdot k}, \quad k = \lfloor (h-1)/12 \rfloor + 1$$
For a 12-month horizon ($h \in \{1, \dots, 12\}$), $\hat{y}_{T+h} = y_{T+h-12}$.
The in-sample residual variance is derived from 12-month seasonal differences:
$$e_t = y_t - y_{t-12}, \quad t = 13, \dots, T$$
$$\hat{\sigma}_{\text{naive}}^2 = \frac{1}{T-12}\sum_{t=13}^T e_t^2$$
Under a seasonal random walk with additive errors, the $h$-step forecast variance scales with seasonal cycles $k$:
$$\text{Var}(\hat{y}_{T+h}) = k \cdot \hat{\sigma}_{\text{naive}}^2$$
Prediction intervals are constructed via standard normal quantiles ($z_{0.10} \approx 1.282$ for 80% CI; $z_{0.025} \approx 1.960$ for 95% CI):
$$\hat{y}_{T+h} \pm z_{\alpha/2} \cdot \sqrt{k} \cdot \hat{\sigma}_{\text{naive}}$$

#### B. Holt-Winters Additive Exponential Smoothing (State-Space ETS)
For time series with linear trend and additive seasonal cycles of period $m=12$ (Holt 1957, Winters 1960):
$$\begin{aligned}
\text{Level: } & l_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(l_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1} \\
\text{Seasonal: } & s_t = \gamma(y_t - l_{t-1} - b_{t-1}) + (1-\gamma)s_{t-m} \\
\text{Point Forecast: } & \hat{y}_{T+h} = l_T + h \cdot b_T + s_{T+h-m \cdot k}
\end{aligned}$$
Smoothing parameters $\alpha, \beta, \gamma \in [0, 1]$ are estimated via Maximum Likelihood / Sum of Squared Errors minimization.
Prediction intervals are derived through **1,000 Monte Carlo innovations simulations** using estimated model residual variance, capturing empirical uncertainty bounds without restrictive asymptotic assumptions:
$$\text{CI}_{80} = [P_{10}(\text{simulations}), P_{90}(\text{simulations})], \quad \text{CI}_{95} = [P_{2.5}(\text{simulations}), P_{97.5}(\text{simulations})]$$

#### C. SARIMAX (Seasonal Autoregressive Integrated Moving Average)
Following Box, Jenkins, Reinsel, & Ljung (2015), the parsimonious seasonal climate model is formulated as:
$$\text{SARIMA}(1, 0, 0) \times (1, 1, 0)_{12}$$
$$(1 - \phi_1 B)(1 - \Phi_1 B^{12})(1 - B^{12}) y_t = \epsilon_t, \quad \epsilon_t \sim \text{i.i.d. } \mathcal{N}(0, \sigma^2)$$
where $B$ is the backshift operator ($B^k y_t = y_{t-k}$). If optimization fails to converge, an automated fallback to $(0, 1, 1) \times (0, 1, 1)_{12}$ is executed.
Prediction intervals are evaluated using the exact state-space Kalman filter covariance matrix:
$$\hat{y}_{T+h} \pm z_{\alpha/2} \sqrt{\text{Var}_{\text{Kalman}}(\hat{y}_{T+h})}$$

### 2. Expanding-Window Chronological Backtesting Protocol
To eliminate future data leakage and simulate genuine operational deployment:
- The historical series of length $T=516$ months (1981–2023) is evaluated across $K=5$ distinct forecast origins:
  - Origin 1: Train 1981–2018 ($N=456$), Test 2019 ($H=12$)
  - Origin 2: Train 1981–2019 ($N=468$), Test 2020 ($H=12$)
  - Origin 3: Train 1981–2020 ($N=480$), Test 2021 ($H=12$)
  - Origin 4: Train 1981–2021 ($N=492$), Test 2022 ($H=12$)
  - Origin 5: Train 1981–2022 ($N=504$), Test 2023 ($H=12$)
- Accuracy metrics for origin $k$ and model $m$:
  $$\text{MAE}_k = \frac{1}{H}\sum_{j=1}^H |y_{T_k+j} - \hat{y}_{T_k+j}^{(m)}|, \quad \text{RMSE}_k = \sqrt{\frac{1}{H}\sum_{j=1}^H (y_{T_k+j} - \hat{y}_{T_k+j}^{(m)})^2}$$
- **Mean Absolute Scaled Error (MASE)** (Hyndman & Koehler 2006):
  $$\text{MASE}^{(m)} = \frac{\frac{1}{K}\sum_{k=1}^K \text{MAE}_k^{(m)}}{\frac{1}{T-12}\sum_{t=13}^T |y_t - y_{t-12}|}$$
  A model achieves true scientific predictive superiority over seasonal persistence if and only if $\text{MASE} < 1.0$.
  The model with the lowest aggregate out-of-sample MAE is designated Tournament Winner.

### 3. Future Anomaly Detection & Uncertainty Propagation
The 12-month forward point projection $\hat{y}_{T+h}$ ($h=1, \dots, 12$) is aligned against the calendar month $m(h) \in \{1, \dots, 12\}$ of the WMO 1981–2010 normal:
- **Expected Absolute Anomaly**: $\hat{A}_{T+h} = \hat{y}_{T+h} - \mu_{m(h)}$
- **Expected Standardized Anomaly (Z-score)**: $\hat{Z}_{T+h} = \frac{\hat{A}_{T+h}}{\sigma_{m(h)}}$
- **95% Anomaly Confidence Band**: $[\hat{y}_{\text{ci95\_low}} - \mu_m, \; \hat{y}_{\text{ci95\_high}} - \mu_m]$
- **95% Z-Score Confidence Band**: $\left[\frac{\hat{y}_{\text{ci95\_low}} - \mu_m}{\sigma_m}, \; \frac{\hat{y}_{\text{ci95\_high}} - \mu_m}{\sigma_m}\right]$
- **Physical Guardrails**:
  - For precipitation: $\hat{y} = \max(0.0, \hat{y})$ and $\text{ci\_low} = \max(0.0, \text{ci\_low})$.
  - For hyper-arid series ($>80\%$ zero-rain months, e.g., Aswan): Displays a dedicated advisory alerting users that background statistical forecasts reflect trace baseline expectations and cannot extrapolate episodic convective spikes.

---

## 3. Technical Decisions & Alternatives Comparison Table

| Method / Component | Status | Selection Reason | Rejected Alternative | Rejection Reason | Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seasonal Naive Baseline Benchmark** | Implemented & Verified | Standard reference benchmark; required to calculate MASE (Hyndman 2006) | Evaluating complex models without a baseline | Without a naive benchmark, error metrics have no scale context | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
| **Holt-Winters Additive ETS** | Implemented & Verified | Optimal for Gaussian deseasonalized temperature series; captures linear trend and 12-mo seasonal cycle | Multiplicative Holt-Winters | Multiplicative seasonal components can explode or diverge with zero-centered or negative numbers | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
| **SARIMAX State-Space Model** | Implemented & Verified | State-space formulation with analytical Kalman filter covariance intervals | Deep Learning (LSTM / Transformer) | 516 monthly points is too small for deep learning without severe overfitting and high compute cost | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
| **Expanding-Window Backtesting** | Implemented & Verified | Strict chronological validation across 5 origins (2018–2022); zero future leakage | Random K-Fold Cross-Validation | K-Fold shuffles time series, leaking future information into past training sets | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
| **Precipitation Non-Negativity Clamping** | Implemented & Verified | Enforces physical reality: precipitation and lower uncertainty bounds cannot be negative | Unbounded Gaussian intervals | Linear/Gaussian models predict unphysical negative rainfall in arid regimes | [`src/forecasting.py`](file:///D:/Space-Apps/src/forecasting.py) |
| **Hyper-Arid Advisory Banner** | Implemented & Verified | Scientifically transparent warning for zero-inflated Aswan precipitation | Suppressing Aswan rainfall or claiming high predictive skill | Arid convective pulses are episodic and cannot be extrapolated deterministically | [`app.py`](file:///D:/Space-Apps/app.py) |
| **Non-Parametric Mann-Kendall** | Implemented & Verified | Distribution-free, handles non-normal climate data, robust to outliers | OLS regression as sole method | OLS assumes normal, homoscedastic errors and is sensitive to individual extreme anomalies | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Theil-Sen Robust Slope** | Implemented & Verified | 29.3% breakdown point against outliers; median pairwise slope | Mean pairwise slope (least squares slope) | Distorted by extreme individual years | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Hamed & Rao (1998) Correction** | Implemented & Verified | Corrects variance for serial correlation in detrended ranks without arbitrary pre-whitening | Pre-whitening (e.g., Kulkarni & von Storch 1995) | Pre-whitening removes real trend slope along with AR(1) noise, deflating trend magnitude | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Pettitt (1979) Change-Point Test** | Implemented & Verified | Detects sudden non-linear regime shifts and step jumps that Mann-Kendall misses | CUSUM or Bai-Perron parametric tests | Parametric break tests assume normality, invalid for zero-inflated precipitation | [`src/rainfall_analysis.py`](file:///D:/Space-Apps/src/rainfall_analysis.py) |
| **Standardized Precipitation Index (SPI)** | Implemented & Verified | WMO reference drought index; Gamma fitting with zero-probability correction $H(x) = q + (1-q)G(x)$ | Ordinary Gaussian Z-score on rainfall | Gaussian Z-scores assume bell symmetry, allowing negative rain and bounding zero-rain above -1.0σ | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **SPI Aridity Guardrail (q < 0.50)** | Implemented & Verified | Prevents SPI from assigning positive (wet) values to zero rain when zeros exceed 50% | Blindly fitting Gamma across all months | When q > 0.50, H(0) > 0.50 causing dry desert months to be classified as wet | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **Persistent Spell Run-Length** | Implemented & Verified | Detects multi-month persistent warm/cold spells (duration ≥ 3 mo, $|Z| \ge 1.0$) | Single-month anomaly thresholding | Single months conflate transient weather noise with persistent climate anomalies | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **Compound Extremes (Hot-Dry)** | Implemented & Verified | Classifies joint compound hazards ($Z_T \ge 1.0$ and $P \le 75\%$) | Univariate temperature analysis | Univariate models ignore concurrent drought-warm spell compounding feedbacks | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **Effective Degrees of Freedom (N_eff)** | Implemented & Verified | Quenouille / Pyper-Peterman correction for serial correlation in cross-variable testing | Standard independent sample $t$-test | Autocorrelation severely inflates Type I false discovery rates in cross-correlation | [`src/anomaly_detection.py`](file:///D:/Space-Apps/src/anomaly_detection.py) |
| **Rainfall Physical Totals (mm)** | Implemented & Verified | Converts daily rate (mm/day) to physically accumulated monthly and annual depth (mm) | Analyzing mm/day rates directly | Daily rates obscure monthly calendar length variations and do not represent water resource volumes | [`src/rainfall_analysis.py`](file:///D:/Space-Apps/src/rainfall_analysis.py) |
| **Percent of Normal (PoN)** | Implemented & Verified | Standard WMO measure for arid precipitation anomalies | Standard Gaussian Z-scores ($Z = A_t / \sigma$) | Z-scores assume bell-shaped symmetry; invalid for zero-inflated skewed rainfall | [`src/rainfall_analysis.py`](file:///D:/Space-Apps/src/rainfall_analysis.py) |
| **Extreme Percentile Thresholds (P90/P10)** | Implemented & Verified | Evaluates extreme frequencies relative to each calendar month's baseline normal | Fixed annual temperature cutoffs (e.g. >35°C) | Fixed cutoffs fail to detect winter heat anomalies and summer cold anomalies | [`src/extremes_analysis.py`](file:///D:/Space-Apps/src/extremes_analysis.py) |
| **OLS Slope for Count Extremes** | Implemented & Verified | Prevents Sen's median slope from collapsing to zero on small tied integer counts | Reporting zero slope or forcing Sen's slope | Sen's median slope of discrete counts with >50% ties is trivially 0, hiding real increases | [`src/extremes_analysis.py`](file:///D:/Space-Apps/src/extremes_analysis.py) |
| **Benjamini-Hochberg FDR** | Implemented & Verified | Controls false discovery rate across 12 monthly tests without being overly conservative | Standard Bonferroni | Standard Bonferroni ($p < \alpha/m$) is excessively conservative, causing high Type II error rates | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Hirsch-Slack Seasonal MK** | Implemented & Qualified | Evaluates multi-decadal monthly series by comparing like months across years | Deseasonalizing by subtracting monthly mean + standard MK | Subtraction assumes constant variance and linear additive seasonal cycle | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **WMO 1981–2010 Baseline Normal** | Implemented & Verified | Standard WMO 30-year climatological normal period | Whole-record mean | Whole-record mean artificially forces anomaly sum to zero and confounds trend detection | [`src/preprocessing.py`](file:///D:/Space-Apps/src/preprocessing.py) |
| **Fixed Dark Space UI Theme** | Implemented & Verified | High-contrast scientific palette (.streamlit/config.toml) with unified Plotly dark charts | Default Streamlit theme following OS | White charts on dark background or mismatched text/card contrast | [`.streamlit/config.toml`](file:///D:/Space-Apps/.streamlit/config.toml) |
| **Top Navbar Navigation** | Implemented & Verified | Responsive horizontal navbar with sub-page tabs; maximizes screen real estate | Permanent left sidebar | Sidebar occupied 25% of horizontal space and cramped high-density statistical charts | [`app.py`](file:///D:/Space-Apps/app.py) |
| **Streamlit Native AppTest** | Implemented & Verified | Fast headless CI testing of all UI pages and widgets | Selenium / Playwright browser automation | Heavyweight, brittle headless browser dependencies requiring webdriver binaries | [`tests/test_streamlit_app.py`](file:///D:/Space-Apps/tests/test_streamlit_app.py) |
| **Multivariate Slope Interaction Model** | Investigated, Not Implemented | Required to formally prove whether Aswan warms significantly faster than Cairo | Raw slope point comparison | Comparing slopes without an interaction test is statistically invalid when CIs overlap | Planned Phase |
