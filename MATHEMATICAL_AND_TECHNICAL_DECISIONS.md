# Mathematical Foundations and Technical Decisions
**Project**: EarthTrend Explorer (NASA Space Apps Challenge 2026: *"Be An Earth System Trend Detective!"*)  
**Team**: Machine Without Learning  
**Status**: Comprehensive Mathematical Documentation & Methodology Log  
**Workspace**: `D:\Space-Apps`  

---

## 1. Mathematical Architecture Overview

`EarthTrend Explorer` is designed to answer four core scientific questions regarding multi-decadal environmental trends:
1. **What is changing?** (Variable identification & trajectory)
2. **Where is it changing?** (Geographic frame & spatial resolution)
3. **How much is it changing?** (Rate of change per decade & cumulative magnitude)
4. **Is the observed trend statistically significant?** (Hypothesis testing & error-rate control)

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

#### Implementation
- **File**: [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py)
- **Function**: `mann_kendall_test()`

---

### 2.3 Standardized Mann-Kendall Test Statistic ($Z$)

#### Formula
$$Z = \begin{cases} \frac{S - 1}{\sqrt{\text{Var}(S)}} & \text{if } S > 0 \\ 0 & \text{if } S = 0 \\ \frac{S + 1}{\sqrt{\text{Var}(S)}} & \text{if } S < 0 \end{cases}$$

$$p = 2 \cdot \left[1 - \Phi(|Z|)\right] = 2 \cdot \left[1 - \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{|Z|} e^{-u^2/2} du\right]$$

#### Definitions
- $Z$: Standard normal test statistic ($Z \sim \mathcal{N}(0, 1)$ asymptotically for $n \ge 10$).
- $\pm 1$: Continuity correction accounting for the discrete integer distribution of $S$ approximated by a continuous normal distribution.
- $\Phi(|Z|)$: Cumulative distribution function of the standard normal distribution.

#### Intuition & Purpose
Converts the rank score $S$ into a standard normal metric $Z$. For Cairo annual T2M ($S = +512.0$):
$$Z = \frac{512.0 - 1.0}{\sqrt{9129.3333}} = \frac{511.0}{95.54754} = \mathbf{+5.348123}$$
$$p = 2 \cdot [1 - \Phi(5.348123)] = \mathbf{8.887 \times 10^{-8}}$$
Because $p < 0.05$ (in fact $p < 10^{-7}$), the null hypothesis of no trend is decisively rejected.

#### Implementation
- **File**: [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py)
- **Function**: `mann_kendall_test()`

---

### 2.4 Kendall's Rank Correlation Coefficient ($\tau$)

#### Formula
$$\tau = \frac{S}{D}$$
$$D = \sqrt{\left[\frac{n(n - 1)}{2} - \frac{1}{2}\sum_{p=1}^g t_p(t_p - 1)\right] \left[\frac{n(n - 1)}{2}\right]}$$
Without ties:
$$\tau = \frac{S}{\frac{1}{2} n (n - 1)} = \frac{2S}{n(n - 1)}$$

#### Definitions
- $\tau$: Kendall's tau rank correlation coefficient ($-1 \le \tau \le 1$).
- $D$: Maximum possible number of discordant/concordant pairs.
- For Cairo annual T2M ($n = 43, S = 512$):
  $$\tau = \frac{512}{\frac{43 \times 42}{2}} = \frac{512}{903} = \mathbf{+0.566999}$$

#### Intuition & Purpose
Provides a normalized, non-parametric effect-size metric representing the net excess of concordant pairs over discordant pairs as a proportion of total pairs.

---

### 2.5 Theil-Sen Robust Median Slope Estimator ($\beta$)

#### Formula
$$Q_{j,k} = \frac{x_j - x_k}{t_j - t_k} \quad \forall \ 1 \le k < j \le n$$

$$\beta = \text{median}\left(\{Q_{j,k} : 1 \le k < j \le n\}\right)$$

$$\text{Intercept } \alpha = \text{median}(x) - \beta \cdot \text{median}(t)$$

#### Definitions
- $N = \frac{n(n - 1)}{2}$: Total number of pairwise combinations ($N = 903$ for $n = 43$).
- $Q_{j,k}$: Pairwise slope between time points $k$ and $j$.
- $\beta$: Sen's slope estimate (units: environmental measurement unit per time unit).

#### Intuition & Purpose
Calculates the rate of change across the period. While ordinary least squares (OLS) minimizes squared residuals (making it vulnerable to a single extreme anomaly), the Theil-Sen estimator takes the median of all pairwise slopes. It has a high **breakdown point of 29.3%**, meaning up to ~29% of measurements can be arbitrary gross outliers without corrupting the slope.

#### Numerical Example (Cairo Annual T2M)
- $\beta = \mathbf{+0.029267^\circ\text{C / year}}$ ($+0.2927^\circ\text{C / decade}$).

---

### 2.6 Analytical 95% Confidence Interval for Sen's Slope (Gilbert 1987)

#### Formula
$$C_\alpha = Z_{1 - \alpha / 2} \cdot \sqrt{\text{Var}(S)}$$

$$M_1 = \left\lfloor \frac{N - C_\alpha}{2} \right\rfloor$$

$$M_2 = \left\lceil \frac{N + C_\alpha}{2} \right\rceil + 1$$

$$\beta_{\text{lower}} = Q_{(M_1)}, \quad \beta_{\text{upper}} = Q_{(M_2)}$$

#### Definitions
- $Z_{1 - \alpha/2}$: Two-tailed standard normal critical value ($1.959964$ for $\alpha = 0.05$).
- $\text{Var}(S)$: Theoretical variance of Mann-Kendall score.
- $Q_{(r)}$: The $r$-th ordered (sorted) pairwise slope in ascending order ($Q_{(1)} \le Q_{(2)} \le \dots \le Q_{(N)}$).

#### Numerical Example (Cairo Annual T2M, $n = 43, N = 903$)
- $\text{Var}(S) = 9129.33 \implies \sqrt{\text{Var}(S)} = 95.548$.
- $C_{0.05} = 1.959964 \times 95.54754 = 187.27$.
- $M_1 = \lfloor (903 - 187.27) / 2 \rfloor = \lfloor 357.86 \rfloor = 357$.
- $M_2 = \lceil (903 + 187.27) / 2 \rceil + 1 = \lceil 545.14 \rceil + 1 = 547$.
- Sorting all 903 pairwise slopes:
  - $Q_{(357)} = \mathbf{+0.020238^\circ\text{C / yr}}$ ($+0.2024^\circ\text{C / decade}$).
  - $Q_{(547)} = \mathbf{+0.038889^\circ\text{C / yr}}$ ($+0.3889^\circ\text{C / decade}$).
- Analytical 95% CI: $[+0.202, +0.389]^\circ\text{C / decade}$.
- **Validation**: Because the lower bound is strictly greater than 0 ($+0.202 > 0$), positive monotonic warming is confirmed at the 95% confidence level.

---

### 2.7 Hamed & Rao (1998) Modified Mann-Kendall Variance Correction

#### Formula
$$\text{Var}^*(S) = \text{Var}(S) \cdot \frac{n}{n^*}$$

$$\frac{n}{n^*} = 1 + \frac{2}{n(n - 1)(n - 2)} \sum_{i=1}^{n-1} (n - i)(n - i - 1)(n - i - 2) \rho_i$$

Where $\rho_i$ is the rank autocorrelation of the detrended series $x^*_t = x_t - \beta \cdot t$:
$$\rho_i = \frac{\frac{1}{n} \sum_{t=1}^{n-i} (R_t - \bar{R})(R_{t+i} - \bar{R})}{\frac{1}{n} \sum_{t=1}^n (R_t - \bar{R})^2}$$

#### Lag Significance Criterion (Eq. 11 of Hamed & Rao 1998)
The summation $\sum$ includes only lags $i$ where:
$$|\rho_i| > \frac{Z_{1 - \alpha/2}}{\sqrt{n}} = \frac{1.959964}{\sqrt{n}}$$

#### Modified Test Statistic
$$Z^* = \frac{S - \text{sign}(S)}{\sqrt{\text{Var}^*(S)}} \quad (S \ne 0)$$

#### Intuition & Purpose
Serial autocorrelation (persistence) increases the sample variance of $S$, causing standard Mann-Kendall tests to reject the null hypothesis too often (inflated Type I error). Hamed & Rao (1998) proved that detrending the series and incorporating significant rank autocorrelations inflates $\text{Var}(S)$ to $\text{Var}^*(S)$, deflating $Z^*$ back to an unbiased level.

---

### 2.8 Hirsch-Slack Seasonal Mann-Kendall Omnibus Test

#### Formula
$$S_g = \sum_{k=1}^{n_g - 1} \sum_{j=k+1}^{n_g} \text{sgn}(x_{g,j} - x_{g,k}) \quad \text{for season } g \in \{1, \dots, m\}$$

$$S_{\text{tot}} = \sum_{g=1}^m S_g$$

$$\text{Var}(S_{\text{tot}}) = \sum_{g=1}^m \text{Var}(S_g) + \sum_{g=1}^m \sum_{h \ne g} \text{Cov}(S_g, S_h)$$

Under the assumption of mutual independence between seasons ($\text{Cov}(S_g, S_h) = 0$):
$$\text{Var}(S_{\text{tot}}) = \sum_{g=1}^m \text{Var}(S_g)$$

$$Z_{\text{seasonal}} = \frac{S_{\text{tot}} - \text{sign}(S_{\text{tot}})}{\sqrt{\text{Var}(S_{\text{tot}})}}$$

$$\beta_{\text{seasonal}} = \text{median}\left(\bigcup_{g=1}^m \{Q_{j,k}^{(g)}\}\right)$$

#### Units Consideration
$\beta_{\text{seasonal}}$ represents change per **year** (per full cycle of length $m$), **not** per month. Decadal rate is computed as $\beta_{\text{seasonal}} \times 10$, not $\times 120$.

---

### 2.9 Multiple Testing Corrections for 12 Calendar Months

When testing $m = 12$ calendar months simultaneously:
$$\alpha_{\text{family}} \le 1 - (1 - \alpha)^m = 1 - (0.95)^{12} \approx \mathbf{45.96\%}$$

To prevent spurious discoveries:

#### 1. Holm-Bonferroni Method (Family-Wise Error Rate Control)
Order unadjusted $p$-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.  
Adjusted $p$-value for the $k$-th hypothesis:
$$p_{(k)}^{\text{Holm}} = \min\left(1, \max_{j \le k} \left[(m - j + 1) \cdot p_{(j)}\right]\right)$$
Reject $H_{0,(k)}$ if $p_{(k)}^{\text{Holm}} < \alpha$.

#### 2. Benjamini-Hochberg Method (False Discovery Rate Control)
Controls the expected proportion of false discoveries: $\text{FDR} \le q$.  
Adjusted $p$-value for the $k$-th hypothesis:
$$p_{(k)}^{\text{BH}} = \min\left(1, \min_{j \ge k} \left[\frac{m}{j} \cdot p_{(j)}\right]\right)$$
Reject $H_{0,(k)}$ if $p_{(k)}^{\text{BH}} < q$.

---

### 2.10 Cumulative Warming Formulations

To eliminate ambiguity, EarthTrend Explorer explicitly differentiates four cumulative change calculations:

1. **Theil-Sen Elapsed Change (Primary)**:
   $$\Delta T_{\text{Sen}} = \beta \cdot (T_{\text{end}} - T_{\text{start}}) = 0.029267 \times 42 = \mathbf{+1.2292^\circ\text{C}} \approx \mathbf{+1.23^\circ\text{C}}$$
2. **Raw Endpoint Difference**:
   $$\Delta T_{\text{endpoint}} = T(t_n) - T(t_1) = 22.6058 - 20.9725 = \mathbf{+1.6333^\circ\text{C}} \approx \mathbf{+1.63^\circ\text{C}}$$
3. **Parametric OLS Linear Fitted Change**:
   $$\Delta T_{\text{OLS}} = m \cdot (T_{\text{end}} - T_{\text{start}}) = 0.030525 \times 42 = \mathbf{+1.2821^\circ\text{C}} \approx \mathbf{+1.28^\circ\text{C}}$$
4. **Theil-Sen Sample Count Multiplier (Fencepost Discrepancy)**:
   $$\Delta T_{\text{fencepost}} = \beta \cdot n = 0.029267 \times 43 = \mathbf{+1.2585^\circ\text{C}} \approx \mathbf{+1.26^\circ\text{C}}$$
5. **Decadal Baseline Climatological Shift**:
   $$\Delta T_{\text{decadal}} = \bar{T}_{2014\text{–}2023} - \bar{T}_{1981\text{–}1990} = 21.9460 - 20.9988 = \mathbf{+0.9473^\circ\text{C}} \approx \mathbf{+0.95^\circ\text{C}}$$

---

## 3. Technical Decisions & Alternatives Comparison Table

| Method / Component | Status | Selection Reason | Rejected Alternative | Rejection Reason | Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Parametric Mann-Kendall** | Implemented & Verified | Distribution-free, handles non-normal climate data, robust to outliers | OLS regression as sole method | OLS assumes normal, homoscedastic errors and is sensitive to individual extreme anomalies | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Theil-Sen Robust Slope** | Implemented & Verified | 29.3% breakdown point against outliers; median pairwise slope | Mean pairwise slope (least squares slope) | Distorted by extreme individual years | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Hamed & Rao (1998) Correction** | Implemented & Verified | Corrects variance for serial correlation in detrended ranks without arbitrary pre-whitening | Pre-whitening (e.g., Kulkarni & von Storch 1995) | Pre-whitening removes real trend slope along with AR(1) noise, deflating trend magnitude | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Benjamini-Hochberg FDR** | Implemented & Verified | Controls false discovery rate across 12 monthly tests without being overly conservative | Standard Bonferroni | Standard Bonferroni ($p < \alpha/m$) is excessively conservative, causing high Type II error rates | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **Hirsch-Slack Seasonal MK** | Implemented & Qualified | Evaluates multi-decadal monthly series by comparing like months across years | Deseasonalizing by subtracting monthly mean + standard MK | Subtraction assumes constant variance and linear additive seasonal cycle | [`src/trend_analysis.py`](file:///D:/Space-Apps/src/trend_analysis.py) |
| **WMO 1981–2010 Baseline Normal** | Implemented & Verified | Standard WMO 30-year climatological normal period | Arbitrary whole-record mean | Whole-record mean artificially forces anomaly sum to zero and confounds trend detection | [`src/preprocessing.py`](file:///D:/Space-Apps/src/preprocessing.py) |
| **Streamlit Native AppTest** | Implemented & Verified | Fast headless CI testing of all UI pages and widgets | Selenium / Playwright browser automation | Heavyweight, brittle headless browser dependencies requiring webdriver binaries | [`tests/test_streamlit_app.py`](file:///D:/Space-Apps/tests/test_streamlit_app.py) |
| **Multivariate Slope Interaction Model** | Investigated, Not Implemented | Required to formally prove whether Aswan warms significantly faster than Cairo | Raw slope point comparison | Comparing slopes without an interaction test is statistically invalid when CIs overlap | Planned Phase |
