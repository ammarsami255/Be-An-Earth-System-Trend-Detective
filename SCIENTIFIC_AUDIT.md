# Scientific Audit Report: EarthTrend Explorer
**Project**: EarthTrend Explorer — NASA Space Apps Challenge 2026  
**Challenge**: *"Be An Earth System Trend Detective!"*  
**Auditor**: Independent Senior Climate Data Scientist & Statistical Reviewer  
**Audit Date**: November 13, 2026  
**Workspace Audited**: `D:\Space-Apps`  
**Repository Branch**: `master` (Git commit: `48d9b81`)

---

## 1. Executive Summary

This independent scientific audit evaluated the data integrity, statistical engines, unit consistency, and scientific claims of the **EarthTrend Explorer** web application. 

### Summary Verdict
- **Core Scientific Finding**: **VERIFIED**. The underlying climate warming trend for Greater Cairo, Egypt ($1981\text{–}2023$) is genuine, statistically robust, and reproducible from raw NASA MERRA-2 data ($p < 10^{-7}$).
- **Data Authenticity**: **VERIFIED**. Datasets for Cairo, Alexandria, and Aswan originate directly from NASA POWER (MERRA-2 atmospheric assimilation model) at native $0.5^\circ \times 0.625^\circ$ grid resolution. No synthetic or fabricated data was used.
- **Statistical Nuance & Reporting Errors**: **CORRECTIONS IDENTIFIED & REMEDIATED**.
  1. *Statistic/Sample Size Conflation*: The previous project summary conflated annual test statistics ($S = +512$, $Z = +5.35$) with the total monthly sample size ($n = 516$), and misstated $S$ as $+555$.
  2. *Seasonal Generalization Error*: The previous report claimed warming was verified across *"all 12 individual calendar months"*. Independent decomposition shows **8 of 12 months** have statistically significant trends ($p < 0.05$), while **4 months** (January, March, April, June) do not.
  3. *Unit Scaling Bug in Seasonal Sen's Slope*: PyMannKendall's `seasonal_test` returns Sen's slope in units per annual cycle ($^\circ\text{C/year}$). The UI previously multiplied this value by $120$ instead of $10$, displaying an exaggerated $+3.63^\circ\text{C/decade}$ instead of the true $+0.303^\circ\text{C/decade}$.
  4. *Elapsed Time vs. Sample Count*: Cumulative warming was previously calculated using $N = 43$ points ($+1.26^\circ\text{C}$) rather than $42$ elapsed years ($+1.23^\circ\text{C}$).
  5. *Autocorrelation Resolution*: Raw annual lag-1 autocorrelation ($r_1 = +0.326$) exceeds Bartlett's white noise bound ($0.299$). However, upon detrending, the residual autocorrelation drops to $-0.248$ (non-significant), confirming via Hamed-Rao testing that the serial persistence is driven by the monotonic trend rather than autoregressive red noise.

All defects have been patched in `src/trend_analysis.py`, `src/visualization.py`, and `app.py`, with 18 automated unit and integration tests passing.

---

## 2. Data Provenance

| Property | Cairo File | Alexandria File | Aswan File |
| :--- | :--- | :--- | :--- |
| **File Path** | `data/raw/nasa_power_cairo_1981_2023.csv` | `data/raw/nasa_power_alexandria_1981_2023.csv` | `data/raw/nasa_power_aswan_1981_2023.csv` |
| **Source Provider** | NASA Langley Research Center (LaRC) | NASA Langley Research Center (LaRC) | NASA Langley Research Center (LaRC) |
| **Underlying Model** | NASA MERRA-2 (GMAO Atmospheric Assimilation) | NASA MERRA-2 (GMAO Atmospheric Assimilation) | NASA MERRA-2 (GMAO Atmospheric Assimilation) |
| **Query Mechanism** | NASA POWER Point API (format=CSV) | NASA POWER Point API (format=CSV) | NASA POWER Point API (format=CSV) |
| **Latitude / Longitude** | $30.0444^\circ\text{N}, 31.2357^\circ\text{E}$ | $31.2001^\circ\text{N}, 29.9187^\circ\text{E}$ | $24.0889^\circ\text{N}, 32.8998^\circ\text{E}$ |
| **Grid Cell Elevation** | $135.44\text{ meters}$ | $8.60\text{ meters}$ | $253.64\text{ meters}$ |
| **Spatial Nature** | **Single $0.5^\circ \times 0.625^\circ$ MERRA-2 Grid Cell** | **Single $0.5^\circ \times 0.625^\circ$ MERRA-2 Grid Cell** | **Single $0.5^\circ \times 0.625^\circ$ MERRA-2 Grid Cell** |
| **Audit Classification** | **VERIFIED** | **VERIFIED** | **VERIFIED** |

> [!IMPORTANT]
> **Spatial Representation Finding**: These datasets represent atmospheric model grid-box assimilations covering approximately $50\text{ km} \times 60\text{ km}$ at each coordinate point. They are **not** administrative city boundary averages. Describing them as municipal spatial averages is inaccurate; they must be identified as single-grid-cell extractions.

---

## 3. Dataset Quality

Each raw CSV file contains 197 lines formatted with an initial 12-line NASA metadata banner followed by a matrix of 172 rows (4 parameters $\times$ 43 years).

| Parameter | Description | Units | Temporal Coverage | Missing (-999) | NaNs | Duplicates | Range [Min, Max] | Mean ± Std |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cairo T2M** | 2m Air Temp | °C | 1981-01 to 2023-12 (516 mo) | 0 | 0 | 0 | $[9.62, 31.67]$ | $21.51 \pm 6.49$ |
| **Cairo T2M_MAX** | 2m Max Temp | °C | 1981-01 to 2023-12 (516 mo) | 0 | 0 | 0 | $[17.84, 46.55]$ | $35.29 \pm 6.89$ |
| **Cairo T2M_MIN** | 2m Min Temp | °C | 1981-01 to 2023-12 (516 mo) | 0 | 0 | 0 | $[0.44, 21.46]$ | $11.14 \pm 5.89$ |
| **Cairo PRECTOTCORR** | Corrected Precip | mm/day | 1981-01 to 2023-12 (516 mo) | 0 | 0 | 0 | $[0.00, 6.22]$ | $0.17 \pm 0.46$ |
| **Alexandria T2M** | 2m Air Temp | °C | 1981-01 to 2023-12 (516 mo) | 0 | 0 | 0 | $[12.19, 29.63]$ | $21.11 \pm 5.15$ |
| **Aswan T2M** | 2m Air Temp | °C | 1981-01 to 2023-12 (516 mo) | 0 | 0 | 0 | $[10.28, 36.56]$ | $25.66 \pm 7.01$ |

**Audit Classification**: **VERIFIED**. All three datasets contain complete, unbroken temporal series spanning exactly 43 years (516 monthly values each) with zero missing entries and zero out-of-bounds readings.

---

## 4. Statistical Verification

Independent recalculation was performed using `scipy.stats` and `pymannkendall` directly on the raw extracted series.

### Detailed Test Matrix for Greater Cairo Temperature (`T2M`)

| Analysis Stream | Series Type | $n$ | Date Range | Kendall $S$ | $\text{Var}(S)$ | Standardized $Z$ | Kendall $\tau$ | $p$-value | Sen's Slope | OLS Slope ($R^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Annual (NASA ANN)** | Annual Mean | 43 | 1981–2023 | $+507.0$ | $9,125.67$ | $+5.297$ | $+0.561$ | $1.18 \times 10^{-7}$ | $+0.0294^\circ\text{C/yr}$ | $+0.0305^\circ\text{C/yr}$ ($0.504$) |
| **Annual (Calc Mean)** | Annual Mean | 43 | 1981–2023 | $+512.0$ | $9,129.33$ | $+5.348$ | $+0.567$ | $8.89 \times 10^{-8}$ | $+0.0293^\circ\text{C/yr}$ | $+0.0305^\circ\text{C/yr}$ ($0.503$) |
| **Raw Monthly Series** | Monthly Steps | 516 | 1981–2023 | $+7,156.0$ | $1.53 \times 10^7$ | $+1.829$ | $+0.054$ | $0.0675$ (N.S.) | $+0.0029^\circ\text{C/mo}$ | $+0.0029^\circ\text{C/mo}$ ($0.004$) |
| **Seasonal MK (Hirsch-Slack)** | Monthly Cycles | 516 | 1981–2023 | $+2,665.0$ | $109,531.67$ | $+8.049$ | $+0.246$ | $8.88 \times 10^{-16}$ | $+0.0303^\circ\text{C/yr}$ | $+0.0305^\circ\text{C/yr}$ ($0.503$) |

### Discrepancy Clarification
- **The "S = +555" Claim**: The previous project report cited $S = +555$. Re-computation shows the actual Kendall score for the 43 annual mean temperatures is **$S = +512.0$** (calculated 12-month mean) or **$S = +507.0$** (NASA ANN column). The Kendall's Tau calculation: $\tau = 512 / (43 \times 42 / 2) = 512 / 903 = +0.5670$. The value $+555$ was an inadvertent typo in the previous summary table.
- **Monthly vs. Annual Conflation**: Applying a non-seasonal Mann-Kendall test directly to 516 raw monthly observations yields **$p = 0.0675 \ge 0.05$** ($Z = +1.83$), failing to achieve significance because the $20^\circ\text{C}$ annual cycle drowns out the monotonic slope. The $Z = +5.35$ and $p = 8.89 \times 10^{-8}$ values strictly belong to the **Annual series ($n = 43$)**.

**Audit Classification**: **NEEDS CORRECTION** (Remediated in code and UI).

---

## 5. Sen's Slope and Units

### Mathematical Formulation
The Theil-Sen estimator computes the median of all pairwise differences:
$$Q_{ij} = \frac{y_j - y_i}{x_j - x_i} \quad \text{for } 1 \le i < j \le n$$

### Independent Verification of Rates
- **Annual Sen's Slope**: $+0.029267^\circ\text{C / year}$
- **Decadal Sen's Rate**: $+0.29267^\circ\text{C / decade} \approx \mathbf{+0.293^\circ\text{C / decade}}$
- **Analytical 95% Confidence Interval**:
  $$M_1 = \frac{N' - C_\alpha}{2}, \quad M_2 = \frac{N' + C_\alpha}{2} + 1, \quad C_\alpha = 1.96 \cdot \sqrt{\text{Var}(S)}$$
  For $n = 43, N' = 903, \sqrt{\text{Var}(S)} = 95.55 \implies C_\alpha = 187.27$:
  $$\text{Rank Lower} = 357, \quad \text{Rank Upper} = 546$$
  $$95\%\text{ CI} = [+0.020238, +0.038889]^\circ\text{C/year} \implies \mathbf{[+0.202, +0.389]^\circ\text{C/decade}}$$
- **Cumulative Warming**:
  - Over $42$ elapsed years ($2023 - 1981 = 42$):
    $$\Delta T_{\text{Sen}} = 0.029267 \times 42 = \mathbf{+1.229^\circ\text{C}} \approx \mathbf{+1.23^\circ\text{C}}$$
    $$\Delta T_{\text{OLS}} = 0.030525 \times 42 = \mathbf{+1.282^\circ\text{C}} \approx \mathbf{+1.28^\circ\text{C}}$$
  - If multiplied by observation count ($N = 43$):
    $$\Delta T = 0.029267 \times 43 = +1.258^\circ\text{C} \approx +1.26^\circ\text{C}$$

> [!WARNING]
> **Unit Defect in Seasonal UI**: PyMannKendall `seasonal_test` calculates slopes across like seasons ($j - k$ in years). The output `res.slope = 0.030286` is already in $^\circ\text{C/year}$. Multiplying by 120 resulted in an erroneous display of $+3.634^\circ\text{C/decade}$. The correct decadal multiplier is $10$, yielding **$+0.303^\circ\text{C/decade}$**.

**Audit Classification**: **NEEDS CORRECTION** (Remediated in `app.py`).

---

## 6. Seasonal Analysis (Month-by-Month Breakdown)

To test the previous claim that *"warming is verified across all 12 individual calendar months"*, each month was audited independently across all 43 years ($n = 43$ each, $\alpha = 0.05$):

| Month | Climatological Mean | MK Trend | $p$-value | $Z$-statistic | Kendall $\tau$ | Sen's Slope (°C/dec) | OLS Slope (°C/dec) | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **January** | $11.95^\circ\text{C}$ | No trend | $0.3516$ | $+0.93$ | $+0.100$ | $+0.172$ | $+0.173$ | **NO** |
| **February** | $13.14^\circ\text{C}$ | Increasing | $0.0382$ | $+2.07$ | $+0.220$ | $+0.333$ | $+0.348$ | **YES** |
| **March** | $16.27^\circ\text{C}$ | No trend | $0.0920$ | $+1.69$ | $+0.179$ | $+0.327$ | $+0.350$ | **NO** |
| **April** | $20.83^\circ\text{C}$ | No trend | $0.8670$ | $-0.17$ | $-0.019$ | $-0.042$ | $+0.025$ | **NO** |
| **May** | $25.14^\circ\text{C}$ | Increasing | $0.0467$ | $+1.99$ | $+0.212$ | $+0.255$ | $+0.267$ | **YES** |
| **June** | $28.37^\circ\text{C}$ | No trend | $0.0940$ | $+1.67$ | $+0.178$ | $+0.215$ | $+0.211$ | **NO** |
| **July** | $29.75^\circ\text{C}$ | Increasing | $0.00006$ | $+4.01$ | $+0.425$ | $+0.328$ | $+0.322$ | **YES** |
| **August** | $29.65^\circ\text{C}$ | Increasing | $0.00009$ | $+3.91$ | $+0.414$ | $+0.370$ | $+0.377$ | **YES** |
| **September** | $27.29^\circ\text{C}$ | Increasing | $0.00087$ | $+3.33$ | $+0.353$ | $+0.358$ | $+0.398$ | **YES** |
| **October** | $23.48^\circ\text{C}$ | Increasing | $0.0180$ | $+2.37$ | $+0.251$ | $+0.346$ | $+0.311$ | **YES** |
| **November** | $18.39^\circ\text{C}$ | Increasing | $0.00157$ | $+3.16$ | $+0.336$ | $\mathbf{+0.504}$ | $+0.489$ | **YES (Fastest)** |
| **December** | $13.83^\circ\text{C}$ | Increasing | $0.00456$ | $+2.84$ | $+0.301$ | $+0.385$ | $+0.393$ | **YES** |

### Key Scientific Insights
1. **Warming is present in 8 of the 12 calendar months**, primarily from mid-summer through early winter (July to December).
2. **November is the fastest warming month** in Greater Cairo at $+0.504^\circ\text{C/decade}$.
3. **Late winter and spring (Jan, Mar, Apr, Jun) do not show statistically significant monotonic warming**. April exhibits virtually zero trend ($-0.042^\circ\text{C/decade}$, $p = 0.867$).
4. The claim that *"all 12 individual months exhibit warming"* was an overgeneralized deduction from aggregate Seasonal Mann-Kendall test.

**Audit Classification**: **NEEDS CORRECTION** (Explicit 12-month breakdown table added to UI).

---

## 7. Autocorrelation & Serial Persistence

### Audit of Autocorrelation Diagnostics
- **Series Examined**: Cairo Annual Mean Temperature ($n = 43$).
- **Bartlett's Two-Tailed 95% White Noise Bound**:
  $$\text{CI}_{95} = \pm \frac{1.96}{\sqrt{43}} = \pm \mathbf{0.2989}$$
- **Raw Lag-1 Autocorrelation**:
  $$r_1 = \mathbf{+0.3261} > 0.2989 \quad \text{(Statistically Significant)}$$
- **Detrended OLS Residual Lag-1 Autocorrelation**:
  $$r_{1,\text{res}} = \mathbf{-0.2477} \quad (|{-0.2477}| < 0.2989 \implies \text{Within White Noise Bounds})$$

### Hamed & Rao (1998) Modified Mann-Kendall Test Result
- $\text{Var}^*(S) = 9,129.33$ (Identical to original variance because residual autocorrelation ranks fall below significance).
- $Z^* = +5.348, \quad p = 8.89 \times 10^{-8}$.
- **Scientific Conclusion**: The apparent lag-1 autocorrelation in the raw annual series is an expected byproduct of the monotonic warming trend itself. Once detrended, the residuals show no significant red-noise memory. The observed trend is therefore robust against autoregressive inflation.

**Audit Classification**: **VERIFIED**.

---

## 8. Regional Comparison Audit

The three Egyptian datasets were compared under identical processing pipelines ($1981\text{–}2023$, $43$ annual points each):

| Station / Grid Cell | Latitude / Longitude | Elevation | Climate Regime | Mean Temp | Sen's Slope (°C/dec) | OLS Slope (°C/dec) | Net Change (42 Yrs) | Mann-Kendall $p$-value |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Aswan** | $24.0889^\circ\text{N}, 32.8998^\circ\text{E}$ | $253.6\text{ m}$ | Upper Egypt Hyper-Arid | $25.66^\circ\text{C}$ | $\mathbf{+0.434}$ | $+0.460$ | $\mathbf{+1.82^\circ\text{C}}$ | $8.61 \times 10^{-9}$ |
| **Greater Cairo** | $30.0444^\circ\text{N}, 31.2357^\circ\text{E}$ | $135.4\text{ m}$ | Nile Valley / Delta Basin | $21.51^\circ\text{C}$ | $\mathbf{+0.293}$ | $+0.305$ | $\mathbf{+1.23^\circ\text{C}}$ | $8.89 \times 10^{-8}$ |
| **Alexandria** | $31.2001^\circ\text{N}, 29.9187^\circ\text{E}$ | $8.6\text{ m}$ | Mediterranean Marine Coast | $21.11^\circ\text{C}$ | $\mathbf{+0.261}$ | $+0.281$ | $\mathbf{+1.10^\circ\text{C}}$ | $3.70 \times 10^{-8}$ |

### Physical Climate Gradient
The comparison reveals a coherent geographic gradient:
1. **Aswan (Inland Desert)** warms fastest ($+0.434^\circ\text{C/decade}$), experiencing $+1.82^\circ\text{C}$ net warming due to high solar irradiance and lack of marine moisture buffering.
2. **Greater Cairo (Urban Basin)** exhibits intermediate warming ($+0.293^\circ\text{C/decade}$, $+1.23^\circ\text{C}$ net shift).
3. **Alexandria (Coastal)** warms slowest ($+0.261^\circ\text{C/decade}$, $+1.10^\circ\text{C}$ net shift), buffered by the thermal inertia of the Mediterranean Sea.

**Audit Classification**: **VERIFIED**.

---

## 9. Confirmed Results (Ready for Judges)

1. **Greater Cairo Warming Rate**: $+0.293^\circ\text{C / decade}$ (Theil-Sen median) with $95\%$ CI $[+0.202, +0.389]^\circ\text{C/decade}$.
2. **Cumulative Net Change**: $+1.23^\circ\text{C}$ across the 42 elapsed years from 1981 to 2023.
3. **Statistical Significance**: Mann-Kendall test rejects $H_0$ ($S = +512$, $Z = +5.35$, $p = 8.89 \times 10^{-8}$, $\tau = +0.567$).
4. **Seasonal Hirsch-Slack Test**: Confirms multi-decadal upward trajectory ($Z = +8.05$, $p < 10^{-15}$, Sen's slope $+0.303^\circ\text{C/decade}$).
5. **Autocorrelation Robustness**: Hamed-Rao test confirms trend significance ($p < 10^{-7}$) after ranking detrended residuals.
6. **Cross-Regional Consistency**: All three Egyptian grid cells exhibit statistically significant warming ($p < 10^{-7}$).

---

## 10. Incorrect or Unsupported Claims (Remediated)

| Previous Claim / Code Implementation | Audit Finding | Correction Made |
| :--- | :--- | :--- |
| "516 monthly observations: S = +555, Z = 5.35, p < 0.0001" | Conflated annual statistics with monthly sample size; $S = 512$, not $555$. | Clearly split: Annual series ($n = 43, S = +512, Z = 5.35$); Monthly series ($n = 516, S_{\text{seasonal}} = 2665, Z = 8.05$). |
| "Warming verified across all 12 individual calendar months" | False generalization. 8 months are significant; 4 months (Jan, Mar, Apr, Jun) are non-significant. | Added 12-month calendar breakdown table showing exact per-month p-values. |
| Seasonal Sen's slope displayed as $+3.63^\circ\text{C/decade}$ | Multiplied annual rate by 120 instead of 10. | Changed scaling factor from 120 to 10 in UI, correctly displaying $+0.303^\circ\text{C/decade}$. |
| Cumulative warming reported as $+1.26^\circ\text{C}$ | Multiplied slope by $N = 43$ points instead of 42 elapsed years. | Standardized to $42$ elapsed years ($+1.23^\circ\text{C}$), while noting $N = 43$ points ($+1.26^\circ\text{C}$). |
| City name implied city-wide administrative spatial average | Data represents single MERRA-2 $0.5^\circ \times 0.625^\circ$ grid cells. | Added spatial representation notice in Data Explorer and Regional Comparison pages. |

---

## 11. Recommended Corrections & Implementation Status

All recommended corrections have been implemented and verified:
1. **`src/trend_analysis.py`**:
   - Added `compute_calendar_month_breakdown()` to provide per-month statistics.
   - Added `hamed_rao_modified_mann_kendall()` for autocorrelation-corrected inference.
   - Updated `compute_autocorrelation()` to evaluate both raw and detrended residual ACF.
   - Standardized elapsed time calculation in `synthesize_scientific_answers()`.
2. **`src/visualization.py`**:
   - Updated `plot_autocorrelation_acf()` to display raw vs. detrended residual ACF traces.
3. **`app.py`**:
   - Fixed Seasonal Sen's slope display multiplier (from 120 to 10).
   - Embedded Month-by-Month Calendar Breakdown table in the Trend Analysis tab.
   - Displayed Hamed-Rao modified Mann-Kendall statistics alongside standard Mann-Kendall.
   - Added spatial domain callout notices clarifying native MERRA-2 grid cell scale.
4. **`tests/`**:
   - Added unit tests for seasonal Sen's slope units, detrended autocorrelation, calendar breakdown, and Hamed-Rao test.
   - Total test suite expanded to 18 passing tests.

### Final Auditor Statement
> With these corrections applied, **EarthTrend Explorer** is mathematically accurate, scientifically defensible, transparent regarding its assumptions and limitations, and fully ready to present to NASA Space Apps judges.
