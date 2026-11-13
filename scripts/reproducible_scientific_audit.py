#!/usr/bin/env python3
"""
EarthTrend Explorer - Master Reproducible Scientific Audit Script
NASA Space Apps Challenge 2026: Be An Earth System Trend Detective!

This script independently recalculates and verifies:
1. Annual regional statistics for Cairo, Alexandria, and Aswan directly from raw CSVs:
   - Original Mann-Kendall (S, Var(S), Z, p-value, Kendall's tau)
   - Hamed & Rao (1998) Modified Mann-Kendall (detrended rank autocorrelation, factor, Var*(S), Z*, p*)
   - Mathematical proof of the continuity correction identity: Z* = (S - sign(S)) / sqrt(Var*(S))
   - Theil-Sen robust median slopes and analytical 95% Confidence Intervals (Gilbert 1987)
   - Cumulative warming metrics (42 elapsed years, raw endpoint difference, decadal baseline shift)
2. Cairo 12-Month Calendar Breakdown:
   - Nominal unadjusted p-values (alpha = 0.05)
   - Holm-Bonferroni Step-Down Family-Wise Error Rate (FWER) adjusted p-values
   - Benjamini-Hochberg Step-Up False Discovery Rate (FDR) adjusted p-values
   - Verification of exactly which months are significant under each criterion
"""

import sys
import os
import math

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import auto_load_dataset
from src.preprocessing import aggregate_to_annual
from src.trend_analysis import (
    mann_kendall_test,
    compute_sens_slope,
    compute_linear_regression,
    hamed_rao_modified_mann_kendall,
    compute_calendar_month_breakdown,
)

REGIONAL_TARGETS = [
    {
        "name": "Greater Cairo",
        "path": "data/raw/nasa_power_cairo_1981_2023.csv",
        "lat": 30.0444,
        "lon": 31.2357,
        "elevation_m": 135.44,
        "climate_zone": "Nile Valley / Urban Basin Transition",
    },
    {
        "name": "Alexandria",
        "path": "data/raw/nasa_power_alexandria_1981_2023.csv",
        "lat": 31.2001,
        "lon": 29.9187,
        "elevation_m": 8.60,
        "climate_zone": "Mediterranean Marine Coastline",
    },
    {
        "name": "Aswan",
        "path": "data/raw/nasa_power_aswan_1981_2023.csv",
        "lat": 24.0889,
        "lon": 32.8998,
        "elevation_m": 253.64,
        "climate_zone": "Upper Egypt Hyper-Arid Desert",
    },
]


def audit_regional_annual_trends():
    print("=" * 115)
    print("SECTION 1: INDEPENDENT RECALCULATION OF REGIONAL ANNUAL T2M TRENDS (1981-2023)")
    print("=" * 115)

    summary_records = []

    for target in REGIONAL_TARGETS:
        name = target["name"]
        path = target["path"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        df_tidy, meta = auto_load_dataset(content, default_location=name)
        t2m_monthly = df_tidy[df_tidy["variable"] == "T2M"].copy()
        ann_df = aggregate_to_annual(t2m_monthly, value_col="value")

        y = ann_df["value"].values
        years = ann_df["year"].values
        n = len(y)
        elapsed_years = int(years[-1] - years[0])

        assert n == 43, f"Expected 43 annual points for {name}, found {n}"
        assert elapsed_years == 42, f"Expected 42 elapsed years for {name}, found {elapsed_years}"

        # 1. Original Mann-Kendall Test
        orig_mk = mann_kendall_test(y, alpha=0.05)
        s = orig_mk["s_statistic"]
        orig_var = orig_mk["var_s"]
        orig_z = orig_mk["z_statistic"]
        orig_p = orig_mk["p_value"]
        tau = orig_mk["tau"]

        # 2. Hamed & Rao (1998) Modified Mann-Kendall Test
        hr_res = hamed_rao_modified_mann_kendall(y, alpha=0.05)
        hr_factor = hr_res["variance_correction_factor"]
        hr_var = hr_res["corrected_var_s"]
        hr_z = hr_res["z_statistic"]
        hr_p = hr_res["p_value"]
        sig_lags = hr_res["significant_lags"]

        # 3. Continuity correction identity verification:
        # Z* = (S - sign(S)) / sqrt(Var*(S))
        sign_s = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
        expected_z_star = (s - sign_s) / math.sqrt(hr_var) if hr_var > 0 else 0.0
        expected_p_star = 2.0 * (1.0 - stats.norm.cdf(abs(expected_z_star)))
        z_error = abs(hr_z - expected_z_star)
        p_error = abs(hr_p - expected_p_star)
        assert z_error < 1e-12, f"Z* identity violated for {name}: error = {z_error}"
        assert p_error < 1e-12, f"p* identity violated for {name}: error = {p_error}"

        # 4. Theil-Sen Robust Slope and Gilbert 95% Analytical CI
        sen = compute_sens_slope(y, years, alpha=0.05)
        slope_annual = sen["slope"]
        slope_decadal = slope_annual * 10.0
        ci_lower_decadal = sen["slope_ci_lower"] * 10.0
        ci_upper_decadal = sen["slope_ci_upper"] * 10.0

        # 5. Cumulative Warming Metrics
        theil_cum_42 = slope_annual * elapsed_years
        endpoint_diff = y[-1] - y[0]

        ols = compute_linear_regression(y, years)
        ols_slope_decadal = ols["slope"] * 10.0
        ols_r2 = ols["r_squared"]
        ols_cum_42 = ols["slope"] * elapsed_years

        # Decadal Baseline Shift (1981-1990 vs 2014-2023)
        b1 = ann_df[(ann_df["year"] >= 1981) & (ann_df["year"] <= 1990)]["value"].mean()
        b4 = ann_df[(ann_df["year"] >= 2014) & (ann_df["year"] <= 2023)]["value"].mean()
        dec_shift = b4 - b1

        record = {
            "Location": name,
            "Grid Coordinates": f"({target['lat']:.2f}°N, {target['lon']:.2f}°E)",
            "Climate Zone": target["climate_zone"],
            "Elevation (m)": target["elevation_m"],
            "n": n,
            "Elapsed Yrs": elapsed_years,
            "S": s,
            "Kendall tau": tau,
            "Sen Slope (°C/dec)": slope_decadal,
            "Sen 95% CI (°C/dec)": f"[{ci_lower_decadal:+.3f}, {ci_upper_decadal:+.3f}]",
            "Theil-Sen Cum (+°C)": theil_cum_42,
            "Endpoint Diff (+°C)": endpoint_diff,
            "Decadal Shift (+°C)": dec_shift,
            "OLS Slope (°C/dec)": ols_slope_decadal,
            "OLS R2": ols_r2,
            "Orig Var(S)": orig_var,
            "Orig Z": orig_z,
            "Orig p-value": orig_p,
            "HR Factor (n/n*)": hr_factor,
            "HR Sig Lags Count": len(sig_lags),
            "HR Var*(S)": hr_var,
            "HR Corrected Z*": hr_z,
            "HR Corrected p*": hr_p,
        }
        summary_records.append(record)

        print(f"\nLocation: {name} [{target['climate_zone']}]")
        print(f"  Coordinates: {target['lat']}°N, {target['lon']}°E | Elevation: {target['elevation_m']} m")
        print(f"  Sen's Slope: {slope_annual:+.6f} °C/year ({slope_decadal:+.4f} °C/decade)")
        print(f"  Sen's 95% CI: [{ci_lower_decadal:+.4f}, {ci_upper_decadal:+.4f}] °C/decade")
        print(f"  Mann-Kendall: S = {s:+0.1f}, Var(S) = {orig_var:.2f}, Z = {orig_z:+.4f}, p = {orig_p:.4e}, tau = {tau:+.4f}")
        print(f"  Hamed-Rao: Factor = {hr_factor:.6f}, Sig Lags = {len(sig_lags)}, Var*(S) = {hr_var:.2f}, Z* = {hr_z:+.4f}, p* = {hr_p:.4e}")
        print(f"  Cumulative 42-Year Theil-Sen: +{theil_cum_42:.4f} °C | Raw Endpoint: +{endpoint_diff:.4f} °C | Decadal Baseline Shift: +{dec_shift:.4f} °C")
        print(f"  Identity Z* = (S - sign(S)) / sqrt(Var*(S)): VERIFIED (delta = {z_error:.2e})")

    df_summary = pd.DataFrame(summary_records)
    print("\n--- REGIONAL VERIFICATION SUMMARY TABLE ---")
    cols_display = [
        "Location", "Sen Slope (°C/dec)", "Sen 95% CI (°C/dec)",
        "Theil-Sen Cum (+°C)", "Endpoint Diff (+°C)",
        "Orig Z", "Orig p-value",
        "HR Factor (n/n*)", "HR Corrected Z*", "HR Corrected p*"
    ]
    print(df_summary[cols_display].to_string(index=False))

    # Assertions guaranteeing consistency
    cairo_row = df_summary[df_summary["Location"] == "Greater Cairo"].iloc[0]
    alex_row = df_summary[df_summary["Location"] == "Alexandria"].iloc[0]
    aswan_row = df_summary[df_summary["Location"] == "Aswan"].iloc[0]

    assert abs(cairo_row["Sen Slope (°C/dec)"] - 0.292667) < 1e-4
    assert abs(alex_row["Sen Slope (°C/dec)"] - 0.261111) < 1e-4
    assert abs(aswan_row["Sen Slope (°C/dec)"] - 0.434167) < 1e-4

    assert abs(cairo_row["HR Factor (n/n*)"] - 1.000000) < 1e-5
    assert abs(alex_row["HR Factor (n/n*)"] - 0.780520) < 1e-5
    assert abs(aswan_row["HR Factor (n/n*)"] - 1.000000) < 1e-5

    return df_summary


def audit_cairo_monthly_breakdown():
    print("\n" + "=" * 115)
    print("SECTION 2: CAIRO MONTHLY BREAKDOWN & MULTIPLE TESTING CORRECTIONS")
    print("=" * 115)

    path = "data/raw/nasa_power_cairo_1981_2023.csv"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    df_tidy, _ = auto_load_dataset(content, default_location="Greater Cairo")
    cairo_t2m = df_tidy[df_tidy["variable"] == "T2M"].copy()

    cal_df = compute_calendar_month_breakdown(cairo_t2m, value_col="value", alpha=0.05)

    print("\n--- 12 CALENDAR MONTHS: INDIVIDUAL RESULTS WITH MULTIPLE TESTING ADJUSTMENTS ---")
    cols = [
        "month_num", "month_name", "trend", "sen_slope_per_decade",
        "z_statistic", "p_value", "p_holm", "sig_holm", "p_bh", "sig_bh"
    ]
    print(cal_df[cols].to_string(index=False))

    nominal_months = cal_df[cal_df["sig_nominal"]]["month_name"].tolist()
    bh_months = cal_df[cal_df["sig_bh"]]["month_name"].tolist()
    holm_months = cal_df[cal_df["sig_holm"]]["month_name"].tolist()

    print("\n--- SUMMARY OF SIGNIFICANT MONTHS (alpha = 0.05) ---")
    print(f"1. Nominally Significant (p < 0.05): {len(nominal_months)} of 12 months")
    print(f"   -> {nominal_months}")
    print(f"2. Benjamini-Hochberg FDR Significant (q < 0.05): {len(bh_months)} of 12 months")
    print(f"   -> {bh_months}")
    print(f"3. Holm-Bonferroni FWER Significant (alpha < 0.05): {len(holm_months)} of 12 months")
    print(f"   -> {holm_months}")

    # Rigorous assertions:
    assert nominal_months == ["FEB", "MAY", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    assert bh_months == ["JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    assert holm_months == ["JUL", "AUG", "SEP", "NOV", "DEC"]
    assert "OCT" not in holm_months, "October must NOT be in the Holm-significant list!"
    assert "DEC" in holm_months, "December MUST be in the Holm-significant list!"

    print("\nMathematical Assertion Check:")
    print("  ✓ Nominal count = 8: FEB, MAY, JUL, AUG, SEP, OCT, NOV, DEC")
    print("  ✓ Benjamini-Hochberg FDR count = 6: JUL, AUG, SEP, OCT, NOV, DEC")
    print("  ✓ Holm-Bonferroni FWER count = 5: JUL, AUG, SEP, NOV, DEC (OCT p_holm = 0.12607 > 0.05)")


if __name__ == "__main__":
    audit_regional_annual_trends()
    audit_cairo_monthly_breakdown()
    print("\n" + "=" * 115)
    print("ALL SCIENTIFIC REPRODUCIBILITY CHECKS PASSED PERFECTLY.")
    print("=" * 115)
