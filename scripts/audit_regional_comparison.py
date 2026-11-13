import os
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

from src.data_loader import auto_load_dataset
from src.preprocessing import aggregate_to_annual
from src.trend_analysis import mann_kendall_test, compute_sens_slope, compute_linear_regression, hamed_rao_modified_mann_kendall

locations = [
    ("Greater Cairo", "data/raw/nasa_power_cairo_1981_2023.csv", 30.0444, 31.2357),
    ("Alexandria", "data/raw/nasa_power_alexandria_1981_2023.csv", 31.2001, 29.9187),
    ("Aswan", "data/raw/nasa_power_aswan_1981_2023.csv", 24.0889, 32.8998),
]

print("=" * 90)
print("REGIONAL COMPARISON AUDIT (CAIRO, ALEXANDRIA, ASWAN)")
print("=" * 90)

results = []

for loc_name, path, lat, lon in locations:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    df_tidy, meta = auto_load_dataset(content, default_location=loc_name)
    
    # 1. Date coverage check
    t2m_tidy = df_tidy[df_tidy["variable"] == "T2M"].copy()
    start_date = t2m_tidy["date"].min()
    end_date = t2m_tidy["date"].max()
    monthly_n = len(t2m_tidy)
    units = t2m_tidy["units"].iloc[0]
    
    # 2. Annual aggregation
    ann_df = aggregate_to_annual(t2m_tidy, value_col="value")
    ann_n = len(ann_df)
    ann_start = ann_df["year"].min()
    ann_end = ann_df["year"].max()
    elapsed_years = ann_end - ann_start
    
    # Values
    y = ann_df["value"].values
    years = ann_df["year"].values
    
    # Recalculate Mann-Kendall
    mk = mann_kendall_test(y, alpha=0.05)
    sen = compute_sens_slope(y, years, alpha=0.05)
    ols = compute_linear_regression(y, years)
    hr = hamed_rao_modified_mann_kendall(y, alpha=0.05)
    
    # Endpoints
    y_first = y[0]  # 1981
    y_last = y[-1]   # 2023
    endpoint_diff = y_last - y_first
    
    # Cumulative warming over 42 elapsed years
    theil_cum_42 = sen["slope"] * elapsed_years
    ols_cum_42 = ols["slope"] * elapsed_years
    
    # Decadal shifts (1981-1990 vs 2014-2023)
    dec1 = ann_df[(ann_df["year"] >= 1981) & (ann_df["year"] <= 1990)]["value"].mean()
    dec4 = ann_df[(ann_df["year"] >= 2014) & (ann_df["year"] <= 2023)]["value"].mean()
    dec_shift = dec4 - dec1
    
    results.append({
        "location": loc_name,
        "grid_cell": f"({lat:.2f}°N, {lon:.2f}°E)",
        "start_date": str(start_date)[:10],
        "end_date": str(end_date)[:10],
        "monthly_n": monthly_n,
        "annual_n": ann_n,
        "elapsed_years": elapsed_years,
        "units": units,
        "s_stat": mk["s_statistic"],
        "z_stat": mk["z_statistic"],
        "tau": mk["tau"],
        "p_val": mk["p_value"],
        "sen_slope_annual": sen["slope"],
        "sen_slope_decadal": sen["slope"] * 10.0,
        "ci_lower_decadal": sen["slope_ci_lower"] * 10.0,
        "ci_upper_decadal": sen["slope_ci_upper"] * 10.0,
        "theil_cum_42": theil_cum_42,
        "ols_slope_decadal": ols["slope"] * 10.0,
        "ols_r2": ols["r_squared"],
        "ols_cum_42": ols_cum_42,
        "y_1981": y_first,
        "y_2023": y_last,
        "endpoint_diff": endpoint_diff,
        "dec_shift": dec_shift,
        "hr_z": hr["z_statistic"],
        "hr_p": hr["p_value"],
        "hr_corr_factor": hr["variance_correction_factor"],
        "hr_sig_lags": hr["significant_lags_count"],
    })

res_df = pd.DataFrame(results)

print("\n--- SUMMARY TABLE OF RECALCULATED REGIONAL STATISTICS (T2M) ---")
summary_cols = [
    "location", "grid_cell", "annual_n", "elapsed_years", "sen_slope_decadal",
    "ci_lower_decadal", "ci_upper_decadal", "theil_cum_42", "endpoint_diff",
    "z_stat", "p_val", "hr_corr_factor"
]
print(res_df[summary_cols].to_string(index=False))

print("\n--- DETAILED BREAKDOWN PER LOCATION ---")
for r in results:
    print(f"\nLocation: {r['location']} [MERRA-2 Grid Cell: {r['grid_cell']}]")
    print(f"  Temporal span: {r['start_date']} to {r['end_date']} (516 months, 43 annual points, 42 elapsed years)")
    print(f"  Sen's Slope: {r['sen_slope_annual']:+.6f} °C/yr ({r['sen_slope_decadal']:+.4f} °C/decade)")
    print(f"  95% Confidence Interval: [{r['ci_lower_decadal']:+.4f}, {r['ci_upper_decadal']:+.4f}] °C/decade")
    print(f"  Theil-Sen Cumulative Change (42 elapsed years): {r['theil_cum_42']:+.4f} °C")
    print(f"  OLS Linear Model: slope = {r['ols_slope_decadal']:+.4f} °C/decade, R² = {r['ols_r2']:.4f}, cum = {r['ols_cum_42']:+.4f} °C")
    print(f"  Endpoint Difference: 2023 ({r['y_2023']:.3f}°C) - 1981 ({r['y_1981']:.3f}°C) = {r['endpoint_diff']:+.4f} °C")
    print(f"  Decadal Baseline Shift (1981-1990 vs 2014-2023): {r['dec_shift']:+.4f} °C")
    print(f"  Mann-Kendall: S = {r['s_stat']:+.1f}, Z = {r['z_stat']:+.2f}, Tau = {r['tau']:+.4f}, p = {r['p_val']:.6e}")
    print(f"  Hamed-Rao Autocorrelation Check: factor = {r['hr_corr_factor']:.4f}, sig lags = {r['hr_sig_lags']}, Z* = {r['hr_z']:+.2f}, p* = {r['hr_p']:.6e}")
