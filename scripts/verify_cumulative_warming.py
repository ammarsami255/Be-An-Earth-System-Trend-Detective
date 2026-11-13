import sys, os
sys.path.insert(0, ".")
import pandas as pd
from src.data_loader import auto_load_dataset
from src.preprocessing import aggregate_to_annual
from src.trend_analysis import compute_sens_slope, compute_linear_regression

with open("data/raw/nasa_power_cairo_1981_2023.csv", "r", encoding="utf-8") as f:
    df_tidy, meta = auto_load_dataset(f.read())

cairo_t2m = df_tidy[df_tidy["variable"] == "T2M"]
ann = aggregate_to_annual(cairo_t2m, value_col="value")

y1981 = ann[ann["year"] == 1981]["value"].values[0]
y2023 = ann[ann["year"] == 2023]["value"].values[0]
endpoint_diff = y2023 - y1981

sen = compute_sens_slope(ann["value"].values, ann["year"].values)
slope_sen = sen["slope"]
sen_42 = slope_sen * 42.0
sen_43 = slope_sen * 43.0

ols = compute_linear_regression(ann["value"].values, ann["year"].values)
ols_slope = ols["slope"]
ols_42 = ols_slope * 42.0

d1 = ann[(ann["year"] >= 1981) & (ann["year"] <= 1990)]["value"].mean()
d4 = ann[(ann["year"] >= 2014) & (ann["year"] <= 2023)]["value"].mean()
dec_shift = d4 - d1

print("=" * 70)
print("CUMULATIVE WARMING METRICS FOR CAIRO T2M (1981-2023)")
print("=" * 70)
print(f"1. Theil-Sen Estimated Change over 42 Elapsed Years (1981-2023):")
print(f"   Slope = {slope_sen:.6f} °C/yr (+{slope_sen*10:.4f} °C/decade)")
print(f"   Total 42-year change = {sen_42:.6f} °C (Rounds to: +{sen_42:.2f} °C)")
print(f"\n2. Raw Endpoint Difference (Observed 2023 vs Observed 1981):")
print(f"   Observed 1981 mean = {y1981:.6f} °C")
print(f"   Observed 2023 mean = {y2023:.6f} °C")
print(f"   Difference = {endpoint_diff:.6f} °C (Rounds to: +{endpoint_diff:.2f} °C)")
print(f"\n3. OLS Linear Model Fitted Change over 42 Elapsed Years:")
print(f"   OLS Slope = {ols_slope:.6f} °C/yr (+{ols_slope*10:.4f} °C/decade, R² = {ols['r_squared']:.4f})")
print(f"   Total 42-year fitted change = {ols_42:.6f} °C (Rounds to: +{ols_42:.2f} °C)")
print(f"\n4. Theil-Sen Multiplied by 43 Annual Points (Fencepost Discrepancy):")
print(f"   Slope * 43 = {sen_43:.6f} °C (Rounds to: +{sen_43:.2f} °C)")
print(f"\n5. Decadal Baseline Shift (1981-1990 mean vs 2014-2023 mean):")
print(f"   1981-1990 Mean: {d1:.6f} °C")
print(f"   2014-2023 Mean: {d4:.6f} °C")
print(f"   Decadal Shift: {dec_shift:.6f} °C (Rounds to: +{dec_shift:.2f} °C)")
