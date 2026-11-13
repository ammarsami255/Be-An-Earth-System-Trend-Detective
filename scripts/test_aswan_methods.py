import sys, os
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.data_loader import auto_load_dataset
from src.preprocessing import aggregate_to_annual
from src.trend_analysis import compute_sens_slope, mann_kendall_test, hamed_rao_modified_mann_kendall

with open('data/raw/nasa_power_aswan_1981_2023.csv', 'r', encoding='utf-8') as f:
    content = f.read()

df_tidy, meta = auto_load_dataset(content, "Aswan")
t2m = df_tidy[df_tidy["variable"] == "T2M"]
ann_agg = aggregate_to_annual(t2m)

# Method 1: aggregate_to_annual (mean of 12 months)
y_agg = ann_agg["value"].values
years_agg = ann_agg["year"].values
sen_agg = compute_sens_slope(y_agg, years_agg)
mk_agg = mann_kendall_test(y_agg)
hr_agg = hamed_rao_modified_mann_kendall(y_agg)

# Method 2: NASA POWER ANN column from metadata["annual_df"]
df_ann_meta = meta["annual_df"]
df_ann_t2m = df_ann_meta[df_ann_meta["variable"] == "T2M"].sort_values("year")
y_ann = df_ann_t2m["annual_value"].values
years_ann = df_ann_t2m["year"].values
sen_ann = compute_sens_slope(y_ann, years_ann)
mk_ann = mann_kendall_test(y_ann)
hr_ann = hamed_rao_modified_mann_kendall(y_ann)

print("=== ASWAN COMPARISON: MONTHLY-DERIVED ANNUAL MEAN vs NASA POWER ANN COLUMN ===")
print("Method 1 (Monthly-derived annual mean, aggregate_to_annual):")
print(f"  Sen's Slope: {sen_agg['slope']*10:.6f} °C/decade")
print(f"  Sen's 95% CI: [{sen_agg['slope_ci_lower']*10:.6f}, {sen_agg['slope_ci_upper']*10:.6f}]")
print(f"  MK S: {mk_agg['s_statistic']}, Z: {mk_agg['z_statistic']:.6f}, p: {mk_agg['p_value']:.8e}")
print(f"  HR Factor: {hr_agg['variance_correction_factor']:.6f}, Z*: {hr_agg['z_statistic']:.6f}, p*: {hr_agg['p_value']:.8e}")

print("\nMethod 2 (NASA POWER ANN column):")
print(f"  Sen's Slope: {sen_ann['slope']*10:.6f} °C/decade")
print(f"  Sen's 95% CI: [{sen_ann['slope_ci_lower']*10:.6f}, {sen_ann['slope_ci_upper']*10:.6f}]")
print(f"  MK S: {mk_ann['s_statistic']}, Z: {mk_ann['z_statistic']:.6f}, p: {mk_ann['p_value']:.8e}")
print(f"  HR Factor: {hr_ann['variance_correction_factor']:.6f}, Z*: {hr_ann['z_statistic']:.6f}, p*: {hr_ann['p_value']:.8e}")

print(f"\nMax diff between monthly mean and ANN: {np.max(np.abs(y_agg - y_ann)):.8f}")
