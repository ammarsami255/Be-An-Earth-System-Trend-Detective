import sys, os
sys.path.insert(0, ".")
import math
import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import auto_load_dataset
from src.preprocessing import aggregate_to_annual
from src.trend_analysis import mann_kendall_test, compute_sens_slope, hamed_rao_modified_mann_kendall

locations = [
    ("Greater Cairo", "data/raw/nasa_power_cairo_1981_2023.csv", 30.0444, 31.2357),
    ("Alexandria", "data/raw/nasa_power_alexandria_1981_2023.csv", 31.2001, 29.9187),
    ("Aswan", "data/raw/nasa_power_aswan_1981_2023.csv", 24.0889, 32.8998),
]

print("=" * 105)
print("COMPREHENSIVE SIDE-BY-SIDE AUDIT: ORIGINAL MK VS. HAMED-RAO MODIFIED MK")
print("=" * 105)

data_rows = []

for name, path, lat, lon in locations:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    df_tidy, meta = auto_load_dataset(content, default_location=name)
    t2m = df_tidy[df_tidy["variable"] == "T2M"].copy()
    ann_df = aggregate_to_annual(t2m, value_col="value")
    
    y = ann_df["value"].values
    years = ann_df["year"].values
    n = len(y)
    
    # 1. Original Mann-Kendall Test
    orig_mk = mann_kendall_test(y, alpha=0.05)
    s = orig_mk["s_statistic"]
    orig_var = orig_mk["var_s"]
    orig_z = orig_mk["z_statistic"]
    orig_p = orig_mk["p_value"]
    
    # 2. Hamed-Rao Modified Mann-Kendall Test
    hr_res = hamed_rao_modified_mann_kendall(y, alpha=0.05)
    hr_var = hr_res["corrected_var_s"]
    hr_factor = hr_res["variance_correction_factor"]
    hr_z = hr_res["z_statistic"]
    hr_p = hr_res["p_value"]
    sig_lags = hr_res["significant_lags"]
    
    # 3. Sen's Slope
    sen = compute_sens_slope(y, years, alpha=0.05)
    slope = sen["slope"]
    ci_lower = sen["slope_ci_lower"]
    ci_upper = sen["slope_ci_upper"]
    
    # 4. Mathematical Verification of continuity correction identity:
    # Z* = (S - sign(S)) / sqrt(Var*(S))
    sign_s = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
    manual_hr_z = (s - sign_s) / math.sqrt(hr_var) if hr_var > 0 else 0.0
    manual_hr_p = 2.0 * (1.0 - stats.norm.cdf(abs(manual_hr_z)))
    
    # Identity assertion
    z_diff = abs(hr_z - manual_hr_z)
    p_diff = abs(hr_p - manual_hr_p)
    assert z_diff < 1e-12, f"Z* identity failed for {name}: {hr_z} vs {manual_hr_z}"
    assert p_diff < 1e-12, f"p* identity failed for {name}: {hr_p} vs {manual_hr_p}"
    
    data_rows.append({
        "Location": name,
        "Grid Cell": f"({lat:.2f}°N, {lon:.2f}°E)",
        "n": n,
        "S": s,
        "Sen Slope (°C/dec)": slope * 10.0,
        "Sen 95% CI (°C/dec)": f"[{ci_lower*10:+.3f}, {ci_upper*10:+.3f}]",
        # Original MK
        "Orig Var(S)": orig_var,
        "Orig Z": orig_z,
        "Orig p-value": orig_p,
        "Orig Sig?": orig_mk["is_significant"],
        # Hamed-Rao
        "HR Factor (n/n*)": hr_factor,
        "HR Sig Lags": len(sig_lags),
        "HR Var*(S)": hr_var,
        "HR Z*": hr_z,
        "HR p*-value": hr_p,
        "HR Sig?": hr_res["is_significant"],
        "Identity Verified": "YES" if z_diff < 1e-12 else "NO",
    })

audit_df = pd.DataFrame(data_rows)
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 1000)

print("\n--- SIDE-BY-SIDE SUMMARY TABLE ---")
print(audit_df[[
    "Location", "S", "Sen Slope (°C/dec)", "Sen 95% CI (°C/dec)",
    "Orig Var(S)", "Orig Z", "Orig p-value",
    "HR Factor (n/n*)", "HR Var*(S)", "HR Z*", "HR p*-value"
]].to_string(index=False))

print("\n--- DETAILED REGIONAL VERIFICATION ---")
for r in data_rows:
    print(f"\nLocation: {r['Location']} {r['Grid Cell']} (n={r['n']})")
    print(f"  Test Statistic S: {r['S']:+.1f}")
    print(f"  Sen's Slope: {r['Sen Slope (°C/dec)']/10:+.6f} °C/yr ({r['Sen Slope (°C/dec)']:+.3f} °C/decade)")
    print(f"  Sen's 95% CI: {r['Sen 95% CI (°C/dec)']} °C/decade")
    print(f"  [Standard Mann-Kendall]:")
    print(f"    - Var(S) = {r['Orig Var(S)']:.4f}")
    print(f"    - Z      = {r['Orig Z']:+.4f}")
    print(f"    - p-val  = {r['Orig p-value']:.6e} ({'Significant at 0.05' if r['Orig Sig?'] else 'Non-significant'})")
    print(f"  [Hamed & Rao (1998) Modified MK]:")
    print(f"    - Significant Lags = {r['HR Sig Lags']}")
    print(f"    - Factor (n/n*)    = {r['HR Factor (n/n*)']:.6f}")
    print(f"    - Var*(S)          = {r['HR Var*(S)']:.4f}")
    print(f"    - Z*               = {r['HR Z*']:+.4f}")
    print(f"    - p*-val           = {r['HR p*-value']:.6e} ({'Significant at 0.05' if r['HR Sig?'] else 'Non-significant'})")
    print(f"  Identity Z* = (S - sign(S)) / sqrt(Var*(S)): {r['Identity Verified']}")
