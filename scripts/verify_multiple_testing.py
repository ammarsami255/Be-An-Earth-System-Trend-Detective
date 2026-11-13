import sys, os
sys.path.insert(0, ".")
import pandas as pd
from src.data_loader import auto_load_dataset
from src.trend_analysis import compute_calendar_month_breakdown

with open("data/raw/nasa_power_cairo_1981_2023.csv", "r", encoding="utf-8") as f:
    df_tidy, meta = auto_load_dataset(f.read())

cairo_t2m = df_tidy[df_tidy["variable"] == "T2M"]
breakdown = compute_calendar_month_breakdown(cairo_t2m, value_col="value", alpha=0.05)

print("=" * 90)
print("12-MONTH CALENDAR BREAKDOWN & MULTIPLE TESTING ADJUSTMENTS (CAIRO T2M 1981-2023)")
print("=" * 90)

cols = [
    "month_name", "trend", "sen_slope_per_decade", "z_statistic",
    "p_value", "p_holm", "p_bh", "sig_nominal", "sig_holm", "sig_bh"
]
df_disp = breakdown[cols].copy()
df_disp.rename(columns={
    "month_name": "Month",
    "trend": "Trend",
    "sen_slope_per_decade": "Sen Slope (°C/dec)",
    "z_statistic": "Z",
    "p_value": "Nominal p",
    "p_holm": "Holm p",
    "p_bh": "BH p",
    "sig_nominal": "Sig (Nominal)",
    "sig_holm": "Sig (Holm)",
    "sig_bh": "Sig (BH)",
}, inplace=True)

print(df_disp.to_string(index=False))

print("\nSUMMARY COUNTS:")
print(f"Nominally significant (p < 0.05): {breakdown['sig_nominal'].sum()} / 12 months")
print(f"  Months: {breakdown[breakdown['sig_nominal']]['month_name'].tolist()}")
print(f"Benjamini-Hochberg FDR significant (q < 0.05): {breakdown['sig_bh'].sum()} / 12 months")
print(f"  Months: {breakdown[breakdown['sig_bh']]['month_name'].tolist()}")
print(f"Holm-Bonferroni FWER significant (alpha < 0.05): {breakdown['sig_holm'].sum()} / 12 months")
print(f"  Months: {breakdown[breakdown['sig_holm']]['month_name'].tolist()}")
