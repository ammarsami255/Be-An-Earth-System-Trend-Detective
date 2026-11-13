import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from src.data_loader import auto_load_dataset, parse_nasa_power_header

raw_files = [
    "data/raw/nasa_power_cairo_1981_2023.csv",
    "data/raw/nasa_power_alexandria_1981_2023.csv",
    "data/raw/nasa_power_aswan_1981_2023.csv",
]

for fpath in raw_files:
    print("=" * 70)
    print(f"RAW CSV FILE: {fpath}")
    print("=" * 70)

    # 1. Inspect raw CSV line structure and header
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("PARAMETER,YEAR"):
            header_idx = i
            break
            
    print(f"Total lines in file: {len(lines)}")
    print(f"Metadata header comment lines: {header_idx}")
    print(f"Header line {header_idx}: {lines[header_idx].strip()}")
    
    # Read raw table as exported by NASA POWER
    df_raw = pd.read_csv(fpath, skiprows=header_idx)
    print(f"\n1. Raw df.shape: {df_raw.shape}")
    print(f"2. Raw df.columns:\n   {list(df_raw.columns)}")
    print(f"3. Raw df.dtypes:\n{df_raw.dtypes}")
    print(f"\n4. Raw df.head(10):\n{df_raw.head(10)}")
    
    # Unique parameters
    params = df_raw["PARAMETER"].unique()
    print(f"\n5. Unique PARAMETERS ({len(params)}): {list(params)}")
    print(f"   Value counts per parameter:\n{df_raw['PARAMETER'].value_counts()}")
    
    # Check years per parameter
    print(f"\n6. Year ranges per parameter:")
    print(df_raw.groupby("PARAMETER")["YEAR"].agg(["min", "max", "count"]))

    # Missing / fill values (-999) in monthly columns
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    fill_count = (df_raw[months] == -999.0).sum().sum()
    nan_count = df_raw[months].isna().sum().sum()
    print(f"\n7. Fill values (-999.0) across all months: {fill_count}")
    print(f"   NaN values across all months: {nan_count}")

    # Now load using preprocessing pipeline into tidy format
    content_str = "".join(lines)
    df_tidy, meta = auto_load_dataset(content_str)
    print(f"\n8. Tidy Processed df.shape: {df_tidy.shape}")
    print(f"   Tidy df.columns: {list(df_tidy.columns)}")
    print(f"   Metadata location: {meta.get('location')}")
    print(f"   Unique timestamps: {df_tidy['date'].nunique()}")
    print(f"   First timestamp: {df_tidy['date'].min()}")
    print(f"   Last timestamp: {df_tidy['date'].max()}")
    print(f"   Date range: {df_tidy['date'].min().strftime('%Y-%m-%d')} to {df_tidy['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Observations per variable:")
    for col in df_tidy["variable"].unique():
        var_sub = df_tidy[df_tidy["variable"] == col]
        valid_n = var_sub["value"].notna().sum()
        missing_n = var_sub["is_missing"].sum()
        print(f"     - {col}: {valid_n} monthly observations ({missing_n} missing/fill values)")

