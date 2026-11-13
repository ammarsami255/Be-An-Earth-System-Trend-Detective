"""
EarthTrend Explorer - Data Loader Module
NASA Space Apps Challenge 2026

Handles fetching, reading, validating, and formatting NASA Earth Science datasets:
- NASA POWER API (MERRA-2 monthly meteorology: T2M, T2M_MAX, T2M_MIN, PRECTOTCORR)
- NASA Giovanni CSV exports
- Generic time-series CSVs
"""

from __future__ import annotations
import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import requests


# Predefined coordinates for common Egyptian regions for demonstration & comparison
EGYPT_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "Greater Cairo": {
        "lat": 30.0444,
        "lon": 31.2357,
        "description": "Urban metropolis in Nile Delta / Valley transition",
        "elevation_m": 135.44,
    },
    "Alexandria": {
        "lat": 31.2001,
        "lon": 29.9187,
        "description": "Mediterranean coastal marine climate",
        "elevation_m": 7.0,
    },
    "Aswan": {
        "lat": 24.0889,
        "lon": 32.8998,
        "description": "Upper Egypt hyper-arid desert climate",
        "elevation_m": 108.0,
    },
    "Giza": {
        "lat": 29.9870,
        "lon": 31.2118,
        "description": "Western bank of Nile, Greater Cairo Metropolitan Area",
        "elevation_m": 75.0,
    },
    "Luxor": {
        "lat": 25.6872,
        "lon": 32.6396,
        "description": "Southern Egypt Nile Valley arid climate",
        "elevation_m": 89.0,
    },
}

VARIABLE_METADATA: Dict[str, Dict[str, str]] = {
    "T2M": {
        "name": "Temperature at 2 Meters",
        "units": "°C",
        "description": "MERRA-2 monthly mean air temperature at 2 meters above surface",
        "source": "NASA MERRA-2 via NASA POWER",
        "valid_min": -50.0,
        "valid_max": 65.0,
    },
    "T2M_MAX": {
        "name": "Maximum Temperature at 2 Meters",
        "units": "°C",
        "description": "MERRA-2 monthly mean of daily maximum temperature at 2 meters",
        "source": "NASA MERRA-2 via NASA POWER",
        "valid_min": -40.0,
        "valid_max": 70.0,
    },
    "T2M_MIN": {
        "name": "Minimum Temperature at 2 Meters",
        "units": "°C",
        "description": "MERRA-2 monthly mean of daily minimum temperature at 2 meters",
        "source": "NASA MERRA-2 via NASA POWER",
        "valid_min": -60.0,
        "valid_max": 50.0,
    },
    "PRECTOTCORR": {
        "name": "Corrected Total Precipitation",
        "units": "mm/day",
        "description": "MERRA-2 bias-corrected total surface precipitation",
        "source": "NASA MERRA-2 via NASA POWER",
        "valid_min": 0.0,
        "valid_max": 500.0,
    },
}

MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def parse_nasa_power_header(content: str) -> Dict[str, Any]:
    """
    Extracts metadata from the standard NASA POWER header block.
    """
    metadata: Dict[str, Any] = {
        "source": "NASA POWER / MERRA-2",
        "format": "NASA_POWER_MONTHLY",
        "raw_header": "",
        "location": "Unknown",
        "latitude": None,
        "longitude": None,
        "elevation_m": None,
        "dates": None,
        "fill_value": -999.0,
        "parameters": {},
    }

    if "-BEGIN HEADER-" in content and "-END HEADER-" in content:
        header_text = content.split("-END HEADER-")[0].split("-BEGIN HEADER-")[-1].strip()
        metadata["raw_header"] = header_text

        # Extract coordinates
        coord_match = re.search(r"Latitude\s+([\-0-9.]+)\s+Longitude\s+([\-0-9.]+)", header_text, re.IGNORECASE)
        if coord_match:
            metadata["latitude"] = float(coord_match.group(1))
            metadata["longitude"] = float(coord_match.group(2))

        # Extract elevation
        elev_match = re.search(r"region\s*=\s*([\-0-9.]+)\s*meters", header_text, re.IGNORECASE)
        if elev_match:
            metadata["elevation_m"] = float(elev_match.group(1))

        # Extract dates
        dates_match = re.search(r"Dates\s*\([^)]*\)\s*:\s*([0-9/]+)\s+through\s+([0-9/]+)", header_text, re.IGNORECASE)
        if dates_match:
            metadata["dates"] = f"{dates_match.group(1)} to {dates_match.group(2)}"

        # Extract fill value
        fill_match = re.search(r"range:\s*([\-0-9.]+)", header_text)
        if fill_match:
            metadata["fill_value"] = float(fill_match.group(1))

        # Match location to known Egyptian presets if near
        if metadata["latitude"] is not None and metadata["longitude"] is not None:
            lat = metadata["latitude"]
            lon = metadata["longitude"]
            for name, loc_data in EGYPT_LOCATIONS.items():
                if abs(lat - loc_data["lat"]) < 0.3 and abs(lon - loc_data["lon"]) < 0.3:
                    metadata["location"] = name
                    break
            if metadata["location"] == "Unknown":
                metadata["location"] = f"Lat {lat:.2f}°, Lon {lon:.2f}°"

    return metadata


def load_nasa_power_csv(content: str, location_override: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses a NASA POWER monthly CSV file into a standardized tidy long-format DataFrame.
    Output columns: ['date', 'year', 'month', 'variable', 'value', 'units', 'location']
    Also preserves annual values in metadata.
    """
    metadata = parse_nasa_power_header(content)
    if location_override:
        metadata["location"] = location_override

    # Find where the data table starts (after -END HEADER- or PARAMETER,YEAR line)
    lines = content.splitlines()
    data_start_idx = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("PARAMETER,YEAR") or line.strip().startswith("PARAMETER, YEAR"):
            data_start_idx = idx
            break

    data_csv = "\n".join(lines[data_start_idx:])
    raw_df = pd.read_csv(io.StringIO(data_csv))
    raw_df.columns = [c.strip().upper() for c in raw_df.columns]

    # Convert wide matrix (YEAR, JAN..DEC, ANN) to tidy long format
    tidy_rows: List[Dict[str, Any]] = []
    annual_rows: List[Dict[str, Any]] = []

    fill_val = metadata.get("fill_value", -999.0)

    for _, row in raw_df.iterrows():
        param = str(row["PARAMETER"]).strip().upper()
        year = int(row["YEAR"])
        units = VARIABLE_METADATA.get(param, {}).get("units", "units")

        # Annual average
        if "ANN" in row and pd.notna(row["ANN"]):
            ann_val = float(row["ANN"])
            if ann_val != fill_val:
                annual_rows.append({
                    "year": year,
                    "variable": param,
                    "annual_value": ann_val,
                    "units": units,
                    "location": metadata["location"],
                })

        # Monthly values
        for month_num, month_col in enumerate(MONTH_NAMES, start=1):
            if month_col in row and pd.notna(row[month_col]):
                val = float(row[month_col])
                is_missing = (val == fill_val)
                actual_val = np.nan if is_missing else val
                date_val = pd.Timestamp(year=year, month=month_num, day=1)

                tidy_rows.append({
                    "date": date_val,
                    "year": year,
                    "month": month_num,
                    "month_name": month_col,
                    "variable": param,
                    "value": actual_val,
                    "units": units,
                    "location": metadata["location"],
                    "is_missing": is_missing,
                })

    df = pd.DataFrame(tidy_rows)
    df.sort_values(by=["variable", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    metadata["annual_df"] = pd.DataFrame(annual_rows)
    metadata["available_variables"] = sorted(df["variable"].unique().tolist())
    metadata["total_records"] = len(df)
    metadata["missing_records"] = int(df["is_missing"].sum())
    metadata["start_date"] = df["date"].min() if not df.empty else None
    metadata["end_date"] = df["date"].max() if not df.empty else None

    return df, metadata


def load_giovanni_csv(content: str, location_name: str = "NASA Giovanni Region") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses NASA Giovanni time series CSV export.
    Typically contains comment lines with '#' and columns like (time, value) or (Date, Value).
    """
    lines = content.splitlines()
    header_comments: List[str] = []
    data_lines: List[str] = []

    for line in lines:
        if line.strip().startswith("#"):
            header_comments.append(line.strip())
        elif line.strip():
            data_lines.append(line.strip())

    if not data_lines:
        raise ValueError("No data rows found in Giovanni CSV file.")

    raw_df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    time_col = None
    value_col = None

    for col in raw_df.columns:
        c_low = col.lower().strip()
        if any(k in c_low for k in ["time", "date", "timestamp", "year"]):
            time_col = col
        elif any(k in c_low for k in ["val", "temp", "t2m", "precip", "mean", "m2t"]):
            value_col = col

    if time_col is None:
        time_col = raw_df.columns[0]
    if value_col is None:
        value_col = raw_df.columns[1] if len(raw_df.columns) > 1 else raw_df.columns[0]

    raw_df["date"] = pd.to_datetime(raw_df[time_col], errors="coerce")
    raw_df["value"] = pd.to_numeric(raw_df[value_col], errors="coerce")
    raw_df.dropna(subset=["date"], inplace=True)

    raw_df["year"] = raw_df["date"].dt.year
    raw_df["month"] = raw_df["date"].dt.month
    raw_df["month_name"] = raw_df["date"].dt.strftime("%b").str.upper()
    var_name = value_col.strip()
    raw_df["variable"] = var_name
    raw_df["units"] = "°C" if "temp" in var_name.lower() or "t2m" in var_name.lower() else "units"
    raw_df["location"] = location_name
    raw_df["is_missing"] = raw_df["value"].isna()

    tidy_df = raw_df[["date", "year", "month", "month_name", "variable", "value", "units", "location", "is_missing"]].copy()
    tidy_df.sort_values(by="date", inplace=True)
    tidy_df.reset_index(drop=True, inplace=True)

    metadata: Dict[str, Any] = {
        "source": "NASA Giovanni",
        "format": "GIOVANNI_TIME_SERIES",
        "raw_header": "\n".join(header_comments),
        "location": location_name,
        "available_variables": [var_name],
        "total_records": len(tidy_df),
        "missing_records": int(tidy_df["is_missing"].sum()),
        "start_date": tidy_df["date"].min(),
        "end_date": tidy_df["date"].max(),
    }
    return tidy_df, metadata


def load_generic_csv(
    content: str,
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
    variable_name: str = "Observation",
    units: str = "units",
    location: str = "Custom Region",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses a generic tabular CSV file.
    Infers date column and numerical value column if not explicitly supplied.
    """
    raw_df = pd.read_csv(io.StringIO(content))

    # Detect date/time column
    if time_col is None or time_col not in raw_df.columns:
        date_candidates = [c for c in raw_df.columns if any(k in c.lower() for k in ["date", "time", "year", "day"])]
        if date_candidates:
            time_col = date_candidates[0]
        else:
            time_col = raw_df.columns[0]

    # Detect numerical value column
    if value_col is None or value_col not in raw_df.columns:
        num_candidates = [c for c in raw_df.columns if c != time_col and pd.api.types.is_numeric_dtype(raw_df[c])]
        if num_candidates:
            value_col = num_candidates[0]
        else:
            other_cols = [c for c in raw_df.columns if c != time_col]
            value_col = other_cols[0] if other_cols else time_col

    dates = pd.to_datetime(raw_df[time_col], errors="coerce")
    values = pd.to_numeric(raw_df[value_col], errors="coerce")

    tidy_df = pd.DataFrame({
        "date": dates,
        "year": dates.dt.year,
        "month": dates.dt.month,
        "month_name": dates.dt.strftime("%b").str.upper(),
        "variable": variable_name,
        "value": values,
        "units": units,
        "location": location,
        "is_missing": values.isna() | (values == -999.0),
    })

    # Convert fill value to NaN
    tidy_df.loc[tidy_df["value"] == -999.0, "value"] = np.nan
    tidy_df.dropna(subset=["date"], inplace=True)
    tidy_df.sort_values(by="date", inplace=True)
    tidy_df.reset_index(drop=True, inplace=True)

    metadata: Dict[str, Any] = {
        "source": "Custom CSV Upload",
        "format": "GENERIC_TABULAR_CSV",
        "raw_header": f"Columns: {list(raw_df.columns)}",
        "location": location,
        "available_variables": [variable_name],
        "total_records": len(tidy_df),
        "missing_records": int(tidy_df["is_missing"].sum()),
        "start_date": tidy_df["date"].min(),
        "end_date": tidy_df["date"].max(),
    }
    return tidy_df, metadata


def auto_load_dataset(content: str, default_location: str = "Selected Region") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Intelligently identifies dataset format and parses accordingly.
    Supports:
    1. NASA POWER standard monthly/annual format.
    2. NASA Giovanni format.
    3. Generic tabular CSV format.
    """
    if not content or not content.strip():
        raise ValueError("The provided file content is empty.")

    # Check for NASA POWER signature
    if "-BEGIN HEADER-" in content or "NASA/POWER" in content or "PARAMETER,YEAR" in content:
        return load_nasa_power_csv(content, location_override=default_location if default_location != "Selected Region" else None)

    # Check for NASA Giovanni signature
    if any(line.startswith("#") for line in content.splitlines()[:15]):
        return load_giovanni_csv(content, location_name=default_location)

    # Fallback to generic CSV
    return load_generic_csv(content, location=default_location)


def fetch_nasa_power_point(
    latitude: float,
    longitude: float,
    start_year: int = 1981,
    end_year: int = 2023,
    parameters: Union[List[str], str] = ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"),
    community: str = "AG",
    timeout: int = 45,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Fetches real NASA Earth Science data directly from the official NASA POWER API.
    Returns:
        (df_tidy, metadata_dict, raw_csv_text)
    """
    if isinstance(parameters, (list, tuple)):
        param_str = ",".join(parameters)
    else:
        param_str = str(parameters)

    endpoint = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": param_str,
        "community": community,
        "longitude": longitude,
        "latitude": latitude,
        "start": start_year,
        "end": end_year,
        "format": "CSV",
    }

    response = requests.get(endpoint, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"NASA POWER API returned HTTP {response.status_code}: {response.text[:300]}"
        )

    raw_csv = response.text
    df, metadata = load_nasa_power_csv(raw_csv)
    metadata["api_url"] = response.url
    return df, metadata, raw_csv


def validate_variable_data(df: pd.DataFrame, variable_name: str) -> Dict[str, Any]:
    """
    Validates numerical values, bounds, and completeness of an extracted variable series.
    """
    var_df = df[df["variable"] == variable_name].copy()
    if var_df.empty:
        return {
            "valid": False,
            "error": f"Variable '{variable_name}' not found in dataset.",
        }

    total_pts = len(var_df)
    valid_pts = int(var_df["value"].notna().sum())
    missing_pts = total_pts - valid_pts
    pct_missing = (missing_pts / total_pts) * 100.0 if total_pts > 0 else 0.0

    meta = VARIABLE_METADATA.get(variable_name, {})
    vmin = meta.get("valid_min", -np.inf)
    vmax = meta.get("valid_max", np.inf)

    clean_vals = var_df["value"].dropna()
    out_of_bounds = int(((clean_vals < vmin) | (clean_vals > vmax)).sum())

    return {
        "valid": True,
        "variable": variable_name,
        "units": meta.get("units", var_df["units"].iloc[0] if "units" in var_df else "units"),
        "total_records": total_pts,
        "valid_records": valid_pts,
        "missing_records": missing_pts,
        "pct_missing": round(pct_missing, 2),
        "min_value": round(float(clean_vals.min()), 2) if not clean_vals.empty else None,
        "max_value": round(float(clean_vals.max()), 2) if not clean_vals.empty else None,
        "mean_value": round(float(clean_vals.mean()), 2) if not clean_vals.empty else None,
        "std_value": round(float(clean_vals.std()), 2) if not clean_vals.empty else None,
        "out_of_bounds_count": out_of_bounds,
    }
