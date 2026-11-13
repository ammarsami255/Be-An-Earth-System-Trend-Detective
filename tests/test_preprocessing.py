"""
Unit tests for data loader and preprocessing modules.
EarthTrend Explorer - NASA Space Apps 2026
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import auto_load_dataset, parse_nasa_power_header, validate_variable_data
from src.preprocessing import (
    aggregate_to_annual,
    calculate_climatology,
    check_and_deduplicate_timestamps,
    compute_monthly_anomalies,
    filter_by_date_range,
    handle_missing_values,
)


def test_parse_nasa_power_header():
    sample_header = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Monthly and Annual 
Dates (month/day/year): 01/01/1981 through 12/31/2023 in LST
Location: Latitude  30.0444   Longitude 31.2357 
Elevation from MERRA-2: Average for 0.5 x 0.625 degree lat/lon region = 135.44 meters
The value for missing source data that cannot be computed or is outside of the sources availability range: -999 
Parameter(s): 
T2M             MERRA-2 Temperature at 2 Meters (C) 
-END HEADER-
PARAMETER,YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,ANN
T2M,1981,10.04,12.5,16.95,21.74,23.96,28.63,29.19,28.83,26.21,23.97,16.03,13.62,21.02
"""
    meta = parse_nasa_power_header(sample_header)
    assert meta["latitude"] == 30.0444
    assert meta["longitude"] == 31.2357
    assert meta["elevation_m"] == 135.44
    assert meta["location"] == "Greater Cairo"
    assert meta["fill_value"] == -999.0


def test_auto_load_dataset():
    sample_csv = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Monthly and Annual 
Dates: 01/01/1981 through 12/31/1982
Location: Latitude 30.04 Longitude 31.23
The value for missing source data: -999
-END HEADER-
PARAMETER,YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,ANN
T2M,1981,10.0,12.0,16.0,21.0,24.0,28.0,29.0,28.0,26.0,24.0,16.0,13.0,20.5
T2M,1982,11.0,11.0,14.0,21.0,23.0,28.0,28.0,28.0,26.0,24.0,16.0,12.0,20.2
"""
    df, meta = auto_load_dataset(sample_csv, "Cairo Test")
    assert len(df) == 24
    assert "date" in df.columns
    assert "value" in df.columns
    assert df["value"].iloc[0] == 10.0
    assert df["year"].iloc[0] == 1981
    assert df["month"].iloc[0] == 1


def test_check_and_deduplicate_timestamps():
    dates = pd.date_range("2020-01-01", periods=5, freq="MS").tolist()
    # Add duplicate of index 1
    dates.append(dates[1])
    vals = [10.0, 12.0, 14.0, 16.0, 18.0, 99.0]
    df = pd.DataFrame({"date": dates, "value": vals})

    clean_df, report = check_and_deduplicate_timestamps(df, strategy="first")
    assert report["has_duplicates"] is True
    assert report["duplicate_count"] == 2
    assert len(clean_df) == 5
    # First was kept
    assert clean_df.loc[clean_df["date"] == dates[1], "value"].iloc[0] == 12.0


def test_handle_missing_values():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
        "value": [10.0, -999.0, 14.0, np.nan, 18.0, 20.0],
    })

    # Drop strategy
    clean_drop, rep_drop = handle_missing_values(df, method="drop")
    assert rep_drop["missing_count"] == 2
    assert len(clean_drop) == 4

    # Interpolation strategy
    clean_interp, rep_interp = handle_missing_values(df, method="interpolate")
    assert len(clean_interp) == 6
    # Interpolated between 10 and 14 -> 12
    assert clean_interp["value"].iloc[1] == 12.0
    # Interpolated between 14 and 18 -> 16
    assert clean_interp["value"].iloc[3] == 16.0


def test_aggregate_to_annual():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    # Year 2020: 10.0 every month -> mean 10.0
    # Year 2021: 20.0 every month -> mean 20.0
    vals = [10.0] * 12 + [20.0] * 12
    df = pd.DataFrame({"date": dates, "value": vals, "year": dates.year})

    ann_df = aggregate_to_annual(df, min_months_required=12)
    assert len(ann_df) == 2
    assert ann_df.loc[ann_df["year"] == 2020, "annual_mean"].iloc[0] == 10.0
    assert ann_df.loc[ann_df["year"] == 2021, "annual_mean"].iloc[0] == 20.0


def test_compute_monthly_anomalies():
    dates = pd.date_range("1990-01-01", periods=24, freq="MS")
    # Base pattern repeated twice
    base_pattern = [10, 12, 16, 20, 24, 28, 30, 29, 26, 22, 16, 12]
    # Year 1: base pattern
    # Year 2: base pattern + 2.0 degrees
    vals = base_pattern + [x + 2.0 for x in base_pattern]
    df = pd.DataFrame({"date": dates, "value": vals, "year": dates.year, "month": dates.month})

    anom_df = compute_monthly_anomalies(df, baseline_start_year=1990, baseline_end_year=1990)
    # Year 1 anomalies should be exactly 0.0
    assert np.allclose(anom_df.iloc[:12]["anomaly"], 0.0)
    # Year 2 anomalies should be exactly +2.0
    assert np.allclose(anom_df.iloc[12:]["anomaly"], 2.0)
