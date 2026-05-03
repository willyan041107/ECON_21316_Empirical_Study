"""
Parse NOAA/NHC Atlantic HURDAT2 best-track data.

Input:
    data/raw/hurdat2/hurdat2_atlantic_1851_2025.txt

Outputs:
    data/intermediate/hurdat2_atlantic_tracks.csv
    data/intermediate/hurdat2_atlantic_tracks_relevant_years.csv

Relevant years:
    For LOSFA graduation years 1999-2025, the first-pass storm year is
    graduation_year - 1, so relevant hurricane seasons are 1998-2024.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "hurdat2" / "hurdat2_atlantic_1851_2025.txt"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

OUT_ALL = INTERMEDIATE_DIR / "hurdat2_atlantic_tracks.csv"
OUT_RELEVANT = INTERMEDIATE_DIR / "hurdat2_atlantic_tracks_relevant_years.csv"


def parse_lat_lon(value: str) -> float:
    """Convert HURDAT2 latitude/longitude strings into signed decimal degrees."""
    value = value.strip()

    if not value:
        return float("nan")

    direction = value[-1]
    number = float(value[:-1])

    if direction in {"S", "W"}:
        return -number

    return number


def parse_int(value: str) -> int | None:
    """Parse integer value, treating -999 as missing."""
    value = value.strip()

    if value == "-999" or value == "":
        return None

    return int(value)


def parse_hurdat2(path: Path) -> pd.DataFrame:
    """Parse HURDAT2 text file into one row per storm-track point."""
    records = []

    current_storm_id = None
    current_storm_name = None
    expected_points = None

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]

            # Storm header row:
            # AL011851, UNNAMED, 14,
            if parts[0].startswith("AL") and len(parts[0]) == 8:
                current_storm_id = parts[0]
                current_storm_name = parts[1]
                expected_points = int(parts[2])
                continue

            # Track point row.
            date = parts[0]
            time = parts[1]
            record_identifier = parts[2]
            status = parts[3]
            latitude = parse_lat_lon(parts[4])
            longitude = parse_lat_lon(parts[5])
            max_wind_kt = parse_int(parts[6])
            min_pressure_mb = parse_int(parts[7])

            timestamp = pd.to_datetime(date + time, format="%Y%m%d%H%M")

            records.append(
                {
                    "storm_id": current_storm_id,
                    "storm_name": current_storm_name,
                    "storm_year": int(date[:4]),
                    "timestamp": timestamp,
                    "date": date,
                    "time": time,
                    "record_identifier": record_identifier,
                    "status": status,
                    "latitude": latitude,
                    "longitude": longitude,
                    "max_wind_kt": max_wind_kt,
                    "min_pressure_mb": min_pressure_mb,
                    "expected_points_for_storm": expected_points,
                    "is_tropical_storm_force": max_wind_kt is not None and max_wind_kt >= 34,
                    "is_hurricane_force": max_wind_kt is not None and max_wind_kt >= 64,
                    "is_major_hurricane_force": max_wind_kt is not None and max_wind_kt >= 96,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    """Run HURDAT2 parser."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing HURDAT2 raw file: {RAW_PATH}")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    tracks = parse_hurdat2(RAW_PATH)

    tracks.to_csv(OUT_ALL, index=False)

    relevant = tracks[
        (tracks["storm_year"] >= 1998)
        & (tracks["storm_year"] <= 2024)
    ].copy()

    relevant.to_csv(OUT_RELEVANT, index=False)

    print("Saved:")
    print(" -", OUT_ALL.relative_to(PROJECT_ROOT))
    print(" -", OUT_RELEVANT.relative_to(PROJECT_ROOT))

    print("\nAll tracks shape:", tracks.shape)
    print("Relevant tracks shape:", relevant.shape)

    print("\nYears:")
    print(tracks["storm_year"].min(), "to", tracks["storm_year"].max())

    print("\nRelevant years:")
    print(relevant["storm_year"].min(), "to", relevant["storm_year"].max())

    print("\nStatus counts in relevant years:")
    print(relevant["status"].value_counts(dropna=False))

    print("\nRelevant hurricane-force track points:")
    print(int(relevant["is_hurricane_force"].sum()))

    print("\nSample relevant rows:")
    print(
        relevant[
            [
                "storm_id",
                "storm_name",
                "storm_year",
                "timestamp",
                "status",
                "latitude",
                "longitude",
                "max_wind_kt",
                "is_hurricane_force",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
