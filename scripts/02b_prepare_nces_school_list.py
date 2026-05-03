"""
Prepare NCES CCD school-level data for LOSFA school matching.

Input:
    data/raw/nces/*.csv

Output:
    data/intermediate/nces_louisiana_schools.csv
    data/intermediate/nces_louisiana_high_school_candidates.csv

Goal:
    Standardize NCES school identifiers, school names, parish/county names,
    district names, grade span, school level, operational status, and coordinates.

This script does not match LOSFA to NCES yet. It only prepares the NCES side.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NCES_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nces"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"


COLUMN_ALIASES = {
    "nces_school_id": ["NCESSCH", "NCES_SCHOOL_ID", "SCHOOL_ID", "SCHID"],
    "nces_district_id": ["LEAID", "NCES_LEAID", "DISTRICT_ID"],
    "nces_school_name": ["SCH_NAME", "SCHOOL_NAME", "NAME"],
    "nces_district_name": ["LEA_NAME", "DISTRICT_NAME", "LEA_NAME"],
    "state": ["LSTATE", "STATE", "ST", "STATE_ABBR"],
    "state_name": ["STATENAME", "STATE_NAME"],
    "county_name": ["CONAME", "COUNTY_NAME", "COUNTY", "NMCNTY"],
    "city": ["LCITY", "CITY"],
    "zip_code": ["LZIP", "ZIP", "ZIPCODE"],
    "latitude": ["LATCOD", "LATITUDE", "LAT"],
    "longitude": ["LONCOD", "LONGITUDE", "LON", "LONG"],
    "school_type": ["SCH_TYPE", "SCH_TYPE_TEXT", "SCHOOL_TYPE"],
    "school_level": ["LEVEL", "LEVEL_TEXT", "SCHOOL_LEVEL"],
    "operational_status": ["SY_STATUS", "UPDATED_STATUS", "STATUS", "OPERATIONAL_STATUS"],
    "grade_low": ["GSLO", "GRADE_LOW", "LOW_GRADE"],
    "grade_high": ["GSHI", "GRADE_HIGH", "HIGH_GRADE"],
    "charter": ["CHARTER_TEXT", "CHARTER"],
    "virtual": ["VIRTUAL", "VIRTUAL_TEXT"],
}


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV file using a few common encodings."""
    for encoding in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(f"Could not read file with common encodings: {path}")


def find_first_csv() -> Path:
    """Find the first NCES CSV under data/raw/nces."""
    csv_files = sorted(NCES_RAW_DIR.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {NCES_RAW_DIR}")

    if len(csv_files) > 1:
        print("Multiple CSV files found. Using the first one:")
        for path in csv_files:
            print(" -", path.relative_to(PROJECT_ROOT))

    return csv_files[0]


def get_column(df: pd.DataFrame, standard_name: str) -> pd.Series:
    """Return a standardized column if any alias exists; otherwise return missing values."""
    aliases = COLUMN_ALIASES[standard_name]

    for alias in aliases:
        if alias in df.columns:
            return df[alias]

    return pd.Series([pd.NA] * len(df))


def normalize_school_name(name: object) -> str:
    """Normalize school names for later matching."""
    if pd.isna(name):
        return ""

    text = str(name).upper().strip()
    text = text.replace("&", " AND ")
    text = text.replace("’", "'")
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\bSAINT\b", "ST", text)
    text = re.sub(r"\bSCH\b", "SCHOOL", text)
    text = re.sub(r"\bHS\b", "HIGH SCHOOL", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_parish_name(name: object) -> str:
    """Normalize Louisiana parish/county names from NCES county field."""
    if pd.isna(name):
        return ""

    text = str(name).upper().strip()
    text = text.replace(" PARISH", "")
    text = text.replace(" COUNTY", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def grade_to_num(value: object) -> float | None:
    """Convert NCES grade labels to numeric grade values."""
    if pd.isna(value):
        return None

    text = str(value).upper().strip()

    if text in {"PK", "PREK", "PRE-K"}:
        return -1

    if text in {"KG", "K", "KINDERGARTEN"}:
        return 0

    if text in {"UG", "AE", "N", "M", "UN"}:
        return None

    match = re.search(r"\d+", text)
    if match:
        return float(match.group())

    return None


def prepare_nces_school_list(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize NCES school-level data."""
    out = pd.DataFrame()

    for standard_name in COLUMN_ALIASES:
        out[standard_name] = get_column(df, standard_name)

    out["nces_school_name_clean"] = out["nces_school_name"].apply(normalize_school_name)
    out["nces_parish"] = out["county_name"].apply(normalize_parish_name)

    out["grade_low_num"] = out["grade_low"].apply(grade_to_num)
    out["grade_high_num"] = out["grade_high"].apply(grade_to_num)

    out["state_upper"] = out["state"].fillna("").astype(str).str.upper().str.strip()
    out["state_name_upper"] = out["state_name"].fillna("").astype(str).str.upper().str.strip()

    # Keep Louisiana rows if state information exists.
    has_state_info = (out["state_upper"] != "").any() or (out["state_name_upper"] != "").any()

    if has_state_info:
        out = out[
            (out["state_upper"] == "LA") |
            (out["state_name_upper"] == "LOUISIANA")
        ].copy()

    # Coordinates should be numeric if present.
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    school_level_text = out["school_level"].fillna("").astype(str).str.upper()
    school_name_text = out["nces_school_name_clean"].fillna("").astype(str)

    out["high_school_candidate"] = (
        (out["grade_high_num"] >= 12) |
        school_level_text.str.contains("HIGH|SECONDARY", regex=True, na=False) |
        school_name_text.str.contains(r"\bHIGH SCHOOL\b|\bSENIOR HIGH\b", regex=True, na=False)
    )

    out["has_coordinates"] = out["latitude"].notna() & out["longitude"].notna()

    wanted_columns = [
        "nces_school_id",
        "nces_district_id",
        "nces_school_name",
        "nces_school_name_clean",
        "nces_district_name",
        "county_name",
        "nces_parish",
        "state",
        "city",
        "zip_code",
        "latitude",
        "longitude",
        "has_coordinates",
        "school_type",
        "school_level",
        "operational_status",
        "grade_low",
        "grade_high",
        "grade_low_num",
        "grade_high_num",
        "charter",
        "virtual",
        "high_school_candidate",
    ]

    return out[wanted_columns].copy()


def main() -> None:
    """Prepare the NCES Louisiana school list."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    input_path = find_first_csv()
    print("Reading:", input_path.relative_to(PROJECT_ROOT))

    raw = read_csv_safely(input_path)

    print("\nRaw shape:", raw.shape)
    print("\nRaw columns:")
    print(list(raw.columns))

    nces = prepare_nces_school_list(raw)

    all_path = INTERMEDIATE_DIR / "nces_louisiana_schools.csv"
    hs_path = INTERMEDIATE_DIR / "nces_louisiana_high_school_candidates.csv"

    nces.to_csv(all_path, index=False)
    nces[nces["high_school_candidate"]].to_csv(hs_path, index=False)

    print("\nSaved:")
    print(all_path.relative_to(PROJECT_ROOT))
    print(hs_path.relative_to(PROJECT_ROOT))

    print("\nPrepared shape:", nces.shape)
    print("High school candidates:", int(nces["high_school_candidate"].sum()))
    print("Rows with coordinates:", int(nces["has_coordinates"].sum()))

    print("\nState counts:")
    print(nces["state"].value_counts(dropna=False).head(20))

    print("\nSchool level counts:")
    print(nces["school_level"].value_counts(dropna=False).head(20))

    print("\nOperational status counts:")
    print(nces["operational_status"].value_counts(dropna=False).head(20))

    print("\nSample high school candidates:")
    print(
        nces[nces["high_school_candidate"]][
            [
                "nces_school_id",
                "nces_school_name",
                "nces_parish",
                "nces_district_name",
                "grade_low",
                "grade_high",
                "latitude",
                "longitude",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
