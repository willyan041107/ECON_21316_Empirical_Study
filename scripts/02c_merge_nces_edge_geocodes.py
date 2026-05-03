"""
Merge NCES EDGE public school geocodes into the prepared NCES Louisiana school list.

Inputs:
    data/intermediate/nces_louisiana_schools.csv
    data/raw/nces/edge_public_school_geocodes/extracted/EDGE_GEOCODE_PUBLICSCH_2425.xlsx

Outputs:
    data/intermediate/nces_louisiana_schools_with_geocode.csv
    data/intermediate/nces_louisiana_high_school_candidates_with_geocode.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
EDGE_DIR = PROJECT_ROOT / "data" / "raw" / "nces" / "edge_public_school_geocodes" / "extracted"


def normalize_id(value: object) -> str:
    """Normalize NCES school IDs for merging."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text.zfill(12)


def find_edge_file() -> Path:
    """Find the EDGE geocode file, preferring Excel."""
    xlsx_files = sorted(EDGE_DIR.rglob("*.xlsx"))
    txt_files = sorted(EDGE_DIR.rglob("*.TXT")) + sorted(EDGE_DIR.rglob("*.txt"))

    if xlsx_files:
        return xlsx_files[0]

    if txt_files:
        return txt_files[0]

    raise FileNotFoundError(f"No EDGE Excel or TXT file found in {EDGE_DIR}")


def read_edge_file(path: Path) -> pd.DataFrame:
    """Read EDGE geocode file."""
    print("Reading EDGE file:", path.relative_to(PROJECT_ROOT))

    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, dtype=str)

    return pd.read_csv(path, dtype=str, sep=None, engine="python", low_memory=False)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find a column by possible names, case-insensitive."""
    lookup = {col.upper(): col for col in df.columns}

    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]

    raise KeyError(f"Could not find any of these columns: {candidates}")


def normalize_parish_name(value: object) -> str:
    """Normalize Louisiana parish/county names."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = text.replace(" PARISH", "")
    text = text.replace(" COUNTY", "")
    text = " ".join(text.split())

    return text


def prepare_edge(edge_raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize EDGE geocode fields."""
    nces_col = find_column(edge_raw, ["NCESSCH", "NCESID", "NCES_SCHOOL_ID"])
    lat_col = find_column(edge_raw, ["LAT", "LATITUDE", "LATCOD"])
    lon_col = find_column(edge_raw, ["LON", "LONGITUDE", "LONCOD"])

    try:
        state_col = find_column(edge_raw, ["LSTATE", "STATE", "ST"])
        state = edge_raw[state_col].astype(str).str.upper().str.strip()
    except KeyError:
        state = pd.Series([""] * len(edge_raw))

    try:
        county_col = find_column(edge_raw, ["NMCNTY", "CONAME", "COUNTY_NAME", "COUNTY"])
        edge_county_name = edge_raw[county_col]
    except KeyError:
        edge_county_name = pd.Series([pd.NA] * len(edge_raw))

    edge = pd.DataFrame(
        {
            "nces_school_id": edge_raw[nces_col].apply(normalize_id),
            "edge_latitude": pd.to_numeric(edge_raw[lat_col], errors="coerce"),
            "edge_longitude": pd.to_numeric(edge_raw[lon_col], errors="coerce"),
            "edge_state": state,
            "edge_county_name": edge_county_name,
        }
    )

    edge["edge_parish"] = edge["edge_county_name"].apply(normalize_parish_name)

    if (edge["edge_state"] != "").any():
        edge = edge[edge["edge_state"].eq("LA")].copy()

    edge = edge.drop_duplicates(subset=["nces_school_id"])

    return edge


def to_bool(series: pd.Series) -> pd.Series:
    """Convert string or boolean-like values to boolean."""
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    """Merge EDGE geocodes into NCES school list."""
    nces_path = INTERMEDIATE_DIR / "nces_louisiana_schools.csv"

    if not nces_path.exists():
        raise FileNotFoundError(f"Missing file: {nces_path}")

    nces = pd.read_csv(nces_path, dtype=str, low_memory=False)
    nces["nces_school_id"] = nces["nces_school_id"].apply(normalize_id)

    edge_path = find_edge_file()
    edge_raw = read_edge_file(edge_path)

    print("\nEDGE raw shape:", edge_raw.shape)
    print("EDGE raw columns:")
    print(list(edge_raw.columns))

    edge = prepare_edge(edge_raw)

    print("\nEDGE prepared shape:", edge.shape)
    print("EDGE rows with coordinates:", int(edge["edge_latitude"].notna().sum()))

    merged = nces.merge(edge, on="nces_school_id", how="left", validate="one_to_one")

    merged["latitude"] = pd.to_numeric(merged["latitude"], errors="coerce")
    merged["longitude"] = pd.to_numeric(merged["longitude"], errors="coerce")

    merged["latitude"] = merged["latitude"].fillna(merged["edge_latitude"])
    merged["longitude"] = merged["longitude"].fillna(merged["edge_longitude"])
    merged["has_coordinates"] = merged["latitude"].notna() & merged["longitude"].notna()

    all_path = INTERMEDIATE_DIR / "nces_louisiana_schools_with_geocode.csv"
    hs_path = INTERMEDIATE_DIR / "nces_louisiana_high_school_candidates_with_geocode.csv"

    merged.to_csv(all_path, index=False)

    hs = merged[to_bool(merged["high_school_candidate"])].copy()
    hs.to_csv(hs_path, index=False)

    print("\nSaved:")
    print(all_path.relative_to(PROJECT_ROOT))
    print(hs_path.relative_to(PROJECT_ROOT))

    print("\nMerged shape:", merged.shape)
    print("Rows with coordinates:", int(merged["has_coordinates"].sum()))
    print("High school candidates:", len(hs))
    print("High school candidates with coordinates:", int(hs["has_coordinates"].sum()))

    print("\nSample high school candidates:")
    print(
        hs[
            [
                "nces_school_id",
                "nces_school_name",
                "nces_parish",
                "nces_district_name",
                "grade_low",
                "grade_high",
                "latitude",
                "longitude",
                "has_coordinates",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
