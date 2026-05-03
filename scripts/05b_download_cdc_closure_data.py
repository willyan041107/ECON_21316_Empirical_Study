"""
Download CDC Prolonged Unplanned School Closures data.

Official source:
    Data.CDC.gov dataset identifier: 5iuf-feyd
    Prolonged Unplanned School Closures: USA, 2011-2019

Outputs:
    data/raw/closures/cdc/cdc_prolonged_unplanned_school_closures_2011_2019.csv
    data/raw/closures/cdc/cdc_prolonged_unplanned_school_closures_2011_2019_metadata.json
    data/intermediate/cdc_closure_download_summary.txt
    data/intermediate/cdc_closure_columns.csv

This script only downloads and inspects the dataset.
It does not yet standardize records into the closure schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CDC_DIR = PROJECT_ROOT / "data" / "raw" / "closures" / "cdc"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

CSV_OUT = RAW_CDC_DIR / "cdc_prolonged_unplanned_school_closures_2011_2019.csv"
METADATA_OUT = RAW_CDC_DIR / "cdc_prolonged_unplanned_school_closures_2011_2019_metadata.json"
SUMMARY_OUT = INTERMEDIATE_DIR / "cdc_closure_download_summary.txt"
COLUMNS_OUT = INTERMEDIATE_DIR / "cdc_closure_columns.csv"

DATASET_ID = "5iuf-feyd"

# Socrata CSV API endpoint. $limit avoids the default small row limit.
CSV_URL = "https://data.cdc.gov/resource/5iuf-feyd.csv?$limit=50000"
METADATA_URL = "https://data.cdc.gov/api/views/5iuf-feyd"


def download_url(url: str, output_path: Path) -> None:
    """Download a URL to a local file."""
    request = Request(
        url,
        headers={
            "User-Agent": "ECON_21316_Empirical_Study/1.0",
        },
    )

    with urlopen(request, timeout=120) as response:
        content = response.read()

    output_path.write_bytes(content)


def main() -> None:
    """Download CDC closure data and print basic diagnostics."""
    RAW_CDC_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CDC closure CSV...")
    download_url(CSV_URL, CSV_OUT)

    print("Downloading CDC metadata...")
    download_url(METADATA_URL, METADATA_OUT)

    df = pd.read_csv(CSV_OUT, dtype=str, low_memory=False)

    with METADATA_OUT.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    columns = pd.DataFrame(
        {
            "column_name": list(df.columns),
            "nonmissing_count": [df[col].notna().sum() for col in df.columns],
            "sample_values": [
                "; ".join(df[col].dropna().astype(str).head(5).tolist())
                for col in df.columns
            ],
        }
    )
    columns.to_csv(COLUMNS_OUT, index=False)

    # Try to detect state-like columns for a first Louisiana check.
    possible_state_cols = [
        col for col in df.columns
        if "state" in col.lower() or col.lower() in {"st", "state_name"}
    ]

    summary_lines = []
    summary_lines.append("CDC Prolonged Unplanned School Closures Download Summary")
    summary_lines.append("=" * 65)
    summary_lines.append("")
    summary_lines.append(f"Dataset ID: {DATASET_ID}")
    summary_lines.append(f"Rows: {len(df):,}")
    summary_lines.append(f"Columns: {len(df.columns):,}")
    summary_lines.append("")
    summary_lines.append("Title from metadata:")
    summary_lines.append(str(metadata.get("name", "")))
    summary_lines.append("")
    summary_lines.append("Columns:")
    for col in df.columns:
        summary_lines.append(f" - {col}")
    summary_lines.append("")

    if possible_state_cols:
        summary_lines.append("Possible state columns:")
        for col in possible_state_cols:
            summary_lines.append(f" - {col}")
            summary_lines.append(df[col].value_counts(dropna=False).head(20).to_string())
            summary_lines.append("")

            louisiana_mask = df[col].astype(str).str.upper().isin(["LA", "LOUISIANA"])
            if louisiana_mask.any():
                louisiana = df[louisiana_mask].copy()
                summary_lines.append(f"Louisiana rows using column `{col}`: {len(louisiana):,}")
                summary_lines.append(louisiana.head(20).to_string(index=False))
                summary_lines.append("")
    else:
        summary_lines.append("No obvious state column detected. Inspect cdc_closure_columns.csv.")

    SUMMARY_OUT.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print("Saved:")
    print(" -", CSV_OUT.relative_to(PROJECT_ROOT))
    print(" -", METADATA_OUT.relative_to(PROJECT_ROOT))
    print(" -", COLUMNS_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_OUT.relative_to(PROJECT_ROOT))

    print()
    print("Shape:", df.shape)
    print("Columns:")
    print(list(df.columns))

    print()
    print("Possible state columns:", possible_state_cols)

    print()
    print("Open summary with:")
    print(f"open {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
