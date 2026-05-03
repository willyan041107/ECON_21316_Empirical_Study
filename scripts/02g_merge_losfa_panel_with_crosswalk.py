"""
Merge LOSFA school-year panel with finalized school crosswalk using the same
canonical school-key logic as 02d.

Inputs:
    data/intermediate/losfa_panel_clean.csv
    data/intermediate/school_crosswalk.csv
    scripts/02d_build_losfa_nces_crosswalk.py

Outputs:
    data/intermediate/losfa_panel_with_crosswalk.csv
    data/intermediate/losfa_crosswalk_coverage_by_year.csv
    data/intermediate/losfa_unmatched_after_canonical_merge.csv

This script also backs up losfa_panel_clean.csv and removes accidental aggregate
rows such as ORLEANS SubTotal.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

LOSFA_PATH = INTERMEDIATE_DIR / "losfa_panel_clean.csv"
LOSFA_BACKUP_PATH = INTERMEDIATE_DIR / "losfa_panel_clean_before_subtotal_cleanup.csv"
CROSSWALK_PATH = INTERMEDIATE_DIR / "school_crosswalk.csv"

OUTPUT_PANEL_PATH = INTERMEDIATE_DIR / "losfa_panel_with_crosswalk.csv"
COVERAGE_BY_YEAR_PATH = INTERMEDIATE_DIR / "losfa_crosswalk_coverage_by_year.csv"
UNMATCHED_PATH = INTERMEDIATE_DIR / "losfa_unmatched_after_canonical_merge.csv"


def load_02d_module():
    """Load canonicalization functions from 02d crosswalk script."""
    module_path = PROJECT_ROOT / "scripts" / "02d_build_losfa_nces_crosswalk.py"
    spec = importlib.util.spec_from_file_location("crosswalk_02d", module_path)

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 02d module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main() -> None:
    """Merge LOSFA panel to school crosswalk and report coverage."""
    crosswalk_02d = load_02d_module()

    losfa = pd.read_csv(LOSFA_PATH, dtype=str, low_memory=False)
    crosswalk = pd.read_csv(CROSSWALK_PATH, dtype=str, low_memory=False)

    # Backup original LOSFA panel before cleaning aggregate rows.
    if not LOSFA_BACKUP_PATH.exists():
        losfa.to_csv(LOSFA_BACKUP_PATH, index=False)

    original_rows = len(losfa)

    losfa["hs_name"] = losfa["hs_name"].fillna("").astype(str)

    aggregate_mask = losfa["hs_name"].str.upper().str.contains(
        r"\bSUBTOTAL\b|\bGRAND TOTAL\b|\bSTATE TOTAL\b",
        regex=True,
        na=False,
    )

    removed_aggregate = losfa[aggregate_mask].copy()
    losfa = losfa[~aggregate_mask].copy()

    # Save cleaned LOSFA panel back to the standard path.
    losfa.to_csv(LOSFA_PATH, index=False)

    # Build canonical LOSFA school key using 02d functions.
    losfa["losfa_parish"] = losfa["parish"].apply(crosswalk_02d.normalize_parish)
    losfa["losfa_hs_name"] = losfa["hs_name"].fillna("").astype(str).str.strip()

    losfa["losfa_match_name_original"] = losfa["losfa_hs_name"].apply(
        crosswalk_02d.normalize_name
    )

    losfa["losfa_match_name_stripped"] = losfa.apply(
        lambda row: crosswalk_02d.strip_leading_parish_prefix(
            row["losfa_hs_name"],
            row["losfa_parish"],
        ),
        axis=1,
    )

    losfa["losfa_match_name"] = losfa.apply(
        crosswalk_02d.choose_canonical_losfa_name,
        axis=1,
    )

    losfa["losfa_school_key"] = (
        losfa["losfa_parish"] + " | " + losfa["losfa_match_name"]
    )

    keep_cols = [
        "losfa_school_key",
        "final_match_status",
        "accepted_match_source",
        "nces_school_id",
        "nces_school_name",
        "nces_parish",
        "nces_district_name",
        "latitude",
        "longitude",
        "has_final_nces_match",
        "has_coordinates",
        "usable_for_hurricane_exposure",
        "api_decision",
        "api_confidence",
        "api_reason",
    ]
    keep_cols = [col for col in keep_cols if col in crosswalk.columns]

    merged = losfa.merge(
        crosswalk[keep_cols],
        on="losfa_school_key",
        how="left",
        validate="many_to_one",
    )

    merged["students_processed"] = pd.to_numeric(
        merged["students_processed"],
        errors="coerce",
    )
    merged["graduation_year"] = pd.to_numeric(
        merged["graduation_year"],
        errors="coerce",
    )

    merged["usable_for_hurricane_exposure"] = (
        merged["usable_for_hurricane_exposure"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .eq("true")
    )

    merged["has_crosswalk_row"] = merged["final_match_status"].notna()

    merged.to_csv(OUTPUT_PANEL_PATH, index=False)

    coverage = merged.groupby("graduation_year").agg(
        rows=("hs_name", "size"),
        crosswalk_rows=("has_crosswalk_row", "sum"),
        usable_rows=("usable_for_hurricane_exposure", "sum"),
        students=("students_processed", "sum"),
        usable_students=("students_processed", lambda x: x[merged.loc[x.index, "usable_for_hurricane_exposure"]].sum()),
    )
    coverage["share_rows_usable"] = coverage["usable_rows"] / coverage["rows"]
    coverage["share_students_usable"] = coverage["usable_students"] / coverage["students"]
    coverage.to_csv(COVERAGE_BY_YEAR_PATH)

    unmatched = (
        merged[~merged["usable_for_hurricane_exposure"]]
        .groupby(["parish", "hs_name", "final_match_status"], dropna=False)
        .agg(
            years=("graduation_year", "nunique"),
            students=("students_processed", "sum"),
        )
        .reset_index()
        .sort_values("students", ascending=False)
    )
    unmatched.to_csv(UNMATCHED_PATH, index=False)

    print("Original LOSFA rows:", original_rows)
    print("Removed aggregate rows:", len(removed_aggregate))
    if len(removed_aggregate) > 0:
        print(removed_aggregate[["graduation_year", "parish", "hs_name", "students_processed"]].to_string(index=False))

    print("\nSaved:")
    print(" -", OUTPUT_PANEL_PATH.relative_to(PROJECT_ROOT))
    print(" -", COVERAGE_BY_YEAR_PATH.relative_to(PROJECT_ROOT))
    print(" -", UNMATCHED_PATH.relative_to(PROJECT_ROOT))

    print("\nMerged rows:", len(merged))
    print("Rows with any crosswalk row:", int(merged["has_crosswalk_row"].sum()))
    print("Usable rows:", int(merged["usable_for_hurricane_exposure"].sum()))
    print("Share usable rows:", round(merged["usable_for_hurricane_exposure"].mean(), 3))

    total_students = merged["students_processed"].sum()
    usable_students = merged.loc[
        merged["usable_for_hurricane_exposure"],
        "students_processed",
    ].sum()

    print("\nStudents total:", int(total_students))
    print("Students usable:", int(usable_students))
    print("Share students usable:", round(usable_students / total_students, 3))

    print("\nFinal match status counts by school-year rows:")
    print(merged["final_match_status"].value_counts(dropna=False))

    print("\nCoverage by year, recent years:")
    print(coverage.tail(10).to_string())

    print("\nTop unmatched after canonical merge:")
    print(unmatched.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
