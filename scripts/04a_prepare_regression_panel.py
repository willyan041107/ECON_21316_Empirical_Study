"""
Prepare regression-ready school-year panel.

Input:
    data/processed/analysis_panel_with_exposure.csv

Outputs:
    data/processed/regression_panel_full_with_flags.csv
    data/processed/regression_panel.csv
    data/processed/regression_panel_by_year.csv
    data/processed/regression_panel_summary.txt

Main sample:
    usable_for_hurricane_exposure == True
    valid_for_main_analysis == True
    exposure_index_pointmax is nonmissing
    eligibility_rate is nonmissing
    students_processed >= min_students

Final regression unit:
    NCES school × graduation year

Main outcome:
    eligibility_rate

Main exposure:
    exposure_index_pointmax

Robustness exposure:
    exposure_index_stormmax
    within_50km_hurricane
    within_100km_hurricane
    within_50km_major_hurricane
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_panel_with_exposure.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FULL_WITH_FLAGS_OUT = PROCESSED_DIR / "regression_panel_full_with_flags.csv"
REGRESSION_PANEL_OUT = PROCESSED_DIR / "regression_panel.csv"
BY_YEAR_OUT = PROCESSED_DIR / "regression_panel_by_year.csv"
SUMMARY_OUT = PROCESSED_DIR / "regression_panel_summary.txt"


COUNT_COLUMNS = [
    "students_processed",
    "opportunity_eligible",
    "opportunity_recipients",
    "performance_eligible",
    "performance_recipients",
    "honors_eligible",
    "honors_recipients",
    "excellence_eligible",
    "excellence_recipients",
    "topstech_eligible",
    "topstech_recipients",
    "total_eligible",
    "total_recipients",
]

RATE_COLUMNS = [
    "eligibility_rate",
    "recipient_rate",
    "acceptance_rate",
]

EXPOSURE_NUMERIC_COLUMNS = [
    "exposure_index_pointmax",
    "exposure_index_stormmax",
    "min_distance_any_tropical_km",
    "min_distance_hurricane_km",
    "min_distance_major_hurricane_km",
    "max_exposure_distance_km",
    "max_exposure_wind_kt",
    "nearest_storm_wind_kt",
    "num_relevant_storms",
]

EXPOSURE_STRING_COLUMNS = [
    "nearest_storm_id",
    "nearest_storm_name",
    "max_exposure_storm_id",
    "max_exposure_storm_name",
]

EXPOSURE_BOOLEAN_COLUMNS = [
    "within_50km_any_tropical",
    "within_100km_any_tropical",
    "within_50km_hurricane",
    "within_100km_hurricane",
    "within_50km_major_hurricane",
]

IDENTITY_COLUMNS = [
    "nces_school_id",
    "nces_school_name",
    "nces_parish",
    "nces_district_name",
    "latitude",
    "longitude",
    "graduation_year",
    "storm_year",
]


def to_bool(series: pd.Series) -> pd.Series:
    """Convert string-like boolean values to boolean."""
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def normalize_id(value: object) -> str:
    """Normalize NCES ID as string."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if text.lower() in {"nan", "none", ""}:
        return ""

    return text.zfill(12)


def first_nonmissing(series: pd.Series) -> object:
    """Return the first nonmissing and nonempty value."""
    clean = series.dropna()

    for value in clean:
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            return value

    return pd.NA


def join_unique(series: pd.Series) -> str:
    """Join unique nonmissing values."""
    values = []

    for value in series.dropna().astype(str):
        value = value.strip()
        if value and value.lower() not in {"nan", "none"} and value not in values:
            values.append(value)

    return "; ".join(values)


def prepare_full_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data types and create sample flags."""
    out = df.copy()

    for col in COUNT_COLUMNS + RATE_COLUMNS + EXPOSURE_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["graduation_year", "storm_year", "latitude", "longitude"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in EXPOSURE_BOOLEAN_COLUMNS:
        if col in out.columns:
            out[col] = to_bool(out[col])

    for col in ["usable_for_hurricane_exposure", "valid_for_main_analysis"]:
        if col in out.columns:
            out[col] = to_bool(out[col])
        else:
            out[col] = False

    out["nces_school_id"] = out["nces_school_id"].apply(normalize_id)

    out["has_valid_outcome"] = (
        out["eligibility_rate"].notna()
        & out["eligibility_rate"].between(0, 1)
    )

    out["has_valid_main_exposure"] = out["exposure_index_pointmax"].notna()

    out["has_valid_school_id"] = out["nces_school_id"].astype(str).str.len().gt(0)

    out["base_regression_sample"] = (
        out["usable_for_hurricane_exposure"]
        & out["valid_for_main_analysis"]
        & out["has_valid_outcome"]
        & out["has_valid_main_exposure"]
        & out["has_valid_school_id"]
        & out["students_processed"].notna()
        & (out["students_processed"] > 0)
    )

    return out


def collapse_to_nces_school_year(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse LOSFA rows to NCES school × graduation year."""
    rows = []

    group_cols = ["nces_school_id", "graduation_year"]

    for (nces_school_id, graduation_year), group in df.groupby(group_cols, dropna=False):
        group = group.copy()

        exposure_sort = group["exposure_index_pointmax"].fillna(-np.inf)
        exposure_row = group.loc[exposure_sort.idxmax()]

        row = {
            "nces_school_id": normalize_id(nces_school_id),
            "graduation_year": int(graduation_year),
            "school_fe": normalize_id(nces_school_id),
            "year_fe": str(int(graduation_year)),
            "num_losfa_rows_aggregated": len(group),
            "losfa_school_keys": join_unique(group.get("losfa_school_key", pd.Series(dtype=str))),
            "losfa_hs_names": join_unique(group.get("hs_name", pd.Series(dtype=str))),
            "losfa_parishes": join_unique(group.get("parish", pd.Series(dtype=str))),
        }

        for col in [
            "nces_school_name",
            "nces_parish",
            "nces_district_name",
            "storm_year",
            "latitude",
            "longitude",
            "final_match_status",
            "accepted_match_source",
        ]:
            if col in group.columns:
                row[col] = first_nonmissing(group[col])

        for col in COUNT_COLUMNS:
            if col in group.columns:
                row[col] = group[col].sum(skipna=True)

        students = row.get("students_processed", np.nan)
        total_eligible = row.get("total_eligible", np.nan)
        total_recipients = row.get("total_recipients", np.nan)

        row["eligibility_rate"] = (
            total_eligible / students
            if pd.notna(students) and students > 0 and pd.notna(total_eligible)
            else np.nan
        )

        row["recipient_rate"] = (
            total_recipients / students
            if pd.notna(students) and students > 0 and pd.notna(total_recipients)
            else np.nan
        )

        row["acceptance_rate"] = (
            total_recipients / total_eligible
            if pd.notna(total_eligible) and total_eligible > 0 and pd.notna(total_recipients)
            else np.nan
        )

        for col in EXPOSURE_NUMERIC_COLUMNS:
            if col in group.columns:
                row[col] = exposure_row[col]

        for col in EXPOSURE_STRING_COLUMNS:
            if col in group.columns:
                row[col] = exposure_row[col]

        for col in EXPOSURE_BOOLEAN_COLUMNS:
            if col in group.columns:
                row[col] = bool(group[col].fillna(False).any())

        rows.append(row)

    collapsed = pd.DataFrame(rows)

    collapsed["parish_fe"] = collapsed["nces_parish"].fillna("").astype(str)
    collapsed["district_fe"] = collapsed["nces_district_name"].fillna("").astype(str)

    collapsed["log_students_processed"] = np.log(collapsed["students_processed"].replace(0, np.nan))

    collapsed["main_outcome"] = collapsed["eligibility_rate"]
    collapsed["main_exposure"] = collapsed["exposure_index_pointmax"]

    collapsed["robust_exposure_stormmax"] = collapsed["exposure_index_stormmax"]

    collapsed["treat_within_50km_hurricane"] = collapsed["within_50km_hurricane"].astype(int)
    collapsed["treat_within_100km_hurricane"] = collapsed["within_100km_hurricane"].astype(int)
    collapsed["treat_within_50km_major_hurricane"] = collapsed["within_50km_major_hurricane"].astype(int)

    return collapsed.sort_values(["nces_school_id", "graduation_year"]).reset_index(drop=True)


def add_standardized_exposures(df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized exposure variables."""
    out = df.copy()

    for source_col, new_col in [
        ("exposure_index_pointmax", "exposure_pointmax_z"),
        ("exposure_index_stormmax", "exposure_stormmax_z"),
    ]:
        mean = out[source_col].mean()
        std = out[source_col].std(ddof=0)

        if pd.notna(std) and std > 0:
            out[new_col] = (out[source_col] - mean) / std
        else:
            out[new_col] = np.nan

    p90 = out["exposure_index_pointmax"].quantile(0.90)
    p75 = out["exposure_index_pointmax"].quantile(0.75)

    out["high_exposure_pointmax_p75"] = (out["exposure_index_pointmax"] >= p75).astype(int)
    out["high_exposure_pointmax_p90"] = (out["exposure_index_pointmax"] >= p90).astype(int)

    return out


def make_by_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create regression-panel summary by graduation year."""
    return (
        df.groupby("graduation_year")
        .agg(
            rows=("nces_school_id", "size"),
            unique_schools=("nces_school_id", "nunique"),
            students=("students_processed", "sum"),
            mean_eligibility_rate=("eligibility_rate", "mean"),
            mean_recipient_rate=("recipient_rate", "mean"),
            mean_acceptance_rate=("acceptance_rate", "mean"),
            mean_exposure_pointmax=("exposure_index_pointmax", "mean"),
            p90_exposure_pointmax=("exposure_index_pointmax", lambda x: x.quantile(0.90)),
            max_exposure_pointmax=("exposure_index_pointmax", "max"),
            mean_exposure_stormmax=("exposure_index_stormmax", "mean"),
            share_within_50km_hurricane=("within_50km_hurricane", "mean"),
            share_within_100km_hurricane=("within_100km_hurricane", "mean"),
        )
        .reset_index()
        .sort_values("graduation_year")
    )


def write_summary(full: pd.DataFrame, base: pd.DataFrame, regression: pd.DataFrame, by_year: pd.DataFrame, min_students: int) -> None:
    """Write plain-text summary."""
    lines = []

    lines.append("Regression Panel Preparation Summary")
    lines.append("=" * 45)
    lines.append("")
    lines.append(f"Input rows: {len(full):,}")
    lines.append(f"Rows in base regression sample before NCES-year collapse: {len(base):,}")
    lines.append(f"Rows in final regression panel after NCES-year collapse: {len(regression):,}")
    lines.append(f"Minimum students threshold: {min_students}")
    lines.append("")
    lines.append(f"Unique NCES schools: {regression['nces_school_id'].nunique():,}")
    lines.append(f"Graduation years: {int(regression['graduation_year'].min())}–{int(regression['graduation_year'].max())}")
    lines.append(f"Total students_processed: {int(regression['students_processed'].sum()):,}")
    lines.append("")
    lines.append("Outcome summary:")
    lines.append(regression[["eligibility_rate", "recipient_rate", "acceptance_rate"]].describe().to_string())
    lines.append("")
    lines.append("Exposure summary:")
    lines.append(regression[[
        "exposure_index_pointmax",
        "exposure_index_stormmax",
        "within_50km_hurricane",
        "within_100km_hurricane",
        "within_50km_major_hurricane",
    ]].describe(include="all").to_string())
    lines.append("")
    lines.append("Rows by year:")
    lines.append(by_year.to_string(index=False))

    SUMMARY_OUT.write_text("\n".join(lines))


def main() -> None:
    """Prepare regression-ready panel."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-students",
        type=int,
        default=20,
        help="Minimum students_processed for final regression panel.",
    )
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)

    full = prepare_full_panel(raw)
    full.to_csv(FULL_WITH_FLAGS_OUT, index=False)

    base = full[full["base_regression_sample"]].copy()

    collapsed = collapse_to_nces_school_year(base)

    collapsed["meets_min_students"] = collapsed["students_processed"] >= args.min_students

    regression = collapsed[
        collapsed["meets_min_students"]
        & collapsed["eligibility_rate"].notna()
        & collapsed["eligibility_rate"].between(0, 1)
        & collapsed["exposure_index_pointmax"].notna()
    ].copy()

    regression = add_standardized_exposures(regression)

    duplicate_school_years = regression.duplicated(["nces_school_id", "graduation_year"]).sum()
    if duplicate_school_years != 0:
        raise ValueError(f"Duplicate NCES school-year rows remain: {duplicate_school_years}")

    by_year = make_by_year_summary(regression)

    regression.to_csv(REGRESSION_PANEL_OUT, index=False)
    by_year.to_csv(BY_YEAR_OUT, index=False)

    write_summary(full, base, regression, by_year, args.min_students)

    print("Saved:")
    print(" -", FULL_WITH_FLAGS_OUT.relative_to(PROJECT_ROOT))
    print(" -", REGRESSION_PANEL_OUT.relative_to(PROJECT_ROOT))
    print(" -", BY_YEAR_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_OUT.relative_to(PROJECT_ROOT))

    print("\n=== Summary ===")
    print("Input rows:", len(full))
    print("Base sample rows before collapse:", len(base))
    print("Final regression rows:", len(regression))
    print("Unique NCES schools:", regression["nces_school_id"].nunique())
    print("Years:", int(regression["graduation_year"].min()), "to", int(regression["graduation_year"].max()))
    print("Total students:", int(regression["students_processed"].sum()))
    print("Duplicate NCES school-year rows:", duplicate_school_years)

    print("\nOutcome summary:")
    print(regression[["eligibility_rate", "recipient_rate", "acceptance_rate"]].describe().to_string())

    print("\nExposure summary:")
    print(regression[[
        "exposure_index_pointmax",
        "exposure_index_stormmax",
        "exposure_pointmax_z",
        "within_50km_hurricane",
        "within_100km_hurricane",
        "within_50km_major_hurricane",
    ]].describe(include="all").to_string())

    print("\nRecent years summary:")
    print(by_year.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
