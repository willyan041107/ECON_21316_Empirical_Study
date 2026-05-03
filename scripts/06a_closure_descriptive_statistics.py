"""
Create descriptive statistics for auto-collected school closure measures.

Input:
    data/processed/regression_panel_with_auto_closures.csv

Outputs:
    data/analysis/closure_descriptive_sample_summary.csv
    data/analysis/closure_descriptive_variable_summary.csv
    data/analysis/closure_by_graduation_year.csv
    data/analysis/closure_by_status_broad.csv
    data/analysis/closure_by_status_strict.csv
    data/analysis/closure_by_exposure_quartile.csv
    data/analysis/closure_event_summary_broad.csv
    data/analysis/closure_top_school_years_broad.csv
    data/analysis/closure_consistency_checks.csv
    reports/06a_closure_descriptive_statistics.txt

Purpose:
    Summarize the strict and broad hurricane-related closure measures before
    using them in regressions.

Notes:
    This script is descriptive only. It does not estimate causal effects.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "regression_panel_with_auto_closures.csv"

ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
REPORTS_DIR = PROJECT_ROOT / "reports"

SAMPLE_SUMMARY_OUT = ANALYSIS_DIR / "closure_descriptive_sample_summary.csv"
VARIABLE_SUMMARY_OUT = ANALYSIS_DIR / "closure_descriptive_variable_summary.csv"
YEAR_SUMMARY_OUT = ANALYSIS_DIR / "closure_by_graduation_year.csv"
BROAD_STATUS_OUT = ANALYSIS_DIR / "closure_by_status_broad.csv"
STRICT_STATUS_OUT = ANALYSIS_DIR / "closure_by_status_strict.csv"
EXPOSURE_QUARTILE_OUT = ANALYSIS_DIR / "closure_by_exposure_quartile.csv"
EVENT_SUMMARY_OUT = ANALYSIS_DIR / "closure_event_summary_broad.csv"
TOP_SCHOOL_YEARS_OUT = ANALYSIS_DIR / "closure_top_school_years_broad.csv"
CONSISTENCY_CHECKS_OUT = ANALYSIS_DIR / "closure_consistency_checks.csv"
REPORT_OUT = REPORTS_DIR / "06a_closure_descriptive_statistics.txt"


OUTCOME_COLS = [
    "eligibility_rate",
    "recipient_rate",
    "acceptance_rate",
]

EXPOSURE_COLS = [
    "exposure_index_pointmax",
    "exposure_index_stormmax",
    "within_50km_hurricane",
    "within_100km_hurricane",
    "within_50km_major_hurricane",
]

CLOSURE_COLS = [
    "closure_days_hurricane_related_strict",
    "closure_any_hurricane_related_strict",
    "closure_max_consecutive_days_strict",
    "closure_source_record_count_strict",
    "closure_source_url_count_strict",
    "closure_days_hurricane_related_broad",
    "closure_any_hurricane_related_broad",
    "closure_max_consecutive_days_broad",
    "closure_source_record_count_broad",
    "closure_source_url_count_broad",
]


def normalize_id(value: object, width: int = 12) -> str:
    """Normalize numeric IDs to string IDs."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if text.isdigit():
        return text.zfill(width)

    return text


def parse_numeric_or_bool(series: pd.Series) -> pd.Series:
    """Convert numeric and boolean-like strings to numeric values."""
    as_text = series.astype(str).str.strip().str.lower()

    bool_mapped = as_text.map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "nan": None,
            "none": None,
            "null": None,
            "": None,
        }
    )

    numeric = pd.to_numeric(series, errors="coerce")

    return numeric.fillna(bool_mapped)


def find_students_column(df: pd.DataFrame) -> str | None:
    """Find the student-count column used for weighting."""
    candidates = [
        "total_students_processed",
        "students_processed",
        "total_processed",
        "processed_students",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str | None) -> float:
    """Compute weighted mean when possible; otherwise return ordinary mean."""
    if value_col not in df.columns:
        return float("nan")

    values = pd.to_numeric(df[value_col], errors="coerce")

    if weight_col is None or weight_col not in df.columns:
        return float(values.mean())

    weights = pd.to_numeric(df[weight_col], errors="coerce")

    valid = values.notna() & weights.notna() & weights.gt(0)

    if not valid.any():
        return float(values.mean())

    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def prepare_panel(path: Path) -> pd.DataFrame:
    """Read and clean regression panel with closure variables."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path, dtype=str, low_memory=False)

    if "nces_school_id" in df.columns:
        df["nces_school_id"] = df["nces_school_id"].apply(normalize_id)

    numeric_cols = [
        "graduation_year",
        "latitude",
        "longitude",
        *OUTCOME_COLS,
        *EXPOSURE_COLS,
        *CLOSURE_COLS,
    ]

    students_col = find_students_column(df)

    if students_col is not None:
        numeric_cols.append(students_col)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = parse_numeric_or_bool(df[col])

    for col in CLOSURE_COLS:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def make_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create high-level sample summary."""
    students_col = find_students_column(df)

    rows = [
        ("panel_rows", len(df)),
        ("unique_nces_schools", df["nces_school_id"].nunique() if "nces_school_id" in df.columns else None),
        ("first_graduation_year", int(df["graduation_year"].min()) if "graduation_year" in df.columns else None),
        ("last_graduation_year", int(df["graduation_year"].max()) if "graduation_year" in df.columns else None),
    ]

    if students_col is not None:
        rows.extend(
            [
                ("total_students_processed", df[students_col].sum()),
                ("mean_students_per_school_year", df[students_col].mean()),
                ("median_students_per_school_year", df[students_col].median()),
            ]
        )

    rows.extend(
        [
            ("strict_rows_with_closure", int((df["closure_days_hurricane_related_strict"] > 0).sum())),
            ("broad_rows_with_closure", int((df["closure_days_hurricane_related_broad"] > 0).sum())),
            ("strict_total_school_closure_days", int(df["closure_days_hurricane_related_strict"].sum())),
            ("broad_total_school_closure_days", int(df["closure_days_hurricane_related_broad"].sum())),
            ("strict_mean_closure_days_all_rows", df["closure_days_hurricane_related_strict"].mean()),
            ("broad_mean_closure_days_all_rows", df["closure_days_hurricane_related_broad"].mean()),
            ("strict_mean_closure_days_conditional", df.loc[df["closure_days_hurricane_related_strict"] > 0, "closure_days_hurricane_related_strict"].mean()),
            ("broad_mean_closure_days_conditional", df.loc[df["closure_days_hurricane_related_broad"] > 0, "closure_days_hurricane_related_broad"].mean()),
            ("strict_share_rows_with_closure", (df["closure_days_hurricane_related_strict"] > 0).mean()),
            ("broad_share_rows_with_closure", (df["closure_days_hurricane_related_broad"] > 0).mean()),
        ]
    )

    for col in OUTCOME_COLS + EXPOSURE_COLS:
        if col in df.columns:
            rows.append((f"mean_{col}", df[col].mean()))

    return pd.DataFrame(rows, columns=["statistic", "value"])


def make_variable_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize main outcome, exposure, and closure variables."""
    variables = [
        *[col for col in OUTCOME_COLS if col in df.columns],
        *[col for col in EXPOSURE_COLS if col in df.columns],
        *CLOSURE_COLS,
    ]

    rows = []

    for var in variables:
        if var not in df.columns:
            continue

        series = pd.to_numeric(df[var], errors="coerce").dropna()

        if series.empty:
            continue

        rows.append(
            {
                "variable": var,
                "count": int(series.count()),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p25": series.quantile(0.25),
                "median": series.median(),
                "p75": series.quantile(0.75),
                "max": series.max(),
            }
        )

    return pd.DataFrame(rows)


def make_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize closure and outcome variables by graduation year."""
    students_col = find_students_column(df)

    rows = []

    for year, group in df.groupby("graduation_year", dropna=False):
        row = {
            "graduation_year": int(year) if pd.notna(year) else None,
            "rows": len(group),
            "unique_schools": group["nces_school_id"].nunique() if "nces_school_id" in group.columns else None,
            "strict_rows_with_closure": int((group["closure_days_hurricane_related_strict"] > 0).sum()),
            "broad_rows_with_closure": int((group["closure_days_hurricane_related_broad"] > 0).sum()),
            "strict_total_closure_days": int(group["closure_days_hurricane_related_strict"].sum()),
            "broad_total_closure_days": int(group["closure_days_hurricane_related_broad"].sum()),
            "strict_max_closure_days": int(group["closure_days_hurricane_related_strict"].max()),
            "broad_max_closure_days": int(group["closure_days_hurricane_related_broad"].max()),
            "strict_share_rows_with_closure": (group["closure_days_hurricane_related_strict"] > 0).mean(),
            "broad_share_rows_with_closure": (group["closure_days_hurricane_related_broad"] > 0).mean(),
        }

        if students_col is not None:
            row["students"] = group[students_col].sum()

        for col in OUTCOME_COLS + EXPOSURE_COLS:
            if col in group.columns:
                row[f"mean_{col}"] = group[col].mean()
                row[f"student_weighted_{col}"] = weighted_mean(group, col, students_col)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("graduation_year")


def make_status_summary(df: pd.DataFrame, measure_type: str) -> pd.DataFrame:
    """Compare rows with and without closure under a selected measure."""
    students_col = find_students_column(df)
    any_col = f"closure_any_hurricane_related_{measure_type}"
    days_col = f"closure_days_hurricane_related_{measure_type}"

    rows = []

    for status, group in df.groupby(any_col, dropna=False):
        row = {
            "measure_type": measure_type,
            "closure_any": int(status) if pd.notna(status) else None,
            "rows": len(group),
            "unique_schools": group["nces_school_id"].nunique() if "nces_school_id" in group.columns else None,
            "mean_closure_days": group[days_col].mean(),
            "total_closure_days": group[days_col].sum(),
            "max_closure_days": group[days_col].max(),
        }

        if students_col is not None:
            row["students"] = group[students_col].sum()

        for col in OUTCOME_COLS + EXPOSURE_COLS:
            if col in group.columns:
                row[f"mean_{col}"] = group[col].mean()
                row[f"student_weighted_{col}"] = weighted_mean(group, col, students_col)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("closure_any")


def make_exposure_quartile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize closure measures by exposure quartile."""
    if "exposure_index_pointmax" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    valid = out["exposure_index_pointmax"].notna()

    out.loc[valid, "exposure_pointmax_quartile"] = pd.qcut(
        out.loc[valid, "exposure_index_pointmax"],
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )

    students_col = find_students_column(out)

    rows = []

    for quartile, group in out.groupby("exposure_pointmax_quartile", dropna=True):
        row = {
            "exposure_pointmax_quartile": str(quartile),
            "rows": len(group),
            "mean_exposure_pointmax": group["exposure_index_pointmax"].mean(),
            "strict_rows_with_closure": int((group["closure_days_hurricane_related_strict"] > 0).sum()),
            "broad_rows_with_closure": int((group["closure_days_hurricane_related_broad"] > 0).sum()),
            "strict_share_with_closure": (group["closure_days_hurricane_related_strict"] > 0).mean(),
            "broad_share_with_closure": (group["closure_days_hurricane_related_broad"] > 0).mean(),
            "strict_mean_closure_days": group["closure_days_hurricane_related_strict"].mean(),
            "broad_mean_closure_days": group["closure_days_hurricane_related_broad"].mean(),
            "strict_total_closure_days": group["closure_days_hurricane_related_strict"].sum(),
            "broad_total_closure_days": group["closure_days_hurricane_related_broad"].sum(),
        }

        if students_col is not None:
            row["students"] = group[students_col].sum()

        for col in OUTCOME_COLS:
            if col in group.columns:
                row[f"mean_{col}"] = group[col].mean()
                row[f"student_weighted_{col}"] = weighted_mean(group, col, students_col)

        rows.append(row)

    return pd.DataFrame(rows)


def split_semicolon_values(value: object) -> list[str]:
    """Split semicolon-separated metadata values."""
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [piece.strip() for piece in text.split(";") if piece.strip()]


def make_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize broad closure rows by event ID."""
    event_col = "closure_event_ids_broad"

    if event_col not in df.columns:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        events = split_semicolon_values(row.get(event_col, ""))

        for event in events:
            rows.append(
                {
                    "event_id": event,
                    "nces_school_id": row.get("nces_school_id", ""),
                    "graduation_year": row.get("graduation_year", None),
                    "closure_days_hurricane_related_broad": row.get("closure_days_hurricane_related_broad", 0),
                    "closure_max_consecutive_days_broad": row.get("closure_max_consecutive_days_broad", 0),
                    "closure_source_record_count_broad": row.get("closure_source_record_count_broad", 0),
                    "eligibility_rate": row.get("eligibility_rate", None),
                    "exposure_index_pointmax": row.get("exposure_index_pointmax", None),
                }
            )

    if not rows:
        return pd.DataFrame()

    exploded = pd.DataFrame(rows)

    for col in [
        "graduation_year",
        "closure_days_hurricane_related_broad",
        "closure_max_consecutive_days_broad",
        "closure_source_record_count_broad",
        "eligibility_rate",
        "exposure_index_pointmax",
    ]:
        exploded[col] = pd.to_numeric(exploded[col], errors="coerce")

    summary = (
        exploded.groupby("event_id", dropna=False)
        .agg(
            school_year_rows=("nces_school_id", "size"),
            unique_schools=("nces_school_id", "nunique"),
            graduation_years=("graduation_year", lambda x: "; ".join(str(int(v)) for v in sorted(x.dropna().unique()))),
            total_closure_days=("closure_days_hurricane_related_broad", "sum"),
            mean_closure_days=("closure_days_hurricane_related_broad", "mean"),
            max_closure_days=("closure_days_hurricane_related_broad", "max"),
            mean_eligibility_rate=("eligibility_rate", "mean"),
            mean_exposure_pointmax=("exposure_index_pointmax", "mean"),
            total_source_records=("closure_source_record_count_broad", "sum"),
        )
        .reset_index()
        .sort_values("total_closure_days", ascending=False)
    )

    return summary


def make_top_school_years(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Return school-year rows with the highest broad closure days."""
    cols = [
        "nces_school_id",
        "nces_school_name",
        "nces_district_name",
        "nces_parish",
        "graduation_year",
        "closure_days_hurricane_related_broad",
        "closure_max_consecutive_days_broad",
        "closure_source_record_count_broad",
        "closure_source_url_count_broad",
        "closure_event_ids_broad",
        "closure_storm_names_broad",
        "eligibility_rate",
        "recipient_rate",
        "acceptance_rate",
        "exposure_index_pointmax",
        "exposure_index_stormmax",
    ]

    existing_cols = [col for col in cols if col in df.columns]

    return (
        df[df["closure_days_hurricane_related_broad"] > 0][existing_cols]
        .sort_values(
            ["closure_days_hurricane_related_broad", "closure_source_record_count_broad"],
            ascending=[False, False],
        )
        .head(n)
        .copy()
    )


def make_consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Create consistency checks for closure variables."""
    rows = []

    rows.append(
        {
            "check": "panel_row_count",
            "value": len(df),
            "passes": len(df) > 0,
        }
    )

    if {"nces_school_id", "graduation_year"}.issubset(df.columns):
        duplicate_rows = df.duplicated(["nces_school_id", "graduation_year"]).sum()
        rows.append(
            {
                "check": "duplicate_nces_school_id_graduation_year_rows",
                "value": int(duplicate_rows),
                "passes": duplicate_rows == 0,
            }
        )

    strict_days = df["closure_days_hurricane_related_strict"]
    broad_days = df["closure_days_hurricane_related_broad"]

    rows.append(
        {
            "check": "broad_days_greater_or_equal_strict_days_all_rows",
            "value": int((broad_days >= strict_days).sum()),
            "passes": bool((broad_days >= strict_days).all()),
        }
    )

    rows.append(
        {
            "check": "strict_any_matches_positive_days",
            "value": int(((df["closure_any_hurricane_related_strict"] == 1) == (strict_days > 0)).sum()),
            "passes": bool(((df["closure_any_hurricane_related_strict"] == 1) == (strict_days > 0)).all()),
        }
    )

    rows.append(
        {
            "check": "broad_any_matches_positive_days",
            "value": int(((df["closure_any_hurricane_related_broad"] == 1) == (broad_days > 0)).sum()),
            "passes": bool(((df["closure_any_hurricane_related_broad"] == 1) == (broad_days > 0)).all()),
        }
    )

    rows.append(
        {
            "check": "negative_broad_closure_days",
            "value": int((broad_days < 0).sum()),
            "passes": bool((broad_days >= 0).all()),
        }
    )

    rows.append(
        {
            "check": "negative_strict_closure_days",
            "value": int((strict_days < 0).sum()),
            "passes": bool((strict_days >= 0).all()),
        }
    )

    return pd.DataFrame(rows)


def write_report(
    sample_summary: pd.DataFrame,
    variable_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    broad_status: pd.DataFrame,
    strict_status: pd.DataFrame,
    exposure_quartile: pd.DataFrame,
    event_summary: pd.DataFrame,
    top_school_years: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    """Write a text report with main descriptive outputs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sections = []

    sections.append("06a Closure Descriptive Statistics")
    sections.append("=" * 80)
    sections.append("")

    sections.append("Sample Summary")
    sections.append("-" * 80)
    sections.append(sample_summary.to_string(index=False))
    sections.append("")

    sections.append("Variable Summary")
    sections.append("-" * 80)
    sections.append(variable_summary.to_string(index=False))
    sections.append("")

    sections.append("Closure by Graduation Year")
    sections.append("-" * 80)
    sections.append(year_summary.to_string(index=False))
    sections.append("")

    sections.append("Closure Status Summary: Broad")
    sections.append("-" * 80)
    sections.append(broad_status.to_string(index=False))
    sections.append("")

    sections.append("Closure Status Summary: Strict")
    sections.append("-" * 80)
    sections.append(strict_status.to_string(index=False))
    sections.append("")

    sections.append("Closure by Exposure Quartile")
    sections.append("-" * 80)
    sections.append(exposure_quartile.to_string(index=False) if not exposure_quartile.empty else "No exposure quartile summary.")
    sections.append("")

    sections.append("Broad Closure Event Summary")
    sections.append("-" * 80)
    sections.append(event_summary.to_string(index=False) if not event_summary.empty else "No event summary.")
    sections.append("")

    sections.append("Top Broad Closure School-Year Rows")
    sections.append("-" * 80)
    sections.append(top_school_years.to_string(index=False) if not top_school_years.empty else "No closure rows.")
    sections.append("")

    sections.append("Consistency Checks")
    sections.append("-" * 80)
    sections.append(checks.to_string(index=False))
    sections.append("")

    REPORT_OUT.write_text("\n".join(sections))


def main() -> None:
    """Run 06a closure descriptive statistics."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_panel(INPUT_PATH)

    sample_summary = make_sample_summary(df)
    variable_summary = make_variable_summary(df)
    year_summary = make_year_summary(df)
    broad_status = make_status_summary(df, "broad")
    strict_status = make_status_summary(df, "strict")
    exposure_quartile = make_exposure_quartile_summary(df)
    event_summary = make_event_summary(df)
    top_school_years = make_top_school_years(df)
    checks = make_consistency_checks(df)

    sample_summary.to_csv(SAMPLE_SUMMARY_OUT, index=False)
    variable_summary.to_csv(VARIABLE_SUMMARY_OUT, index=False)
    year_summary.to_csv(YEAR_SUMMARY_OUT, index=False)
    broad_status.to_csv(BROAD_STATUS_OUT, index=False)
    strict_status.to_csv(STRICT_STATUS_OUT, index=False)
    exposure_quartile.to_csv(EXPOSURE_QUARTILE_OUT, index=False)
    event_summary.to_csv(EVENT_SUMMARY_OUT, index=False)
    top_school_years.to_csv(TOP_SCHOOL_YEARS_OUT, index=False)
    checks.to_csv(CONSISTENCY_CHECKS_OUT, index=False)

    write_report(
        sample_summary=sample_summary,
        variable_summary=variable_summary,
        year_summary=year_summary,
        broad_status=broad_status,
        strict_status=strict_status,
        exposure_quartile=exposure_quartile,
        event_summary=event_summary,
        top_school_years=top_school_years,
        checks=checks,
    )

    print("Saved:")
    print(" -", SAMPLE_SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", VARIABLE_SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", YEAR_SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", BROAD_STATUS_OUT.relative_to(PROJECT_ROOT))
    print(" -", STRICT_STATUS_OUT.relative_to(PROJECT_ROOT))
    print(" -", EXPOSURE_QUARTILE_OUT.relative_to(PROJECT_ROOT))
    print(" -", EVENT_SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", TOP_SCHOOL_YEARS_OUT.relative_to(PROJECT_ROOT))
    print(" -", CONSISTENCY_CHECKS_OUT.relative_to(PROJECT_ROOT))
    print(" -", REPORT_OUT.relative_to(PROJECT_ROOT))

    print()
    print("Sample Summary:")
    print(sample_summary.to_string(index=False))

    print()
    print("Closure by Graduation Year:")
    print(
        year_summary[
            [
                "graduation_year",
                "rows",
                "broad_rows_with_closure",
                "broad_total_closure_days",
                "broad_max_closure_days",
                "mean_eligibility_rate",
                "mean_exposure_index_pointmax",
            ]
        ].to_string(index=False)
    )

    print()
    print("Closure by Exposure Quartile:")
    print(exposure_quartile.to_string(index=False) if not exposure_quartile.empty else "No exposure quartile summary.")

    print()
    print("Consistency Checks:")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
