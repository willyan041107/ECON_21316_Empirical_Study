"""
Expand CDC Louisiana closure records to school-date and school-year panels.

Inputs:
    data/intermediate/cdc_louisiana_closure_records_clean.csv
    data/processed/regression_panel.csv

Outputs:
    data/intermediate/cdc_closure_school_date_panel.csv
    data/intermediate/cdc_closure_school_year_panel.csv
    data/intermediate/cdc_closure_expansion_review.csv
    data/intermediate/cdc_closure_school_year_summary.csv
    data/processed/regression_panel_with_cdc_closures.csv
    data/processed/cdc_closure_merge_summary.txt

Main idea:
    CDC records may be district-level or school-specific.
    District-level closures are expanded to all matched public high schools
    in the regression-panel school universe.

Main treatment variables:
    cdc_closure_days_hurricane_related
    cdc_closure_any_hurricane_related
    cdc_closure_event_count_hurricane_related
    cdc_max_consecutive_closure_days_hurricane_related
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CDC_CLEAN_PATH = INTERMEDIATE_DIR / "cdc_louisiana_closure_records_clean.csv"
REGRESSION_PANEL_PATH = PROCESSED_DIR / "regression_panel.csv"

SCHOOL_DATE_OUT = INTERMEDIATE_DIR / "cdc_closure_school_date_panel.csv"
SCHOOL_YEAR_OUT = INTERMEDIATE_DIR / "cdc_closure_school_year_panel.csv"
EXPANSION_REVIEW_OUT = INTERMEDIATE_DIR / "cdc_closure_expansion_review.csv"
SUMMARY_BY_YEAR_OUT = INTERMEDIATE_DIR / "cdc_closure_school_year_summary.csv"
MERGED_PANEL_OUT = PROCESSED_DIR / "regression_panel_with_cdc_closures.csv"
SUMMARY_TEXT_OUT = PROCESSED_DIR / "cdc_closure_merge_summary.txt"


def normalize_text(value: object) -> str:
    """Normalize names for deterministic matching."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = text.replace("&", " AND ")
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = re.sub(r"\bSAINT\b", "ST", text)
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_parish(value: object) -> str:
    """Normalize Louisiana parish names."""
    text = normalize_text(value)

    if not text:
        return ""

    text = text.replace(" PARISH", "").strip()

    aliases = {
        "ST JOHN": "ST JOHN THE BAPTIST",
        "ST JOHN BAPTIST": "ST JOHN THE BAPTIST",
        "ST JOHN THE BAPTIST": "ST JOHN THE BAPTIST",
        "ST BERNARD": "ST BERNARD",
        "ST CHARLES": "ST CHARLES",
        "ST TAMMANY": "ST TAMMANY",
        "LA SALLE": "LA SALLE",
        "LASALLE": "LA SALLE",
    }

    return aliases.get(text, text)


def normalize_id(value: object, width: int = 12) -> str:
    """Normalize numeric IDs as strings."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if text.lower() in {"nan", "none", ""}:
        return ""

    if text.isdigit():
        return text.zfill(width)

    return text


def parse_bool(value: object) -> bool:
    """Parse string-like booleans."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    """Parse date safely."""
    if pd.isna(value):
        return pd.NaT

    return pd.to_datetime(value, errors="coerce").normalize()


def graduation_year_from_date(date: pd.Timestamp) -> int | None:
    """
    Map closure date to graduation year.

    Academic year:
        Aug 1 of t-1 through May 31 of t -> graduation_year t.
    """
    if pd.isna(date):
        return None

    if int(date.month) >= 8:
        return int(date.year) + 1

    return int(date.year)


def closure_weekdays(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    """
    Generate weekday closure dates from start_date through day before reopened/end_date.

    CDC datereopened is interpreted as the date schools reopened, so closure dates
    run from closure_start_date through closure_end_date - 1 day.
    """
    if pd.isna(start_date) or pd.isna(end_date) or end_date <= start_date:
        return []

    dates = pd.date_range(
        start=start_date,
        end=end_date - pd.Timedelta(days=1),
        freq="D",
    )

    return [date.normalize() for date in dates if date.weekday() < 5]


def build_school_universe(regression: pd.DataFrame) -> pd.DataFrame:
    """Build one row per NCES school in the regression panel."""
    school_cols = [
        "nces_school_id",
        "nces_school_name",
        "nces_district_name",
        "nces_parish",
        "latitude",
        "longitude",
    ]

    existing_cols = [col for col in school_cols if col in regression.columns]

    schools = (
        regression[existing_cols]
        .drop_duplicates("nces_school_id")
        .copy()
    )

    schools["nces_school_id"] = schools["nces_school_id"].apply(lambda x: normalize_id(x, 12))
    schools["district_norm"] = schools["nces_district_name"].apply(normalize_text)
    schools["parish_norm"] = schools["nces_parish"].apply(normalize_parish)

    return schools


def match_district_record_to_schools(record: pd.Series, schools: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Match one district-level CDC record to public high schools.

    Matching priority:
        1. exact normalized district name
        2. parish fallback only for ordinary parish school systems
    """
    district_norm = normalize_text(record.get("district_name", ""))
    parish_norm = normalize_parish(record.get("parish", ""))

    exact = schools[schools["district_norm"].eq(district_norm)].copy()

    if not exact.empty:
        return exact, "exact_district_name"

    # Fallback only when the CDC record appears to describe a parish public school system.
    # Avoid expanding special charter operators such as KIPP or RSD to all Orleans schools.
    ordinary_parish_system = (
        district_norm.endswith("PARISH")
        or " PARISH " in district_norm
        or district_norm in {
            "ORLEANS PARISH",
            "JEFFERSON PARISH",
            "LAFOURCHE PARISH",
            "TERREBONNE PARISH",
            "ST BERNARD PARISH",
            "ST CHARLES PARISH",
            "ST JOHN THE BAPTIST PARISH",
            "PLAQUEMINES PARISH",
            "UNION PARISH",
            "WEBSTER PARISH",
        }
    )

    if ordinary_parish_system and parish_norm:
        parish_match = schools[schools["parish_norm"].eq(parish_norm)].copy()

        if not parish_match.empty:
            return parish_match, "parish_fallback_public_system"

    return pd.DataFrame(columns=schools.columns), "no_school_match"


def match_school_record_to_schools(record: pd.Series, schools: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Match one school-specific CDC record to public high schools."""
    nces_school_id = normalize_id(record.get("nces_school_id", ""), 12)

    if nces_school_id:
        matched = schools[schools["nces_school_id"].eq(nces_school_id)].copy()

        if not matched.empty:
            return matched, "exact_nces_school_id"

    school_name_norm = normalize_text(record.get("school_name", ""))
    parish_norm = normalize_parish(record.get("parish", ""))

    candidates = schools.copy()

    if parish_norm:
        candidates = candidates[candidates["parish_norm"].eq(parish_norm)].copy()

    candidates = candidates[
        candidates["nces_school_name"].apply(normalize_text).eq(school_name_norm)
    ].copy()

    if not candidates.empty:
        return candidates, "exact_school_name_parish"

    return pd.DataFrame(columns=schools.columns), "no_school_match"


def expand_records_to_school_dates(clean: pd.DataFrame, schools: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand CDC closure records to school-date rows."""
    expanded_rows = []
    review_rows = []

    main_records = clean[clean["include_in_main_closure_measure"].apply(parse_bool)].copy()

    for _, record in main_records.iterrows():
        closure_scope = str(record.get("closure_scope", "")).strip()
        start_date = parse_date(record.get("closure_start_date"))
        end_date = parse_date(record.get("closure_end_date"))
        dates = closure_weekdays(start_date, end_date)

        if closure_scope == "district_system":
            matched_schools, match_method = match_district_record_to_schools(record, schools)
        elif closure_scope == "school_specific":
            matched_schools, match_method = match_school_record_to_schools(record, schools)
        else:
            matched_schools = pd.DataFrame(columns=schools.columns)
            match_method = "unsupported_scope"

        if matched_schools.empty or not dates:
            review_rows.append(
                {
                    "closure_record_id": record.get("closure_record_id", ""),
                    "raw_record_id": record.get("raw_record_id", ""),
                    "eventname": record.get("eventname", ""),
                    "closure_scope": closure_scope,
                    "district_name": record.get("district_name", ""),
                    "school_name": record.get("school_name", ""),
                    "parish": record.get("parish", ""),
                    "closure_start_date": record.get("closure_start_date", ""),
                    "closure_end_date": record.get("closure_end_date", ""),
                    "match_method": match_method,
                    "issue": "no_matched_schools_or_no_valid_dates",
                    "notes": "",
                }
            )
            continue

        for _, school in matched_schools.iterrows():
            for date in dates:
                expanded_rows.append(
                    {
                        "closure_record_id": record.get("closure_record_id", ""),
                        "raw_record_id": record.get("raw_record_id", ""),
                        "source_id": record.get("source_id", ""),
                        "eventname": record.get("eventname", ""),
                        "storm_name": record.get("storm_name", ""),
                        "closure_reason_category": record.get("closure_reason_category", ""),
                        "closure_scope": closure_scope,
                        "closure_match_method": match_method,
                        "closure_date": date.date().isoformat(),
                        "graduation_year": graduation_year_from_date(date),
                        "nces_school_id": school["nces_school_id"],
                        "nces_school_name": school.get("nces_school_name", ""),
                        "nces_district_name": school.get("nces_district_name", ""),
                        "nces_parish": school.get("nces_parish", ""),
                        "latitude": school.get("latitude", ""),
                        "longitude": school.get("longitude", ""),
                        "hurricane_related": parse_bool(record.get("hurricane_related", False)),
                        "named_storm_closure": bool(str(record.get("storm_name", "")).strip()),
                        "instructional_closure": parse_bool(record.get("instructional_closure", False)),
                    }
                )

    school_dates = pd.DataFrame(expanded_rows)
    review = pd.DataFrame(review_rows)

    if school_dates.empty:
        return school_dates, review

    # Deduplicate overlapping closure announcements for the same school-date.
    grouped = (
        school_dates
        .groupby(["nces_school_id", "closure_date"], dropna=False)
        .agg(
            graduation_year=("graduation_year", "first"),
            nces_school_name=("nces_school_name", "first"),
            nces_district_name=("nces_district_name", "first"),
            nces_parish=("nces_parish", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            closure_record_ids=("closure_record_id", lambda x: "; ".join(sorted(set(x.astype(str))))),
            eventnames=("eventname", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            storm_names=("storm_name", lambda x: "; ".join(sorted(set(v for v in x.dropna().astype(str) if v.strip())))),
            closure_reason_categories=("closure_reason_category", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            closure_scopes=("closure_scope", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            closure_match_methods=("closure_match_method", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            hurricane_related=("hurricane_related", "max"),
            named_storm_closure=("named_storm_closure", "max"),
            instructional_closure=("instructional_closure", "max"),
        )
        .reset_index()
    )

    return grouped, review


def max_consecutive_schooldays(dates: pd.Series) -> int:
    """
    Compute maximum run of consecutive schooldays.

    Friday to Monday counts as consecutive schooldays.
    """
    parsed = pd.to_datetime(dates, errors="coerce").dropna().sort_values().drop_duplicates()

    if parsed.empty:
        return 0

    max_run = 1
    current_run = 1

    previous = parsed.iloc[0]

    for current in parsed.iloc[1:]:
        expected_next_schoolday = previous + pd.offsets.BDay(1)

        if current.normalize() == expected_next_schoolday.normalize():
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1

        previous = current

    return max(max_run, current_run)


def aggregate_school_dates_to_year(school_dates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate school-date closure rows to NCES school × graduation year."""
    if school_dates.empty:
        return pd.DataFrame()

    school_dates = school_dates.copy()
    school_dates["graduation_year"] = pd.to_numeric(school_dates["graduation_year"], errors="coerce")
    school_dates = school_dates[school_dates["graduation_year"].notna()].copy()
    school_dates["graduation_year"] = school_dates["graduation_year"].astype(int)

    rows = []

    for (school_id, grad_year), group in school_dates.groupby(["nces_school_id", "graduation_year"]):
        group = group.copy()

        named = group[group["named_storm_closure"]].copy()

        rows.append(
            {
                "nces_school_id": school_id,
                "graduation_year": int(grad_year),
                "cdc_closure_days_hurricane_related": len(group),
                "cdc_closure_any_hurricane_related": int(len(group) > 0),
                "cdc_closure_event_count_hurricane_related": len(
                    set(
                        "; ".join(group["closure_record_ids"].astype(str))
                        .split("; ")
                    )
                ),
                "cdc_max_consecutive_closure_days_hurricane_related": max_consecutive_schooldays(group["closure_date"]),
                "cdc_closure_days_named_storm": len(named),
                "cdc_closure_any_named_storm": int(len(named) > 0),
                "cdc_storm_names": "; ".join(sorted(set(
                    name
                    for piece in group["storm_names"].dropna().astype(str)
                    for name in piece.split("; ")
                    if name.strip()
                ))),
                "cdc_eventnames": "; ".join(sorted(set(
                    name
                    for piece in group["eventnames"].dropna().astype(str)
                    for name in piece.split("; ")
                    if name.strip()
                ))),
                "cdc_closure_record_ids": "; ".join(sorted(set(
                    rid
                    for piece in group["closure_record_ids"].dropna().astype(str)
                    for rid in piece.split("; ")
                    if rid.strip()
                ))),
            }
        )

    return pd.DataFrame(rows)


def merge_to_regression_panel(regression: pd.DataFrame, closure_year: pd.DataFrame) -> pd.DataFrame:
    """Merge school-year closure measures into regression panel."""
    panel = regression.copy()

    panel["nces_school_id"] = panel["nces_school_id"].apply(lambda x: normalize_id(x, 12))
    panel["graduation_year"] = pd.to_numeric(panel["graduation_year"], errors="coerce").astype(int)

    if closure_year.empty:
        merged = panel.copy()
    else:
        closure_year = closure_year.copy()
        closure_year["nces_school_id"] = closure_year["nces_school_id"].apply(lambda x: normalize_id(x, 12))
        closure_year["graduation_year"] = pd.to_numeric(closure_year["graduation_year"], errors="coerce").astype(int)

        merged = panel.merge(
            closure_year,
            on=["nces_school_id", "graduation_year"],
            how="left",
            validate="one_to_one",
        )

    numeric_closure_cols = [
        "cdc_closure_days_hurricane_related",
        "cdc_closure_any_hurricane_related",
        "cdc_closure_event_count_hurricane_related",
        "cdc_max_consecutive_closure_days_hurricane_related",
        "cdc_closure_days_named_storm",
        "cdc_closure_any_named_storm",
    ]

    for col in numeric_closure_cols:
        if col not in merged.columns:
            merged[col] = 0

        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    for col in ["cdc_storm_names", "cdc_eventnames", "cdc_closure_record_ids"]:
        if col not in merged.columns:
            merged[col] = ""

        merged[col] = merged[col].fillna("")

    return merged


def make_summary_by_year(merged: pd.DataFrame) -> pd.DataFrame:
    """Summarize CDC closure measures by graduation year in regression panel."""
    return (
        merged.groupby("graduation_year")
        .agg(
            rows=("nces_school_id", "size"),
            students=("students_processed", lambda x: pd.to_numeric(x, errors="coerce").sum()),
            schools_with_cdc_closure=("cdc_closure_any_hurricane_related", "sum"),
            total_cdc_closure_days=("cdc_closure_days_hurricane_related", "sum"),
            mean_cdc_closure_days=("cdc_closure_days_hurricane_related", "mean"),
            max_cdc_closure_days=("cdc_closure_days_hurricane_related", "max"),
            schools_with_named_storm_closure=("cdc_closure_any_named_storm", "sum"),
            total_named_storm_closure_days=("cdc_closure_days_named_storm", "sum"),
        )
        .reset_index()
        .sort_values("graduation_year")
    )


def write_summary_text(
    clean: pd.DataFrame,
    school_dates: pd.DataFrame,
    school_year: pd.DataFrame,
    review: pd.DataFrame,
    merged: pd.DataFrame,
    summary_year: pd.DataFrame,
) -> None:
    """Write a plain-text summary."""
    lines = []

    lines.append("CDC Closure Expansion and Merge Summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"CDC clean records: {len(clean):,}")
    lines.append(f"CDC records included in main measure: {int(clean['include_in_main_closure_measure'].apply(parse_bool).sum()):,}")
    lines.append(f"Expanded school-date rows: {len(school_dates):,}")
    lines.append(f"School-year closure rows: {len(school_year):,}")
    lines.append(f"Expansion review rows: {len(review):,}")
    lines.append(f"Merged regression panel rows: {len(merged):,}")
    lines.append("")
    lines.append("Closure measures in merged regression panel:")
    lines.append(str(merged[[
        "cdc_closure_days_hurricane_related",
        "cdc_closure_any_hurricane_related",
        "cdc_closure_event_count_hurricane_related",
        "cdc_max_consecutive_closure_days_hurricane_related",
        "cdc_closure_days_named_storm",
    ]].describe()))
    lines.append("")
    lines.append("Summary by graduation year:")
    lines.append(summary_year.to_string(index=False))
    lines.append("")
    lines.append("Expansion review rows:")
    if review.empty:
        lines.append("None")
    else:
        lines.append(review.to_string(index=False))

    SUMMARY_TEXT_OUT.write_text("\n".join(lines))


def main() -> None:
    """Run CDC closure expansion and merge."""
    if not CDC_CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing CDC clean file: {CDC_CLEAN_PATH}")

    if not REGRESSION_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing regression panel: {REGRESSION_PANEL_PATH}")

    clean = pd.read_csv(CDC_CLEAN_PATH, dtype=str, low_memory=False)
    regression = pd.read_csv(REGRESSION_PANEL_PATH, dtype=str, low_memory=False)

    schools = build_school_universe(regression)

    school_dates, review = expand_records_to_school_dates(clean, schools)
    school_year = aggregate_school_dates_to_year(school_dates)

    merged = merge_to_regression_panel(regression, school_year)
    summary_year = make_summary_by_year(merged)

    school_dates.to_csv(SCHOOL_DATE_OUT, index=False)
    school_year.to_csv(SCHOOL_YEAR_OUT, index=False)
    review.to_csv(EXPANSION_REVIEW_OUT, index=False)
    summary_year.to_csv(SUMMARY_BY_YEAR_OUT, index=False)
    merged.to_csv(MERGED_PANEL_OUT, index=False)

    write_summary_text(
        clean=clean,
        school_dates=school_dates,
        school_year=school_year,
        review=review,
        merged=merged,
        summary_year=summary_year,
    )

    print("Saved:")
    print(" -", SCHOOL_DATE_OUT.relative_to(PROJECT_ROOT))
    print(" -", SCHOOL_YEAR_OUT.relative_to(PROJECT_ROOT))
    print(" -", EXPANSION_REVIEW_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_BY_YEAR_OUT.relative_to(PROJECT_ROOT))
    print(" -", MERGED_PANEL_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_TEXT_OUT.relative_to(PROJECT_ROOT))

    print()
    print("CDC clean records:", len(clean))
    print("Included CDC records:", int(clean["include_in_main_closure_measure"].apply(parse_bool).sum()))
    print("Expanded school-date rows:", len(school_dates))
    print("School-year closure rows:", len(school_year))
    print("Expansion review rows:", len(review))
    print("Merged regression panel rows:", len(merged))

    print()
    print("Closure days summary in merged panel:")
    print(
        merged[
            [
                "cdc_closure_days_hurricane_related",
                "cdc_closure_any_hurricane_related",
                "cdc_closure_event_count_hurricane_related",
                "cdc_max_consecutive_closure_days_hurricane_related",
                "cdc_closure_days_named_storm",
            ]
        ]
        .describe()
        .to_string()
    )

    print()
    print("Summary by graduation year:")
    print(summary_year.to_string(index=False))

    print()
    print("Expansion review rows:")
    if review.empty:
        print("None")
    else:
        print(review.to_string(index=False))


if __name__ == "__main__":
    main()
