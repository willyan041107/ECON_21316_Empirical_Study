"""
Expand validated closure records to school-date and school-year panels.

Inputs:
    data/intermediate/closure_auto_records_validated_strict.csv
    data/intermediate/closure_auto_records_validated_broad.csv
    data/processed/regression_panel.csv

Outputs:
    data/intermediate/closure_auto_school_date_panel_strict.csv
    data/intermediate/closure_auto_school_date_panel_broad.csv
    data/intermediate/closure_auto_school_year_panel_strict.csv
    data/intermediate/closure_auto_school_year_panel_broad.csv
    data/intermediate/closure_auto_expansion_review.csv
    data/intermediate/closure_auto_expansion_summary.csv
    data/processed/regression_panel_with_auto_closures.csv

Purpose:
    Convert source-level closure records into school-year closure variables.

Important:
    This script deduplicates at the nces_school_id x closure_date level before
    aggregating to school-year. This prevents duplicate news/source reports from
    mechanically inflating closure-day counts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STRICT_RECORDS_PATH = INTERMEDIATE_DIR / "closure_auto_records_validated_strict.csv"
BROAD_RECORDS_PATH = INTERMEDIATE_DIR / "closure_auto_records_validated_broad.csv"
REGRESSION_PANEL_PATH = PROCESSED_DIR / "regression_panel.csv"

STRICT_SCHOOL_DATE_OUT = INTERMEDIATE_DIR / "closure_auto_school_date_panel_strict.csv"
BROAD_SCHOOL_DATE_OUT = INTERMEDIATE_DIR / "closure_auto_school_date_panel_broad.csv"

STRICT_SCHOOL_YEAR_OUT = INTERMEDIATE_DIR / "closure_auto_school_year_panel_strict.csv"
BROAD_SCHOOL_YEAR_OUT = INTERMEDIATE_DIR / "closure_auto_school_year_panel_broad.csv"

EXPANSION_REVIEW_OUT = INTERMEDIATE_DIR / "closure_auto_expansion_review.csv"
EXPANSION_SUMMARY_OUT = INTERMEDIATE_DIR / "closure_auto_expansion_summary.csv"

MERGED_PANEL_OUT = PROCESSED_DIR / "regression_panel_with_auto_closures.csv"


VALID_SCOPES = {
    "district_system",
    "parishwide",
    "statewide",
    "school_specific",
}


def normalize_text(value: object) -> str:
    """Normalize generic text."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_upper(value: object) -> str:
    """Normalize text to uppercase alphanumeric form."""
    text = normalize_text(value).upper()
    text = text.replace("&", " AND ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = re.sub(r"[^A-Z0-9\s.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_parish(value: object) -> str:
    """Normalize Louisiana parish names."""
    text = normalize_upper(value)
    text = text.replace(" PARISH", "").strip()

    aliases = {
        "ST JOHN": "ST. JOHN THE BAPTIST",
        "ST. JOHN": "ST. JOHN THE BAPTIST",
        "ST JOHN THE BAPTIST": "ST. JOHN THE BAPTIST",
        "ST. JOHN THE BAPTIST": "ST. JOHN THE BAPTIST",
        "ST TAMMANY": "ST. TAMMANY",
        "ST. TAMMANY": "ST. TAMMANY",
        "ST CHARLES": "ST. CHARLES",
        "ST. CHARLES": "ST. CHARLES",
        "ST JAMES": "ST. JAMES",
        "ST. JAMES": "ST. JAMES",
        "ST BERNARD": "ST. BERNARD",
        "ST. BERNARD": "ST. BERNARD",
        "ST MARY": "ST. MARY",
        "ST. MARY": "ST. MARY",
        "EAST BATON ROUGE": "EAST BATON ROUGE",
        "WEST BATON ROUGE": "WEST BATON ROUGE",
        "POINTE COUPEE": "POINTE COUPEE",
        "LA SALLE": "LA SALLE",
        "LASALLE": "LA SALLE",
    }

    return aliases.get(text, text)


def normalize_storm(value: object) -> str:
    """Normalize storm names."""
    text = normalize_upper(value)
    text = text.replace("HURRICANE", "")
    text = text.replace("TROPICAL STORM", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_school_name(value: object) -> str:
    """Normalize school name for exact matching."""
    text = normalize_upper(value)

    replacements = {
        " HIGH SCHOOL": " HIGH",
        " SENIOR HIGH SCHOOL": " HIGH",
        " JR SR HIGH SCHOOL": " HIGH",
        " JUNIOR SENIOR HIGH SCHOOL": " HIGH",
        " SCHOOL": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_id(value: object, width: int = 12) -> str:
    """Normalize numeric IDs to strings."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    if text.isdigit():
        return text.zfill(width)

    return text


def parse_bool(value: object) -> bool:
    """Parse boolean-like values."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    """Parse date safely."""
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null", "nat"}:
        return pd.NaT

    parsed = pd.to_datetime(text, errors="coerce")

    if pd.isna(parsed):
        return pd.NaT

    return parsed.normalize()


def infer_graduation_year(date: pd.Timestamp) -> int | None:
    """Map closure date to graduation year."""
    if pd.isna(date):
        return None

    if int(date.month) >= 8:
        return int(date.year) + 1

    return int(date.year)


def date_range_schooldays(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    """Return weekday closure dates from start_date through end_date inclusive."""
    if pd.isna(start_date) or pd.isna(end_date) or end_date < start_date:
        return []

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    return [date.normalize() for date in dates if date.weekday() < 5]


def max_consecutive_schooldays(dates: pd.Series) -> int:
    """Compute the maximum run of consecutive schooldays."""
    parsed = pd.to_datetime(dates, errors="coerce").dropna().sort_values().drop_duplicates()

    if parsed.empty:
        return 0

    max_run = 1
    current_run = 1
    previous = parsed.iloc[0]

    for current in parsed.iloc[1:]:
        expected_next = previous + pd.offsets.BDay(1)

        if current.normalize() == expected_next.normalize():
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1

        previous = current

    return max(max_run, current_run)


def build_school_universe(regression: pd.DataFrame) -> pd.DataFrame:
    """Build one row per NCES school in regression panel."""
    required_cols = [
        "nces_school_id",
        "nces_school_name",
        "nces_district_name",
        "nces_parish",
        "latitude",
        "longitude",
    ]

    existing_cols = [col for col in required_cols if col in regression.columns]

    schools = (
        regression[existing_cols]
        .drop_duplicates("nces_school_id")
        .copy()
    )

    schools["nces_school_id"] = schools["nces_school_id"].apply(lambda x: normalize_id(x, 12))
    schools["school_norm"] = schools["nces_school_name"].apply(normalize_school_name)
    schools["district_norm"] = schools["nces_district_name"].apply(normalize_upper)
    schools["parish_norm"] = schools["nces_parish"].apply(normalize_parish)

    return schools


def standardize_records(records: pd.DataFrame, measure_type: str) -> pd.DataFrame:
    """Standardize strict or broad closure records before expansion."""
    out = records.copy()

    out["measure_type"] = measure_type

    for col in [
        "auto_closure_record_id",
        "target_event_id",
        "parish",
        "target_parish",
        "district_name",
        "school_name",
        "storm_name",
        "closure_scope",
        "closure_start_date",
        "closure_end_date",
        "url",
        "local_file_path",
        "evidence_text",
    ]:
        if col not in out.columns:
            out[col] = ""

        out[col] = out[col].fillna("").astype(str).map(normalize_text)

    out["parish_norm"] = out["parish"].apply(normalize_parish)

    missing_parish_mask = out["parish_norm"].eq("")
    out.loc[missing_parish_mask, "parish_norm"] = out.loc[
        missing_parish_mask, "target_parish"
    ].apply(normalize_parish)

    out["district_norm"] = out["district_name"].apply(normalize_upper)
    out["school_norm"] = out["school_name"].apply(normalize_school_name)
    out["storm_norm"] = out["storm_name"].apply(normalize_storm)
    out["scope_norm"] = out["closure_scope"].str.lower().str.strip()

    out["closure_start_parsed"] = out["closure_start_date"].apply(parse_date)
    out["closure_end_parsed"] = out["closure_end_date"].apply(parse_date)

    out["source_confidence"] = pd.to_numeric(out.get("confidence", 1), errors="coerce").fillna(1)

    return out


def match_parish_or_district_record(record: pd.Series, schools: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Match parishwide/district-level record to schools."""
    parish_norm = record.get("parish_norm", "")
    district_norm = record.get("district_norm", "")

    if district_norm:
        exact_district = schools[schools["district_norm"].eq(district_norm)].copy()

        if not exact_district.empty:
            return exact_district, "exact_district"

    if parish_norm:
        parish_match = schools[schools["parish_norm"].eq(parish_norm)].copy()

        if not parish_match.empty:
            return parish_match, "parish_fallback"

    return pd.DataFrame(columns=schools.columns), "no_match"


def match_school_specific_record(record: pd.Series, schools: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Match a school-specific record to schools."""
    school_norm = record.get("school_norm", "")
    parish_norm = record.get("parish_norm", "")
    district_norm = record.get("district_norm", "")

    if not school_norm:
        return pd.DataFrame(columns=schools.columns), "missing_school_name"

    candidates = schools.copy()

    if parish_norm:
        candidates = candidates[candidates["parish_norm"].eq(parish_norm)].copy()

    if district_norm and not candidates.empty:
        district_candidates = candidates[candidates["district_norm"].eq(district_norm)].copy()

        if not district_candidates.empty:
            candidates = district_candidates

    matched = candidates[candidates["school_norm"].eq(school_norm)].copy()

    if not matched.empty:
        return matched, "exact_school_name"

    # Fallback: allow containment match for school-specific records.
    contains = candidates[
        candidates["school_norm"].apply(lambda x: bool(x) and (x in school_norm or school_norm in x))
    ].copy()

    if not contains.empty:
        return contains, "school_name_contains"

    return pd.DataFrame(columns=schools.columns), "no_match"


def expand_records_to_school_dates(records: pd.DataFrame, schools: pd.DataFrame, measure_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand source-level closure records to school-date rows."""
    expanded_rows = []
    review_rows = []

    records = standardize_records(records, measure_type)

    for _, record in records.iterrows():
        record_id = record.get("auto_closure_record_id", "")
        scope = record.get("scope_norm", "")

        start_date = record.get("closure_start_parsed", pd.NaT)
        end_date = record.get("closure_end_parsed", pd.NaT)
        dates = date_range_schooldays(start_date, end_date)

        if not dates:
            review_rows.append(
                {
                    "measure_type": measure_type,
                    "auto_closure_record_id": record_id,
                    "target_event_id": record.get("target_event_id", ""),
                    "parish": record.get("parish", ""),
                    "target_parish": record.get("target_parish", ""),
                    "district_name": record.get("district_name", ""),
                    "school_name": record.get("school_name", ""),
                    "storm_name": record.get("storm_name", ""),
                    "closure_scope": record.get("closure_scope", ""),
                    "closure_start_date": record.get("closure_start_date", ""),
                    "closure_end_date": record.get("closure_end_date", ""),
                    "issue": "no_valid_schoolday_dates",
                    "match_method": "",
                    "url": record.get("url", ""),
                }
            )
            continue

        if scope in {"district_system", "parishwide", "statewide"}:
            matched_schools, match_method = match_parish_or_district_record(record, schools)
        elif scope == "school_specific":
            matched_schools, match_method = match_school_specific_record(record, schools)
        else:
            matched_schools, match_method = match_parish_or_district_record(record, schools)

        if matched_schools.empty:
            review_rows.append(
                {
                    "measure_type": measure_type,
                    "auto_closure_record_id": record_id,
                    "target_event_id": record.get("target_event_id", ""),
                    "parish": record.get("parish", ""),
                    "target_parish": record.get("target_parish", ""),
                    "district_name": record.get("district_name", ""),
                    "school_name": record.get("school_name", ""),
                    "storm_name": record.get("storm_name", ""),
                    "closure_scope": record.get("closure_scope", ""),
                    "closure_start_date": record.get("closure_start_date", ""),
                    "closure_end_date": record.get("closure_end_date", ""),
                    "issue": "no_matched_public_school",
                    "match_method": match_method,
                    "url": record.get("url", ""),
                }
            )
            continue

        for _, school in matched_schools.iterrows():
            for closure_date in dates:
                graduation_year = infer_graduation_year(closure_date)

                expanded_rows.append(
                    {
                        "measure_type": measure_type,
                        "auto_closure_record_id": record_id,
                        "target_event_id": record.get("target_event_id", ""),
                        "storm_name": record.get("storm_name", ""),
                        "storm_norm": record.get("storm_norm", ""),
                        "parish_norm": record.get("parish_norm", ""),
                        "closure_scope": record.get("closure_scope", ""),
                        "closure_match_method": match_method,
                        "closure_date": closure_date.date().isoformat(),
                        "graduation_year": graduation_year,
                        "nces_school_id": school["nces_school_id"],
                        "nces_school_name": school.get("nces_school_name", ""),
                        "nces_district_name": school.get("nces_district_name", ""),
                        "nces_parish": school.get("nces_parish", ""),
                        "latitude": school.get("latitude", ""),
                        "longitude": school.get("longitude", ""),
                        "source_confidence": record.get("source_confidence", ""),
                        "url": record.get("url", ""),
                        "evidence_text": record.get("evidence_text", ""),
                    }
                )

    school_dates = pd.DataFrame(expanded_rows)
    review = pd.DataFrame(review_rows)

    if school_dates.empty:
        return school_dates, review

    deduped = (
        school_dates
        .groupby(["measure_type", "nces_school_id", "closure_date"], dropna=False)
        .agg(
            graduation_year=("graduation_year", "first"),
            nces_school_name=("nces_school_name", "first"),
            nces_district_name=("nces_district_name", "first"),
            nces_parish=("nces_parish", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            auto_closure_record_ids=("auto_closure_record_id", lambda x: "; ".join(sorted(set(x.astype(str))))),
            target_event_ids=("target_event_id", lambda x: "; ".join(sorted(set(x.astype(str))))),
            storm_names=("storm_norm", lambda x: "; ".join(sorted(set(v for v in x.astype(str) if v.strip())))),
            parish_norms=("parish_norm", lambda x: "; ".join(sorted(set(v for v in x.astype(str) if v.strip())))),
            closure_scopes=("closure_scope", lambda x: "; ".join(sorted(set(x.astype(str))))),
            closure_match_methods=("closure_match_method", lambda x: "; ".join(sorted(set(x.astype(str))))),
            source_count=("auto_closure_record_id", "nunique"),
            source_urls=("url", lambda x: "; ".join(sorted(set(v for v in x.astype(str) if v.strip())))),
            max_source_confidence=("source_confidence", "max"),
        )
        .reset_index()
    )

    return deduped, review


def aggregate_school_year(school_dates: pd.DataFrame, measure_type: str) -> pd.DataFrame:
    """Aggregate school-date rows to school-year closure measures."""
    if school_dates.empty:
        return pd.DataFrame()

    df = school_dates.copy()
    df["graduation_year"] = pd.to_numeric(df["graduation_year"], errors="coerce")
    df = df[df["graduation_year"].notna()].copy()
    df["graduation_year"] = df["graduation_year"].astype(int)

    rows = []

    for (school_id, grad_year), group in df.groupby(["nces_school_id", "graduation_year"], dropna=False):
        rows.append(
            {
                "nces_school_id": school_id,
                "graduation_year": int(grad_year),
                f"closure_days_hurricane_related_{measure_type}": len(group),
                f"closure_any_hurricane_related_{measure_type}": int(len(group) > 0),
                f"closure_max_consecutive_days_{measure_type}": max_consecutive_schooldays(group["closure_date"]),
                f"closure_source_record_count_{measure_type}": len(
                    set(
                        record_id
                        for piece in group["auto_closure_record_ids"].astype(str)
                        for record_id in piece.split("; ")
                        if record_id.strip()
                    )
                ),
                f"closure_source_url_count_{measure_type}": len(
                    set(
                        url
                        for piece in group["source_urls"].astype(str)
                        for url in piece.split("; ")
                        if url.strip()
                    )
                ),
                f"closure_event_ids_{measure_type}": "; ".join(
                    sorted(
                        set(
                            event_id
                            for piece in group["target_event_ids"].astype(str)
                            for event_id in piece.split("; ")
                            if event_id.strip()
                        )
                    )
                ),
                f"closure_storm_names_{measure_type}": "; ".join(
                    sorted(
                        set(
                            storm
                            for piece in group["storm_names"].astype(str)
                            for storm in piece.split("; ")
                            if storm.strip()
                        )
                    )
                ),
                f"closure_record_ids_{measure_type}": "; ".join(
                    sorted(
                        set(
                            record_id
                            for piece in group["auto_closure_record_ids"].astype(str)
                            for record_id in piece.split("; ")
                            if record_id.strip()
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def merge_closure_panels(regression: pd.DataFrame, strict_year: pd.DataFrame, broad_year: pd.DataFrame) -> pd.DataFrame:
    """Merge strict and broad school-year closure measures into regression panel."""
    panel = regression.copy()
    panel["nces_school_id"] = panel["nces_school_id"].apply(lambda x: normalize_id(x, 12))
    panel["graduation_year"] = pd.to_numeric(panel["graduation_year"], errors="coerce").astype(int)

    merged = panel.copy()

    for closure_year in [strict_year, broad_year]:
        if closure_year.empty:
            continue

        closure_year = closure_year.copy()
        closure_year["nces_school_id"] = closure_year["nces_school_id"].apply(lambda x: normalize_id(x, 12))
        closure_year["graduation_year"] = pd.to_numeric(closure_year["graduation_year"], errors="coerce").astype(int)

        merged = merged.merge(
            closure_year,
            on=["nces_school_id", "graduation_year"],
            how="left",
            validate="one_to_one",
        )

    numeric_cols = [
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

    for col in numeric_cols:
        if col not in merged.columns:
            merged[col] = 0

        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    text_cols = [
        "closure_event_ids_strict",
        "closure_storm_names_strict",
        "closure_record_ids_strict",
        "closure_event_ids_broad",
        "closure_storm_names_broad",
        "closure_record_ids_broad",
    ]

    for col in text_cols:
        if col not in merged.columns:
            merged[col] = ""

        merged[col] = merged[col].fillna("")

    return merged


def make_expansion_summary(
    strict_records: pd.DataFrame,
    broad_records: pd.DataFrame,
    strict_dates: pd.DataFrame,
    broad_dates: pd.DataFrame,
    strict_year: pd.DataFrame,
    broad_year: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """Create summary table for expansion outputs."""
    rows = [
        {"metric": "strict_source_records", "value": len(strict_records)},
        {"metric": "broad_source_records", "value": len(broad_records)},
        {"metric": "strict_school_date_rows_deduped", "value": len(strict_dates)},
        {"metric": "broad_school_date_rows_deduped", "value": len(broad_dates)},
        {"metric": "strict_school_year_rows", "value": len(strict_year)},
        {"metric": "broad_school_year_rows", "value": len(broad_year)},
        {"metric": "merged_regression_panel_rows", "value": len(merged)},
        {
            "metric": "strict_total_school_closure_days",
            "value": int(pd.to_numeric(strict_year.get("closure_days_hurricane_related_strict", 0), errors="coerce").sum()) if not strict_year.empty else 0,
        },
        {
            "metric": "broad_total_school_closure_days",
            "value": int(pd.to_numeric(broad_year.get("closure_days_hurricane_related_broad", 0), errors="coerce").sum()) if not broad_year.empty else 0,
        },
        {
            "metric": "strict_panel_rows_with_closure",
            "value": int((merged["closure_days_hurricane_related_strict"] > 0).sum()),
        },
        {
            "metric": "broad_panel_rows_with_closure",
            "value": int((merged["closure_days_hurricane_related_broad"] > 0).sum()),
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    """Run closure expansion to school-year panel."""
    if not STRICT_RECORDS_PATH.exists():
        raise FileNotFoundError(f"Missing strict closure records: {STRICT_RECORDS_PATH}")

    if not BROAD_RECORDS_PATH.exists():
        raise FileNotFoundError(f"Missing broad closure records: {BROAD_RECORDS_PATH}")

    if not REGRESSION_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing regression panel: {REGRESSION_PANEL_PATH}")

    strict_records = pd.read_csv(STRICT_RECORDS_PATH, dtype=str, low_memory=False)
    broad_records = pd.read_csv(BROAD_RECORDS_PATH, dtype=str, low_memory=False)
    regression = pd.read_csv(REGRESSION_PANEL_PATH, dtype=str, low_memory=False)

    schools = build_school_universe(regression)

    strict_dates, strict_review = expand_records_to_school_dates(strict_records, schools, "strict")
    broad_dates, broad_review = expand_records_to_school_dates(broad_records, schools, "broad")

    strict_year = aggregate_school_year(strict_dates, "strict")
    broad_year = aggregate_school_year(broad_dates, "broad")

    merged = merge_closure_panels(regression, strict_year, broad_year)

    review = pd.concat([strict_review, broad_review], ignore_index=True)

    summary = make_expansion_summary(
        strict_records=strict_records,
        broad_records=broad_records,
        strict_dates=strict_dates,
        broad_dates=broad_dates,
        strict_year=strict_year,
        broad_year=broad_year,
        merged=merged,
    )

    strict_dates.to_csv(STRICT_SCHOOL_DATE_OUT, index=False)
    broad_dates.to_csv(BROAD_SCHOOL_DATE_OUT, index=False)
    strict_year.to_csv(STRICT_SCHOOL_YEAR_OUT, index=False)
    broad_year.to_csv(BROAD_SCHOOL_YEAR_OUT, index=False)
    review.to_csv(EXPANSION_REVIEW_OUT, index=False)
    summary.to_csv(EXPANSION_SUMMARY_OUT, index=False)
    merged.to_csv(MERGED_PANEL_OUT, index=False)

    print("Saved:")
    print(" -", STRICT_SCHOOL_DATE_OUT.relative_to(PROJECT_ROOT))
    print(" -", BROAD_SCHOOL_DATE_OUT.relative_to(PROJECT_ROOT))
    print(" -", STRICT_SCHOOL_YEAR_OUT.relative_to(PROJECT_ROOT))
    print(" -", BROAD_SCHOOL_YEAR_OUT.relative_to(PROJECT_ROOT))
    print(" -", EXPANSION_REVIEW_OUT.relative_to(PROJECT_ROOT))
    print(" -", EXPANSION_SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", MERGED_PANEL_OUT.relative_to(PROJECT_ROOT))

    print()
    print("Expansion summary:")
    print(summary.to_string(index=False))

    print()
    print("Broad school-year rows by graduation year:")
    if broad_year.empty:
        print("None")
    else:
        print(
            broad_year.groupby("graduation_year")
            .agg(
                rows=("nces_school_id", "size"),
                total_closure_days=("closure_days_hurricane_related_broad", "sum"),
                max_closure_days=("closure_days_hurricane_related_broad", "max"),
            )
            .reset_index()
            .sort_values("graduation_year")
            .to_string(index=False)
        )

    print()
    print("Top broad closure school-year rows:")
    if broad_year.empty:
        print("None")
    else:
        print(
            broad_year.sort_values("closure_days_hurricane_related_broad", ascending=False)
            .head(30)
            .to_string(index=False)
        )

    print()
    print("Expansion review rows:", len(review))
    if not review.empty:
        print(review.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
