"""
Validate and rescue auto-extracted closure records.

Input:
    data/intermediate/closure_auto_extracted_records_clean.csv

Outputs:
    data/intermediate/closure_auto_records_validated_strict.csv
    data/intermediate/closure_auto_records_validated_broad.csv
    data/intermediate/closure_auto_records_rescued_from_review.csv
    data/intermediate/closure_auto_records_validation_summary.csv
    data/intermediate/closure_auto_records_validation_reason_summary.csv

Purpose:
    05g extracts source-level closure records.
    05g2 creates two usable versions:

    1. strict:
        Original high-confidence main-measure records from 05g.

    2. broad:
        Strict records plus conservatively rescued records from review.

Broad rescue rules:
    - Keep hurricane-related instructional closures with matching target storm.
    - Allow single-day closure inference when start date exists but end date is missing.
    - Allow reopen-date inference when reopen_date exists.
    - Allow interval borrowing from another source for the same event × parish × storm.
    - Infer parish from target_parish if source parish is missing.
    - Do not rescue non-instructional, storm-mismatched, non-hurricane, or implausible-date records.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_extracted_records_clean.csv"

STRICT_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_records_validated_strict.csv"
BROAD_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_records_validated_broad.csv"
RESCUED_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_records_rescued_from_review.csv"
SUMMARY_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_records_validation_summary.csv"
REASON_SUMMARY_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_records_validation_reason_summary.csv"


EVENT_WINDOWS = {
    "2005_katrina_rita": ("2005-08-20", "2006-06-30"),
    "2012_isaac": ("2012-08-20", "2012-09-30"),
    "2020_laura_delta_zeta": ("2020-08-20", "2020-11-30"),
    "2021_ida": ("2021-08-25", "2022-01-31"),
    "2024_francine": ("2024-09-05", "2024-10-15"),
}


VALID_SCOPES = {
    "district_system",
    "parishwide",
    "school_specific",
    "statewide",
}


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


def date_to_string(value: pd.Timestamp | pd.NaT) -> str:
    """Convert parsed date to ISO string."""
    if pd.isna(value):
        return ""

    return value.date().isoformat()


def count_weekdays_inclusive(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Count weekdays from start_date through end_date inclusive."""
    if pd.isna(start_date) or pd.isna(end_date) or end_date < start_date:
        return 0

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return int(sum(date.weekday() < 5 for date in dates))


def infer_graduation_year(date: pd.Timestamp) -> int | None:
    """Map closure date to graduation year."""
    if pd.isna(date):
        return None

    if int(date.month) >= 8:
        return int(date.year) + 1

    return int(date.year)


def normalize_text(value: object) -> str:
    """Normalize generic text."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_upper(value: object) -> str:
    """Normalize text to uppercase alphanumeric-ish form."""
    text = normalize_text(value).upper()
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
        "ST CHARLES": "ST. CHARLES",
        "ST JAMES": "ST. JAMES",
        "ST BERNARD": "ST. BERNARD",
        "ST MARY": "ST. MARY",
        "EAST BATON ROUGE": "EAST BATON ROUGE",
        "WEST BATON ROUGE": "WEST BATON ROUGE",
        "POINTE COUPEE": "POINTE COUPEE",
    }

    return aliases.get(text, text)


def normalize_storm_name(value: object) -> str:
    """Normalize storm name for matching."""
    text = normalize_upper(value)
    text = text.replace("HURRICANE", "")
    text = text.replace("TROPICAL STORM", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_expected_storms(value: object) -> set[str]:
    """Parse expected target storm names."""
    text = normalize_text(value)

    if not text:
        return set()

    storms = set()

    for part in re.split(r"[;,]", text):
        storm = normalize_storm_name(part)
        if storm:
            storms.add(storm)

    return storms


def storm_matches_expected(row: pd.Series) -> bool:
    """Return True if extracted storm_name matches expected target storm."""
    expected = parse_expected_storms(row.get("expected_storms", ""))
    found = normalize_storm_name(row.get("storm_name", ""))

    if not expected or not found:
        return False

    return any(found == storm or found in storm or storm in found for storm in expected)


def event_window_ok(row: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Check whether closure dates fall in plausible event window."""
    event_id = normalize_text(row.get("target_event_id", ""))

    if event_id not in EVENT_WINDOWS:
        return True

    lower = parse_date(EVENT_WINDOWS[event_id][0])
    upper = parse_date(EVENT_WINDOWS[event_id][1])

    if pd.isna(start) or pd.isna(end):
        return False

    return not (end < lower or start > upper)


def scope_is_valid_or_rescuable(row: pd.Series) -> bool:
    """Check whether closure scope is usable."""
    scope = normalize_text(row.get("closure_scope", "")).lower()

    if scope in VALID_SCOPES:
        return True

    # If source gives parish-level information but scope is unclear,
    # allow broad measure to treat it as parishwide.
    parish = normalize_parish(row.get("parish", ""))
    target_parish = normalize_parish(row.get("target_parish", ""))

    return bool(parish or target_parish)


def text_suggests_single_day_closure(row: pd.Series) -> bool:
    """
    Decide whether start-only record can be treated as single-day closure.

    This is intentionally broader than the 05g strict rule.
    """
    text = " ".join(
        [
            str(row.get("evidence_text", "")),
            str(row.get("closure_reason_raw", "")),
            str(row.get("source_level_notes", "")),
            str(row.get("review_notes", "")),
            str(row.get("deterministic_review_notes", "")),
        ]
    ).lower()

    closure_terms = [
        "closed",
        "closure",
        "closures",
        "classes canceled",
        "classes cancelled",
        "school canceled",
        "school cancelled",
        "campuses closed",
        "facilities closed",
        "early dismissal",
        "remote learning",
        "virtual learning",
    ]

    return any(term in text for term in closure_terms)


def build_canonical_intervals(df: pd.DataFrame) -> dict[tuple[str, str, str], tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build canonical intervals from records that already have usable start and end dates.

    Key:
        target_event_id × parish_norm × storm_norm
    """
    intervals: dict[tuple[str, str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}

    for _, row in df.iterrows():
        if not parse_bool(row.get("hurricane_related", False)):
            continue

        if not parse_bool(row.get("instructional_closure", False)):
            continue

        if not storm_matches_expected(row):
            continue

        confidence = pd.to_numeric(row.get("confidence", 0), errors="coerce")
        if pd.isna(confidence) or confidence < 0.80:
            continue

        start = parse_date(row.get("closure_start_date", ""))
        end = parse_date(row.get("closure_end_date", ""))

        if pd.isna(start) or pd.isna(end) or end < start:
            continue

        parish = normalize_parish(row.get("parish", "")) or normalize_parish(row.get("target_parish", ""))
        storm = normalize_storm_name(row.get("storm_name", ""))
        event = normalize_text(row.get("target_event_id", ""))

        if not parish or not storm or not event:
            continue

        if not event_window_ok(row, start, end):
            continue

        key = (event, parish, storm)
        intervals.setdefault(key, []).append((start, end))

    canonical = {}

    for key, ranges in intervals.items():
        starts = [x[0] for x in ranges]
        ends = [x[1] for x in ranges]

        canonical[key] = (min(starts), max(ends))

    return canonical


def rescue_dates(row: pd.Series, canonical_intervals: dict) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, str]:
    """Determine broad start/end date and rescue rule."""
    start = parse_date(row.get("closure_start_date", ""))
    end = parse_date(row.get("closure_end_date", ""))
    reopen = parse_date(row.get("reopen_date", ""))

    # Original complete interval.
    if pd.notna(start) and pd.notna(end) and end >= start:
        return start, end, "original_complete_interval"

    # Reopen-date inference: closure ends the previous calendar day.
    if pd.notna(start) and pd.isna(end) and pd.notna(reopen):
        inferred_end = reopen - pd.Timedelta(days=1)
        if inferred_end >= start:
            return start, inferred_end, "end_inferred_from_reopen_date"

    # Single-day rescue.
    if pd.notna(start) and pd.isna(end) and text_suggests_single_day_closure(row):
        return start, start, "single_day_inferred_from_start_date"

    # Borrow interval from another source for same event × parish × storm.
    event = normalize_text(row.get("target_event_id", ""))
    parish = normalize_parish(row.get("parish", "")) or normalize_parish(row.get("target_parish", ""))
    storm = normalize_storm_name(row.get("storm_name", ""))

    key = (event, parish, storm)

    if key in canonical_intervals:
        borrowed_start, borrowed_end = canonical_intervals[key]
        return borrowed_start, borrowed_end, "borrowed_interval_same_event_parish_storm"

    return pd.NaT, pd.NaT, "unresolved_dates"


def validate_broad_record(row: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[bool, str]:
    """Validate a broad-measure closure record."""
    reasons = []

    if not parse_bool(row.get("hurricane_related", False)):
        reasons.append("not_hurricane_related")

    if not parse_bool(row.get("instructional_closure", False)):
        reasons.append("not_instructional_closure")

    if not storm_matches_expected(row):
        reasons.append("storm_mismatch")

    parish = normalize_parish(row.get("parish", "")) or normalize_parish(row.get("target_parish", ""))

    if not parish:
        reasons.append("missing_parish")

    if not scope_is_valid_or_rescuable(row):
        reasons.append("bad_scope")

    confidence = pd.to_numeric(row.get("confidence", 0), errors="coerce")
    if pd.isna(confidence) or confidence < 0.80:
        reasons.append("low_confidence")

    if pd.isna(start) or pd.isna(end) or end < start:
        reasons.append("unusable_dates")

    if pd.notna(start) and pd.notna(end) and not event_window_ok(row, start, end):
        reasons.append("outside_event_window")

    closure_days = count_weekdays_inclusive(start, end)

    if closure_days <= 0:
        reasons.append("nonpositive_closure_days")

    include = len(reasons) == 0

    return include, "; ".join(reasons)


def prepare_record(row: pd.Series, canonical_intervals: dict) -> dict:
    """Prepare one validated broad record."""
    start, end, rescue_rule = rescue_dates(row, canonical_intervals)

    include_broad, broad_exclusion_reasons = validate_broad_record(row, start, end)

    parish_norm = normalize_parish(row.get("parish", "")) or normalize_parish(row.get("target_parish", ""))
    storm_norm = normalize_storm_name(row.get("storm_name", ""))

    graduation_year = infer_graduation_year(start)
    academic_year_start = graduation_year - 1 if graduation_year else None

    out = row.to_dict()

    out["original_include_in_main_closure_measure"] = parse_bool(row.get("include_in_main_closure_measure", False))
    out["validated_broad_include"] = include_broad
    out["broad_rescue_rule"] = rescue_rule
    out["broad_exclusion_reasons"] = broad_exclusion_reasons
    out["parish_norm"] = parish_norm
    out["storm_norm"] = storm_norm
    out["storm_matches_expected"] = storm_matches_expected(row)
    out["broad_closure_start_date"] = date_to_string(start)
    out["broad_closure_end_date"] = date_to_string(end)
    out["broad_closure_days"] = count_weekdays_inclusive(start, end)
    out["broad_graduation_year"] = graduation_year
    out["broad_academic_year_start"] = academic_year_start

    # For downstream 05h, broad output should use the rescued dates as the
    # active closure dates.
    if include_broad:
        out["closure_start_date"] = out["broad_closure_start_date"]
        out["closure_end_date"] = out["broad_closure_end_date"]
        out["closure_days"] = out["broad_closure_days"]
        out["graduation_year"] = out["broad_graduation_year"]
        out["academic_year_start"] = out["broad_academic_year_start"]
        out["parish"] = parish_norm
        out["storm_name"] = storm_norm

        scope = normalize_text(row.get("closure_scope", "")).lower()
        if scope not in VALID_SCOPES:
            out["closure_scope"] = "parishwide"

    return out


def make_summary(strict: pd.DataFrame, broad: pd.DataFrame, rescued: pd.DataFrame, all_validated: pd.DataFrame) -> pd.DataFrame:
    """Create validation summary."""
    rows = [
        {"metric": "input_clean_records", "value": len(all_validated)},
        {"metric": "strict_records", "value": len(strict)},
        {"metric": "broad_records", "value": len(broad)},
        {"metric": "rescued_from_review_records", "value": len(rescued)},
        {"metric": "strict_total_closure_days_source_level", "value": pd.to_numeric(strict.get("closure_days", 0), errors="coerce").sum()},
        {"metric": "broad_total_closure_days_source_level", "value": pd.to_numeric(broad.get("closure_days", 0), errors="coerce").sum()},
    ]

    if not broad.empty:
        for event_id, count in broad["target_event_id"].value_counts(dropna=False).items():
            rows.append({"metric": f"broad_records_event_{event_id}", "value": count})

    return pd.DataFrame(rows)


def main() -> None:
    """Run validation and rescue."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)

    canonical_intervals = build_canonical_intervals(df)

    validated_rows = [prepare_record(row, canonical_intervals) for _, row in df.iterrows()]
    validated = pd.DataFrame(validated_rows)

    strict = validated[validated["original_include_in_main_closure_measure"]].copy()
    broad = validated[validated["validated_broad_include"]].copy()
    rescued = broad[~broad["original_include_in_main_closure_measure"]].copy()

    strict.to_csv(STRICT_OUT, index=False)
    broad.to_csv(BROAD_OUT, index=False)
    rescued.to_csv(RESCUED_OUT, index=False)

    summary = make_summary(strict, broad, rescued, validated)
    summary.to_csv(SUMMARY_OUT, index=False)

    reason_summary = (
        validated.groupby(["validated_broad_include", "broad_exclusion_reasons"], dropna=False)
        .agg(
            records=("auto_closure_record_id", "size"),
        )
        .reset_index()
        .sort_values(["validated_broad_include", "records"], ascending=[False, False])
    )
    reason_summary.to_csv(REASON_SUMMARY_OUT, index=False)

    print("Saved:")
    print(" -", STRICT_OUT.relative_to(PROJECT_ROOT))
    print(" -", BROAD_OUT.relative_to(PROJECT_ROOT))
    print(" -", RESCUED_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", REASON_SUMMARY_OUT.relative_to(PROJECT_ROOT))

    print()
    print("Input clean records:", len(df))
    print("Strict records:", len(strict))
    print("Broad records:", len(broad))
    print("Rescued from review:", len(rescued))

    print()
    print("Broad records by event:")
    if broad.empty:
        print("None")
    else:
        print(broad["target_event_id"].value_counts(dropna=False).to_string())

    print()
    print("Broad rescue-rule counts:")
    if broad.empty:
        print("None")
    else:
        print(broad["broad_rescue_rule"].value_counts(dropna=False).to_string())

    print()
    print("Broad exclusion reason summary:")
    print(reason_summary.to_string(index=False))

    print()
    print("Rescued records preview:")
    if rescued.empty:
        print("None")
    else:
        cols = [
            "auto_closure_record_id",
            "target_event_id",
            "parish",
            "storm_name",
            "closure_scope",
            "closure_start_date",
            "closure_end_date",
            "closure_days",
            "broad_rescue_rule",
            "confidence",
            "url",
        ]
        existing_cols = [col for col in cols if col in rescued.columns]
        print(rescued[existing_cols].head(80).to_string(index=False))


if __name__ == "__main__":
    main()
