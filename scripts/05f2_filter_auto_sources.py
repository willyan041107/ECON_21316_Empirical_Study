"""
Filter auto-collected closure source screenings.

Inputs:
    data/intermediate/closure_auto_source_screening.csv
    data/intermediate/closure_source_priority_queue_top.csv

Outputs:
    data/intermediate/closure_auto_source_screening_filtered.csv
    data/intermediate/closure_auto_source_filter_summary.csv

Purpose:
    Keep only source candidates that match the target event's expected storm names
    and storm year before structured closure extraction.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENING_PATH = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_source_screening.csv"
QUEUE_PATH = PROJECT_ROOT / "data" / "intermediate" / "closure_source_priority_queue_top.csv"

FILTERED_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_source_screening_filtered.csv"
SUMMARY_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_source_filter_summary.csv"


def normalize_text(value: object) -> str:
    """Normalize storm/parish text for matching."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = text.replace("HURRICANE", "")
    text = text.replace("TROPICAL STORM", "")
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_list_like(value: object) -> list[str]:
    """Parse list-like, semicolon-separated, or comma-separated values."""
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # Handle strings like [Laura, Delta] or [Hurricane Laura]
    text = text.strip("[]")
    text = text.replace("'", "").replace('"', "")

    return [x.strip() for x in re.split(r"[;,]", text) if x.strip()]


def parse_date_year(value: object) -> int | None:
    """Extract year from date-like text."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    match = re.search(r"(19|20)\d{2}", text)

    if not match:
        return None

    return int(match.group(0))


def get_queue_value(row: pd.Series, name: str) -> object:
    """Get queue metadata from explicitly prefixed queue columns."""
    return row.get(f"queue_{name}", "")


def expected_storms_from_queue(row: pd.Series) -> list[str]:
    """Get target storm names from the priority queue metadata."""
    storms = parse_list_like(get_queue_value(row, "storm_names"))

    if storms:
        return [normalize_text(s) for s in storms if normalize_text(s)]

    event_name = normalize_text(get_queue_value(row, "event_name"))
    return [event_name] if event_name else []


def source_storms_from_screening(row: pd.Series) -> list[str]:
    """Get storm names found by OpenRouter screening."""
    storms = parse_list_like(row.get("storm_names", ""))

    return [normalize_text(s) for s in storms if normalize_text(s)]


def has_storm_match(expected: list[str], found: list[str]) -> bool:
    """Check whether found source storms match expected target storms."""
    if not expected or not found:
        return False

    for expected_storm in expected:
        for found_storm in found:
            if (
                expected_storm == found_storm
                or expected_storm in found_storm
                or found_storm in expected_storm
            ):
                return True

    return False


def has_year_match(row: pd.Series, storm_year: int | None) -> bool:
    """
    Check whether extracted closure dates match target storm year.

    If no date is extracted, keep the source if the storm name matches.
    The extraction step can recover exact dates from the source text later.
    """
    if storm_year is None:
        return False

    years = []

    for col in ["closure_start_date_guess", "closure_end_date_guess"]:
        year = parse_date_year(row.get(col, ""))
        if year is not None:
            years.append(year)

    if not years:
        return True

    return any(year == storm_year for year in years)


def has_parish_match(row: pd.Series) -> bool:
    """Check whether extracted parish/district roughly matches target parish."""
    target_parish = normalize_text(get_queue_value(row, "target_parish"))
    parish_display = normalize_text(str(get_queue_value(row, "parish_display")).replace("PARISH", ""))

    found_parish = normalize_text(str(row.get("parish", "")).replace("PARISH", ""))
    found_district = normalize_text(str(row.get("district_name", "")).replace("PARISH", ""))

    targets = {target_parish, parish_display}
    targets = {x for x in targets if x}

    found = " ".join([found_parish, found_district]).strip()

    if not targets or not found:
        return False

    return any(t in found or found in t for t in targets)


def main() -> None:
    """Filter source screenings."""
    if not SCREENING_PATH.exists():
        raise FileNotFoundError(f"Missing screening file: {SCREENING_PATH}")

    if not QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing priority queue file: {QUEUE_PATH}")

    screening = pd.read_csv(SCREENING_PATH, dtype=str, low_memory=False)
    queue = pd.read_csv(QUEUE_PATH, dtype=str, low_memory=False)

    queue_cols = [
        "priority_queue_id",
        "event_id",
        "event_name",
        "storm_names",
        "storm_year",
        "target_parish",
        "parish_display",
        "search_window_start",
        "search_window_end",
    ]

    queue_small = queue[queue_cols].copy()
    queue_small = queue_small.rename(
        columns={col: f"queue_{col}" for col in queue_cols if col != "priority_queue_id"}
    )

    merged = screening.merge(
        queue_small,
        on="priority_queue_id",
        how="left",
        validate="many_to_one",
    )

    rows = []

    for _, row in merged.iterrows():
        usable = str(row.get("usable_source", "")).strip().lower() == "true"

        storm_year_raw = get_queue_value(row, "storm_year")
        try:
            storm_year = int(float(storm_year_raw))
        except Exception:
            storm_year = None

        expected_storms = expected_storms_from_queue(row)
        found_storms = source_storms_from_screening(row)

        storm_match = has_storm_match(expected_storms, found_storms)
        year_match = has_year_match(row, storm_year)
        parish_match = has_parish_match(row)

        keep = usable and storm_match and year_match
        needs_manual_review = keep and not parish_match

        if not usable:
            reason = "not_usable_source"
        elif not storm_match:
            reason = "storm_mismatch"
        elif not year_match:
            reason = "year_mismatch"
        elif needs_manual_review:
            reason = "keep_but_parish_review"
        else:
            reason = "keep"

        row_dict = row.to_dict()
        row_dict["target_event_id"] = get_queue_value(row, "event_id")
        row_dict["target_storm_year"] = storm_year
        row_dict["expected_storms"] = "; ".join(expected_storms)
        row_dict["found_storms"] = "; ".join(found_storms)
        row_dict["storm_match"] = storm_match
        row_dict["year_match"] = year_match
        row_dict["parish_match"] = parish_match
        row_dict["keep_for_extraction"] = keep
        row_dict["needs_manual_review_after_filter"] = needs_manual_review
        row_dict["filter_reason"] = reason

        rows.append(row_dict)

    filtered = pd.DataFrame(rows)
    filtered.to_csv(FILTERED_OUT, index=False)

    summary = (
        filtered.groupby(["keep_for_extraction", "filter_reason"], dropna=False)
        .agg(
            rows=("candidate_id", "size"),
            unique_urls=("url", "nunique"),
        )
        .reset_index()
        .sort_values(["keep_for_extraction", "rows"], ascending=[False, False])
    )

    summary.to_csv(SUMMARY_OUT, index=False)

    print("Saved:")
    print(" -", FILTERED_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_OUT.relative_to(PROJECT_ROOT))

    print()
    print("Rows screened:", len(filtered))
    print("Rows kept for extraction:", int(filtered["keep_for_extraction"].sum()))

    print()
    print("Filter summary:")
    print(summary.to_string(index=False))

    print()
    print("Kept sources:")
    kept = filtered[filtered["keep_for_extraction"]].copy()

    if kept.empty:
        print("None")
    else:
        print(
            kept[
                [
                    "candidate_id",
                    "target_event_id",
                    "queue_target_parish",
                    "expected_storms",
                    "found_storms",
                    "closure_start_date_guess",
                    "closure_end_date_guess",
                    "confidence",
                    "filter_reason",
                    "url",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
