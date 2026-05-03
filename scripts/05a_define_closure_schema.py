"""
Define school-closure data schema and coding protocol.

This script does not collect closure data. It creates standardized templates
that all closure sources should follow.

Closure days are treated as the main instructional-disruption treatment variable,
not as a pre-treatment control.

Primary final unit:
    NCES school × graduation year

Raw unit:
    closure announcement / closure record

Preferred intermediate unit:
    school × closure date

Main treatment:
    closure_days_hurricane_related
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CLOSURE_DIR = PROJECT_ROOT / "data" / "raw" / "closures"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SOURCES_INVENTORY_OUT = INTERMEDIATE_DIR / "closure_sources_inventory.csv"
RAW_TEMPLATE_OUT = INTERMEDIATE_DIR / "closure_records_raw_template.csv"
CLEAN_TEMPLATE_OUT = INTERMEDIATE_DIR / "closure_records_clean_template.csv"
MANUAL_REVIEW_TEMPLATE_OUT = INTERMEDIATE_DIR / "closure_manual_review_template.csv"
CODING_PROTOCOL_OUT = PROCESSED_DIR / "closure_data_coding_protocol.md"


def make_directories() -> None:
    """Create folder structure for closure data."""
    folders = [
        RAW_CLOSURE_DIR,
        RAW_CLOSURE_DIR / "cdc",
        RAW_CLOSURE_DIR / "gohsep",
        RAW_CLOSURE_DIR / "district_announcements",
        RAW_CLOSURE_DIR / "news",
        RAW_CLOSURE_DIR / "manual",
        INTERMEDIATE_DIR,
        PROCESSED_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def make_sources_inventory() -> pd.DataFrame:
    """Create source inventory template."""
    columns = [
        "source_id",
        "source_type",
        "source_name",
        "source_url_or_path",
        "source_year_start",
        "source_year_end",
        "geographic_scope",
        "source_priority",
        "status",
        "notes",
    ]

    rows = [
        {
            "source_id": "CDC_PUSC_BASELINE",
            "source_type": "official_dataset",
            "source_name": "CDC Prolonged Unplanned School Closures",
            "source_url_or_path": "",
            "source_year_start": "",
            "source_year_end": "",
            "geographic_scope": "United States",
            "source_priority": "high",
            "status": "to_prepare",
            "notes": "Baseline official source. Use if record clearly identifies Louisiana school/district closure.",
        },
        {
            "source_id": "GOHSEP_INCIDENTS",
            "source_type": "official_emergency_context",
            "source_name": "Louisiana GOHSEP incidents / emergency information",
            "source_url_or_path": "",
            "source_year_start": "",
            "source_year_end": "",
            "geographic_scope": "Louisiana",
            "source_priority": "medium",
            "status": "to_prepare",
            "notes": "Use for disaster context only. Do not count as school closure unless closure is explicitly stated.",
        },
        {
            "source_id": "DISTRICT_ANNOUNCEMENTS",
            "source_type": "school_system_announcement",
            "source_name": "Louisiana parish/district school closure announcements",
            "source_url_or_path": "",
            "source_year_start": "",
            "source_year_end": "",
            "geographic_scope": "Parish / district",
            "source_priority": "high",
            "status": "to_collect",
            "notes": "Preferred source for closure dates when district explicitly announces schools closed.",
        },
        {
            "source_id": "LOCAL_NEWS",
            "source_type": "news",
            "source_name": "Local news school closure reports",
            "source_url_or_path": "",
            "source_year_start": "",
            "source_year_end": "",
            "geographic_scope": "Parish / district / school",
            "source_priority": "medium",
            "status": "to_collect",
            "notes": "Use when official district source unavailable. Must explicitly mention school closure.",
        },
    ]

    return pd.DataFrame(rows, columns=columns)


def make_raw_template() -> pd.DataFrame:
    """Create raw closure extraction template."""
    columns = [
        "raw_record_id",
        "source_id",
        "source_type",
        "source_url_or_file",
        "source_title",
        "source_publication_date",
        "extraction_method",
        "extracted_text_snippet",
        "closure_scope_raw",
        "state_raw",
        "parish_raw",
        "district_raw",
        "school_raw",
        "closure_dates_raw",
        "closure_start_date_raw",
        "closure_end_date_raw",
        "closure_days_reported_raw",
        "closure_reason_raw",
        "storm_name_raw",
        "hurricane_related_raw",
        "confidence_raw",
        "needs_manual_review_raw",
        "raw_notes",
    ]

    return pd.DataFrame(columns=columns)


def make_clean_template() -> pd.DataFrame:
    """Create cleaned closure-record template."""
    columns = [
        "closure_record_id",
        "raw_record_id",
        "source_id",
        "source_type",
        "source_url_or_file",
        "source_title",
        "source_publication_date",
        "closure_scope",
        "state",
        "parish",
        "district_name",
        "school_name",
        "nces_school_id",
        "nces_school_name",
        "closure_start_date",
        "closure_end_date",
        "closure_days",
        "academic_year_start",
        "graduation_year",
        "closure_reason_raw",
        "closure_reason_category",
        "storm_name",
        "hurricane_related",
        "instructional_closure",
        "include_in_main_closure_measure",
        "confidence",
        "needs_manual_review",
        "review_notes",
    ]

    return pd.DataFrame(columns=columns)


def make_manual_review_template() -> pd.DataFrame:
    """Create manual-review template for ambiguous closure records."""
    columns = [
        "closure_record_id",
        "raw_record_id",
        "issue_type",
        "parish",
        "district_name",
        "school_name",
        "closure_start_date",
        "closure_end_date",
        "closure_reason_raw",
        "source_url_or_file",
        "source_title",
        "current_decision",
        "manual_decision",
        "manual_nces_school_id",
        "manual_graduation_year",
        "manual_hurricane_related",
        "manual_include_in_main_measure",
        "reviewer_notes",
    ]

    return pd.DataFrame(columns=columns)


def write_coding_protocol() -> None:
    """Write closure-data coding protocol."""
    text = """# School Closure Data Coding Protocol

## Purpose

The closure data measure actual instructional disruption. In this project, hurricane-related closure days are treated as the main treatment-intensity variable in the school-closure analysis.

They are not pre-treatment controls. They occur after the disaster shock and may mediate the relationship between physical hurricane exposure and TOPS eligibility.

## Raw unit

The raw unit is a closure announcement or closure record.

Examples:

- A district announcement that all schools are closed from Monday to Wednesday.
- A news article reporting that a parish school system closed due to Hurricane Ida.
- A school-specific notice that one high school closed due to storm damage.

## Preferred cleaned unit

The preferred cleaned unit is a closure record with start and end date, closure scope, parish/district/school identifiers, and reason category.

Later scripts will expand these records to school-date level and aggregate to NCES school × graduation year.

## Academic-year mapping

For graduation year t, the relevant school year is:

- August 1 of t-1 through May 31 of t.

Example:

- graduation_year = 2021
- closure window = 2020-08-01 to 2021-05-31

Closures outside this window should not be counted for that graduating cohort in the main specification.

## Closure scope

Use one of:

- `school_specific`
- `district_system`
- `parishwide`
- `statewide`
- `unclear`

Coding rules:

- If the source says all schools in a parish/district are closed, code as `district_system` or `parishwide`.
- If the source names a specific school, code as `school_specific`.
- If the source only mentions a general emergency declaration, do not count it as a school closure unless school closure is explicitly stated.

## Closure reason category

Use one of:

- `hurricane`
- `tropical_storm`
- `severe_weather`
- `flood`
- `power_outage`
- `covid`
- `other`
- `unclear`

Main treatment uses:

- `hurricane_related == True`
- `instructional_closure == True`
- `include_in_main_closure_measure == True`

## Hurricane-related coding

Set `hurricane_related = True` if the closure is explicitly related to:

- a named hurricane,
- a named tropical storm,
- hurricane landfall,
- tropical storm conditions,
- hurricane evacuation,
- hurricane-related flooding, damage, or power outage.

Set `hurricane_related = False` for:

- COVID closure,
- normal holiday,
- teacher work day,
- unrelated illness,
- ordinary maintenance,
- non-weather administrative closure.

Set `needs_manual_review = True` if the source is ambiguous.

## Instructional closure

Set `instructional_closure = True` if students lost ordinary in-person instructional time.

Set `instructional_closure = False` if:

- the source only reports office closure,
- extracurricular cancellation only,
- school was open virtually and no instructional loss is indicated,
- the record is only an emergency declaration with no school closure.

## Main closure measure

The main treatment variable will be:

`closure_days_hurricane_related`

At the school-year level, this is the number of instructional closure days during the academic-year window that are hurricane-related.

Additional variables:

- `closure_any_hurricane_related`
- `closure_event_count_hurricane_related`
- `max_consecutive_closure_days_hurricane_related`

## Source hierarchy

Preferred sources:

1. Official school/district/parish school-system announcements.
2. CDC official school closure data when available and clearly identifiable.
3. Local news reports explicitly describing school closures.
4. GOHSEP / emergency information only as contextual evidence, not as closure proof.

## API use

OpenRouter / LLM extraction may be used only to convert source text into structured records.

The API should not invent dates, schools, parishes, or closure reasons. If the text is ambiguous, the record must be flagged for manual review.
"""

    CODING_PROTOCOL_OUT.write_text(text)


def main() -> None:
    """Create closure data schema files."""
    make_directories()

    sources = make_sources_inventory()
    raw_template = make_raw_template()
    clean_template = make_clean_template()
    manual_review_template = make_manual_review_template()

    sources.to_csv(SOURCES_INVENTORY_OUT, index=False)
    raw_template.to_csv(RAW_TEMPLATE_OUT, index=False)
    clean_template.to_csv(CLEAN_TEMPLATE_OUT, index=False)
    manual_review_template.to_csv(MANUAL_REVIEW_TEMPLATE_OUT, index=False)

    write_coding_protocol()

    print("Created closure data folders and templates:")
    print(" -", SOURCES_INVENTORY_OUT.relative_to(PROJECT_ROOT))
    print(" -", RAW_TEMPLATE_OUT.relative_to(PROJECT_ROOT))
    print(" -", CLEAN_TEMPLATE_OUT.relative_to(PROJECT_ROOT))
    print(" -", MANUAL_REVIEW_TEMPLATE_OUT.relative_to(PROJECT_ROOT))
    print(" -", CODING_PROTOCOL_OUT.relative_to(PROJECT_ROOT))

    print()
    print("Next step:")
    print("Place CDC closure data in data/raw/closures/cdc/ if available, then run 05b_prepare_cdc_closure_data.py.")


if __name__ == "__main__":
    main()
