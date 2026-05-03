"""
Build targeted source-collection queue for hurricane-related school closure data.

This script does not scrape the web and does not call OpenRouter.

It creates structured templates for collecting official district announcements,
local news articles, and archived web pages that mention school closures.

Inputs:
    data/processed/regression_panel.csv

Outputs:
    data/intermediate/closure_target_events.csv
    data/intermediate/closure_source_search_queue.csv
    data/intermediate/closure_manual_source_inventory.csv
    data/raw/closures/manual_sources/README.md

Design:
    Closure days are the main instructional-disruption treatment variable.
    This queue targets major Louisiana hurricane / tropical storm years and
    event-parish combinations that matter for the current regression sample.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGRESSION_PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "regression_panel.csv"

RAW_MANUAL_DIR = PROJECT_ROOT / "data" / "raw" / "closures" / "manual_sources"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

TARGET_EVENTS_OUT = INTERMEDIATE_DIR / "closure_target_events.csv"
SOURCE_QUEUE_OUT = INTERMEDIATE_DIR / "closure_source_search_queue.csv"
MANUAL_SOURCE_INVENTORY_OUT = INTERMEDIATE_DIR / "closure_manual_source_inventory.csv"
README_OUT = RAW_MANUAL_DIR / "README.md"


TARGET_EVENTS = [
    {
        "event_id": "2002_isidore_lili",
        "event_name": "Isidore / Lili",
        "storm_names": "Isidore; Lili",
        "storm_year": 2002,
        "graduation_year": 2003,
        "search_window_start": "2002-09-20",
        "search_window_end": "2002-10-15",
        "priority": "high",
        "notes": "Early major Louisiana tropical storm / hurricane disruption period.",
    },
    {
        "event_id": "2005_katrina_rita",
        "event_name": "Katrina / Rita",
        "storm_names": "Katrina; Rita",
        "storm_year": 2005,
        "graduation_year": 2006,
        "search_window_start": "2005-08-25",
        "search_window_end": "2005-10-15",
        "priority": "very_high",
        "notes": "Critical event; CDC data does not cover 2005.",
    },
    {
        "event_id": "2008_gustav_ike",
        "event_name": "Gustav / Ike",
        "storm_names": "Gustav; Ike",
        "storm_year": 2008,
        "graduation_year": 2009,
        "search_window_start": "2008-08-25",
        "search_window_end": "2008-09-20",
        "priority": "very_high",
        "notes": "Important high-exposure year in HURDAT measure.",
    },
    {
        "event_id": "2012_isaac",
        "event_name": "Tropical Storm Isaac",
        "storm_names": "Isaac",
        "storm_year": 2012,
        "graduation_year": 2013,
        "search_window_start": "2012-08-20",
        "search_window_end": "2012-09-15",
        "priority": "high",
        "notes": "Partially covered by CDC; useful for validating extraction.",
    },
    {
        "event_id": "2016_louisiana_flood_aug",
        "event_name": "Louisiana August Flood / Storm",
        "storm_names": "",
        "storm_year": 2016,
        "graduation_year": 2017,
        "search_window_start": "2016-08-10",
        "search_window_end": "2016-08-31",
        "priority": "medium",
        "notes": "CDC contains LA Storm - Aug. Include as disaster-related robustness, not main named-storm measure unless source links to tropical system.",
    },
    {
        "event_id": "2017_harvey",
        "event_name": "Harvey",
        "storm_names": "Harvey",
        "storm_year": 2017,
        "graduation_year": 2018,
        "search_window_start": "2017-08-25",
        "search_window_end": "2017-09-10",
        "priority": "medium",
        "notes": "HURDAT exposure present; check Louisiana school closures.",
    },
    {
        "event_id": "2019_barry",
        "event_name": "Barry",
        "storm_names": "Barry",
        "storm_year": 2019,
        "graduation_year": 2020,
        "search_window_start": "2019-07-10",
        "search_window_end": "2019-07-20",
        "priority": "high",
        "notes": "Important pre-2020 named storm event.",
    },
    {
        "event_id": "2020_laura_delta_zeta",
        "event_name": "Laura / Delta / Zeta",
        "storm_names": "Laura; Delta; Zeta",
        "storm_year": 2020,
        "graduation_year": 2021,
        "search_window_start": "2020-08-20",
        "search_window_end": "2020-11-05",
        "priority": "very_high",
        "notes": "Highest exposure year in current diagnostics.",
    },
    {
        "event_id": "2021_ida",
        "event_name": "Ida",
        "storm_names": "Ida",
        "storm_year": 2021,
        "graduation_year": 2022,
        "search_window_start": "2021-08-25",
        "search_window_end": "2021-09-30",
        "priority": "very_high",
        "notes": "High exposure and likely widespread closures.",
    },
    {
        "event_id": "2024_francine",
        "event_name": "Francine",
        "storm_names": "Francine",
        "storm_year": 2024,
        "graduation_year": 2025,
        "search_window_start": "2024-09-05",
        "search_window_end": "2024-09-20",
        "priority": "high",
        "notes": "Relevant to 2025 graduation cohort; recipient outcomes may be incomplete, but eligibility is usable.",
    },
]


def normalize_text(value: object) -> str:
    """Normalize text for source-query construction."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def prepare_panel() -> pd.DataFrame:
    """Read regression panel and prepare exposure-related fields."""
    panel = pd.read_csv(REGRESSION_PANEL_PATH, dtype=str, low_memory=False)

    for col in [
        "graduation_year",
        "storm_year",
        "exposure_index_pointmax",
        "within_100km_hurricane",
        "within_50km_hurricane",
    ]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    return panel


def get_event_parishes(panel: pd.DataFrame, event: dict) -> pd.DataFrame:
    """
    Select parishes/districts most relevant for one event.

    Include schools in the event's graduation year if they have high exposure or
    are within 100km of hurricane-force track points.
    """
    graduation_year = event["graduation_year"]

    subset = panel[panel["graduation_year"].eq(graduation_year)].copy()

    if subset.empty:
        return pd.DataFrame()

    high_exposure_mask = (
        subset["within_100km_hurricane"].fillna(0).eq(1)
        | (subset["exposure_index_pointmax"].fillna(0) >= 0.50)
    )

    targeted = subset[high_exposure_mask].copy()

    if targeted.empty:
        targeted = subset.sort_values("exposure_index_pointmax", ascending=False).head(30).copy()

    grouped = (
        targeted.groupby(["nces_parish", "nces_district_name"], dropna=False)
        .agg(
            schools=("nces_school_id", "nunique"),
            max_exposure=("exposure_index_pointmax", "max"),
            mean_exposure=("exposure_index_pointmax", "mean"),
            any_within_100km_hurricane=("within_100km_hurricane", "max"),
            any_within_50km_hurricane=("within_50km_hurricane", "max"),
        )
        .reset_index()
        .sort_values(["any_within_100km_hurricane", "max_exposure"], ascending=[False, False])
    )

    grouped["nces_parish"] = grouped["nces_parish"].fillna("").astype(str)
    grouped["nces_district_name"] = grouped["nces_district_name"].fillna("").astype(str)

    return grouped


def make_target_events() -> pd.DataFrame:
    """Create target-events table."""
    return pd.DataFrame(TARGET_EVENTS)


def make_source_queue(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Create targeted source-search queue."""
    rows = []

    for _, event in events.iterrows():
        event_dict = event.to_dict()
        event_parishes = get_event_parishes(panel, event_dict)

        for _, target in event_parishes.iterrows():
            parish = normalize_text(target["nces_parish"])
            district = normalize_text(target["nces_district_name"])
            event_name = normalize_text(event["event_name"])

            query_terms = [
                f'"{district}" school closure {event_name}',
                f'"{parish}" "schools closed" {event_name}',
                f'"{district}" "schools closed" hurricane',
                f'"{parish}" "school closures" "{event["search_window_start"][:4]}"',
            ]

            rows.append(
                {
                    "queue_id": f"{event['event_id']}__{parish.replace(' ', '_')}__{len(rows)+1}",
                    "event_id": event["event_id"],
                    "event_name": event["event_name"],
                    "storm_names": event["storm_names"],
                    "storm_year": event["storm_year"],
                    "graduation_year": event["graduation_year"],
                    "search_window_start": event["search_window_start"],
                    "search_window_end": event["search_window_end"],
                    "priority": event["priority"],
                    "target_parish": parish,
                    "target_district": district,
                    "target_schools_count": target["schools"],
                    "max_exposure": target["max_exposure"],
                    "mean_exposure": target["mean_exposure"],
                    "any_within_100km_hurricane": target["any_within_100km_hurricane"],
                    "suggested_search_query_1": query_terms[0],
                    "suggested_search_query_2": query_terms[1],
                    "suggested_search_query_3": query_terms[2],
                    "suggested_search_query_4": query_terms[3],
                    "source_url_found": "",
                    "source_title_found": "",
                    "source_type_found": "",
                    "collection_status": "not_started",
                    "notes": "",
                }
            )

    return pd.DataFrame(rows)


def make_manual_source_inventory() -> pd.DataFrame:
    """Create template for manually collected source files."""
    columns = [
        "source_id",
        "event_id",
        "event_name",
        "storm_year",
        "graduation_year",
        "target_parish",
        "target_district",
        "source_type",
        "source_title",
        "source_url",
        "local_file_path",
        "date_accessed",
        "collection_notes",
        "ready_for_api_extraction",
    ]

    return pd.DataFrame(columns=columns)


def write_readme() -> None:
    """Write instructions for manual closure source collection."""
    text = """# Manual Closure Source Collection

Place manually saved closure source files in this folder.

Recommended file types:

- `.txt`: copied page text
- `.html`: saved webpage source
- `.md`: notes copied from an official/news source
- `.pdf`: downloaded district announcement, if available

For each source file, add one row to:

`data/intermediate/closure_manual_source_inventory.csv`

Required fields:

- `source_id`: unique ID, e.g. `SRC_2020_LAURA_CALCASIEU_001`
- `event_id`: must match `closure_target_events.csv`
- `source_type`: official_district, official_state, news, archive, other
- `source_title`
- `source_url`
- `local_file_path`: relative path, e.g. `data/raw/closures/manual_sources/SRC_2020_LAURA_CALCASIEU_001.txt`
- `ready_for_api_extraction`: True

Important coding rule:

The source must explicitly mention school closure, district closure, school reopening,
remote instruction due to storm, or closure dates. Emergency declarations alone are
not enough.
"""
    README_OUT.write_text(text)


def main() -> None:
    """Build source-collection queue."""
    RAW_MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    panel = prepare_panel()

    events = make_target_events()
    queue = make_source_queue(panel, events)
    inventory = make_manual_source_inventory()

    events.to_csv(TARGET_EVENTS_OUT, index=False)
    queue.to_csv(SOURCE_QUEUE_OUT, index=False)
    inventory.to_csv(MANUAL_SOURCE_INVENTORY_OUT, index=False)
    write_readme()

    print("Saved:")
    print(" -", TARGET_EVENTS_OUT.relative_to(PROJECT_ROOT))
    print(" -", SOURCE_QUEUE_OUT.relative_to(PROJECT_ROOT))
    print(" -", MANUAL_SOURCE_INVENTORY_OUT.relative_to(PROJECT_ROOT))
    print(" -", README_OUT.relative_to(PROJECT_ROOT))

    print()
    print("Target events:", len(events))
    print("Source-search queue rows:", len(queue))

    print()
    print("Queue counts by event:")
    print(queue["event_id"].value_counts().to_string())

    print()
    print("Top 30 source-search priorities:")
    print(
        queue[
            [
                "event_id",
                "target_parish",
                "target_district",
                "target_schools_count",
                "max_exposure",
                "suggested_search_query_1",
                "collection_status",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
