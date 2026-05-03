"""
Prioritize and collapse closure source-search queue.

Input:
    data/intermediate/closure_source_search_queue.csv

Outputs:
    data/intermediate/closure_source_priority_queue.csv
    data/intermediate/closure_source_priority_queue_top.csv
    data/intermediate/closure_source_priority_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "closure_source_search_queue.csv"

OUT_PRIORITY = PROJECT_ROOT / "data" / "intermediate" / "closure_source_priority_queue.csv"
OUT_TOP = PROJECT_ROOT / "data" / "intermediate" / "closure_source_priority_queue_top.csv"
OUT_SUMMARY = PROJECT_ROOT / "data" / "intermediate" / "closure_source_priority_summary.csv"


PRIORITY_ORDER = {
    "very_high": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
}


EVENT_ORDER = {
    "2020_laura_delta_zeta": 1,
    "2021_ida": 2,
    "2005_katrina_rita": 3,
    "2008_gustav_ike": 4,
    "2012_isaac": 5,
    "2024_francine": 6,
    "2019_barry": 7,
    "2002_isidore_lili": 8,
    "2017_harvey": 9,
    "2016_louisiana_flood_aug": 10,
}


def normalize_text(value: object) -> str:
    """Normalize text for grouping and query construction."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def parse_bool_numeric(value: object) -> int:
    """Parse True/False, 1/0, yes/no, or numeric indicators into 0/1."""
    if pd.isna(value):
        return 0

    text = str(value).strip().lower()

    if text in {"true", "1", "1.0", "yes"}:
        return 1

    return 0


def is_parish_system_name(district: str) -> bool:
    """Return True if district looks like a normal parish school system."""
    text = normalize_text(district).upper()

    if not text:
        return False

    ordinary_patterns = [
        " PARISH",
        "PARISH SCHOOL",
        "SCHOOL BOARD",
        "PUBLIC SCHOOLS",
        "SCHOOL DISTRICT",
    ]

    special_operator_patterns = [
        "KIPP",
        "CHARTER",
        "FOUNDATION",
        "LABORATORY",
        "RECOVERY SCHOOL DISTRICT",
        "NEW ORLEANS COLLEGE PREPARATORY",
        "INNOVATIONS",
        "MAX CHARTER",
    ]

    if any(pattern in text for pattern in special_operator_patterns):
        return False

    return any(pattern in text for pattern in ordinary_patterns)


def parish_display_name(parish: str) -> str:
    """Create display-friendly parish name."""
    parish = normalize_text(parish).upper()

    special = {
        "ST. JOHN THE BAPTIST": "St. John the Baptist Parish",
        "ST JOHN THE BAPTIST": "St. John the Baptist Parish",
        "ST. TAMMANY": "St. Tammany Parish",
        "ST TAMMANY": "St. Tammany Parish",
        "ST. CHARLES": "St. Charles Parish",
        "ST CHARLES": "St. Charles Parish",
        "ST. BERNARD": "St. Bernard Parish",
        "ST BERNARD": "St. Bernard Parish",
        "ST. JAMES": "St. James Parish",
        "ST JAMES": "St. James Parish",
        "ST. MARY": "St. Mary Parish",
        "ST MARY": "St. Mary Parish",
        "ST. LANDRY": "St. Landry Parish",
        "ST LANDRY": "St. Landry Parish",
        "ST. MARTIN": "St. Martin Parish",
        "ST MARTIN": "St. Martin Parish",
        "LA SALLE": "LaSalle Parish",
        "LASALLE": "LaSalle Parish",
        "EAST BATON ROUGE": "East Baton Rouge Parish",
        "WEST BATON ROUGE": "West Baton Rouge Parish",
        "JEFFERSON DAVIS": "Jefferson Davis Parish",
        "POINTE COUPEE": "Pointe Coupee Parish",
        "DE SOTO": "DeSoto Parish",
    }

    if parish in special:
        return special[parish]

    if not parish:
        return ""

    return parish.title() + " Parish"


def clean_event_name(event_name: str) -> str:
    """Clean event name for search query."""
    return normalize_text(event_name).replace("/", " ")


def choose_representative_district(group: pd.DataFrame) -> str:
    """Choose a district name for search."""
    districts = group["target_district"].dropna().astype(str).map(normalize_text).tolist()
    parish_systems = [d for d in districts if is_parish_system_name(d)]

    if parish_systems:
        return pd.Series(parish_systems).value_counts().index[0]

    return ""


def build_queries(row: pd.Series) -> dict:
    """Create compact search queries for one event-parish target."""
    event_name = clean_event_name(row["event_name"])
    storm_names_raw = normalize_text(row.get("storm_names", ""))
    parish_name = row["parish_display"]
    district = normalize_text(row.get("representative_district", ""))
    year = str(int(row["storm_year"]))

    storm_names = [s.strip() for s in storm_names_raw.split(";") if s.strip()]
    primary_storm = storm_names[0] if storm_names else event_name

    if storm_names:
        storm_phrase = f"Hurricane {primary_storm}"
        broader_phrase = " ".join([f'"Hurricane {s}"' for s in storm_names])
    else:
        storm_phrase = event_name
        broader_phrase = event_name

    search_unit = district if district else parish_name

    query_1 = f'"{search_unit}" "schools closed" "{storm_phrase}"'
    query_2 = f'"{search_unit}" "school closure" hurricane {year}'
    query_3 = f'"{parish_name}" "schools closed" "{year}" hurricane'
    query_4 = f'"{parish_name}" "reopen" schools "{primary_storm}"'
    query_5 = f'"{parish_name}" "school board" closure {broader_phrase}'

    return {
        "priority_search_query_1": query_1,
        "priority_search_query_2": query_2,
        "priority_search_query_3": query_3,
        "priority_search_query_4": query_4,
        "priority_search_query_5": query_5,
    }


def prepare_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Clean original queue fields."""
    out = df.copy()

    for col in ["max_exposure", "mean_exposure", "target_schools_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["any_within_100km_hurricane"] = out["any_within_100km_hurricane"].apply(parse_bool_numeric)
    out["priority_rank"] = out["priority"].map(PRIORITY_ORDER).fillna(99).astype(int)
    out["event_rank"] = out["event_id"].map(EVENT_ORDER).fillna(99).astype(int)
    out["target_parish"] = out["target_parish"].map(normalize_text)
    out["target_district"] = out["target_district"].map(normalize_text)

    return out


def collapse_to_event_parish(queue: pd.DataFrame) -> pd.DataFrame:
    """Collapse original queue to one row per event × parish."""
    rows = []

    group_cols = [
        "event_id",
        "event_name",
        "storm_names",
        "storm_year",
        "graduation_year",
        "search_window_start",
        "search_window_end",
        "priority",
        "target_parish",
    ]

    for keys, group in queue.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))

        row["priority_rank"] = int(group["priority_rank"].min())
        row["event_rank"] = int(group["event_rank"].min())
        row["target_rows_collapsed"] = len(group)
        row["approx_target_schools_count"] = int(group["target_schools_count"].fillna(0).sum())
        row["max_exposure"] = group["max_exposure"].max()
        row["mean_exposure"] = group["mean_exposure"].mean()
        row["any_within_100km_hurricane"] = int(group["any_within_100km_hurricane"].fillna(0).max())
        row["representative_district"] = choose_representative_district(group)
        row["parish_display"] = parish_display_name(row["target_parish"])

        raw_districts = [
            d for d in group["target_district"].dropna().astype(str).map(normalize_text).unique()
            if d
        ]
        row["collapsed_districts_or_schools"] = "; ".join(raw_districts[:20])

        row.update(build_queries(pd.Series(row)))

        row["source_url_found"] = ""
        row["source_title_found"] = ""
        row["source_type_found"] = ""
        row["local_file_path"] = ""
        row["collection_status"] = "not_started"
        row["collection_notes"] = ""

        rows.append(row)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        [
            "event_rank",
            "priority_rank",
            "any_within_100km_hurricane",
            "max_exposure",
            "approx_target_schools_count",
        ],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)

    out["priority_queue_id"] = [
        f"PQ_{i+1:04d}_{row.event_id}_{str(row.target_parish).replace(' ', '_').replace('.', '')}"
        for i, row in out.iterrows()
    ]

    first_cols = [
        "priority_queue_id",
        "event_id",
        "event_name",
        "storm_names",
        "storm_year",
        "graduation_year",
        "priority",
        "target_parish",
        "parish_display",
        "representative_district",
        "target_rows_collapsed",
        "approx_target_schools_count",
        "max_exposure",
        "mean_exposure",
        "any_within_100km_hurricane",
        "priority_search_query_1",
        "priority_search_query_2",
        "priority_search_query_3",
        "priority_search_query_4",
        "priority_search_query_5",
    ]

    remaining_cols = [col for col in out.columns if col not in first_cols]

    return out[first_cols + remaining_cols]


def select_top_queue(priority: pd.DataFrame, max_parishes_per_event: int, include_medium: bool) -> pd.DataFrame:
    """Select first-round top queue."""
    allowed_priorities = ["very_high", "high"]

    if include_medium:
        allowed_priorities.append("medium")

    subset = priority[priority["priority"].isin(allowed_priorities)].copy()

    selected_frames = []

    for _, group in subset.groupby("event_id", sort=False):
        selected_frames.append(group.head(max_parishes_per_event).copy())

    if not selected_frames:
        return pd.DataFrame(columns=priority.columns)

    return pd.concat(selected_frames, ignore_index=True)


def make_summary(priority: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    """Summarize queue counts."""
    rows = []

    for label, data in [("all_priority_queue", priority), ("top_collection_queue", top)]:
        grouped = (
            data.groupby(["event_id", "priority"], dropna=False)
            .agg(
                rows=("priority_queue_id", "size"),
                parishes=("target_parish", "nunique"),
                approx_target_schools=("approx_target_schools_count", "sum"),
                max_exposure=("max_exposure", "max"),
            )
            .reset_index()
        )
        grouped["queue_type"] = label
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Run prioritization."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-parishes-per-event",
        type=int,
        default=12,
        help="Maximum event-parish rows to keep per event in the top queue.",
    )
    parser.add_argument(
        "--include-medium",
        action="store_true",
        help="Include medium-priority events in the top queue.",
    )
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input queue: {INPUT_PATH}")

    raw = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)

    queue = prepare_queue(raw)
    priority = collapse_to_event_parish(queue)
    top = select_top_queue(
        priority=priority,
        max_parishes_per_event=args.max_parishes_per_event,
        include_medium=args.include_medium,
    )
    summary = make_summary(priority, top)

    priority.to_csv(OUT_PRIORITY, index=False)
    top.to_csv(OUT_TOP, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("Saved:")
    print(" -", OUT_PRIORITY.relative_to(PROJECT_ROOT))
    print(" -", OUT_TOP.relative_to(PROJECT_ROOT))
    print(" -", OUT_SUMMARY.relative_to(PROJECT_ROOT))

    print()
    print("Original queue rows:", len(raw))
    print("Collapsed event-parish rows:", len(priority))
    print("Top collection rows:", len(top))

    print()
    print("Top queue counts by event:")
    print(top["event_id"].value_counts().to_string())

    print()
    print("Top 40 collection tasks:")
    print(
        top[
            [
                "priority_queue_id",
                "event_id",
                "target_parish",
                "representative_district",
                "approx_target_schools_count",
                "max_exposure",
                "any_within_100km_hurricane",
                "priority_search_query_1",
                "collection_status",
            ]
        ]
        .head(40)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
