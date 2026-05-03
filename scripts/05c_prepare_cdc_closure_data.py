"""
Prepare Louisiana CDC closure records into the standardized closure schema.

Input:
    data/raw/closures/cdc/cdc_prolonged_unplanned_school_closures_2011_2019.csv

Outputs:
    data/intermediate/cdc_louisiana_closure_records_clean.csv
    data/intermediate/cdc_louisiana_closure_records_review.csv
    data/intermediate/cdc_louisiana_closure_summary_by_event.csv
    data/intermediate/cdc_louisiana_closure_summary_by_year.csv

Notes:
    CDC records are official baseline closure records but only cover 2011-2019.
    This script standardizes Louisiana records only.

Main coding:
    closure days = weekdays from dateclosure through the day before datereopened.
    This approximates instructional days, without excluding holidays.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CDC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "closures"
    / "cdc"
    / "cdc_prolonged_unplanned_school_closures_2011_2019.csv"
)

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

CLEAN_OUT = INTERMEDIATE_DIR / "cdc_louisiana_closure_records_clean.csv"
REVIEW_OUT = INTERMEDIATE_DIR / "cdc_louisiana_closure_records_review.csv"
SUMMARY_EVENT_OUT = INTERMEDIATE_DIR / "cdc_louisiana_closure_summary_by_event.csv"
SUMMARY_YEAR_OUT = INTERMEDIATE_DIR / "cdc_louisiana_closure_summary_by_year.csv"


PARISH_ALIASES = {
    "ORLEANS": "ORLEANS",
    "JEFFERSON": "JEFFERSON",
    "ST. BERNARD": "ST. BERNARD",
    "ST BERNARD": "ST. BERNARD",
    "LAFOURCHE": "LAFOURCHE",
    "TERREBONNE": "TERREBONNE",
    "ST. CHARLES": "ST. CHARLES",
    "ST CHARLES": "ST. CHARLES",
    "ST. JOHN THE BAPTIST": "ST. JOHN THE BAPTIST",
    "ST JOHN THE BAPTIST": "ST. JOHN THE BAPTIST",
    "PLAQUEMINES": "PLAQUEMINES",
    "WEBSTER": "WEBSTER",
    "UNION": "UNION",
    "ST. TAMMANY": "ST. TAMMANY",
    "ST TAMMANY": "ST. TAMMANY",
    "CALCASIEU": "CALCASIEU",
    "CAMERON": "CAMERON",
    "EAST BATON ROUGE": "EAST BATON ROUGE",
    "ST. JOHN": "ST. JOHN THE BAPTIST",
    "ST JOHN": "ST. JOHN THE BAPTIST",
}


def normalize_text(value: object) -> str:
    """Normalize text for matching."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_nces_school_id(value: object) -> str:
    """Normalize public NCES school ID if available."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if not text or text.lower() in {"nan", "none"}:
        return ""

    # Public NCES school IDs are usually 12-digit numeric IDs.
    if text.isdigit() and len(text) <= 12:
        return text.zfill(12)

    # Private/nonpublic IDs in CDC may be alphanumeric. Do not treat these as
    # public NCES school IDs.
    return ""


def normalize_nces_district_id(value: object) -> str:
    """Normalize NCES district ID if available."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if not text or text.lower() in {"nan", "none"}:
        return ""

    if text.isdigit():
        return text.zfill(7)

    return ""


def infer_parish_from_district(distname: object) -> str:
    """Infer Louisiana parish from district name when possible."""
    text = normalize_text(distname)

    if not text:
        return ""

    # New Orleans charter / RSD records are generally Orleans-based in this CDC extract.
    if "ORLEANS" in text or text.startswith("RSD-") or "RECOVERY SCHOOL DISTRICT" in text:
        return "ORLEANS"

    for pattern, parish in PARISH_ALIASES.items():
        if pattern in text:
            return parish

    return ""


def infer_closure_scope(row: pd.Series) -> str:
    """Infer closure scope from CDC type field."""
    record_type = normalize_text(row.get("type", ""))

    if record_type == "D":
        return "district_system"

    if record_type == "S":
        return "school_specific"

    return "unclear"


def categorize_reason(row: pd.Series) -> str:
    """Assign broad closure reason category."""
    text = " ".join(
        [
            normalize_text(row.get("eventname", "")),
            normalize_text(row.get("eventtype", "")),
            normalize_text(row.get("eventtype_2", "")),
            normalize_text(row.get("eventdesc", "")),
        ]
    )

    if "COVID" in text:
        return "covid"

    if "HURRICANE" in text:
        return "hurricane"

    if "TROPICAL STORM" in text:
        return "tropical_storm"

    if "ICE" in text or "SNOW" in text:
        return "severe_weather"

    if "FLOOD" in text:
        return "flood"

    if "RAIN" in text or "STORM" in text or "WEATHER" in text:
        return "severe_weather"

    if "FIRE" in text or "BUILDING" in text or "UTILITY" in text:
        return "other"

    if "ENVIRONMENTAL" in text:
        return "other"

    return "unclear"


def infer_storm_name(row: pd.Series) -> str:
    """Infer named storm from event text."""
    text = " ".join(
        [
            normalize_text(row.get("eventname", "")),
            normalize_text(row.get("eventdesc", "")),
        ]
    )

    known_storms = [
        "ISAAC",
        "KATRINA",
        "RITA",
        "GUSTAV",
        "IKE",
        "LAURA",
        "DELTA",
        "ZETA",
        "IDA",
        "FRANCINE",
        "BARRY",
        "LILI",
        "ISIDORE",
        "HARVEY",
    ]

    for storm in known_storms:
        if storm in text:
            return storm.title()

    return ""


def is_hurricane_related(row: pd.Series) -> bool:
    """Return True if closure appears hurricane/tropical-storm related."""
    reason = categorize_reason(row)
    storm_name = infer_storm_name(row)

    if reason in {"hurricane", "tropical_storm"}:
        return True

    if storm_name:
        return True

    return False


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    """Parse CDC date field."""
    if pd.isna(value):
        return pd.NaT

    return pd.to_datetime(value, errors="coerce").normalize()


def count_weekdays_exclusive_reopen(start: pd.Timestamp, reopened: pd.Timestamp) -> int:
    """
    Count weekdays from closure start through the day before reopening.

    Example:
        Closed 2012-08-27, reopened 2012-09-04 means count 2012-08-27
        through 2012-09-03.
    """
    if pd.isna(start) or pd.isna(reopened):
        return 0

    if reopened <= start:
        return 0

    days = pd.date_range(start=start, end=reopened - pd.Timedelta(days=1), freq="D")
    return int(sum(day.weekday() < 5 for day in days))


def infer_graduation_year(start_date: pd.Timestamp) -> int | None:
    """
    Map closure start date to graduation year.

    Academic year window:
        Aug 1 of t-1 through May 31 of t -> graduation_year t
    """
    if pd.isna(start_date):
        return None

    year = int(start_date.year)
    month = int(start_date.month)

    if month >= 8:
        return year + 1

    return year


def build_review_notes(row: pd.Series) -> str:
    """Build notes explaining possible review issues."""
    notes = []

    if row["closure_scope"] == "unclear":
        notes.append("unclear_scope")

    if pd.isna(row["closure_start_date"]) or pd.isna(row["closure_end_date"]):
        notes.append("missing_or_invalid_dates")

    if row["closure_days"] <= 0:
        notes.append("nonpositive_closure_days")

    if row["closure_scope"] == "school_specific" and not row["nces_school_id"]:
        notes.append("school_specific_without_public_nces_id")

    if row["closure_reason_category"] == "unclear":
        notes.append("unclear_reason")

    return "; ".join(notes)


def standardize_cdc_louisiana(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize CDC Louisiana closure records into clean schema."""
    la = df[df["state"].astype(str).str.upper().eq("LA")].copy()

    records = []

    for _, row in la.iterrows():
        start_date = parse_date(row.get("dateclosure"))
        reopened_date = parse_date(row.get("datereopened"))
        closure_days = count_weekdays_exclusive_reopen(start_date, reopened_date)

        closure_scope = infer_closure_scope(row)
        reason_category = categorize_reason(row)
        storm_name = infer_storm_name(row)
        hurricane_related = is_hurricane_related(row)

        nces_school_id = normalize_nces_school_id(row.get("schoolncesid"))
        nces_district_id = normalize_nces_district_id(row.get("districtncesid"))

        graduation_year = infer_graduation_year(start_date)
        academic_year_start = graduation_year - 1 if graduation_year else None

        clean = {
            "closure_record_id": f"CDC_{row.get('uniqueid')}",
            "raw_record_id": row.get("uniqueid", ""),
            "source_id": "CDC_PUSC_BASELINE",
            "source_type": "official_dataset",
            "source_url_or_file": str(RAW_CDC_PATH.relative_to(PROJECT_ROOT)),
            "source_title": "CDC Prolonged Unplanned School Closures: USA, 2011-2019",
            "source_publication_date": "",
            "closure_scope": closure_scope,
            "state": "LA",
            "parish": infer_parish_from_district(row.get("distname")),
            "district_name": row.get("distname", ""),
            "districtncesid_raw": row.get("districtncesid", ""),
            "nces_district_id": nces_district_id,
            "school_name": row.get("schoolname", ""),
            "schoolncesid_raw": row.get("schoolncesid", ""),
            "nces_school_id": nces_school_id,
            "nces_school_name": row.get("schoolname", ""),
            "closure_start_date": start_date.date().isoformat() if pd.notna(start_date) else "",
            "closure_end_date": reopened_date.date().isoformat() if pd.notna(reopened_date) else "",
            "closure_days": closure_days,
            "academic_year_start": academic_year_start,
            "graduation_year": graduation_year,
            "closure_reason_raw": " | ".join(
                [
                    str(row.get("eventname", "")),
                    str(row.get("eventtype", "")),
                    str(row.get("eventtype_2", "")),
                    str(row.get("eventdesc", "")),
                ]
            ),
            "closure_reason_category": reason_category,
            "storm_name": storm_name,
            "hurricane_related": hurricane_related,
            "instructional_closure": closure_days > 0,
            "include_in_main_closure_measure": (
                hurricane_related
                and closure_days > 0
                and closure_scope in {"district_system", "school_specific"}
            ),
            "confidence": 0.95 if hurricane_related and closure_days > 0 else 0.80,
            "needs_manual_review": False,
            "review_notes": "",
            "eventname": row.get("eventname", ""),
            "eventtype": row.get("eventtype", ""),
            "eventtype_2": row.get("eventtype_2", ""),
            "eventdesc": row.get("eventdesc", ""),
            "distweb": row.get("distweb", ""),
            "schoolweb": row.get("schoolweb", ""),
        }

        records.append(clean)

    out = pd.DataFrame(records)

    if out.empty:
        return out

    out["review_notes"] = out.apply(build_review_notes, axis=1)
    out["needs_manual_review"] = out["review_notes"].astype(str).str.len().gt(0)

    # Do not include records with serious review issues in main closure measure.
    out.loc[out["needs_manual_review"], "include_in_main_closure_measure"] = False

    return out


def main() -> None:
    """Prepare CDC Louisiana closure data."""
    if not RAW_CDC_PATH.exists():
        raise FileNotFoundError(f"Missing CDC raw file: {RAW_CDC_PATH}")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_CDC_PATH, dtype=str, low_memory=False)

    clean = standardize_cdc_louisiana(raw)
    clean.to_csv(CLEAN_OUT, index=False)

    review = clean[clean["needs_manual_review"]].copy()
    review.to_csv(REVIEW_OUT, index=False)

    summary_event = (
        clean.groupby(
            [
                "eventname",
                "closure_reason_category",
                "storm_name",
                "hurricane_related",
                "include_in_main_closure_measure",
            ],
            dropna=False,
        )
        .agg(
            records=("closure_record_id", "size"),
            total_closure_days=("closure_days", "sum"),
            district_records=("closure_scope", lambda x: (x == "district_system").sum()),
            school_records=("closure_scope", lambda x: (x == "school_specific").sum()),
        )
        .reset_index()
        .sort_values(["hurricane_related", "records"], ascending=[False, False])
    )
    summary_event.to_csv(SUMMARY_EVENT_OUT, index=False)

    summary_year = (
        clean.groupby(
            [
                "graduation_year",
                "hurricane_related",
                "include_in_main_closure_measure",
            ],
            dropna=False,
        )
        .agg(
            records=("closure_record_id", "size"),
            total_closure_days=("closure_days", "sum"),
        )
        .reset_index()
        .sort_values(["graduation_year", "hurricane_related"])
    )
    summary_year.to_csv(SUMMARY_YEAR_OUT, index=False)

    print("Saved:")
    print(" -", CLEAN_OUT.relative_to(PROJECT_ROOT))
    print(" -", REVIEW_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_EVENT_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_YEAR_OUT.relative_to(PROJECT_ROOT))

    print()
    print("CDC Louisiana records:", len(clean))
    print("Manual review records:", len(review))

    print()
    print("Closure scope counts:")
    print(clean["closure_scope"].value_counts(dropna=False).to_string())

    print()
    print("Reason category counts:")
    print(clean["closure_reason_category"].value_counts(dropna=False).to_string())

    print()
    print("Hurricane-related counts:")
    print(clean["hurricane_related"].value_counts(dropna=False).to_string())

    print()
    print("Included in main closure measure:")
    print(clean["include_in_main_closure_measure"].value_counts(dropna=False).to_string())

    print()
    print("Summary by event:")
    print(summary_event.to_string(index=False))

    print()
    print("Review rows:")
    if review.empty:
        print("None")
    else:
        print(
            review[
                [
                    "closure_record_id",
                    "eventname",
                    "closure_scope",
                    "district_name",
                    "school_name",
                    "closure_start_date",
                    "closure_end_date",
                    "closure_days",
                    "review_notes",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
