"""
Extract structured closure records from filtered auto-collected sources.

Inputs:
    data/intermediate/closure_auto_source_screening_filtered.csv
    data/raw/closures/auto_sources/*.txt

Outputs:
    data/intermediate/closure_auto_extracted_records_raw.csv
    data/intermediate/closure_auto_extracted_records_clean.csv
    data/intermediate/closure_auto_extraction_review.csv
    data/intermediate/closure_auto_extraction_summary.csv

Purpose:
    Convert retained source texts into standardized closure records.

This script extracts closure records only.
It does not yet expand district/parish-level closures to school-year panel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILTERED_PATH = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_source_screening_refined.csv"

RAW_RECORDS_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_extracted_records_raw.csv"
CLEAN_RECORDS_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_extracted_records_clean.csv"
REVIEW_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_extraction_review.csv"
SUMMARY_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_extraction_summary.csv"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def load_project_env() -> None:
    """Load project-level .env."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)


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


def count_weekdays_inclusive(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """
    Count weekdays from start_date through end_date inclusive.

    If source says schools closed on Oct. 15 and Oct. 16, count 2 weekdays.
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return 0

    if end_date < start_date:
        return 0

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return int(sum(date.weekday() < 5 for date in dates))


def infer_graduation_year(date: pd.Timestamp) -> int | None:
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


def normalize_text(value: object) -> str:
    """Normalize text values."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def read_source_text(local_file_path: str) -> str:
    """Read local source text."""
    path = PROJECT_ROOT / local_file_path

    if not path.exists():
        raise FileNotFoundError(f"Missing source text file: {path}")

    return path.read_text(encoding="utf-8", errors="ignore")


def call_openrouter_extract(
    source_text: str,
    metadata: dict,
    api_key: str,
    model: str,
) -> dict:
    """Call OpenRouter to extract closure records."""
    clipped = source_text[:18000]

    system_prompt = """
You extract structured K-12 school closure records from source text.

Return only valid JSON. Do not include markdown.

Rules:
- Extract only closures explicitly stated in the source text.
- Do not invent dates, parishes, districts, or school names.
- General emergency declarations are not enough unless school closure, cancellation, remote learning, or reopening is explicitly stated.
- If a source lists multiple parishes/districts, return one record per parish/district when possible.
- If exact closure end date is not stated, leave closure_end_date empty and set needs_manual_review true.
- Use Louisiana K-12 school/district/parish-school-system context.
- For source evidence, provide a short paraphrase or quote under 40 words.
"""

    user_payload = {
        "task": "Extract hurricane-related Louisiana K-12 school closure records.",
        "target_metadata": metadata,
        "return_schema": {
            "records": [
                {
                    "closure_scope": "district_system | parishwide | school_specific | statewide | unclear",
                    "state": "LA",
                    "parish": "parish name if stated or strongly implied",
                    "district_name": "district / school system name if stated",
                    "school_name": "school name if school-specific, otherwise empty",
                    "closure_start_date": "YYYY-MM-DD or empty",
                    "closure_end_date": "YYYY-MM-DD or empty",
                    "reopen_date": "YYYY-MM-DD or empty",
                    "closure_reason_raw": "reason text from source",
                    "closure_reason_category": "hurricane | tropical_storm | severe_weather | flood | power_outage | other | unclear",
                    "storm_name": "named storm if stated",
                    "hurricane_related": "boolean",
                    "instructional_closure": "boolean",
                    "remote_learning": "boolean",
                    "evidence_text": "short source evidence under 40 words",
                    "confidence": "number from 0 to 1",
                    "needs_manual_review": "boolean",
                    "review_notes": "short notes if ambiguous",
                }
            ],
            "source_level_notes": "short summary",
        },
        "source_text": clipped,
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ECON_21316_Empirical_Study",
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return json.loads(content)


def parse_expected_storms(value: object) -> set[str]:
    """Parse expected storm names from filtered source metadata."""
    if pd.isna(value):
        return set()

    text = str(value).upper().strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return set()

    parts = re.split(r"[;,]", text)

    cleaned = set()
    for part in parts:
        storm = normalize_storm_name(part)
        if storm:
            cleaned.add(storm)

    return cleaned


def normalize_storm_name(value: object) -> str:
    """Normalize storm name for matching."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = text.replace("HURRICANE", "")
    text = text.replace("TROPICAL STORM", "")
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def storm_matches_expected(row: pd.Series) -> bool:
    """Return True if extracted storm name matches the target event storms."""
    expected = parse_expected_storms(row.get("expected_storms", ""))
    found = normalize_storm_name(row.get("storm_name", ""))

    if not expected or not found:
        return False

    return any(found == storm or found in storm or storm in found for storm in expected)


def evidence_suggests_single_day_closure(row: pd.Series) -> bool:
    """
    Return True when the source evidence likely describes a one-day closure.

    This is intentionally conservative. It only helps when the start date exists
    but the end date is missing.
    """
    evidence = " ".join(
        [
            str(row.get("evidence_text", "")),
            str(row.get("closure_reason_raw", "")),
            str(row.get("source_level_notes", "")),
        ]
    ).lower()

    single_day_patterns = [
        "closed today",
        "closed on",
        "closed wednesday",
        "closed thursday",
        "closed friday",
        "closed monday",
        "closed tuesday",
        "schools closed wednesday",
        "schools closed thursday",
        "schools closed friday",
        "classes canceled",
        "classes cancelled",
        "school canceled",
        "school cancelled",
        "campuses closed",
        "facilities closed",
    ]

    multi_day_warning_patterns = [
        "until further notice",
        "remain closed",
        "closed through",
        "closed until",
        "reopen",
        "reopened",
        "will reopen",
    ]

    if any(pattern in evidence for pattern in multi_day_warning_patterns):
        return False

    return any(pattern in evidence for pattern in single_day_patterns)


def should_infer_single_day_closure(row: pd.Series) -> bool:
    """Decide whether to infer closure_end_date = closure_start_date."""
    start = parse_date(row.get("closure_start_date", ""))
    end = parse_date(row.get("closure_end_date", ""))

    if pd.isna(start) or pd.notna(end):
        return False

    if not parse_bool(row.get("hurricane_related", False)):
        return False

    if not parse_bool(row.get("instructional_closure", False)):
        return False

    if not storm_matches_expected(row):
        return False

    try:
        confidence = float(row.get("confidence", 0) or 0)
    except Exception:
        confidence = 0

    if confidence < 0.85:
        return False

    return evidence_suggests_single_day_closure(row)


def make_review_notes(row: pd.Series) -> str:
    """Create deterministic review notes."""
    notes = []

    if not normalize_text(row.get("parish", "")):
        notes.append("missing_parish")

    if not normalize_text(row.get("district_name", "")) and row.get("closure_scope") in {
        "district_system",
        "parishwide",
    }:
        notes.append("missing_district_name")

    if pd.isna(parse_date(row.get("closure_start_date", ""))):
        notes.append("missing_or_invalid_start_date")

    if pd.isna(parse_date(row.get("closure_end_date", ""))):
        notes.append("missing_or_invalid_end_date")

    if not storm_matches_expected(row):
        notes.append("storm_mismatch")

    if not parse_bool(row.get("hurricane_related", False)):
        notes.append("not_hurricane_related")

    if not parse_bool(row.get("instructional_closure", False)):
        notes.append("not_instructional_closure")

    try:
        confidence = float(row.get("confidence", 0) or 0)
    except Exception:
        confidence = 0

    if confidence < 0.75:
        notes.append("low_confidence")

    if parse_bool(row.get("needs_manual_review", False)):
        notes.append("api_flagged_manual_review")

    return "; ".join(notes)


def clean_extracted_records(raw_records: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich extracted closure records."""
    if raw_records.empty:
        return raw_records

    df = raw_records.copy()

    for col in [
        "closure_scope",
        "state",
        "parish",
        "district_name",
        "school_name",
        "closure_reason_raw",
        "closure_reason_category",
        "storm_name",
        "evidence_text",
        "review_notes",
    ]:
        if col not in df.columns:
            df[col] = ""

        df[col] = df[col].fillna("").astype(str).map(normalize_text)

    for col in [
        "hurricane_related",
        "instructional_closure",
        "remote_learning",
        "needs_manual_review",
    ]:
        if col not in df.columns:
            df[col] = False

        df[col] = df[col].apply(parse_bool)

    if "confidence" not in df.columns:
        df["confidence"] = 0

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)

    for col in ["closure_start_date", "closure_end_date", "reopen_date"]:
        if col not in df.columns:
            df[col] = ""

        parsed = df[col].apply(parse_date)
        df[col] = parsed.apply(lambda x: x.date().isoformat() if pd.notna(x) else "")

    # If closure_end_date missing but reopen_date available, use day before reopen_date
    # only when source clearly describes reopening after closure.
    for idx, row in df.iterrows():
        start = parse_date(row.get("closure_start_date", ""))
        end = parse_date(row.get("closure_end_date", ""))
        reopen = parse_date(row.get("reopen_date", ""))

        if pd.isna(end) and pd.notna(reopen):
            inferred_end = reopen - pd.Timedelta(days=1)
            if pd.notna(start) and inferred_end >= start:
                df.loc[idx, "closure_end_date"] = inferred_end.date().isoformat()

    # Conservative single-day closure inference:
    # if the source says schools/classes were closed on a specific date but no
    # end date is stated, treat it as a one-school-day closure.
    df["single_day_closure_inferred"] = False

    for idx, row in df.iterrows():
        if should_infer_single_day_closure(row):
            start = parse_date(row.get("closure_start_date", ""))
            df.loc[idx, "closure_end_date"] = start.date().isoformat()
            df.loc[idx, "single_day_closure_inferred"] = True

    df["closure_days"] = 0
    df["academic_year_start"] = None
    df["graduation_year"] = None

    for idx, row in df.iterrows():
        start = parse_date(row.get("closure_start_date", ""))
        end = parse_date(row.get("closure_end_date", ""))

        df.loc[idx, "closure_days"] = count_weekdays_inclusive(start, end)

        grad_year = infer_graduation_year(start)
        df.loc[idx, "graduation_year"] = grad_year
        df.loc[idx, "academic_year_start"] = grad_year - 1 if grad_year else None

    df["deterministic_review_notes"] = df.apply(make_review_notes, axis=1)

    df["storm_matches_expected"] = df.apply(storm_matches_expected, axis=1)

    serious_review_terms = [
        "missing_parish",
        "missing_district_name",
        "missing_or_invalid_start_date",
        "missing_or_invalid_end_date",
        "storm_mismatch",
        "not_hurricane_related",
        "not_instructional_closure",
        "low_confidence",
    ]

    def has_serious_review_issue(notes: object) -> bool:
        """Return True if deterministic notes contain serious exclusion reasons."""
        text = str(notes)
        return any(term in text for term in serious_review_terms)

    df["has_serious_review_issue"] = df["deterministic_review_notes"].apply(has_serious_review_issue)

    df["include_in_main_closure_measure"] = (
        df["hurricane_related"]
        & df["instructional_closure"]
        & df["storm_matches_expected"]
        & df["closure_days"].gt(0)
        & df["closure_scope"].isin(["district_system", "parishwide", "school_specific"])
        & ~df["has_serious_review_issue"]
    )

    df["auto_closure_record_id"] = [
        f"AUTO_CLOSURE_{i+1:05d}" for i in range(len(df))
    ]

    first_cols = [
        "auto_closure_record_id",
        "candidate_id",
        "priority_queue_id",
        "target_event_id",
        "target_parish",
        "url",
        "local_file_path",
        "closure_scope",
        "state",
        "parish",
        "district_name",
        "school_name",
        "closure_start_date",
        "closure_end_date",
        "reopen_date",
        "closure_days",
        "academic_year_start",
        "graduation_year",
        "closure_reason_raw",
        "closure_reason_category",
        "storm_name",
        "hurricane_related",
        "instructional_closure",
        "remote_learning",
        "confidence",
        "needs_manual_review",
        "deterministic_review_notes",
        "include_in_main_closure_measure",
        "evidence_text",
    ]

    remaining_cols = [col for col in df.columns if col not in first_cols]

    return df[first_cols + remaining_cols]


def main() -> None:
    """Run extraction from filtered sources."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Maximum kept sources to extract.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between OpenRouter calls.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override OPENROUTER_MODEL.",
    )

    args = parser.parse_args()

    load_project_env()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = args.model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

    if not FILTERED_PATH.exists():
        raise FileNotFoundError(f"Missing filtered sources file: {FILTERED_PATH}")

    filtered = pd.read_csv(FILTERED_PATH, dtype=str, low_memory=False)
    keep_col = "refined_keep_for_extraction"
    if keep_col not in filtered.columns:
        keep_col = "keep_for_extraction"

    kept = filtered[filtered[keep_col].astype(str).str.lower().eq("true")].copy()

    if args.max_sources is not None:
        kept = kept.head(args.max_sources).copy()

    raw_rows = []

    print("Sources kept for extraction:", len(kept))

    for _, source in kept.iterrows():
        candidate_id = source["candidate_id"]
        local_file_path = source["local_file_path"]

        print()
        print("Extracting:", candidate_id)
        print("Source:", local_file_path)

        try:
            source_text = read_source_text(local_file_path)

            metadata = {
                "candidate_id": candidate_id,
                "priority_queue_id": source.get("priority_queue_id", ""),
                "target_event_id": source.get("target_event_id", source.get("event_id", "")),
                "target_parish": source.get("queue_target_parish", source.get("target_parish", "")),
                "expected_storms": source.get("expected_storms", ""),
                "found_storms": source.get("found_storms", ""),
                "target_storm_year": source.get("target_storm_year", ""),
                "url": source.get("url", ""),
                "local_file_path": local_file_path,
            }

            extraction = call_openrouter_extract(
                source_text=source_text,
                metadata=metadata,
                api_key=api_key,
                model=model,
            )

            records = extraction.get("records", [])

            if not isinstance(records, list):
                records = []

            print("Records extracted:", len(records))

            for record_idx, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    continue

                row = {
                    "candidate_id": candidate_id,
                    "priority_queue_id": source.get("priority_queue_id", ""),
                    "target_event_id": metadata["target_event_id"],
                    "target_parish": metadata["target_parish"],
                    "expected_storms": metadata["expected_storms"],
                    "found_storms": metadata["found_storms"],
                    "target_storm_year": metadata["target_storm_year"],
                    "url": source.get("url", ""),
                    "local_file_path": local_file_path,
                    "source_record_index": record_idx,
                    "source_level_notes": extraction.get("source_level_notes", ""),
                    **record,
                }

                raw_rows.append(row)

        except Exception as exc:
            print("Extraction failed:", type(exc).__name__, exc)
            raw_rows.append(
                {
                    "candidate_id": candidate_id,
                    "priority_queue_id": source.get("priority_queue_id", ""),
                    "target_event_id": source.get("target_event_id", source.get("event_id", "")),
                    "target_parish": source.get("queue_target_parish", source.get("target_parish", "")),
                    "url": source.get("url", ""),
                    "local_file_path": local_file_path,
                    "source_record_index": "",
                    "extraction_error": f"{type(exc).__name__}: {exc}",
                }
            )

        time.sleep(args.sleep)

    raw_records = pd.DataFrame(raw_rows)
    raw_records.to_csv(RAW_RECORDS_OUT, index=False)

    clean_records = clean_extracted_records(raw_records)
    clean_records.to_csv(CLEAN_RECORDS_OUT, index=False)

    if clean_records.empty:
        review = clean_records
    else:
        review = clean_records[
            ~clean_records["include_in_main_closure_measure"]
            | clean_records["deterministic_review_notes"].astype(str).str.len().gt(0)
        ].copy()

    review.to_csv(REVIEW_OUT, index=False)

    if clean_records.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            clean_records.groupby(
                [
                    "target_event_id",
                    "parish",
                    "storm_name",
                    "closure_reason_category",
                    "include_in_main_closure_measure",
                ],
                dropna=False,
            )
            .agg(
                records=("auto_closure_record_id", "size"),
                total_closure_days=("closure_days", "sum"),
                min_start_date=("closure_start_date", "min"),
                max_end_date=("closure_end_date", "max"),
                mean_confidence=("confidence", "mean"),
            )
            .reset_index()
            .sort_values(["target_event_id", "parish", "storm_name"])
        )

    summary.to_csv(SUMMARY_OUT, index=False)

    print()
    print("Saved:")
    print(" -", RAW_RECORDS_OUT.relative_to(PROJECT_ROOT))
    print(" -", CLEAN_RECORDS_OUT.relative_to(PROJECT_ROOT))
    print(" -", REVIEW_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_OUT.relative_to(PROJECT_ROOT))

    print()
    print("Raw extracted records:", len(raw_records))
    print("Clean extracted records:", len(clean_records))
    print("Included in main measure:", int(clean_records["include_in_main_closure_measure"].sum()) if not clean_records.empty else 0)
    print("Review records:", len(review))

    print()
    print("Summary:")
    if summary.empty:
        print("None")
    else:
        print(summary.to_string(index=False))

    print()
    print("Review rows:")
    if review.empty:
        print("None")
    else:
        cols = [
            "auto_closure_record_id",
            "target_event_id",
            "target_parish",
            "parish",
            "district_name",
            "storm_name",
            "closure_start_date",
            "closure_end_date",
            "closure_days",
            "deterministic_review_notes",
            "url",
        ]
        print(review[cols].to_string(index=False))


if __name__ == "__main__":
    main()
