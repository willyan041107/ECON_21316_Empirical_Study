"""
Refine filtered auto-collected closure sources before final extraction.

Input:
    data/intermediate/closure_auto_source_screening_filtered.csv

Outputs:
    data/intermediate/closure_auto_source_screening_refined.csv
    data/intermediate/closure_auto_source_refine_summary.csv

Purpose:
    05f2 keeps sources matching target storm and year, but some sources are still noisy.
    This script removes weak domains, obvious non-source pages, and sources with
    closure-date guesses far outside the target search window.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_source_screening_filtered.csv"

REFINED_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_source_screening_refined.csv"
SUMMARY_OUT = PROJECT_ROOT / "data" / "intermediate" / "closure_auto_source_refine_summary.csv"


BLOCKED_DOMAINS = {
    "wikipedia.org",
    "en.wikipedia.org",
    "kids.kiddle.co",
    "ground.news",
}

LOW_VALUE_DOMAINS = {
    "newsweek.com",
}

SOURCE_PRIORITY_DOMAINS = {
    "wdsu.com",
    "nola.com",
    "katc.com",
    "kplctv.com",
    "wafb.com",
    "fox8live.com",
    "theadvocate.com",
    "theadvocate.com",
    "houmatoday.com",
    "houmatimes.com",
    "theadvertiser.com",
    "thetownTalk.com",
    "thetowntalk.com",
    "ktbs.com",
    "ksla.com",
    "wrkf.org",
    "wbrz.com",
    "weeklycitizen.com",
    "donaldsonvillechief.com",
    "assumptionschools.com",
    "nolapublicschools.com",
    "wearescpps.org",
    "tpsd.org",
}


def get_domain(url: object) -> str:
    """Extract normalized domain from URL."""
    if pd.isna(url):
        return ""

    domain = urlparse(str(url)).netloc.lower()
    domain = domain.removeprefix("www.")

    return domain


def parse_bool(value: object) -> bool:
    """Parse boolean-like values."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    """Parse date safely."""
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return pd.NaT

    return pd.to_datetime(text, errors="coerce").normalize()


def date_within_event_window(row: pd.Series) -> bool:
    """
    Check whether extracted date guesses are plausible for the target event.

    Rule:
    - If no dates were extracted, do not reject here.
    - If dates exist, at least one date should fall within the event search window
      plus a buffer.
    - For Katrina/Rita 2005 and Ida 2021, allow longer reopening tails.
    """
    start_window = parse_date(row.get("queue_search_window_start", ""))
    end_window = parse_date(row.get("queue_search_window_end", ""))

    if pd.isna(start_window) or pd.isna(end_window):
        return True

    event_id = str(row.get("target_event_id", ""))

    if event_id == "2005_katrina_rita":
        buffer_before = 10
        buffer_after = 240
    elif event_id == "2021_ida":
        buffer_before = 10
        buffer_after = 90
    elif event_id == "2020_laura_delta_zeta":
        buffer_before = 10
        buffer_after = 90
    else:
        buffer_before = 10
        buffer_after = 45

    lower = start_window - pd.Timedelta(days=buffer_before)
    upper = end_window + pd.Timedelta(days=buffer_after)

    extracted_dates = []

    for col in ["closure_start_date_guess", "closure_end_date_guess"]:
        date = parse_date(row.get(col, ""))
        if pd.notna(date):
            extracted_dates.append(date)

    if not extracted_dates:
        return True

    return any(lower <= date <= upper for date in extracted_dates)


def should_block_domain(row: pd.Series) -> bool:
    """Return True if source domain should be excluded."""
    domain = get_domain(row.get("url", ""))

    if domain in BLOCKED_DOMAINS:
        return True

    if any(domain.endswith("." + blocked) for blocked in BLOCKED_DOMAINS):
        return True

    return False


def looks_like_school_closure_source(row: pd.Series) -> bool:
    """
    Keep sources that are likely useful for closure extraction.

    This uses OpenRouter screening metadata and URL/title context.
    """
    usable = parse_bool(row.get("keep_for_extraction", False))

    if not usable:
        return False

    confidence = pd.to_numeric(row.get("confidence", 0), errors="coerce")

    if pd.isna(confidence):
        confidence = 0

    if confidence < 0.85:
        return False

    closure_scope = str(row.get("closure_scope", "")).strip().lower()

    if closure_scope not in {
        "district_system",
        "parishwide",
        "school_specific",
        "statewide",
    }:
        return False

    return True


def classify_refine_decision(row: pd.Series) -> tuple[bool, str]:
    """Classify whether to keep source after refinement."""
    if not parse_bool(row.get("keep_for_extraction", False)):
        return False, "not_kept_by_05f2"

    if should_block_domain(row):
        return False, "blocked_domain"

    if not looks_like_school_closure_source(row):
        return False, "low_confidence_or_bad_scope"

    if not date_within_event_window(row):
        return False, "date_outside_event_window"

    domain = get_domain(row.get("url", ""))

    if domain in LOW_VALUE_DOMAINS:
        return False, "low_value_domain"

    if parse_bool(row.get("needs_manual_review_after_filter", False)):
        return True, "keep_manual_review"

    return True, "keep"


def main() -> None:
    """Refine filtered sources."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)

    decisions = df.apply(classify_refine_decision, axis=1)

    out = df.copy()
    out["source_domain"] = out["url"].apply(get_domain)
    out["refined_keep_for_extraction"] = [decision[0] for decision in decisions]
    out["refine_reason"] = [decision[1] for decision in decisions]

    out.to_csv(REFINED_OUT, index=False)

    summary = (
        out.groupby(["refined_keep_for_extraction", "refine_reason"], dropna=False)
        .agg(
            rows=("candidate_id", "size"),
            unique_urls=("url", "nunique"),
        )
        .reset_index()
        .sort_values(["refined_keep_for_extraction", "rows"], ascending=[False, False])
    )

    summary.to_csv(SUMMARY_OUT, index=False)

    kept = out[out["refined_keep_for_extraction"]].copy()

    print("Saved:")
    print(" -", REFINED_OUT.relative_to(PROJECT_ROOT))
    print(" -", SUMMARY_OUT.relative_to(PROJECT_ROOT))

    print()
    print("05f2 kept rows:", int(out["keep_for_extraction"].astype(str).str.lower().eq("true").sum()))
    print("Refined kept rows:", len(kept))

    print()
    print("Refine summary:")
    print(summary.to_string(index=False))

    print()
    print("Refined kept sources by event:")
    if kept.empty:
        print("None")
    else:
        print(kept["target_event_id"].value_counts().to_string())

    print()
    print("Top refined kept sources:")
    if kept.empty:
        print("None")
    else:
        cols = [
            "candidate_id",
            "target_event_id",
            "queue_target_parish",
            "expected_storms",
            "found_storms",
            "closure_start_date_guess",
            "closure_end_date_guess",
            "confidence",
            "source_domain",
            "refine_reason",
            "url",
        ]
        existing_cols = [col for col in cols if col in kept.columns]
        print(kept[existing_cols].head(80).to_string(index=False))


if __name__ == "__main__":
    main()
