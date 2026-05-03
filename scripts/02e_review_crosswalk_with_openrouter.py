"""
Use OpenRouter API to review ambiguous LOSFA-to-NCES crosswalk cases.

Inputs:
    data/intermediate/losfa_nces_crosswalk_review_priority.csv
    data/intermediate/losfa_nces_crosswalk_candidates.csv
    .env with OPENROUTER_API_KEY

Output:
    data/intermediate/losfa_nces_crosswalk_api_review.csv

Important:
    The API is not allowed to invent NCES IDs.
    It must choose only from the provided candidate list, or return no_match / unsure / historical_closed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

REVIEW_PATH = INTERMEDIATE_DIR / "losfa_nces_crosswalk_review_priority.csv"
CANDIDATES_PATH = INTERMEDIATE_DIR / "losfa_nces_crosswalk_candidates.csv"
OUTPUT_PATH = INTERMEDIATE_DIR / "losfa_nces_crosswalk_api_review.csv"


def get_client() -> OpenAI:
    """Create OpenRouter client."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        raise ValueError("OPENROUTER_API_KEY is missing. Please paste it into .env.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def build_prompt(row: pd.Series, candidates: pd.DataFrame) -> str:
    """Build a strict review prompt."""
    candidate_records = []

    candidates = candidates.copy()
    candidates["nces_school_id"] = candidates["nces_school_id"].astype(str)

    for _, cand in candidates.iterrows():
        candidate_records.append(
            {
                "candidate_rank": int(cand["candidate_rank"]),
                "nces_school_id": str(cand["nces_school_id"]),
                "nces_school_name": str(cand["nces_school_name"]),
                "nces_parish": str(cand["nces_parish"]),
                "nces_district_name": str(cand["nces_district_name"]),
                "grade_low": str(cand.get("grade_low", "")),
                "grade_high": str(cand.get("grade_high", "")),
                "match_score": float(cand["match_score"]),
                "latitude": str(cand.get("latitude", "")),
                "longitude": str(cand.get("longitude", "")),
            }
        )

    payload = {
        "task": "Review whether a LOSFA school identity matches one of the provided NCES public school candidates.",
        "strict_rules": [
            "Do not invent NCES school IDs.",
            "Choose a match only from the provided candidates.",
            "You may select a lower-ranked candidate if it is clearly better than the top-ranked candidate.",
            "Jr./Sr. High School, Senior High School, and High School can refer to the same institution when parish/name align and the NCES school serves grade 12.",
            "Magnet, academy, charter, and abbreviated names may reflect renaming or official wording changes; do not reject solely because of these words.",
            "Do not reject solely because the NCES school serves grades 6-12 or 7-12; it can still be the high-school institution.",
            "If the LOSFA school is explicitly closed/historical and no current candidate is a clear continuation, return historical_closed.",
            "If the candidate is only a weak guess, return unsure.",
            "Return JSON only.",
        ],
        "losfa_school": {
            "losfa_school_key": str(row["losfa_school_key"]),
            "losfa_parish": str(row["losfa_parish"]),
            "losfa_hs_name": str(row["losfa_hs_name"]),
            "losfa_hs_name_variants": str(row.get("losfa_hs_name_variants", "")),
            "losfa_hs_type_history": str(row.get("losfa_hs_type_history", "")),
            "first_year": str(row.get("first_year", "")),
            "last_year": str(row.get("last_year", "")),
            "years_observed": str(row.get("years_observed", "")),
            "total_students_processed": str(row.get("total_students_processed", "")),
        },
        "provided_candidates": candidate_records,
        "allowed_output_schema": {
            "decision": "match | no_match | unsure | historical_closed",
            "selected_nces_school_id": "exact NCES ID string from provided_candidates, or empty string",
            "selected_candidate_rank": "candidate rank as integer, or null",
            "confidence": "number between 0 and 1",
            "reason": "brief explanation",
        },
    }

    return json.dumps(payload, ensure_ascii=False)


def parse_json_response(text: str) -> dict:
    """Parse JSON response safely."""
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "decision": "parse_error",
            "selected_nces_school_id": "",
            "selected_candidate_rank": None,
            "confidence": 0,
            "reason": text[:500],
        }


def load_existing_output() -> pd.DataFrame:
    """Load existing output if present so the script can resume."""
    if OUTPUT_PATH.exists():
        return pd.read_csv(OUTPUT_PATH, dtype=str, low_memory=False)

    return pd.DataFrame()


def main() -> None:
    """Run API review."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-score",
        type=float,
        default=90.0,
        help="Only review rows with match_score >= this value.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Maximum rows to review in one run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls.",
    )
    args = parser.parse_args()

    client = get_client()
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    review = pd.read_csv(REVIEW_PATH, dtype=str, low_memory=False)
    candidates = pd.read_csv(CANDIDATES_PATH, dtype=str, low_memory=False)

    review["match_score_numeric"] = pd.to_numeric(review["match_score"], errors="coerce")
    review["years_observed_numeric"] = pd.to_numeric(review["years_observed"], errors="coerce")
    review["total_students_processed_numeric"] = pd.to_numeric(
        review["total_students_processed"], errors="coerce"
    )

    todo = review[review["match_score_numeric"] >= args.min_score].copy()

    todo = todo.sort_values(
        ["years_observed_numeric", "total_students_processed_numeric", "match_score_numeric"],
        ascending=[False, False, False],
    )

    existing = load_existing_output()
    completed_keys = set(existing["losfa_school_key"].astype(str)) if not existing.empty else set()

    todo = todo[~todo["losfa_school_key"].astype(str).isin(completed_keys)].copy()
    todo = todo.head(args.max_rows)

    print("Model:", model)
    print("Rows selected for API review:", len(todo))
    print("Existing completed rows:", len(completed_keys))
    print("Output:", OUTPUT_PATH)

    output_rows = []

    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        key = str(row["losfa_school_key"])

        cand = candidates[candidates["losfa_school_key"].astype(str).eq(key)].copy()
        cand["candidate_rank_numeric"] = pd.to_numeric(cand["candidate_rank"], errors="coerce")
        cand = cand.sort_values("candidate_rank_numeric").head(5)

        prompt = build_prompt(row, cand)

        print(f"[{i}/{len(todo)}] Reviewing: {row['losfa_parish']} | {row['losfa_hs_name']}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful data linkage reviewer. Return valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
            )

            content = response.choices[0].message.content
            parsed = parse_json_response(content)

        except Exception as exc:
            parsed = {
                "decision": "api_error",
                "selected_nces_school_id": "",
                "selected_candidate_rank": None,
                "confidence": 0,
                "reason": str(exc),
            }

        output = {
            "losfa_school_key": key,
            "losfa_parish": row.get("losfa_parish", ""),
            "losfa_hs_name": row.get("losfa_hs_name", ""),
            "losfa_hs_name_variants": row.get("losfa_hs_name_variants", ""),
            "first_year": row.get("first_year", ""),
            "last_year": row.get("last_year", ""),
            "years_observed": row.get("years_observed", ""),
            "total_students_processed": row.get("total_students_processed", ""),
            "top_candidate_nces_school_id": row.get("nces_school_id", ""),
            "top_candidate_nces_school_name": row.get("nces_school_name", ""),
            "top_candidate_match_score": row.get("match_score", ""),
            "api_decision": parsed.get("decision", ""),
            "api_selected_nces_school_id": parsed.get("selected_nces_school_id", ""),
            "api_selected_candidate_rank": parsed.get("selected_candidate_rank", ""),
            "api_confidence": parsed.get("confidence", ""),
            "api_reason": parsed.get("reason", ""),
        }

        output_rows.append(output)

        combined = pd.concat([existing, pd.DataFrame(output_rows)], ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)

        time.sleep(args.sleep)

    print()
    print("Done.")
    if OUTPUT_PATH.exists():
        final = pd.read_csv(OUTPUT_PATH, dtype=str, low_memory=False)
        print("Total API-reviewed rows saved:", len(final))
        print()
        print(final["api_decision"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
