"""
Finalize LOSFA-to-NCES school crosswalk.

Inputs:
    data/intermediate/school_crosswalk_preliminary.csv
    data/intermediate/losfa_nces_crosswalk_api_review.csv
    data/intermediate/losfa_nces_crosswalk_candidates.csv

Output:
    data/intermediate/school_crosswalk.csv

Rules:
    1. Keep exact and high-confidence fuzzy auto matches.
    2. Accept API matches only if:
        api_decision == match
        api_confidence >= 0.85
        selected NCES ID exists in the candidate list
    3. Preserve historical_closed / excluded_nonpublic / unresolved cases.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

PRELIM_PATH = INTERMEDIATE_DIR / "school_crosswalk_preliminary.csv"
API_PATH = INTERMEDIATE_DIR / "losfa_nces_crosswalk_api_review.csv"
CANDIDATES_PATH = INTERMEDIATE_DIR / "losfa_nces_crosswalk_candidates.csv"
OUTPUT_PATH = INTERMEDIATE_DIR / "school_crosswalk.csv"


def normalize_id(value: object) -> str:
    """Normalize NCES school ID as a 12-character string."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if text.lower() in {"nan", "none", ""}:
        return ""

    return text.zfill(12)


def main() -> None:
    """Finalize school crosswalk."""
    prelim = pd.read_csv(PRELIM_PATH, dtype=str, low_memory=False)
    candidates = pd.read_csv(CANDIDATES_PATH, dtype=str, low_memory=False)

    if API_PATH.exists():
        api = pd.read_csv(API_PATH, dtype=str, low_memory=False)
    else:
        api = pd.DataFrame()

    prelim["nces_school_id"] = prelim["nces_school_id"].apply(normalize_id)
    candidates["nces_school_id"] = candidates["nces_school_id"].apply(normalize_id)

    # Start from preliminary crosswalk.
    final = prelim.copy()

    final["manual_review_flag"] = final["manual_review_flag"].astype(str)
    final["accepted_match_source"] = ""
    final["api_decision"] = ""
    final["api_confidence"] = ""
    final["api_reason"] = ""

    auto_mask = final["match_method"].isin(["exact_name_parish", "fuzzy_high_confidence"])
    final.loc[auto_mask, "accepted_match_source"] = final.loc[auto_mask, "match_method"]

    # Apply API results.
    if not api.empty:
        api = api.copy()
        api["api_selected_nces_school_id"] = api["api_selected_nces_school_id"].apply(normalize_id)
        api["api_confidence_num"] = pd.to_numeric(api["api_confidence"], errors="coerce")

        valid_candidate_ids = set(
            zip(
                candidates["losfa_school_key"].astype(str),
                candidates["nces_school_id"].astype(str),
            )
        )

        for _, row in api.iterrows():
            key = str(row["losfa_school_key"])
            selected_id = row["api_selected_nces_school_id"]

            row_mask = final["losfa_school_key"].astype(str).eq(key)

            final.loc[row_mask, "api_decision"] = str(row.get("api_decision", ""))
            final.loc[row_mask, "api_confidence"] = str(row.get("api_confidence", ""))
            final.loc[row_mask, "api_reason"] = str(row.get("api_reason", ""))

            if (
                row.get("api_decision") == "match"
                and row["api_confidence_num"] >= 0.85
                and (key, selected_id) in valid_candidate_ids
            ):
                selected_candidate = candidates[
                    (candidates["losfa_school_key"].astype(str).eq(key))
                    & (candidates["nces_school_id"].astype(str).eq(selected_id))
                ].iloc[0]

                for col in [
                    "nces_school_id",
                    "nces_school_name",
                    "nces_match_name",
                    "nces_parish",
                    "nces_district_name",
                    "grade_low",
                    "grade_high",
                    "school_level",
                    "operational_status",
                    "latitude",
                    "longitude",
                ]:
                    if col in final.columns and col in selected_candidate.index:
                        final.loc[row_mask, col] = selected_candidate[col]

                final.loc[row_mask, "final_match_status"] = "api_accepted_match"
                final.loc[row_mask, "manual_review_flag"] = "False"
                final.loc[row_mask, "accepted_match_source"] = "api_match"

            elif row.get("api_decision") == "historical_closed":
                final.loc[row_mask, "final_match_status"] = "historical_closed"
                final.loc[row_mask, "manual_review_flag"] = "True"
                final.loc[row_mask, "accepted_match_source"] = "api_historical_closed"

    # Clean final status labels.
    final["final_match_status"] = final["final_match_status"].fillna("needs_review")
    final["manual_review_flag"] = final["manual_review_flag"].fillna("True")

    # A row is usable for exposure if it has a matched NCES school and coordinates.
    final["latitude"] = pd.to_numeric(final["latitude"], errors="coerce")
    final["longitude"] = pd.to_numeric(final["longitude"], errors="coerce")

    final["has_final_nces_match"] = final["nces_school_id"].fillna("").astype(str).ne("")
    final["has_coordinates"] = final["latitude"].notna() & final["longitude"].notna()

    final["usable_for_hurricane_exposure"] = (
        final["has_final_nces_match"]
        & final["has_coordinates"]
        & final["final_match_status"].isin(
            ["exact_name_parish", "fuzzy_high_confidence", "api_accepted_match"]
        )
    )

    final.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH.relative_to(PROJECT_ROOT))
    print("Rows:", len(final))

    print("\nFinal match status counts:")
    print(final["final_match_status"].value_counts(dropna=False))

    print("\nAccepted match source counts:")
    print(final["accepted_match_source"].value_counts(dropna=False))

    print("\nUsable for hurricane exposure:")
    print(final["usable_for_hurricane_exposure"].value_counts(dropna=False))

    print("\nRemaining manual review rows:")
    remaining = final[
        final["final_match_status"].isin(["needs_review", "historical_closed"])
    ]
    print(len(remaining))

    print("\nTop remaining review rows:")
    show_cols = [
        "losfa_parish",
        "losfa_hs_name",
        "losfa_hs_name_variants",
        "nces_school_name",
        "match_score",
        "score_margin",
        "final_match_status",
        "api_decision",
        "api_reason",
    ]
    show_cols = [col for col in show_cols if col in remaining.columns]

    print(
        remaining[show_cols]
        .head(40)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
