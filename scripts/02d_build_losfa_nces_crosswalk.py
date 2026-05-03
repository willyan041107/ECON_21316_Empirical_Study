"""
Build LOSFA-to-NCES school crosswalk.

This version is conservative:
1. It first canonicalizes LOSFA school-name variants.
2. It matches only to NCES Louisiana high-school candidates.
3. It auto-accepts exact parish-name matches.
4. It auto-accepts fuzzy matches only when the score is very high and clearly separated.
5. It sends ambiguous cases to review.

No API is used in this version.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

LOSFA_PANEL_PATH = INTERMEDIATE_DIR / "losfa_panel_clean.csv"
NCES_HS_PATH = INTERMEDIATE_DIR / "nces_louisiana_high_school_candidates_with_geocode.csv"

TOP_N_CANDIDATES = 5
AUTO_FUZZY_THRESHOLD = 97
MIN_SCORE_MARGIN = 3


def normalize_parish(value: object) -> str:
    """Normalize Louisiana parish names."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = text.replace(" PARISH", "")
    text = text.replace(" COUNTY", "")
    text = re.sub(r"\bSAINT\b", "ST", text)
    text = re.sub(r"\s+", " ", text).strip()

    aliases = {
        "LASALLE": "LA SALLE",
        "LA SALLE": "LA SALLE",
        "ST JOHN THE BAPTIST": "ST. JOHN THE BAPTIST",
        "ST. JOHN THE BAPTIST": "ST. JOHN THE BAPTIST",
    }

    return aliases.get(text, text)


def normalize_name(value: object) -> str:
    """Normalize school names for matching."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    text = text.replace("&", " AND ")
    text = text.replace("’", "'")
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")

    text = re.sub(r"\bSAINT\b", "ST", text)
    text = re.sub(r"\bSCHOO\b", "SCHOOL", text)
    text = re.sub(r"\bSCH\b", "SCHOOL", text)
    text = re.sub(r"\bSC\b", "SCHOOL", text)
    text = re.sub(r"\bHS\b", "HIGH SCHOOL", text)
    text = re.sub(r"\bTECH\b", "TECHNOLOGY", text)
    text = re.sub(r"\bMAG\b", "MAGNET", text)

    text = re.sub(r"\bCLSD\b", "CLOSED", text)

    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parish_aliases_for_school_prefix(parish: str) -> list[str]:
    """Return possible school-name prefixes representing a parish."""
    parish_clean = normalize_parish(parish)
    aliases = {normalize_name(parish_clean)}

    if parish_clean == "LA SALLE":
        aliases.add("LASALLE")

    if parish_clean.startswith("ST. "):
        rest = parish_clean.replace("ST. ", "")
        aliases.add(normalize_name("ST " + rest))
        aliases.add(normalize_name("SAINT " + rest))

    return sorted(aliases, key=len, reverse=True)


def strip_leading_parish_prefix(name: str, parish: str) -> str:
    """
    Strip a leading parish prefix only when the remainder is informative.

    Examples:
        ORLEANS BENJAMIN FRANKLIN HIGH SCHOOL -> BENJAMIN FRANKLIN HIGH SCHOOL
        EAST BATON ROUGE LSU LABORATORY SCHOOL -> LSU LABORATORY SCHOOL

    But:
        LAFAYETTE HIGH SCHOOL should stay LAFAYETTE HIGH SCHOOL
        CADDO PARISH MAGNET HIGH SCHOOL should stay CADDO PARISH MAGNET HIGH SCHOOL
    """
    normalized = normalize_name(name)

    for alias in parish_aliases_for_school_prefix(parish):
        if normalized.startswith(alias + " "):
            remainder = normalized[len(alias):].strip()

            if is_informative_stripped_name(remainder):
                return remainder

    return normalized


def is_informative_stripped_name(name: str) -> bool:
    """Return True if a stripped school name is specific enough to use."""
    if not name:
        return False

    generic_exact = {
        "HIGH SCHOOL",
        "SENIOR HIGH SCHOOL",
        "JUNIOR HIGH SCHOOL",
        "CHARTER SCHOOL",
        "PUBLIC CHARTER SCHOOL",
        "MAGNET SCHOOL",
        "ACADEMY",
        "SCHOOL",
        "HIGH",
        "PARISH HIGH SCHOOL",
        "PARISH MAGNET HIGH SCHOOL",
        "PARISH TECHNICAL AND CAREER",
    }

    if name in generic_exact:
        return False

    if name.startswith("PARISH "):
        return False

    if len(name.split()) < 2:
        return False

    return True


def choose_canonical_losfa_name(row: pd.Series) -> str:
    """Choose the canonical LOSFA match name used to group school aliases."""
    original = row["losfa_match_name_original"]
    stripped = row["losfa_match_name_stripped"]

    if stripped != original and is_informative_stripped_name(stripped):
        return stripped

    return original


def clean_type_history(values: pd.Series) -> str:
    """Create compact LOSFA school-type history."""
    clean_values = sorted(set(values.dropna().astype(str)))
    return ";".join(clean_values)


def build_losfa_unique_schools(losfa: pd.DataFrame) -> pd.DataFrame:
    """Build one row per canonical LOSFA school identity."""
    losfa = losfa.copy()

    losfa["losfa_parish"] = losfa["parish"].apply(normalize_parish)
    losfa["losfa_hs_name"] = losfa["hs_name"].fillna("").astype(str).str.strip()

    bad_name = losfa["losfa_hs_name"].str.upper().str.contains(
        r"\bSUBTOTAL\b|\bGRAND TOTAL\b|\bSTATE TOTAL\b",
        regex=True,
        na=False,
    )
    losfa = losfa[~bad_name].copy()

    losfa["losfa_match_name_original"] = losfa["losfa_hs_name"].apply(normalize_name)
    losfa["losfa_match_name_stripped"] = losfa.apply(
        lambda row: strip_leading_parish_prefix(row["losfa_hs_name"], row["losfa_parish"]),
        axis=1,
    )
    losfa["losfa_match_name"] = losfa.apply(choose_canonical_losfa_name, axis=1)

    group_cols = ["losfa_parish", "losfa_match_name"]

    grouped = (
        losfa.groupby(group_cols, dropna=False)
        .agg(
            losfa_hs_name=("losfa_hs_name", "first"),
            losfa_hs_name_variants=("losfa_hs_name", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            losfa_match_name_original_variants=("losfa_match_name_original", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            losfa_match_name_stripped_variants=("losfa_match_name_stripped", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
            first_year=("graduation_year", "min"),
            last_year=("graduation_year", "max"),
            years_observed=("graduation_year", "nunique"),
            observations=("graduation_year", "size"),
            losfa_hs_type_history=("hs_type", clean_type_history),
            total_students_processed=("students_processed", "sum"),
        )
        .reset_index()
    )

    grouped["losfa_school_key"] = grouped["losfa_parish"] + " | " + grouped["losfa_match_name"]

    type_history = grouped["losfa_hs_type_history"].fillna("")

    grouped["only_nonpublic_losfa"] = (
        type_history.eq("NONPUBLIC") |
        type_history.eq("NONPUBLIC;UNKNOWN")
    )
    grouped["has_public_losfa"] = type_history.str.contains("PUBLIC", regex=False)

    grouped["match_scope"] = grouped["only_nonpublic_losfa"].map(
        {
            True: "excluded_nonpublic_losfa",
            False: "eligible_for_nces_public_match",
        }
    )

    ordered = [
        "losfa_school_key",
        "losfa_parish",
        "losfa_hs_name",
        "losfa_hs_name_variants",
        "losfa_match_name",
        "losfa_match_name_original_variants",
        "losfa_match_name_stripped_variants",
        "first_year",
        "last_year",
        "years_observed",
        "observations",
        "losfa_hs_type_history",
        "has_public_losfa",
        "only_nonpublic_losfa",
        "match_scope",
        "total_students_processed",
    ]

    return grouped[ordered].sort_values(["losfa_parish", "losfa_match_name"]).reset_index(drop=True)


def prepare_nces(nces: pd.DataFrame) -> pd.DataFrame:
    """Prepare NCES candidate schools."""
    nces = nces.copy()

    nces["nces_parish"] = nces["nces_parish"].apply(normalize_parish)
    nces["nces_match_name"] = nces["nces_school_name"].apply(normalize_name)

    nces["latitude"] = pd.to_numeric(nces["latitude"], errors="coerce")
    nces["longitude"] = pd.to_numeric(nces["longitude"], errors="coerce")

    return nces


def combined_score(query: str, candidate: str) -> float:
    """Compute conservative fuzzy score without token_set_ratio."""
    return max(
        fuzz.WRatio(query, candidate),
        fuzz.token_sort_ratio(query, candidate),
    )


def make_query_names(row: pd.Series) -> list[str]:
    """Create all usable LOSFA query names for one canonical school."""
    raw_values = [
        row.get("losfa_match_name", ""),
        row.get("losfa_match_name_original_variants", ""),
        row.get("losfa_match_name_stripped_variants", ""),
    ]

    names: list[str] = []

    for raw in raw_values:
        for piece in str(raw).split(";"):
            piece = piece.strip()
            if piece and piece not in names:
                names.append(piece)

    return names


def get_candidates_for_school(
    row: pd.Series,
    nces_same_parish: dict[str, pd.DataFrame],
    nces_all: pd.DataFrame,
) -> list[dict]:
    """Return top NCES candidates for one LOSFA school."""
    parish = row["losfa_parish"]
    query_names = make_query_names(row)

    pool = nces_same_parish.get(parish)

    if pool is None or pool.empty:
        pool = nces_all
        candidate_scope = "statewide_no_same_parish_pool"
    else:
        candidate_scope = "same_parish"

    scored = []

    pool = pool.reset_index(drop=True)

    for _, candidate in pool.iterrows():
        candidate_name = candidate["nces_match_name"]

        best_score = -1.0
        best_query = ""

        for query in query_names:
            score = combined_score(query, candidate_name)

            if score > best_score:
                best_score = score
                best_query = query

        scored.append((candidate, best_score, best_query))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    top = scored[:TOP_N_CANDIDATES]

    candidates = []

    for rank, (candidate, score, best_query) in enumerate(top, start=1):
        candidates.append(
            {
                "losfa_school_key": row["losfa_school_key"],
                "losfa_parish": row["losfa_parish"],
                "losfa_hs_name": row["losfa_hs_name"],
                "losfa_hs_name_variants": row["losfa_hs_name_variants"],
                "losfa_match_name": row["losfa_match_name"],
                "score_name_used": best_query,
                "losfa_hs_type_history": row["losfa_hs_type_history"],
                "first_year": row["first_year"],
                "last_year": row["last_year"],
                "years_observed": row["years_observed"],
                "candidate_rank": rank,
                "candidate_scope": candidate_scope,
                "match_score": round(float(score), 2),
                "nces_school_id": candidate["nces_school_id"],
                "nces_school_name": candidate["nces_school_name"],
                "nces_match_name": candidate["nces_match_name"],
                "nces_parish": candidate["nces_parish"],
                "nces_district_name": candidate["nces_district_name"],
                "grade_low": candidate.get("grade_low", pd.NA),
                "grade_high": candidate.get("grade_high", pd.NA),
                "school_level": candidate.get("school_level", pd.NA),
                "operational_status": candidate.get("operational_status", pd.NA),
                "latitude": candidate.get("latitude", pd.NA),
                "longitude": candidate.get("longitude", pd.NA),
            }
        )

    return candidates


def add_second_best_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add second-best score to candidate-rank-1 rows."""
    second = (
        candidates[candidates["candidate_rank"] == 2][
            ["losfa_school_key", "match_score"]
        ]
        .rename(columns={"match_score": "second_best_score"})
    )

    best = candidates[candidates["candidate_rank"] == 1].copy()
    best = best.merge(second, on="losfa_school_key", how="left")
    best["score_margin"] = best["match_score"] - best["second_best_score"].fillna(0)

    return best


def classify_best_candidate(row: pd.Series) -> str:
    """Classify a best candidate."""
    if pd.isna(row.get("nces_school_id")):
        return "no_candidate"

    same_parish = row["losfa_parish"] == row["nces_parish"]
    exact = row["score_name_used"] == row["nces_match_name"]

    if same_parish and exact:
        return "exact_name_parish"

    if (
        same_parish
        and row["match_score"] >= AUTO_FUZZY_THRESHOLD
        and row["score_margin"] >= MIN_SCORE_MARGIN
    ):
        return "fuzzy_high_confidence"

    if row["match_score"] >= 85:
        return "review_ambiguous"

    return "review_low_score"


def build_crosswalk(
    losfa_unique: pd.DataFrame,
    nces: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build candidate, auto, review, excluded, and preliminary crosswalk files."""
    nces_same_parish = {
        parish: group.reset_index(drop=True)
        for parish, group in nces.groupby("nces_parish", dropna=False)
    }

    eligible = losfa_unique[
        losfa_unique["match_scope"].eq("eligible_for_nces_public_match")
    ].copy()

    excluded = losfa_unique[
        losfa_unique["match_scope"].eq("excluded_nonpublic_losfa")
    ].copy()

    all_candidates = []

    for _, row in eligible.iterrows():
        all_candidates.extend(get_candidates_for_school(row, nces_same_parish, nces))

    candidates = pd.DataFrame(all_candidates)

    if candidates.empty:
        raise RuntimeError("No candidates generated.")

    best = add_second_best_scores(candidates)
    best["match_method"] = best.apply(classify_best_candidate, axis=1)

    auto = best[best["match_method"].isin(["exact_name_parish", "fuzzy_high_confidence"])].copy()
    review = best[best["match_method"].isin(["review_ambiguous", "review_low_score", "no_candidate"])].copy()

    preliminary = best.copy()
    preliminary["final_match_status"] = preliminary["match_method"].where(
        preliminary["match_method"].isin(["exact_name_parish", "fuzzy_high_confidence"]),
        "needs_review",
    )
    preliminary["manual_review_flag"] = preliminary["final_match_status"].eq("needs_review")

    excluded_rows = []
    for _, row in excluded.iterrows():
        excluded_rows.append(
            {
                "losfa_school_key": row["losfa_school_key"],
                "losfa_parish": row["losfa_parish"],
                "losfa_hs_name": row["losfa_hs_name"],
                "losfa_hs_name_variants": row["losfa_hs_name_variants"],
                "losfa_match_name": row["losfa_match_name"],
                "score_name_used": pd.NA,
                "losfa_hs_type_history": row["losfa_hs_type_history"],
                "first_year": row["first_year"],
                "last_year": row["last_year"],
                "years_observed": row["years_observed"],
                "candidate_rank": pd.NA,
                "candidate_scope": "not_searched_nonpublic_losfa",
                "match_score": pd.NA,
                "second_best_score": pd.NA,
                "score_margin": pd.NA,
                "nces_school_id": pd.NA,
                "nces_school_name": pd.NA,
                "nces_match_name": pd.NA,
                "nces_parish": pd.NA,
                "nces_district_name": pd.NA,
                "grade_low": pd.NA,
                "grade_high": pd.NA,
                "school_level": pd.NA,
                "operational_status": pd.NA,
                "latitude": pd.NA,
                "longitude": pd.NA,
                "match_method": "excluded_nonpublic_losfa",
                "final_match_status": "excluded_nonpublic_losfa",
                "manual_review_flag": False,
            }
        )

    preliminary = pd.concat([preliminary, pd.DataFrame(excluded_rows)], ignore_index=True)

    return candidates, auto, review, excluded, preliminary


def main() -> None:
    """Run crosswalk construction."""
    losfa = pd.read_csv(LOSFA_PANEL_PATH, dtype=str, low_memory=False)
    nces = pd.read_csv(NCES_HS_PATH, dtype=str, low_memory=False)

    losfa["graduation_year"] = pd.to_numeric(losfa["graduation_year"], errors="coerce")
    losfa["students_processed"] = pd.to_numeric(losfa["students_processed"], errors="coerce")

    losfa_unique = build_losfa_unique_schools(losfa)
    nces = prepare_nces(nces)

    candidates, auto, review, excluded, preliminary = build_crosswalk(losfa_unique, nces)

    paths = {
        "losfa_unique": INTERMEDIATE_DIR / "losfa_unique_schools.csv",
        "candidates": INTERMEDIATE_DIR / "losfa_nces_crosswalk_candidates.csv",
        "auto": INTERMEDIATE_DIR / "losfa_nces_crosswalk_auto.csv",
        "review": INTERMEDIATE_DIR / "losfa_nces_crosswalk_review.csv",
        "nonpublic": INTERMEDIATE_DIR / "losfa_nonpublic_excluded.csv",
        "preliminary": INTERMEDIATE_DIR / "school_crosswalk_preliminary.csv",
    }

    losfa_unique.to_csv(paths["losfa_unique"], index=False)
    candidates.to_csv(paths["candidates"], index=False)
    auto.to_csv(paths["auto"], index=False)
    review.to_csv(paths["review"], index=False)
    excluded.to_csv(paths["nonpublic"], index=False)
    preliminary.to_csv(paths["preliminary"], index=False)

    print("Saved outputs:")
    for path in paths.values():
        print(" -", path.relative_to(PROJECT_ROOT))

    print("\n=== Summary ===")
    print("Unique canonical LOSFA schools:", len(losfa_unique))
    print("Eligible for NCES public matching:", int(losfa_unique["match_scope"].eq("eligible_for_nces_public_match").sum()))
    print("Excluded as LOSFA nonpublic:", len(excluded))
    print("Auto matches:", len(auto))
    print("Review needed:", len(review))
    print("Unique NCES IDs in auto:", auto["nces_school_id"].nunique())

    print("\nMatch method counts:")
    print(preliminary["match_method"].value_counts(dropna=False))

    print("\nSample auto matches:")
    print(
        auto[
            [
                "losfa_parish",
                "losfa_hs_name",
                "losfa_hs_name_variants",
                "nces_school_name",
                "nces_parish",
                "match_score",
                "score_margin",
                "match_method",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSample review rows:")
    print(
        review[
            [
                "losfa_parish",
                "losfa_hs_name",
                "losfa_hs_name_variants",
                "nces_school_name",
                "nces_parish",
                "match_score",
                "score_margin",
                "match_method",
            ]
        ]
        .sort_values("match_score", ascending=False)
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
