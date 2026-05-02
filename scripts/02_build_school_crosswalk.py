"""
Section 2: Build LOSFA-to-NCES school crosswalk.

Goal:
Match LOSFA school names to NCES CCD school IDs.

Inputs:
- data/intermediate/losfa_panel_clean.csv
- data/raw/nces/

Outputs:
- data/intermediate/school_crosswalk.csv

Key variables:
- losfa_school_name
- losfa_parish
- nces_school_id
- nces_school_name
- district_id
- district_name
- latitude
- longitude
- match_method
- match_confidence
- manual_review_flag

Important:
Use exact matching first, fuzzy matching second, and LLM only as a support tool for ambiguous candidates.
"""


def main() -> None:
    """Create a school identifier crosswalk between LOSFA and NCES."""
    # TODO: Load LOSFA school list.
    # TODO: Load NCES school list.
    # TODO: Normalize school names.
    # TODO: Exact match within parish.
    # TODO: Fuzzy match unmatched schools.
    # TODO: Save crosswalk with confidence labels.
    pass


if __name__ == "__main__":
    main()
