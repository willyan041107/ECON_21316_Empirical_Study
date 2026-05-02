"""
Section 5: Build final main analysis panel.

Goal:
Merge LOSFA outcomes, NCES school data, and hurricane exposure data.

Inputs:
- data/intermediate/losfa_panel_clean.csv
- data/intermediate/school_crosswalk.csv
- data/intermediate/school_year_hurricane_exposure.csv

Outputs:
- data/processed/analysis_panel_main.csv

Main unit:
- school_id × graduation_year

Important:
This should become the main regression-ready dataset.
"""


def main() -> None:
    """Build the main school-by-cohort analysis panel."""
    # TODO: Load clean LOSFA panel.
    # TODO: Merge NCES crosswalk.
    # TODO: Merge hurricane exposure.
    # TODO: Create fixed-effect IDs.
    # TODO: Save final main panel.
    pass


if __name__ == "__main__":
    main()
