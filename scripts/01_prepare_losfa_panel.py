"""
Section 1: Prepare LOSFA TOPS outcome panel.

Goal:
Create a clean school-by-graduation-year panel from LOSFA files.

Inputs:
- data/raw/losfa/

Outputs:
- data/intermediate/losfa_panel_clean.csv

Key variables to construct:
- school_name
- parish
- graduation_year
- total_processed_students
- total_eligible_students
- total_recipients
- eligibility_rate
- recipient_rate
- eligibility_conversion_rate
- award-type counts and shares if available

Important checks:
- Each school-year should be unique.
- total_eligible_students <= total_processed_students.
- total_recipients <= total_processed_students.
- Rates should be between 0 and 1.
"""


def main() -> None:
    """Prepare the LOSFA school-by-cohort outcome panel."""
    # TODO: Load raw LOSFA files.
    # TODO: Standardize column names.
    # TODO: Extract graduation year.
    # TODO: Construct outcome rates.
    # TODO: Save clean LOSFA panel.
    pass


if __name__ == "__main__":
    main()
