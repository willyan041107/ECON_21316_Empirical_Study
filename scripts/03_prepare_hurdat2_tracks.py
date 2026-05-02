"""
Section 3: Prepare NOAA HURDAT2 hurricane track data.

Goal:
Convert raw HURDAT2 text data into structured storm track points.

Inputs:
- data/raw/hurdat2/

Outputs:
- data/intermediate/hurdat2_track_points.csv

Key variables:
- storm_id
- storm_name
- datetime
- latitude
- longitude
- max_wind
- storm_status
- storm_year

Important:
Do not use LLM for this step. This should be deterministic parsing.
"""


def main() -> None:
    """Parse raw HURDAT2 data into structured storm track points."""
    # TODO: Load raw HURDAT2 file.
    # TODO: Parse storm headers and track rows.
    # TODO: Convert latitude and longitude to numeric values.
    # TODO: Save structured storm track file.
    pass


if __name__ == "__main__":
    main()
