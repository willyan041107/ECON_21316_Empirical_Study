"""
Section 4: Calculate school-level hurricane exposure.

Goal:
Use school coordinates and hurricane tracks to construct treatment variables.

Inputs:
- data/intermediate/school_crosswalk.csv
- data/intermediate/hurdat2_track_points.csv

Outputs:
- data/intermediate/school_year_hurricane_exposure.csv

Treatment variables:
- within_50km
- min_distance_km
- max_wind_nearby
- max_exposure_intensity
- storm_names

Important:
Do not use LLM to calculate exposure. Use rule-based geospatial calculations.
"""


def main() -> None:
    """Calculate hurricane exposure for each school and graduation cohort."""
    # TODO: Load school coordinates.
    # TODO: Load hurricane track points.
    # TODO: Define exposure window for each graduation cohort.
    # TODO: Calculate distance from each school to storm tracks.
    # TODO: Construct Within50 and intensity exposure variables.
    # TODO: Save school-year exposure file.
    pass


if __name__ == "__main__":
    main()
