"""
Compute school-year hurricane exposure from HURDAT2 tracks.

Inputs:
    data/intermediate/losfa_panel_with_crosswalk.csv
    data/intermediate/hurdat2_atlantic_tracks_relevant_years.csv

Outputs:
    data/intermediate/school_year_hurricane_exposure.csv
    data/processed/analysis_panel_with_exposure.csv

Main exposure definition:
    For school s and graduation year t, use storm_year = t - 1.

    exposure_index_stormmax =
        max over storms h in storm_year of:
            storm_max_wind_kt_h / (min_distance_to_storm_path_sht + c)

    where c is a smoothing constant, default c = 25 km.

This script also constructs threshold exposure variables:
    within_50km_any_tropical
    within_100km_any_tropical
    within_50km_hurricane
    within_100km_hurricane
    within_50km_major_hurricane
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LOSFA_PANEL_PATH = INTERMEDIATE_DIR / "losfa_panel_with_crosswalk.csv"
TRACKS_PATH = INTERMEDIATE_DIR / "hurdat2_atlantic_tracks_relevant_years.csv"

EXPOSURE_OUT = INTERMEDIATE_DIR / "school_year_hurricane_exposure.csv"
ANALYSIS_PANEL_OUT = PROCESSED_DIR / "analysis_panel_with_exposure.csv"

SMOOTHING_KM = 25.0
EARTH_RADIUS_KM = 6371.0

TROPICAL_STATUSES = {"TD", "TS", "HU", "SS", "SD"}


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute pairwise haversine distance in kilometers."""
    lat1_rad = np.radians(lat1)[:, None]
    lon1_rad = np.radians(lon1)[:, None]
    lat2_rad = np.radians(lat2)[None, :]
    lon2_rad = np.radians(lon2)[None, :]

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def prepare_school_years(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep usable school-year rows with valid coordinates."""
    out = panel.copy()

    out["graduation_year"] = pd.to_numeric(out["graduation_year"], errors="coerce")
    out["students_processed"] = pd.to_numeric(out["students_processed"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    out["usable_for_hurricane_exposure"] = (
        out["usable_for_hurricane_exposure"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    if "valid_for_main_analysis" in out.columns:
        out["valid_for_main_analysis"] = (
            out["valid_for_main_analysis"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
    else:
        out["valid_for_main_analysis"] = True

    out = out[
        out["usable_for_hurricane_exposure"]
        & out["valid_for_main_analysis"]
        & out["latitude"].notna()
        & out["longitude"].notna()
        & out["graduation_year"].notna()
    ].copy()

    out["storm_year"] = out["graduation_year"].astype(int) - 1
    out["source_panel_row_id"] = pd.to_numeric(
        out["source_panel_row_id"],
        errors="coerce",
    ).astype(int)

    # One row per LOSFA school-year observation.
    out["school_year_id"] = (
        out["losfa_school_key"].astype(str)
        + " | "
        + out["graduation_year"].astype(int).astype(str)
    )

    return out


def prepare_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Prepare HURDAT2 track points for distance calculation."""
    out = tracks.copy()

    out["storm_year"] = pd.to_numeric(out["storm_year"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["max_wind_kt"] = pd.to_numeric(out["max_wind_kt"], errors="coerce")

    out["status"] = out["status"].fillna("").astype(str).str.strip()

    # First-pass exposure uses tropical/subtropical statuses only.
    out = out[
        out["status"].isin(TROPICAL_STATUSES)
        & out["storm_year"].notna()
        & out["latitude"].notna()
        & out["longitude"].notna()
        & out["max_wind_kt"].notna()
    ].copy()

    out["storm_year"] = out["storm_year"].astype(int)

    return out


def summarize_exposure_for_year(
    schools_year: pd.DataFrame,
    tracks_year: pd.DataFrame,
) -> pd.DataFrame:
    """Compute exposure variables for one storm year."""
    if schools_year.empty:
        return pd.DataFrame()

    if tracks_year.empty:
        out = schools_year[[
            "source_panel_row_id",
            "school_year_id",
            "losfa_school_key",
            "graduation_year",
            "storm_year",
            "nces_school_id",
            "nces_school_name",
            "latitude",
            "longitude",
        ]].copy()

        out["num_relevant_storms"] = 0
        out["min_distance_any_tropical_km"] = np.nan
        out["nearest_storm_id"] = pd.NA
        out["nearest_storm_name"] = pd.NA
        out["nearest_storm_wind_kt"] = np.nan
        out["exposure_index_stormmax"] = 0.0
        out["exposure_index_pointmax"] = 0.0
        out["max_exposure_storm_id"] = pd.NA
        out["max_exposure_storm_name"] = pd.NA
        out["max_exposure_distance_km"] = np.nan
        out["max_exposure_wind_kt"] = np.nan
        out["within_50km_any_tropical"] = False
        out["within_100km_any_tropical"] = False
        out["within_50km_hurricane"] = False
        out["within_100km_hurricane"] = False
        out["within_50km_major_hurricane"] = False

        return out

    school_lat = schools_year["latitude"].to_numpy(dtype=float)
    school_lon = schools_year["longitude"].to_numpy(dtype=float)
    track_lat = tracks_year["latitude"].to_numpy(dtype=float)
    track_lon = tracks_year["longitude"].to_numpy(dtype=float)

    distance_matrix = haversine_distance_km(
        school_lat,
        school_lon,
        track_lat,
        track_lon,
    )

    storm_ids = tracks_year["storm_id"].astype(str).to_numpy()
    unique_storm_ids = tracks_year["storm_id"].astype(str).drop_duplicates().tolist()

    output_rows = []

    for school_idx, (_, school) in enumerate(schools_year.reset_index(drop=True).iterrows()):
        distances = distance_matrix[school_idx, :]

        storm_summaries = []

        for storm_id in unique_storm_ids:
            mask = storm_ids == storm_id
            storm_points = tracks_year.loc[mask].copy()
            storm_distances = distances[mask]

            min_distance = float(np.nanmin(storm_distances))
            nearest_point_idx = int(np.nanargmin(storm_distances))

            storm_max_wind = float(storm_points["max_wind_kt"].max())

            # Point-level version: max over track points of wind / distance.
            point_exposure_values = (
                storm_points["max_wind_kt"].to_numpy(dtype=float)
                / (storm_distances + SMOOTHING_KM)
            )
            max_point_exposure = float(np.nanmax(point_exposure_values))

            storm_summaries.append(
                {
                    "storm_id": storm_id,
                    "storm_name": storm_points["storm_name"].iloc[0],
                    "min_distance_km": min_distance,
                    "nearest_point_wind_kt": float(storm_points["max_wind_kt"].iloc[nearest_point_idx]),
                    "storm_max_wind_kt": storm_max_wind,
                    "stormmax_exposure": storm_max_wind / (min_distance + SMOOTHING_KM),
                    "pointmax_exposure": max_point_exposure,
                    "min_distance_hurricane_km": (
                        float(np.nanmin(storm_distances[storm_points["max_wind_kt"].to_numpy(dtype=float) >= 64]))
                        if (storm_points["max_wind_kt"].to_numpy(dtype=float) >= 64).any()
                        else np.nan
                    ),
                    "min_distance_major_km": (
                        float(np.nanmin(storm_distances[storm_points["max_wind_kt"].to_numpy(dtype=float) >= 96]))
                        if (storm_points["max_wind_kt"].to_numpy(dtype=float) >= 96).any()
                        else np.nan
                    ),
                }
            )

        storm_df = pd.DataFrame(storm_summaries)

        nearest = storm_df.sort_values("min_distance_km").iloc[0]
        max_exposure = storm_df.sort_values("stormmax_exposure", ascending=False).iloc[0]

        min_distance_hurricane = storm_df["min_distance_hurricane_km"].min(skipna=True)
        min_distance_major = storm_df["min_distance_major_km"].min(skipna=True)

        row = {
            "source_panel_row_id": school["source_panel_row_id"],
            "school_year_id": school["school_year_id"],
            "losfa_school_key": school["losfa_school_key"],
            "graduation_year": int(school["graduation_year"]),
            "storm_year": int(school["storm_year"]),
            "nces_school_id": school["nces_school_id"],
            "nces_school_name": school["nces_school_name"],
            "latitude": school["latitude"],
            "longitude": school["longitude"],
            "num_relevant_storms": len(unique_storm_ids),
            "min_distance_any_tropical_km": nearest["min_distance_km"],
            "nearest_storm_id": nearest["storm_id"],
            "nearest_storm_name": nearest["storm_name"],
            "nearest_storm_wind_kt": nearest["nearest_point_wind_kt"],
            "exposure_index_stormmax": max_exposure["stormmax_exposure"],
            "exposure_index_pointmax": storm_df["pointmax_exposure"].max(),
            "max_exposure_storm_id": max_exposure["storm_id"],
            "max_exposure_storm_name": max_exposure["storm_name"],
            "max_exposure_distance_km": max_exposure["min_distance_km"],
            "max_exposure_wind_kt": max_exposure["storm_max_wind_kt"],
            "min_distance_hurricane_km": min_distance_hurricane,
            "min_distance_major_hurricane_km": min_distance_major,
            "within_50km_any_tropical": nearest["min_distance_km"] <= 50,
            "within_100km_any_tropical": nearest["min_distance_km"] <= 100,
            "within_50km_hurricane": (
                pd.notna(min_distance_hurricane)
                and min_distance_hurricane <= 50
            ),
            "within_100km_hurricane": (
                pd.notna(min_distance_hurricane)
                and min_distance_hurricane <= 100
            ),
            "within_50km_major_hurricane": (
                pd.notna(min_distance_major)
                and min_distance_major <= 50
            ),
        }

        output_rows.append(row)

    return pd.DataFrame(output_rows)


def compute_exposure(schools: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Compute exposure for all school-year observations."""
    exposure_frames = []

    for storm_year, schools_year in schools.groupby("storm_year"):
        tracks_year = tracks[tracks["storm_year"] == storm_year].copy()

        print(
            f"Computing storm_year={storm_year}: "
            f"{len(schools_year)} school-year rows, "
            f"{tracks_year['storm_id'].nunique()} storms, "
            f"{len(tracks_year)} track points"
        )

        exposure_frames.append(
            summarize_exposure_for_year(
                schools_year=schools_year,
                tracks_year=tracks_year,
            )
        )

    return pd.concat(exposure_frames, ignore_index=True)


def main() -> None:
    """Run school-year hurricane exposure construction."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(LOSFA_PANEL_PATH, dtype=str, low_memory=False)
    panel = panel.reset_index(drop=False).rename(columns={"index": "source_panel_row_id"})

    tracks = pd.read_csv(TRACKS_PATH, dtype=str, low_memory=False)

    schools = prepare_school_years(panel)
    tracks = prepare_tracks(tracks)

    exposure = compute_exposure(schools, tracks)

    exposure.to_csv(EXPOSURE_OUT, index=False)

    # Merge exposure back using the original panel row id.
    # This avoids row expansion when the same school has aliases or repeated names.
    panel["source_panel_row_id"] = pd.to_numeric(
        panel["source_panel_row_id"],
        errors="coerce",
    ).astype(int)

    exposure["source_panel_row_id"] = pd.to_numeric(
        exposure["source_panel_row_id"],
        errors="coerce",
    ).astype(int)

    duplicate_exposure_rows = exposure["source_panel_row_id"].duplicated().sum()
    if duplicate_exposure_rows != 0:
        raise ValueError(
            f"Exposure has duplicate source_panel_row_id values: {duplicate_exposure_rows}"
        )

    print("\nPanel rows before exposure merge:", len(panel))
    print("Exposure rows before merge:", len(exposure))

    merged = panel.merge(
        exposure,
        on="source_panel_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_exposure"),
    )

    merged.to_csv(ANALYSIS_PANEL_OUT, index=False)

    print("\nSaved:")
    print(" -", EXPOSURE_OUT.relative_to(PROJECT_ROOT))
    print(" -", ANALYSIS_PANEL_OUT.relative_to(PROJECT_ROOT))

    print("\nExposure rows:", len(exposure))
    print("Analysis panel rows:", len(merged))

    print("\nExposure summary:")
    print(
        exposure[
            [
                "exposure_index_stormmax",
                "exposure_index_pointmax",
                "min_distance_any_tropical_km",
                "min_distance_hurricane_km",
                "within_50km_hurricane",
                "within_100km_hurricane",
                "within_50km_major_hurricane",
            ]
        ].describe(include="all").to_string()
    )

    print("\nTop 20 school-years by stormmax exposure:")
    print(
        exposure.sort_values("exposure_index_stormmax", ascending=False)[
            [
                "graduation_year",
                "storm_year",
                "nces_school_name",
                "max_exposure_storm_name",
                "max_exposure_distance_km",
                "max_exposure_wind_kt",
                "exposure_index_stormmax",
                "within_50km_hurricane",
                "within_100km_hurricane",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
