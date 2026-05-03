"""
Create diagnostics for HURDAT2-based hurricane exposure variables.

Inputs:
    data/processed/analysis_panel_with_exposure.csv

Outputs:
    data/intermediate/exposure_by_year.csv
    data/intermediate/top_exposed_school_years_pointmax.csv
    data/intermediate/top_exposed_school_years_stormmax.csv
    data/intermediate/exposure_indicator_summary.csv
    data/intermediate/exposure_by_storm.csv
    data/processed/exposure_diagnostics_report.md
    outputs/figures/exposure_mean_by_year.png
    outputs/figures/exposure_p90_by_year.png
    outputs/figures/exposure_pointmax_vs_stormmax.png

Purpose:
    This script does not construct new exposure measures. It checks whether the
    exposure variables from 03b are empirically reasonable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "analysis_panel_with_exposure.csv"

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

YEAR_OUT = INTERMEDIATE_DIR / "exposure_by_year.csv"
TOP_POINTMAX_OUT = INTERMEDIATE_DIR / "top_exposed_school_years_pointmax.csv"
TOP_STORMMAX_OUT = INTERMEDIATE_DIR / "top_exposed_school_years_stormmax.csv"
INDICATOR_OUT = INTERMEDIATE_DIR / "exposure_indicator_summary.csv"
STORM_OUT = INTERMEDIATE_DIR / "exposure_by_storm.csv"
REPORT_OUT = PROCESSED_DIR / "exposure_diagnostics_report.md"


NUMERIC_COLUMNS = [
    "graduation_year",
    "storm_year",
    "students_processed",
    "eligibility_rate",
    "recipient_rate",
    "acceptance_rate",
    "exposure_index_pointmax",
    "exposure_index_stormmax",
    "min_distance_any_tropical_km",
    "min_distance_hurricane_km",
    "min_distance_major_hurricane_km",
    "max_exposure_distance_km",
    "max_exposure_wind_kt",
]

BOOLEAN_COLUMNS = [
    "usable_for_hurricane_exposure",
    "valid_for_main_analysis",
    "within_50km_any_tropical",
    "within_100km_any_tropical",
    "within_50km_hurricane",
    "within_100km_hurricane",
    "within_50km_major_hurricane",
]


def to_bool(series: pd.Series) -> pd.Series:
    """Convert string-like boolean values to bool."""
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted mean with missing-value handling."""
    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return float("nan")

    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def prepare_panel(path: Path) -> pd.DataFrame:
    """Read and clean the analysis panel."""
    df = pd.read_csv(path, dtype=str, low_memory=False)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = to_bool(df[col])

    df["has_exposure"] = df["exposure_index_pointmax"].notna()

    return df


def make_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize exposure variables by graduation year."""
    exposed = df[df["has_exposure"]].copy()

    summary = exposed.groupby("graduation_year").agg(
        rows=("hs_name", "size"),
        students=("students_processed", "sum"),
        mean_pointmax=("exposure_index_pointmax", "mean"),
        median_pointmax=("exposure_index_pointmax", "median"),
        p90_pointmax=("exposure_index_pointmax", lambda x: x.quantile(0.90)),
        p95_pointmax=("exposure_index_pointmax", lambda x: x.quantile(0.95)),
        max_pointmax=("exposure_index_pointmax", "max"),
        mean_stormmax=("exposure_index_stormmax", "mean"),
        median_stormmax=("exposure_index_stormmax", "median"),
        p90_stormmax=("exposure_index_stormmax", lambda x: x.quantile(0.90)),
        max_stormmax=("exposure_index_stormmax", "max"),
        share_within_50km_hurricane=("within_50km_hurricane", "mean"),
        share_within_100km_hurricane=("within_100km_hurricane", "mean"),
        share_within_50km_major_hurricane=("within_50km_major_hurricane", "mean"),
    ).reset_index()

    weighted_rows = []

    for year, group in exposed.groupby("graduation_year"):
        weighted_rows.append(
            {
                "graduation_year": year,
                "student_weighted_mean_pointmax": weighted_mean(
                    group["exposure_index_pointmax"],
                    group["students_processed"],
                ),
                "student_weighted_mean_stormmax": weighted_mean(
                    group["exposure_index_stormmax"],
                    group["students_processed"],
                ),
            }
        )

    weighted = pd.DataFrame(weighted_rows)

    summary = summary.merge(weighted, on="graduation_year", how="left")

    return summary.sort_values("graduation_year")


def make_top_school_years(df: pd.DataFrame, exposure_col: str, n: int = 100) -> pd.DataFrame:
    """Return top exposed school-year observations."""
    cols = [
        "graduation_year",
        "storm_year",
        "parish",
        "hs_name",
        "nces_school_id",
        "nces_school_name",
        "nces_parish",
        "max_exposure_storm_id",
        "max_exposure_storm_name",
        "max_exposure_distance_km",
        "max_exposure_wind_kt",
        "exposure_index_pointmax",
        "exposure_index_stormmax",
        "min_distance_any_tropical_km",
        "min_distance_hurricane_km",
        "within_50km_hurricane",
        "within_100km_hurricane",
        "within_50km_major_hurricane",
        "students_processed",
        "eligibility_rate",
        "recipient_rate",
        "acceptance_rate",
    ]

    existing_cols = [col for col in cols if col in df.columns]

    return (
        df[df["has_exposure"]]
        .sort_values(exposure_col, ascending=False)
        [existing_cols]
        .head(n)
        .copy()
    )


def make_indicator_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize threshold indicator variables."""
    exposed = df[df["has_exposure"]].copy()

    rows = []

    for col in [
        "within_50km_any_tropical",
        "within_100km_any_tropical",
        "within_50km_hurricane",
        "within_100km_hurricane",
        "within_50km_major_hurricane",
    ]:
        if col not in exposed.columns:
            continue

        rows.append(
            {
                "indicator": col,
                "count_true": int(exposed[col].sum()),
                "count_total": len(exposed),
                "share_true": float(exposed[col].mean()),
                "students_true": float(exposed.loc[exposed[col], "students_processed"].sum()),
                "students_total": float(exposed["students_processed"].sum()),
                "student_share_true": float(
                    exposed.loc[exposed[col], "students_processed"].sum()
                    / exposed["students_processed"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def make_storm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize which storms drive maximum exposure."""
    exposed = df[df["has_exposure"]].copy()

    summary = exposed.groupby(
        ["storm_year", "max_exposure_storm_id", "max_exposure_storm_name"],
        dropna=False,
    ).agg(
        exposed_school_years=("hs_name", "size"),
        exposed_students=("students_processed", "sum"),
        mean_pointmax=("exposure_index_pointmax", "mean"),
        max_pointmax=("exposure_index_pointmax", "max"),
        mean_stormmax=("exposure_index_stormmax", "mean"),
        max_stormmax=("exposure_index_stormmax", "max"),
        min_distance_km=("max_exposure_distance_km", "min"),
        schools_within_50km_hurricane=("within_50km_hurricane", "sum"),
        schools_within_100km_hurricane=("within_100km_hurricane", "sum"),
    ).reset_index()

    return summary.sort_values(["storm_year", "max_pointmax"], ascending=[True, False])


def save_figures(year_summary: pd.DataFrame, df: pd.DataFrame) -> None:
    """Save basic diagnostic figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(year_summary["graduation_year"], year_summary["mean_pointmax"], marker="o")
    plt.xlabel("Graduation Year")
    plt.ylabel("Mean Point-Level Exposure")
    plt.title("Mean Hurricane Exposure by Graduation Year")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "exposure_mean_by_year.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(year_summary["graduation_year"], year_summary["p90_pointmax"], marker="o")
    plt.xlabel("Graduation Year")
    plt.ylabel("90th Percentile Point-Level Exposure")
    plt.title("90th Percentile Hurricane Exposure by Graduation Year")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "exposure_p90_by_year.png", dpi=300)
    plt.close()

    exposed = df[df["has_exposure"]].copy()

    plt.figure(figsize=(6, 6))
    plt.scatter(
        exposed["exposure_index_pointmax"],
        exposed["exposure_index_stormmax"],
        alpha=0.4,
        s=12,
    )
    plt.xlabel("Point-Level Exposure")
    plt.ylabel("Storm-Max Exposure")
    plt.title("Point-Level vs. Storm-Max Exposure")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "exposure_pointmax_vs_stormmax.png", dpi=300)
    plt.close()


def write_report(
    df: pd.DataFrame,
    year_summary: pd.DataFrame,
    indicator_summary: pd.DataFrame,
    storm_summary: pd.DataFrame,
    top_pointmax: pd.DataFrame,
) -> None:
    """Write a markdown diagnostics report."""
    exposed = df[df["has_exposure"]].copy()

    corr = exposed[["exposure_index_pointmax", "exposure_index_stormmax"]].corr().iloc[0, 1]

    high_years = year_summary.sort_values("mean_pointmax", ascending=False).head(8)

    report = []
    report.append("# Hurricane Exposure Diagnostics\n")
    report.append("## Basic coverage\n")
    report.append(f"- Analysis panel rows: {len(df):,}\n")
    report.append(f"- Rows with exposure: {len(exposed):,}\n")
    report.append(f"- Rows without exposure: {len(df) - len(exposed):,}\n")
    report.append(f"- Correlation between pointmax and stormmax exposure: {corr:.3f}\n")

    report.append("\n## Highest mean exposure years\n")
    report.append(high_years[
        [
            "graduation_year",
            "rows",
            "mean_pointmax",
            "p90_pointmax",
            "max_pointmax",
            "mean_stormmax",
            "max_stormmax",
        ]
    ].to_markdown(index=False))

    report.append("\n\n## Threshold indicator summary\n")
    report.append(indicator_summary.to_markdown(index=False))

    report.append("\n\n## Top storms by max point-level exposure\n")
    report.append(
        storm_summary.sort_values("max_pointmax", ascending=False).head(15)[
            [
                "storm_year",
                "max_exposure_storm_name",
                "exposed_school_years",
                "exposed_students",
                "mean_pointmax",
                "max_pointmax",
                "schools_within_50km_hurricane",
                "schools_within_100km_hurricane",
            ]
        ].to_markdown(index=False)
    )

    report.append("\n\n## Top 20 school-years by point-level exposure\n")
    report.append(
        top_pointmax.head(20)[
            [
                "graduation_year",
                "parish",
                "hs_name",
                "max_exposure_storm_name",
                "max_exposure_distance_km",
                "max_exposure_wind_kt",
                "exposure_index_pointmax",
                "within_50km_hurricane",
            ]
        ].to_markdown(index=False)
    )

    REPORT_OUT.write_text("\n".join(report))


def main() -> None:
    """Run exposure diagnostics."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_panel(ANALYSIS_PANEL_PATH)

    year_summary = make_year_summary(df)
    top_pointmax = make_top_school_years(df, "exposure_index_pointmax", n=100)
    top_stormmax = make_top_school_years(df, "exposure_index_stormmax", n=100)
    indicator_summary = make_indicator_summary(df)
    storm_summary = make_storm_summary(df)

    year_summary.to_csv(YEAR_OUT, index=False)
    top_pointmax.to_csv(TOP_POINTMAX_OUT, index=False)
    top_stormmax.to_csv(TOP_STORMMAX_OUT, index=False)
    indicator_summary.to_csv(INDICATOR_OUT, index=False)
    storm_summary.to_csv(STORM_OUT, index=False)

    save_figures(year_summary, df)
    write_report(df, year_summary, indicator_summary, storm_summary, top_pointmax)

    exposed = df[df["has_exposure"]].copy()
    corr = exposed[["exposure_index_pointmax", "exposure_index_stormmax"]].corr().iloc[0, 1]

    print("Saved:")
    print(" -", YEAR_OUT.relative_to(PROJECT_ROOT))
    print(" -", TOP_POINTMAX_OUT.relative_to(PROJECT_ROOT))
    print(" -", TOP_STORMMAX_OUT.relative_to(PROJECT_ROOT))
    print(" -", INDICATOR_OUT.relative_to(PROJECT_ROOT))
    print(" -", STORM_OUT.relative_to(PROJECT_ROOT))
    print(" -", REPORT_OUT.relative_to(PROJECT_ROOT))
    print(" -", (FIGURE_DIR / "exposure_mean_by_year.png").relative_to(PROJECT_ROOT))
    print(" -", (FIGURE_DIR / "exposure_p90_by_year.png").relative_to(PROJECT_ROOT))
    print(" -", (FIGURE_DIR / "exposure_pointmax_vs_stormmax.png").relative_to(PROJECT_ROOT))

    print("\nRows:", len(df))
    print("Rows with exposure:", len(exposed))
    print("Correlation pointmax/stormmax:", round(corr, 3))

    print("\nHighest mean pointmax exposure years:")
    print(
        year_summary.sort_values("mean_pointmax", ascending=False)[
            [
                "graduation_year",
                "rows",
                "mean_pointmax",
                "p90_pointmax",
                "max_pointmax",
                "mean_stormmax",
                "max_stormmax",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nIndicator summary:")
    print(indicator_summary.to_string(index=False))

    print("\nTop 20 school-years by pointmax exposure:")
    print(
        top_pointmax[
            [
                "graduation_year",
                "parish",
                "hs_name",
                "max_exposure_storm_name",
                "max_exposure_distance_km",
                "max_exposure_wind_kt",
                "exposure_index_pointmax",
                "within_50km_hurricane",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
