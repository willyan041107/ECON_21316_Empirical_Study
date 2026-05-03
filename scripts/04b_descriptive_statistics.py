"""
Create descriptive statistics for the regression-ready panel.

Input:
    data/processed/regression_panel.csv

Outputs:
    data/processed/descriptive_sample_summary.csv
    data/processed/descriptive_variable_summary.csv
    data/processed/descriptive_by_year.csv
    data/processed/descriptive_by_exposure_quartile.csv
    data/processed/descriptive_indicator_summary.csv
    data/processed/descriptive_robustness_outcomes_no_2025.csv
    data/processed/descriptive_statistics_report.md
    outputs/figures/descriptive_mean_eligibility_by_year.png
    outputs/figures/descriptive_mean_exposure_by_year.png
    outputs/figures/descriptive_eligibility_by_exposure_quartile.png

Main outcome:
    eligibility_rate

Robustness outcomes:
    recipient_rate
    acceptance_rate

Main exposure:
    exposure_index_pointmax
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "regression_panel.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

SAMPLE_SUMMARY_OUT = PROCESSED_DIR / "descriptive_sample_summary.csv"
VARIABLE_SUMMARY_OUT = PROCESSED_DIR / "descriptive_variable_summary.csv"
BY_YEAR_OUT = PROCESSED_DIR / "descriptive_by_year.csv"
BY_EXPOSURE_QUARTILE_OUT = PROCESSED_DIR / "descriptive_by_exposure_quartile.csv"
INDICATOR_SUMMARY_OUT = PROCESSED_DIR / "descriptive_indicator_summary.csv"
ROBUSTNESS_NO_2025_OUT = PROCESSED_DIR / "descriptive_robustness_outcomes_no_2025.csv"
REPORT_OUT = PROCESSED_DIR / "descriptive_statistics_report.md"


MAIN_OUTCOME = "eligibility_rate"
ROBUSTNESS_OUTCOMES = ["recipient_rate", "acceptance_rate"]

MAIN_EXPOSURE = "exposure_index_pointmax"
ROBUSTNESS_EXPOSURES = [
    "exposure_index_stormmax",
    "within_50km_hurricane",
    "within_100km_hurricane",
    "within_50km_major_hurricane",
]

NUMERIC_COLUMNS = [
    "graduation_year",
    "students_processed",
    "eligibility_rate",
    "recipient_rate",
    "acceptance_rate",
    "exposure_index_pointmax",
    "exposure_index_stormmax",
    "exposure_pointmax_z",
    "exposure_stormmax_z",
    "min_distance_any_tropical_km",
    "min_distance_hurricane_km",
    "min_distance_major_hurricane_km",
]

BOOLEAN_COLUMNS = [
    "within_50km_any_tropical",
    "within_100km_any_tropical",
    "within_50km_hurricane",
    "within_100km_hurricane",
    "within_50km_major_hurricane",
]


def to_bool(series: pd.Series) -> pd.Series:
    """Convert string-like boolean values to boolean."""
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted mean with missing-value handling."""
    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return np.nan

    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a small DataFrame to a markdown table without requiring tabulate."""
    if df.empty:
        return "_No rows._"

    display = df.copy()

    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    headers = list(display.columns)
    rows = display.astype(str).values.tolist()

    table = []
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        table.append("| " + " | ".join(row) + " |")

    return "\n".join(table)


def prepare_panel(path: Path) -> pd.DataFrame:
    """Read and clean regression panel."""
    df = pd.read_csv(path, dtype=str, low_memory=False)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = to_bool(df[col])

    return df


def make_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create high-level sample summary."""
    rows = [
        ("school_year_observations", len(df)),
        ("unique_nces_schools", df["nces_school_id"].nunique()),
        ("first_graduation_year", int(df["graduation_year"].min())),
        ("last_graduation_year", int(df["graduation_year"].max())),
        ("total_students_processed", int(df["students_processed"].sum())),
        ("mean_students_per_school_year", df["students_processed"].mean()),
        ("median_students_per_school_year", df["students_processed"].median()),
        ("mean_eligibility_rate", df["eligibility_rate"].mean()),
        ("mean_recipient_rate", df["recipient_rate"].mean()),
        ("mean_acceptance_rate", df["acceptance_rate"].mean()),
        ("mean_pointmax_exposure", df["exposure_index_pointmax"].mean()),
        ("mean_stormmax_exposure", df["exposure_index_stormmax"].mean()),
        ("share_within_50km_hurricane", df["within_50km_hurricane"].mean()),
        ("share_within_100km_hurricane", df["within_100km_hurricane"].mean()),
        ("share_within_50km_major_hurricane", df["within_50km_major_hurricane"].mean()),
    ]

    return pd.DataFrame(rows, columns=["statistic", "value"])


def make_variable_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create Table 1-style variable summary."""
    variables = [
        "eligibility_rate",
        "recipient_rate",
        "acceptance_rate",
        "students_processed",
        "exposure_index_pointmax",
        "exposure_index_stormmax",
        "exposure_pointmax_z",
        "min_distance_any_tropical_km",
        "min_distance_hurricane_km",
        "within_50km_hurricane",
        "within_100km_hurricane",
        "within_50km_major_hurricane",
    ]

    rows = []

    for var in variables:
        if var not in df.columns:
            continue

        series = df[var]

        # Convert boolean indicators to 0/1 before computing summary statistics.
        # This avoids numpy boolean quantile errors and makes means interpretable as shares.
        if series.dtype == bool:
            series = series.astype(int)

        rows.append(
            {
                "variable": var,
                "count": int(series.notna().sum()),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p25": series.quantile(0.25),
                "median": series.quantile(0.50),
                "p75": series.quantile(0.75),
                "max": series.max(),
            }
        )

    return pd.DataFrame(rows)


def make_by_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive statistics by graduation year."""
    rows = []

    for year, group in df.groupby("graduation_year"):
        rows.append(
            {
                "graduation_year": int(year),
                "rows": len(group),
                "unique_schools": group["nces_school_id"].nunique(),
                "students": group["students_processed"].sum(),
                "mean_eligibility_rate": group["eligibility_rate"].mean(),
                "student_weighted_eligibility_rate": weighted_mean(
                    group["eligibility_rate"],
                    group["students_processed"],
                ),
                "mean_recipient_rate": group["recipient_rate"].mean(),
                "mean_acceptance_rate": group["acceptance_rate"].mean(),
                "mean_exposure_pointmax": group["exposure_index_pointmax"].mean(),
                "p90_exposure_pointmax": group["exposure_index_pointmax"].quantile(0.90),
                "max_exposure_pointmax": group["exposure_index_pointmax"].max(),
                "mean_exposure_stormmax": group["exposure_index_stormmax"].mean(),
                "share_within_50km_hurricane": group["within_50km_hurricane"].mean(),
                "share_within_100km_hurricane": group["within_100km_hurricane"].mean(),
                "share_within_50km_major_hurricane": group["within_50km_major_hurricane"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("graduation_year")


def make_exposure_quartile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize outcomes by pointmax exposure quartile."""
    out = df.copy()

    out["exposure_pointmax_quartile"] = pd.qcut(
        out["exposure_index_pointmax"],
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )

    rows = []

    for quartile, group in out.groupby("exposure_pointmax_quartile", observed=True):
        rows.append(
            {
                "exposure_pointmax_quartile": str(quartile),
                "rows": len(group),
                "students": group["students_processed"].sum(),
                "mean_exposure_pointmax": group["exposure_index_pointmax"].mean(),
                "mean_eligibility_rate": group["eligibility_rate"].mean(),
                "student_weighted_eligibility_rate": weighted_mean(
                    group["eligibility_rate"],
                    group["students_processed"],
                ),
                "mean_recipient_rate": group["recipient_rate"].mean(),
                "mean_acceptance_rate": group["acceptance_rate"].mean(),
                "share_within_50km_hurricane": group["within_50km_hurricane"].mean(),
                "share_within_100km_hurricane": group["within_100km_hurricane"].mean(),
            }
        )

    return pd.DataFrame(rows)


def make_indicator_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize exposure indicator variables."""
    rows = []

    for col in [
        "within_50km_any_tropical",
        "within_100km_any_tropical",
        "within_50km_hurricane",
        "within_100km_hurricane",
        "within_50km_major_hurricane",
    ]:
        if col not in df.columns:
            continue

        rows.append(
            {
                "indicator": col,
                "count_true": int(df[col].sum()),
                "count_total": len(df),
                "share_true": df[col].mean(),
                "students_true": df.loc[df[col], "students_processed"].sum(),
                "students_total": df["students_processed"].sum(),
                "student_share_true": (
                    df.loc[df[col], "students_processed"].sum()
                    / df["students_processed"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def make_robustness_no_2025_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize robustness outcomes excluding 2025 because recipients may be incomplete."""
    no_2025 = df[df["graduation_year"] < 2025].copy()

    variables = ["recipient_rate", "acceptance_rate"]

    rows = []

    for var in variables:
        rows.append(
            {
                "sample": "excluding_2025",
                "variable": var,
                "count": int(no_2025[var].notna().sum()),
                "mean": no_2025[var].mean(),
                "std": no_2025[var].std(),
                "min": no_2025[var].min(),
                "p25": no_2025[var].quantile(0.25),
                "median": no_2025[var].quantile(0.50),
                "p75": no_2025[var].quantile(0.75),
                "max": no_2025[var].max(),
            }
        )

    return pd.DataFrame(rows)


def save_figures(by_year: pd.DataFrame, by_quartile: pd.DataFrame) -> None:
    """Save descriptive figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(
        by_year["graduation_year"],
        by_year["mean_eligibility_rate"],
        marker="o",
        label="Mean eligibility rate",
    )
    plt.xlabel("Graduation Year")
    plt.ylabel("Eligibility Rate")
    plt.title("Mean TOPS Eligibility Rate by Graduation Year")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "descriptive_mean_eligibility_by_year.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        by_year["graduation_year"],
        by_year["mean_exposure_pointmax"],
        marker="o",
        label="Mean pointmax exposure",
    )
    plt.xlabel("Graduation Year")
    plt.ylabel("Mean Pointmax Exposure")
    plt.title("Mean Hurricane Exposure by Graduation Year")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "descriptive_mean_exposure_by_year.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(
        by_quartile["exposure_pointmax_quartile"].astype(str),
        by_quartile["mean_eligibility_rate"],
    )
    plt.xlabel("Pointmax Exposure Quartile")
    plt.ylabel("Mean Eligibility Rate")
    plt.title("Eligibility Rate by Exposure Quartile")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "descriptive_eligibility_by_exposure_quartile.png", dpi=300)
    plt.close()


def write_report(
    sample_summary: pd.DataFrame,
    variable_summary: pd.DataFrame,
    by_year: pd.DataFrame,
    by_quartile: pd.DataFrame,
    indicator_summary: pd.DataFrame,
    robustness_no_2025: pd.DataFrame,
) -> None:
    """Write markdown report."""
    report = []

    report.append("# Descriptive Statistics Report\n")
    report.append("## Main empirical setup\n")
    report.append("- Main outcome: `eligibility_rate`.\n")
    report.append("- Robustness outcomes: `recipient_rate`, `acceptance_rate`.\n")
    report.append("- Main exposure: `exposure_index_pointmax`.\n")
    report.append("- Robustness exposure: `exposure_index_stormmax` and hurricane-distance indicators.\n")

    report.append("\n## Sample summary\n")
    report.append(df_to_markdown(sample_summary))

    report.append("\n\n## Variable summary\n")
    report.append(df_to_markdown(variable_summary))

    report.append("\n\n## Exposure indicator summary\n")
    report.append(df_to_markdown(indicator_summary))

    report.append("\n\n## Highest exposure years by mean pointmax exposure\n")
    highest_years = by_year.sort_values("mean_exposure_pointmax", ascending=False).head(10)
    report.append(
        df_to_markdown(
            highest_years[
                [
                    "graduation_year",
                    "rows",
                    "students",
                    "mean_eligibility_rate",
                    "mean_exposure_pointmax",
                    "p90_exposure_pointmax",
                    "share_within_100km_hurricane",
                ]
            ]
        )
    )

    report.append("\n\n## Outcomes by exposure quartile\n")
    report.append(df_to_markdown(by_quartile))

    report.append("\n\n## Robustness outcomes excluding 2025\n")
    report.append(
        "Recipient and acceptance outcomes may be incomplete for the latest cohort, "
        "so robustness checks using these outcomes should exclude 2025.\n\n"
    )
    report.append(df_to_markdown(robustness_no_2025))

    REPORT_OUT.write_text("\n".join(report))


def main() -> None:
    """Run descriptive statistics."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing regression panel: {INPUT_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_panel(INPUT_PATH)

    sample_summary = make_sample_summary(df)
    variable_summary = make_variable_summary(df)
    by_year = make_by_year_summary(df)
    by_quartile = make_exposure_quartile_summary(df)
    indicator_summary = make_indicator_summary(df)
    robustness_no_2025 = make_robustness_no_2025_summary(df)

    sample_summary.to_csv(SAMPLE_SUMMARY_OUT, index=False)
    variable_summary.to_csv(VARIABLE_SUMMARY_OUT, index=False)
    by_year.to_csv(BY_YEAR_OUT, index=False)
    by_quartile.to_csv(BY_EXPOSURE_QUARTILE_OUT, index=False)
    indicator_summary.to_csv(INDICATOR_SUMMARY_OUT, index=False)
    robustness_no_2025.to_csv(ROBUSTNESS_NO_2025_OUT, index=False)

    save_figures(by_year, by_quartile)

    write_report(
        sample_summary=sample_summary,
        variable_summary=variable_summary,
        by_year=by_year,
        by_quartile=by_quartile,
        indicator_summary=indicator_summary,
        robustness_no_2025=robustness_no_2025,
    )

    print("Saved:")
    print(" -", SAMPLE_SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", VARIABLE_SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", BY_YEAR_OUT.relative_to(PROJECT_ROOT))
    print(" -", BY_EXPOSURE_QUARTILE_OUT.relative_to(PROJECT_ROOT))
    print(" -", INDICATOR_SUMMARY_OUT.relative_to(PROJECT_ROOT))
    print(" -", ROBUSTNESS_NO_2025_OUT.relative_to(PROJECT_ROOT))
    print(" -", REPORT_OUT.relative_to(PROJECT_ROOT))
    print(" -", (FIGURE_DIR / "descriptive_mean_eligibility_by_year.png").relative_to(PROJECT_ROOT))
    print(" -", (FIGURE_DIR / "descriptive_mean_exposure_by_year.png").relative_to(PROJECT_ROOT))
    print(" -", (FIGURE_DIR / "descriptive_eligibility_by_exposure_quartile.png").relative_to(PROJECT_ROOT))

    print("\n=== Sample Summary ===")
    print(sample_summary.to_string(index=False))

    print("\n=== Main Variables ===")
    print(
        variable_summary[
            variable_summary["variable"].isin(
                [
                    "eligibility_rate",
                    "recipient_rate",
                    "acceptance_rate",
                    "exposure_index_pointmax",
                    "exposure_index_stormmax",
                    "within_50km_hurricane",
                    "within_100km_hurricane",
                ]
            )
        ].to_string(index=False)
    )

    print("\n=== Highest Exposure Years ===")
    print(
        by_year.sort_values("mean_exposure_pointmax", ascending=False)[
            [
                "graduation_year",
                "rows",
                "students",
                "mean_eligibility_rate",
                "mean_exposure_pointmax",
                "p90_exposure_pointmax",
                "share_within_100km_hurricane",
            ]
        ].head(10).to_string(index=False)
    )

    print("\n=== Outcomes by Exposure Quartile ===")
    print(by_quartile.to_string(index=False))

    print("\n=== Robustness Outcomes Excluding 2025 ===")
    print(robustness_no_2025.to_string(index=False))


if __name__ == "__main__":
    main()
