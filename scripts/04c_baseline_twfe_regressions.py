"""
Run baseline TWFE regressions.

Input:
    data/processed/regression_panel.csv

Outputs:
    data/processed/regression_results_baseline.csv
    data/processed/regression_results_detailed.csv
    data/processed/regression_table_baseline.md
    data/processed/regression_model_summaries.txt

Main outcome:
    eligibility_rate

Main exposure:
    exposure_index_pointmax

Main preferred model:
    eligibility_rate_st = beta * exposure_index_pointmax_st
                          + school FE + year FE + error_st

Standard errors:
    Clustered by NCES school.

Weights:
    Weighted models use students_processed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "regression_panel.csv"

OUT_RESULTS = PROJECT_ROOT / "data" / "processed" / "regression_results_baseline.csv"
OUT_DETAILS = PROJECT_ROOT / "data" / "processed" / "regression_results_detailed.csv"
OUT_TABLE = PROJECT_ROOT / "data" / "processed" / "regression_table_baseline.md"
OUT_SUMMARIES = PROJECT_ROOT / "data" / "processed" / "regression_model_summaries.txt"


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
    "within_50km_hurricane",
    "within_100km_hurricane",
    "within_50km_major_hurricane",
]


def parse_bool_indicator(series: pd.Series) -> pd.Series:
    """Parse boolean indicators from True/False, 1/0, yes/no strings."""
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
        .astype(int)
    )


def prepare_panel(path: Path) -> pd.DataFrame:
    """Read and clean regression panel."""
    df = pd.read_csv(path, dtype=str, low_memory=False)

    indicator_cols = [
        "within_50km_hurricane",
        "within_100km_hurricane",
        "within_50km_major_hurricane",
    ]

    for col in NUMERIC_COLUMNS:
        if col in df.columns and col not in indicator_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["nces_school_id"] = df["nces_school_id"].astype(str)
    df["graduation_year"] = pd.to_numeric(df["graduation_year"], errors="coerce")
    df = df[df["graduation_year"].notna()].copy()
    df["graduation_year"] = df["graduation_year"].astype(int)

    df["year_fe"] = df["graduation_year"].astype(str)
    df["school_fe"] = df["nces_school_id"].astype(str)

    for col in indicator_cols:
        df[col] = parse_bool_indicator(df[col])

    print("\nIndicator checks after parsing:")
    for col in indicator_cols:
        print(col)
        print(df[col].value_counts(dropna=False).to_string())

    return df


def fit_model(
    df: pd.DataFrame,
    model_name: str,
    outcome: str,
    exposure: str,
    fixed_effects: str,
    weighted: bool,
    exclude_2025: bool = False,
):
    """Fit one regression model with school-clustered standard errors."""
    sample = df.copy()

    if exclude_2025:
        sample = sample[sample["graduation_year"] < 2025].copy()

    needed = [outcome, exposure, "nces_school_id", "students_processed", "graduation_year"]
    sample = sample.dropna(subset=needed).copy()

    if weighted:
        sample = sample[sample["students_processed"] > 0].copy()

    if fixed_effects == "year":
        formula = f"{outcome} ~ {exposure} + C(graduation_year)"
    elif fixed_effects == "school_year":
        formula = f"{outcome} ~ {exposure} + C(nces_school_id) + C(graduation_year)"
    else:
        formula = f"{outcome} ~ {exposure}"

    if weighted:
        model = smf.wls(formula=formula, data=sample, weights=sample["students_processed"])
    else:
        model = smf.ols(formula=formula, data=sample)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["nces_school_id"]},
    )

    return {
        "model_name": model_name,
        "outcome": outcome,
        "exposure": exposure,
        "fixed_effects": fixed_effects,
        "weighted": weighted,
        "exclude_2025": exclude_2025,
        "formula": formula,
        "sample": sample,
        "result": result,
    }


def extract_main_result(model_info: dict) -> dict:
    """Extract coefficient table row for the exposure variable."""
    result = model_info["result"]
    exposure = model_info["exposure"]
    sample = model_info["sample"]

    coef = result.params.get(exposure, np.nan)
    se = result.bse.get(exposure, np.nan)
    pval = result.pvalues.get(exposure, np.nan)

    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    return {
        "model_name": model_info["model_name"],
        "outcome": model_info["outcome"],
        "exposure": exposure,
        "coef": coef,
        "std_error": se,
        "p_value": pval,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "n_obs": int(result.nobs),
        "n_schools": sample["nces_school_id"].nunique(),
        "first_year": int(sample["graduation_year"].min()),
        "last_year": int(sample["graduation_year"].max()),
        "weighted": model_info["weighted"],
        "fixed_effects": model_info["fixed_effects"],
        "exclude_2025": model_info["exclude_2025"],
        "r_squared": result.rsquared,
    }


def stars(p_value: float) -> str:
    """Return conventional significance stars."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def make_markdown_table(results: pd.DataFrame) -> str:
    """Create compact markdown regression table."""
    display = results.copy()

    display["estimate"] = display.apply(
        lambda row: f"{row['coef']:.4f}{stars(row['p_value'])}",
        axis=1,
    )
    display["se"] = display["std_error"].map(lambda x: f"({x:.4f})")
    display["p_value"] = display["p_value"].map(lambda x: f"{x:.3f}")
    display["r_squared"] = display["r_squared"].map(lambda x: f"{x:.3f}")

    cols = [
        "model_name",
        "outcome",
        "exposure",
        "estimate",
        "se",
        "p_value",
        "n_obs",
        "n_schools",
        "fixed_effects",
        "weighted",
        "exclude_2025",
        "r_squared",
    ]

    table = display[cols]

    lines = []
    lines.append("# Baseline TWFE Regression Results\n")
    lines.append("Main outcome: `eligibility_rate`. Robustness outcomes: `recipient_rate` and `acceptance_rate`.\n")
    lines.append("Standard errors are clustered by NCES school. Weighted models use `students_processed`.\n")
    lines.append(table.to_markdown(index=False))
    lines.append("\nSignificance: * p < 0.10, ** p < 0.05, *** p < 0.01.\n")

    return "\n".join(lines)


def main() -> None:
    """Run baseline regressions."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing regression panel: {INPUT_PATH}")

    df = prepare_panel(INPUT_PATH)

    model_specs = [
        {
            "model_name": "M1_year_fe",
            "outcome": "eligibility_rate",
            "exposure": "exposure_index_pointmax",
            "fixed_effects": "year",
            "weighted": False,
            "exclude_2025": False,
        },
        {
            "model_name": "M2_twfe_unweighted",
            "outcome": "eligibility_rate",
            "exposure": "exposure_index_pointmax",
            "fixed_effects": "school_year",
            "weighted": False,
            "exclude_2025": False,
        },
        {
            "model_name": "M3_twfe_weighted_main",
            "outcome": "eligibility_rate",
            "exposure": "exposure_index_pointmax",
            "fixed_effects": "school_year",
            "weighted": True,
            "exclude_2025": False,
        },
        {
            "model_name": "M4_twfe_weighted_stormmax",
            "outcome": "eligibility_rate",
            "exposure": "exposure_index_stormmax",
            "fixed_effects": "school_year",
            "weighted": True,
            "exclude_2025": False,
        },
        {
            "model_name": "M5_twfe_weighted_100km_hurricane",
            "outcome": "eligibility_rate",
            "exposure": "within_100km_hurricane",
            "fixed_effects": "school_year",
            "weighted": True,
            "exclude_2025": False,
        },
        {
            "model_name": "M6_twfe_weighted_50km_hurricane",
            "outcome": "eligibility_rate",
            "exposure": "within_50km_hurricane",
            "fixed_effects": "school_year",
            "weighted": True,
            "exclude_2025": False,
        },
        {
            "model_name": "M7_twfe_weighted_pointmax_z",
            "outcome": "eligibility_rate",
            "exposure": "exposure_pointmax_z",
            "fixed_effects": "school_year",
            "weighted": True,
            "exclude_2025": False,
        },
        {
            "model_name": "R1_recipient_rate_no2025",
            "outcome": "recipient_rate",
            "exposure": "exposure_index_pointmax",
            "fixed_effects": "school_year",
            "weighted": True,
            "exclude_2025": True,
        },
        {
            "model_name": "R2_acceptance_rate_no2025",
            "outcome": "acceptance_rate",
            "exposure": "exposure_index_pointmax",
            "fixed_effects": "school_year",
            "weighted": True,
            "exclude_2025": True,
        },
    ]

    model_outputs = []
    result_rows = []
    summary_text = []

    for spec in model_specs:
        print(f"Running {spec['model_name']}...")

        model_info = fit_model(df=df, **spec)
        model_outputs.append(model_info)
        result_rows.append(extract_main_result(model_info))

        summary_text.append("=" * 90)
        summary_text.append(spec["model_name"])
        summary_text.append("=" * 90)
        summary_text.append(str(model_info["result"].summary()))
        summary_text.append("\n\n")

    results = pd.DataFrame(result_rows)

    # Detailed model/sample metadata.
    detail_rows = []
    for model_info in model_outputs:
        sample = model_info["sample"]
        detail_rows.append(
            {
                "model_name": model_info["model_name"],
                "formula": model_info["formula"],
                "n_obs": int(model_info["result"].nobs),
                "n_schools": sample["nces_school_id"].nunique(),
                "weighted": model_info["weighted"],
                "fixed_effects": model_info["fixed_effects"],
                "exclude_2025": model_info["exclude_2025"],
                "mean_outcome": sample[model_info["outcome"]].mean(),
                "mean_exposure": sample[model_info["exposure"]].mean(),
                "r_squared": model_info["result"].rsquared,
            }
        )

    details = pd.DataFrame(detail_rows)

    results.to_csv(OUT_RESULTS, index=False)
    details.to_csv(OUT_DETAILS, index=False)
    OUT_TABLE.write_text(make_markdown_table(results))
    OUT_SUMMARIES.write_text("\n".join(summary_text))

    print("\nSaved:")
    print(" -", OUT_RESULTS.relative_to(PROJECT_ROOT))
    print(" -", OUT_DETAILS.relative_to(PROJECT_ROOT))
    print(" -", OUT_TABLE.relative_to(PROJECT_ROOT))
    print(" -", OUT_SUMMARIES.relative_to(PROJECT_ROOT))

    print("\n=== Baseline Results ===")
    print(
        results[
            [
                "model_name",
                "outcome",
                "exposure",
                "coef",
                "std_error",
                "p_value",
                "n_obs",
                "n_schools",
                "fixed_effects",
                "weighted",
                "exclude_2025",
                "r_squared",
            ]
        ].to_string(index=False)
    )

    print("\nPreferred main model:")
    preferred = results[results["model_name"].eq("M3_twfe_weighted_main")].iloc[0]
    print(
        f"Coefficient on exposure_index_pointmax = {preferred['coef']:.4f}, "
        f"SE = {preferred['std_error']:.4f}, "
        f"p = {preferred['p_value']:.3f}, "
        f"N = {int(preferred['n_obs'])}."
    )


if __name__ == "__main__":
    main()
