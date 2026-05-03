"""
Run baseline TWFE regressions using auto-collected school closure measures.

Input:
    data/processed/regression_panel_with_auto_closures.csv

Outputs:
    data/analysis/closure_twfe_baseline_results.csv
    reports/06b_baseline_closure_twfe_regressions.txt

Purpose:
    Estimate the association between hurricane-related school closure days and
    TOPS outcomes using school and graduation-year fixed effects.

Main outcome:
    eligibility_rate

Main closure variable:
    closure_days_hurricane_related_broad

Robustness:
    strict closure days, any-closure indicator, max consecutive closure days,
    recipient_rate and acceptance_rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "regression_panel_with_auto_closures.csv"

ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_OUT = ANALYSIS_DIR / "closure_twfe_baseline_results.csv"
REPORT_OUT = REPORTS_DIR / "06b_baseline_closure_twfe_regressions.txt"


MAIN_OUTCOME = "eligibility_rate"

ROBUSTNESS_OUTCOMES = [
    "recipient_rate",
    "acceptance_rate",
]

CLOSURE_VARS = [
    "closure_days_hurricane_related_broad",
    "closure_days_hurricane_related_strict",
    "closure_any_hurricane_related_broad",
    "closure_max_consecutive_days_broad",
]

EXPOSURE_CONTROLS = [
    "exposure_index_pointmax",
]


def parse_numeric_or_bool(series: pd.Series) -> pd.Series:
    """Convert numeric and boolean-like columns to numeric."""
    text = series.astype(str).str.strip().str.lower()

    bool_map = text.map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "": np.nan,
            "nan": np.nan,
            "none": np.nan,
            "null": np.nan,
        }
    )

    numeric = pd.to_numeric(series, errors="coerce")

    return numeric.fillna(bool_map)


def normalize_id(value: object, width: int = 12) -> str:
    """Normalize NCES school IDs."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if text.isdigit():
        return text.zfill(width)

    return text


def find_students_column(df: pd.DataFrame) -> str:
    """Find student-count column for WLS weights."""
    candidates = [
        "students_processed",
        "total_students_processed",
        "total_processed",
        "processed_students",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError("Could not find student-count column for weights.")


def prepare_panel(path: Path) -> tuple[pd.DataFrame, str]:
    """Read and clean regression panel."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path, dtype=str, low_memory=False)

    df["nces_school_id"] = df["nces_school_id"].apply(normalize_id)
    df["graduation_year"] = parse_numeric_or_bool(df["graduation_year"]).astype("Int64")

    students_col = find_students_column(df)

    numeric_cols = [
        students_col,
        MAIN_OUTCOME,
        *ROBUSTNESS_OUTCOMES,
        *CLOSURE_VARS,
        *EXPOSURE_CONTROLS,
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = parse_numeric_or_bool(df[col])

    for col in CLOSURE_VARS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df["graduation_year"].notna()].copy()
    df["graduation_year"] = df["graduation_year"].astype(int)

    return df, students_col


def fit_model(
    df: pd.DataFrame,
    model_name: str,
    outcome: str,
    closure_var: str,
    students_col: str,
    fixed_effects: str,
    weighted: bool,
    controls: list[str] | None = None,
    exclude_2025: bool = False,
) -> dict:
    """Fit one OLS/WLS model with clustered standard errors."""
    controls = controls or []

    needed_cols = [
        outcome,
        closure_var,
        "nces_school_id",
        "graduation_year",
        students_col,
        *controls,
    ]

    work = df.copy()

    if exclude_2025:
        work = work[work["graduation_year"] != 2025].copy()

    work = work[needed_cols].dropna().copy()
    work = work[work[students_col] > 0].copy()

    if fixed_effects == "year":
        fe_terms = "C(graduation_year)"
    elif fixed_effects == "school_year":
        fe_terms = "C(nces_school_id) + C(graduation_year)"
    else:
        raise ValueError(f"Unknown fixed_effects: {fixed_effects}")

    control_terms = ""
    if controls:
        control_terms = " + " + " + ".join(controls)

    formula = f"{outcome} ~ {closure_var}{control_terms} + {fe_terms}"

    if weighted:
        model = smf.wls(formula=formula, data=work, weights=work[students_col])
    else:
        model = smf.ols(formula=formula, data=work)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": work["nces_school_id"]},
    )

    coef = result.params.get(closure_var, np.nan)
    se = result.bse.get(closure_var, np.nan)
    p_value = result.pvalues.get(closure_var, np.nan)

    return {
        "model_name": model_name,
        "outcome": outcome,
        "closure_variable": closure_var,
        "coef": coef,
        "std_error": se,
        "p_value": p_value,
        "n_obs": int(result.nobs),
        "n_schools": work["nces_school_id"].nunique(),
        "fixed_effects": fixed_effects,
        "weighted": weighted,
        "controls": "; ".join(controls) if controls else "",
        "exclude_2025": exclude_2025,
        "r_squared": result.rsquared,
        "mean_outcome": work[outcome].mean(),
        "mean_closure_var": work[closure_var].mean(),
        "sd_closure_var": work[closure_var].std(),
    }


def run_models(df: pd.DataFrame, students_col: str) -> pd.DataFrame:
    """Run baseline and robustness models."""
    specs = [
        {
            "model_name": "M1_year_fe_broad_days",
            "outcome": MAIN_OUTCOME,
            "closure_var": "closure_days_hurricane_related_broad",
            "fixed_effects": "year",
            "weighted": False,
            "controls": [],
            "exclude_2025": False,
        },
        {
            "model_name": "M2_twfe_unweighted_broad_days",
            "outcome": MAIN_OUTCOME,
            "closure_var": "closure_days_hurricane_related_broad",
            "fixed_effects": "school_year",
            "weighted": False,
            "controls": [],
            "exclude_2025": False,
        },
        {
            "model_name": "M3_twfe_weighted_broad_days_main",
            "outcome": MAIN_OUTCOME,
            "closure_var": "closure_days_hurricane_related_broad",
            "fixed_effects": "school_year",
            "weighted": True,
            "controls": [],
            "exclude_2025": False,
        },
        {
            "model_name": "M4_twfe_weighted_broad_days_exposure_control",
            "outcome": MAIN_OUTCOME,
            "closure_var": "closure_days_hurricane_related_broad",
            "fixed_effects": "school_year",
            "weighted": True,
            "controls": ["exposure_index_pointmax"],
            "exclude_2025": False,
        },
        {
            "model_name": "M5_twfe_weighted_strict_days",
            "outcome": MAIN_OUTCOME,
            "closure_var": "closure_days_hurricane_related_strict",
            "fixed_effects": "school_year",
            "weighted": True,
            "controls": [],
            "exclude_2025": False,
        },
        {
            "model_name": "M6_twfe_weighted_broad_any",
            "outcome": MAIN_OUTCOME,
            "closure_var": "closure_any_hurricane_related_broad",
            "fixed_effects": "school_year",
            "weighted": True,
            "controls": [],
            "exclude_2025": False,
        },
        {
            "model_name": "M7_twfe_weighted_broad_max_consecutive",
            "outcome": MAIN_OUTCOME,
            "closure_var": "closure_max_consecutive_days_broad",
            "fixed_effects": "school_year",
            "weighted": True,
            "controls": [],
            "exclude_2025": False,
        },
        {
            "model_name": "R1_recipient_rate_no2025_broad_days",
            "outcome": "recipient_rate",
            "closure_var": "closure_days_hurricane_related_broad",
            "fixed_effects": "school_year",
            "weighted": True,
            "controls": [],
            "exclude_2025": True,
        },
        {
            "model_name": "R2_acceptance_rate_no2025_broad_days",
            "outcome": "acceptance_rate",
            "closure_var": "closure_days_hurricane_related_broad",
            "fixed_effects": "school_year",
            "weighted": True,
            "controls": [],
            "exclude_2025": True,
        },
        {
            "model_name": "R3_eligibility_no2025_broad_days",
            "outcome": MAIN_OUTCOME,
            "closure_var": "closure_days_hurricane_related_broad",
            "fixed_effects": "school_year",
            "weighted": True,
            "controls": [],
            "exclude_2025": True,
        },
    ]

    rows = []

    for spec in specs:
        rows.append(
            fit_model(
                df=df,
                students_col=students_col,
                **spec,
            )
        )

    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame) -> None:
    """Write a text report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("06b Baseline Closure TWFE Regressions")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Main outcome: eligibility_rate")
    lines.append("Main closure variable: closure_days_hurricane_related_broad")
    lines.append("Standard errors clustered by NCES school.")
    lines.append("")
    lines.append("Results")
    lines.append("-" * 80)
    lines.append(results.to_string(index=False))
    lines.append("")

    main = results[results["model_name"].eq("M3_twfe_weighted_broad_days_main")]

    if not main.empty:
        row = main.iloc[0]
        lines.append("Preferred main model")
        lines.append("-" * 80)
        lines.append(
            f"Coefficient on broad closure days = {row['coef']:.6f}, "
            f"SE = {row['std_error']:.6f}, p = {row['p_value']:.4f}, "
            f"N = {int(row['n_obs'])}, schools = {int(row['n_schools'])}."
        )
        lines.append(
            "Interpretation: one additional hurricane-related school closure day "
            "is associated with this change in eligibility_rate, conditional on "
            "school and graduation-year fixed effects."
        )
        lines.append("")

    REPORT_OUT.write_text("\n".join(lines))


def main() -> None:
    """Run baseline closure TWFE regressions."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df, students_col = prepare_panel(INPUT_PATH)
    results = run_models(df, students_col)

    results.to_csv(RESULTS_OUT, index=False)
    write_report(results)

    print("Saved:")
    print(" -", RESULTS_OUT.relative_to(PROJECT_ROOT))
    print(" -", REPORT_OUT.relative_to(PROJECT_ROOT))

    print()
    print("=== Baseline Closure TWFE Results ===")
    print(results.to_string(index=False))

    main = results[results["model_name"].eq("M3_twfe_weighted_broad_days_main")]

    if not main.empty:
        row = main.iloc[0]
        print()
        print("Preferred main model:")
        print(
            f"Coefficient on closure_days_hurricane_related_broad = {row['coef']:.6f}, "
            f"SE = {row['std_error']:.6f}, p = {row['p_value']:.4f}, "
            f"N = {int(row['n_obs'])}."
        )


if __name__ == "__main__":
    main()
