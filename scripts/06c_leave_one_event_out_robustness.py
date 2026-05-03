"""
Run leave-one-event-out robustness checks for closure TWFE regressions.

Input:
    data/processed/regression_panel_with_auto_closures.csv

Outputs:
    data/analysis/closure_leave_one_event_out_results.csv
    reports/06c_leave_one_event_out_robustness.txt

Purpose:
    Check whether the baseline closure-days coefficient is driven by one major
    hurricane event.

Main model:
    eligibility_rate ~ closure_days_hurricane_related_broad
        + school fixed effects
        + graduation-year fixed effects

Weights:
    students_processed

Standard errors:
    clustered by NCES school
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "regression_panel_with_auto_closures.csv"

ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_OUT = ANALYSIS_DIR / "closure_leave_one_event_out_results.csv"
REPORT_OUT = REPORTS_DIR / "06c_leave_one_event_out_robustness.txt"


OUTCOME = "eligibility_rate"
MAIN_CLOSURE_VAR = "closure_days_hurricane_related_broad"
STUDENT_COL_CANDIDATES = [
    "students_processed",
    "total_students_processed",
    "total_processed",
    "processed_students",
]

EVENT_YEAR_MAP = {
    "2005_katrina_rita": 2006,
    "2020_laura_delta_zeta": 2021,
    "2021_ida": 2022,
    "2024_francine": 2025,
}


def normalize_id(value: object, width: int = 12) -> str:
    """Normalize NCES school ID as string."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if text.isdigit():
        return text.zfill(width)

    return text


def parse_numeric_or_bool(series: pd.Series) -> pd.Series:
    """Convert numeric and boolean-like string columns to numeric."""
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


def find_students_column(df: pd.DataFrame) -> str:
    """Find student-count column for weights."""
    for col in STUDENT_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError("Could not find student-count column.")


def contains_event(value: object, event_id: str) -> bool:
    """Return True if semicolon-separated event list contains event_id."""
    if pd.isna(value):
        return False

    pieces = [piece.strip() for piece in str(value).split(";") if piece.strip()]

    return event_id in pieces


def safe_event_var_name(event_id: str) -> str:
    """Create safe variable suffix from event ID."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", event_id)


def prepare_panel(path: Path) -> tuple[pd.DataFrame, str]:
    """Read and clean panel."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path, dtype=str, low_memory=False)

    df["nces_school_id"] = df["nces_school_id"].apply(normalize_id)

    students_col = find_students_column(df)

    numeric_cols = [
        "graduation_year",
        students_col,
        OUTCOME,
        MAIN_CLOSURE_VAR,
        "closure_days_hurricane_related_strict",
        "closure_any_hurricane_related_broad",
        "closure_max_consecutive_days_broad",
        "exposure_index_pointmax",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = parse_numeric_or_bool(df[col])

    df["graduation_year"] = df["graduation_year"].astype("Int64")
    df = df[df["graduation_year"].notna()].copy()
    df["graduation_year"] = df["graduation_year"].astype(int)

    df[MAIN_CLOSURE_VAR] = pd.to_numeric(df[MAIN_CLOSURE_VAR], errors="coerce").fillna(0)

    if "closure_event_ids_broad" not in df.columns:
        df["closure_event_ids_broad"] = ""

    df["closure_event_ids_broad"] = df["closure_event_ids_broad"].fillna("").astype(str)

    return df, students_col


def fit_twfe(
    df: pd.DataFrame,
    model_name: str,
    closure_var: str,
    students_col: str,
    sample_description: str,
    excluded_event: str = "",
    excluded_year: int | None = None,
) -> dict:
    """Fit weighted school-year TWFE model."""
    needed_cols = [
        OUTCOME,
        closure_var,
        "nces_school_id",
        "graduation_year",
        students_col,
    ]

    work = df[needed_cols].dropna().copy()
    work = work[work[students_col] > 0].copy()

    formula = f"{OUTCOME} ~ {closure_var} + C(nces_school_id) + C(graduation_year)"

    model = smf.wls(
        formula=formula,
        data=work,
        weights=work[students_col],
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": work["nces_school_id"]},
    )

    coef = result.params.get(closure_var, np.nan)
    se = result.bse.get(closure_var, np.nan)
    p_value = result.pvalues.get(closure_var, np.nan)

    return {
        "model_name": model_name,
        "sample_description": sample_description,
        "excluded_event": excluded_event,
        "excluded_graduation_year": excluded_year,
        "outcome": OUTCOME,
        "closure_variable": closure_var,
        "coef": coef,
        "std_error": se,
        "p_value": p_value,
        "n_obs": int(result.nobs),
        "n_schools": work["nces_school_id"].nunique(),
        "r_squared": result.rsquared,
        "adjusted_r_squared": result.rsquared_adj,
        "mean_outcome": work[OUTCOME].mean(),
        "mean_closure_var": work[closure_var].mean(),
        "sd_closure_var": work[closure_var].std(),
        "closure_rows": int((work[closure_var] > 0).sum()),
        "total_closure_days": float(work[closure_var].sum()),
    }


def run_leave_one_event_out(df: pd.DataFrame, students_col: str) -> pd.DataFrame:
    """Run baseline plus leave-one-event-out models."""
    rows = []

    # Baseline preferred model.
    rows.append(
        fit_twfe(
            df=df,
            model_name="baseline_full_sample",
            closure_var=MAIN_CLOSURE_VAR,
            students_col=students_col,
            sample_description="Full sample, broad closure days",
        )
    )

    for event_id, grad_year in EVENT_YEAR_MAP.items():
        safe_event = safe_event_var_name(event_id)

        event_mask = df["closure_event_ids_broad"].apply(lambda x: contains_event(x, event_id))

        # Version 1: keep full sample, set this event's closure days to zero.
        zeroed = df.copy()
        leaveout_var = f"closure_days_leaveout_{safe_event}"
        zeroed[leaveout_var] = zeroed[MAIN_CLOSURE_VAR]
        zeroed.loc[event_mask, leaveout_var] = 0

        rows.append(
            fit_twfe(
                df=zeroed,
                model_name=f"zero_out_{event_id}",
                closure_var=leaveout_var,
                students_col=students_col,
                sample_description=f"Full sample, closure days from {event_id} set to zero",
                excluded_event=event_id,
                excluded_year=grad_year,
            )
        )

        # Version 2: drop treated rows associated with this event.
        dropped_treated = df[~event_mask].copy()

        rows.append(
            fit_twfe(
                df=dropped_treated,
                model_name=f"drop_treated_rows_{event_id}",
                closure_var=MAIN_CLOSURE_VAR,
                students_col=students_col,
                sample_description=f"Drop rows whose broad closure event includes {event_id}",
                excluded_event=event_id,
                excluded_year=grad_year,
            )
        )

        # Version 3: drop the entire graduation year for this event.
        dropped_year = df[df["graduation_year"] != grad_year].copy()

        rows.append(
            fit_twfe(
                df=dropped_year,
                model_name=f"drop_graduation_year_{grad_year}_{event_id}",
                closure_var=MAIN_CLOSURE_VAR,
                students_col=students_col,
                sample_description=f"Drop all rows from graduation_year {grad_year}",
                excluded_event=event_id,
                excluded_year=grad_year,
            )
        )

    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame) -> None:
    """Write text report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("06c Leave-One-Event-Out Robustness")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Preferred baseline model:")
    lines.append("eligibility_rate ~ closure_days_hurricane_related_broad + school FE + year FE")
    lines.append("Weighted by students_processed; SE clustered by NCES school.")
    lines.append("")
    lines.append("Results")
    lines.append("-" * 80)
    lines.append(results.to_string(index=False))
    lines.append("")

    baseline = results[results["model_name"].eq("baseline_full_sample")]

    if not baseline.empty:
        row = baseline.iloc[0]
        lines.append("Baseline reference")
        lines.append("-" * 80)
        lines.append(
            f"Baseline coefficient = {row['coef']:.6f}, "
            f"SE = {row['std_error']:.6f}, "
            f"p = {row['p_value']:.4f}, "
            f"R2 = {row['r_squared']:.4f}, "
            f"Adj. R2 = {row['adjusted_r_squared']:.4f}."
        )
        lines.append("")

    lines.append("Interpretation guide")
    lines.append("-" * 80)
    lines.append(
        "If coefficients remain negative after excluding individual events, "
        "the main result is not mechanically driven by a single hurricane event. "
        "If the coefficient changes sharply after excluding one event, that event "
        "is an important source of identifying variation."
    )
    lines.append("")

    REPORT_OUT.write_text("\n".join(lines))


def main() -> None:
    """Run leave-one-event-out robustness."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df, students_col = prepare_panel(INPUT_PATH)

    results = run_leave_one_event_out(df, students_col)

    results.to_csv(RESULTS_OUT, index=False)
    write_report(results)

    print("Saved:")
    print(" -", RESULTS_OUT.relative_to(PROJECT_ROOT))
    print(" -", REPORT_OUT.relative_to(PROJECT_ROOT))

    print()
    print("=== Leave-One-Event-Out Results ===")
    display_cols = [
        "model_name",
        "excluded_event",
        "excluded_graduation_year",
        "coef",
        "std_error",
        "p_value",
        "n_obs",
        "n_schools",
        "r_squared",
        "adjusted_r_squared",
        "closure_rows",
        "total_closure_days",
    ]
    print(results[display_cols].to_string(index=False))

    print()
    print("Compact coefficient comparison:")
    compact = results[
        [
            "model_name",
            "excluded_event",
            "coef",
            "std_error",
            "p_value",
            "closure_rows",
            "total_closure_days",
        ]
    ].copy()
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
