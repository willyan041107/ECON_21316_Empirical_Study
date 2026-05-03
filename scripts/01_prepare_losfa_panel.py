"""
Section 1: Prepare LOSFA TOPS outcome panel.

This script parses LOSFA "TOPS Eligible and Recipients by HS Graduation Year"
PDF files and constructs a clean school-by-graduation-year outcome panel.

Main unit:
    high school × HS graduation year

Main outcome:
    eligibility_rate = total_eligible / students_processed

Robustness outcomes:
    recipient_rate = total_recipients / students_processed
    acceptance_rate = total_recipients / total_eligible

Supported LOSFA table formats:
    Older format:
        Students Processed + 5 award groups × 4 tokens = 21 tokens
        Award groups: Opportunity, Performance, Honors, TOPS Tech, Total

    Newer format:
        Students Processed + 6 award groups × 4 tokens = 25 tokens
        Award groups: Opportunity, Performance, Honors, Excellence, TOPS Tech, Total

Inputs:
    data/raw/losfa/TOPS_BY_HS_GY_*.pdf

Outputs:
    data/intermediate/losfa_panel_clean.csv
    data/intermediate/losfa_parse_review.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pdfplumber


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSFA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "losfa"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

EXPECTED_TAIL_LENGTHS = [25, 21]

LOUISIANA_PARISHES = [
    "ACADIA",
    "ALLEN",
    "ASCENSION",
    "ASSUMPTION",
    "AVOYELLES",
    "BEAUREGARD",
    "BIENVILLE",
    "BOSSIER",
    "CADDO",
    "CALCASIEU",
    "CALDWELL",
    "CAMERON",
    "CATAHOULA",
    "CLAIBORNE",
    "CONCORDIA",
    "DE SOTO",
    "EAST BATON ROUGE",
    "EAST CARROLL",
    "EAST FELICIANA",
    "EVANGELINE",
    "FRANKLIN",
    "GRANT",
    "IBERIA",
    "IBERVILLE",
    "JACKSON",
    "JEFFERSON",
    "JEFFERSON DAVIS",
    "LAFAYETTE",
    "LAFOURCHE",
    "LA SALLE",
    "LINCOLN",
    "LIVINGSTON",
    "MADISON",
    "MOREHOUSE",
    "NATCHITOCHES",
    "ORLEANS",
    "OUACHITA",
    "PLAQUEMINES",
    "POINTE COUPEE",
    "RAPIDES",
    "RED RIVER",
    "RICHLAND",
    "SABINE",
    "ST. BERNARD",
    "ST. CHARLES",
    "ST. HELENA",
    "ST. JAMES",
    "ST. JOHN THE BAPTIST",
    "ST. LANDRY",
    "ST. MARTIN",
    "ST. MARY",
    "ST. TAMMANY",
    "TANGIPAHOA",
    "TENSAS",
    "TERREBONNE",
    "UNION",
    "VERMILION",
    "VERNON",
    "WASHINGTON",
    "WEBSTER",
    "WEST BATON ROUGE",
    "WEST CARROLL",
    "WEST FELICIANA",
    "WINN",
]


def normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace in a string."""
    return " ".join(text.strip().split())


def build_parish_prefixes() -> list[tuple[str, str]]:
    """
    Build possible PDF parish prefixes.

    Each tuple is:
        possible PDF prefix, canonical parish name.
    """
    prefixes = []

    for parish in LOUISIANA_PARISHES:
        prefixes.append((parish, parish))

        if parish.startswith("ST. "):
            rest = parish.removeprefix("ST. ")
            prefixes.append((f"SAINT {rest}", parish))
            prefixes.append((f"ST {rest}", parish))

    return sorted(prefixes, key=lambda item: len(item[0]), reverse=True)


PARISH_PREFIXES_BY_LENGTH = build_parish_prefixes()


def extract_graduation_year_from_filename(pdf_path: Path) -> int:
    """Extract graduation cohort year from a LOSFA filename."""
    match = re.search(r"(\d{4})", pdf_path.stem)
    if not match:
        raise ValueError(f"Could not extract year from filename: {pdf_path.name}")
    return int(match.group(1))


def is_header_or_footer(line: str) -> bool:
    """Return True if a line is a repeated header, footer, or timestamp."""
    header_starts = (
        "TOPS Eligible and Recipients",
        "Graduation Year",
        "AWARD NAME",
        "Parish HSName",
        "Processed",
    )

    if any(line.startswith(prefix) for prefix in header_starts):
        return True

    if re.match(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+(AM|PM)", line):
        return True

    return False


def is_subtotal_or_total_row(line: str) -> bool:
    """Return True if a line is a subtotal or total row."""
    lowered = line.lower()
    return (
        lowered.startswith("subtotal")
        or lowered.startswith("total")
        or lowered.startswith("grand total")
        or lowered.startswith("state total")
    )


def looks_like_continuation_line(line: str) -> bool:
    """Return True if a line likely belongs to a broken school name."""
    if not line:
        return False

    if is_header_or_footer(line) or is_subtotal_or_total_row(line):
        return False

    if any(char.isdigit() for char in line):
        return False

    return bool(re.fullmatch(r"[A-Z&.'()/,\-\s]+", line))


def is_numeric_like(token: str) -> bool:
    """Check whether a token looks like a number, percent, or NaN."""
    token = token.strip()

    if token in {"NaN", "∞", "inf", "INF"}:
        return True

    if re.fullmatch(r"-?\d{1,3}(,\d{3})*", token):
        return True

    if re.fullmatch(r"-?\d+(\.\d+)?", token):
        return True

    if re.fullmatch(r"-?\d+(\.\d+)?%", token):
        return True

    return False


def parse_int(token: str) -> int | None:
    """Convert a numeric token to int."""
    token = token.replace(",", "").strip()

    if token in {"NaN", "∞", "inf", "INF"}:
        return None

    try:
        return int(float(token))
    except ValueError:
        return None


def parse_percent(token: str) -> float | None:
    """Convert a percent token such as '69.6%' to decimal form."""
    token = token.replace("%", "").strip()

    if token == "NaN":
        return None

    try:
        return float(token) / 100
    except ValueError:
        return None


def split_prefix_and_tail(line: str) -> tuple[str, list[str], int] | None:
    """
    Split a LOSFA data row into text prefix and numeric tail.

    Supports both:
        25-token newer format
        21-token older format
    """
    tokens = line.split()

    for tail_length in EXPECTED_TAIL_LENGTHS:
        if len(tokens) < tail_length + 1:
            continue

        tail = tokens[-tail_length:]
        prefix_tokens = tokens[:-tail_length]

        if all(is_numeric_like(token) for token in tail) and parse_int(tail[0]) is not None:
            return " ".join(prefix_tokens), tail, tail_length

    return None


def strip_parish_prefix(
    prefix: str,
    current_parish: str | None,
    allow_new_parish: bool,
) -> tuple[str | None, str, bool]:
    """
    Identify parish prefix while avoiding parish-name collisions.

    A new parish is only recognized when:
        1. It is the first school row, or
        2. A SubTotal row was just seen.

    This prevents school names like MADISON PREPARATORY ACADEMY from being
    mistaken as a new MADISON parish section.
    """
    normalized_prefix = normalize_spaces(prefix)

    # Fix duplicate parish prefix inside the same group.
    # Example:
    #     NATCHITOCHES NATCHITOCHES CENTRAL HIGH SCHOOL
    # should become:
    #     NATCHITOCHES CENTRAL HIGH SCHOOL
    if current_parish is not None:
        current_aliases = [
            alias
            for alias, canonical in PARISH_PREFIXES_BY_LENGTH
            if canonical == current_parish
        ]

        for alias in current_aliases:
            if normalized_prefix == alias or normalized_prefix.startswith(alias + " "):
                remainder = normalized_prefix[len(alias):].strip()

                for second_alias in current_aliases:
                    if remainder == second_alias or remainder.startswith(second_alias + " "):
                        return current_parish, remainder, False

    if allow_new_parish or current_parish is None:
        for alias, canonical in PARISH_PREFIXES_BY_LENGTH:
            if normalized_prefix == alias or normalized_prefix.startswith(alias + " "):
                school_part = normalized_prefix[len(alias):].strip()
                parish_switched = canonical != current_parish
                return canonical, school_part, parish_switched

    return current_parish, normalized_prefix, False


def split_school_and_type(school_part: str) -> tuple[str, str, str]:
    """Split school name and high-school type."""
    tokens = school_part.split()

    if not tokens:
        return "", "UNKNOWN", "missing_school_name"

    if tokens[-1] in {"PUBLIC", "NONPUBLIC"}:
        hs_type = tokens[-1]
        hs_name = " ".join(tokens[:-1])
        return hs_name, hs_type, "ok"

    return school_part, "UNKNOWN", "missing_hstype"


def parse_data_tail(tail: list[str], tail_length: int) -> dict[str, int | float | str | None]:
    """Extract numeric fields from either old or new LOSFA tail format."""
    students_processed = parse_int(tail[0])

    opportunity_eligible = parse_int(tail[1])
    opportunity_recipients = parse_int(tail[3])

    performance_eligible = parse_int(tail[5])
    performance_recipients = parse_int(tail[7])

    honors_eligible = parse_int(tail[9])
    honors_recipients = parse_int(tail[11])

    if tail_length == 25:
        table_format = "new_25_token"
        excellence_eligible = parse_int(tail[13])
        excellence_recipients = parse_int(tail[15])

        topstech_eligible = parse_int(tail[17])
        topstech_recipients = parse_int(tail[19])

        total_eligible = parse_int(tail[21])
        pdf_total_percent_eligible = parse_percent(tail[22])
        total_recipients = parse_int(tail[23])
        pdf_total_percent_acceptance = parse_percent(tail[24])

    elif tail_length == 21:
        table_format = "old_21_token"
        excellence_eligible = None
        excellence_recipients = None

        topstech_eligible = parse_int(tail[13])
        topstech_recipients = parse_int(tail[15])

        total_eligible = parse_int(tail[17])
        pdf_total_percent_eligible = parse_percent(tail[18])
        total_recipients = parse_int(tail[19])
        pdf_total_percent_acceptance = parse_percent(tail[20])

    else:
        raise ValueError(f"Unsupported tail length: {tail_length}")

    return {
        "table_format": table_format,
        "students_processed": students_processed,
        "opportunity_eligible": opportunity_eligible,
        "opportunity_recipients": opportunity_recipients,
        "performance_eligible": performance_eligible,
        "performance_recipients": performance_recipients,
        "honors_eligible": honors_eligible,
        "honors_recipients": honors_recipients,
        "excellence_eligible": excellence_eligible,
        "excellence_recipients": excellence_recipients,
        "topstech_eligible": topstech_eligible,
        "topstech_recipients": topstech_recipients,
        "total_eligible": total_eligible,
        "pdf_total_percent_eligible": pdf_total_percent_eligible,
        "total_recipients": total_recipients,
        "pdf_total_percent_acceptance": pdf_total_percent_acceptance,
    }


def safe_divide(numerator: int | float | None, denominator: int | float | None) -> float | None:
    """Divide safely, returning None if denominator is zero or missing."""
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
        return None

    if denominator == 0:
        return None

    return numerator / denominator


def make_quality_flag(row: dict) -> str:
    """Create a compact quality flag for suspicious parsed rows."""
    flags = []

    if row["parish"] is None:
        flags.append("missing_parish")

    if row["hs_name"] == "":
        flags.append("missing_school_name")

    if row["hs_type"] == "UNKNOWN":
        flags.append("missing_hstype")

    processed = row["students_processed"]
    eligible = row["total_eligible"]
    recipients = row["total_recipients"]

    if processed is None:
        flags.append("missing_processed")

    if eligible is None:
        flags.append("missing_total_eligible")

    if recipients is None:
        flags.append("missing_total_recipients")

    if processed is not None and eligible is not None and eligible > processed:
        flags.append("eligible_gt_processed")

    if processed is not None and recipients is not None and recipients > processed:
        flags.append("recipients_gt_processed")

    if eligible is not None and recipients is not None and recipients > eligible:
        flags.append("recipients_gt_eligible")

    return ";".join(flags) if flags else "ok"


def recalculate_outcomes(row: dict) -> dict:
    """Recalculate core outcome rates after parsing or aggregation."""
    row["eligibility_rate"] = safe_divide(row["total_eligible"], row["students_processed"])
    row["recipient_rate"] = safe_divide(row["total_recipients"], row["students_processed"])
    row["acceptance_rate"] = safe_divide(row["total_recipients"], row["total_eligible"])
    row["is_public"] = row["hs_type"] == "PUBLIC"
    row["data_quality_flag"] = make_quality_flag(row)
    return row


def parse_pdf(pdf_path: Path) -> tuple[list[dict], list[dict]]:
    """Parse one LOSFA PDF into school-level records."""
    graduation_year = extract_graduation_year_from_filename(pdf_path)

    records = []
    review_rows = []

    current_parish = None
    allow_new_parish = True
    pending_prefix_lines: list[str] = []
    last_record_index: int | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""

            for raw_line in text.splitlines():
                line = normalize_spaces(raw_line)

                if not line:
                    continue

                if is_header_or_footer(line):
                    continue

                if is_subtotal_or_total_row(line):
                    allow_new_parish = True
                    pending_prefix_lines = []
                    continue

                split_result = split_prefix_and_tail(line)

                if split_result is None:
                    if looks_like_continuation_line(line):
                        # If the previous parsed school name clearly ended mid-name,
                        # append this fragment to the previous record.
                        if (
                            last_record_index is not None
                            and line in {"SCHOOL", "ACADEMY", "CENTER"}
                            and records[last_record_index]["hs_name"].endswith(("HIGH", "SCHOO", "SC"))
                        ):
                            records[last_record_index]["hs_name"] = normalize_spaces(
                                records[last_record_index]["hs_name"] + " " + line
                            )
                            records[last_record_index]["hs_name_clean"] = records[last_record_index][
                                "hs_name"
                            ].upper()
                        else:
                            pending_prefix_lines.append(line)
                    else:
                        review_rows.append(
                            {
                                "source_file": pdf_path.name,
                                "graduation_year": graduation_year,
                                "page_number": page_number,
                                "raw_line": line,
                                "issue": "could_not_parse_line",
                            }
                        )
                    continue

                prefix, tail, tail_length = split_result

                if pending_prefix_lines:
                    prefix = normalize_spaces(" ".join(pending_prefix_lines + [prefix]))
                    pending_prefix_lines = []

                current_parish, school_part, parish_switched = strip_parish_prefix(
                    prefix=prefix,
                    current_parish=current_parish,
                    allow_new_parish=allow_new_parish,
                )

                hs_name, hs_type, parse_status = split_school_and_type(school_part)

                row = {
                    "graduation_year": graduation_year,
                    "source_file": pdf_path.name,
                    "page_number": page_number,
                    "parish": current_parish,
                    "hs_name": hs_name,
                    "hs_name_clean": hs_name.upper(),
                    "hs_type": hs_type,
                    "parse_status": parse_status,
                    "parish_switched_on_this_row": parish_switched,
                }

                row.update(parse_data_tail(tail, tail_length))
                row = recalculate_outcomes(row)

                records.append(row)
                last_record_index = len(records) - 1

                allow_new_parish = False

    if pending_prefix_lines:
        review_rows.append(
            {
                "source_file": pdf_path.name,
                "graduation_year": graduation_year,
                "page_number": None,
                "raw_line": " ".join(pending_prefix_lines),
                "issue": "unused_pending_prefix_line",
            }
        )

    return records, review_rows


def choose_hs_type(values: pd.Series) -> str:
    """
    Choose one high-school type when duplicate school-year rows are aggregated.

    Priority:
        PUBLIC > NONPUBLIC > UNKNOWN
    """
    clean_values = set(values.dropna().astype(str))

    if "PUBLIC" in clean_values:
        return "PUBLIC"

    if "NONPUBLIC" in clean_values:
        return "NONPUBLIC"

    return "UNKNOWN"


def aggregate_duplicate_school_year_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate duplicate school-year rows.

    The empirical unit is school × graduation year. Some LOSFA files contain
    repeated rows for the same school in the same year, sometimes with one row
    marked PUBLIC and another marked UNKNOWN. Counts are summed and rates are
    recalculated.
    """
    if df.empty:
        return df

    count_columns = [
        "students_processed",
        "opportunity_eligible",
        "opportunity_recipients",
        "performance_eligible",
        "performance_recipients",
        "honors_eligible",
        "honors_recipients",
        "excellence_eligible",
        "excellence_recipients",
        "topstech_eligible",
        "topstech_recipients",
        "total_eligible",
        "total_recipients",
    ]

    group_columns = [
        "graduation_year",
        "source_file",
        "parish",
        "hs_name_clean",
        "hs_name",
    ]

    grouped = (
        df.groupby(group_columns, dropna=False)
        .agg(
            {
                **{column: "sum" for column in count_columns},
                "hs_type": choose_hs_type,
                "page_number": "min",
                "table_format": lambda x: ";".join(sorted(set(x.dropna().astype(str)))),
                "parse_status": lambda x: ";".join(sorted(set(x.dropna().astype(str)))),
                "parish_switched_on_this_row": "max",
            }
        )
        .reset_index()
    )

    grouped["pdf_total_percent_eligible"] = pd.NA
    grouped["pdf_total_percent_acceptance"] = pd.NA

    recalculated_rows = []
    for row in grouped.to_dict("records"):
        recalculated_rows.append(recalculate_outcomes(row))

    out = pd.DataFrame(recalculated_rows)

    ordered_columns = [
        "graduation_year",
        "source_file",
        "page_number",
        "parish",
        "hs_name",
        "hs_name_clean",
        "hs_type",
        "parse_status",
        "table_format",
        "parish_switched_on_this_row",
        "students_processed",
        "opportunity_eligible",
        "opportunity_recipients",
        "performance_eligible",
        "performance_recipients",
        "honors_eligible",
        "honors_recipients",
        "excellence_eligible",
        "excellence_recipients",
        "topstech_eligible",
        "topstech_recipients",
        "total_eligible",
        "total_recipients",
        "pdf_total_percent_eligible",
        "pdf_total_percent_acceptance",
        "eligibility_rate",
        "recipient_rate",
        "acceptance_rate",
        "is_public",
        "data_quality_flag",
    ]

    return out[ordered_columns]


def parse_all_pdfs(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse every LOSFA PDF in the raw data directory."""
    pdf_files = sorted(raw_dir.glob("TOPS_BY_HS_GY_*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    all_records = []
    all_review_rows = []

    for pdf_path in pdf_files:
        print(f"Parsing {pdf_path.name}...")
        records, review_rows = parse_pdf(pdf_path)
        all_records.extend(records)
        all_review_rows.extend(review_rows)

    df = pd.DataFrame(all_records)
    review_df = pd.DataFrame(all_review_rows)

    df = aggregate_duplicate_school_year_rows(df)

    return df, review_df


def save_outputs(df: pd.DataFrame, review_df: pd.DataFrame) -> None:
    """Save clean panel and review file."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = INTERMEDIATE_DIR / "losfa_panel_clean.csv"
    review_path = INTERMEDIATE_DIR / "losfa_parse_review.csv"

    df.to_csv(clean_path, index=False)
    review_df.to_csv(review_path, index=False)

    print()
    print("Saved clean panel to:", clean_path)
    print("Saved parse review file to:", review_path)
    print()
    print("Rows parsed:", len(df))

    if not df.empty:
        print("Unique graduation years:", df["graduation_year"].nunique())
        print("Rows needing review:", (df["data_quality_flag"] != "ok").sum())

        print()
        print("Rows by graduation year:")
        print(df["graduation_year"].value_counts().sort_index())

        print()
        print("HS type counts:")
        print(df["hs_type"].value_counts(dropna=False))

        print()
        print("Table format counts:")
        print(df["table_format"].value_counts(dropna=False))

        print()
        print("Data quality flag counts:")
        print(df["data_quality_flag"].value_counts(dropna=False).head(20))


def main() -> None:
    """Run the LOSFA parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--one-file",
        type=str,
        default=None,
        help="Optional single PDF filename inside data/raw/losfa/ to parse first.",
    )
    args = parser.parse_args()

    if args.one_file:
        pdf_path = LOSFA_RAW_DIR / args.one_file
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        records, review_rows = parse_pdf(pdf_path)
        df = pd.DataFrame(records)
        df = aggregate_duplicate_school_year_rows(df)
        review_df = pd.DataFrame(review_rows)
    else:
        df, review_df = parse_all_pdfs(LOSFA_RAW_DIR)

    save_outputs(df, review_df)


if __name__ == "__main__":
    main()
