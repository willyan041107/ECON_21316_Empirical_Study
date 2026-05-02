# ECON_21316_Empirical_Study

This project studies whether hurricane exposure reduces Louisiana high school students' probability of qualifying for and receiving TOPS merit-based scholarships.

## Main Unit of Observation

School × graduation cohort year.

## Data Sections

1. LOSFA TOPS outcome data
2. NCES CCD school identifiers and school characteristics
3. NOAA HURDAT2 hurricane track data
4. School-level hurricane exposure measures
5. School closure / disruption data using LLM extraction
6. FAFSA completion and mechanism data
7. Final analysis panel
8. Quality checks and empirical outputs

## Important Rule

LLM API should be used for text extraction, classification, and fuzzy matching support. It should not directly create final treatment variables, outcome rates, or regression-ready data without rule-based validation.
