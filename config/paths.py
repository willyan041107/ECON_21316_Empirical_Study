"""
Centralized project paths.

All scripts should import paths from this file instead of hard-coding paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"

LOSFA_RAW_DIR = RAW_DIR / "losfa"
NCES_RAW_DIR = RAW_DIR / "nces"
HURDAT2_RAW_DIR = RAW_DIR / "hurdat2"
CLOSURE_RAW_DIR = RAW_DIR / "closures"
FAFSA_RAW_DIR = RAW_DIR / "fafsa"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
