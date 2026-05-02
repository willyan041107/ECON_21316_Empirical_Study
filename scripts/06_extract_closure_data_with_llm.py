"""
Section 6: Extract school closure data using LLM API.

Goal:
Use OpenRouter API to extract structured school closure information from messy text sources.

Inputs:
- data/raw/closures/text/

Outputs:
- data/intermediate/closure_events_raw.jsonl
- data/intermediate/closure_events_clean.csv
- data/intermediate/school_year_closure_days.csv

Fields to extract:
- source_file
- source_url
- announcement_date
- closure_start_date
- closure_end_date
- school_name
- district_name
- parish
- reason
- hurricane_related
- event_name
- closure_type
- confidence
- raw_quote

Important:
Every LLM-extracted record must keep raw_quote and source information.
Do not let the LLM create the final school-year closure variable without validation.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI


def get_openrouter_client() -> OpenAI:
    """Create an OpenRouter client using the API key stored in .env."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        raise ValueError("Please paste your OpenRouter API key into the .env file.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def main() -> None:
    """Extract structured closure records from raw text files."""
    # TODO: Load text files from data/raw/closures/text/.
    # TODO: Send each text to OpenRouter with a strict JSON extraction prompt.
    # TODO: Save raw JSONL output.
    # TODO: Validate dates and source quotes.
    # TODO: Aggregate closure records to school-year closure days.
    pass


if __name__ == "__main__":
    main()
