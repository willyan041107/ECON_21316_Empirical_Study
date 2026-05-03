"""
Automatically collect candidate school-closure sources using Search API + OpenRouter.

Inputs:
    data/intermediate/closure_source_priority_queue_top.csv

Outputs:
    data/intermediate/closure_auto_source_candidates.csv
    data/intermediate/closure_auto_source_screening.csv
    data/raw/closures/auto_sources/*.txt

Required environment variables:
    BRAVE_SEARCH_API_KEY
    OPENROUTER_API_KEY
    OPENROUTER_MODEL

Purpose:
    This script searches for closure-related webpages, fetches text, and asks
    OpenRouter to screen whether each source contains usable school closure
    information.

Important:
    This script collects source candidates. It does not yet create final closure
    records. The next step will extract structured closure dates from approved
    source texts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUEUE_PATH = PROJECT_ROOT / "data" / "intermediate" / "closure_source_priority_queue_top.csv"

RAW_AUTO_DIR = PROJECT_ROOT / "data" / "raw" / "closures" / "auto_sources"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

CANDIDATES_OUT = INTERMEDIATE_DIR / "closure_auto_source_candidates.csv"
SCREENING_OUT = INTERMEDIATE_DIR / "closure_auto_source_screening.csv"

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "openai/gpt-4o-mini"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

BLOCKED_DOMAINS = {
    "facebook.com",
    "www.facebook.com",
    "x.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "www.youtube.com",
    "linkedin.com",
    "www.linkedin.com",
}


def load_project_env() -> None:
    """Load project .env with override=True."""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=True)


def safe_filename(value: str, max_len: int = 120) -> str:
    """Create filesystem-safe filename."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")

    if len(text) > max_len:
        text = text[:max_len]

    return text or "source"


def url_hash(url: str) -> str:
    """Short stable hash for URL."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def is_blocked_url(url: str) -> bool:
    """Skip social/media domains that are usually hard to parse."""
    domain = urlparse(url).netloc.lower().replace("m.", "www.")

    return domain in BLOCKED_DOMAINS


def brave_search(query: str, api_key: str, count: int = 10) -> list[dict]:
    """Run Brave Search API query."""
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        "User-Agent": USER_AGENT,
    }

    params = {
        "q": query,
        "count": count,
        "search_lang": "en",
        "country": "us",
        "safesearch": "moderate",
    }

    response = requests.get(
        BRAVE_SEARCH_URL,
        headers=headers,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("web", {}).get("results", [])

    return results


def fetch_page_text(url: str) -> tuple[str, str]:
    """
    Fetch webpage text.

    Returns:
        title, text
    """
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if "pdf" in content_type:
        return "", ""

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return title, text


def likely_relevant_text(text: str) -> bool:
    """Cheap keyword filter before calling OpenRouter."""
    lower = text.lower()

    school_terms = [
        "school",
        "schools",
        "students",
        "classes",
        "campus",
        "district",
        "parish",
    ]

    closure_terms = [
        "closed",
        "closure",
        "closures",
        "reopen",
        "reopened",
        "cancel",
        "canceled",
        "cancelled",
        "dismiss",
        "remote",
        "virtual",
    ]

    storm_terms = [
        "hurricane",
        "tropical storm",
        "storm",
        "ida",
        "laura",
        "delta",
        "zeta",
        "katrina",
        "rita",
        "gustav",
        "ike",
        "isaac",
        "francine",
        "barry",
        "lili",
        "isidore",
    ]

    return (
        any(term in lower for term in school_terms)
        and any(term in lower for term in closure_terms)
        and any(term in lower for term in storm_terms)
    )


def call_openrouter_screening(
    source_text: str,
    metadata: dict,
    api_key: str,
    model: str,
) -> dict:
    """Ask OpenRouter to screen source relevance and extract minimal metadata."""
    clipped = source_text[:12000]

    system_prompt = """
You screen webpages for a school-closure dataset.

Return only valid JSON. Do not include markdown.

A usable source must explicitly mention school closures, school reopening,
school cancellation, virtual/remote learning because of a storm, or specific
closure dates. General emergency declarations are not enough unless school
closure is stated.

Focus on Louisiana K-12 public schools, districts, school boards, or parish
school systems.
"""

    user_prompt = {
        "task": "Screen this source for hurricane-related Louisiana K-12 school closure data.",
        "metadata": metadata,
        "return_schema": {
            "usable_source": "boolean",
            "confidence": "number from 0 to 1",
            "reason": "short explanation",
            "closure_scope": "district_system | parishwide | school_specific | statewide | unclear | none",
            "parish": "Louisiana parish if stated or strongly implied",
            "district_name": "district/school system if stated",
            "school_names": ["list of school names if school-specific"],
            "storm_names": ["named storms mentioned"],
            "closure_dates_mentioned": "boolean",
            "reopening_dates_mentioned": "boolean",
            "closure_start_date_guess": "YYYY-MM-DD or empty",
            "closure_end_date_guess": "YYYY-MM-DD or empty",
            "text_evidence": "short quote or paraphrase under 40 words",
            "needs_manual_review": "boolean",
        },
        "source_text": clipped,
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ECON_21316_Empirical_Study",
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=90,
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return json.loads(content)


def save_source_text(queue_id: str, url: str, title: str, text: str) -> str:
    """Save fetched source text to raw auto_sources directory."""
    RAW_AUTO_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{safe_filename(queue_id)}__{url_hash(url)}.txt"
    path = RAW_AUTO_DIR / filename

    content = []
    content.append(f"URL: {url}")
    content.append(f"TITLE: {title}")
    content.append("")
    content.append(text)

    path.write_text("\n".join(content), encoding="utf-8", errors="ignore")

    return str(path.relative_to(PROJECT_ROOT))


def main() -> None:
    """Run source search and screening."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-queue-rows",
        type=int,
        default=20,
        help="Maximum priority queue rows to process.",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=5,
        help="Search results per query.",
    )
    parser.add_argument(
        "--queries-per-row",
        type=int,
        default=3,
        help="Number of priority_search_query columns to use per queue row.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between search/fetch calls.",
    )
    parser.add_argument(
        "--skip-openrouter",
        action="store_true",
        help="Only search/fetch sources; do not call OpenRouter screening.",
    )

    args = parser.parse_args()

    load_project_env()

    brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    if not brave_key:
        raise RuntimeError("Missing BRAVE_SEARCH_API_KEY in .env")

    if not args.skip_openrouter and not openrouter_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

    if not QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing queue: {QUEUE_PATH}")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_AUTO_DIR.mkdir(parents=True, exist_ok=True)

    queue = pd.read_csv(QUEUE_PATH, dtype=str, low_memory=False).head(args.max_queue_rows).copy()

    candidate_rows = []
    screening_rows = []
    seen_urls = set()

    query_cols = [f"priority_search_query_{i}" for i in range(1, args.queries_per_row + 1)]

    for _, qrow in queue.iterrows():
        queue_id = qrow["priority_queue_id"]

        print(f"\n=== Processing {queue_id}: {qrow['event_id']} / {qrow['target_parish']} ===")

        for query_col in query_cols:
            query = str(qrow.get(query_col, "")).strip()

            if not query or query.lower() == "nan":
                continue

            print("Search:", query)

            try:
                results = brave_search(query=query, api_key=brave_key, count=args.results_per_query)
            except Exception as exc:
                print("Search failed:", exc)
                continue

            time.sleep(args.sleep)

            for rank, result in enumerate(results, start=1):
                url = result.get("url", "")
                title = result.get("title", "")
                description = result.get("description", "")

                if not url or url in seen_urls or is_blocked_url(url):
                    continue

                seen_urls.add(url)

                candidate_id = f"{queue_id}__{query_col}__R{rank}__{url_hash(url)}"

                candidate = {
                    "candidate_id": candidate_id,
                    "priority_queue_id": queue_id,
                    "event_id": qrow.get("event_id", ""),
                    "event_name": qrow.get("event_name", ""),
                    "target_parish": qrow.get("target_parish", ""),
                    "parish_display": qrow.get("parish_display", ""),
                    "representative_district": qrow.get("representative_district", ""),
                    "storm_year": qrow.get("storm_year", ""),
                    "graduation_year": qrow.get("graduation_year", ""),
                    "query_col": query_col,
                    "query": query,
                    "search_rank": rank,
                    "url": url,
                    "search_title": title,
                    "search_description": description,
                    "fetch_status": "",
                    "local_file_path": "",
                    "screening_status": "",
                }

                try:
                    page_title, text = fetch_page_text(url)
                    candidate["fetch_status"] = "ok"
                    candidate["page_title"] = page_title
                    candidate["text_characters"] = len(text)

                    if len(text) < 500:
                        candidate["fetch_status"] = "too_short"
                        candidate_rows.append(candidate)
                        continue

                    local_path = save_source_text(queue_id, url, page_title or title, text)
                    candidate["local_file_path"] = local_path

                    if not likely_relevant_text(text):
                        candidate["screening_status"] = "keyword_filter_not_relevant"
                        candidate_rows.append(candidate)
                        continue

                    if args.skip_openrouter:
                        candidate["screening_status"] = "not_screened"
                        candidate_rows.append(candidate)
                        continue

                    metadata = {
                        "candidate_id": candidate_id,
                        "url": url,
                        "title": page_title or title,
                        "event_id": qrow.get("event_id", ""),
                        "event_name": qrow.get("event_name", ""),
                        "target_parish": qrow.get("target_parish", ""),
                        "parish_display": qrow.get("parish_display", ""),
                        "storm_year": qrow.get("storm_year", ""),
                        "search_query": query,
                    }

                    screening = call_openrouter_screening(
                        source_text=text,
                        metadata=metadata,
                        api_key=openrouter_key,
                        model=model,
                    )
                    candidate["screening_status"] = "screened"

                    screening_row = {
                        "candidate_id": candidate_id,
                        "priority_queue_id": queue_id,
                        "url": url,
                        "local_file_path": local_path,
                        "event_id": qrow.get("event_id", ""),
                        "target_parish": qrow.get("target_parish", ""),
                        **screening,
                    }
                    screening_rows.append(screening_row)

                    print(
                        "  Screened:",
                        "usable=",
                        screening.get("usable_source"),
                        "conf=",
                        screening.get("confidence"),
                        "url=",
                        url,
                    )

                except Exception as exc:
                    candidate["fetch_status"] = f"failed: {type(exc).__name__}: {exc}"
                    print("  Fetch/screen failed:", candidate["fetch_status"])

                candidate_rows.append(candidate)
                time.sleep(args.sleep)

    candidates = pd.DataFrame(candidate_rows)
    screening = pd.DataFrame(screening_rows)

    candidates.to_csv(CANDIDATES_OUT, index=False)
    screening.to_csv(SCREENING_OUT, index=False)

    print("\nSaved:")
    print(" -", CANDIDATES_OUT.relative_to(PROJECT_ROOT))
    print(" -", SCREENING_OUT.relative_to(PROJECT_ROOT))
    print(" -", RAW_AUTO_DIR.relative_to(PROJECT_ROOT))

    print("\nCandidates:", len(candidates))
    print("Screened:", len(screening))

    if not screening.empty and "usable_source" in screening.columns:
        print("\nUsable source counts:")
        print(screening["usable_source"].value_counts(dropna=False).to_string())

        print("\nTop usable sources:")
        usable = screening[screening["usable_source"].astype(str).str.lower().eq("true")].copy()
        if usable.empty:
            print("None")
        else:
            print(
                usable[
                    [
                        "candidate_id",
                        "event_id",
                        "target_parish",
                        "confidence",
                        "closure_scope",
                        "storm_names",
                        "closure_start_date_guess",
                        "closure_end_date_guess",
                        "url",
                    ]
                ]
                .head(20)
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()
