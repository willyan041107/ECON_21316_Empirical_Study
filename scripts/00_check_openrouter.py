"""
Check whether the OpenRouter API key is correctly loaded.

Before running:
1. Open the .env file.
2. Replace PASTE_YOUR_KEY_HERE with your actual OpenRouter API key.
3. Run this script from the project root:

    python scripts/00_check_openrouter.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    """Send a minimal test request to OpenRouter."""
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        raise ValueError("OPENROUTER_API_KEY is missing. Please paste it into the .env file.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: API connection works.",
            }
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
