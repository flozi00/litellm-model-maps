#!/usr/bin/env python3
"""
Sync model cost maps from LiteLLM and provider-specific sources.

This script:
1. Fetches the base model cost data from the LiteLLM GitHub repository
2. Scrapes additional model pricing from providers (e.g., Together AI)
3. Merges the data, with scraped data taking precedence for new models
   (existing LiteLLM entries are NOT overwritten)
4. Saves the result to model_prices_and_context_window.json
"""

import json
import logging
import re
import sys
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/"
    "litellm_internal_staging/model_prices_and_context_window.json"
)

TOGETHER_MODELS_URL = "https://www.together.ai/models"
TOGETHER_MODEL_DETAIL_BASE = "https://www.together.ai/models/"
TOGETHER_PROVIDER_NAME = "together_ai"

OUTPUT_FILE = "model_prices_and_context_window.json"

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.0  # seconds between provider detail page requests


def fetch_litellm_prices() -> dict[str, Any]:
    """Fetch the base model price data from the LiteLLM GitHub repository."""
    logger.info("Fetching LiteLLM model prices from %s", LITELLM_PRICES_URL)
    response = requests.get(LITELLM_PRICES_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    logger.info("Fetched %d LiteLLM model entries", len(data))
    return data


def _parse_price_string(price_str: str) -> float | None:
    """
    Parse a price string like '$0.20' or '0.20' into a float per token.

    Together AI prices are typically shown per 1M tokens, so we divide by 1_000_000.
    Returns None if parsing fails.
    """
    if not price_str:
        return None
    cleaned = price_str.replace("$", "").replace(",", "").strip()
    match = re.search(r"[\d.]+", cleaned)
    if match:
        try:
            per_million = float(match.group())
            return per_million / 1_000_000
        except ValueError:
            return None
    return None


def _slug_to_model_id(slug: str) -> str:
    """Convert a Together AI URL slug to a model identifier."""
    return slug.strip("/").lower()


def scrape_together_ai_model_list() -> list[str]:
    """
    Scrape the list of model slugs from the Together AI models page.

    Returns a list of URL path slugs (e.g. 'meta-llama-3-1-8b-instruct-turbo').
    """
    logger.info("Scraping Together AI model list from %s", TOGETHER_MODELS_URL)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; LiteLLM-ModelMapSync/1.0; "
            "+https://github.com/flozi00/litellm-model-maps)"
        )
    }
    try:
        response = requests.get(
            TOGETHER_MODELS_URL, headers=headers, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch Together AI model list: %s", exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    slugs: list[str] = []

    # Together AI renders model cards as anchor elements pointing to /models/<slug>
    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        match = re.match(r"^/models/([^/?#]+)$", href)
        if match:
            slug = match.group(1)
            if slug not in slugs:
                slugs.append(slug)

    logger.info("Found %d model slugs on Together AI models page", len(slugs))
    return slugs


def scrape_together_ai_model_detail(slug: str) -> dict[str, Any] | None:
    """
    Scrape pricing and metadata for a single Together AI model.

    Returns a dict compatible with the LiteLLM model price format, or None on failure.
    """
    url = f"{TOGETHER_MODEL_DETAIL_BASE}{slug}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; LiteLLM-ModelMapSync/1.0; "
            "+https://github.com/flozi00/litellm-model-maps)"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch Together AI model page %s: %s", url, exc)
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = soup.get_text(separator=" ", strip=True)

    model_data: dict[str, Any] = {
        "litellm_provider": TOGETHER_PROVIDER_NAME,
        "mode": "chat",
        "source": url,
    }

    # --- Extract context window / max tokens ---
    ctx_match = re.search(
        r"(?i)context\s*(?:window|length)[:\s]+([0-9,]+)\s*[Kk]?(?:\s*tokens)?",
        page_text,
    )
    if ctx_match:
        raw = ctx_match.group(1).replace(",", "")
        multiplier = 1000 if re.search(r"[Kk]", page_text[ctx_match.end():ctx_match.end() + 5]) else 1
        try:
            tokens = int(raw) * multiplier
            if tokens < 1000:
                tokens *= 1000  # assume K suffix was omitted
            model_data["max_tokens"] = tokens
            model_data["max_input_tokens"] = tokens
            model_data["max_output_tokens"] = tokens
        except ValueError:
            pass

    # --- Extract pricing ---
    # Together AI pages show "Input $X.XX / 1M tokens" and "Output $X.XX / 1M tokens"
    input_match = re.search(
        r"(?i)input\s*\$?\s*([\d.]+)\s*(?:/\s*1M|per\s*(?:million|1M))",
        page_text,
    )
    output_match = re.search(
        r"(?i)output\s*\$?\s*([\d.]+)\s*(?:/\s*1M|per\s*(?:million|1M))",
        page_text,
    )

    if input_match:
        try:
            model_data["input_cost_per_token"] = float(input_match.group(1)) / 1_000_000
        except ValueError:
            pass

    if output_match:
        try:
            model_data["output_cost_per_token"] = float(output_match.group(1)) / 1_000_000
        except ValueError:
            pass

    # --- Detect embedding models ---
    if re.search(r"(?i)\bembed(ding)?\b", page_text):
        model_data["mode"] = "embedding"

    # --- Detect image generation models ---
    if re.search(r"(?i)\b(image.gen|text.to.image|stable.diffusion|diffusion)\b", page_text):
        model_data["mode"] = "image_generation"

    return model_data


def _derive_together_model_key(slug: str) -> str:
    """
    Derive the LiteLLM model key for a Together AI model.

    LiteLLM uses the format `together_ai/<model_id>` for Together AI models.
    """
    return f"together_ai/{slug}"


def scrape_together_ai_models() -> dict[str, Any]:
    """
    Scrape all Together AI models and return a dict of LiteLLM-format model entries.
    """
    slugs = scrape_together_ai_model_list()
    if not slugs:
        logger.warning("No Together AI model slugs found; skipping Together AI scraping")
        return {}

    scraped: dict[str, Any] = {}
    for i, slug in enumerate(slugs, 1):
        logger.info(
            "Scraping Together AI model %d/%d: %s", i, len(slugs), slug
        )
        detail = scrape_together_ai_model_detail(slug)
        if detail:
            key = _derive_together_model_key(slug)
            scraped[key] = detail

        # Polite crawl delay
        if i < len(slugs):
            time.sleep(REQUEST_DELAY)

    logger.info("Scraped %d Together AI model entries", len(scraped))
    return scraped


def merge_model_data(
    base: dict[str, Any],
    *provider_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge provider-scraped model data into the base LiteLLM data.

    Strategy:
    - Existing entries in `base` are NOT overwritten (LiteLLM is authoritative).
    - New entries from providers are added to the merged result.
    - If a provider entry already exists in base, it is skipped.
    """
    merged = dict(base)
    for provider_dict in provider_data:
        added = 0
        updated = 0
        for key, value in provider_dict.items():
            if key not in merged:
                merged[key] = value
                added += 1
            else:
                # Update only missing fields in existing entries
                existing = merged[key]
                changed = False
                for field, field_value in value.items():
                    if field not in existing:
                        existing[field] = field_value
                        changed = True
                if changed:
                    updated += 1
        logger.info(
            "Merged provider data: %d new entries added, %d existing entries updated",
            added,
            updated,
        )
    return merged


def save_model_data(data: dict[str, Any], path: str) -> None:
    """Save the model data to a JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
    logger.info("Saved %d model entries to %s", len(data), path)


def main() -> None:
    # 1. Fetch base data from LiteLLM
    base_data = fetch_litellm_prices()

    # 2. Scrape provider-specific data
    together_data = scrape_together_ai_models()

    # 3. Merge
    merged = merge_model_data(base_data, together_data)

    # 4. Save
    save_model_data(merged, OUTPUT_FILE)
    logger.info("Sync complete.")


if __name__ == "__main__":
    main()
