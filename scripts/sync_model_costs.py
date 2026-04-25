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
import os
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

# Together AI REST API (primary source — returns all models with pricing)
TOGETHER_API_URL = "https://api.together.xyz/v1/models"

# Together AI website (used as fallback when API is unavailable)
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


def fetch_together_ai_models_via_api(api_key: str | None = None) -> list[dict]:
    """
    Fetch the full model list from the Together AI REST API.

    Together AI API response format (list of objects):
    {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "type": "chat",          # language | chat | code | embedding | image | diffusion
        "context_length": 8192,
        "pricing": {
            "input": 0.18,       # per million tokens
            "output": 0.18,      # per million tokens
            ...
        }
    }

    Set TOGETHER_API_KEY environment variable to authenticate requests.
    Without a key the API may return 401; the caller handles that gracefully.
    """
    logger.info("Fetching Together AI models from REST API: %s", TOGETHER_API_URL)
    headers: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; LiteLLM-ModelMapSync/1.0; "
            "+https://github.com/flozi00/litellm-model-maps)"
        ),
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(TOGETHER_API_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    # API may return a plain list or a paginated {"data": [...]} object
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data", [])
    return []


def _api_model_to_litellm_entry(model: dict) -> tuple[str, dict] | None:
    """
    Convert a Together AI API model object to a (litellm_key, entry) pair.

    Returns None if the model has no usable ID.
    """
    model_id = (model.get("id") or "").strip()
    if not model_id:
        return None

    key = f"together_ai/{model_id}"

    # Map Together AI model type to LiteLLM mode
    model_type = (model.get("type") or "").lower()
    if "embed" in model_type:
        mode = "embedding"
    elif "image" in model_type or "diffusion" in model_type:
        mode = "image_generation"
    else:
        # language, chat, code → all are chat-compatible
        mode = "chat"

    entry: dict[str, Any] = {
        "litellm_provider": TOGETHER_PROVIDER_NAME,
        "mode": mode,
        "source": TOGETHER_API_URL,
    }

    # Context window
    ctx = model.get("context_length")
    if ctx is not None:
        try:
            tokens = int(ctx)
            if tokens > 0:
                entry["max_tokens"] = tokens
                entry["max_input_tokens"] = tokens
                entry["max_output_tokens"] = tokens
        except (ValueError, TypeError):
            pass

    # Pricing — API gives per-million-token rates; LiteLLM wants per-token
    pricing = model.get("pricing") or {}
    in_price = pricing.get("input")
    out_price = pricing.get("output")
    if in_price is not None:
        try:
            val = float(in_price)
            if val >= 0:
                entry["input_cost_per_token"] = val / 1_000_000
        except (ValueError, TypeError):
            pass
    if out_price is not None:
        try:
            val = float(out_price)
            if val >= 0:
                entry["output_cost_per_token"] = val / 1_000_000
        except (ValueError, TypeError):
            pass

    return key, entry


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


def _extract_next_data(html: str) -> dict | None:
    """
    Extract the embedded Next.js page data from a page's HTML.

    Next.js server-side-rendered pages embed their initial props inside a
    ``<script id="__NEXT_DATA__" type="application/json">`` tag.  Parsing
    this avoids the need for a headless browser to execute JavaScript.

    Returns the parsed JSON dict, or None if the tag is absent or unparseable.
    """
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag and tag.string:
        try:
            return json.loads(tag.string)
        except json.JSONDecodeError:
            pass
    return None


def _find_slugs_in_next_data(data: Any) -> list[str]:
    """
    Recursively search a Next.js data tree for Together AI model slugs.

    Looks for objects that contain a "slug", "model_slug", or "modelSlug" key.
    Returns a deduplicated list preserving first-seen order.
    """
    seen: set[str] = set()
    found: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("slug", "model_slug", "modelSlug"):
                val = node.get(key)
                if isinstance(val, str) and val and val not in seen:
                    seen.add(val)
                    found.append(val)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return found


def scrape_together_ai_model_list() -> list[str]:
    """
    Scrape the list of model slugs from the Together AI models page.

    Attempts three strategies in order:
    1. Parse ``__NEXT_DATA__`` embedded JSON for explicit slug fields.
    2. Find anchor elements with ``href="/models/<slug>"`` in the raw HTML.

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

    html = response.text

    # --- Strategy 1: Parse Next.js embedded page data (__NEXT_DATA__) ---
    # Next.js SSR/SSG pages embed their initial props as JSON in the HTML.
    # Together AI's website is a Next.js app; if the model list is server-rendered
    # the slugs will appear in the __NEXT_DATA__ script tag.
    next_data = _extract_next_data(html)
    if next_data:
        next_slugs = _find_slugs_in_next_data(next_data)
        if next_slugs:
            logger.info(
                "Found %d model slugs via __NEXT_DATA__", len(next_slugs)
            )
            return next_slugs

    # --- Strategy 2: Parse anchor tags (works for server-rendered HTML) ---
    soup = BeautifulSoup(html, "html.parser")
    slugs: list[str] = []

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
    Fetch all Together AI models and return a dict of LiteLLM-format model entries.

    Tries three strategies in order of reliability:
    1. Together AI REST API (``/v1/models``) — returns structured data with pricing.
       Uses the ``TOGETHER_API_KEY`` environment variable for authentication when set.
    2. HTML scraping with Next.js ``__NEXT_DATA__`` extraction — works when Together AI
       uses server-side rendering for its models page.
    3. HTML anchor-tag scraping with per-model detail page visits — last resort; only
       works if the page renders server-side HTML links.
    """
    api_key = os.environ.get("TOGETHER_API_KEY")

    # --- Strategy 1: REST API ---
    try:
        api_models = fetch_together_ai_models_via_api(api_key=api_key)
    except requests.RequestException as exc:
        logger.warning(
            "Together AI API request failed (%s); falling back to HTML scraping", exc
        )
        api_models = []

    if api_models:
        scraped: dict[str, Any] = {}
        for model in api_models:
            result = _api_model_to_litellm_entry(model)
            if result is not None:
                key, entry = result
                scraped[key] = entry
        logger.info("Fetched %d Together AI model entries from REST API", len(scraped))
        return scraped

    # --- Strategies 2 & 3: HTML scraping (with __NEXT_DATA__ + anchor fallback) ---
    slugs = scrape_together_ai_model_list()
    if not slugs:
        logger.warning("No Together AI model slugs found; skipping Together AI scraping")
        return {}

    scraped = {}
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

    logger.info("Scraped %d Together AI model entries via HTML", len(scraped))
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
