#!/usr/bin/env python3
"""
Sync model cost maps from LiteLLM and provider-specific sources.

This script:
1. Fetches the base model cost data from the LiteLLM GitHub repository
2. Scrapes additional model pricing from providers (Together AI, DeepInfra, Fireworks AI)
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

# DeepInfra REST API (OpenAI-compatible — returns model list; pricing may not be included)
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/models"

# DeepInfra website (fallback for pricing via HTML scraping)
DEEPINFRA_MODELS_URL = "https://deepinfra.com/models"
DEEPINFRA_MODEL_DETAIL_BASE = "https://deepinfra.com/"
DEEPINFRA_PROVIDER_NAME = "deepinfra"

# Fireworks AI supplemental model data
FIREWORKS_PROVIDER_NAME = "fireworks_ai"
FIREWORKS_MODELS_URL = "https://fireworks.ai/models"
FIREWORKS_MODEL_DETAIL_BASE = "https://fireworks.ai/models/fireworks/"
FIREWORKS_APP_MODEL_DETAIL_BASE = "https://app.fireworks.ai/models/fireworks/"
FIREWORKS_SUPPLEMENTAL_MODELS: dict[str, dict[str, Any]] = {
    "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro": {
        "litellm_provider": FIREWORKS_PROVIDER_NAME,
        "mode": "chat",
        "max_tokens": 1_000_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 1_000_000,
        "input_cost_per_token": 1.74 / 1_000_000,
        "cache_read_input_token_cost": 0.15 / 1_000_000,
        "output_cost_per_token": 3.48 / 1_000_000,
        "source": f"{FIREWORKS_APP_MODEL_DETAIL_BASE}deepseek-v4-pro",
    },
    "fireworks_ai/accounts/fireworks/models/kimi-k2p6": {
        "litellm_provider": FIREWORKS_PROVIDER_NAME,
        "mode": "chat",
        "max_tokens": 262_144,
        "max_input_tokens": 262_144,
        "max_output_tokens": 262_144,
        "input_cost_per_token": 0.95 / 1_000_000,
        "cache_read_input_token_cost": 0.16 / 1_000_000,
        "output_cost_per_token": 4.00 / 1_000_000,
        "source": "https://fireworks.ai/models/fireworks/kimi-k2p6",
        "supports_function_calling": True,
    },
    "fireworks_ai/kimi-k2p6": {
        "litellm_provider": FIREWORKS_PROVIDER_NAME,
        "mode": "chat",
        "max_tokens": 262_144,
        "max_input_tokens": 262_144,
        "max_output_tokens": 262_144,
        "input_cost_per_token": 0.95 / 1_000_000,
        "cache_read_input_token_cost": 0.16 / 1_000_000,
        "output_cost_per_token": 4.00 / 1_000_000,
        "source": "https://fireworks.ai/models/fireworks/kimi-k2p6",
        "supports_function_calling": True,
    },
}

# Path segments that are definitely not model org names on deepinfra.com
_DEEPINFRA_NON_ORG_PATHS = frozenset(
    {
        "docs",
        "blog",
        "pricing",
        "about",
        "company",
        "careers",
        "contact",
        "api",
        "help",
        "faq",
        "login",
        "signup",
        "terms",
        "privacy",
        "models",
        "inference",
    }
)

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
    Recursively search a Next.js data tree for Together AI model slugs or model IDs.

    Prefers the actual API model ID (``id`` field in ``org/model`` format such as
    ``MiniMaxAI/MiniMax-M2.7``) over a URL slug when both are present in the same
    object — this avoids the slug→key mismatch bug where the URL slug
    (e.g. ``minimax-m2-7``) does not match the API model ID.

    Falls back to ``slug``, ``model_slug``, or ``modelSlug`` keys when no ``id`` field
    with an ``org/model``-format value is found.

    Returns a deduplicated list preserving first-seen order.
    """
    _ORG_MODEL_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.:+-]+$")
    seen: set[str] = set()
    found: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            # Prefer the actual API model id (e.g. "MiniMaxAI/MiniMax-M2.7") over
            # the URL slug.  The `id` key is used by the Together AI REST API and
            # sometimes also appears in __NEXT_DATA__.
            id_val = node.get("id")
            if isinstance(id_val, str) and _ORG_MODEL_RE.match(id_val):
                if id_val not in seen:
                    seen.add(id_val)
                    found.append(id_val)
            else:
                # Fall back to slug keys
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

    If the page contains an "Endpoint" field (e.g. ``MiniMaxAI/MiniMax-M2.7``), the
    dict will also contain a private ``_model_id`` key with the canonical API model ID.
    The caller (``scrape_together_ai_models``) pops this key and uses it to build the
    correct LiteLLM key instead of the URL slug.
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

    # --- Extract the canonical API model ID from the "Endpoint" field ---------------
    # Together AI detail pages show an "Endpoint" row like "MiniMaxAI/MiniMax-M2.7".
    # We match a case-insensitive word boundary around "endpoint" followed by a value
    # in org/model format (e.g. "MiniMaxAI/MiniMax-M2.7").
    endpoint_match = re.search(
        r"(?i)\bEndpoint\b[:\s]+([A-Za-z0-9_.-]+/[A-Za-z0-9_.:+-]+)",
        page_text,
    )
    if endpoint_match:
        model_data["_model_id"] = endpoint_match.group(1)

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
    if re.search(r"(?i)\b(image[ _-]gen|text[ _-]to[ _-]image|stable[ _-]diffusion|diffusion)\b", page_text):
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
            # Use the canonical API model ID extracted from the detail page if
            # available (e.g. "MiniMaxAI/MiniMax-M2.7" from the "Endpoint" field).
            # Fall back to the URL slug only when no endpoint ID was found.
            model_id = detail.pop("_model_id", slug)
            key = _derive_together_model_key(model_id)
            scraped[key] = detail

        # Polite crawl delay
        if i < len(slugs):
            time.sleep(REQUEST_DELAY)

    logger.info("Scraped %d Together AI model entries via HTML", len(scraped))
    return scraped


# ---------------------------------------------------------------------------
# DeepInfra provider
# ---------------------------------------------------------------------------


def fetch_deepinfra_models_via_api(api_key: str | None = None) -> list[dict]:
    """
    Fetch the model list from the DeepInfra OpenAI-compatible REST API.

    DeepInfra exposes an OpenAI-compatible ``GET /v1/openai/models`` endpoint
    that returns model metadata.  The standard OpenAI format does not carry
    pricing; if DeepInfra extends the response with ``pricing`` / ``context_length``
    fields we parse them, otherwise the caller falls back to HTML scraping.

    Set the ``DEEPINFRA_API_KEY`` environment variable to authenticate requests.
    """
    logger.info("Fetching DeepInfra models from REST API: %s", DEEPINFRA_API_URL)
    headers: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; LiteLLM-ModelMapSync/1.0; "
            "+https://github.com/flozi00/litellm-model-maps)"
        ),
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(DEEPINFRA_API_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    # OpenAI-compatible format: {"object": "list", "data": [...]}
    if isinstance(data, dict):
        return data.get("data", [])
    if isinstance(data, list):
        return data
    return []


def _deepinfra_api_model_to_litellm_entry(model: dict) -> tuple[str, dict] | None:
    """
    Convert a DeepInfra API model object to a (litellm_key, entry) pair.

    Returns None if the model has no usable ID.
    """
    model_id = (model.get("id") or "").strip()
    if not model_id:
        return None

    key = f"deepinfra/{model_id}"

    # Infer mode from the model id / owned_by heuristics
    model_id_lower = model_id.lower()
    if "embed" in model_id_lower:
        mode = "embedding"
    elif any(w in model_id_lower for w in ("flux", "stable-diffusion", "wan-")):
        mode = "image_generation"
    else:
        mode = "chat"

    entry: dict[str, Any] = {
        "litellm_provider": DEEPINFRA_PROVIDER_NAME,
        "mode": mode,
        "source": DEEPINFRA_API_URL,
    }

    # Pricing — some DeepInfra API responses include a ``pricing`` object with
    # per-million-token rates (same convention as Together AI).
    pricing = model.get("pricing") or {}
    _PRICE_IN_KEYS = ("input", "prompt", "input_cost")
    _PRICE_OUT_KEYS = ("output", "completion", "output_cost")
    in_price = next((pricing[k] for k in _PRICE_IN_KEYS if k in pricing), None)
    out_price = next((pricing[k] for k in _PRICE_OUT_KEYS if k in pricing), None)
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

    # Context window
    ctx = model.get("context_length") or model.get("context_window")
    if ctx is not None:
        try:
            tokens = int(ctx)
            if tokens > 0:
                entry["max_tokens"] = tokens
                entry["max_input_tokens"] = tokens
                entry["max_output_tokens"] = tokens
        except (ValueError, TypeError):
            pass

    return key, entry


def _find_deepinfra_entries_in_next_data(data: Any) -> dict[str, dict]:
    """
    Recursively search a DeepInfra Next.js data tree for model entries.

    Looks for dicts that contain a ``modelId`` or ``model_id`` field matching
    the ``org/model`` pattern, optionally accompanied by pricing / context_length
    fields.  Returns a ``{litellm_key: entry}`` dict (may be empty).
    """
    _ORG_MODEL_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

    results: dict[str, dict] = {}
    seen_ids: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            # Look for an explicit model-id field
            model_id: str | None = None
            for id_key in ("modelId", "model_id"):
                val = node.get(id_key)
                if isinstance(val, str) and _ORG_MODEL_RE.match(val):
                    model_id = val
                    break

            if model_id and model_id not in seen_ids:
                seen_ids.add(model_id)
                entry: dict[str, Any] = {
                    "litellm_provider": DEEPINFRA_PROVIDER_NAME,
                    "mode": "chat",
                    "source": DEEPINFRA_MODELS_URL,
                }

                # Pricing — use next() so a legitimate price of 0 is not skipped
                pricing = node.get("pricing") or {}
                in_p = next(
                    (pricing[k] for k in ("input_cost", "input", "prompt_cost") if k in pricing),
                    None,
                )
                out_p = next(
                    (pricing[k] for k in ("output_cost", "output", "completion_cost") if k in pricing),
                    None,
                )
                if in_p is not None:
                    try:
                        entry["input_cost_per_token"] = float(in_p) / 1_000_000
                    except (ValueError, TypeError):
                        pass
                if out_p is not None:
                    try:
                        entry["output_cost_per_token"] = float(out_p) / 1_000_000
                    except (ValueError, TypeError):
                        pass

                # Context window
                ctx = node.get("context_length") or node.get("contextLength")
                if ctx is not None:
                    try:
                        tokens = int(ctx)
                        if tokens > 0:
                            entry["max_tokens"] = tokens
                            entry["max_input_tokens"] = tokens
                            entry["max_output_tokens"] = tokens
                    except (ValueError, TypeError):
                        pass

                # Mode
                task = (node.get("type") or node.get("task") or "").lower()
                if "embed" in task:
                    entry["mode"] = "embedding"
                elif "image" in task or "diffusion" in task:
                    entry["mode"] = "image_generation"

                results[f"deepinfra/{model_id}"] = entry

            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return results


def scrape_deepinfra_model_list() -> list[str]:
    """
    Scrape the list of model IDs from the DeepInfra models page.

    Attempts two strategies in order:
    1. Parse ``__NEXT_DATA__`` JSON for explicit ``modelId`` / ``model_id`` fields.
    2. Find anchor elements with ``href="/<org>/<model>"`` in the raw HTML.

    Returns a list of model IDs in ``org/model`` format.
    """
    logger.info("Scraping DeepInfra model list from %s", DEEPINFRA_MODELS_URL)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; LiteLLM-ModelMapSync/1.0; "
            "+https://github.com/flozi00/litellm-model-maps)"
        )
    }
    try:
        response = requests.get(
            DEEPINFRA_MODELS_URL, headers=headers, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch DeepInfra model list: %s", exc)
        return []

    html = response.text

    # --- Strategy 1: Parse Next.js embedded page data (__NEXT_DATA__) ---
    next_data = _extract_next_data(html)
    if next_data:
        entries = _find_deepinfra_entries_in_next_data(next_data)
        if entries:
            logger.info(
                "Found %d DeepInfra model entries via __NEXT_DATA__", len(entries)
            )
            # Return model IDs so scrape_deepinfra_models can skip detail scraping
            # when we already have full entries — the caller handles this distinction.
            return list(entries.keys())  # keys are "deepinfra/<org>/<model>" already

    # --- Strategy 2: Parse anchor tags ---
    soup = BeautifulSoup(html, "html.parser")
    slugs: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        # Match /<org>/<model> — exactly two non-empty path segments
        match = re.match(r"^/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$", href)
        if match:
            org, model_name = match.group(1), match.group(2)
            # Filter out well-known non-model path prefixes
            if org.lower() not in _DEEPINFRA_NON_ORG_PATHS:
                slug = f"{org}/{model_name}"
                if slug not in seen:
                    seen.add(slug)
                    slugs.append(slug)

    logger.info("Found %d DeepInfra model slugs via HTML anchors", len(slugs))
    return slugs


def scrape_deepinfra_model_detail(model_id: str) -> dict[str, Any] | None:
    """
    Scrape pricing and metadata from a DeepInfra model detail page.

    ``model_id`` should be in ``org/model-name`` format (e.g.
    ``deepseek-ai/DeepSeek-V4-Pro``).

    Returns a dict compatible with the LiteLLM model price format, or None on
    failure.
    """
    url = f"{DEEPINFRA_MODEL_DETAIL_BASE}{model_id}"
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
        logger.warning("Failed to fetch DeepInfra model page %s: %s", url, exc)
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = soup.get_text(separator=" ", strip=True)

    model_data: dict[str, Any] = {
        "litellm_provider": DEEPINFRA_PROVIDER_NAME,
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
        suffix_window = page_text[ctx_match.end() : ctx_match.end() + 5]
        multiplier = 1000 if re.search(r"[Kk]", suffix_window) else 1
        try:
            tokens = int(raw) * multiplier
            if tokens < 1000:
                tokens *= 1000  # assume K suffix was omitted
            model_data["max_tokens"] = tokens
            model_data["max_input_tokens"] = tokens
            model_data["max_output_tokens"] = tokens
        except ValueError:
            pass

    # --- Extract pricing (shown as "Input $X.XX / 1M tokens") ---
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
    if re.search(r"(?i)\b(image[ _-]gen|text[ _-]to[ _-]image|stable[ _-]diffusion|diffusion|flux)\b", page_text):
        model_data["mode"] = "image_generation"

    return model_data


def scrape_deepinfra_models() -> dict[str, Any]:
    """
    Fetch all DeepInfra models and return a dict of LiteLLM-format model entries.

    Tries three strategies in order of reliability:
    1. DeepInfra REST API (``/v1/openai/models``) — returns model IDs; pricing is
       included only if DeepInfra extends the standard OpenAI response.  Uses the
       ``DEEPINFRA_API_KEY`` environment variable for authentication when set.
    2. HTML scraping with Next.js ``__NEXT_DATA__`` extraction — works when
       DeepInfra embeds the model catalogue (including pricing) server-side.
    3. HTML anchor-tag scraping with per-model detail page visits — last resort.
    """
    api_key = os.environ.get("DEEPINFRA_API_KEY")

    # --- Strategy 1: REST API ---
    try:
        api_models = fetch_deepinfra_models_via_api(api_key=api_key)
    except requests.RequestException as exc:
        logger.warning(
            "DeepInfra API request failed (%s); falling back to HTML scraping", exc
        )
        api_models = []

    if api_models:
        scraped: dict[str, Any] = {}
        for model in api_models:
            result = _deepinfra_api_model_to_litellm_entry(model)
            if result is not None:
                key, entry = result
                scraped[key] = entry
        logger.info("Fetched %d DeepInfra model entries from REST API", len(scraped))
        return scraped

    # --- Strategies 2 & 3: HTML scraping ---
    raw_slugs = scrape_deepinfra_model_list()
    if not raw_slugs:
        logger.warning("No DeepInfra model slugs found; skipping DeepInfra scraping")
        return {}

    # scrape_deepinfra_model_list() returns "deepinfra/<org>/<model>" keys when it
    # found full entries via __NEXT_DATA__, or plain "<org>/<model>" slugs otherwise.
    if raw_slugs and raw_slugs[0].startswith("deepinfra/"):
        # __NEXT_DATA__ path already yielded complete entries; rebuild the dict by
        # re-running the extraction (list→set of keys already processed above).
        next_data = _extract_next_data(
            requests.get(
                DEEPINFRA_MODELS_URL,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; LiteLLM-ModelMapSync/1.0; "
                        "+https://github.com/flozi00/litellm-model-maps)"
                    )
                },
                timeout=REQUEST_TIMEOUT,
            ).text
        )
        if next_data:
            entries = _find_deepinfra_entries_in_next_data(next_data)
            if entries:
                logger.info(
                    "Using %d DeepInfra entries from __NEXT_DATA__", len(entries)
                )
                return entries

    # Plain slug list — visit each model detail page
    scraped = {}
    for i, slug in enumerate(raw_slugs, 1):
        logger.info("Scraping DeepInfra model %d/%d: %s", i, len(raw_slugs), slug)
        detail = scrape_deepinfra_model_detail(slug)
        if detail:
            scraped[f"deepinfra/{slug}"] = detail

        if i < len(raw_slugs):
            time.sleep(REQUEST_DELAY)

    logger.info("Scraped %d DeepInfra model entries via HTML", len(scraped))
    return scraped


def _parse_token_count(number: str, unit: str = "") -> int | None:
    """Parse displayed token counts such as ``262.1k`` or ``1M``."""
    try:
        value = float(number.replace(",", ""))
    except ValueError:
        return None

    unit = unit.lower()
    if unit == "m":
        return int(round(value * 1_000_000))
    if unit == "k":
        tokens = value * 1_000
        if "." in number:
            return int(round(tokens / 1024) * 1024)
        return int(round(tokens))
    return int(round(value))


def _find_fireworks_model_slugs_in_next_data(data: Any) -> list[str]:
    """Recursively search Next.js data for Fireworks model detail page slugs."""
    seen: set[str] = set()
    found: list[str] = []

    def _add(slug: str) -> None:
        slug = slug.strip("/")
        if slug and slug not in seen:
            seen.add(slug)
            found.append(slug)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            for match in re.finditer(r"/models/fireworks/([A-Za-z0-9_.-]+)", node):
                _add(match.group(1))
            for match in re.finditer(
                r"\b(?:accounts/)?fireworks(?:/models)?/([A-Za-z0-9_.-]+)\b",
                node,
            ):
                _add(match.group(1))

    _walk(data)
    return found


def scrape_fireworks_model_list() -> list[str]:
    """Scrape Fireworks model slugs from the public model library."""
    logger.info("Scraping Fireworks AI model list from %s", FIREWORKS_MODELS_URL)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; LiteLLM-ModelMapSync/1.0; "
            "+https://github.com/flozi00/litellm-model-maps)"
        )
    }
    try:
        response = requests.get(
            FIREWORKS_MODELS_URL, headers=headers, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch Fireworks AI model list: %s", exc)
        return []

    html = response.text
    next_data = _extract_next_data(html)
    if next_data:
        next_slugs = _find_fireworks_model_slugs_in_next_data(next_data)
        if next_slugs:
            logger.info(
                "Found %d Fireworks AI model slugs via __NEXT_DATA__",
                len(next_slugs),
            )
            return next_slugs

    soup = BeautifulSoup(html, "html.parser")
    slugs: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        match = re.match(r"^/models/fireworks/([^/?#]+)$", href)
        if match:
            slug = match.group(1)
            if slug not in slugs:
                slugs.append(slug)

    logger.info("Found %d Fireworks AI model slugs on model list page", len(slugs))
    return slugs


def scrape_fireworks_model_detail(slug: str) -> dict[str, Any] | None:
    """Scrape pricing and metadata for a single Fireworks model detail page."""
    url = f"{FIREWORKS_MODEL_DETAIL_BASE}{slug}"
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
        logger.warning("Failed to fetch Fireworks AI model page %s: %s", url, exc)
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))

    entry: dict[str, Any] = {
        "litellm_provider": FIREWORKS_PROVIDER_NAME,
        "mode": "chat",
        "source": url,
    }

    context_match = re.search(
        r"(?i)context\s+length\s+([0-9,.]+)\s*([kKmM]?)\s*tokens?",
        page_text,
    )
    if context_match:
        tokens = _parse_token_count(context_match.group(1), context_match.group(2))
        if tokens and tokens > 0:
            entry["max_tokens"] = tokens
            entry["max_input_tokens"] = tokens
            entry["max_output_tokens"] = tokens

    price_triplet = re.search(
        r"\$\s*([0-9.]+)\s*/\s*\$\s*([0-9.]+)\s*/\s*\$\s*([0-9.]+)"
        r".{0,120}?(?:input\s*/\s*cached\s+input\s*/\s*output)",
        page_text,
        re.IGNORECASE,
    )
    if price_triplet:
        entry["input_cost_per_token"] = float(price_triplet.group(1)) / 1_000_000
        entry["cache_read_input_token_cost"] = (
            float(price_triplet.group(2)) / 1_000_000
        )
        entry["output_cost_per_token"] = float(price_triplet.group(3)) / 1_000_000
    else:
        input_match = re.search(r"(?i)\binput\b\s*\$?\s*([0-9.]+)", page_text)
        cache_match = re.search(
            r"(?i)\bcached\s+input\b\s*\$?\s*([0-9.]+)",
            page_text,
        )
        output_match = re.search(r"(?i)\boutput\b\s*\$?\s*([0-9.]+)", page_text)
        if input_match:
            entry["input_cost_per_token"] = float(input_match.group(1)) / 1_000_000
        if cache_match:
            entry["cache_read_input_token_cost"] = (
                float(cache_match.group(1)) / 1_000_000
            )
        if output_match:
            entry["output_cost_per_token"] = float(output_match.group(1)) / 1_000_000

    if re.search(r"(?i)function\s+calling\s+supported", page_text):
        entry["supports_function_calling"] = True

    return entry


def _fireworks_entries_for_slug(slug: str, detail: dict[str, Any]) -> dict[str, Any]:
    """Build the LiteLLM Fireworks keys for a scraped Fireworks model slug."""
    return {
        f"fireworks_ai/accounts/fireworks/models/{slug}": dict(detail),
        f"fireworks_ai/{slug}": dict(detail),
    }


def scrape_fireworks_models() -> dict[str, Any]:
    """Fetch Fireworks AI models and return LiteLLM-format model entries."""
    scraped: dict[str, Any] = {}
    slugs = scrape_fireworks_model_list()
    for i, slug in enumerate(slugs, 1):
        logger.info("Scraping Fireworks AI model %d/%d: %s", i, len(slugs), slug)
        try:
            detail = scrape_fireworks_model_detail(slug)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to parse Fireworks AI model %s page; skipping it: %s",
                slug,
                exc,
            )
            detail = None
        if detail:
            scraped.update(_fireworks_entries_for_slug(slug, detail))

        if i < len(slugs):
            time.sleep(REQUEST_DELAY)

    logger.info("Scraped %d Fireworks AI model entries via HTML", len(scraped))
    logger.info(
        "Loaded %d Fireworks AI supplemental model entries",
        len(FIREWORKS_SUPPLEMENTAL_MODELS),
    )
    for key, value in FIREWORKS_SUPPLEMENTAL_MODELS.items():
        if key not in scraped:
            scraped[key] = dict(value)
        else:
            for field, field_value in value.items():
                scraped[key].setdefault(field, field_value)
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
    sorted_data = {
        key: data[key]
        for key in (["sample_spec"] if "sample_spec" in data else [])
        + sorted(key for key in data if key != "sample_spec")
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sorted_data, fh, indent=4, ensure_ascii=False)
    logger.info("Saved %d model entries to %s", len(data), path)


def main() -> None:
    # 1. Fetch base data from LiteLLM
    base_data = fetch_litellm_prices()

    # 2. Scrape provider-specific data
    together_data = scrape_together_ai_models()
    deepinfra_data = scrape_deepinfra_models()
    fireworks_data = scrape_fireworks_models()

    # 3. Merge (LiteLLM is authoritative; providers fill in missing entries/fields)
    merged = merge_model_data(base_data, together_data, deepinfra_data, fireworks_data)

    # 4. Save
    save_model_data(merged, OUTPUT_FILE)
    logger.info("Sync complete.")


if __name__ == "__main__":
    main()
