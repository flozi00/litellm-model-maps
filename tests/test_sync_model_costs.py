"""
Tests for scripts/sync_model_costs.py
"""

import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import pytest

# Add the repo root to sys.path so we can import the script as a module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the module under test
import importlib.util

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "sync_model_costs.py"
)
spec = importlib.util.spec_from_file_location("sync_model_costs", SCRIPT_PATH)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class TestParsePrice:
    def test_dollar_sign_prefix(self):
        result = sync._parse_price_string("$0.20")
        assert result == pytest.approx(0.20 / 1_000_000)

    def test_plain_number(self):
        result = sync._parse_price_string("1.50")
        assert result == pytest.approx(1.50 / 1_000_000)

    def test_none_on_empty(self):
        assert sync._parse_price_string("") is None

    def test_none_on_non_numeric(self):
        assert sync._parse_price_string("free") is None


class TestSlugToModelKey:
    def test_basic_slug(self):
        key = sync._derive_together_model_key("meta-llama-3-8b")
        assert key == "together_ai/meta-llama-3-8b"

    def test_slash_in_slug(self):
        key = sync._derive_together_model_key("mistralai/Mistral-7B")
        assert key == "together_ai/mistralai/Mistral-7B"


class TestMergeModelData:
    def test_new_entries_added(self):
        base = {"model_a": {"litellm_provider": "openai", "mode": "chat"}}
        provider = {"model_b": {"litellm_provider": "together_ai", "mode": "chat"}}
        merged = sync.merge_model_data(base, provider)
        assert "model_a" in merged
        assert "model_b" in merged

    def test_existing_entries_not_overwritten(self):
        base = {
            "model_a": {
                "litellm_provider": "openai",
                "input_cost_per_token": 1e-6,
            }
        }
        provider = {
            "model_a": {
                "litellm_provider": "together_ai",
                "input_cost_per_token": 9e-6,
            }
        }
        merged = sync.merge_model_data(base, provider)
        # LiteLLM value must win
        assert merged["model_a"]["input_cost_per_token"] == 1e-6
        assert merged["model_a"]["litellm_provider"] == "openai"

    def test_missing_fields_filled_from_provider(self):
        base = {"model_a": {"litellm_provider": "openai", "mode": "chat"}}
        provider = {
            "model_a": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_tokens": 128000,
            }
        }
        merged = sync.merge_model_data(base, provider)
        # max_tokens was missing in base → should be filled
        assert merged["model_a"]["max_tokens"] == 128000

    def test_multiple_providers(self):
        base = {"model_a": {"litellm_provider": "openai"}}
        p1 = {"model_b": {"litellm_provider": "together_ai"}}
        p2 = {"model_c": {"litellm_provider": "anyscale"}}
        merged = sync.merge_model_data(base, p1, p2)
        assert len(merged) == 3


class TestFetchLiteLLMPrices:
    def test_returns_dict(self):
        sample = {
            "gpt-4": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 3e-05,
                "output_cost_per_token": 6e-05,
            }
        }
        mock_response = MagicMock()
        mock_response.json.return_value = sample
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            result = sync.fetch_litellm_prices()

        assert isinstance(result, dict)
        assert "gpt-4" in result

    def test_raises_on_http_error(self):
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                sync.fetch_litellm_prices()


class TestSaveModelData:
    def test_saves_valid_json(self):
        data = {"model_a": {"litellm_provider": "openai", "mode": "chat"}}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            sync.save_model_data(data, tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded == data
        finally:
            os.unlink(tmp_path)


class TestScrapeTogetherAIModelList:
    def test_parses_model_links(self):
        html = """
        <html><body>
          <a href="/models/llama-3-8b">Llama 3 8B</a>
          <a href="/models/mistral-7b">Mistral 7B</a>
          <a href="/about">About</a>
          <a href="/models/llama-3-8b">Llama 3 8B (duplicate)</a>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            slugs = sync.scrape_together_ai_model_list()

        assert "llama-3-8b" in slugs
        assert "mistral-7b" in slugs
        # /about should not be included
        assert "about" not in slugs
        # Duplicates should be removed
        assert slugs.count("llama-3-8b") == 1

    def test_returns_empty_list_on_error(self):
        import requests

        with patch("requests.get", side_effect=requests.RequestException("timeout")):
            slugs = sync.scrape_together_ai_model_list()

        assert slugs == []


class TestScrapeTogetherAIModelDetail:
    def test_parses_pricing(self):
        html = """
        <html><body>
          <p>Context Window: 131072 tokens</p>
          <p>Input $0.18 / 1M tokens</p>
          <p>Output $0.18 / 1M tokens</p>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            detail = sync.scrape_together_ai_model_detail("llama-3-8b")

        assert detail is not None
        assert detail["litellm_provider"] == "together_ai"
        assert detail["input_cost_per_token"] == pytest.approx(0.18 / 1_000_000)
        assert detail["output_cost_per_token"] == pytest.approx(0.18 / 1_000_000)

    def test_returns_none_on_error(self):
        import requests

        with patch("requests.get", side_effect=requests.RequestException("timeout")):
            detail = sync.scrape_together_ai_model_detail("llama-3-8b")

        assert detail is None

    def test_detects_embedding_mode(self):
        html = "<html><body><p>Text embedding model</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            detail = sync.scrape_together_ai_model_detail("bge-large")

        assert detail is not None
        assert detail["mode"] == "embedding"

    def test_detects_image_generation_mode(self):
        html = "<html><body><p>Stable Diffusion text-to-image model</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            detail = sync.scrape_together_ai_model_detail("stable-diffusion-xl")

        assert detail is not None
        assert detail["mode"] == "image_generation"
