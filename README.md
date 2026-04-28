# litellm-model-maps

Automatically synced model cost maps for [LiteLLM](https://github.com/BerriAI/litellm), enhanced with additional provider pricing scraped from their official websites.

## Overview

This repository maintains an up-to-date `model_prices_and_context_window.json` file that:

1. **Fetches** the base model cost data from the [LiteLLM upstream file](https://github.com/BerriAI/litellm/blob/litellm_internal_staging/model_prices_and_context_window.json).
2. **Scrapes** additional model listings and pricing from provider websites (e.g. [Together AI](https://www.together.ai/models)).
3. **Merges** the data — existing LiteLLM entries are authoritative and are never overwritten; new models found via scraping are added.

## Automated Sync

A GitHub Actions workflow (`.github/workflows/sync.yml`) runs **daily at 02:00 UTC** and on every manual trigger. It:

- Runs `scripts/sync_model_costs.py`
- Commits and pushes any changes to `model_prices_and_context_window.json`

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the sync script (outputs model_prices_and_context_window.json in the repo root)
python scripts/sync_model_costs.py
```

## Supported Providers

| Provider | Source |
|---|---|
| **LiteLLM (base)** | [model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/litellm_internal_staging/model_prices_and_context_window.json) |
| **Together AI** | [together.ai/models](https://www.together.ai/models) |
| **Fireworks AI** | [fireworks.ai/pricing](https://fireworks.ai/pricing#serverless-pricing) |

## Output Format

The output JSON follows the same schema as the LiteLLM upstream file. Each model entry looks like:

```json
{
    "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
        "litellm_provider": "together_ai",
        "mode": "chat",
        "max_tokens": 131072,
        "max_input_tokens": 131072,
        "max_output_tokens": 131072,
        "input_cost_per_token": 1.8e-07,
        "output_cost_per_token": 1.8e-07,
        "source": "https://www.together.ai/models/meta-llama-3-1-8b-instruct-turbo"
    }
}
```
