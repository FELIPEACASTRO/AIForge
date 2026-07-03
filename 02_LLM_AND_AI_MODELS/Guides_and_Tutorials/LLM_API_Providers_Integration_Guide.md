# LLM API Providers — Integration Guide

> How to integrate the major LLM APIs (endpoints, auth, models, official docs) — verified live in June 2026. Most providers are **OpenAI-compatible**, so one client pattern covers almost all of them. 🔒 **Never hardcode or commit API keys** — use environment variables / secret managers.

## The universal pattern (OpenAI-compatible)

Most providers accept the exact same request shape — only `base_url`, `model`, and the key change:

```python
import os, requests
def chat(base_url, model, key, prompt, max_tokens=800):
    r = requests.post(f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# example: chat("https://openrouter.ai/api/v1", "qwen/qwen-2.5-72b-instruct", os.environ["OPENROUTER_API_KEY"], "...")
```

Or with the official `openai` SDK: `OpenAI(base_url=..., api_key=...)` works for every OpenAI-compatible row below.

## Provider matrix (verified June 2026)

| Provider | Base URL | Auth | Example models | OpenAI-compat? | Official docs |
|---|---|---|---|---|---|
| **Anthropic (Claude)** | `https://api.anthropic.com/v1/messages` | `x-api-key` + `anthropic-version: 2023-06-01` headers | `claude-opus-4-8`, `claude-sonnet-5`, `claude-haiku-4-5` | ❌ (own schema: `messages` API; SDK `anthropic`) | https://docs.anthropic.com/ |
| **OpenAI** | `https://api.openai.com/v1` | `Authorization: Bearer sk-...` | `gpt-4o`, `gpt-4o-mini`, `o3` | ✅ (it *is* the standard; newer `/v1/responses` API too) | https://platform.openai.com/docs |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` | `?key=AIza...` query param or `x-goog-api-key` header | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` | ➖ (own schema `contents/parts`; an OpenAI-compat endpoint exists at `/v1beta/openai/`) | https://ai.google.dev/gemini-api/docs |
| **OpenRouter** (gateway to 300+ models) | `https://openrouter.ai/api/v1` | `Authorization: Bearer sk-or-v1-...` | any slug: `qwen/qwen-2.5-72b-instruct`, `deepseek/deepseek-chat`, `google/gemma-3-27b-it`, `mistralai/mistral-small-3.2-24b-instruct` | ✅ | https://openrouter.ai/docs |
| **Hugging Face Inference Providers** | `https://router.huggingface.co/v1` | `Authorization: Bearer hf_...` | `Qwen/Qwen2.5-72B-Instruct` and other hosted hub models | ✅ | https://huggingface.co/docs/inference-providers |
| **DeepSeek** | `https://api.deepseek.com/v1` | `Authorization: Bearer sk-...` | `deepseek-chat`, `deepseek-reasoner` | ✅ | https://api-docs.deepseek.com/ |
| **Perplexity** (web-grounded answers) | `https://api.perplexity.ai` | `Authorization: Bearer pplx-...` | `sonar`, `sonar-pro`, `sonar-reasoning` | ✅ | https://docs.perplexity.ai/ |
| **xAI (Grok)** | `https://api.x.ai/v1` | `Authorization: Bearer xai-...` | `grok-4`, `grok-3-mini` | ✅ | https://docs.x.ai/ |
| **Cerebras** (ultra-fast inference) | `https://api.cerebras.ai/v1` | `Authorization: Bearer csk-...` | check `GET /v1/models` — catalog rotates (Llama/Qwen/GPT-OSS family) | ✅ | https://inference-docs.cerebras.ai/ |

Notes from live testing (June 2026): model **slugs change often** — on 404 "model does not exist", call `GET {base_url}/models` to list what your key can access (e.g., Cerebras rotated its Llama slugs; OpenRouter retired `mistral-large-2411` in favor of newer Mistral slugs). Common failure codes: `401` invalid/revoked key, `402` no balance (DeepSeek), `403` key blocked (xAI), `429` quota (free tiers).

## Provider-specific quirks

- **Anthropic**: requires the `anthropic-version` header; responses come in `content[].text`; system prompt is a top-level `system` field, not a message.
- **Gemini**: request body is `{"contents":[{"parts":[{"text":...}]}]}` and generation params live in `generationConfig` (`maxOutputTokens`); free tier has strict per-model RPM/TPD quotas (429 on `gemini-2.5-pro` is common — fall back to `flash`/`flash-lite`).
- **OpenRouter**: one key → hundreds of models (Anthropic/OpenAI/Google/Meta/Qwen/DeepSeek…); optional headers `HTTP-Referer`/`X-Title` for attribution; `:free` model variants exist with rate limits; great for **multi-model fan-out without holding 10 vendor keys**.
- **HF Inference Providers**: the router picks a backing provider (Together/Fireworks/etc.); availability varies per model; the classic `api-inference.huggingface.co` serverless endpoint still exists for raw `pipeline` tasks.
- **Perplexity**: answers are **web-search-grounded with citations** — best for "what's new" queries rather than pure reasoning.
- **Cerebras/DeepSeek/xAI**: strictly OpenAI-compatible — just swap `base_url`.
- **Kaggle** (data, not LLM): `pip install kaggle`, credentials in `~/.kaggle/kaggle.json` (`KAGGLE_USERNAME`/`KAGGLE_KEY` env vars also work).

## Multi-model orchestration pattern (no repeated answers)

To aggregate several models without duplicated content: give **each model a different sub-question/angle** (not the same prompt), then merge and de-duplicate:

```python
ANGLES = {
  "openrouter:qwen/qwen-2.5-72b-instruct": "Angle A ...",
  "openrouter:deepseek/deepseek-chat":     "Angle B ...",
  "gemini-2.5-flash":                       "Angle C ...",
}
# fan out with ThreadPoolExecutor -> collect -> dedupe by normalized key -> VERIFY each claim before publishing
```

⚠️ **Anti-hallucination rule:** treat every model answer as a *lead*, not a fact — verify names/URLs/papers against primary sources before publishing (models routinely invent repos and citations).

## Security checklist
- Keys in env vars / secret managers only; `.gitignore` patterns for `*.key`, `.env`, `*CHAVE*`, `kaggle.json`.
- Scan diffs for `sk-`, `hf_`, `AIza`, `pplx-`, `xai-`, `csk-`, `sk-or-` patterns before every commit.
- Rotate any key that ever touches a chat, log, or upload.

## Related in AIForge
- [`../README.md`](../README.md) (Guides & Tutorials) · [`../../02_LLM_AND_AI_MODELS/`](../../02_LLM_AND_AI_MODELS/) · [`../../04_MLOPS_AND_PRODUCTION_AI/`](../../04_MLOPS_AND_PRODUCTION_AI/) (inference/serving)

**Keywords:** LLM API integration, OpenAI-compatible API, OpenRouter, Gemini API, Anthropic API, DeepSeek API, Perplexity Sonar, Cerebras inference, Hugging Face inference providers, multi-model orchestration, integração de APIs de LLM, guia de integração.
