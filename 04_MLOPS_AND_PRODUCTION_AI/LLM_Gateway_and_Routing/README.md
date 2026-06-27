# LLM Gateway and Routing

> An LLM gateway is a unified proxy in front of many model providers (OpenAI-compatible API, fallbacks, load balancing, rate limiting, semantic caching, observability, budget control); routing is the policy layer that picks *which* model serves each request to optimize cost, quality, and latency.

## Why it matters

LLM gateways sit between model serving and applications, decoupling app code from any single vendor and giving platform teams one control point for cost, reliability, and governance. Multi-provider routing plus fallbacks turns a single-provider outage or rate limit into a non-event, while semantic caching and weak/strong model routing routinely cut spend 40-90% with little quality loss. As fleets grow to dozens of models, the routing policy — not the model — becomes the dominant lever on the cost/quality frontier.

## Taxonomy

| Layer | What it does | Examples |
|---|---|---|
| **Unified proxy / gateway** | One OpenAI-compatible endpoint over N providers; auth, keys, virtual budgets | LiteLLM, Portkey, Kong AI Gateway, Cloudflare AI Gateway, OpenRouter |
| **Reliability routing** | Fallbacks, retries, load balancing across deployments/regions | LiteLLM Router, Portkey configs, Bifrost |
| **Cost/quality routing** | Weak↔strong model selection per query difficulty | RouteLLM, Hybrid LLM, Arch-Router, Martian |
| **Cascade routing** | Query cheap model first, escalate on low confidence | FrugalGPT, AutoMix |
| **Caching** | Exact-match + semantic (embedding-similarity) response cache | GPTCache, Portkey, Cloudflare AI Gateway, Redis Semantic Cache |
| **Observability / governance** | Logging, tracing, guardrails, rate/spend limits | Helicone, Portkey, Langfuse, Kong |

## Key tools and gateways

| Tool | Type | Notes | Link |
|---|---|---|---|
| LiteLLM | Proxy + SDK | OpenAI-compatible, 100+ providers, router with fallbacks/load-balancing, budgets, admin UI; self-host default | https://github.com/BerriAI/litellm |
| Portkey AI Gateway | Proxy (OSS + cloud) | Fast edge gateway, configs, semantic caching, guardrails, observability | https://github.com/Portkey-AI/gateway |
| Kong AI Gateway | API-gateway plugin | AI proxy/routing on Kong; fits orgs already running Kong | https://docs.konghq.com/gateway/latest/ai-gateway/ |
| Cloudflare AI Gateway | Managed (edge) | Zero-ops caching, rate limiting, analytics on Cloudflare's edge | https://developers.cloudflare.com/ai-gateway/ |
| OpenRouter | Managed marketplace | Neutral router/marketplace; broad model + pricing catalog | https://openrouter.ai/ |
| Bifrost (Maxim) | OSS gateway | High-throughput Go gateway, OpenAI-compatible, failover | https://github.com/maximhq/bifrost |
| Helicone | Observability proxy | Logging/tracing layer, often fronted by LiteLLM | https://github.com/Helicone/helicone |
| Langfuse | Observability | LLM tracing/evals/prompt mgmt; pairs with gateways | https://github.com/langfuse/langfuse |

## Routing methods and caching

| Method / Tool | Approach | Link |
|---|---|---|
| RouteLLM | Learned router (matrix factorization, BERT, LLM classifier) trained on preference data; strong↔weak | https://github.com/lm-sys/RouteLLM |
| Hybrid LLM | Quality- and budget-aware router; tunable quality threshold at test time | https://arxiv.org/abs/2404.14618 |
| Arch-Router (Katanemo) | 1.5B preference-aligned router mapping queries to domain/action | https://huggingface.co/katanemo/Arch-Router-1.5B |
| FrugalGPT | LLM cascade: query cheap models first, escalate on low score | https://arxiv.org/abs/2305.05176 |
| GPTCache | Semantic cache library; pluggable embeddings + vector stores | https://github.com/zilliztech/GPTCache |
| Redis Semantic Cache | Vector-similarity cache via RedisVL | https://github.com/redis/redis-vl-python |
| vLLM Semantic Router | Intelligent semantic routing in the vLLM production stack | https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/semantic-router-integration.html |

## Benchmarks

| Benchmark | What it measures | Link |
|---|---|---|
| RouterBench | Multi-LLM routing systems; 405k+ inference outcomes | https://arxiv.org/abs/2403.12031 |
| RouteLLM eval suite | Router accuracy vs cost on MT-Bench, MMLU, GSM8K | https://github.com/lm-sys/RouteLLM |
| RouterArena | Open platform for comprehensive router comparison | https://arxiv.org/abs/2510.00202 |

## Key papers

| Paper | Year | Link |
|---|---|---|
| FrugalGPT: How to Use LLMs While Reducing Cost and Improving Performance | 2023 | https://arxiv.org/abs/2305.05176 |
| RouterBench: A Benchmark for Multi-LLM Routing System | 2024 | https://arxiv.org/abs/2403.12031 |
| Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing | 2024 | https://arxiv.org/abs/2404.14618 |
| RouteLLM: Learning to Route LLMs with Preference Data | 2024 | https://arxiv.org/abs/2406.18665 |
| Arch-Router: Aligning LLM Routing with Human Preferences | 2025 | https://arxiv.org/abs/2506.16655 |
| RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers | 2025 | https://arxiv.org/abs/2510.00202 |

## Cross-references in AIForge

- [LLM Inference](../LLM_Inference/) — the serving layer a gateway sits in front of
- [Inference Optimization](../Inference_Optimization/) — throughput/latency techniques routing builds on
- [Cost Optimization and FinOps](../Cost_Optimization_and_FinOps/) — budgets and spend control that routing/caching drive
- [AI Observability](../AI_Observability/) — tracing/logging gateways emit
- [Guardrails and Safety](../Guardrails_and_Safety/) — policy enforcement at the gateway

## Sources

- https://github.com/BerriAI/litellm
- https://github.com/Portkey-AI/gateway
- https://docs.konghq.com/gateway/latest/ai-gateway/
- https://developers.cloudflare.com/ai-gateway/
- https://github.com/lm-sys/RouteLLM
- https://www.lmsys.org/blog/2024-07-01-routellm/
- https://github.com/zilliztech/GPTCache
- https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/semantic-router-integration.html
- https://arxiv.org/abs/2305.05176
- https://arxiv.org/abs/2403.12031
- https://arxiv.org/abs/2404.14618
- https://arxiv.org/abs/2406.18665
- https://arxiv.org/abs/2506.16655
- https://arxiv.org/abs/2510.00202
