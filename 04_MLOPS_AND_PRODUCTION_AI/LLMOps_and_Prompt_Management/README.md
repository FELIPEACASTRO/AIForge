# LLMOps and Prompt Management

> The operational discipline of building, versioning, evaluating, observing, and continuously improving LLM-powered applications — with the prompt (and its variants, datasets, and eval scores) treated as a first-class, version-controlled artifact rather than a string buried in application code.

## Why it matters

LLM behavior is steered primarily by prompts, models, and context — none of which are captured by traditional MLOps tooling built for weights and feature pipelines. A one-line prompt edit can silently change accuracy, cost, and latency across millions of calls, so teams need prompt registries, A/B testing, trace-level observability, and offline/online evals to ship changes safely. LLMOps closes this loop: decouple prompts from code, measure every change against datasets, trace production calls, and roll back fast. It is distinct from general AI observability (collection/monitoring) in that it owns the full prompt lifecycle — authoring, experimentation, deployment, and governance.

## Taxonomy

| Sub-area | What it covers | Representative tooling |
|---|---|---|
| Prompt registry / versioning | Decouple prompts from code; version, label (e.g. `prod`/`staging`), deploy & roll back | PromptLayer, Langfuse Prompts, MLflow Prompt Registry |
| Prompt experimentation / playground | Side-by-side runs, model/param sweeps, structured-output testing | Agenta, Langfuse Playground, PromptHub |
| LLM tracing & observability | Span-level capture of prompts, completions, tool calls, tokens, cost, latency | Langfuse, Helicone, Arize Phoenix, LangSmith |
| Offline evaluation | Dataset-driven scoring (LLM-as-judge, heuristics, assertions) on commit/merge | promptfoo, DeepEval, OpenAI Evals |
| Online evaluation & monitoring | Live scoring, drift/regression alerting, human feedback capture | Langfuse, Arize, LangSmith |
| Automatic prompt optimization | Programmatic compilation/search/evolution of prompts to a metric | DSPy, TextGrad, GEPA, APE |
| Gateway / governance | Caching, routing, rate-limit, PII redaction, cost guardrails | Helicone, LiteLLM, Portkey |

## Key tools / frameworks

| Tool | Focus | License / type | Link |
|---|---|---|---|
| Langfuse | Tracing + prompt management + evals + datasets + playground | Open source (MIT core) | https://github.com/langfuse/langfuse |
| Helicone | Observability + prompt versioning + AI gateway (OpenAI-compatible) | Open source | https://github.com/Helicone/helicone |
| promptfoo | Prompt/RAG/agent testing, eval & red-teaming (CLI + CI) | Open source | https://github.com/promptfoo/promptfoo |
| Agenta | LLMOps: prompt mgmt + evaluation + observability, dev/non-dev collab | Open source | https://github.com/Agenta-AI/agenta |
| LangSmith | Tracing, prompt hub, eval (tight LangChain/LangGraph integration) | Commercial (free tier) | https://docs.smith.langchain.com/ |
| PromptLayer | Prompt CMS / registry, A/B testing, analytics for prompt lifecycle | Commercial | https://www.promptlayer.com/ |
| DSPy | Programming (not prompting) LMs; compile pipelines to a metric | Open source | https://github.com/stanfordnlp/dspy |
| Arize Phoenix | OSS AI observability & evaluation, OpenInference tracing | Open source | https://github.com/Arize-ai/phoenix |
| OpenInference | OpenTelemetry-complementary semantic conventions for LLM spans | Open standard | https://github.com/Arize-ai/openinference |
| MLflow (GenAI) | Prompt Registry + tracing + LLM evaluation in MLflow | Open source | https://github.com/mlflow/mlflow |
| OpenAI Evals | Framework + registry of evals for LLMs and prompt chains | Open source | https://github.com/openai/evals |
| DeepEval | Pytest-style LLM eval (RAG, agents) with metrics | Open source | https://github.com/confident-ai/deepeval |
| LiteLLM | Unified gateway/SDK for 100+ providers (routing, cost, caching) | Open source | https://github.com/BerriAI/litellm |
| TextGrad | "Backprop" textual feedback to optimize prompts/components | Open source | https://github.com/zou-group/textgrad |
| GEPA | Reflective/genetic-Pareto prompt evolution optimizer | Open source | https://github.com/gepa-ai/gepa |

## Standards & conventions

| Standard | Purpose | Link |
|---|---|---|
| OpenTelemetry GenAI semantic conventions | Standard span attributes for LLM/agent traces (`gen_ai.*`) | https://opentelemetry.io/docs/specs/semconv/gen-ai/ |
| OpenTelemetry semantic-conventions-genai repo | Source of the evolving GenAI conventions | https://github.com/open-telemetry/semantic-conventions-genai |
| OpenInference spec | Vendor-neutral LLM tracing conventions on top of OTel | https://arize-ai.github.io/openinference/spec/semantic_conventions.html |

## Key papers

| Paper | Year | Link |
|---|---|---|
| The Prompt Report: A Systematic Survey of Prompt Engineering Techniques | 2024 | https://arxiv.org/abs/2406.06608 |
| A Systematic Survey of Prompt Engineering in LLMs: Techniques and Applications | 2024 | https://arxiv.org/abs/2402.07927 |
| DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines | 2023 | https://arxiv.org/abs/2310.03714 |
| Large Language Models Are Human-Level Prompt Engineers (APE) | 2022 | https://arxiv.org/abs/2211.01910 |
| TextGrad: Automatic "Differentiation" via Text | 2024 | https://arxiv.org/abs/2406.07496 |
| GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning | 2025 | https://arxiv.org/abs/2507.19457 |
| Is It Time To Treat Prompts As Code? A Multi-Use-Case Study (DSPy) | 2025 | https://arxiv.org/abs/2507.03620 |

## Cross-references in AIForge

- [AI Observability](../AI_Observability/) — collection/monitoring layer that LLMOps prompt tracing builds on.
- [LLM Gateway and Routing](../LLM_Gateway_and_Routing/) — gateways (LiteLLM, Helicone) that front prompt traffic with caching, routing, and cost controls.
- [Guardrails and Safety](../Guardrails_and_Safety/) — assertion/guardrail layer often wired into prompt eval pipelines.
- [AB Testing and Canary](../AB_Testing_and_Canary/) — staged rollout patterns applied to prompt versions.

## Sources

- https://github.com/langfuse/langfuse
- https://github.com/Helicone/helicone
- https://github.com/Agenta-AI/agenta
- https://github.com/stanfordnlp/dspy
- https://github.com/Arize-ai/phoenix
- https://github.com/open-telemetry/semantic-conventions-genai
- https://opentelemetry.io/docs/specs/semconv/gen-ai/
- https://www.zenml.io/blog/best-prompt-management-tools
- https://agenta.ai/blog/top-open-source-prompt-management-platforms
- https://arxiv.org/abs/2406.06608
- https://arxiv.org/abs/2310.03714
- https://arxiv.org/abs/2406.07496
- https://arxiv.org/abs/2507.19457

_Expanded from the seed gap-sweep section. Contributions welcome (see CONTRIBUTING.md)._
