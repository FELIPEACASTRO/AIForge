# Guardrails and Safety

> Production controls that validate, filter, and constrain LLM inputs and outputs — blocking prompt injection/jailbreaks, redacting PII, detecting toxicity and unsafe content, and enforcing schema/format — so GenAI systems stay safe, compliant, and reliable.

## Why it matters

LLMs are non-deterministic and adversarially attackable: a single crafted prompt can jailbreak a model, exfiltrate secrets, leak PII, or produce harmful, off-brand, or non-compliant output. Guardrails are the runtime safety layer between users, the model, and downstream tools — the production analogue of input validation and output sanitization. They have become a baseline requirement for any production GenAI deployment and a core part of OWASP's Top 10 for LLM Applications.

## Taxonomy

| Layer | What it does | Example controls |
|---|---|---|
| **Input guardrails** | Inspect/clean the user prompt before it reaches the LLM | Prompt-injection & jailbreak detection, PII redaction, topic/off-topic filtering, language/length checks |
| **Output guardrails** | Validate/transform the model response before returning it | Toxicity & hate-speech filtering, hallucination/groundedness checks, schema/JSON validation, secret & PII scrubbing |
| **Retrieval / tool guardrails** | Constrain RAG context and tool/agent actions | Context grounding, allow-listed tools, code-execution sandboxing, privilege/scope limits |
| **Policy & taxonomy** | Define the harm categories enforced | MLCommons hazard taxonomy, OWASP LLM Top 10, custom org policy |
| **Approach** | How a guardrail decides | (a) **Classifier/guard models** (Llama Guard, ShieldGemma), (b) **rule/validator frameworks** (Guardrails AI, NeMo), (c) **programmable dialogue flows** (Colang) |

## Key models (guard / classifier models)

| Model | Vendor | Notes | Link |
|---|---|---|---|
| Llama Guard 3 (8B / 1B / Vision) | Meta | Input+output safety classifier, multilingual, 14-category MLCommons taxonomy | https://huggingface.co/meta-llama/Llama-Guard-3-8B |
| Prompt Guard 2 | Meta | Small DeBERTa-based jailbreak/prompt-injection input filter | https://github.com/meta-llama/PurpleLlama |
| ShieldGemma (2B / 9B / 27B) | Google | Gemma2-based content moderation; reports +10.8% AU-PRC vs Llama Guard | https://huggingface.co/google/shieldgemma-2b |
| Granite Guardian 3 | IBM | Harm + RAG hallucination/groundedness + jailbreak + bias detection | https://huggingface.co/ibm-granite/granite-guardian-3.0-8b |
| WildGuard (7B) | AI2 | One-stop moderation: harm, jailbreak, and refusal detection | https://huggingface.co/allenai/wildguard |
| ToxicChat-T5 / toxicity classifiers | LMSYS | Toxicity detection trained on real chatbot traffic | https://huggingface.co/lmsys/toxicchat-t5-large-v1.0 |

## Key frameworks and tools

| Tool | Type | Strengths | Link |
|---|---|---|---|
| NVIDIA NeMo Guardrails | Programmable dialogue rails (Colang) | Topical/dialogue flow control, full pipeline, vendor-neutral | https://github.com/NVIDIA/NeMo-Guardrails |
| Guardrails AI | Validator framework + Hub | Pluggable validators, structured output, RAIL spec | https://github.com/guardrails-ai/guardrails |
| Meta PurpleLlama | Guard models + tooling | Llama Guard, Prompt Guard, Code Shield, CyberSecEval | https://github.com/meta-llama/PurpleLlama |
| Microsoft Presidio | PII detection/anonymization | Configurable recognizers, redaction, image/text | https://github.com/microsoft/presidio |
| ProtectAI LLM Guard | Input/output scanners | Toxicity, injection, PII, secrets, ban-topics scanners | https://github.com/protectai/llm-guard |
| Rebuff | Prompt-injection detector | Self-hardening, canary tokens (note: higher latency/FPR) | https://github.com/protectai/rebuff |
| Vigil | Prompt-injection scanner | YARA + embedding + transformer signatures | https://github.com/deadbits/vigil-llm |
| Lakera Guard | Hosted API | Low-latency injection/PII/content filtering | https://www.lakera.ai/lakera-guard |
| OpenAI Moderation API | Hosted API | Free content moderation across harm categories | https://platform.openai.com/docs/guides/moderation |
| Azure AI Content Safety | Hosted API | Content + jailbreak (Prompt Shields) + groundedness | https://learn.microsoft.com/azure/ai-services/content-safety/overview |

## Benchmarks and datasets

| Benchmark / dataset | Focus | Link |
|---|---|---|
| JailbreakBench | Open robustness benchmark for jailbreaks (NeurIPS 2024) | https://github.com/JailbreakBench/jailbreakbench |
| HarmBench | Standardized automated red-teaming framework | https://github.com/centerforaisafety/HarmBench |
| AdvBench (GCG) | 520 adversarial harmful-behavior prompts | https://github.com/llm-attacks/llm-attacks |
| ToxicChat | Real-world Vicuna chat prompts annotated for toxicity/jailbreak | https://huggingface.co/datasets/lmsys/toxic-chat |
| WildGuardMix | 87K adversarial + benign moderation examples | https://huggingface.co/datasets/allenai/wildguardmix |
| AILuminate (MLCommons) | Standardized AI safety benchmark + hazard taxonomy | https://mlcommons.org/benchmarks/ailuminate/ |
| SafetyPrompts.com | Index of LLM safety datasets | https://safetyprompts.com/ |

## Key papers

| Year | Paper | Link |
|---|---|---|
| 2023 | Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations | https://arxiv.org/abs/2312.06674 |
| 2024 | ShieldGemma: Generative AI Content Moderation Based on Gemma | https://arxiv.org/abs/2407.21772 |
| 2024 | WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals | https://arxiv.org/abs/2406.18495 |
| 2024 | Granite Guardian (comprehensive LLM safeguarding) | https://arxiv.org/abs/2412.07724 |
| 2022 | Constitutional AI: Harmlessness from AI Feedback | https://arxiv.org/abs/2212.08073 |
| 2024 | JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs | https://arxiv.org/abs/2404.01318 |
| 2024 | HarmBench: A Standardized Evaluation Framework for Automated Red Teaming | https://arxiv.org/abs/2402.04249 |
| 2025 | SoK: Evaluating Jailbreak Guardrails for Large Language Models | https://arxiv.org/abs/2506.10597 |

## Cross-references in AIForge

- [AI Observability](../AI_Observability/) — tracing and monitoring guardrail decisions in production.
- [LLMOps and Prompt Management](../LLMOps_and_Prompt_Management/) — prompt versioning and evaluation alongside safety policies.
- [AI Agents](../AI_Agents/) — agentic systems where tool/action guardrails are critical.
- [LLM Gateway and Routing](../LLM_Gateway_and_Routing/) — gateways that enforce guardrails as middleware across providers.

## Sources

- https://github.com/NVIDIA/NeMo-Guardrails
- https://github.com/guardrails-ai/guardrails
- https://github.com/meta-llama/PurpleLlama
- https://github.com/microsoft/presidio
- https://huggingface.co/meta-llama/Llama-Guard-3-8B
- https://arxiv.org/abs/2312.06674
- https://arxiv.org/abs/2407.21772
- https://arxiv.org/abs/2406.18495
- https://arxiv.org/abs/2412.07724
- https://github.com/JailbreakBench/jailbreakbench
- https://arxiv.org/abs/2506.10597
- https://learn.microsoft.com/azure/ai-services/content-safety/overview
- https://github.com/protectai/llm-guard
- https://mlcommons.org/benchmarks/ailuminate/

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
