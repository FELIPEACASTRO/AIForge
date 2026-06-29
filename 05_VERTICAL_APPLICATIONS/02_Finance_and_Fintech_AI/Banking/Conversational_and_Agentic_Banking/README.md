# Conversational and Agentic Banking

> AI assistants and tool-using agents that let customers and bankers interact with accounts, products, and policies in natural language — spanning rule-based chatbots, retrieval-grounded LLM copilots, and autonomous multi-step agents operating inside compliance guardrails.

## Why it matters

Banking support is high-volume, repetitive, and unforgiving of errors: a wrong balance, a missed disclosure, or a hallucinated fee can be a regulatory incident. Conversational AI deflects routine call/branch volume at scale — Bank of America's Erica passed 3 billion interactions and reportedly handles work equivalent to ~11,000 staff — while agentic systems are now moving from Q&A into action (transfers, disputes, onboarding, AML case prep). The central tension is autonomy vs. control: banks want agents that *do* things, but every action must be grounded, logged, auditable, and reversible.

## Core concepts (end-to-end)

A modern conversational/agentic banking stack typically layers:

1. **Channel & ASR/TTS** — mobile app, web, IVR voice (speech-to-text + text-to-speech), messaging.
2. **NLU / intent routing** — classify intent + extract entities (amount, payee, date). Classic platforms (Rasa, Lex, Dialogflow) or an LLM router.
3. **Input guardrails** — jailbreak/prompt-injection detection, PII redaction, topic/scope rails (banking-only), authentication & step-up checks.
4. **Retrieval (RAG)** — ground answers in *bank-owned* sources: product T&Cs, fee schedules, policy/compliance docs, KB articles, account context via internal APIs. Retrieval rails filter/validate chunks.
5. **Reasoning / orchestration** — LLM (or rules) plans steps; an **agent loop** selects tools (balance lookup, transfer API, dispute filing, fraud check) with approval thresholds for sensitive actions.
6. **Output guardrails** — hallucination/grounding checks, disclosure injection, fact/format validation, fallback to human handoff.
7. **Action + audit** — execute via APIs with full traceability; log prompt, retrieval, tool calls, and decision rationale for model risk & examiners.
8. **Human-in-the-loop & eval** — escalation paths, offline evaluation, continuous monitoring for drift, toxicity, and accuracy.

Two postures coexist in production: **deterministic assistants** (BofA's Erica deliberately avoids generative LLMs for accuracy in customer-facing answers) and **generative copilots/agents** (banker-facing or tightly-scoped, RAG-grounded). Most banks deploy agentic AI first in lower-risk back-office flows (AML case prep, KYC support, compliance drafting) before fully autonomous customer actions.

## Techniques / Models

| Technique | Role in conversational/agentic banking |
|---|---|
| Intent classification + NER (NLU) | Map utterances to intents; extract slots (payee, amount). Backbone of rule-based assistants like Erica. |
| Retrieval-Augmented Generation (RAG) | Ground LLM answers in policies, fee schedules, KB; reduce hallucination; cite sources. |
| LLM agents / tool use (ReAct, function calling) | Plan multi-step tasks; call account/transfer/dispute APIs within thresholds. |
| Domain-adapted / fine-tuned LLMs | Banking-specific models (e.g., Kasisto KAI-GPT, FinGPT) for tone, jargon, accuracy. |
| Guardrail models (classifiers) | Jailbreak/prompt-injection detection, toxicity, PII, topical scope (Llama Guard, NeMo rails). |
| Hallucination / groundedness checks | Self-consistency, NLI entailment vs. retrieved context, faithfulness scoring. |
| Embeddings + vector search | Semantic retrieval over docs/KB; reranking for precision. |
| ASR / TTS (speech) | Voice IVR and voice assistants. |
| Sentiment / emotion detection | Detect frustration; trigger human handoff. |
| Reinforcement learning / RLHF | Align tone, refusal behavior, and policy compliance. |

## Tools, Vendors & Open-Source

| Name | Type | What it does | URL |
|---|---|---|---|
| Kasisto KAI / KAI-GPT | Vendor | Banking-specific conversational AI & domain LLM | https://kasisto.com/products/ai-agents/ |
| Bank of America Erica | In-house product | Deterministic virtual assistant (reference case) | https://info.bankofamerica.com/en/digital-banking/erica |
| IBM watsonx Assistant | Vendor | Enterprise conversational AI, used in banking | https://www.ibm.com/products/watsonx-assistant |
| Amazon Lex | Cloud | ASR + NLU bot platform (Alexa tech) | https://aws.amazon.com/lex/ |
| Google Dialogflow CX | Cloud | NLU/agent platform with flows + generative | https://cloud.google.com/dialogflow |
| Rasa | Open-source | Self-hosted conversational AI (CALM, LLM-native) | https://rasa.com/ |
| NVIDIA NeMo Guardrails | Open-source | Programmable input/dialog/retrieval/output rails | https://github.com/NVIDIA-NeMo/Guardrails |
| Guardrails AI | Open-source | Output validation framework (validators) | https://www.guardrailsai.com/ |
| Llama Guard | Open-source | LLM-based safety/input-output classifier | https://huggingface.co/meta-llama/Llama-Guard-3-8B |
| Amazon Bedrock Guardrails | Cloud | Managed content/PII/topic guardrails | https://aws.amazon.com/bedrock/guardrails/ |
| LangChain | Open-source | Agent/tool orchestration framework | https://www.langchain.com/ |
| LlamaIndex | Open-source | RAG/indexing & retrieval framework | https://www.llamaindex.ai/ |
| FinGPT | Open-source | Open financial LLMs + data pipeline | https://github.com/AI4Finance-Foundation/FinGPT |
| Backbase | Vendor | Banking engagement platform with agentic AI | https://www.backbase.com/ |

## Datasets & Benchmarks

Conversational-banking-specific public datasets are scarce (most are proprietary), but these are relevant for grounding, NLU, and financial-LLM evaluation:

| Dataset / Benchmark | Use | URL |
|---|---|---|
| Banking77 | Intent classification (77 banking intents) | https://huggingface.co/datasets/PolyAI/banking77 |
| FinBen | Holistic financial LLM benchmark (36 datasets, RAG/agent eval; NeurIPS 2024) | https://arxiv.org/abs/2402.12659 |
| FinQA | Numerical reasoning QA over financial reports | https://github.com/czyssrs/FinQA |
| ConvFinQA | Multi-turn conversational financial QA | https://github.com/czyssrs/ConvFinQA |
| TAT-QA | QA over hybrid tabular + textual financial data | https://github.com/NExTplusplus/TAT-QA |
| Open FinLLM Leaderboard | Public leaderboard for financial LLMs | https://arxiv.org/pdf/2501.10963 |

## Regulations & Standards

- **DORA (Digital Operational Resilience Act, EU)** — applies from 17 Jan 2025; ICT/third-party resilience, central for agentic workflows. https://www.eiopa.europa.eu/digital-operational-resilience-act-dora_en
- **EU AI Act** — risk tiers, transparency for AI systems including chatbots (disclosure that users interact with AI). https://artificialintelligenceact.eu/
- **SR 11-7 (US Fed/OCC) — Model Risk Management** — governance, validation, documentation for models incl. LLMs. https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- **GDPR / LGPD** — data protection, PII handling, automated decisioning rights. https://gdpr.eu/
- **PSD2** — strong customer authentication & open-banking APIs that agents may invoke. https://www.eba.europa.eu/regulation-and-policy/payment-services-and-electronic-money
- **CFPB guidance on chatbots in consumer finance** — risks of inaccurate/automated customer service. https://www.consumerfinance.gov/data-research/research-reports/chatbots-in-consumer-finance/
- **NIST AI Risk Management Framework** — voluntary governance for trustworthy AI. https://www.nist.gov/itl/ai-risk-management-framework
- **OWASP Top 10 for LLM / Agentic AI** — prompt injection, excessive agency, insecure tool use. https://genai.owasp.org/

## Key papers

- *FinBen: A Holistic Financial Benchmark for Large Language Models* (2024). https://arxiv.org/abs/2402.12659
- *FinGPT: Open-Source Financial Large Language Models* (2023). https://arxiv.org/abs/2306.06031
- *NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails* (2023). https://arxiv.org/abs/2310.10501
- *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations* (2023). https://arxiv.org/abs/2312.06674
- *GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning* (2024). https://arxiv.org/abs/2406.09187
- *Towards Policy-Compliant Agents: Learning Efficient Guardrails for Policy Violation Detection* (2025). https://arxiv.org/abs/2510.03485

## Cross-references in AIForge

- Sibling: [Fraud Detection](../Fraud_Detection/) — agents can trigger fraud/AML checks
- Sibling: [Customer Onboarding & KYC — agentic eKYC funnels
- Sibling: [Credit Risk & Underwriting — copilots for credit decisions
- Parent: [Finance & Fintech AI](../../)
- Fundamentals: [LLMs & Transformers
- Fundamentals: [Retrieval-Augmented Generation
- Security: [AI Agent Security

> Note: adjust relative paths above to match the exact sibling/fundamentals directory names in this repository.

## Sources

- Bank of America — Erica milestones: https://newsroom.bankofamerica.com/content/newsroom/press-releases/2025/08/a-decade-of-ai-innovation--bofa-s-virtual-assistant-erica-surpas.html
- American Banker — Erica scale: https://www.americanbanker.com/news/how-bank-of-americas-erica-does-the-work-of-11-000-people
- Kasisto AI Agents: https://kasisto.com/products/ai-agents/
- Deloitte — Agentic AI in banking: https://www.deloitte.com/us/en/insights/industry/financial-services/agentic-ai-banking.html
- NVIDIA NeMo Guardrails: https://github.com/NVIDIA-NeMo/Guardrails / https://developer.nvidia.com/nemo-guardrails
- Guardrails AI: https://www.guardrailsai.com/
- AWS — Bedrock Guardrails: https://aws.amazon.com/bedrock/guardrails/
- Rasa: https://rasa.com/
- FinGPT (AI4Finance): https://github.com/AI4Finance-Foundation/FinGPT
- FinBen (arXiv / NeurIPS 2024): https://arxiv.org/abs/2402.12659
- OWASP Top 10 for Agentic AI (financial institutions): https://www.myabt.com/blog/owasp-top-10-agentic-ai-financial-institutions
- CFPB — Chatbots in consumer finance: https://www.consumerfinance.gov/data-research/research-reports/chatbots-in-consumer-finance/
