# 27 Customer Support AI

> AI systems that automate and augment customer service — RAG-grounded support agents, ticket deflection/routing, agent-assist & call summarization, voice IVR with ASR/TTS, and CSAT/sentiment analytics. Distinct from generic [Conversational AI](../17_Conversational_AI/) (dialogue systems), this vertical is policy-driven, tool-using, and business-adherence critical.

## Why it matters

Customer support is one of the highest-ROI LLM deployment areas: AI-handled tickets cost roughly $0.50–$1.05 each versus $8–$12 for human-handled ones, and mature agentic deployments push first-contact resolution into the 70–85% range. Unlike open-ended chat, support agents must follow domain-specific policies, call backend tools (refunds, account changes), and remain robust to adversarial or confused users — making *business-adherence* and *reliability over repeated trials*, not just fluency, the key success metrics. Benchmarks like τ-bench show even frontier function-calling models still succeed on <50% of realistic tasks, so this remains an open, fast-moving problem.

## Taxonomy

| Sub-area | What it does | Representative tech |
|---|---|---|
| RAG support agents | Answer questions grounded in KB/docs/tickets | LangChain/LangGraph, LlamaIndex, contextual retrieval |
| Ticket deflection & self-service | Resolve common queries before a human is involved | Intent classification, FAQ retrieval, agentic workflows |
| Routing & triage | Classify, prioritize, and route tickets to teams/agents | Intent + sentiment models, multi-label classifiers |
| Agent-assist (copilot) | Real-time suggestions, KB lookup, reply drafting | Retrieval + generation in the agent console |
| Call/chat summarization | After-call wrap-up notes, ticket summaries | Abstractive/extractive summarization LLMs |
| Voice IVR / contact center | Speech-to-speech support, full-duplex voice agents | ASR + TTS + dialog (Riva, Whisper, Coqui) |
| Analytics & QA | CSAT/sentiment, compliance/policy-adherence scoring | Sentiment models, LLM judges, journey-coverage metrics |

## Key frameworks & tools

| Tool | Role | Link |
|---|---|---|
| Rasa | Open-source conversational AI for support assistants | https://github.com/RasaHQ/rasa |
| LangChain | LLM app + RAG framework (retrieval chains) | https://github.com/langchain-ai/langchain |
| LangGraph | Stateful, multi-step agent loops (plan-execute-verify) | https://github.com/langchain-ai/langgraph |
| LlamaIndex | Data-first RAG over PDFs/DBs/APIs | https://github.com/run-llama/llama_index |
| Open Agent Platform | No-code agent building platform (LangChain) | https://github.com/langchain-ai/open-agent-platform |
| NVIDIA Riva | Speech AI (ASR/TTS) for contact centers | https://developer.nvidia.com/riva |
| Whisper | Open-source ASR for voice support | https://github.com/openai/whisper |
| Coqui TTS | Open-source text-to-speech for voice agents | https://github.com/coqui-ai/TTS |
| Anthropic contextual retrieval | RAG technique reducing retrieval failures for support KBs | https://www.anthropic.com/news/contextual-retrieval |

## Benchmarks

| Benchmark | Focus | Link |
|---|---|---|
| τ-bench | Tool-Agent-User interaction (retail, airline); pass^k reliability metric | https://github.com/sierra-research/tau-bench |
| τ²-bench | Dual-control conversational agents (adds telecom domain, Dec-POMDP) | https://github.com/sierra-research/tau2-bench |
| JourneyBench (Beyond IVR) | Policy-aware support agents; User Journey Coverage Score | https://aclanthology.org/2026.eacl-industry.15/ |
| ABCD | Action-Based Conversations Dataset: policy-adherent retail support | https://github.com/asappresearch/abcd |
| MultiWOZ | Multi-domain task-oriented dialogue benchmark | https://github.com/budzianowski/multiwoz |
| Schema-Guided Dialogue (SGD) | 16k+ multi-domain dialogues, dialogue state tracking | https://github.com/google-research-datasets/dstc8-schema-guided-dialogue |

## Datasets

| Dataset | Description | Link |
|---|---|---|
| Bitext customer-support intents | 27 intents, support-bot training data | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset |
| ABCD | 10,042 grounded retail-support conversations w/ 55 intents | https://github.com/asappresearch/abcd |
| MultiWOZ | Large-scale Wizard-of-Oz task-oriented dialogue | https://github.com/budzianowski/multiwoz |
| Schema-Guided Dialogue | 16k+ conversations, 16 domains | https://github.com/google-research-datasets/dstc8-schema-guided-dialogue |
| STAR | Schema-guided dialog for transfer learning | https://github.com/RasaHQ/STAR |

## Key papers

| Paper | Year | Link |
|---|---|---|
| τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains | 2024 | https://arxiv.org/abs/2406.12045 |
| τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment | 2025 | https://arxiv.org/abs/2506.07982 |
| Action-Based Conversations Dataset (ABCD) | 2021 | https://arxiv.org/abs/2104.00783 |
| Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset | 2019 | https://arxiv.org/abs/1909.05855 |
| Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence | 2026 | https://arxiv.org/abs/2601.00596 |
| Prompting and Fine-Tuning of Small LLMs for Length-Controllable Telephone Call Summarization | 2024 | https://arxiv.org/abs/2410.18624 |
| Leveraging Explicit Procedural Instructions for Data-Efficient Action Prediction | 2023 | https://arxiv.org/abs/2306.03959 |

## Cross-references in AIForge

- [17 Conversational AI](../17_Conversational_AI/) — underlying dialogue systems and NLU/NLG.
- [25 Telecom and Network AI](../25_Telecom_and_Network_AI/) — telecom-domain support (τ²-bench dual-control).
- [07 Retail and Ecommerce](../07_Retail_and_Ecommerce/) — order/refund support workflows (τ-bench retail, ABCD).
- [26 HR and Recruiting AI](../26_HR_and_Recruiting_AI/) — adjacent enterprise conversational assistants.

## Sources

- https://arxiv.org/abs/2406.12045 — τ-bench
- https://github.com/sierra-research/tau-bench
- https://arxiv.org/abs/2506.07982 — τ²-bench
- https://github.com/sierra-research/tau2-bench
- https://arxiv.org/abs/2601.00596 — Beyond IVR / JourneyBench
- https://aclanthology.org/2026.eacl-industry.15/
- https://arxiv.org/abs/2104.00783 — ABCD
- https://arxiv.org/abs/1909.05855 — Schema-Guided Dialogue
- https://arxiv.org/abs/2410.18624 — call summarization with small LLMs
- https://github.com/RasaHQ/rasa
- https://github.com/langchain-ai/langgraph
- https://developer.nvidia.com/riva
- https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
