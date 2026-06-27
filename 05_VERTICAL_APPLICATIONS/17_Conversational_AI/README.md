# 17 Conversational AI

> Conversational AI builds systems that hold natural multi-turn dialogue with humans — open-domain chatbots, task-oriented assistants, RAG-grounded copilots, and real-time voice agents — combining NLU, dialogue management, generation, and (for voice) speech I/O.

## Why it matters

Conversational interfaces are the dominant way people now interact with LLMs, from customer-support bots to voice assistants and agentic copilots. The field spans decades of task-oriented dialogue research (state tracking, slot filling) and the recent shift to LLM-driven open-ended chat plus low-latency STT→LLM→TTS voice pipelines. Getting it right requires not just a strong base model but dialogue management, retrieval grounding, latency engineering, and rigorous evaluation of helpfulness and safety.

## Taxonomy

| Sub-area | What it covers | Representative tooling |
|---|---|---|
| Open-domain chat | Free-form, multi-turn conversation; chit-chat and assistant-style replies | DialoGPT, instruction-tuned LLMs, FastChat |
| Task-oriented dialogue (TOD) | Goal completion via intent/slot/state tracking + APIs | Rasa, SimpleTOD, TOD-BERT |
| RAG-grounded assistants | Chat grounded in retrieved documents/knowledge | LangChain, LlamaIndex, Haystack |
| Voice / multimodal agents | Real-time STT → LLM → TTS pipelines, barge-in, telephony | Pipecat, LiveKit Agents, TEN |
| Agentic / tool-using chat | Dialogue that calls tools, plans, and acts | LangGraph, agent frameworks |
| Evaluation & arenas | Human-pref and LLM-judge benchmarking of chat quality | Chatbot Arena/LMArena, MT-Bench |

## Key frameworks & tools

| Tool | Focus | Link |
|---|---|---|
| Rasa | Open-source TOD: NLU, intents/entities, dialogue policies | https://github.com/RasaHQ/rasa |
| Pipecat | Real-time voice & multimodal agent pipelines (STT/LLM/TTS) | https://github.com/pipecat-ai/pipecat |
| LiveKit Agents | WebRTC-based realtime voice/video AI agents | https://github.com/livekit/agents |
| TEN Framework | Real-time multimodal conversational AI agents | https://github.com/TEN-framework/TEN-Agent |
| FastChat | Serving, training & eval of chat LLMs; powers LMSYS arena | https://github.com/lm-sys/FastChat |
| LangChain | Chains/agents, memory, tool use for chat apps | https://github.com/langchain-ai/langchain |
| LlamaIndex | Data ingestion + indexing for RAG chatbots | https://github.com/run-llama/llama_index |
| Haystack | Production RAG/QA pipelines (deepset) | https://github.com/deepset-ai/haystack |
| ParlAI | Research platform for dialogue models & tasks | https://github.com/facebookresearch/ParlAI |
| ConvoKit | Toolkit for analyzing conversational dynamics | https://github.com/CornellNLP/ConvoKit |

## Key models

| Model | Type | Link |
|---|---|---|
| DialoGPT | Open-domain response generation (Reddit-pretrained GPT-2) | https://huggingface.co/microsoft/DialoGPT-large |
| BlenderBot 3 | Open-domain chat with retrieval & memory | https://huggingface.co/facebook/blenderbot-3B |
| TOD-BERT | Pretrained encoder for task-oriented dialogue | https://github.com/jasonwu0731/ToD-BERT |
| SimpleTOD | Single causal LM for end-to-end TOD | https://github.com/salesforce/simpletod |
| Vicuna | Instruction-tuned chat LLM (from FastChat) | https://github.com/lm-sys/FastChat#vicuna-weights |

## Benchmarks & datasets

| Resource | Task | Link |
|---|---|---|
| MultiWOZ | Multi-domain TOD / dialogue state tracking (10k dialogues) | https://github.com/budzianowski/multiwoz |
| Schema-Guided Dialogue (SGD) | Scalable multi-domain DST, unseen-service generalization | https://github.com/google-research-datasets/dstc8-schema-guided-dialogue |
| DialoGLUE | NLU benchmark suite for task-oriented dialogue | https://github.com/alexa/dialoglue |
| MT-Bench | 80 multi-turn questions, LLM-as-judge scoring | https://huggingface.co/spaces/lmsys/mt-bench |
| Chatbot Arena / LMArena | Crowdsourced pairwise human-preference Elo | https://lmarena.ai/ |
| STAR | Schema-guided transfer-learning dialog dataset | https://arxiv.org/abs/2010.11853 |

## Key papers

| Paper | Year | Link |
|---|---|---|
| DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation | 2019 | https://arxiv.org/abs/1911.00536 |
| MultiWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset for TOD | 2018 | https://arxiv.org/abs/1810.00278 |
| Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset | 2019 | https://arxiv.org/abs/1909.05855 |
| TOD-BERT: Pre-trained NLU for Task-Oriented Dialogue | 2020 | https://arxiv.org/abs/2004.06871 |
| A Simple Language Model for Task-Oriented Dialogue (SimpleTOD) | 2020 | https://arxiv.org/abs/2005.00796 |
| MultiWOZ 2.2: Annotation Corrections and State Tracking Baselines | 2020 | https://arxiv.org/abs/2007.12720 |
| DialoGLUE: A Natural Language Understanding Benchmark for TOD | 2020 | https://arxiv.org/abs/2009.13570 |
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | 2023 | https://arxiv.org/abs/2306.05685 |
| Recipes for Building an Open-Domain Chatbot (BlenderBot) | 2020 | https://arxiv.org/abs/2004.13637 |

## Cross-references in AIForge

- [../27_Customer_Support_AI](../27_Customer_Support_AI) — applied support bots and deflection
- [../25_Telecom_and_Network_AI](../25_Telecom_and_Network_AI) — IVR and voice-agent deployment context
- [../12_Business_and_Marketing_AI](../12_Business_and_Marketing_AI) — conversational marketing & sales assistants
- [../05_Education_AI](../05_Education_AI) — tutoring and dialogue-based learning agents

## Sources

- https://github.com/pipecat-ai/pipecat
- https://github.com/livekit/agents
- https://github.com/RasaHQ/rasa
- https://github.com/lm-sys/FastChat
- https://github.com/budzianowski/multiwoz
- https://arxiv.org/abs/1911.00536
- https://arxiv.org/abs/1810.00278
- https://arxiv.org/abs/1909.05855
- https://arxiv.org/abs/2004.06871
- https://arxiv.org/abs/2306.05685
- https://lmarena.ai/
- https://www.firecrawl.dev/blog/best-open-source-rag-frameworks

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
