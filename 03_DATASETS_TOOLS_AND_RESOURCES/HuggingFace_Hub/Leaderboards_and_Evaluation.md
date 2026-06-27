# HuggingFace Leaderboards & Evaluation Spaces — The "Competitions" of HF

> A curated, link-verified index of the major evaluation leaderboards and "arenas" hosted on (or anchored to) the Hugging Face Hub. These are the closest thing the HF ecosystem has to Kaggle-style competitions: public, continuously-updated rankings where models compete on standardized benchmarks or crowd-sourced human preference. Every entry below carries at least one real, public link.

**Two evaluation paradigms dominate:**

| Paradigm | How it ranks | Examples |
|---|---|---|
| `static_benchmark` | Fixed datasets, automated scoring (lm-eval-harness / lighteval / VLMEvalKit), normalized accuracy | Open LLM Leaderboard v2, MTEB, BigCodeBench, Open ASR, Open Medical-LLM |
| `human_preference_arena` | Anonymous A/B votes → Bradley-Terry / Elo rating | LMArena (Chatbot Arena), TTS Arena, Vision Arena / WildVision, MTEB Arena |

---

## 1. General LLM Leaderboards

### Open LLM Leaderboard v2 (archived)
- **Org:** `open-llm-leaderboard` / `OpenEvals` (Hugging Face)
- **Status:** Live June 2024 → retired/archived in 2025 ("performances are plateauing"). Now read-only.
- **What it measures:** Six harder benchmarks chosen to fix v1 saturation, run with EleutherAI's `lm-evaluation-harness` and normalized scores:
  - `IFEval` — instruction-following / format compliance
  - `BBH` — BIG-Bench Hard (23 challenging reasoning tasks)
  - `MATH` (Lvl 5) — competition math, hardest split
  - `GPQA` — Google-Proof Q&A, expert-written graduate science
  - `MUSR` — multistep soft reasoning
  - `MMLU-Pro` — harder, deduplicated MMLU with 10 options
- **Why it mattered:** The de-facto open-weights ranking for ~2 years; thousands of community submissions. Top open models in the v2 era were Qwen2.5-72B-Instruct-class and Llama-3.x-70B derivatives.
- **Links:** [Space (archived)](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) · [Org](https://huggingface.co/open-llm-leaderboard) · ["Make it steep again" blog](https://huggingface.co/spaces/open-llm-leaderboard/blog) · [v2 collection](https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-2) · [Archived 2024-2025 collection](https://huggingface.co/collections/OpenEvals/archived-open-llm-leaderboard-2024-2025)

### Chatbot Arena / LMArena (LMSYS → Arena.ai)
- **Org:** LMSYS / LMArena-AI (rebranded to **"Arena"** on Jan 28, 2026; still `arena.ai`). HF mirror: `lmarena-ai/arena-leaderboard`.
- **What it measures:** Blind, side-by-side human preference. Users prompt two anonymous models, vote for the better reply (or tie / both-bad). Votes → **Bradley-Terry MLE** (not raw win-rate, not vanilla Elo) → a stable rating. Beating a strong opponent moves score more than beating a weak one.
- **Scale:** Launched May 2023; **millions of blind votes** across 300+ models — the largest human-preference LLM eval in existence.
- **Notable methodology adds:** **Style Control** filter (mid-2025, decouples answer quality from length/markdown), **Arena Expert** track (Nov 2025, top ~5% hardest prompts), vision & other category boards.
- **Top tier (as of June 2026, per public snapshots):** clustered within ~55 Elo — Claude Opus 4.x, GPT-5.5 (Pro), Gemini 3.x Pro.
- **Links:** [HF arena-leaderboard Space](https://huggingface.co/spaces/lmarena-ai/arena-leaderboard) · [arena.ai/leaderboard](https://arena.ai/leaderboard) · [LMSYS Elo update blog](https://www.lmsys.org/blog/2023-12-07-leaderboard/) · [Leaderboard changelog](https://arena.ai/blog/leaderboard-changelog/)

---

## 2. Embeddings & Retrieval

### MTEB — Massive Text Embedding Benchmark
- **Org:** `mteb` / `embeddings-benchmark`
- **What it measures:** Text-embedding quality across **8 task types** (retrieval, reranking, clustering, classification, STS, pair-classification, summarization, bitext mining). Original paper: 58 datasets / 112 languages, 33 models benchmarked — established the standard way to compare embedding models.
- **Live board:** Interactive leaderboard Space; rankings span language and (now) multimodal tasks. 2025 top open-weight models are built on **Qwen3** and NVIDIA's late-2025 embedding releases.
- **Links:** [MTEB Leaderboard Space](https://huggingface.co/spaces/mteb/leaderboard) · [Legacy board](https://huggingface.co/spaces/mteb/leaderboard_legacy) · [GitHub](https://github.com/embeddings-benchmark/mteb/) · [arXiv 2210.07316](https://arxiv.org/abs/2210.07316) · [User-guide blog](https://huggingface.co/blog/lyon-nlp-group/mteb-leaderboard-best-practices)

### MMTEB — Massive **Multilingual** Text Embedding Benchmark
- **What it measures:** Community-driven expansion of MTEB — **500+ quality-controlled tasks across 250+ languages**, adding instruction-following, long-document retrieval, and code retrieval. Powers the multilingual views of the MTEB leaderboard.
- **Links:** [arXiv 2502.13595](https://arxiv.org/abs/2502.13595) · [mteb org](https://huggingface.co/mteb)

### MTEB Arena
- **What it measures:** Human-preference voting for embeddings — pick which model retrieves the better document / clusters better, etc. The Bradley-Terry/arena analogue for retrieval, complementing the static MTEB board.
- **Link:** [mteb org (Arena listed)](https://huggingface.co/mteb)

---

## 3. Vision-Language (VLM) Leaderboards

### Open VLM Leaderboard (OpenCompass / VLMEvalKit)
- **Org:** `opencompass`
- **What it measures:** Vision-language models scored with **VLMEvalKit** (200+ LMMs, 80+ image & video benchmarks; supports open weights + commercial APIs). Main board + per-dataset breakdowns; companion **Open VLM Video Leaderboard** and **Open LMM Subjective Leaderboard**.
- **Links:** [Open VLM Leaderboard Space](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) · [Video board](https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard) · [Subjective board](https://huggingface.co/spaces/opencompass/openlmm_subjective_leaderboard) · [VLMEvalKit GitHub](https://github.com/open-compass/vlmevalkit) · [VLMEvalKit paper](https://huggingface.co/papers/2407.11691)

### Vision Arena / WildVision-Arena
- **Org:** `WildVision` (+ related `lmarena-ai` VisionArena datasets)
- **What it measures:** Human-preference arena for VLMs — upload an image + prompt, two anonymous VLMs answer, vote; **Elo** rating updated every few hours over 20+ VLMs. Continuous, anonymous, crowd-sourced.
- **Datasets:** VisionArena-Chat (~200K convs), VisionArena-Battle (~30K paired votes), VisionArena-Bench (500-prompt auto proxy of the live ranking).
- **Links:** [Vision Arena Space](https://huggingface.co/spaces/WildVision/vision-arena) · [WildVision GitHub](https://github.com/WildVision-AI/WildVision-Arena) · [WildVision paper (arXiv 2406.11069)](https://arxiv.org/abs/2406.11069) · [VisionArena-Battle dataset](https://huggingface.co/datasets/lmarena-ai/VisionArena-Battle)

---

## 4. Code Generation

### Big Code Models Leaderboard
- **Org:** `bigcode`
- **What it measures:** Open-source **code-generation** models — browse/search/filter open code LLMs with detailed scores and throughput plots. Historically anchored on HumanEval-family pass@k for multiple languages.
- **Link:** [Big Code Models Leaderboard Space](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)

### BigCodeBench Leaderboard
- **What it measures:** The "next generation of HumanEval" — code generation with **diverse library/function calls and complex instructions**, designed because HumanEval's 164 algorithmic problems are too simple to be representative of real-world software work. Hosted on HF Space + GitHub Pages.
- **Links:** [BigCodeBench Leaderboard Space](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) · [HF blog write-up](https://github.com/huggingface/blog/blob/main/leaderboard-bigcodebench.md) · [bigcode org](https://huggingface.co/bigcode)

---

## 5. Speech (ASR & TTS)

### Open ASR Leaderboard
- **Org:** `hf-audio` (Hugging Face)
- **What it measures:** Speech-to-text across many public ASR datasets. Two headline metrics: **WER** (word error rate, quality) and **RTFx** (inverse real-time factor, speed). 2025 refresh added **multilingual** and **long-form** tracks.
- **Trends / leaders:** Conformer-encoder + LLM-decoder models lead English accuracy — e.g. **NVIDIA Canary-Qwen-2.5B (~5.63% WER)**, IBM Granite-Speech-3.3-8B, Microsoft Phi-4-Multimodal; CTC/TDT models like **Parakeet** dominate throughput (RTFx ~2800 vs Whisper-large-v3 ~68).
- **Links:** [Open ASR Leaderboard Space](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) · [2025 trends blog](https://huggingface.co/blog/open-asr-leaderboard) · [GitHub](https://github.com/huggingface/open_asr_leaderboard) · [Paper (arXiv 2510.06961)](https://arxiv.org/html/2510.06961v1)

### TTS Arena V2
- **Org:** `TTS-AGI`
- **What it measures:** Crowd-sourced **text-to-speech** preference (Chatbot-Arena-style for audio). Submit text, listen to two anonymous TTS systems, vote; **Elo** ranking. V2 is a clean restart (V1 votes excluded) covering open + proprietary voices.
- **Links:** [TTS Arena V2 Space](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2) · [Legacy Space](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) · [Launch blog](https://huggingface.co/blog/arena-tts) · [About page](https://tts-agi-tts-arena-v2.hf.space/about)
- **Related:** [Artificial Analysis Speech Arena](https://huggingface.co/spaces/ArtificialAnalysis/Speech-Arena-Leaderboard) · [Arabic TTS Arena](https://huggingface.co/blog/Navid-AI/introducing-arabic-tts-arena)

---

## 6. Agents & Reasoning

### GAIA — General AI Assistants
- **Org:** `gaia-benchmark` (Hugging Face × Meta-AI × AutoGPT contributors)
- **What it measures:** **466 human-verified** real-world questions requiring reasoning, multimodality, tool use, and long-horizon web browsing — conceptually simple for humans, hard for AI. **3 difficulty levels** (L1 → L3). Public leaderboard on 300 held-out test questions; 150+ validation set for dev.
- **Headline gap:** humans **~92%** vs GPT-4-with-plugins **~15%** at release. GAIA is the capstone benchmark of HF's Agents Course.
- **Links:** [GAIA Leaderboard Space](https://huggingface.co/spaces/gaia-benchmark/leaderboard) · [Dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) · [Agents-course "What is GAIA"](https://huggingface.co/learn/agents-course/unit4/what-is-gaia) · [Meta paper page](https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants/)

### Hallucinations Leaderboard
- **Org:** `hallucinations-leaderboard` (led by Univ. of Edinburgh contributors)
- **What it measures:** Factuality + faithfulness across **14 datasets grouped into 7 concepts** (closed/open-book QA, summarization, reading comprehension, instruction-following, fact-checking, self-consistency, hallucination detection). Uses EleutherAI `lm-evaluation-harness` (zero/few-shot). Includes a **SelfCheckGPT** task (generate 6 Wikipedia passages, check self-consistency).
- **Links:** [Hallucinations Leaderboard Space](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard) · [Launch blog](https://huggingface.co/blog/leaderboard-hallucinations) · [Paper (arXiv 2404.05904)](https://arxiv.org/pdf/2404.05904)
- **Related (RAG/summarization hallucination):** [Vectara Hallucination Eval Leaderboard](https://huggingface.co/spaces/vectara/Hallucination-evaluation-leaderboard) · [Vectara GitHub](https://github.com/vectara/hallucination-leaderboard) · [How-to-build-it blog](https://huggingface.co/blog/leaderboard-vectara)

---

## 7. Efficiency / Hardware

### LLM-Perf Leaderboard
- **Org:** `optimum` (Hugging Face)
- **What it measures:** The **performance** axis (not quality): prefill & end-to-end **latency**, **throughput** (tok/s), **memory**, and **energy** (kWh via CodeCarbon, accounting for GPU/CPU/RAM/location) across hardware, backends, and quantization. All runs go through **`optimum-benchmark`** for reproducibility.
- **Links:** [LLM-Perf Leaderboard Space](https://huggingface.co/spaces/optimum/llm-perf-leaderboard) · [optimum-benchmark GitHub](https://github.com/huggingface/optimum-benchmark) · [llm-perf-backend GitHub](https://github.com/huggingface/llm-perf-backend) · [Results dataset](https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard/viewer)

---

## 8. Domain-Specific

### Open Medical-LLM Leaderboard
- **Org:** `openlifescienceai` (Pal, Minervini, Motzfeldt, Gema, Alex; 2024)
- **What it measures:** Medical QA across **MedQA (USMLE)**, **PubMedQA**, **MedMCQA**, and medicine/biology subsets of MMLU — clinical knowledge + reasoning over biomedical literature.
- **Links:** [Open Medical-LLM Leaderboard Space](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) · [Launch blog](https://huggingface.co/blog/leaderboard-medicalllm)

---

## 9. Multilingual / Regional Leaderboards

| Leaderboard | Language | Org | What it measures | Link |
|---|---|---|---|---|
| Open Arabic LLM Leaderboard (OALL v1/v2) | Arabic | `OALL` (2A2I × TII × HF) | v1: 14 Arabic benchmarks (reading comp, sentiment, QA…); v2 raises difficulty | [Space](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard) · [v2 blog](https://huggingface.co/blog/leaderboard-arabic-v2) |
| Open Portuguese LLM Leaderboard | Portuguese | `eduagarcia` | Tracks/ranks open LLMs on PT-native tasks | [Space](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard) |
| Open Ko-LLM Leaderboard | Korean | `upstage` (× NIA) | **Ko-H5**: Korean MMLU/ARC/HellaSwag/TruthfulQA + Korean CommonGen; private test set to block contamination; 1,000+ submissions | [Space](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard) · [Blog](https://huggingface.co/blog/leaderboard-upstage) · [Paper (arXiv 2405.20574)](https://arxiv.org/html/2405.20574v2) |

> These language-specific boards (Korean, Polish, Portuguese, Arabic…) form a broad HF ecosystem of regionally-grounded LLM evaluation, each adapting the leaderboard template to native datasets rather than English translations.

---

## 10. Building & Hosting Your Own (Infra)

### Leaderboards & Evaluations Docs + Template
- **What it is:** Official HF docs + a **4-component template** to spin up your own leaderboard:
  1. **Frontend Space** — displays results, explains the eval, optionally accepts submissions
  2. **Requests dataset** — submission queue + eval status
  3. **Results dataset** — finished scores, pulled by frontend
  4. **Backend Space** (optional) — runs pending evals via **lm-eval-harness** or **lighteval**, can be fully automated / semi-auto / manual
- **Links:** [Leaderboards docs index](https://huggingface.co/docs/leaderboards/index) · [Building a leaderboard (template)](https://huggingface.co/docs/leaderboards/leaderboards/building_page) · [Finding the right leaderboard](https://huggingface.co/docs/leaderboards/en/leaderboards/finding_page) · [leaderboards GitHub](https://github.com/huggingface/leaderboards) · [Accessing benchmark data](https://huggingface.co/docs/hub/en/leaderboard-data-guide)

### The Big Benchmarks Collection
- Curated HF collection aggregating the canonical benchmark leaderboards (LLM, code, embeddings, perf) in one place.
- **Link:** [Big Benchmarks Collection](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection)

---

## 11. HF Community Competitions & Events

Unlike static leaderboards, these were **time-boxed, open-to-the-community challenges** (the most Kaggle-like part of HF).

| Event | Year | What it was | Link |
|---|---|---|---|
| **`huggingface/competitions`** platform | ongoing | HF's own framework for hosting Kaggle-style competitions (private test sets, submission scoring, leaderboard) | [Docs](https://huggingface.co/docs/competitions/index) · [Org](https://huggingface.co/competitions) · [GitHub](https://github.com/huggingface/competitions) |
| **Robust Speech Recognition Challenge** | 2022 | Open-to-community sprint to fine-tune ASR in **70+ languages** (Jan 24 → Feb 7), community-built Whisper/XLS-R models | [Forum thread](https://discuss.huggingface.co/t/open-to-the-community-robust-speech-recognition-challenge/13614) |
| **`ai-competition` (Challenge) org** | — | HF org hosting community AI challenges | [Org](https://huggingface.co/ai-competition) |

> The platform supports fully-private test sets and automated scoring, making it the canonical route for running a true "competition" (vs. a perpetual leaderboard) on the Hub.

---

## Quick-Reference Cheat Sheet

| Leaderboard | Modality | Ranking | Canonical link |
|---|---|---|---|
| Open LLM Leaderboard v2 (archived) | text LLM | static (harness) | `spaces/open-llm-leaderboard/open_llm_leaderboard` |
| LMArena / Chatbot Arena | text LLM | Bradley-Terry | `arena.ai/leaderboard` |
| MTEB / MMTEB | embeddings | static | `spaces/mteb/leaderboard` |
| MTEB Arena | embeddings | preference | `huggingface.co/mteb` |
| Open VLM (OpenCompass) | VLM | static (VLMEvalKit) | `spaces/opencompass/open_vlm_leaderboard` |
| Vision Arena / WildVision | VLM | Elo | `spaces/WildVision/vision-arena` |
| Big Code / BigCodeBench | code | static (pass@k) | `spaces/bigcode/bigcodebench-leaderboard` |
| Open ASR | speech→text | WER / RTFx | `spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena V2 | text→speech | Elo | `spaces/TTS-AGI/TTS-Arena-V2` |
| GAIA | agents | static (test set) | `spaces/gaia-benchmark/leaderboard` |
| Hallucinations | text LLM | static (harness) | `spaces/hallucinations-leaderboard/leaderboard` |
| LLM-Perf | efficiency | latency/energy | `spaces/optimum/llm-perf-leaderboard` |
| Open Medical-LLM | medical QA | static | `spaces/openlifescienceai/open_medical_llm_leaderboard` |
| OALL (Arabic) | Arabic LLM | static | `spaces/OALL/Open-Arabic-LLM-Leaderboard` |
| Open PT-LLM (Portuguese) | PT LLM | static | `spaces/eduagarcia/open_pt_llm_leaderboard` |
| Open Ko-LLM (Korean) | KO LLM | static (Ko-H5) | `spaces/upstage/open-ko-llm-leaderboard` |

---

## Sources

- Open LLM Leaderboard v2: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard · https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-2 · https://huggingface.co/collections/OpenEvals/archived-open-llm-leaderboard-2024-2025
- LMArena / Chatbot Arena: https://huggingface.co/spaces/lmarena-ai/arena-leaderboard · https://arena.ai/leaderboard · https://www.lmsys.org/blog/2023-12-07-leaderboard/ · https://arena.ai/blog/leaderboard-changelog/
- MTEB / MMTEB / Arena: https://huggingface.co/spaces/mteb/leaderboard · https://arxiv.org/abs/2210.07316 · https://arxiv.org/abs/2502.13595 · https://github.com/embeddings-benchmark/mteb/
- Open VLM / VLMEvalKit: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard · https://github.com/open-compass/vlmevalkit · https://huggingface.co/papers/2407.11691
- Vision Arena / WildVision: https://huggingface.co/spaces/WildVision/vision-arena · https://arxiv.org/abs/2406.11069 · https://huggingface.co/datasets/lmarena-ai/VisionArena-Battle
- Big Code / BigCodeBench: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard · https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard · https://github.com/huggingface/blog/blob/main/leaderboard-bigcodebench.md
- Open ASR: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard · https://huggingface.co/blog/open-asr-leaderboard · https://arxiv.org/html/2510.06961v1 · https://github.com/huggingface/open_asr_leaderboard
- TTS Arena V2: https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2 · https://huggingface.co/blog/arena-tts
- GAIA: https://huggingface.co/spaces/gaia-benchmark/leaderboard · https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants/ · https://huggingface.co/learn/agents-course/unit4/what-is-gaia
- Hallucinations: https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard · https://huggingface.co/blog/leaderboard-hallucinations · https://arxiv.org/pdf/2404.05904 · https://huggingface.co/spaces/vectara/Hallucination-evaluation-leaderboard
- LLM-Perf: https://huggingface.co/spaces/optimum/llm-perf-leaderboard · https://github.com/huggingface/optimum-benchmark
- Open Medical-LLM: https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard · https://huggingface.co/blog/leaderboard-medicalllm
- Multilingual: https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard · https://huggingface.co/blog/leaderboard-arabic-v2 · https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard · https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard · https://huggingface.co/blog/leaderboard-upstage
- Infra / docs: https://huggingface.co/docs/leaderboards/index · https://github.com/huggingface/leaderboards · https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection
- Competitions: https://huggingface.co/docs/competitions/index · https://github.com/huggingface/competitions · https://discuss.huggingface.co/t/open-to-the-community-robust-speech-recognition-challenge/13614

---
_Curated via public-source research (Kaggle Discussions, official solution repos, arXiv, blogs). Verify any specific link before relying on it; gold write-ups live in each competition's Discussion tab._
