# 🔬 Research Breakthroughs & Science AI — 2026 Radar

> The research edge of 2026: new post-transformer architectures, diffusion LLMs and SSMs; AI reaching the top of olympiad mathematics and formal proving; and a wave of multimodal and scientific foundation models for biology, weather, medicine and audio.

**Refreshed:** 2026-10-09 · **Scope:** releases through October 2026.

> ⚠️ Fast-moving field: entries reflect public reporting at refresh time. Always confirm versions, licenses, and benchmarks at the source links before relying on them. Benchmark and capability figures below are attributed to their source, not asserted as independently verified fact.

---

## 🧱 New architectures — post-transformer, diffusion LLMs & SSMs

Newest first.

| Item | Date | What's new |
|---|---|---|
| [DiffusionGemma (technical report)](https://arxiv.org/abs/2608.00146) | 2026-07-31 | Google's first open-weight text-diffusion LLM, fine-tuned from the Gemma 4 MoE; per the report it generates text by denoising blocks of tokens in parallel rather than left-to-right autoregression. |
| [DeepSeek-V4 — Compressed Sparse Attention + Heavily Compressed Attention](https://arxiv.org/abs/2606.19348) | 2026-06 | MoE models (V4-Pro reported 1.6T/49B active, plus V4-Flash) with a novel interleaved CSA+HCA sparse-attention scheme; the paper reports native 1M-token context at ~27% FLOPs and ~10% KV cache of V3.2. |
| [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569) | 2026-03-16 | Latest SSM/post-transformer architecture (Dao, Gu, Kolter et al.): exponential-trapezoidal discretization, complex-valued state spaces for state-tracking and a MIMO formulation; the paper reports it beats Gated DeltaNet at 1.5B scale (ICLR 2026). |
| [Qwen3.5 — Gated DeltaNet linear-attention + sparse MoE hybrid](https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894) | 2026-02-17 | Open-weight native-multimodal family (flagship 397B/17B active) hybridizing Gated DeltaNet linear attention (~3:1) with full attention and high-sparsity MoE; Alibaba reports 8.6x/19x decode throughput vs Qwen3-Max at 32k/256k context. |
| [Nested Learning + the "Hope" architecture](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) | 2025-11-07 | Google Research paradigm treating a model as nested optimization problems to attack catastrophic forgetting; a proof-of-concept "Hope" self-modifying architecture with a Continuum Memory System (NeurIPS 2025). |
| [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | 2025-10-30 | Moonshot AI hybrid linear-attention model using Kimi Delta Attention (a per-channel-decay refinement of Gated DeltaNet) with gated MLA; the paper reports ~75% KV-cache reduction and up to 6x decode throughput while beating full attention in fair comparisons. |
| [Every Attention Matters: Ring-linear 2.0 hybrid architecture](https://arxiv.org/abs/2510.19338) | 2025-10-22 | Ling Team / inclusionAI "Ring-mini/flash-linear-2.0" models interleaving linear and softmax attention for long-context reasoning and RL; reported ~1/10 inference cost of a 32B dense model, plus an FP8 operator library (linghe). |
| [Mercury: Ultra-Fast Diffusion LLM](https://arxiv.org/abs/2506.17298) | 2025-06 | Inception Labs' first commercial diffusion LLM line, generating text via parallel denoising for order-of-magnitude higher throughput than autoregressive models. (Secondary reports describe a Mercury 2.5 in Sept 2026; the primary spec sheet was not confirmable.) |

---

## 🧠 Interpretability & the science of models

Newest first.

| Item | Date | What's new |
|---|---|---|
| [Characterizing interference weights in a tiny language model](https://transformer-circuits.pub/) | 2026-08 | Anthropic / Transformer Circuits mechanistic-interpretability study identifying "interference weights" in a 1-layer transformer by measuring their causal effect on outputs and loss; part of the 2026 Circuits thread. |
| [Verbalizable Representations Form a Global Workspace in Language Models](https://arxiv.org/abs/2607.15495) | 2026-07 | Anthropic result (Gurnee, Lindsey et al.) introducing the "Jacobian lens" / "J-space": a small privileged set of representations the model can report, hold and reason with — a functional analogue of a conscious global workspace that, per the paper, surfaces hidden strategic deliberation in alignment audits. |

---

## ➗ AI for mathematics, theorem proving & algorithm discovery

Newest first.

| Item | Date | What's new |
|---|---|---|
| [An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics (NVIDIA)](https://arxiv.org/abs/2609.10712) | 2026-09-09 | Fully open natural-language, test-time pipeline on Nemotron 3 Ultra that, per the paper, reaches the IMO 2026 gold threshold (30/42); releases checkpoints, data, code and a 200-problem olympiad benchmark — described as the first open recipe for IMO gold. |
| [dots-note-3.0 — reported first AI perfect 42/42 at IMO 2026](https://studio-dots-ai.github.io/dots_imo_2026/en.html) | 2026-07 | Xiaohongshu/RedNote "dots" team; reported as an officially IMO-committee-marked perfect score (6/6 problems, 42/42) at IMO 2026 Shanghai using a Proof-Verify-Refine workflow that reads raw LaTeX and mixes natural-language reasoning with Python — no formal translation. |
| [Mistral Leanstral 1.5](https://mistral.ai/news/leanstral-1-5) | 2026-07-02 | Mistral's formal-math/proof model (119B total / 6B active, Apache-2.0) for practical proof engineering in Lean 4 and verifying real code; a strong open addition to the theorem-proving cluster. |
| [Pythagoras-Prover: Efficient Formal Proving via Augmented Lean Formalisation](https://arxiv.org/abs/2606.12594) | 2026-06-10 | Open family of compute-efficient Lean 4 theorem provers with both autoregressive and diffusion variants; the paper reports a 4B model surpassing DeepSeek-Prover-V2-671B on MiniF2F at ~167x fewer parameters. |
| [Seed-Prover 1.5 (ByteDance Seed)](https://arxiv.org/abs/2512.17260) | 2025-12-19 | Formal theorem-proving model trained with large-scale agentic RL plus a test-time workflow linking natural-language and Lean proofs; the paper reports solving 88% of PutnamBench. |
| [IMO-Bench / "Towards Robust Mathematical Reasoning" (Google DeepMind)](https://arxiv.org/abs/2511.01846) | 2025-11-03 | Suite of olympiad-level benchmarks (IMO-AnswerBench, ProofBench, GradingBench, LeanProofBench) for robust short-answer and proof-writing evaluation; a standard yardstick for AI math reasoning. |
| [Aristotle: IMO-level Automated Theorem Proving (Harmonic)](https://arxiv.org/abs/2510.01346) | 2025-10-01 | Formal ATP that pairs informal reasoning with Lean verification; reported to reach gold-medal-equivalent performance on IMO 2025 problems — a reference point for formally verified olympiad proving. |
| [AlphaEvolve (Google DeepMind)](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | 2025-05-14 | Gemini-powered evolutionary coding agent that discovers novel algorithms — e.g. a 4x4 complex matrix multiply in 48 scalar multiplications (beating Strassen) and new kissing-number bounds. (Widely reported 2026 impact updates were not confirmed on a primary source.) |
| [DeepSeek-Prover-V2 (Lean 4 formal theorem proving)](https://arxiv.org/abs/2504.21801) | 2025-04 | Reference open-source Lean 4 prover using a recursive subgoal pipeline driven by DeepSeek-V3; a strong MiniF2F/PutnamBench baseline that the newer 2026 provers measure against. |

---

## 🧬 AI for science — proteins, genomics & biology

Newest first.

| Item | Date | What's new |
|---|---|---|
| [TorchCraft: Unified binder design by inverting an all-atom structure predictor](https://arxiv.org/abs/2609.19770) | 2026-09-17 | One framework that designs protein binders (minibinders, VHHs, cyclic peptides, ligand-binding proteins) by optimizing sequence logits through frozen AlphaFold 3 weights, with experimental validation and no post-hoc sequence redesign. |
| [AlphaGenome Atlas + AVI score (Google DeepMind)](https://www.marktechpost.com/2026/09/08/google-deepmind-releases-alphagenome-atlas-with-precomputed-molecular-effect-predictions-and-avi-scores-for-9-billion-human-dna-variants/) | 2026-09-08 | Precomputed AlphaGenome predictions (~27,000 per variant) for every possible single-nucleotide change in the human genome (~9B variants, ~1PB), plus a new AlphaGenome Variant Impact (AVI) score combining AlphaGenome + AlphaMissense; free non-commercial via web/API. (Reported across multiple outlets; DeepMind's own page was not bot-reachable.) |
| [ESM world model — ESMC + ESMFold2 + ESM Atlas (Arc / CZ Biohub)](https://biohub.org/news/world-model-of-protein-biology/) | 2026-05-27 | Fully open (commercial-use) "world model of protein biology": the ESMC language model, ESMFold2 structure prediction (SOTA claims on protein-protein and antibody-antigen), and ESM Atlas mapping 6.8B proteins / 1.1B predicted structures; reported to design functional binders at therapeutic affinity. |
| [Boltz-2 affinity fine-tuning framework (Biogen)](https://www.biorxiv.org/content/10.64898/2026.05.26.727958v1) | 2026-05-26 | Open framework for adapting the open co-folding+affinity model Boltz-2 to a project's own assay data for protein-ligand potency prediction, in some cases approaching free-energy-perturbation (FEP) accuracy; code public. (Independent evals contest its ranking precision.) |
| [Evo 2 — genome foundation model across all domains of life (Arc Institute + NVIDIA, Nature)](https://www.nature.com/articles/s41586-026-10176-5) | 2026-03-04 | Largest fully-open DNA/genome foundation model (7B/40B params, ~9.3T base pairs across 128k+ species, up to 1M-token single-nucleotide context, StripedHyena 2); zero-shot variant-effect prediction (incl. BRCA1) and genome-scale sequence design. Peer-reviewed in Nature. |
| [Proteina-Complexa (NVIDIA) protein binder design](https://developer.nvidia.com/blog/designing-protein-binders-using-the-generative-model-proteina-complexa/) | 2026 | Generative model that co-designs atomistic structure + sequence of protein binders with inference-time compute; per NVIDIA it produces binders against most targets including nano- and picomolar affinities, with source code, checkpoints and datasets released. (Exact 2026 release date not pinned down.) |
| [AlphaGenome (Google DeepMind)](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) | 2025-06-25 | Regulatory-genomics model taking up to 1 Mb of DNA at single-base resolution, jointly predicting many molecular modalities and scoring a variant's effects in ~1s; first to model RNA splice junctions directly from sequence (Nature pub. Jan 2026). |
| [Boltz-2 (MIT + Recursion)](https://jeremywohlwend.com/assets/boltz2.pdf) | 2025-06 | The first open co-folding model to jointly predict biomolecular structure and binding affinity, per the report approaching FEP physics accuracy up to ~1000x faster; extends Boltz-1 with a dual affinity head and pocket/contact constraints. |

---

## 🌦️ Weather, Earth-system & medical foundation models

Newest first.

| Item | Date | What's new |
|---|---|---|
| [WeatherNext 3 (Google DeepMind)](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/introducing-weathernext-3/) | 2026-09-03 | Newest flagship weather AI: per the blog it learns from near-real-time geostationary satellite data for hourly refreshes at ~5 km (temp/moisture), ~5x sharper than WeatherNext 2, adding precipitation and clean-energy variables (100 m wind, solar); now powers Search, Gemini and Maps. |
| [WeatherNext Cyclones (open weights + Nature paper)](https://deepmind.google/blog/weathernext-ai-model-achieves-breakthrough-in-forecasting-cyclones/) | 2026-08-06 | Open-sourced (code Apache-2.0, weights CC BY 4.0) cyclone model that jointly predicts track, intensity and size; per the Nature paper "Operational Tropical Cyclone Forecasting with AI" it gains ~24h+ lead time vs operational systems. The google-deepmind/weathernext repo also ships WeatherNext 2 and a Colab-runnable 2-mini. |
| [MedGemma 1.5 (4B) + MedASR (Google)](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/) | 2026-01-13 | Open medical multimodal model adding high-dimensional imaging (CT/MRI/whole-slide histopathology), longitudinal chest X-ray analysis, anatomical localization and lab-report extraction; released with MedASR, a medical-dictation speech-to-text model. Free for research and commercial use. |
| [Microsoft Aurora (open-sourced Earth-system foundation model)](https://github.com/microsoft/aurora) | 2026-01 | A 1.3B-parameter foundation model of the Earth system, open on GitHub/Hugging Face, with fine-tuned checkpoints beyond weather: air quality/atmospheric chemistry (5-day global at 0.4°, CAMS-level) and ocean waves, running ~thousands of times faster than IFS (Nature, 2025). |

---

## 🧩 Multimodal & omni foundation models

Newest first.

| Item | Date | What's new |
|---|---|---|
| [Qwen3.8-Omni-Flash (Alibaba Qwen)](https://arxiv.org/abs/2609.25611) | 2026-09-22 | Natively omni-modal *agentic* model (sparse MoE from Qwen3.8-Next) extended to a 1M-token context for long-horizon multimodal agent tasks (video editing, long-form A/V translation, music-conditioned video); ships open Qwen-MM-Plugins and Qwen-Live-Harness frameworks. |
| [GLM-5.3-Flash (Z.ai / Zhipu AI)](https://emergent.sh/news/glm-5-3-flash-officially-launched) | 2026-08-26 | Speed-optimized multimodal variant of the GLM-5.3 family (alongside the Aug GLM-5.3 flagship), combining text+vision under an MIT license; some sources report it as the first natively text/image/video model of the GLM-5 series (~320B/18B active) — those specifics vary by source. |
| [Gemini Omni (Google)](https://www.shacknews.com/article/149206/google-gemini-omni-multimodal-gen-ai) | 2026-05-19 | Google's native multimodal "create anything from anything" family unveiled at I/O 2026, starting with Gemini Omni Flash for video; fuses Gemini with Veo/Nano Banana/Genie generative-media models, with SynthID + C2PA watermarking. (Primary blog.google post not surfaced; scope beyond video reported inconsistently.) |
| [NVIDIA Nemotron 3 Nano Omni](https://blogs.nvidia.com/blog/nemotron) | 2026-04-29 | Open (weights + data + recipe, commercial-use) hybrid Mamba-Transformer MoE omni model, 30B total / 3B active, taking text/image/audio/video/documents/charts/UIs as input; NVIDIA claims leads on 6 doc-intelligence and A/V leaderboards and ~9x throughput vs Qwen3-Omni. |
| [Qwen3.5-Omni (Alibaba Qwen)](https://www.marktechpost.com/2026/03/30/alibaba-qwen-team-releases-qwen3-5-omni-a-native-multimodal-model-for-text-audio-video-and-realtime-interaction/) | 2026-03-30 | Native Thinker-Talker multimodal model for unified text/audio/video understanding and real-time interaction; the spring-2026 step between 2025's Qwen3-Omni and Sept-2026's Qwen3.8-Omni. Open weights. |
| [HunyuanImage-3.0-Instruct (Tencent)](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) | 2026-01-26 | Instruct/editing upgrade to Tencent's open native-multimodal 80B-MoE (13B active) autoregressive image model: adds reasoning-based prompt handling and image-to-image editing (add/remove, style transfer, multi-image blend), plus a distilled fast checkpoint. |

---

## 🔊 Audio, speech & music foundation models

Newest first.

| Item | Date | What's new |
|---|---|---|
| [Suno v6 / v6-wild / v6-mini](https://musicbusinessworldwide.com/suno-v6-ai-music-models-launch-in-partnership-with-wmg-bmg-and-believe) | 2026-09-09 | Suno's first fully licensed-music models (trained with Warner Music Group, BMG and Believe), retiring all prior models; adds plain-language section/lyric editing, multi-song mashups and text/audio/image/video inputs. v6-mini free; v6/v6-wild gated to paid tiers. (Primary suno.com post not surfaced; corroborated by MBW.) |
| [VibeVoice-ASR-Streaming (Microsoft)](https://github.com/microsoft/VibeVoice) | 2026-09-03 | Open unified *streaming* ASR model (VibeVoice-ASR-Streaming-7B) that continuously transcribes speaker-attributed "who said what" as audio arrives, with custom hotwords and 10 languages; builds on the Jan-2026 VibeVoice-ASR. |
| [Lyria 3.5 (Google DeepMind)](https://deepmind.google/models/lyria/) | 2026-07-29 | DeepMind's newest music-generation model (listed as "our newest" on the official model page): richer melody/structure, better lyrics and prompt adherence, variable song length up to ~3 min, 44.1 kHz stereo, SynthID; reached the Gemini app + API/AI Studio on Sep 4. |
| [GPT-Live (GPT-Live-1 / GPT-Live-1 mini) (OpenAI)](https://openai.com/index/introducing-gpt-live/) | 2026-07-08 | New generation of ChatGPT voice models that listen and speak simultaneously (full-duplex) for more natural real-time conversation, rolling out in ChatGPT and the API; audio carries SynthID watermarking (added Jul 31). (OpenAI page 403s to automated fetch; corroborated by CNBC.) |
| [GPT-Realtime-2 (+ Realtime-Translate, Realtime-Whisper) (OpenAI)](https://techcrunch.com/2026/05/07/openai-launches-new-voice-intelligence-features-in-its-api/) | 2026-05-07 | Native speech-to-speech API model with GPT-5-class reasoning, larger context and adjustable reasoning effort; accepts text/audio/image, outputs text/audio. Shipped with GPT-Realtime-Translate and a GPT-Realtime-Whisper streaming STT. |
| [Mistral Voxtral TTS](https://techcrunch.com/2026/03/26/mistral-releases-a-new-open-source-model-for-speech-generation/) | 2026-03-26 | Mistral's first open-source text-to-speech model, extending the Voxtral family to speech generation for voice assistants/support; supports 9 languages (EN/FR/DE/ES/NL/PT/IT/HI/AR). |
| [Omnilingual ASR (Meta AI)](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/) | 2025-11 | Open suite of automatic speech-recognition models covering 1,600+ languages (including many low-resource languages), with an extensible design to add new languages from few examples; a breadth benchmark leader for multilingual ASR (released late 2025, still current). |

---

## Related

- Radar index: [Frontier AI 2026 — Innovation Radar README](./README.md) · sibling pages [Open-Source LLMs](./Open_Source_LLMs_2026.md), [Reasoning & Agents](./Reasoning_and_Agents_2026.md), [Robotics & World Models](./Robotics_and_World_Models_2026.md)
- Stable pillars: [`01_AI_FUNDAMENTALS_AND_THEORY/State_Space_Models`](../01_AI_FUNDAMENTALS_AND_THEORY/State_Space_Models/) · [`02_LLM_AND_AI_MODELS/Scientific_Models`](../02_LLM_AND_AI_MODELS/Scientific_Models/) · [`02_LLM_AND_AI_MODELS/Multimodal_Models`](../02_LLM_AND_AI_MODELS/Multimodal_Models/) · [`02_LLM_AND_AI_MODELS/Audio_Models`](../02_LLM_AND_AI_MODELS/Audio_Models/) · [`05_VERTICAL_APPLICATIONS/15_Science_AI`](../05_VERTICAL_APPLICATIONS/15_Science_AI/)

**Sources:** [DiffusionGemma](https://arxiv.org/abs/2608.00146) · [DeepSeek-V4 (CSA+HCA)](https://arxiv.org/abs/2606.19348) · [Mamba-3](https://arxiv.org/abs/2603.15569) · [Qwen3.5](https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894) · [Nested Learning / Hope](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) · [Kimi Linear](https://arxiv.org/abs/2510.26692) · [Ring-linear 2.0](https://arxiv.org/abs/2510.19338) · [Mercury](https://arxiv.org/abs/2506.17298) · [Interference weights (Transformer Circuits)](https://transformer-circuits.pub/) · [Global Workspace in LMs](https://arxiv.org/abs/2607.15495) · [Open Recipe for IMO Gold (Nemotron)](https://arxiv.org/abs/2609.10712) · [dots-note-3.0 IMO 2026](https://studio-dots-ai.github.io/dots_imo_2026/en.html) · [Mistral Leanstral 1.5](https://mistral.ai/news/leanstral-1-5) · [Pythagoras-Prover](https://arxiv.org/abs/2606.12594) · [Seed-Prover 1.5](https://arxiv.org/abs/2512.17260) · [IMO-Bench](https://arxiv.org/abs/2511.01846) · [Aristotle (Harmonic)](https://arxiv.org/abs/2510.01346) · [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) · [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) · [TorchCraft](https://arxiv.org/abs/2609.19770) · [AlphaGenome Atlas + AVI](https://www.marktechpost.com/2026/09/08/google-deepmind-releases-alphagenome-atlas-with-precomputed-molecular-effect-predictions-and-avi-scores-for-9-billion-human-dna-variants/) · [ESM world model](https://biohub.org/news/world-model-of-protein-biology/) · [Boltz-2 affinity fine-tuning](https://www.biorxiv.org/content/10.64898/2026.05.26.727958v1) · [Evo 2 (Nature)](https://www.nature.com/articles/s41586-026-10176-5) · [Proteina-Complexa](https://developer.nvidia.com/blog/designing-protein-binders-using-the-generative-model-proteina-complexa/) · [AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) · [Boltz-2](https://jeremywohlwend.com/assets/boltz2.pdf) · [WeatherNext 3](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/introducing-weathernext-3/) · [WeatherNext Cyclones](https://deepmind.google/blog/weathernext-ai-model-achieves-breakthrough-in-forecasting-cyclones/) · [MedGemma 1.5 + MedASR](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/) · [Microsoft Aurora](https://github.com/microsoft/aurora) · [Qwen3.8-Omni-Flash](https://arxiv.org/abs/2609.25611) · [GLM-5.3-Flash](https://emergent.sh/news/glm-5-3-flash-officially-launched) · [Gemini Omni](https://www.shacknews.com/article/149206/google-gemini-omni-multimodal-gen-ai) · [Nemotron 3 Nano Omni](https://blogs.nvidia.com/blog/nemotron) · [Qwen3.5-Omni](https://www.marktechpost.com/2026/03/30/alibaba-qwen-team-releases-qwen3-5-omni-a-native-multimodal-model-for-text-audio-video-and-realtime-interaction/) · [HunyuanImage-3.0-Instruct](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) · [Suno v6](https://musicbusinessworldwide.com/suno-v6-ai-music-models-launch-in-partnership-with-wmg-bmg-and-believe) · [VibeVoice-ASR-Streaming](https://github.com/microsoft/VibeVoice) · [Lyria 3.5](https://deepmind.google/models/lyria/) · [GPT-Live](https://openai.com/index/introducing-gpt-live/) · [GPT-Realtime-2](https://techcrunch.com/2026/05/07/openai-launches-new-voice-intelligence-features-in-its-api/) · [Mistral Voxtral TTS](https://techcrunch.com/2026/03/26/mistral-releases-a-new-open-source-model-for-speech-generation/) · [Omnilingual ASR](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/)

**Keywords:** AI research breakthroughs 2026, new architectures, post-transformer, diffusion LLM, DiffusionGemma, Mercury, state space models, Mamba-3, linear attention, Kimi Linear, Ring-linear 2.0, Gated DeltaNet, Qwen3.5, sparse attention, DeepSeek-V4, nested learning, Hope architecture, mechanistic interpretability, global workspace, AI for math, IMO 2026 gold, Nemotron olympiad math, dots-note-3.0, automated theorem proving, Lean 4, Leanstral, Pythagoras-Prover, Seed-Prover, Aristotle, DeepSeek-Prover-V2, AlphaEvolve, AI for science, protein design, protein binders, TorchCraft, Proteina-Complexa, Boltz-2, ESM, Evo 2, genome foundation model, AlphaGenome Atlas, AVI score, weather AI, WeatherNext 3, cyclone forecasting, Microsoft Aurora, MedGemma, medical AI, multimodal foundation models, omni models, Qwen3.8-Omni, GLM-5.3-Flash, Gemini Omni, Nemotron Nano Omni, audio AI, speech recognition, Suno v6, Lyria 3.5, GPT-Live, VibeVoice, Omnilingual ASR, Voxtral TTS, scientific foundation models, October 2026.
