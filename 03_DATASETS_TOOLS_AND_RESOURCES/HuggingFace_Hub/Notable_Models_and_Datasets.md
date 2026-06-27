# HuggingFace — Notable Models & Datasets Worth Knowing (2024–2026)

A curated, skimmable index of the highest-impact open-weight models and public datasets on the Hugging Face Hub across every modality. Focus: permissively-licensed, widely-downloaded, and trending resources from the 2024–2026 cycle. Download/like counts are point-in-time and drift continuously on the Hub — treat as relative magnitude, not absolute truth. Every entry links to a canonical source (HF Hub page, official repo, paper, or vendor blog).

> Legend: `T` = total params, `A` = active params (MoE), `ctx` = context window.

---

## 1. Text LLMs — Open-Weight Flagships & Workhorses

| Model | Org | Size | License | Why it matters | Link |
|---|---|---|---|---|---|
| **DeepSeek-R1** | DeepSeek | 671B T / 37B A, 128K ctx | MIT | Jan-2025 reasoning model that matched OpenAI o1 at a fraction of cost; RL-trained reasoning. Ships distilled 1.5B/7B/8B/14B/32B/70B variants. | [hf.co](https://huggingface.co/deepseek-ai/DeepSeek-R1) · [paper](https://arxiv.org/pdf/2501.12948) |
| **DeepSeek-V3** | DeepSeek | 671B T / 37B A | MIT | MoE base/chat trained on 14.8T tokens for only ~2.788M H800-hours; backbone for R1. MTP module included. | [hf.co](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| **DeepSeek-V3.2** | DeepSeek | MoE | open weights | Late-cycle refresh of the V3 line on the Hub. | [hf.co](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) |
| **Qwen3-235B-A22B** | Alibaba Qwen | 235B T / 22B A | Apache-2.0 | Apr-2025 flagship; hybrid thinking/non-thinking; ~36T training tokens; leads GPQA-Diamond (77.2) & AIME'24 (85.7) among open weights. | [tech report](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf) · [hf.co org](https://huggingface.co/Qwen) |
| **Qwen2.5-1.5B-Instruct** | Alibaba Qwen | 1.5B | Apache-2.0 | Reportedly the single most-downloaded text LLM on the Hub (~20% of Alibaba's downloads) — small-model deployment workhorse. | [hf.co stats](https://huggingface.co/blog/lbourdois/huggingface-models-stats) |
| **Llama 3.2 3B Instruct** | Meta | 3B | Llama 3.2 license | >2.18M downloads; the most widely deployed small Llama; on-device/edge default. | [models sorted by downloads](https://huggingface.co/models?sort=downloads) |
| **Llama 4 Scout / Maverick** | Meta | 109B/400B T, 17B A | Llama 4 license | Apr-2025; first MoE Llamas. Scout's headline 10M-token context is the longest of any open model. | [year in review](https://www.interconnects.ai/p/2025-open-models-year-in-review) |
| **gpt-oss-120b / 20b** | OpenAI | 120B / 20B (MoE) | Apache-2.0 | Aug-6-2025: OpenAI's first open-weight LLMs since GPT-2. 20B runs near-frontier on a single high-end consumer GPU; strong agentic/tool use. | [hf.co](https://huggingface.co/openai/gpt-oss-20b) · [collection](https://huggingface.co/collections/openai/gpt-oss) · [HF blog](https://huggingface.co/blog/welcome-openai-gpt-oss) |
| **Gemma 3** | Google DeepMind | 1B–27B | Gemma license | Native-multimodal, strong multilingual at <30B, the efficiency champion of 2025; huge fine-tune ecosystem. | [open-models review](https://www.digit.in/features/general/gpt-oss-to-gemma-3-top-5-open-weight-models-you-must-try.html) |
| **Mistral 3 / Ministral 3** | Mistral AI | MoE flagship + 9 edge models | Apache-2.0 | Dec-2025 full-line refresh: frontier MoE (Mistral Large 3) plus compact edge models, all permissive. | [open-source LLM blog](https://huggingface.co/blog/daya-shankar/open-source-llms) |
| **SmolLM3-3B** | Hugging Face | 3B | Apache-2.0 | Strong on-device 3B with long-context and reasoning; reference open-recipe small model. | [year in review](https://www.interconnects.ai/p/2025-open-models-year-in-review) |
| **SmolLM2-1.7B-Instruct** | Hugging Face | 1.7B | Apache-2.0 | Fully open data+recipe small model; the SmolTalk training mix is public. | [hf.co](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) |

**Quick reasoning-benchmark notes (open weights):** DeepSeek-R1 ~97.3 MATH-500; Qwen3-235B leads GPQA-Diamond/AIME'24; both routinely cited as the open frontier for math/reasoning in 2025. See [open-models comparison](https://www.codesota.com/llm/open-models).

---

## 2. Embeddings & Rerankers (RAG stack)

| Model | Org | Sizes | License | Why it matters | Link |
|---|---|---|---|---|---|
| **Qwen3-Embedding-8B** | Alibaba Qwen | 0.6B / 4B / 8B | Apache-2.0 | #1 on MTEB **multilingual** leaderboard (70.58, Jun-5-2025); first local family competitive with commercial embedding APIs across the board. | [hf.co](https://huggingface.co/Qwen/Qwen3-Embedding-8B) · [blog](https://qwenlm.github.io/blog/qwen3-embedding/) · [paper](https://arxiv.org/pdf/2506.05176) |
| **Qwen3-Embedding-0.6B** | Alibaba Qwen | 0.6B | Apache-2.0 | Tiny, fast retrieval encoder; the efficient end of the SOTA family. | [hf.co](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| **Qwen3-Reranker** | Alibaba Qwen | 0.6B–8B | Apache-2.0 | Cross-encoder companion to Qwen3-Embedding; strong across retrieval scenarios. | [blog](https://qwenlm.github.io/blog/qwen3-embedding/) |
| **EmbeddingGemma-300m** | Google | 300M | Gemma license | Multilingual (>100 langs) compact retrieval encoder; popular on-device search/RAG choice. | [analyticsvidhya roundup](https://www.analyticsvidhya.com/blog/2025/11/top-open-source-models-on-huggingface/) |
| **BGE-M3** | BAAI | ~560M | MIT | Multi-functional (dense + sparse + multi-vector), multilingual, long-doc; ~4.7M pulls. The default multilingual RAG encoder. | [morphllm benchmark](https://www.morphllm.com/ollama-embedding-models) |
| **nomic-embed-text** | Nomic AI | 137M | Apache-2.0 | ~73.8M pulls — among the most-pulled open embedders; fully open data + reproducible. | [morphllm benchmark](https://www.morphllm.com/ollama-embedding-models) |
| **mxbai-embed-large** | Mixedbread | 335M | Apache-2.0 | ~11.4M pulls; strong MTEB-English v1 results; popular default English encoder. | [morphllm benchmark](https://www.morphllm.com/ollama-embedding-models) |
| **Qwen3-VL-Embedding / Reranker** | Alibaba Qwen | — | open weights | Unified framework for SOTA **multimodal** retrieval & ranking (text+image). | [paper](https://arxiv.org/pdf/2601.04720) |

> MTEB caveat: leaderboards differ — Qwen3 & EmbeddingGemma report MTEB-multilingual; mxbai/nomic report MTEB-English v1; BGE-M3 has no single comparable score (mixes dense/sparse/multi-vector). Source: [morphllm](https://www.morphllm.com/ollama-embedding-models).

---

## 3. Multimodal / Vision-Language (VLM)

| Model | Org | Sizes | License | Why it matters | Link |
|---|---|---|---|---|---|
| **Qwen2.5-VL** | Alibaba Qwen | 3B / 7B / 72B | Apache-2.0 (most) | Pretrained on 4.1T tokens; window-attention ViT, dynamic FPS, upgraded MRoPE; SOTA OCR, doc parsing, hour-long video, object grounding. | [hf.co 7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · [collection](https://huggingface.co/collections/Qwen/qwen25-vl) · [transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl) |
| **Qwen3-VL** | Alibaba Qwen | multiple | open weights | Next-gen Qwen multimodal series; successor to 2.5-VL. | [GitHub](https://github.com/QwenLM/Qwen3-VL) |
| **Qwen2.5-Omni** | Alibaba Qwen | 3B / 7B | Apache-2.0 | End-to-end any-to-any: text+audio+vision+video in, real-time speech out. Hit #1 HF Trending on 2025-04-02. | [GitHub](https://github.com/QwenLM/Qwen2.5-Omni) |
| **InternVL3** | OpenGVLab | up to 78B | varies | Apr-2025; jointly trained on text + vision-language from scratch (no post-hoc alignment); top-tier open MLLM. | [VLM survey context](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl) |

---

## 4. Vision Foundation / Segmentation / Backbones

| Model | Org | License | Why it matters | Link |
|---|---|---|---|---|
| **SAM 2** | Meta | Apache-2.0 | Promptable visual segmentation across **images and video**; added to Transformers 2025-08-14. | [transformers docs](https://huggingface.co/docs/transformers/en/model_doc/sam2) |
| **SAM 3** | Meta | open weights | Late-2025; promptable **concept** segmentation with text phrases; unified image+video via single Perception Encoder backbone. | [SAM2→SAM3 paper](https://arxiv.org/pdf/2512.06032) |
| **DINOv3** | Meta | open weights | Updated SSL recipe + 1.689B-image pretraining set (LVD-1689M); high-quality dense features; replaces DINOv2 backbone. | [paper page](https://huggingface.co/papers/2508.10104) |

---

## 5. Audio — ASR / TTS

| Model | Org | Size | License | Why it matters | Link |
|---|---|---|---|---|---|
| **Kokoro-82M** | hexgrad | 82M | Apache-2.0 | Open-weight TTS punching far above its size; #1 in TTS Spaces Arena pre-release; <$1 / 1M chars served. ONNX + MLX ports widely used. | [hf.co](https://huggingface.co/hexgrad/Kokoro-82M) · [ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) · [Space](https://huggingface.co/spaces/hexgrad/Kokoro-TTS) |
| **F5-TTS** | community | — | open | Flow-matching TTS with strong zero-shot voice cloning; common Kokoro alternative. | [TTS roundup](https://www.digitalocean.com/community/tutorials/best-text-to-speech-models) |
| **VibeVoice-1.5B** | Microsoft | 1.5B | open weights | Expressive long-form / conversational TTS; trended on the Hub in 2025. | [analyticsvidhya roundup](https://www.analyticsvidhya.com/blog/2025/11/top-open-source-models-on-huggingface/) |
| **Whisper** (large-v3 / turbo) | OpenAI | up to 1.55B | MIT | Still the default open multilingual ASR; backbone of countless Hub ASR pipelines & voice agents. | [speech-to-speech repo](https://github.com/huggingface/speech-to-speech) |
| **Parakeet-TDT** | NVIDIA | — | open | Low-latency (sub-100ms) streaming ASR with auto language detection; popular in real-time voice agents. | [speech-to-speech repo](https://github.com/huggingface/speech-to-speech) |

---

## 6. Code Models

| Model | Org | Sizes | License | Why it matters | Link |
|---|---|---|---|---|---|
| **Qwen2.5-Coder-32B-Instruct** | Alibaba Qwen | 0.5B–32B | Apache-2.0 (most) | 32B variant = SOTA open code LLM at release, matching GPT-4o on coding; six sizes for every footprint. | [hf.co 32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) · [hf.co 7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) · [collection](https://huggingface.co/collections/Qwen/qwen25-coder) |
| **Qwen3-Coder** | Alibaba Qwen | multiple | open weights | Code edition of Qwen3; agentic coding focus. | [GitHub](https://github.com/QwenLM/Qwen3-Coder) · [collection](https://huggingface.co/collections/Qwen/qwen3-coder) |
| **StarCoder2** | BigCode | 3B/7B/15B | BigCode OpenRAIL-M | Fully-open training data (The Stack v2); reference permissive code-LLM family. | [Qwen-coder comparison](https://medium.com/@orami98/i-installed-quantized-and-benchmarked-the-top-open-source-code-llms-8ffc3cfec238) |

---

## 7. Image Generation (Diffusion / DiT)

| Model | Org | License | Why it matters | Link |
|---|---|---|---|---|
| **FLUX.1 [schnell]** | Black Forest Labs | Apache-2.0 | 1–4-step high-quality generation via latent adversarial diffusion distillation; the permissive FLUX entry point. | [hf.co](https://huggingface.co/black-forest-labs/FLUX.1-schnell) · [diffusers docs](https://huggingface.co/docs/diffusers/en/api/pipelines/flux) |
| **FLUX.2** | Black Forest Labs | varies | Nov-2025; native 4-megapixel resolution, improved DiT backbone. | [diffusers FLUX](https://huggingface.co/docs/diffusers/en/api/pipelines/flux) |
| **Qwen-Image** | Alibaba Qwen | Apache-2.0 | Aug-2025; best-in-class **text rendering** (English + Chinese logographic), broad style range; GGUF quants available. | [hf.co](https://huggingface.co/Qwen/Qwen-Image) · [diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage) · [GGUF](https://huggingface.co/unsloth/Qwen-Image-GGUF) |
| **Stable Diffusion 3.5** | Stability AI | Stability Community License | Runs on consumer hardware; massive LoRA/fine-tune ecosystem (ComfyUI, A1111, Forge). | [image-gen roundup](https://www.kdnuggets.com/best-free-image-generators-on-hugging-face-right-now) |

---

## 8. Video Generation

| Model | Org | Size | License | Why it matters | Link |
|---|---|---|---|---|---|
| **Wan2.1-T2V-14B** | Wan-AI (Alibaba) | 14B | Apache-2.0 | New SOTA open T2V at release; first video model to render both Chinese and English text. | [hf.co](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| **Wan2.2-S2V-14B** | Wan-AI | 14B | Apache-2.0 | Speech/audio-to-video addition to the Wan line; widely quantized (GGUF). | [hf.co](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) · [GitHub](https://github.com/Wan-Video/Wan2.2) |
| **HunyuanVideo** | Tencent | 13B | open weights | Open video foundation model rivaling/beating Runway Gen-3 & Luma 1.6 in human eval. | [hf.co](https://huggingface.co/tencent/HunyuanVideo) |
| **HunyuanVideo-I2V** | Tencent | — | open weights | Image-to-video variant of HunyuanVideo. | [hf.co](https://huggingface.co/tencent/HunyuanVideo-I2V) |

---

## 9. Datasets — Pretraining Corpora

| Dataset | Org | Scale | License | Why it matters | Link |
|---|---|---|---|---|---|
| **FineWeb** | HuggingFaceFW | ~15T tokens (gpt2), 96 CC dumps 2013→2024 | ODC-By | Largest public clean web pretraining corpus; processed/dedup'd via `datatrove`. v1.2.0 added May–Dec 2024; v1.3.0 (Jan-2025) +~400B tokens. | [hf.co](https://huggingface.co/datasets/HuggingFaceFW/fineweb) · [collection](https://huggingface.co/collections/HuggingFaceFW/fineweb) |
| **FineWeb-Edu** | HuggingFaceFW | ~1.3T+ tokens, 1.53B rows | ODC-By | Educational-quality filtered subset of FineWeb; the go-to high-signal pretraining mix. v1.4.0 (Jul-2025) added Jan–Jun 2025 snapshots. | [hf.co](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| **FineWeb-2** | HuggingFaceFW | ~20TB, 5B docs, 3T+ words | ODC-By | Multilingual FineWeb across 1000+ languages; the multilingual pretraining baseline. | [hf.co](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| **FineTranslations** | Hugging Face | trillion-token parallel | open | Jan-2026; large multilingual parallel-text corpus for translation/multilingual pretraining. | [InfoQ](https://www.infoq.com/news/2026/01/huggingface-fine-translations/) |
| **MixtureVitae** | community | web-scale | permissive-first | Permissive-sourced web pretraining corpus with embedded instruction + reasoning data. | [arXiv](https://arxiv.org/pdf/2509.25531) |

---

## 10. Datasets — Instruction / Preference (Post-Training)

| Dataset | Org | Scale | License | Why it matters | Link |
|---|---|---|---|---|---|
| **SmolTalk** | HuggingFaceTB | ~1M+ samples | open | SFT mix behind SmolLM2; blends OpenHermes2.5 (100k), Smol-Magpie-Ultra, etc. Boosts MMLU/WinoGrande. | [hf.co](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) |
| **SmolTalk2** | HuggingFaceTB | — | open | Updated SFT mixture for the SmolLM3 generation. | [hf.co](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) |
| **UltraFeedback** | OpenBMB | 64K prompts / 256K responses / 380K annotations | MIT | Canonical preference dataset (DPO/RLHF); prompts from UltraChat, ShareGPT, Evol-Instruct, FLAN, etc., scored by 17 LLMs. | [RLHFlow standard](https://huggingface.co/datasets/RLHFlow/UltraFeedback-preference-standard) |
| **Tulu-3 UltraFeedback (cleaned, on-policy 8B)** | Allen AI | — | open | Decontaminated, on-policy preference data from the Tulu 3 post-training recipe; removes TruthfulQA leakage. | [hf.co](https://huggingface.co/datasets/allenai/tulu-3-ultrafeedback-cleaned-on-policy-8b) |
| **OpenHermes-2.5** | Teknium | 384,900 rows | open | Workhorse instruction component reused across many SFT mixes (incl. SmolTalk, Tulu). | [SmolTalk readme](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/blob/main/README.md) |
| **OpenThoughts** | OpenThoughts | — | open | Reasoning-trace data recipes for distilling reasoning models. | [arXiv](https://arxiv.org/pdf/2506.04178) |

---

## 11. Datasets — Evaluation Benchmarks

| Dataset | Focus | Why it matters | Link |
|---|---|---|---|
| **GPQA (Diamond)** | Graduate STEM QA | ~198 expert-verified hard questions; standard open-weight reasoning gate (Qwen3-235B 77.2). | [Qwen3 report](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf) |
| **MMLU-Pro** | Multitask knowledge | Harder, more robust MMLU successor; default knowledge benchmark in 2025 model cards. | [Qwen3 report](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf) |
| **AIME 2025** | Competition math | 30 problems; the headline math-reasoning eval for reasoning LLMs. | [Qwen3 report](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf) |
| **LiveCodeBench** | Code reasoning | Contamination-resistant coding benchmark for reasoning/coding LLMs. | [benchmark index](https://github.com/VyetGokyra/awaresome_LLM_eval_benchmark) |
| **MTEB** | Embedding/retrieval | The leaderboard that anchors embedding/reranker comparisons (multilingual & English tracks). | [Qwen3-Embedding blog](https://qwenlm.github.io/blog/qwen3-embedding/) |

---

## How to use this index

- **RAG stack:** Qwen3-Embedding (or BGE-M3 for multilingual) + Qwen3-Reranker; corpus from FineWeb-Edu for domain pretraining/continued training.
- **On-device / edge:** SmolLM3-3B or Llama 3.2 3B for text; Kokoro-82M for TTS; Whisper-turbo / Parakeet for ASR; EmbeddingGemma-300m for retrieval.
- **Frontier open reasoning:** DeepSeek-R1 or Qwen3-235B-A22B; gpt-oss-120b for permissive agentic workloads.
- **Multimodal:** Qwen2.5-VL (docs/OCR/video) or InternVL3; Qwen2.5-Omni for any-to-any with speech.
- **Generation:** FLUX.1-schnell / Qwen-Image (text-in-image) for images; Wan2.1/2.2 or HunyuanVideo for video.

> ⚠️ Always re-check the live model card for current license terms and download counts before depending on a resource — both change frequently on the Hub.

---

## Sources

- HF model stats: https://huggingface.co/blog/lbourdois/huggingface-models-stats
- HF models by downloads: https://huggingface.co/models?sort=downloads
- Top open models roundup: https://www.analyticsvidhya.com/blog/2025/11/top-open-source-models-on-huggingface/
- DeepSeek-R1: https://huggingface.co/deepseek-ai/DeepSeek-R1 · paper https://arxiv.org/pdf/2501.12948
- DeepSeek-V3: https://huggingface.co/deepseek-ai/DeepSeek-V3 · V3.2 https://huggingface.co/deepseek-ai/DeepSeek-V3.2
- Qwen3 Technical Report: https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf
- gpt-oss: https://huggingface.co/openai/gpt-oss-20b · https://huggingface.co/collections/openai/gpt-oss · https://huggingface.co/blog/welcome-openai-gpt-oss
- 2025 Open Models Year in Review: https://www.interconnects.ai/p/2025-open-models-year-in-review
- Open-source LLM blog: https://huggingface.co/blog/daya-shankar/open-source-llms · https://www.codesota.com/llm/open-models
- Qwen3-Embedding: https://huggingface.co/Qwen/Qwen3-Embedding-8B · https://huggingface.co/Qwen/Qwen3-Embedding-0.6B · https://qwenlm.github.io/blog/qwen3-embedding/ · https://arxiv.org/pdf/2506.05176
- Embedding benchmarks: https://www.morphllm.com/ollama-embedding-models
- Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct · https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl · https://huggingface.co/collections/Qwen/qwen25-vl
- Qwen2.5-Omni: https://github.com/QwenLM/Qwen2.5-Omni · Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- SAM2: https://huggingface.co/docs/transformers/en/model_doc/sam2 · SAM3 paper https://arxiv.org/pdf/2512.06032 · DINOv3 https://huggingface.co/papers/2508.10104
- Kokoro-82M: https://huggingface.co/hexgrad/Kokoro-82M · ONNX https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX · Space https://huggingface.co/spaces/hexgrad/Kokoro-TTS
- speech-to-speech (Whisper/Parakeet): https://github.com/huggingface/speech-to-speech · TTS roundup https://www.digitalocean.com/community/tutorials/best-text-to-speech-models
- Qwen2.5-Coder: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct · https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct · https://huggingface.co/collections/Qwen/qwen25-coder · Qwen3-Coder https://github.com/QwenLM/Qwen3-Coder
- FLUX.1: https://huggingface.co/black-forest-labs/FLUX.1-schnell · https://huggingface.co/docs/diffusers/en/api/pipelines/flux
- Qwen-Image: https://huggingface.co/Qwen/Qwen-Image · https://huggingface.co/docs/diffusers/api/pipelines/qwenimage · GGUF https://huggingface.co/unsloth/Qwen-Image-GGUF
- Wan: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B · https://huggingface.co/Wan-AI/Wan2.2-S2V-14B · https://github.com/Wan-Video/Wan2.2
- HunyuanVideo: https://huggingface.co/tencent/HunyuanVideo · https://huggingface.co/tencent/HunyuanVideo-I2V
- FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb · https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · collection https://huggingface.co/collections/HuggingFaceFW/fineweb
- FineTranslations: https://www.infoq.com/news/2026/01/huggingface-fine-translations/ · MixtureVitae https://arxiv.org/pdf/2509.25531
- SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk · https://huggingface.co/datasets/HuggingFaceTB/smoltalk2 · SmolLM2 https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- UltraFeedback / Tulu: https://huggingface.co/datasets/RLHFlow/UltraFeedback-preference-standard · https://huggingface.co/datasets/allenai/tulu-3-ultrafeedback-cleaned-on-policy-8b
- OpenThoughts: https://arxiv.org/pdf/2506.04178 · eval benchmark index https://github.com/VyetGokyra/awaresome_LLM_eval_benchmark

---
_Curated via public-source research (Kaggle Discussions, official solution repos, arXiv, blogs). Verify any specific link before relying on it; gold write-ups live in each competition's Discussion tab._
