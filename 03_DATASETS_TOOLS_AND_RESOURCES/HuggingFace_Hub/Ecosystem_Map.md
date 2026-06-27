# HuggingFace Hub — The Ecosystem Map

A practical, link-dense index of the HuggingFace (HF) ecosystem: the Hub (models / datasets / spaces), the core open-source libraries, the most-downloaded model families, dataset highlights, Spaces, the inference stack (Inference Providers / Endpoints / TGI), AutoTrain, and Hub file/format features (model cards, safetensors, GGUF). Every entry links a canonical source. `snake_case`-friendly; skim the tables.

> Scope note: HF is three things at once — (1) a **git-backed Hub** of repos, (2) a stack of **Python/Rust libraries**, and (3) a set of **hosted services** (Spaces, Inference, AutoTrain). The map below is organized along those axes.

---

## 1. The Hub — repos, the three repo types

The Hub is a platform of git+LFS repositories. Every repo is `models`, `datasets`, or `spaces`, namespaced as `org_or_user/repo_name`.

| Surface | What it is | Entry point |
|---|---|---|
| Hub docs | Canonical reference for all Hub behavior (repos, cards, formats, security) | https://huggingface.co/docs/hub/index |
| Models | Browse/sort/filter all model repos | https://huggingface.co/models?sort=downloads |
| Datasets | Browse/sort/filter all dataset repos | https://huggingface.co/datasets?sort=downloads |
| Spaces | Browse hosted demo apps | https://huggingface.co/spaces |
| `huggingface_hub` (client) | Python lib + CLI to create/upload/download repos, run inference | https://huggingface.co/docs/huggingface_hub/index |
| `hf` CLI / `hf_transfer` | Fast download/upload, `hf auth login`, `hf download` | https://huggingface.co/docs/huggingface_hub/guides/cli |
| Repo structure & git/LFS | How repos, branches, and large files work | https://huggingface.co/docs/hub/repositories |
| Xet storage backend | Chunk-dedup storage replacing pure LFS for large repos | https://huggingface.co/docs/hub/storage-backends |

**Practical commands**
- `pip install -U huggingface_hub` → `hf auth login` → `hf download <repo>` / `hf upload <repo> .`
- `from huggingface_hub import snapshot_download, hf_hub_download` for programmatic pulls.

---

## 2. Core libraries — the open-source stack

These are the packages most developers actually `pip install`. All are MIT/Apache-2.0 and live under `github.com/huggingface`.

| Library | One-liner | Docs | GitHub |
|---|---|---|---|
| `transformers` | Pretrained models for text/vision/audio/multimodal; `pipeline()` + `AutoModel`/`AutoTokenizer` | https://huggingface.co/docs/transformers/index | https://github.com/huggingface/transformers |
| `diffusers` | State-of-the-art diffusion pipelines (image/video/audio gen) | https://huggingface.co/docs/diffusers/index | https://github.com/huggingface/diffusers |
| `datasets` | Memory-mapped (Arrow) loading/streaming/processing of datasets | https://huggingface.co/docs/datasets/index | https://github.com/huggingface/datasets |
| `tokenizers` | Fast Rust BPE/WordPiece/Unigram tokenizers | https://huggingface.co/docs/tokenizers/index | https://github.com/huggingface/tokenizers |
| `accelerate` | Launch/train PyTorch on any device + FSDP/DeepSpeed/fp8 with one config | https://huggingface.co/docs/accelerate/index | https://github.com/huggingface/accelerate |
| `peft` | Parameter-efficient fine-tuning (LoRA, QLoRA, IA3, prompt tuning) | https://huggingface.co/docs/peft/index | https://github.com/huggingface/peft |
| `trl` | Post-training: SFT, DPO, GRPO, PPO, reward modeling | https://huggingface.co/docs/trl/index | https://github.com/huggingface/trl |
| `evaluate` | Standardized metrics/comparisons/measurements | https://huggingface.co/docs/evaluate/index | https://github.com/huggingface/evaluate |
| `optimum` | Hardware acceleration (ONNX Runtime, OpenVINO, TensorRT, Habana, Neuron) for Transformers/Diffusers/TIMM | https://huggingface.co/docs/optimum/index | https://github.com/huggingface/optimum |
| `text-generation-inference` (TGI) | Production-grade LLM serving (continuous batching, tensor parallel, quant) | https://huggingface.co/docs/text-generation-inference/index | https://github.com/huggingface/text-generation-inference |
| `smolagents` | Barebones agents that "think in code"; sandboxed execution (E2B/Docker/Modal) | https://huggingface.co/docs/smolagents/index | https://github.com/huggingface/smolagents |
| `lerobot` | End-to-end robotics in PyTorch: imitation/RL/VLA policies + datasets | https://huggingface.co/docs/lerobot/index | https://github.com/huggingface/lerobot |
| `gradio` | Build ML web UIs/demos in pure Python; powers most Spaces | https://www.gradio.app/docs | https://github.com/gradio-app/gradio |

**Library notes**
- `transformers` `pipeline("task")` is the fastest path to inference; `AutoClass` covers custom loops.
- `trl` integrates `peft` + `accelerate` so you can SFT/DPO a quantized model on a single GPU and scale to multi-node unchanged. PEFT+TRL LoRA guide: https://huggingface.co/docs/trl/lora_tuning_peft
- `optimum` is the bridge to non-CUDA hardware; sub-packages: `optimum-onnxruntime`, `optimum-intel` (OpenVINO), `optimum-neuron` (AWS Trainium/Inferentia), `optimum-habana` (Gaudi).
- `lerobot` **v0.5.0** (200+ merged PRs, first humanoid, autoregressive VLAs, Python 3.12) release note: https://github.com/huggingface/blog/blob/main/lerobot-release-v050.md ; SmolVLA model+docs: https://huggingface.co/docs/lerobot/smolvla
- `smolagents` `CodeAgent` writes actions as Python code rather than JSON tool calls — a distinct design vs. typical tool-calling agents.

---

## 3. Most-downloaded / important model families

The top ~50 entities account for ~80% of all Hub downloads (data 2025-10-01). Sources: https://huggingface.co/blog/lbourdois/huggingface-models-stats and live https://huggingface.co/models?sort=downloads .

### 3.1 Foundational encoders / embeddings (perennially top-downloaded)

| Family | Why it matters | Canonical page |
|---|---|---|
| BERT (`bert-base-uncased`) | The reference encoder; still massive in production pipelines | https://huggingface.co/google-bert/bert-base-uncased |
| RoBERTa (`roberta-base`) | Robustly-tuned BERT; classification/NLU baseline | https://huggingface.co/FacebookAI/roberta-base |
| DistilBERT | 40% smaller BERT, ~97% performance | https://huggingface.co/distilbert/distilbert-base-uncased |
| Sentence-Transformers (`all-MiniLM-L6-v2`) | De-facto sentence-embedding model for RAG/search | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| CLIP (`openai/clip-vit-...`) | Most-downloaded vision model; image-text embedding | https://huggingface.co/openai/clip-vit-base-patch32 |
| Whisper (`openai/whisper-large-v3`) | Standard open ASR model | https://huggingface.co/openai/whisper-large-v3 |
| BGE / GTE embeddings | Strong open retrieval embeddings (BAAI / Alibaba) | https://huggingface.co/BAAI/bge-m3 |

### 3.2 Open LLM families (instruction / chat)

| Family | Org | Hub org page |
|---|---|---|
| Qwen / Qwen2.5 / Qwen3 | Alibaba | https://huggingface.co/Qwen |
| Llama (3.x / 4) | Meta | https://huggingface.co/meta-llama |
| Mistral / Mixtral | Mistral AI | https://huggingface.co/mistralai |
| Gemma (2 / 3) | Google | https://huggingface.co/google |
| Phi | Microsoft | https://huggingface.co/microsoft |
| DeepSeek (V3 / R1) | DeepSeek | https://huggingface.co/deepseek-ai |
| SmolLM / SmolVLM | HuggingFaceTB (HF) | https://huggingface.co/HuggingFaceTB |
| Falcon | TII | https://huggingface.co/tiiuae |

> Note on rankings: `Qwen2.5-1.5B-Instruct` is among the single most-downloaded LLMs, and small instruct models dominate raw download counts because they are pulled by countless downstream pipelines (per the stats blog above). For *current* numbers always trust the live sorted listing rather than any static figure.

### 3.3 Text-to-image / video (diffusers ecosystem)

| Family | Org | Hub page |
|---|---|---|
| Stable Diffusion / SDXL | Stability AI | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 |
| FLUX.1 | Black Forest Labs | https://huggingface.co/black-forest-labs/FLUX.1-dev |
| Stable Diffusion 3.5 | Stability AI | https://huggingface.co/stabilityai/stable-diffusion-3.5-large |

Browse diffusers-compatible models by downloads: https://huggingface.co/models?library=diffusers&sort=downloads

---

## 4. Datasets Hub — highlights

The Datasets Hub serves training/eval corpora with a unified loader, streaming, and a built-in viewer. Browse: https://huggingface.co/datasets?sort=downloads

| Dataset | Use | Page |
|---|---|---|
| FineWeb / FineWeb-Edu | Large high-quality web pretraining corpus (HF) | https://huggingface.co/datasets/HuggingFaceFW/fineweb |
| The Stack v2 | Permissively-licensed source code (BigCode) | https://huggingface.co/datasets/bigcode/the-stack-v2 |
| Common Voice | Multilingual crowdsourced speech (Mozilla) | https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 |
| SQuAD | Reading-comprehension QA benchmark | https://huggingface.co/datasets/rajpurkar/squad |
| GLUE | NLU benchmark suite | https://huggingface.co/datasets/nyu-mll/glue |
| IMDB | Sentiment-classification baseline | https://huggingface.co/datasets/stanfordnlp/imdb |
| Cosmopedia | Synthetic textbook-style pretraining data (HF) | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia |
| LeRobot datasets | Real-robot trajectories in the `lerobot` format | https://huggingface.co/datasets?other=lerobot |

**Dataset features**
- **Dataset Viewer** (auto preview, full-text/SQL filter): https://huggingface.co/docs/dataset-viewer/index
- **Streaming** (`load_dataset(..., streaming=True)`): iterate TB-scale data without downloading.
- **Dataset cards** (`README.md` + YAML): https://huggingface.co/docs/hub/datasets-cards

---

## 5. Spaces — hosted demos & apps

Spaces are git repos that build and host an app (Gradio, Streamlit, Docker, or static). Docs: https://huggingface.co/docs/hub/spaces

| Topic | Link |
|---|---|
| Spaces overview / SDKs | https://huggingface.co/docs/hub/spaces-overview |
| Gradio SDK | https://huggingface.co/docs/hub/spaces-sdks-gradio |
| Streamlit SDK | https://huggingface.co/docs/hub/spaces-sdks-streamlit |
| Docker SDK | https://huggingface.co/docs/hub/spaces-sdks-docker |
| GPU/hardware tiers | https://huggingface.co/docs/hub/spaces-gpus |
| ZeroGPU (free dynamic GPU) | https://huggingface.co/docs/hub/spaces-zerogpu |
| `gradio` for building the UI | https://www.gradio.app/docs |

**Practical:** push a repo with `app.py` + `requirements.txt` and `sdk: gradio` in `README.md` YAML → Spaces auto-builds and serves it.

---

## 6. Inference stack — serverless, endpoints, serving

HF separates *managed serverless* (Inference Providers), *dedicated autoscaling* (Inference Endpoints), and *self-hosted serving* (TGI).

| Layer | What it is | Link |
|---|---|---|
| **Inference Providers** | Unified serverless access routed to partner providers (fal, Replicate, SambaNova, Together AI, Cerebras, etc.); drop-in OpenAI-compatible chat API. Replaces the old "Inference API (serverless)". | https://huggingface.co/docs/inference-providers/index |
| Providers intro blog | Why/what of Inference Providers | https://huggingface.co/blog/inference-providers |
| Pricing/billing | Routed billing vs. own-key (BYO-key) | https://huggingface.co/docs/inference-providers/pricing |
| HF Inference (own GPUs) | HF's own provider backend | https://huggingface.co/docs/inference-providers/providers/hf-inference |
| **Inference Endpoints** | Dedicated, autoscaling, secure model deployment on managed infra | https://huggingface.co/docs/inference-endpoints/index |
| **TGI** | Self-host LLMs (continuous batching, tensor parallel, quant) | https://huggingface.co/docs/text-generation-inference/index |
| `InferenceClient` | Python/JS client for all of the above | https://huggingface.co/docs/huggingface_hub/guides/inference |

**Key fact:** Inference Providers is a *drop-in replacement for the OpenAI chat-completions API* — point an OpenAI client at the HF router and switch `model` to a Hub repo id. You can use HF-routed billing or bring your own provider API keys.

---

## 7. AutoTrain — no-code / low-code training

AutoTrain Advanced trains/fine-tunes models (LLMs, text/image classification, object detection, etc.) with no code, via UI on Spaces, CLI, or config.

| Surface | Link |
|---|---|
| Product page | https://huggingface.co/autotrain |
| Docs | https://huggingface.co/docs/autotrain/index |
| Quickstart on Spaces | https://huggingface.co/docs/autotrain/quickstart_spaces |
| GitHub | https://github.com/huggingface/autotrain-advanced |
| API | https://huggingface.co/docs/autotrain/autotrain_api |

---

## 8. Hub features — cards, formats, security

| Feature | What/why | Link |
|---|---|---|
| **Model cards** | `README.md` + YAML metadata (license, tags, datasets, metrics, widget) | https://huggingface.co/docs/hub/model-cards |
| Model-card metadata spec | The YAML fields the Hub indexes | https://huggingface.co/docs/hub/model-card-annotated |
| **safetensors** | Safe (no arbitrary code on load), fast zero-copy tensor format; default for new models | https://huggingface.co/docs/safetensors/index |
| **GGUF** | Single-file format (tensors + metadata) optimized for fast load / CPU+GPU inference (llama.cpp/Ollama); Hub parses metadata | https://huggingface.co/docs/hub/gguf |
| Download stats | How the Hub counts downloads | https://huggingface.co/docs/hub/models-download-stats |
| Storage / Xet | Chunk-dedup backend for large repos | https://huggingface.co/docs/hub/storage-backends |
| Security & malware scan | Pickle/secret scanning, signed commits | https://huggingface.co/docs/hub/security |
| Gated & licensed repos | Access-request gating | https://huggingface.co/docs/hub/models-gated |
| Collections | Curate models/datasets/spaces into shareable sets | https://huggingface.co/docs/hub/collections |

**Format guidance**
- Prefer **safetensors** for sharing/serving in PyTorch — it avoids pickle's arbitrary-code-execution risk and loads zero-copy.
- Use **GGUF** when targeting llama.cpp / Ollama / LM Studio local inference; one file bundles quantized weights + metadata.

---

## 9. Quickstart cheat-sheet

```python
# install
pip install -U transformers datasets accelerate huggingface_hub

# auth
hf auth login            # or: from huggingface_hub import login; login()

# fastest inference
from transformers import pipeline
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
pipe("Explain safetensors in one sentence.")

# load a dataset (streaming, no full download)
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# serverless via Inference Providers (OpenAI-compatible)
from huggingface_hub import InferenceClient
client = InferenceClient()                       # uses HF_TOKEN
client.chat_completion(model="meta-llama/Llama-3.1-8B-Instruct",
                       messages=[{"role":"user","content":"hi"}])

# push your own model
from huggingface_hub import create_repo, upload_folder
create_repo("me/my-model")
upload_folder(repo_id="me/my-model", folder_path="./out")
```

---

## Sources

- Hub docs: https://huggingface.co/docs/hub/index
- Models (by downloads): https://huggingface.co/models?sort=downloads
- Model stats blog (2025): https://huggingface.co/blog/lbourdois/huggingface-models-stats
- Download stats docs: https://huggingface.co/docs/hub/models-download-stats
- transformers: https://github.com/huggingface/transformers — https://huggingface.co/docs/transformers/index
- diffusers: https://github.com/huggingface/diffusers — https://huggingface.co/docs/diffusers/index
- datasets: https://github.com/huggingface/datasets — https://huggingface.co/docs/datasets/index
- tokenizers: https://huggingface.co/docs/tokenizers/index
- accelerate: https://github.com/huggingface/accelerate
- peft: https://github.com/huggingface/peft
- trl: https://github.com/huggingface/trl — https://huggingface.co/docs/trl/lora_tuning_peft
- evaluate: https://huggingface.co/docs/evaluate/index
- optimum: https://github.com/huggingface/optimum
- text-generation-inference: https://github.com/huggingface/text-generation-inference
- smolagents: https://github.com/huggingface/smolagents — https://huggingface.co/docs/smolagents/index
- lerobot: https://github.com/huggingface/lerobot — https://huggingface.co/docs/lerobot/index — v0.5.0 release: https://github.com/huggingface/blog/blob/main/lerobot-release-v050.md — SmolVLA: https://huggingface.co/docs/lerobot/smolvla
- gradio: https://www.gradio.app/docs — https://github.com/gradio-app/gradio
- Datasets Hub: https://huggingface.co/datasets?sort=downloads — Dataset Viewer: https://huggingface.co/docs/dataset-viewer/index
- Spaces: https://huggingface.co/docs/hub/spaces — ZeroGPU: https://huggingface.co/docs/hub/spaces-zerogpu
- Inference Providers: https://huggingface.co/docs/inference-providers/index — blog: https://huggingface.co/blog/inference-providers — pricing: https://huggingface.co/docs/inference-providers/pricing — InfoQ overview: https://www.infoq.com/news/2025/02/hugging-face-inference/
- Inference Endpoints: https://huggingface.co/docs/inference-endpoints/index
- AutoTrain: https://huggingface.co/autotrain — https://huggingface.co/docs/autotrain/index — https://github.com/huggingface/autotrain-advanced
- Model cards: https://huggingface.co/docs/hub/model-cards
- safetensors: https://huggingface.co/docs/safetensors/index
- GGUF: https://huggingface.co/docs/hub/gguf
- huggingface_hub client/CLI: https://huggingface.co/docs/huggingface_hub/index

---
_Curated via public-source research (Kaggle Discussions, official solution repos, arXiv, blogs). Verify any specific link before relying on it; gold write-ups live in each competition's Discussion tab._
