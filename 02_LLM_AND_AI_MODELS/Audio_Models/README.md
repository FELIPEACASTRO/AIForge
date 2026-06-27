# Audio Models

> Neural models that perceive, transcribe, translate, or generate audio — covering automatic speech recognition (ASR), text-to-speech (TTS), music & sound generation, neural audio codecs, and end-to-end speech LLMs.

## Why it matters

Audio is one of the four core modalities (alongside text, image, and video) and the most natural human interface. Modern audio models unify ASR, TTS, and dialogue into single foundation models, enable real-time full-duplex voice agents, and power music/sound generation, multilingual speech translation, and on-device assistants. The shift to discrete neural audio codec tokens lets the same transformer/LM stack used for text drive speech, making audio a first-class citizen of the foundation-model era.

## Categories

| Sub-area | What it does | Representative models |
|---|---|---|
| **ASR (speech-to-text)** | Transcribe speech, often multilingual + robust to noise | Whisper, SeamlessM4T, w2v-BERT 2.0 |
| **TTS (text-to-speech)** | Synthesize natural/cloned voice from text | StyleTTS 2, VALL-E 2, XTTS-v2, Bark |
| **Speech translation (S2ST/S2TT)** | Speech↔text across languages | SeamlessM4T, Seamless (streaming) |
| **Music generation** | Text/melody-conditioned music | MusicGen, Stable Audio Open |
| **Sound / general audio generation** | Text-to-audio (effects, ambience) | Stable Audio Open, AudioCraft |
| **Neural audio codecs** | Compress audio into discrete tokens for LMs | EnCodec, SoundStream, Mimi |
| **Speech LLMs / full-duplex dialogue** | Unified audio-text reasoning & real-time conversation | Moshi, Qwen2-Audio, GLM-4-Voice |

## Key models

### ASR & speech translation

| Model | Org | Highlights | Link |
|---|---|---|---|
| Whisper | OpenAI | Encoder-decoder ASR trained on 680k h weak-supervision; multilingual + translation; large-v3 | https://github.com/openai/whisper |
| Whisper large-v3 (weights) | OpenAI | HF checkpoint | https://huggingface.co/openai/whisper-large-v3 |
| SeamlessM4T | Meta | Single model: S2ST, S2TT, T2ST, T2TT, ASR for up to 100 languages | https://github.com/facebookresearch/seamless_communication |

### Text-to-speech & voice cloning

| Model | Org | Highlights | Link |
|---|---|---|---|
| StyleTTS 2 | Columbia (yl4579) | Style diffusion + SLM adversarial training; human-level on LJSpeech/VCTK | https://github.com/yl4579/StyleTTS2 |
| VALL-E 2 | Microsoft | Neural-codec LM; human parity zero-shot TTS from 3-s prompt | https://arxiv.org/abs/2406.05370 |
| Coqui XTTS-v2 | Coqui | Multilingual voice-cloning TTS, 17 languages | https://huggingface.co/coqui/XTTS-v2 |
| Bark | Suno AI | Transformer text-to-audio: speech, music, SFX, nonverbal | https://github.com/suno-ai/bark |

### Music & general audio generation

| Model | Org | Highlights | Link |
|---|---|---|---|
| MusicGen / AudioCraft | Meta (FAIR) | Single-stage LM over EnCodec tokens; text + melody conditioning | https://github.com/facebookresearch/audiocraft |
| Stable Audio Open 1.0 | Stability AI | Open-weights DiT latent diffusion; up to 47 s stereo 44.1 kHz | https://huggingface.co/stabilityai/stable-audio-open-1.0 |

### Neural audio codecs

| Model | Org | Highlights | Link |
|---|---|---|---|
| EnCodec | Meta | Real-time RVQ codec; 24 kHz mono / 48 kHz stereo, 1.5–24 kbps | https://github.com/facebookresearch/encodec |
| Mimi (in Moshi) | Kyutai | Streaming codec underpinning Moshi's speech tokens | https://github.com/kyutai-labs/moshi |

### Speech LLMs / full-duplex dialogue

| Model | Org | Highlights | Link |
|---|---|---|---|
| Moshi | Kyutai | First real-time full-duplex speech-text LM; ~200 ms latency; uses Mimi codec | https://github.com/kyutai-labs/moshi |
| Qwen2-Audio | Alibaba | Audio-language model; voice chat + audio analysis; beats Gemini-1.5-pro on AIR-Bench | https://github.com/QwenLM/Qwen2-Audio |
| GLM-4-Voice | Zhipu/THU | End-to-end human-like spoken chatbot | https://arxiv.org/abs/2412.02612 |

## Datasets & benchmarks

| Dataset | Use | Scale / notes | Link |
|---|---|---|---|
| LibriSpeech | ASR benchmark | ~1000 h, 16 kHz read English (LibriVox) | https://www.openslr.org/12 |
| LibriTTS | TTS corpus | 585 h, 24 kHz, 2,456 speakers (from LibriSpeech) | https://www.openslr.org/60 |
| Common Voice | Multilingual ASR | Crowdsourced, 100+ languages, accent/age/gender metadata | https://commonvoice.mozilla.org/ |
| VCTK | Multi-speaker TTS | 110 English speakers, varied accents | https://datashare.ed.ac.uk/handle/10283/3443 |
| AIR-Bench | Audio LLM eval | Audio-centric instruction-following benchmark | https://github.com/OFA-Sys/AIR-Bench |

## Key papers

| Year | Paper | Link |
|---|---|---|
| 2022 | Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | https://arxiv.org/abs/2212.04356 |
| 2022 | High Fidelity Neural Audio Compression (EnCodec) | https://arxiv.org/abs/2210.13438 |
| 2023 | Neural Codec Language Models are Zero-Shot TTS (VALL-E) | https://arxiv.org/abs/2301.02111 |
| 2023 | Simple and Controllable Music Generation (MusicGen) | https://arxiv.org/abs/2306.05284 |
| 2023 | StyleTTS 2: Towards Human-Level Text-to-Speech | https://arxiv.org/abs/2306.07691 |
| 2023 | SeamlessM4T: Massively Multilingual & Multimodal Machine Translation | https://arxiv.org/abs/2308.11596 |
| 2024 | Qwen2-Audio Technical Report | https://arxiv.org/abs/2407.10759 |
| 2024 | Stable Audio Open | https://arxiv.org/abs/2407.14358 |
| 2024 | VALL-E 2: Human Parity Zero-Shot TTS | https://arxiv.org/abs/2406.05370 |
| 2024 | Moshi: a speech-text foundation model for real-time dialogue | https://arxiv.org/abs/2410.00037 |

## Cross-references in AIForge

- [Multimodal Models](../Multimodal_Models/) — vision-language-audio foundation models
- [Diffusion Models](../Diffusion_Models/) — latent diffusion behind Stable Audio / TTS
- [Text LLMs](../Text_LLMs/) — backbones reused by speech LLMs (Moshi, Qwen2-Audio)
- [Video Models](../Video_Models/) — sibling generative modality with audio tracks

## Sources

- https://github.com/openai/whisper — Whisper repo
- https://arxiv.org/abs/2212.04356 — Whisper paper
- https://github.com/kyutai-labs/moshi · https://arxiv.org/abs/2410.00037 — Moshi
- https://github.com/facebookresearch/audiocraft · https://arxiv.org/abs/2306.05284 — MusicGen
- https://arxiv.org/abs/2306.07691 — StyleTTS 2
- https://github.com/facebookresearch/seamless_communication · https://arxiv.org/abs/2308.11596 — SeamlessM4T
- https://github.com/facebookresearch/encodec · https://arxiv.org/abs/2210.13438 — EnCodec
- https://arxiv.org/abs/2301.02111 · https://arxiv.org/abs/2406.05370 — VALL-E / VALL-E 2
- https://github.com/QwenLM/Qwen2-Audio · https://arxiv.org/abs/2407.10759 — Qwen2-Audio
- https://huggingface.co/stabilityai/stable-audio-open-1.0 · https://arxiv.org/abs/2407.14358 — Stable Audio Open
- https://github.com/suno-ai/bark — Bark
- https://www.openslr.org/12 · https://www.openslr.org/60 — LibriSpeech / LibriTTS

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
