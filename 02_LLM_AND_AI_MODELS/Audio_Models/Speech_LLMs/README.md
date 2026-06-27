# Speech LLMs

> Speech-native and audio-language models that hear, reason over, and generate speech end-to-end — spanning audio understanding (speech-to-text) and full-duplex spoken dialogue (speech-to-speech) — a distinct family beyond classic ASR (Whisper) and standalone TTS.

## Why it matters

Classic voice stacks cascade ASR → text LLM → TTS, which loses paralinguistics (emotion, prosody, speaker identity), accumulates latency, and breaks on overlapping speech. Speech LLMs collapse this pipeline: they ingest audio tokens (from neural codecs) directly into a language model and emit text and/or speech tokens, enabling sub-300 ms latency, full-duplex turn-taking, barge-in, and expressive generation. This is the foundation of modern real-time voice agents (GPT-4o voice mode, Moshi, Qwen-Omni) and a fast-growing research frontier with its own benchmarks, codecs, and surveys.

## Taxonomy

| Approach | What it does | Output | Representative models |
|---|---|---|---|
| **Audio-language understanding (S2T)** | Audio + text in, text out; ASR, audio QA, captioning, emotion/event recognition | Text | Qwen2-Audio, SALMONN, WavLLM, Kimi-Audio |
| **Modular-aligned S2S** | Speech encoder + text LLM + speech decoder, aligned but not jointly tokenized | Text + speech | LLaMA-Omni, Mini-Omni, Freeze-Omni, SLAM-Omni |
| **Native multimodal / omni** | Single model natively tokenizes audio (+vision/video), real-time speech out | Speech (streaming) | Qwen2.5-Omni, Qwen3-Omni, Mini-Omni2, Step-Audio |
| **Full-duplex spoken dialogue** | Models user + assistant streams in parallel; listens while speaking | Speech (duplex) | Moshi, SALMONN-omni, GLM-4-Voice |
| **Neural audio codecs / tokenizers** | Convert waveform ↔ discrete tokens the LLM operates on | Audio tokens | Mimi, EnCodec, SoundStream, SNAC, SpeechTokenizer |
| **TTS / voice synthesis** | Text (or conditioning) → high-fidelity speech | Speech | XTTS, Kokoro, Sesame CSM, F5-TTS |

## Key models

| Model | Org | Type | Link |
|---|---|---|---|
| Qwen2-Audio | Alibaba | Audio understanding (S2T) | https://github.com/QwenLM/Qwen2-Audio |
| Qwen2.5-Omni | Alibaba | Native omni, real-time speech | https://github.com/QwenLM/Qwen2.5-Omni |
| Qwen3-Omni | Alibaba | Native omni (text/audio/image/video) | https://github.com/QwenLM/Qwen3-Omni |
| Kimi-Audio | Moonshot AI | Audio foundation (understand/gen/chat) | https://github.com/MoonshotAI/Kimi-Audio |
| Moshi / Mimi | Kyutai | Full-duplex speech-text foundation | https://github.com/kyutai-labs/moshi |
| GLM-4-Voice | THUDM (Zhipu) | End-to-end spoken chatbot | https://github.com/THUDM/GLM-4-Voice |
| LLaMA-Omni | ICTNLP | Modular-aligned S2S | https://github.com/ictnlp/LLaMA-Omni |
| Mini-Omni / Mini-Omni2 | Gpt-Omni | Streaming S2S, duplex | https://github.com/gpt-omni/mini-omni |
| SALMONN / SALMONN-omni | Tsinghua/ByteDance | Hearing LLM; codec-free full-duplex | https://github.com/bytedance/SALMONN |
| Step-Audio | StepFun | Production speech interaction | https://github.com/stepfun-ai/Step-Audio |
| SpeechGPT | Fudan (OpenMOSS) | Cross-modal conversational LLM | https://github.com/0nutation/SpeechGPT |

## TTS & codecs

| Tool | Type | Link |
|---|---|---|
| Mimi (streaming codec, 12.5 Hz, 1.1 kbps) | Neural audio codec | https://huggingface.co/kyutai/mimi |
| EnCodec | Neural audio codec | https://github.com/facebookresearch/encodec |
| SNAC (multi-scale RVQ) | Neural audio codec | https://github.com/hubertsiuzdak/snac |
| SpeechTokenizer | Semantic+acoustic tokenizer | https://github.com/ZhangXInFD/SpeechTokenizer |
| Kokoro-82M | Lightweight TTS | https://huggingface.co/hexgrad/Kokoro-82M |
| Sesame CSM | Conversational speech model | https://github.com/SesameAILabs/csm |
| XTTS / Coqui TTS | Multilingual voice cloning TTS | https://github.com/coqui-ai/TTS |
| F5-TTS | Flow-matching TTS | https://github.com/SWivid/F5-TTS |

## Benchmarks

| Benchmark | Focus | Link |
|---|---|---|
| VoiceBench | LLM-based voice assistants (real + synthetic spoken instructions) | https://github.com/MatthewCYM/VoiceBench |
| AudioBench | Universal benchmark for audio LLMs | https://arxiv.org/abs/2406.16020 |
| Dynamic-SUPERB | 180+ community speech-processing tasks | https://github.com/dynamic-superb/dynamic-superb |
| URO-Bench | End-to-end spoken dialogue evaluation | https://arxiv.org/abs/2502.17810 |
| WildSpeech-Bench | End-to-end Speech LLMs "in the wild" | https://arxiv.org/abs/2506.21875 |

## Key papers

| Paper | Year | Link |
|---|---|---|
| SpeechGPT: Empowering LLMs with Intrinsic Cross-Modal Conversational Abilities | 2023 | https://arxiv.org/abs/2305.11000 |
| SALMONN: Towards Generic Hearing Abilities for LLMs | 2023 | https://arxiv.org/abs/2310.13289 |
| WavLLM: Towards Robust and Adaptive Speech LLM | 2024 | https://arxiv.org/abs/2404.00656 |
| Qwen2-Audio Technical Report | 2024 | https://arxiv.org/abs/2407.10759 |
| Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming | 2024 | https://arxiv.org/abs/2408.16725 |
| LLaMA-Omni: Seamless Speech Interaction with LLMs | 2024 | https://arxiv.org/abs/2409.06666 |
| Moshi: A Speech-Text Foundation Model for Real-Time Dialogue | 2024 | https://arxiv.org/abs/2410.00037 |
| Recent Advances in Speech Language Models: A Survey (ACL 2025) | 2024 | https://arxiv.org/abs/2410.03751 |
| GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot | 2024 | https://arxiv.org/abs/2412.02612 |
| SALMONN-omni: A Codec-free LLM for Full-duplex Speech | 2024 | https://arxiv.org/abs/2411.18138 |
| Qwen2.5-Omni Technical Report | 2025 | https://arxiv.org/abs/2503.20215 |
| Qwen3-Omni Technical Report | 2025 | https://arxiv.org/abs/2509.17765 |

## Cross-references in AIForge

- [Whisper / OpenAI ASR](../Whisper_OpenAI.md) — classic speech-to-text foundation
- [Music Generation](../Music_Generation/) — sibling audio-generation family
- [Multimodal Models](../../Multimodal_Models/) — omni/vision-language-audio models
- [Text LLMs](../../Text_LLMs/) — backbone language models underlying speech LLMs

## Sources

- Recent Advances in Speech Language Models: A Survey — https://arxiv.org/abs/2410.03751
- Awesome-SpeechLM-Survey (ACL 2025) — https://github.com/dreamtheater123/Awesome-SpeechLM-Survey
- Moshi paper — https://arxiv.org/abs/2410.00037 · repo — https://github.com/kyutai-labs/moshi
- Kyutai neural audio codecs explainer — https://kyutai.org/codec-explainer/
- Qwen2.5-Omni — https://github.com/QwenLM/Qwen2.5-Omni · blog https://qwenlm.github.io/blog/qwen2.5-omni/
- Kimi-Audio — https://github.com/MoonshotAI/Kimi-Audio
- GLM-4-Voice — https://arxiv.org/abs/2412.02612
- VoiceBench — https://github.com/MatthewCYM/VoiceBench · https://arxiv.org/abs/2410.17196

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
