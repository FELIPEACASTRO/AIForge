# Music Generation

> Generative models that synthesize music — instrumental or full songs with vocals — from text prompts, melodies, lyrics, or other audio conditioning, distinct from speech ASR/TTS.

## Why it matters

Music generation is a fast-moving generative-audio family that turns natural-language prompts, lyrics, or reference melodies into high-fidelity audio. The 2023–2026 wave (MusicGen, MusicLM, Stable Audio, AudioLDM, Suno, Udio, YuE, ACE-Step) shifted the field from short instrumental clips toward full multi-minute songs with coherent vocals. It sits at the center of active product (Suno, Udio), open-weight (MusicGen, Stable Audio Open, YuE, ACE-Step), and legal/licensing debates — major-label copyright lawsuits in 2024–2025 made the autoregressive-vs-diffusion and hosted-vs-self-hosted tradeoffs strategically important.

## Taxonomy

| Approach | How it works | Representative systems |
|---|---|---|
| Autoregressive LM over audio tokens | Transformer LM predicts discrete codec tokens (EnCodec/RVQ), decoded to audio | MusicGen, MusicLM, Jukebox, YuE |
| Latent diffusion / flow | Diffusion or flow matching in a compressed VAE latent, text-conditioned | Stable Audio (Open), AudioLDM/AudioLDM 2, Noise2Music, ACE-Step |
| Spectrogram diffusion | Image-style diffusion on mel/STFT spectrograms, then vocoded | Riffusion, MusicLDM |
| Conditioning modality | text-to-music · melody/chord-conditioned · lyrics-to-song (vocals) · continuation/inpainting · stem/multi-track | MusicGen-Melody, Music ControlNet, YuE, Mustango |
| Delivery | Open-weight (self-hostable) vs. closed hosted product | Open: MusicGen, Stable Audio Open, YuE, ACE-Step · Closed: Suno, Udio, MusicLM (Google) |

## Key models & tools

| Model / tool | Org | Type | Open? | Link |
|---|---|---|---|---|
| MusicGen / AudioCraft | Meta | Autoregressive LM (EnCodec) | Weights (MIT code, CC-BY-NC models) | https://github.com/facebookresearch/audiocraft |
| MusicLM | Google | Autoregressive (hierarchical) | Closed (paper + demos) | https://google-research.github.io/seanet/musiclm/examples/ |
| Stable Audio Open 1.0 | Stability AI | Latent diffusion (DiT) | Open weights | https://huggingface.co/stabilityai/stable-audio-open-1.0 |
| AudioLDM 2 | Liu et al. | Latent diffusion (CLAP-guided) | Open weights | https://github.com/haoheliu/AudioLDM2 |
| YuE | M-A-P / multimodal-art-projection | Dual-LLaMA autoregressive, full songs + vocals | Open weights | https://github.com/multimodal-art-projection/YuE |
| ACE-Step | ACE Studio / StepFun | Diffusion + DCAE + linear transformer | Open weights | https://github.com/ace-step/ACE-Step |
| Riffusion | Forsgren & Martiros | Spectrogram diffusion (SD fine-tune) | Open weights | https://github.com/riffusion/riffusion-hobby |
| Jukebox | OpenAI | Hierarchical VQ-VAE + autoregressive | Open weights | https://github.com/openai/jukebox |
| Mustango | Declare Lab | Controllable diffusion (FLAN-T5) | Open weights | https://github.com/AMAAI-Lab/mustango |
| Suno | Suno | Hosted full-song product | Closed | https://suno.com/ |
| Udio | Udio | Hosted full-song product | Closed | https://www.udio.com/ |

## Datasets & benchmarks

| Resource | Use | Link |
|---|---|---|
| MusicCaps | 5.5k text-captioned clips; standard text-to-music eval set (from MusicLM) | https://huggingface.co/datasets/google/MusicCaps |
| Song Describer Dataset | Crowd-sourced music captions for text-to-music eval | https://huggingface.co/datasets/mwitiderrick/SongDescriber |
| FMA (Free Music Archive) | Large CC-licensed music corpus for training/eval | https://github.com/mdeff/fma |
| MTG-Jamendo | 55k full tracks with tags; tagging + generation | https://github.com/MTG/mtg-jamendo-dataset |
| SongEval | Benchmark dataset for song aesthetics evaluation | https://arxiv.org/abs/2505.10793 |
| Frechet Audio Distance (FAD) | Standard distributional fidelity metric for generated audio | https://github.com/gudgud96/frechet-audio-distance |

Common metrics: FAD (fidelity), KL divergence over audio-tagger logits, CLAP score (text–audio alignment), plus human MOS / aesthetics ratings.

## Key papers

| Paper | Venue / ID | Link |
|---|---|---|
| Jukebox: A Generative Model for Music | arXiv 2005.00341 | https://arxiv.org/abs/2005.00341 |
| MusicLM: Generating Music From Text | arXiv 2301.11325 | https://arxiv.org/abs/2301.11325 |
| AudioLDM: Text-to-Audio Generation with Latent Diffusion Models | arXiv 2301.12503 | https://arxiv.org/abs/2301.12503 |
| Noise2Music: Text-conditioned Music Generation with Diffusion | arXiv 2302.03917 | https://arxiv.org/abs/2302.03917 |
| Simple and Controllable Music Generation (MusicGen) | arXiv 2306.05284 (NeurIPS 2023) | https://arxiv.org/abs/2306.05284 |
| AudioLDM 2: Holistic Audio Generation with Self-Supervised Pretraining | arXiv 2308.05734 | https://arxiv.org/abs/2308.05734 |
| Music ControlNet: Multiple Time-varying Controls | arXiv 2311.07069 | https://arxiv.org/abs/2311.07069 |
| Mustango: Toward Controllable Text-to-Music Generation | arXiv 2311.08355 | https://arxiv.org/abs/2311.08355 |
| Stable Audio Open | arXiv 2407.14358 | https://arxiv.org/abs/2407.14358 |
| ACE-Step: A Step Towards Music Generation Foundation Model | arXiv 2506.00045 | https://arxiv.org/abs/2506.00045 |

## Cross-references in AIForge

- [Audio Models — Speech Recognition (Whisper / ASR) — sibling generative-vs-recognition audio family
- [Audio Models index](../) — parent section for all audio model families
- [Diffusion Models](../../Diffusion_Models/) — shared latent-diffusion / flow-matching backbone used by Stable Audio, AudioLDM, ACE-Step
- [Multimodal Models](../../Multimodal_Models/) — text↔audio conditioning and CLAP-style joint embeddings

## Sources

- https://github.com/facebookresearch/audiocraft
- https://arxiv.org/abs/2306.05284
- https://arxiv.org/abs/2301.11325
- https://huggingface.co/stabilityai/stable-audio-open-1.0
- https://arxiv.org/abs/2407.14358
- https://github.com/haoheliu/AudioLDM2
- https://github.com/multimodal-art-projection/YuE
- https://github.com/ace-step/ACE-Step
- https://arxiv.org/abs/2506.00045
- https://arxiv.org/abs/2005.00341
- https://arxiv.org/abs/2302.03917
- https://arxiv.org/abs/2311.08355
- https://www.spheron.network/blog/deploy-open-source-ai-music-generation-gpu-cloud-2026/

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
