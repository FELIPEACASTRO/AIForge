# Video Understanding

> Video understanding is the set of methods that extract spatiotemporal semantics from video — recognizing actions, localizing events in time, answering questions, captioning, and reasoning over long sequences — using spatiotemporal backbones, self-supervised pretraining, and video-language (multimodal) models.

## Why it matters

Video is the dominant modality of real-world sensory data, but it adds a temporal axis that images lack: models must reason about motion, ordering, causality, and long-range dependencies, not just appearance. Modern video foundation models and video-LLMs power search, content moderation, robotics/embodied agents, surveillance, sports/medical analytics, and assistants that can watch and discuss hour-long footage. The field has shifted from supervised CNN/3D-CNN action classifiers to transformer backbones, masked-autoencoder self-supervision, and instruction-tuned video-language models that unify recognition, retrieval, and dialogue.

## Taxonomy

| Sub-area | What it does | Representative approaches |
|---|---|---|
| Spatiotemporal backbones | Encode clips into features | 3D CNNs (I3D, SlowFast), video transformers (ViViT, TimeSformer) |
| Self-supervised pretraining | Learn from unlabeled video | Masked video modeling (VideoMAE), joint-embedding prediction (V-JEPA) |
| Video foundation models | General-purpose video encoders | InternVideo, VideoPrism, VideoMAE V2 |
| Video-language models (Video-LLMs) | Caption, QA, dialogue over video | VideoLLaMA, LLaVA-Video, Qwen2.5-VL, InternVideo2.5 |
| Action recognition / detection | Classify or localize actions | Kinetics/SSv2 classifiers, temporal action detection |
| Temporal grounding / retrieval | Localize moments, text-video retrieval | Moment retrieval, video-text contrastive |
| Long-video understanding | Reason over minutes-to-hours | Token compression, memory, streaming Video-LLMs |

## Key models & backbones

| Model | Type | Link |
|---|---|---|
| TimeSformer | Space-time attention video transformer | https://arxiv.org/abs/2102.05095 |
| ViViT | Pure-transformer video classifier | https://arxiv.org/abs/2103.15691 |
| SlowFast | Dual-pathway 3D CNN | https://arxiv.org/abs/1812.03982 |
| VideoMAE | Masked autoencoder SSL pretraining | https://arxiv.org/abs/2203.12602 |
| VideoMAE V2 | Dual-masking, scaled MAE pretraining | https://github.com/OpenGVLab/VideoMAEv2 |
| V-JEPA 2 | Action-free joint-embedding video model | https://arxiv.org/abs/2506.09985 |
| VideoPrism | General video encoder (Google) | https://arxiv.org/abs/2402.13217 |
| InternVideo / InternVideo2 | Video foundation models & data | https://github.com/OpenGVLab/InternVideo |

## Key video-language models (Video-LLMs)

| Model | Notes | Link |
|---|---|---|
| InternVideo2 | Progressive training to 6B encoder; SOTA video-text | https://arxiv.org/abs/2403.15377 |
| VideoLLaMA 2 | Audio-visual video-language model | https://github.com/DAMO-NLP-SG/VideoLLaMA2 |
| VideoLLaMA 3 | Frontier image+video MLLM (2B/7B) | https://github.com/DAMO-NLP-SG/VideoLLaMA3 |
| LLaVA-Video | Video instruction tuning + synthetic data | https://arxiv.org/abs/2410.02713 |
| LLaVA-OneVision | Unified image/multi-image/video MLLM | https://arxiv.org/abs/2408.03326 |
| Qwen2.5-VL | M-RoPE spatial-temporal encoding; long video | https://arxiv.org/abs/2502.13923 |
| Video-LLaVA | Aligned image/video projection to LLM | https://arxiv.org/abs/2311.10122 |
| Video-ChatGPT | Conversational video understanding | https://arxiv.org/abs/2306.05424 |

## Tools & frameworks

| Tool | Purpose | Link |
|---|---|---|
| MMAction2 | Action recognition/detection toolbox | https://github.com/open-mmlab/mmaction2 |
| PyTorchVideo | Video models & datasets library | https://github.com/facebookresearch/pytorchvideo |
| Hugging Face Transformers (video) | VideoMAE, TimeSformer, ViViT, Video-LLMs | https://huggingface.co/docs/transformers/en/tasks/video_classification |
| Decord | Efficient video reader for DL | https://github.com/dmlc/decord |
| lmms-eval | Eval harness for video/multimodal LLMs | https://github.com/EvolvingLMMs-Lab/lmms-eval |

## Datasets

| Dataset | Focus | Link |
|---|---|---|
| Kinetics (400/600/700) | Trimmed human action clips | https://github.com/cvdfoundation/kinetics-dataset |
| Something-Something V2 | Fine-grained temporal/motion actions | https://arxiv.org/abs/1706.04261 |
| ActivityNet | Untrimmed activity detection (200 classes) | https://activity-net.org/ |
| Charades | Daily indoor activities, multi-label | https://prior.allenai.org/projects/charades |
| Ego4D | Large-scale egocentric video | https://ego4d-data.org/ |
| HowTo100M | Instructional video-text pretraining | https://www.di.ens.fr/willow/research/howto100m/ |

## Benchmarks

| Benchmark | Tests | Link |
|---|---|---|
| Video-MME | Comprehensive MLLM video QA (11s–1h) | https://arxiv.org/abs/2405.21075 |
| MVBench | 20 temporal-reasoning multiple-choice tasks | https://arxiv.org/abs/2311.17005 |
| EgoSchema | Long-form egocentric QA (3-min clips) | https://arxiv.org/abs/2308.09126 |
| NExT-QA | Causal & temporal action reasoning QA | https://arxiv.org/abs/2105.08276 |
| TempCompass | Fine-grained temporal perception | https://arxiv.org/abs/2403.00476 |
| LongVideoBench | Long-context interleaved video-language | https://arxiv.org/abs/2407.15754 |

## Key papers

- **TimeSformer: Is Space-Time Attention All You Need for Video Understanding?** — https://arxiv.org/abs/2102.05095
- **ViViT: A Video Vision Transformer** — https://arxiv.org/abs/2103.15691
- **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training** — https://arxiv.org/abs/2203.12602
- **VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking** — https://arxiv.org/abs/2303.16727
- **InternVideo2: Scaling Foundation Models for Multimodal Video Understanding** — https://arxiv.org/abs/2403.15377
- **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** — https://arxiv.org/abs/2506.09985
- **VideoPrism: A Foundational Visual Encoder for Video Understanding** — https://arxiv.org/abs/2402.13217
- **Qwen2.5-VL Technical Report** — https://arxiv.org/abs/2502.13923
- **Video-MME: The First-Ever Comprehensive Evaluation Benchmark of MLLMs in Video Analysis** — https://arxiv.org/abs/2405.21075
- **The Kinetics Human Action Video Dataset** — https://arxiv.org/abs/1705.06950

## Cross-references in AIForge

- [Vision Transformers](../Vision_Transformers/) — backbone architectures underlying video transformers.
- [Vision Language Models](../Vision_Language_Models/) — image-text models that Video-LLMs extend.
- [Computer Vision](../Computer_Vision/) — broader visual perception context.
- [Long Context Models](../Long_Context_Models/) — techniques for reasoning over long video sequences.

## Sources

- https://github.com/NeeluMadan/ViFM_Survey (Foundation Models for Video Understanding: A Survey)
- https://github.com/OpenGVLab/InternVideo
- https://arxiv.org/abs/2403.15377 (InternVideo2)
- https://arxiv.org/abs/2203.12602 (VideoMAE)
- https://arxiv.org/abs/2506.09985 (V-JEPA 2)
- https://arxiv.org/abs/2402.13217 (VideoPrism)
- https://arxiv.org/abs/2502.13923 (Qwen2.5-VL)
- https://arxiv.org/abs/2405.21075 (Video-MME)
- https://arxiv.org/abs/1705.06950 (Kinetics)
- https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
