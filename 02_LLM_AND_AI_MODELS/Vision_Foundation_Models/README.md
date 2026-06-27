# Vision Foundation Models

> Large pretrained image/video backbones that produce general-purpose visual representations — via self-supervision (DINOv2/v3, JEPA), image-text contrastive learning (CLIP, SigLIP), or promptable segmentation (SAM) — reusable across detection, segmentation, retrieval, depth, and multimodal LLMs.

## Why it matters

Vision foundation models (VFMs) replace task-specific CNNs with a single frozen backbone that transfers to dozens of downstream tasks with little or no fine-tuning. Self-supervised encoders like DINOv3 yield dense features strong enough for segmentation and depth *without labels*, while contrastive encoders like CLIP/SigLIP supply the visual front-end for almost every modern vision-language model (LLaVA, Qwen-VL, etc.). The segmentation family (SAM, SAM 2) made promptable, zero-shot masking a commodity. Picking the right VFM — and knowing how to distill or agglomerate several into one — is now a core architectural decision for any perception or multimodal system.

## Taxonomy

| Paradigm | Training signal | What it's good at | Representative models |
|---|---|---|---|
| **Self-supervised (DINO/SSL)** | Self-distillation, masked image modeling | Dense features, kNN/linear probe, depth, correspondence | DINOv2, DINOv3, iBOT, MAE |
| **Joint-Embedding Predictive (JEPA)** | Predict latent reps of masked regions | Label-free features, video world-models, planning | I-JEPA, V-JEPA 2 |
| **Image-text contrastive** | Image–caption alignment | Zero-shot classification, retrieval, VLM vision encoder | CLIP, OpenCLIP, SigLIP, SigLIP 2, EVA-CLIP, MetaCLIP |
| **Promptable segmentation** | Interactive mask supervision + data engine | Zero-shot / promptable masks in images & video | SAM, SAM 2 |
| **Agglomerative / distilled** | Multi-teacher distillation into one backbone | One backbone with CLIP + DINO + SAM properties | AM-RADIO / RADIOv2.5, Theia |

## Key models

### Self-supervised & JEPA backbones

| Model | Org | Notes | Link |
|---|---|---|---|
| DINOv2 | Meta | Robust label-free ViT features; strong linear/kNN probes | [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) · [code](https://github.com/facebookresearch/dinov2) |
| DINOv3 | Meta | Scales SSL to 7B params; Gram anchoring fixes dense-feature drift; SOTA dense tasks | [arXiv 2508.10104](https://arxiv.org/abs/2508.10104) · [code](https://github.com/facebookresearch/dinov3) |
| I-JEPA | Meta | Predict latent reps of image regions; no hand-crafted augmentations | [arXiv 2301.08243](https://arxiv.org/abs/2301.08243) · [code](https://github.com/facebookresearch/ijepa) |
| V-JEPA 2 | Meta | Video world model (1M+ hrs); understanding, prediction, zero-shot robot control | [arXiv 2506.09985](https://arxiv.org/abs/2506.09985) · [code](https://github.com/facebookresearch/vjepa2) |
| MAE | Meta | Masked autoencoder; scalable self-supervised pretraining baseline | [arXiv 2111.06377](https://arxiv.org/abs/2111.06377) |

### Image-text contrastive encoders

| Model | Org | Notes | Link |
|---|---|---|---|
| CLIP | OpenAI | Original contrastive image-text; zero-shot classification foundation | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) · [code](https://github.com/openai/CLIP) |
| OpenCLIP | LAION / ML Foundations | Open reproduction + scaling laws; many public checkpoints | [GitHub](https://github.com/mlfoundations/open_clip) |
| SigLIP | Google | Sigmoid loss; better at small batch sizes, simpler than softmax CLIP | [arXiv 2303.15343](https://arxiv.org/abs/2303.15343) |
| SigLIP 2 | Google | Multilingual, improved localization + dense features; common VLM encoder | [arXiv 2502.14786](https://arxiv.org/abs/2502.14786) |
| EVA-CLIP / EVA-CLIP-18B | BAAI | Scaling CLIP to 18B params; widely reused vision backbone | [arXiv 2303.15389](https://arxiv.org/abs/2303.15389) · [18B: arXiv 2402.04252](https://arxiv.org/abs/2402.04252) · [code](https://github.com/baaivision/EVA) |
| MetaCLIP | Meta | Demystifies CLIP data curation; reproducible open data pipeline | [arXiv 2309.16671](https://arxiv.org/abs/2309.16671) · [code](https://github.com/facebookresearch/MetaCLIP) |

### Segmentation & agglomerative

| Model | Org | Notes | Link |
|---|---|---|---|
| SAM | Meta | Promptable segmentation; SA-1B (1B+ masks); zero-shot masking | [arXiv 2304.02643](https://arxiv.org/abs/2304.02643) · [code](https://github.com/facebookresearch/segment-anything) |
| SAM 2 | Meta | Images + video, streaming memory; 6x faster than SAM on images | [arXiv 2408.00714](https://arxiv.org/abs/2408.00714) · [code](https://github.com/facebookresearch/sam2) |
| AM-RADIO / RADIOv2.5 | NVIDIA | Multi-teacher distillation of CLIP+DINOv2+SAM into one backbone | [arXiv 2312.06709](https://arxiv.org/abs/2312.06709) · [code](https://github.com/NVlabs/RADIO) |

## Datasets & benchmarks

| Name | Use | Link |
|---|---|---|
| ImageNet-1k | Linear / kNN probe, zero-shot classification | [image-net.org](https://www.image-net.org/) |
| ADE20K | Semantic segmentation transfer | [paper](https://arxiv.org/abs/1608.05442) |
| SA-1B | SAM pretraining (1.1B masks, 11M images) | [dataset](https://ai.meta.com/datasets/segment-anything/) |
| SA-V | SAM 2 video segmentation (largest video mask set) | [dataset](https://ai.meta.com/datasets/segment-anything-video/) |
| COCO | Detection / instance segmentation transfer | [cocodataset.org](https://cocodataset.org/) |
| Something-Something v2 | Motion / temporal understanding (JEPA video) | [paper](https://arxiv.org/abs/1706.04261) |
| LAION-5B | Open image-text pretraining corpus (OpenCLIP) | [laion.ai](https://laion.ai/blog/laion-5b/) |

## Key papers

- **CLIP** — Learning Transferable Visual Models From Natural Language Supervision — [arXiv 2103.00020](https://arxiv.org/abs/2103.00020)
- **SigLIP** — Sigmoid Loss for Language Image Pre-Training — [arXiv 2303.15343](https://arxiv.org/abs/2303.15343)
- **SigLIP 2** — Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features — [arXiv 2502.14786](https://arxiv.org/abs/2502.14786)
- **DINOv2** — Learning Robust Visual Features without Supervision — [arXiv 2304.07193](https://arxiv.org/abs/2304.07193)
- **DINOv3** — Scaling self-supervised learning for vision foundation models — [arXiv 2508.10104](https://arxiv.org/abs/2508.10104)
- **I-JEPA** — Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture — [arXiv 2301.08243](https://arxiv.org/abs/2301.08243)
- **V-JEPA 2** — Self-Supervised Video Models Enable Understanding, Prediction and Planning — [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- **SAM** — Segment Anything — [arXiv 2304.02643](https://arxiv.org/abs/2304.02643)
- **SAM 2** — Segment Anything in Images and Videos — [arXiv 2408.00714](https://arxiv.org/abs/2408.00714)
- **AM-RADIO** — Agglomerative Vision Foundation Model: Reduce All Domains Into One — [arXiv 2312.06709](https://arxiv.org/abs/2312.06709)

## Cross-references in AIForge

- [Multimodal Models](../Multimodal_Models/) — VLMs that consume these vision encoders (CLIP/SigLIP front-ends)
- [Embedding Models](../Embedding_Models/) — contrastive image/text embeddings & retrieval
- [Video Models](../Video_Models/) — video understanding & generation (SAM 2, V-JEPA 2)
- [Diffusion Models](../Diffusion_Models/) — image generation that often conditions on these encoders

## Sources

- https://arxiv.org/abs/2304.07193 (DINOv2)
- https://arxiv.org/abs/2508.10104 (DINOv3)
- https://arxiv.org/abs/2502.14786 (SigLIP 2)
- https://arxiv.org/abs/2303.15343 (SigLIP)
- https://arxiv.org/abs/2103.00020 (CLIP)
- https://arxiv.org/abs/2301.08243 (I-JEPA)
- https://arxiv.org/abs/2506.09985 (V-JEPA 2)
- https://arxiv.org/abs/2304.02643 (SAM)
- https://arxiv.org/abs/2408.00714 (SAM 2)
- https://arxiv.org/abs/2312.06709 (AM-RADIO)
- https://arxiv.org/abs/2402.04252 (EVA-CLIP-18B)
- https://github.com/mlfoundations/open_clip
- https://github.com/NVlabs/RADIO
- https://ai.meta.com/research/vjepa/

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
