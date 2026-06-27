# OCR Document AI

> Neural OCR and document-AI models that turn pixels (scans, PDFs, photos) into structured text, Markdown, tables, formulas and key-value fields — the backbone of enterprise document understanding (invoices, contracts, forms, academic papers).

## Why it matters

Document understanding is one of the largest enterprise AI use cases: invoices, contracts, forms, receipts and scientific papers all arrive as images or PDFs, not clean text. Modern neural OCR has moved from box-and-classify pipelines (Tesseract-style) to end-to-end vision-language models (VLMs) that emit Markdown, tables, LaTeX and reading order in one pass. The current frontier — GOT-OCR2.0, DeepSeek-OCR, Surya, dots.ocr, olmOCR, PaddleOCR-VL — rivals proprietary APIs while being open-weight, and underpins both RAG ingestion and the generation of trillions of training tokens for LLMs.

## Taxonomy

| Approach | Idea | Representative models |
|---|---|---|
| **Classic pipeline OCR** | Detect text regions, recognize crops, post-process layout | Tesseract, PaddleOCR (PP-OCR), docTR |
| **OCR-free document transformers** | Image-to-sequence; no external OCR engine | Donut, Dessurt, Pix2Struct |
| **OCR-free academic / Markdown** | Render full-page document to Markdown/LaTeX | Nougat, GOT-OCR2.0 |
| **Layout-aware multimodal** | Joint text + layout + image pretraining for VQA/KIE | LayoutLMv3, UDOP |
| **Unified OCR VLMs (2024-2026)** | One VLM for layout + OCR + tables + formulas → Markdown | Surya, dots.ocr, olmOCR, PaddleOCR-VL, MinerU2.5, DeepSeek-OCR |
| **Document-to-Markdown toolkits** | Orchestrate detection/OCR/table models into clean MD/JSON | Marker, MinerU, Docling, LlamaParse |
| **Sub-tasks** | Layout analysis, reading order, table recognition (TEDS), formula recognition, KIE/VQA | (cross-cutting across the above) |

## Key models

| Model | Org | What it does | Link |
|---|---|---|---|
| GOT-OCR2.0 | UCAS / StepFun | 580M unified end-to-end OCR-2.0: text, tables, charts, formulas, sheet music | https://github.com/Ucas-HaoranWei/GOT-OCR2.0 |
| DeepSeek-OCR | DeepSeek | Optical context compression; <100 vision tokens/page, beats GOT-OCR2.0 on OmniDocBench | https://github.com/deepseek-ai/DeepSeek-OCR |
| Surya | Datalab | OCR, layout, reading order, table recognition in 90+ languages | https://github.com/datalab-to/surya |
| Nougat | Meta AI | OCR-free academic PDF → Markdown + LaTeX | https://github.com/facebookresearch/nougat |
| Donut | NAVER Clova | OCR-free document understanding transformer (VQA/KIE) | https://github.com/clovaai/donut |
| TrOCR | Microsoft | Transformer encoder-decoder for printed/handwritten line OCR | https://huggingface.co/docs/transformers/model_doc/trocr |
| olmOCR | AllenAI | Toolkit to linearize PDFs into LLM training tokens; olmOCR-2 7B | https://github.com/allenai/olmocr |
| dots.ocr | Xiaohongshu (rednote-hilab) | 3B multilingual layout+OCR VLM, Markdown output | https://github.com/rednote-hilab/dots.ocr |
| PaddleOCR-VL | Baidu | Sub-1B VLM topping OmniDocBench on vendor scores | https://huggingface.co/PaddlePaddle/PaddleOCR-VL |
| LayoutLMv3 | Microsoft | Multimodal text+layout+image pretraining for Doc-AI | https://github.com/microsoft/unilm/tree/master/layoutlmv3 |

## Tools & toolkits

| Tool | Org | Role | Link |
|---|---|---|---|
| PaddleOCR | Baidu | Production OCR + PP-StructureV3 layout/table parsing | https://github.com/PaddlePaddle/PaddleOCR |
| MinerU | OpenDataLab | High-fidelity PDF/document → Markdown (MinerU2.5) | https://github.com/opendatalab/MinerU |
| Marker | Datalab | PDF/EPUB/docs → clean Markdown/JSON | https://github.com/datalab-to/marker |
| Docling | IBM | Document parsing + conversion for RAG pipelines | https://github.com/docling-project/docling |
| docTR | Mindee | Modular detection+recognition OCR (PyTorch/TF) | https://github.com/mindee/doctr |
| Tesseract | Google/OSS | Classic open-source OCR engine (100+ languages) | https://github.com/tesseract-ocr/tesseract |
| Mistral OCR | Mistral AI | API-first OCR for media/text/tables/equations | https://mistral.ai/news/mistral-ocr |

## Benchmarks & datasets

| Benchmark | Focus | Metrics | Link |
|---|---|---|---|
| OmniDocBench (CVPR 2025) | Diverse PDF parsing (9 doc types) | Edit distance, TEDS (tables), CDM (formulas) | https://github.com/opendatalab/OmniDocBench |
| olmOCR-Bench | Robustness of PDF parsing | Unit-test pass rate | https://github.com/allenai/olmocr |
| DocVQA | Document visual question answering | ANLS | https://www.docvqa.org/ |
| FUNSD | Form understanding / KIE | F1 | https://guillaumejaume.github.io/FUNSD/ |
| PubTabNet | Table recognition | TEDS | https://github.com/ibm-aur-nlp/PubTabNet |
| SROIE | Receipt OCR + info extraction | F1 | https://rrc.cvc.uab.es/?ch=13 |

## Key papers

- **GOT-OCR2.0: General OCR Theory — Towards OCR-2.0 via a Unified End-to-end Model** (2024) — https://arxiv.org/abs/2409.01704
- **DeepSeek-OCR: Contexts Optical Compression** (2025) — https://arxiv.org/abs/2510.18234
- **olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models** (2025) — https://arxiv.org/abs/2502.18443
- **Nougat: Neural Optical Understanding for Academic Documents** (2023) — https://arxiv.org/abs/2308.13418
- **OCR-free Document Understanding Transformer (Donut)** (2021) — https://arxiv.org/abs/2111.15664
- **TrOCR: Transformer-based OCR with Pre-trained Models** (2021) — https://arxiv.org/abs/2109.10282
- **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking** (2022) — https://arxiv.org/abs/2204.08387
- **UDOP: Unifying Vision, Text, and Layout for Universal Document Processing** (2022) — https://arxiv.org/abs/2212.02623
- **OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations** (2024) — https://arxiv.org/abs/2412.07626
- **PaddleOCR 3.0 Technical Report** (2025) — https://arxiv.org/abs/2507.05595

## Cross-references in AIForge

- [Multimodal Models](../Multimodal_Models/) — the VLM backbones (Qwen-VL, InternVL) that modern OCR models are built on
- [Vision Foundation Models](../Vision_Foundation_Models/) — image encoders underpinning document VLMs
- [Embedding Models](../Embedding_Models/) — turning parsed documents into vectors for RAG
- [Frameworks](../Frameworks/) — serving and pipeline frameworks for document AI

## Sources

- https://github.com/Ucas-HaoranWei/GOT-OCR2.0
- https://github.com/deepseek-ai/DeepSeek-OCR — paper: https://arxiv.org/abs/2510.18234
- https://github.com/datalab-to/surya
- https://github.com/allenai/olmocr — paper: https://arxiv.org/abs/2502.18443
- https://github.com/opendatalab/OmniDocBench — CVPR 2025
- https://github.com/opendatalab/MinerU
- https://huggingface.co/PaddlePaddle/PaddleOCR-VL — https://arxiv.org/abs/2507.05595
- https://arxiv.org/abs/2308.13418 (Nougat), https://arxiv.org/abs/2111.15664 (Donut), https://arxiv.org/abs/2109.10282 (TrOCR), https://arxiv.org/abs/2204.08387 (LayoutLMv3)

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
