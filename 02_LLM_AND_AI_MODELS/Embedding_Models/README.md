# Embedding Models

> Embedding models map text (and other modalities) into dense vectors so that semantic similarity becomes geometric proximity — the foundation of RAG, semantic search, clustering, classification, and vector databases.

## Why it matters

Embeddings are arguably the most-deployed model family in production AI: every RAG pipeline, semantic search box, deduplication job, and recommendation system depends on them. A small change in the embedding model often moves end-to-end retrieval quality more than swapping the generator LLM. The field moved fast from BERT-era bi-encoders (Sentence-BERT) to weakly-supervised contrastive encoders (E5, GTE, BGE) and now to decoder-LLM-based generalist embedders (NV-Embed, e5-mistral, Qwen3-Embedding) that top the MTEB leaderboard.

## Taxonomy

| Approach | Idea | Representative models |
|---|---|---|
| Bi-encoder (dense, single-vector) | Encode query and doc independently; score by cosine/dot. Fast, ANN-indexable. | Sentence-BERT, E5, GTE, BGE, Nomic, Jina v3 |
| LLM-based generalist | Use a decoder LLM backbone + contrastive fine-tuning; instruction-aware. | NV-Embed, e5-mistral-7b, gte-Qwen2, Qwen3-Embedding |
| Multi-vector / late interaction | One vector per token; token-level MaxSim scoring. Higher quality, larger index. | ColBERT, ColBERTv2, Jina-ColBERT-v2 |
| Sparse / learned-lexical | Produce term-weighted sparse vectors (BM25-like but learned). | SPLADE, BGE-M3 (sparse mode) |
| Multilingual / multi-functional | Single model spanning many languages and dense+sparse+multi-vec output. | BGE-M3, multilingual-e5, Arctic-Embed 2.0, Jina v3 |
| Adaptive-dimension (Matryoshka) | Nested embeddings truncatable to smaller dims with graceful degradation. | Nomic v1.5, Jina v3, OpenAI text-embedding-3, MRL |
| Proprietary API | Closed-weight hosted embedders. | OpenAI text-embedding-3, Voyage, Cohere Embed |

## Key models

| Model | Type | Notes | Link |
|---|---|---|---|
| BGE / FlagEmbedding (BAAI) | Bi-encoder family | BGE-base/large, BGE-M3 (dense+sparse+ColBERT), BGE-en-ICL (in-context) | https://github.com/FlagOpen/FlagEmbedding |
| E5 (Microsoft) | Weakly-supervised contrastive | e5-base/large, multilingual-e5, e5-mistral-7b-instruct | https://huggingface.co/intfloat/e5-large-v2 |
| GTE (Alibaba) | Multi-stage contrastive | gte-large, gte-Qwen2-7B-instruct | https://huggingface.co/thenlper/gte-large |
| Qwen3-Embedding | LLM-based, flexible dims | 0.6B/4B/8B; ranked #1 MTEB multilingual at release (Jun 2025) | https://github.com/QwenLM/Qwen3-Embedding |
| NV-Embed (NVIDIA) | LLM-based generalist | NV-Embed-v2, latent-attention pooling; top English MTEB | https://huggingface.co/nvidia/NV-Embed-v2 |
| nomic-embed-text-v1.5 | Open, long-context, Matryoshka | 8192 ctx, fully open data/weights | https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 |
| Jina Embeddings v3 | Multilingual + task LoRA | 570M, 8192 ctx, Matryoshka dims to 32 | https://huggingface.co/jinaai/jina-embeddings-v3 |
| Arctic-Embed 2.0 (Snowflake) | Multilingual retrieval | Strong multilingual without English regression | https://arxiv.org/abs/2412.04506 |
| OpenAI text-embedding-3 | Proprietary API | small/large, Matryoshka-style dim control | https://platform.openai.com/docs/guides/embeddings |
| Voyage AI | Proprietary API | voyage-3 family, code/finance/law variants | https://docs.voyageai.com/docs/embeddings |
| Cohere Embed | Proprietary API | Embed v3/v4, multilingual, int8/binary | https://docs.cohere.com/docs/embeddings |

## Tools & frameworks

| Tool | Use | Link |
|---|---|---|
| sentence-transformers | Train/serve bi-encoders & cross-encoders | https://github.com/UKPLab/sentence-transformers |
| MTEB | Evaluate embeddings across tasks/languages | https://github.com/embeddings-benchmark/mteb |
| FlagEmbedding | BGE training & fine-tuning recipes | https://github.com/FlagOpen/FlagEmbedding |
| Text Embeddings Inference (TEI) | High-throughput embedding serving (HF) | https://github.com/huggingface/text-embeddings-inference |
| RAGatouille / ColBERT | Late-interaction retrieval | https://github.com/stanford-futuredata/ColBERT |

## Benchmarks

| Benchmark | Scope | Link |
|---|---|---|
| MTEB | 8 task types, 58+ datasets, 112+ languages; the standard leaderboard | https://huggingface.co/spaces/mteb/leaderboard |
| MMTEB | Massively multilingual extension of MTEB | https://github.com/embeddings-benchmark/mteb |
| BEIR | Zero-shot retrieval across 18 datasets | https://github.com/beir-cellar/beir |
| LoCo | Long-context retrieval (used by Nomic) | https://arxiv.org/abs/2402.07440 |

## Key papers

| Paper | arXiv | Why it matters |
|---|---|---|
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (2019) | https://arxiv.org/abs/1908.10084 | Foundational bi-encoder; usable cosine similarity |
| Matryoshka Representation Learning (2022) | https://arxiv.org/abs/2205.13147 | Nested, truncatable embeddings |
| MTEB: Massive Text Embedding Benchmark (2022) | https://arxiv.org/abs/2210.07316 | The evaluation standard |
| E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training (2022) | https://arxiv.org/abs/2212.03533 | Scalable contrastive pre-training |
| GTE: Towards General Text Embeddings with Multi-stage Contrastive Learning (2023) | https://arxiv.org/abs/2308.03281 | Multi-stage contrastive recipe |
| C-Pack / BGE: Packed Resources for General Chinese Embeddings (2023) | https://arxiv.org/abs/2309.07597 | BGE training recipe & resources |
| Nomic Embed: Training a Reproducible Long Context Text Embedder (2024) | https://arxiv.org/abs/2402.01613 | Fully open, 8192-ctx, beats Ada-002 |
| BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity (2024) | https://arxiv.org/abs/2402.03216 | Dense + sparse + multi-vector in one model |
| NV-Embed: Improved Techniques for Training LLMs as Generalist Embedders (2024) | https://arxiv.org/abs/2405.17428 | Latent-attention pooling; LLM-based SOTA |
| jina-embeddings-v3: Multilingual Embeddings With Task LoRA (2024) | https://arxiv.org/abs/2409.10173 | Task-specific LoRA adapters |
| Qwen3 Embedding: Advancing Text Embedding and Reranking (2025) | https://arxiv.org/abs/2506.05176 | #1 MTEB multilingual at release |
| ColBERTv2: Effective and Efficient Retrieval via Late Interaction (2021) | https://arxiv.org/abs/2112.01488 | Multi-vector late-interaction retrieval |

## Cross-references in AIForge

- [Reranker Models](../Reranker_Models/) — cross-encoders that re-score the top-k retrieved by embedding search.
- [Frameworks](../Frameworks/) — RAG and vector-store integrations that consume embeddings.
- [Text LLMs](../Text_LLMs/) — generator backbones; LLM-based embedders reuse these architectures.
- [Multimodal Models](../Multimodal_Models/) — image/audio/video embedding spaces.

## Sources

- https://github.com/FlagOpen/FlagEmbedding
- https://github.com/embeddings-benchmark/mteb
- https://huggingface.co/spaces/mteb/leaderboard
- https://qwenlm.github.io/blog/qwen3-embedding/
- https://github.com/QwenLM/Qwen3-Embedding
- https://huggingface.co/nvidia/NV-Embed-v2
- https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/
- https://arxiv.org/abs/2210.07316
- https://arxiv.org/abs/2402.01613
- https://arxiv.org/abs/2405.17428
- https://arxiv.org/abs/2409.10173
- https://arxiv.org/abs/2506.05176
- https://arxiv.org/abs/2402.03216
- https://arxiv.org/abs/2205.13147
- https://arxiv.org/abs/2412.04506
