# Embedding Infrastructure

> The engine layer beneath every RAG and semantic-search system: the models that turn text/images into dense vectors, the serving stacks that batch and accelerate that inference, and the ANN (approximate nearest neighbor) index libraries (FAISS, hnswlib, ScaNN, DiskANN) that retrieve them at scale — distinct from the managed vector *databases* covered elsewhere in this section.

## Why it matters

Managed vector DBs (Pinecone, Weaviate, Milvus, Qdrant) expose a query API, but the recall, latency, and cost of a retrieval system are set by two layers underneath: the **embedding model** (quality of the vector) and the **ANN index** (speed/recall tradeoff of the search). FAISS, hnswlib, and ScaNN are the kernels those databases embed internally, and high-throughput embedding servers (TEI, Infinity, BEI, vLLM) are what keep GPUs saturated during bulk indexing and online query encoding. Choosing the index family (HNSW vs IVF vs DiskANN), the quantization scheme (PQ/ScaNN/RaBitQ), and the embedding dimensionality (see Matryoshka) is the core engineering of production retrieval.

## Taxonomy

| Layer | What it does | Representative tools |
|---|---|---|
| **Embedding models** | Map text/image → dense vector | BGE, E5, GTE, Nomic, NV-Embed, Qwen3-Embedding, OpenAI/Cohere/Gemini APIs |
| **Embedding serving** | High-throughput inference, dynamic batching, OpenAI-compatible API | sentence-transformers, TEI, Infinity, BEI, vLLM, SGLang |
| **ANN index libraries** | In-process nearest-neighbor search structures | FAISS, hnswlib, ScaNN, DiskANN, Annoy, NMSLIB, Vespa, Voyager |
| **Index algorithms** | Graph / inverted-list / quantization methods | HNSW, NSG, IVF, IVF-PQ, ScaNN (anisotropic VQ), DiskANN/Vamana, SPANN |
| **Compression** | Shrink memory & dimensionality | PQ, OPQ, RaBitQ, scalar/binary quantization, Matryoshka (MRL) |
| **Benchmarks** | Compare recall/QPS and embedding quality | ann-benchmarks, Big-ANN, VIBE, MTEB |

## Key ANN index libraries

| Library | Algorithms | Notes | Link |
|---|---|---|---|
| FAISS | IVF, IVF-PQ, HNSW, ScaNN, flat | Meta's reference C++/Python lib; GPU support; backs Milvus & others | https://github.com/facebookresearch/faiss |
| hnswlib | HNSW | Header-only C++/Python; latency-optimal in-memory graph | https://github.com/nmslib/hnswlib |
| ScaNN | Anisotropic VQ + partitioning | Google's MIPS-optimized library; top of many leaderboards | https://github.com/google-research/google-research/tree/master/scann |
| DiskANN | Vamana (SSD-resident graph) | Microsoft; billion-scale on commodity SSDs | https://github.com/microsoft/DiskANN |
| Annoy | Random projection trees | Spotify; mmap'd, simple, static index | https://github.com/spotify/annoy |
| NMSLIB | HNSW, SW-graph, others | Research-grade ANN library (HNSW origin) | https://github.com/nmslib/nmslib |
| Voyager | HNSW | Spotify's production HNSW (Python/Java) | https://github.com/spotify/voyager |
| USearch | HNSW (multi-metric) | Compact, many-language bindings, on-disk | https://github.com/unum-cloud/usearch |

## Key embedding-serving frameworks

| Tool | Stack | Highlights | Link |
|---|---|---|---|
| sentence-transformers | PyTorch | De-facto library for encoding & training embeddings | https://github.com/UKPLab/sentence-transformers |
| Text Embeddings Inference (TEI) | Rust | Dynamic batching, safetensors, Prometheus/OTel metrics | https://github.com/huggingface/text-embeddings-inference |
| Infinity | PyTorch / ONNX / TensorRT / CTranslate2 | OpenAI-compatible REST; text, rerank, CLIP, ColPali | https://github.com/michaelfeil/infinity |
| FlagEmbedding / BGE | PyTorch | BAAI model family + fine-tuning toolkit | https://github.com/FlagOpen/FlagEmbedding |
| vLLM (embeddings) | PyTorch / CUDA | Pooling/embedding task mode, OpenAI `/embeddings` | https://github.com/vllm-project/vllm |
| Nomic embed | PyTorch | Open weights + data + training code | https://github.com/nomic-ai/contrastors |

## Key embedding models

| Model | Notes | Link |
|---|---|---|
| BGE-M3 | Multilingual, multi-granularity (dense+sparse+ColBERT) | https://huggingface.co/BAAI/bge-m3 |
| E5 (multilingual-e5-large) | Weakly-supervised contrastive pretraining | https://huggingface.co/intfloat/multilingual-e5-large |
| GTE | General Text Embeddings (Alibaba) | https://huggingface.co/thenlper/gte-large |
| Nomic Embed v1.5 | Matryoshka, 8192-token context, open data | https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 |
| NV-Embed-v2 | Llama-based decoder embedder, top MTEB scores | https://huggingface.co/nvidia/NV-Embed-v2 |
| Qwen3-Embedding | Strong open multilingual retriever family | https://huggingface.co/Qwen/Qwen3-Embedding-8B |
| Jina embeddings v3 | Long-context, task-LoRA | https://huggingface.co/jinaai/jina-embeddings-v3 |

## Benchmarks

| Benchmark | Measures | Link |
|---|---|---|
| MTEB | Embedding quality across retrieval/STS/classification (+ leaderboard) | https://huggingface.co/spaces/mteb/leaderboard |
| ann-benchmarks | Recall vs QPS across ANN libraries on standard datasets | https://github.com/erikbern/ann-benchmarks |
| Big-ANN (NeurIPS'23) | Billion-scale & filtered/streaming ANN tracks | https://big-ann-benchmarks.com/neurips23.html |
| VIBE | Vector Index Benchmark for Embeddings (realistic workloads) | https://arxiv.org/abs/2505.17810 |

## Key papers

| Paper | Year | Link |
|---|---|---|
| Efficient and robust ANN search using HNSW graphs (Malkov & Yashunin) | 2016 | https://arxiv.org/abs/1603.09320 |
| Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN) | 2019 | https://arxiv.org/abs/1908.10396 |
| The Faiss Library (Douze et al.) | 2024 | https://arxiv.org/abs/2401.08281 |
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | 2019 | https://arxiv.org/abs/1908.10084 |
| Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5) | 2022 | https://arxiv.org/abs/2212.03533 |
| MTEB: Massive Text Embedding Benchmark | 2022 | https://arxiv.org/abs/2210.07316 |
| Matryoshka Representation Learning (MRL) | 2022 | https://arxiv.org/abs/2205.13147 |
| SPANN: Highly-efficient Billion-scale ANN Search | 2021 | https://arxiv.org/abs/2111.08566 |
| Results of the Big ANN: NeurIPS'23 competition | 2024 | https://arxiv.org/abs/2409.17424 |

## Cross-references in AIForge

- [Vector Databases](../) — managed/server vector DBs (Pinecone, Weaviate, Milvus, Qdrant) that embed these index kernels.
- [Storage and Databases](../../) — broader storage pillar (object stores, feature stores, OLAP).
- [Datasets, Tools and Resources](../../../) — parent pillar index.

## Sources

- https://github.com/facebookresearch/faiss
- https://github.com/nmslib/hnswlib
- https://github.com/google-research/google-research/tree/master/scann
- https://github.com/microsoft/DiskANN
- https://github.com/huggingface/text-embeddings-inference
- https://github.com/michaelfeil/infinity
- https://github.com/erikbern/ann-benchmarks
- https://huggingface.co/spaces/mteb/leaderboard
- https://arxiv.org/abs/1603.09320
- https://arxiv.org/abs/1908.10396
- https://arxiv.org/abs/2401.08281
- https://arxiv.org/abs/2210.07316
- https://arxiv.org/abs/2205.13147
- https://arxiv.org/abs/2505.17810

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
