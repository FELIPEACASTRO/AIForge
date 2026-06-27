# RAG and Retrieval-Augmented Generation

Resources for **RAG (Retrieval-Augmented Generation)**, **hybrid search**, **agentic RAG**, **GraphRAG**, **multimodal retrieval**, and **production RAG architectures**.

## Architectures

| Pattern | Description | Reference |
|---|---|---|
| **Naive RAG** | Embed → retrieve → stuff context | Lewis et al. 2020 — https://arxiv.org/abs/2005.11401 |
| **HyDE** | Hypothetical doc embedding for retrieval | Gao et al. 2022 — https://arxiv.org/abs/2212.10496 |
| **Self-RAG** | Self-reflection on retrieval need | Asai et al. 2023 — https://arxiv.org/abs/2310.11511 |
| **CRAG (Corrective RAG)** | Detect and correct bad retrievals | Yan et al. 2024 — https://arxiv.org/abs/2401.15884 |
| **GraphRAG** | Knowledge-graph-aware retrieval | Microsoft Research 2024 — https://arxiv.org/abs/2404.16130 |
| **Agentic RAG** | Agent loops over retrievers and tools | https://arxiv.org/abs/2501.09136 |
| **ColBERT / Late Interaction** | Token-level interaction for high-precision retrieval | Khattab & Zaharia 2020 — https://arxiv.org/abs/2004.12832 |
| **RAPTOR** | Recursive abstractive tree retrieval | Sarthi et al. 2024 — https://arxiv.org/abs/2401.18059 |
| **Long RAG** | Use long-context LLMs over coarse chunks | https://arxiv.org/abs/2406.15319 |
| **Contextual Retrieval** | Per-chunk context with Claude prompt caching | https://www.anthropic.com/news/contextual-retrieval |

## Embedding Models

- **OpenAI text-embedding-3-large/small** — https://platform.openai.com/docs/guides/embeddings
- **Cohere Embed v3** — https://cohere.com/embed
- **Voyage AI** (recommended by Anthropic) — https://www.voyageai.com/
- **BGE family (BAAI)** — https://huggingface.co/BAAI
- **Nomic Embed** — https://www.nomic.ai/
- **Jina Embeddings v3** — https://jina.ai/embeddings/
- **Mistral Embed** — https://docs.mistral.ai/capabilities/embeddings/
- **ColPali (visual document retrieval)** — https://huggingface.co/vidore/colpali

## Vector Databases

See `03_DATASETS_TOOLS_AND_RESOURCES/Storage_and_Databases/Vector_Databases` for the catalog. Highlights: Pinecone, Weaviate, Qdrant, Milvus, Chroma, LanceDB, pgvector, Vespa, Turbopuffer.

## Frameworks

- **LangChain** — https://www.langchain.com/
- **LlamaIndex** — https://www.llamaindex.ai/
- **Haystack (deepset)** — https://haystack.deepset.ai/
- **Verba (Weaviate)** — https://github.com/weaviate/Verba
- **txtai** — https://neuml.github.io/txtai/
- **R2R** — https://r2r-docs.sciphi.ai/
- **RAGFlow** — https://github.com/infiniflow/ragflow

## Re-Ranking

- **Cohere Rerank v3** — https://cohere.com/rerank
- **Voyage Rerank** — https://docs.voyageai.com/docs/reranker
- **Jina Reranker** — https://jina.ai/reranker/
- **bge-reranker (BAAI)** — https://huggingface.co/BAAI/bge-reranker-large
- **MS MARCO Cross-Encoders** — https://www.sbert.net/

## Evaluation

- **RAGAS** — https://docs.ragas.io/
- **TruLens** — https://www.trulens.org/
- **Phoenix Evals** — https://docs.arize.com/phoenix
- **DeepEval** — https://docs.confident-ai.com/

## Surveys

- **Retrieval-Augmented Generation for LLMs: A Survey** — Gao et al. 2023 — https://arxiv.org/abs/2312.10997
- **A Survey on RAG Meets LLMs** — Fan et al. 2024 — https://arxiv.org/abs/2405.06211
