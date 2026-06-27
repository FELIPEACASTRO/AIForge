# Reranker Models

> Rerankers are the second stage of modern retrieval pipelines: they re-score a candidate set from a fast first-stage retriever (BM25 / bi-encoder) using deeper query–document interaction, trading latency for large precision gains in RAG and search.

## Why it matters

First-stage retrievers (sparse BM25 or dense bi-encoders) encode queries and documents independently, so they miss fine-grained term-level interactions. Rerankers re-score the top-k candidates with full cross-attention (cross-encoders), multi-vector late interaction (ColBERT), or LLM listwise reasoning, typically adding **+5 to +15 nDCG@10** on BEIR/MTEB at the cost of higher latency. Because they only run on a small candidate set (e.g. top-100), the cost is bounded, making rerankers the highest-ROI upgrade for most RAG and search stacks.

## Taxonomy

| Approach | How it works | Latency | Examples |
|---|---|---|---|
| **Cross-encoder (pointwise)** | Query+doc concatenated into one transformer; single relevance score per pair | Medium (full attention per pair) | BGE-reranker, mxbai-rerank, Jina, MiniLM cross-encoders |
| **Late interaction (multi-vector)** | Encode query/doc into token embeddings; MaxSim scoring at query time | Low–Medium (precomputable doc vectors) | ColBERTv2, PLAID, Jina-ColBERT |
| **Seq2seq (generative pointwise)** | Encoder-decoder generates "true/false" or ranking token; logits = relevance | Medium | monoT5, RankT5 |
| **LLM listwise** | LLM is shown the query + a list of docs and outputs a permutation | High (long prompts, multiple passes) | RankGPT, RankZephyr, RankVicuna |
| **LLM pairwise** | LLM compares two docs at a time; aggregate to a ranking | Very high (O(n²) comparisons) | PRP (Pairwise Ranking Prompting) |

## Key models

### Open-weight cross-encoders & late interaction

| Model | Params / type | Notes | Link |
|---|---|---|---|
| BAAI bge-reranker-v2-m3 | ~0.6B, cross-encoder | Multilingual, long context, strong default open reranker | https://huggingface.co/BAAI/bge-reranker-v2-m3 |
| BAAI bge-reranker-v2-gemma | Gemma-2B backbone | LLM-based reranker, higher accuracy | https://huggingface.co/BAAI/bge-reranker-v2-gemma |
| mixedbread mxbai-rerank-large-v2 | 1.5B, Apache-2.0 | Beats Cohere/Voyage on BEIR per vendor; base-v2 is 0.5B | https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2 |
| Jina Reranker v2 (multilingual) | cross-encoder | 100+ languages, function-calling/code aware | https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual |
| Jina Reranker v3 | listwise, "last-not-late" | ~0.6B, sub-200ms latency tier | https://arxiv.org/abs/2509.25085 |
| ColBERTv2 | late interaction | Stanford multi-vector retriever/reranker | https://github.com/stanford-futuredata/ColBERT |
| MS MARCO MiniLM cross-encoders | 6–12 layer | Lightweight, fast baseline rerankers | https://sbert.net/docs/cross_encoder/pretrained_models.html |
| RankZephyr (7B) | listwise LLM | Zero-shot, rivals GPT-4 listwise reranking | https://huggingface.co/castorini/rank_zephyr_7b_v1_full |

### Hosted / API rerankers

| Service | Notes | Link |
|---|---|---|
| Cohere Rerank 3.5 | 100+ languages, 4096-token chunks, strong default API | https://docs.cohere.com/docs/rerank-overview |
| Voyage rerank-2 / rerank-2.5 | Low-latency hosted rerankers | https://docs.voyageai.com/docs/reranker |
| Jina Reranker API | Hosted v2/v3 endpoints | https://jina.ai/reranker/ |
| Mixedbread Rerank API | Hosted mxbai-rerank-v2 | https://www.mixedbread.com/api-reference/endpoints/reranking |

### Frameworks & toolkits

| Tool | Purpose | Link |
|---|---|---|
| Sentence-Transformers CrossEncoder | Train/infer cross-encoder rerankers | https://sbert.net/docs/cross_encoder/pretrained_models.html |
| RankLLM (castorini) | Listwise LLM reranking (RankZephyr/Vicuna/GPT) | https://github.com/castorini/rank_llm |
| FlagEmbedding (BAAI) | BGE reranker training & inference | https://github.com/FlagOpen/FlagEmbedding |
| PyGaggle (castorini) | monoT5/duoT5 neural ranking | https://github.com/castorini/pygaggle |
| rerankers (AnswerDotAI) | Unified Python API across rerankers | https://github.com/AnswerDotAI/rerankers |

## Benchmarks & datasets

| Benchmark | Focus | Link |
|---|---|---|
| BEIR | Zero-shot retrieval/rerank across 18 datasets; nDCG@10 standard | https://github.com/beir-cellar/beir |
| MTEB | Massive embedding/reranking leaderboard | https://huggingface.co/spaces/mteb/leaderboard |
| MS MARCO | Passage ranking training & dev | https://microsoft.github.io/msmarco/ |
| MIRACL | Multilingual retrieval (18 languages) | https://github.com/project-miracl/miracl |
| TREC Deep Learning | Annual passage/document ranking tracks | https://microsoft.github.io/msmarco/TREC-Deep-Learning |

**Common metrics:** nDCG@10 (primary), MRR@10, MAP, Recall@k. Reranking lift is measured as the delta over the first-stage retriever on the same candidate pool.

## Key papers

| Paper | Year | Link |
|---|---|---|
| Document Ranking with a Pretrained Seq2Seq Model (monoT5) | 2020 | https://arxiv.org/abs/2003.06713 |
| ColBERT: Efficient & Effective Passage Search via Late Interaction | 2020 | https://arxiv.org/abs/2004.12832 |
| ColBERTv2: Effective & Efficient Retrieval via Lightweight Late Interaction | 2021 | https://arxiv.org/abs/2112.01488 |
| RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses | 2022 | https://arxiv.org/abs/2210.10634 |
| Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents (RankGPT) | 2023 | https://arxiv.org/abs/2304.09542 |
| RankVicuna: Zero-Shot Listwise Reranking with Open-Source LLMs | 2023 | https://arxiv.org/abs/2309.15088 |
| RankZephyr: Effective and Robust Zero-Shot Listwise Reranking | 2023 | https://arxiv.org/abs/2312.02724 |
| Pairwise Ranking Prompting (PRP) for Effective LLM Ranking | 2023 | https://arxiv.org/abs/2306.17563 |
| jina-reranker-v3: Last but Not Late Interaction for Listwise Reranking | 2025 | https://arxiv.org/abs/2509.25085 |
| Pretrained Transformers for Text Ranking: BERT and Beyond (survey) | 2020 | https://arxiv.org/abs/2010.06467 |

## Cross-references in AIForge

- [Embedding Models](../Embedding_Models/) — first-stage bi-encoders that rerankers refine
- [Text LLMs](../Text_LLMs/) — backbones for LLM-based listwise rerankers
- [Reasoning Models](../Reasoning_Models/) — reasoning-driven listwise ranking
- [Frameworks](../Frameworks/) — RAG/retrieval orchestration that hosts rerankers

## Sources

- https://huggingface.co/BAAI/bge-reranker-v2-m3
- https://github.com/stanford-futuredata/ColBERT
- https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2
- https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
- https://docs.cohere.com/docs/rerank-overview
- https://github.com/castorini/rank_llm
- https://sbert.net/docs/cross_encoder/pretrained_models.html
- https://huggingface.co/blog/train-reranker
- https://arxiv.org/abs/2003.06713
- https://arxiv.org/abs/2112.01488
- https://arxiv.org/abs/2210.10634
- https://arxiv.org/abs/2312.02724
- https://arxiv.org/abs/2509.25085
- https://github.com/beir-cellar/beir

_Seed section — curated from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
