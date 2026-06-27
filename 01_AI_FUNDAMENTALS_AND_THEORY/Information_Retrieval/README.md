# Information Retrieval

> Information Retrieval (IR) is the science of finding and ranking documents (or passages, images, code) relevant to a query from a large collection — spanning classical lexical scoring (TF-IDF, BM25), learning-to-rank, neural dense/sparse retrieval, approximate nearest-neighbor (ANN) indexing, and rank-aware evaluation (nDCG, MRR, Recall@k).

## Why it matters

IR is the foundational theory beneath every search engine, vector database, and Retrieval-Augmented Generation (RAG) pipeline: an LLM is only as good as the passages a retriever feeds it. Modern stacks combine a cheap first-stage retriever (BM25 or dense bi-encoder) over an inverted or ANN index with an expensive neural re-ranker (cross-encoder). Understanding relevance modeling, indexing trade-offs, and proper offline evaluation is what separates a demo from a production system that stays fast and accurate at billion-document scale.

## Taxonomy

| Sub-area | What it covers | Representative methods |
|---|---|---|
| **Lexical / sparse retrieval** | Exact term matching over inverted index | TF-IDF, BM25 (Okapi), query likelihood |
| **Learned sparse retrieval** | Neural term weighting + expansion, inverted-index friendly | SPLADE, DeepImpact, uniCOIL, docT5query |
| **Dense retrieval (bi-encoder)** | Independent query/doc embeddings, cosine/dot similarity | DPR, ANCE, Sentence-BERT, E5, BGE |
| **Late-interaction** | Token-level multi-vector matching | ColBERT, ColBERTv2 |
| **Re-ranking (cross-encoder)** | Joint query-doc scoring of a candidate shortlist | monoBERT, monoT5, RankT5, cross-encoders |
| **Learning to Rank (LTR)** | Supervised ranking over hand/feature signals | RankNet, LambdaRank, LambdaMART |
| **ANN indexing** | Sublinear vector search | HNSW, IVF-PQ, Faiss, ScaNN |
| **Generative retrieval** | Decode document identifiers autoregressively | DSI, differentiable search index |
| **Evaluation** | Rank-aware quality metrics | nDCG, MRR, MAP, Recall@k, BEIR/MTEB |

## Key methods & models

| Method | Type | Link |
|---|---|---|
| BM25 / Okapi (Probabilistic Relevance Framework) | Lexical | https://en.wikipedia.org/wiki/Okapi_BM25 |
| DPR — Dense Passage Retrieval | Dense bi-encoder | https://arxiv.org/abs/2004.04906 |
| Sentence-BERT (SBERT) | Dense bi-encoder | https://arxiv.org/abs/1908.10084 |
| ColBERT — late interaction | Multi-vector | https://arxiv.org/abs/2004.12832 |
| ColBERTv2 | Multi-vector | https://arxiv.org/abs/2112.01488 |
| SPLADE / SPLADE v2 | Learned sparse | https://arxiv.org/abs/2109.10086 |
| docT5query (doc expansion) | Learned sparse | https://arxiv.org/abs/2003.06713 |
| monoT5 — seq2seq re-ranker | Cross-encoder | https://arxiv.org/abs/2003.06713 |
| LambdaMART (RankNet→LambdaRank→LambdaMART) | Learning to Rank | https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/ |
| HNSW — graph ANN index | Indexing | https://arxiv.org/abs/1603.09320 |

## Key tools & frameworks

| Tool | Role | Link |
|---|---|---|
| Pyserini | Reproducible IR (BM25 + dense, Anserini/Lucene) | https://github.com/castorini/pyserini |
| Faiss | Billion-scale dense ANN search (IVF-PQ, HNSW) | https://github.com/facebookresearch/faiss |
| SPLADE | Official learned-sparse implementation | https://github.com/naver/splade |
| Sentence-Transformers | Bi-/cross-encoder training & inference | https://github.com/UKPLab/sentence-transformers |
| ColBERT | Late-interaction retrieval engine | https://github.com/stanford-futuredata/ColBERT |
| BEIR | Zero-shot heterogeneous IR benchmark suite | https://github.com/beir-cellar/beir |
| MTEB | Massive Text Embedding Benchmark + leaderboard | https://github.com/embeddings-benchmark/mteb |
| Elasticsearch / OpenSearch | Production BM25 + vector hybrid search | https://github.com/elastic/elasticsearch |

## Benchmarks & datasets

| Benchmark | Focus | Link |
|---|---|---|
| MS MARCO | 8.8M-passage ranking, 1M+ Bing queries | https://github.com/microsoft/MSMARCO-Passage-Ranking |
| BEIR | 18-task zero-shot retrieval transfer | https://arxiv.org/abs/2104.08663 |
| MTEB | 8 task types, 100+ languages, embedding leaderboard | https://arxiv.org/abs/2210.07316 |
| TREC Deep Learning | Annual passage/document ranking track | https://microsoft.github.io/msmarco/TREC-Deep-Learning |
| mMARCO | Multilingual (13 languages) MS MARCO | https://arxiv.org/abs/2108.13897 |
| Natural Questions / TriviaQA | Open-domain QA retrieval | https://ai.google.com/research/NaturalQuestions |

**Core metrics:** nDCG@k (graded relevance, discounted by rank), MRR (reciprocal rank of first hit), MAP, Recall@k, and latency/throughput at index scale.

## Key papers

| Paper | Year | Link |
|---|---|---|
| The Probabilistic Relevance Framework: BM25 and Beyond (Robertson & Zaragoza) | 2009 | https://dl.acm.org/doi/10.1561/1500000019 |
| Dense Passage Retrieval for Open-Domain QA (Karpukhin et al.) | 2020 | https://arxiv.org/abs/2004.04906 |
| ColBERT: Efficient & Effective Passage Search via Late Interaction (Khattab & Zaharia) | 2020 | https://arxiv.org/abs/2004.12832 |
| ColBERTv2: Effective & Efficient Retrieval via Lightweight Late Interaction | 2021 | https://arxiv.org/abs/2112.01488 |
| SPLADE v2: Sparse Lexical and Expansion Model for IR (Formal et al.) | 2021 | https://arxiv.org/abs/2109.10086 |
| Pretrained Transformers for Text Ranking: BERT and Beyond (Lin et al.) | 2020 | https://arxiv.org/abs/2010.06467 |
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych) | 2019 | https://arxiv.org/abs/1908.10084 |
| BEIR: Heterogeneous Benchmark for Zero-shot IR (Thakur et al.) | 2021 | https://arxiv.org/abs/2104.08663 |
| MTEB: Massive Text Embedding Benchmark (Muennighoff et al.) | 2022 | https://arxiv.org/abs/2210.07316 |
| Efficient & Robust ANN Search via HNSW Graphs (Malkov & Yashunin) | 2016 | https://arxiv.org/abs/1603.09320 |

## Cross-references in AIForge

- [RAG and Retrieval](../RAG_and_Retrieval/) — application-level retrieval-augmented generation built on these IR primitives
- [Natural Language Processing](../Natural_Language_Processing/) — text representation, tokenization, and transformer encoders
- [Recommender Systems](../Recommender_Systems/) — ranking and candidate generation that share LTR and ANN machinery
- [Knowledge_Graphs](../Knowledge_Graphs/) — structured retrieval and entity linking complementary to dense search

## Sources

- https://en.wikipedia.org/wiki/Okapi_BM25
- https://dl.acm.org/doi/10.1561/1500000019
- https://arxiv.org/abs/2004.04906
- https://arxiv.org/abs/2004.12832
- https://arxiv.org/abs/2112.01488
- https://arxiv.org/abs/2109.10086
- https://arxiv.org/abs/2010.06467
- https://arxiv.org/abs/1908.10084
- https://arxiv.org/abs/2104.08663
- https://arxiv.org/abs/2210.07316
- https://arxiv.org/abs/1603.09320
- https://arxiv.org/abs/2108.13897
- https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- https://github.com/castorini/pyserini
- https://github.com/facebookresearch/faiss
- https://github.com/naver/splade
- https://github.com/embeddings-benchmark/mteb
- https://github.com/beir-cellar/beir
- https://github.com/microsoft/MSMARCO-Passage-Ranking

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
