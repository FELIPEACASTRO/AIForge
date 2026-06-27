# Recommender Systems

> Recommender systems predict the items (products, content, ads) a user is most likely to engage with, ranking a massive catalog from sparse interaction signals — the workhorse of personalization at industrial scale.

## Why it matters

Recommendation drives a large share of engagement and revenue on every major platform (search, feed, e-commerce, streaming, ads), making it one of the highest-impact applied-ML areas. The field spans classical matrix factorization, implicit-feedback ranking (BPR), graph and sequential models, deep CTR networks (Wide&Deep, DeepFM, DCN, DLRM), and two-tower retrieval that serves billions of items under tight latency budgets. Modern stacks are typically two-stage — cheap candidate **retrieval** followed by heavy **ranking** — and increasingly fuse LLMs for semantic and generative recommendation.

## Taxonomy

| Family | Idea | Representative methods |
|---|---|---|
| Collaborative filtering (CF) | Learn from user–item interactions only | ALS / WMF, SVD++, NeuMF |
| Implicit-feedback ranking | Optimize pairwise/top-K order from clicks | BPR, WARP, sampled softmax |
| Content / hybrid | Mix side features with CF | LightFM, Wide&Deep |
| Deep CTR (ranking) | Model high-order feature interactions | DeepFM, DCN-v2, DLRM |
| Graph-based CF | Propagate signal over the user–item graph | NGCF, LightGCN, PinSage |
| Sequential / session | Model the order of past interactions | GRU4Rec, SASRec, BERT4Rec |
| Two-tower retrieval | Dual encoders + ANN for fast candidate gen | YouTube DNN, sampling-bias-corrected two-tower |
| LLM / generative rec | Semantic IDs, generative retrieval, LLM rankers | TIGER, P5, LLM-as-ranker |

## Key models

| Model | Type | Link |
|---|---|---|
| BPR — Bayesian Personalized Ranking | Implicit ranking | https://arxiv.org/abs/1205.2618 |
| Neural Collaborative Filtering (NeuMF) | CF | https://arxiv.org/abs/1708.05031 |
| Wide & Deep | Deep CTR | https://arxiv.org/abs/1606.07792 |
| DeepFM | Deep CTR | https://arxiv.org/abs/1804.04950 |
| DCN-v2 (Deep & Cross Network) | Deep CTR | https://arxiv.org/abs/2008.13535 |
| DLRM | Deep CTR | https://arxiv.org/abs/1906.00091 |
| LightGCN | Graph CF | https://arxiv.org/abs/2002.02126 |
| GRU4Rec (Top-k gains) | Sequential | https://arxiv.org/abs/1706.03847 |
| SASRec | Sequential | https://arxiv.org/abs/1808.09781 |
| BERT4Rec | Sequential | https://arxiv.org/abs/1904.06690 |

## Frameworks & tools

| Tool | What it's for | Link |
|---|---|---|
| Microsoft Recommenders | Best-practice examples & utilities | https://github.com/recommenders-team/recommenders |
| RecBole | Unified library, 90+ algorithms, 40+ datasets | https://github.com/RUCAIBox/RecBole |
| TorchRec | Meta's PyTorch library for sharded embeddings at scale | https://github.com/meta-pytorch/torchrec |
| NVIDIA Merlin | GPU end-to-end RecSys (ETL, training, serving) | https://github.com/NVIDIA-Merlin/Merlin |
| LightFM | Hybrid CF + content learning-to-rank | https://github.com/lyst/lightfm |
| Implicit | Fast ALS / BPR for implicit feedback | https://github.com/benfred/implicit |
| RecSys Datasets | Curated dataset collection | https://github.com/RUCAIBox/RecSysDatasets |

## Datasets & benchmarks

| Dataset | Domain | Link |
|---|---|---|
| MovieLens (100K/1M/25M) | Movie ratings (explicit) | https://grouplens.org/datasets/movielens/ |
| Amazon Reviews 2023 | Product reviews + metadata | https://amazon-reviews-2023.github.io/ |
| Criteo 1TB Click Logs | Ad CTR (sparse features) | https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/ |
| Yelp Open Dataset | Business reviews / check-ins | https://www.yelp.com/dataset |
| Taobao User Behavior | Large-scale e-commerce clicks | https://tianchi.aliyun.com/dataset/649 |
| RecSysDatasets index | 40+ standardized datasets | https://github.com/RUCAIBox/RecSysDatasets |

Common offline metrics: **NDCG@K, Recall/HR@K, MAP, MRR** for ranking/retrieval, and **AUC / LogLoss** for CTR. Beware the benchmarking pitfalls of negative sampling and inconsistent eval protocols.

## Key papers

| Paper | Year | Link |
|---|---|---|
| BPR: Bayesian Personalized Ranking from Implicit Feedback | 2012 | https://arxiv.org/abs/1205.2618 |
| Wide & Deep Learning for Recommender Systems | 2016 | https://arxiv.org/abs/1606.07792 |
| Neural Collaborative Filtering | 2017 | https://arxiv.org/abs/1708.05031 |
| DeepFM | 2017 | https://arxiv.org/abs/1804.04950 |
| Self-Attentive Sequential Recommendation (SASRec) | 2018 | https://arxiv.org/abs/1808.09781 |
| BERT4Rec | 2019 | https://arxiv.org/abs/1904.06690 |
| DLRM: Deep Learning Recommendation Model | 2019 | https://arxiv.org/abs/1906.00091 |
| LightGCN | 2020 | https://arxiv.org/abs/2002.02126 |
| DCN V2: Improved Deep & Cross Network | 2020 | https://arxiv.org/abs/2008.13535 |

## Cross-references in AIForge

- [Information Retrieval](../Information_Retrieval/) — ANN search, ranking metrics, retrieval that underpins candidate generation
- [Graph Neural Networks](../Graph_Neural_Networks/) — backbone for graph-based CF (LightGCN, PinSage)
- [RAG and Retrieval](../RAG_and_Retrieval/) — embeddings and two-tower retrieval shared with LLM stacks
- [Tabular Deep Learning](../Tabular_Deep_Learning/) — feature-interaction modeling central to deep CTR

## Sources

- https://arxiv.org/abs/1606.07792
- https://arxiv.org/abs/1804.04950
- https://arxiv.org/abs/2008.13535
- https://arxiv.org/abs/1906.00091
- https://arxiv.org/abs/2002.02126
- https://arxiv.org/abs/1904.06690
- https://arxiv.org/abs/1706.03847
- https://github.com/recommenders-team/recommenders
- https://github.com/RUCAIBox/RecBole
- https://github.com/meta-pytorch/torchrec
- https://github.com/RUCAIBox/RecSysDatasets
- https://grouplens.org/datasets/movielens/
- https://amazon-reviews-2023.github.io/

_Seed section expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
