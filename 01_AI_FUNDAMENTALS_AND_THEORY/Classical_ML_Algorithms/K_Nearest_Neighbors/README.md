# K-Nearest Neighbors

> A non-parametric, instance-based ("lazy") method that predicts a query point's label or value from the *k* closest training examples under a chosen distance metric.

## Why it matters

K-Nearest Neighbors (KNN) is one of the oldest and most intuitive supervised algorithms: it makes no assumptions about the data distribution, requires essentially no training, and adapts to arbitrarily complex decision boundaries. Its core machinery — fast nearest-neighbor retrieval — underpins modern recommendation, deduplication, and vector-database / retrieval-augmented-generation (RAG) systems, where billion-scale approximate nearest neighbor (ANN) search is now production infrastructure. Understanding KNN therefore bridges classical ML theory (Bayes-error bounds) and large-scale similarity search.

## Core concepts

- **Decision rule.** For a query `x`, find the set `N_k(x)` of its *k* nearest training points. Classification uses a (optionally distance-weighted) majority vote; regression averages the neighbors' targets: `ŷ = (1/k) Σ_{i∈N_k(x)} y_i`.
- **Lazy / instance-based learning.** There is no model fit beyond storing the training set; all computation happens at query time. Prediction cost is `O(n·d)` per query for brute force (n = samples, d = dimensions).
- **Distance metrics** define "nearness":
  - Minkowski: `d(x,y) = (Σ_i |x_i − y_i|^p)^(1/p)`; `p=2` → Euclidean (L2), `p=1` → Manhattan (L1), `p→∞` → Chebyshev.
  - Cosine distance: `1 − (x·y)/(‖x‖‖y‖)` — scale-invariant, common for text/embeddings.
  - Mahalanobis: `√((x−y)ᵀ Σ⁻¹ (x−y))` — accounts for feature covariance.
  - Hamming for categorical/binary data.
- **Feature scaling is mandatory.** Because distances mix features additively, unscaled variables with large ranges dominate; standardize or normalize first.
- **Choosing k.** Small *k* → low bias, high variance (noisy, jagged boundaries); large *k* → smoother boundaries but higher bias. Tune via cross-validation; odd *k* avoids ties in binary classification.
- **Curse of dimensionality.** In high *d*, distances concentrate (nearest and farthest neighbors become similar), degrading KNN; mitigate with dimensionality reduction (PCA/UMAP) or learned/metric embeddings.
- **Theoretical guarantee.** Cover & Hart (1967) proved the asymptotic 1-NN error is bounded above by twice the Bayes error; the k-NN error decreases toward the Bayes rate as `k→∞` with `k/n→0`.
- **Weighting.** Distance weighting (`w_i = 1/d_i`) lets closer neighbors count more, often improving accuracy over uniform votes.

## Algorithms / Methods

| Method | Idea | Build / Query complexity | Best for |
|---|---|---|---|
| Brute force | Compute all pairwise distances | Query `O(n·d)` | Small n, or high d where trees fail |
| KD-tree (Bentley 1975) | Axis-aligned binary space partition | Build `O(n log n)`; query `~O(log n)` low d | Low-to-moderate d (≲20), Euclidean/Minkowski |
| Ball tree | Nested hyperspheres | Build `O(n log n)`; query sublinear | Higher d than KD-tree, general metrics |
| LSH (Locality-Sensitive Hashing) | Hash so near points collide | Sublinear query | High-d approximate search |
| HNSW (Malkov & Yashunin 2016) | Hierarchical navigable small-world graph | Build `O(n log n)`; query `~O(log n)` | State-of-the-art ANN, high recall/speed |
| IVF + PQ (inverted file + product quantization) | Cluster + compress vectors | Sublinear, low memory | Billion-scale ANN (FAISS) |
| Tree ensembles (Annoy) | Forest of random projection trees | mmap-able static index | Memory-efficient, disk-resident ANN |
| Edited / condensed NN | Prune redundant/noisy points | Reduces stored set | Faster, more robust 1-NN |

## Tools & libraries

| Tool | What it offers | URL |
|---|---|---|
| scikit-learn `neighbors` | `KNeighborsClassifier/Regressor`, KD-tree, ball tree, multiple metrics | https://scikit-learn.org/stable/modules/neighbors.html |
| FAISS (Meta) | GPU/CPU ANN, IVF, PQ, HNSW; billion-scale | https://github.com/facebookresearch/faiss |
| hnswlib | Header-only C++/Python HNSW implementation | https://github.com/nmslib/hnswlib |
| Annoy (Spotify) | Random-projection forests, mmap static index | https://github.com/spotify/annoy |
| ScaNN (Google) | Anisotropic vector quantization, fast MIPS | https://github.com/google-research/google-research/tree/master/scann |
| nmslib | Non-metric space library (HNSW, SW-graph) | https://github.com/nmslib/nmslib |
| PyNNDescent | Fast approximate KNN graph construction | https://github.com/lmcinnes/pynndescent |
| ann-benchmarks | Reproducible benchmarks of ANN libraries | https://github.com/erikbern/ann-benchmarks |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman), ch. 13 | Book (free PDF) | https://hastie.su.domains/ElemStatLearn/ |
| *An Introduction to Statistical Learning* (ISLR/ISLP), ch. 2 & 4 | Book (free PDF) | https://www.statlearning.com/ |
| *Probabilistic Machine Learning* (Murphy) | Book (free PDF) | https://probml.github.io/pml-book/ |
| scikit-learn Nearest Neighbors user guide | Tutorial / docs | https://scikit-learn.org/stable/modules/neighbors.html |
| StatQuest — KNN, Clearly Explained | Video | https://www.youtube.com/watch?v=HVXime0nQeI |
| Pinecone — Hierarchical Navigable Small Worlds (HNSW) | Tutorial | https://www.pinecone.io/learn/series/faiss/hnsw/ |

## Key papers

- Cover, T. & Hart, P. (1967). *Nearest Neighbor Pattern Classification.* IEEE Trans. Information Theory 13(1):21-27. DOI: https://doi.org/10.1109/TIT.1967.1053964
- Fix, E. & Hodges, J.L. (1951/1989). *Discriminatory Analysis — Nonparametric Discrimination.* (Reprinted) Int. Statistical Review 57(3):238-247. DOI: https://doi.org/10.2307/1403797
- Bentley, J.L. (1975). *Multidimensional Binary Search Trees Used for Associative Searching.* Comm. ACM 18(9):509-517. DOI: https://doi.org/10.1145/361002.361007
- Malkov, Yu.A. & Yashunin, D.A. (2016). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* arXiv: https://arxiv.org/abs/1603.09320
- Johnson, J., Douze, M. & Jégou, H. (2017). *Billion-scale similarity search with GPUs (FAISS).* arXiv: https://arxiv.org/abs/1702.08734
- Guo, R. et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN).* arXiv: https://arxiv.org/abs/1908.10396
- Weinberger, K.Q. & Saul, L.K. (2009). *Distance Metric Learning for Large Margin Nearest Neighbor Classification.* JMLR 10:207-244. https://www.jmlr.org/papers/v10/weinberger09a.html

## Cross-references in AIForge

- [Machine Learning](../../Machine_Learning/) — broader supervised/unsupervised context
- [Model Evaluation](../../Model_Evaluation/) — cross-validation and choosing *k*
- [Optimization Algorithms](../../Optimization_Algorithms/) — metric learning and tuning
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — Bayes error and density estimation links

## Sources

- scikit-learn — Nearest Neighbors: https://scikit-learn.org/stable/modules/neighbors.html
- scikit-learn — KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- Cover & Hart (1967), IEEE TIT: https://doi.org/10.1109/TIT.1967.1053964
- Bentley (1975), Comm. ACM: https://dl.acm.org/doi/10.1145/361002.361007
- Malkov & Yashunin (2016), HNSW: https://arxiv.org/abs/1603.09320
- FAISS: https://github.com/facebookresearch/faiss
- Annoy: https://github.com/spotify/annoy
- hnswlib: https://github.com/nmslib/hnswlib
- ScaNN: https://github.com/google-research/google-research/tree/master/scann
- ann-benchmarks: https://github.com/erikbern/ann-benchmarks
