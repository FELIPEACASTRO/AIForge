# Clustering Algorithms

> Unsupervised methods that partition a dataset into groups (clusters) such that points within a group are more similar to each other than to points in other groups.

## Why it matters

Clustering is the workhorse of exploratory data analysis: it reveals latent structure, segments customers, compresses data via prototypes (vector quantization), detects anomalies, and produces features or pseudo-labels for downstream supervised learning. Because it requires no labels, it scales to the vast majority of real-world data that is never annotated. The choice of algorithm encodes strong assumptions about cluster shape, density, and number, so understanding the trade-offs is essential to avoid misleading results.

## Core concepts

- **Objective (centroid methods).** K-Means minimizes within-cluster sum of squares (inertia): `J = Σ_k Σ_{x∈C_k} ||x − μ_k||²`, where `μ_k` is the centroid of cluster `C_k`. This is NP-hard in general; Lloyd's algorithm is an alternating-minimization heuristic that converges to a local optimum.
- **Distance / similarity.** Most methods depend on a metric — Euclidean (`L2`), Manhattan (`L1`), cosine, or a precomputed affinity/kernel matrix. Feature scaling (standardization) is usually required because distances are scale-sensitive.
- **Density.** DBSCAN defines clusters as maximal sets of *density-connected* points using two parameters: `ε` (neighborhood radius) and `minPts` (minimum points to form a dense region). Core, border, and noise points emerge naturally; no `k` needed and arbitrary shapes are recoverable.
- **Hierarchy.** Agglomerative clustering builds a dendrogram by iteratively merging the closest clusters under a *linkage* criterion (single, complete, average, Ward). Cutting the dendrogram at a height yields a flat partition.
- **Graph / spectral.** Spectral clustering embeds points using the eigenvectors of a graph Laplacian `L = D − W` (or its normalized variants `L_sym`, `L_rw`), then runs K-Means in the embedding. It captures non-convex, manifold-shaped clusters that K-Means cannot.
- **Choosing k / validating.** Internal indices include the **elbow** (inertia vs. `k`), **silhouette coefficient** `s = (b − a)/max(a,b)`, Calinski–Harabasz, and Davies–Bouldin. External indices (with ground truth) include Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

## Algorithms / Methods

| Method | Idea | Needs `k`? | Cluster shape | Handles noise | Complexity (typical) | Notes |
|---|---|---|---|---|---|---|
| **K-Means (Lloyd)** | Minimize within-cluster variance via centroids | Yes | Convex / isotropic | No | `O(n·k·i·d)` | Fast, scalable; sensitive to init & outliers |
| **K-Means++** | Smart seeding for K-Means | Yes | Convex | No | adds `O(n·k)` seeding | `O(log k)`-competitive; default in sklearn |
| **Mini-Batch K-Means** | K-Means on data batches | Yes | Convex | No | sublinear per step | For very large `n`, slight quality loss |
| **K-Medoids (PAM)** | Use actual points as centers | Yes | Convex | Partially | `O(k(n−k)²)` per iter | Robust to outliers; any metric |
| **Gaussian Mixture (EM)** | Soft probabilistic assignment | Yes | Elliptical | No | `O(n·k·d²·i)` | Gives cluster covariances & responsibilities |
| **Agglomerative (Ward, etc.)** | Bottom-up merging w/ linkage | No (cut tree) | Linkage-dependent | No | `O(n²log n)`–`O(n³)` | Dendrogram; deterministic |
| **DBSCAN** | Density-connected regions | No | Arbitrary | Yes | `O(n log n)` w/ index | `ε`, `minPts`; struggles w/ varying density |
| **HDBSCAN** | Hierarchical DBSCAN + stability | No | Arbitrary | Yes | `~O(n log n)` | Robust to varying density; only `min_cluster_size` |
| **OPTICS** | Density ordering, variable `ε` | No | Arbitrary | Yes | `O(n log n)` | Reachability plot; extracts multi-density clusters |
| **Spectral Clustering** | Eigenvectors of graph Laplacian | Yes | Non-convex / manifold | No | `O(n³)` (eigsolve) | Excellent for connectivity; poor scaling |
| **Mean Shift** | Mode-seeking via kernel density | No | Arbitrary blobs | Partially | `O(n²·i)` | Bandwidth-driven; finds number of clusters |
| **Affinity Propagation** | Message passing between exemplars | No | — | No | `O(n²·i)` | Picks exemplars; many clusters, slow |
| **BIRCH** | CF-tree summaries | Optional | Spherical | Partially | `O(n)` | Streaming / very large datasets |

## Tools & libraries

| Tool | What it offers | URL |
|---|---|---|
| **scikit-learn** | K-Means, DBSCAN, OPTICS, Agglomerative, Spectral, Mean Shift, Affinity Propagation, BIRCH, GMM, metrics | https://scikit-learn.org/stable/modules/clustering.html |
| **hdbscan** | High-performance HDBSCAN with soft clustering & outlier scores | https://github.com/scikit-learn-contrib/hdbscan |
| **scikit-learn-extra** | K-Medoids (PAM), common-nearest-neighbor clustering | https://github.com/scikit-learn-contrib/scikit-learn-extra |
| **SciPy** | Hierarchical clustering (`scipy.cluster.hierarchy`), dendrograms, linkage | https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html |
| **FAISS** | GPU/CPU large-scale K-Means & nearest-neighbor for clustering at scale | https://github.com/facebookresearch/faiss |
| **RAPIDS cuML** | GPU-accelerated K-Means, DBSCAN, HDBSCAN, Agglomerative | https://github.com/rapidsai/cuml |
| **PyClustering** | 40+ clustering algorithms (CURE, ROCK, X-Means, etc.) | https://github.com/annoviko/pyclustering |
| **scikit-learn GaussianMixture** | EM-based soft clustering with full covariances | https://scikit-learn.org/stable/modules/mixture.html |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman) — Ch. 14 Unsupervised Learning | Book (free PDF) | https://hastie.su.domains/ElemStatLearn/ |
| *An Introduction to Statistical Learning* (ISLR/ISLP) — Ch. 12 Unsupervised Learning | Book (free PDF) | https://www.statlearning.com/ |
| *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) | Book (free PDF) | https://mml-book.github.io/ |
| *Probabilistic Machine Learning* (Kevin Murphy) — mixture models & clustering | Book (free PDF) | https://probml.github.io/pml-book/ |
| scikit-learn clustering user guide | Tutorial / docs | https://scikit-learn.org/stable/modules/clustering.html |
| "A Tutorial on Spectral Clustering" (von Luxburg) | Tutorial paper | https://arxiv.org/abs/0711.0189 |
| StatQuest — K-means & Hierarchical clustering (Josh Starmer) | Video | https://www.youtube.com/c/joshstarmer |
| Stanford CS229 lecture notes (K-means, EM/GMM) | Course notes | https://cs229.stanford.edu/ |

## Key papers

- Lloyd, S. P. — *Least Squares Quantization in PCM* (Bell Labs 1957; IEEE Trans. Information Theory, 1982). DOI: https://doi.org/10.1109/TIT.1982.1056489 — foundational K-Means algorithm.
- Arthur, D. & Vassilvitskii, S. — *k-means++: The Advantages of Careful Seeding* (SODA 2007). https://dl.acm.org/doi/10.5555/1283383.1283494 (PDF: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- Ester, M., Kriegel, H.-P., Sander, J., Xu, X. — *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise* (KDD 1996). https://cdn.aaai.org/KDD/1996/KDD96-037.pdf — DBSCAN.
- Ankerst, M., Breunig, M., Kriegel, H.-P., Sander, J. — *OPTICS: Ordering Points To Identify the Clustering Structure* (SIGMOD 1999). https://doi.org/10.1145/304182.304187
- Campello, R. J. G. B., Moulavi, D., Sander, J. — *Density-Based Clustering Based on Hierarchical Density Estimates* (PAKDD 2013). https://doi.org/10.1007/978-3-642-37456-2_14 — HDBSCAN.
- Ng, A. Y., Jordan, M. I., Weiss, Y. — *On Spectral Clustering: Analysis and an Algorithm* (NIPS 2001). https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html
- von Luxburg, U. — *A Tutorial on Spectral Clustering* (Statistics and Computing, 2007). https://arxiv.org/abs/0711.0189
- McInnes, L., Healy, J., Astels, S. — *hdbscan: Hierarchical density based clustering* (JOSS, 2017). https://doi.org/10.21105/joss.00205

## Cross-references in AIForge

- [Dimensionality Reduction](../Dimensionality_Reduction/) — PCA/UMAP embeddings often precede clustering (e.g., spectral, HDBSCAN on UMAP).
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — Gaussian mixtures, EM, and Dirichlet-process (nonparametric) clustering.
- [Model Evaluation](../../Model_Evaluation/) — silhouette, ARI, NMI, and internal/external validation indices.
- [Anomaly and OOD Detection](../../Anomaly_and_OOD_Detection/) — density-based clustering (DBSCAN/HDBSCAN) for outlier identification.

## Sources

- scikit-learn — Clustering user guide: https://scikit-learn.org/stable/modules/clustering.html
- DBSCAN (Ester et al., KDD 1996): https://cdn.aaai.org/KDD/1996/KDD96-037.pdf
- HDBSCAN (Campello et al., PAKDD 2013): https://doi.org/10.1007/978-3-642-37456-2_14
- hdbscan JOSS paper / library: https://doi.org/10.21105/joss.00205 · https://github.com/scikit-learn-contrib/hdbscan
- k-means++ (Arthur & Vassilvitskii, SODA 2007): https://dl.acm.org/doi/10.5555/1283383.1283494
- Lloyd (IEEE TIT 1982): https://doi.org/10.1109/TIT.1982.1056489
- Spectral clustering (Ng, Jordan, Weiss, NIPS 2001): https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html
- von Luxburg spectral clustering tutorial: https://arxiv.org/abs/0711.0189
- OPTICS (Ankerst et al., SIGMOD 1999): https://doi.org/10.1145/304182.304187
- ESL: https://hastie.su.domains/ElemStatLearn/ · ISLR: https://www.statlearning.com/ · MML book: https://mml-book.github.io/ · PML (Murphy): https://probml.github.io/pml-book/
