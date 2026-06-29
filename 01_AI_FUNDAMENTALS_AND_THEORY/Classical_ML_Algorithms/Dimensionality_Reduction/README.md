# Dimensionality Reduction (PCA, t-SNE, UMAP)

> Techniques that map high-dimensional data into a lower-dimensional space while preserving as much meaningful structure (variance, neighborhoods, or topology) as possible.

## Why it matters

High-dimensional data suffers from the **curse of dimensionality**: distances concentrate, models overfit, and visualization is impossible. Dimensionality reduction compresses features for faster training, denoising, storage, and 2D/3D exploratory visualization, often revealing cluster and manifold structure invisible in the raw space. It underpins preprocessing pipelines, embedding analysis, and modern representation learning.

## Core concepts

- **Linear vs. nonlinear.** Linear methods (PCA, ICA, factor analysis) find a linear projection; nonlinear / manifold methods (t-SNE, UMAP, Isomap, kernel PCA, autoencoders) capture curved structure.
- **PCA.** Find orthogonal directions of maximum variance. Center data X, compute covariance C = (1/n) XᵀX, take its eigendecomposition C = VΛVᵀ; the top-k eigenvectors (principal components) form the projection. Equivalent to a truncated **SVD** of X = UΣVᵀ. The fraction of variance retained by k components is (Σᵢ₌₁ᵏ λᵢ) / (Σⱼ λⱼ).
- **Kernel PCA.** Apply PCA in a feature space induced by a kernel k(x, y) = ⟨φ(x), φ(y)⟩, solving an eigenproblem on the centered **Gram matrix** K instead of the covariance — enabling nonlinear components without ever computing φ explicitly.
- **ICA.** Unlike PCA (uncorrelated, orthogonal), ICA seeks **statistically independent**, non-Gaussian sources s such that x = As; used for blind source separation (e.g., the cocktail-party problem). Maximizes non-Gaussianity (e.g., negentropy in FastICA).
- **t-SNE.** Converts pairwise distances into conditional probabilities (Gaussian in high-D, Student-t in low-D), then minimizes the **KL divergence** between the two distributions via gradient descent. The heavy-tailed t-distribution mitigates the "crowding problem." Excellent for local cluster structure; **perplexity** (~5–50) tunes neighbor balance. Distances between clusters and cluster sizes are not reliably meaningful.
- **UMAP.** Builds a fuzzy topological graph of the data (k-nearest-neighbor based, grounded in Riemannian geometry / algebraic topology) and optimizes a low-D layout to match it via cross-entropy. Typically faster than t-SNE, preserves more **global structure**, and supports arbitrary embedding dimension and a `transform` for new points.
- **Autoencoders.** Neural networks trained to reconstruct their input through a low-dimensional **bottleneck**; the encoder yields a nonlinear, learnable embedding (variational, denoising, and contractive variants add structure/regularization).
- **Evaluation.** Reconstruction error, explained variance, trustworthiness/continuity, downstream task accuracy, and neighborhood-preservation metrics.

## Algorithms / Methods

| Method | Type | Preserves | Strengths | Caveats |
|---|---|---|---|---|
| **PCA** | Linear, unsupervised | Global variance | Fast, deterministic, interpretable, invertible | Linear only; sensitive to scaling |
| **Truncated SVD / LSA** | Linear | Variance (no centering) | Works on sparse matrices (text/TF-IDF) | Not centered |
| **Kernel PCA** | Nonlinear | Variance in feature space | Captures nonlinear structure | Kernel/parameter choice; O(n²) Gram matrix |
| **ICA (FastICA)** | Linear, statistical | Independence, non-Gaussianity | Source separation, signal/EEG | Assumes independent non-Gaussian sources |
| **Factor Analysis** | Linear, probabilistic | Shared latent factors + noise | Models per-feature noise | Gaussian assumptions |
| **Isomap** | Nonlinear (graph) | Geodesic distances | Global manifold geometry | Sensitive to neighborhood size, noise |
| **LLE / Laplacian Eigenmaps** | Nonlinear (graph) | Local neighborhoods | Local structure | Less robust globally |
| **MDS** | Linear/nonlinear | Pairwise distances | Distance-faithful layouts | O(n²) memory |
| **t-SNE** | Nonlinear, manifold | Local neighborhoods | Best-in-class cluster visualization | Slow; non-deterministic; distorts global geometry |
| **UMAP** | Nonlinear, manifold | Local + some global | Fast, scalable, reusable transform | Stochastic; hyperparameter sensitivity |
| **Autoencoder / VAE** | Nonlinear, learned | Reconstruction (+ latent prior) | Flexible, scalable, generative (VAE) | Needs training/tuning; less interpretable |

## Tools & libraries

| Tool | Use | URL |
|---|---|---|
| scikit-learn `decomposition` | PCA, KernelPCA, TruncatedSVD, FastICA, NMF, FactorAnalysis | https://scikit-learn.org/stable/api/sklearn.decomposition.html |
| scikit-learn `manifold` | t-SNE, Isomap, LLE, MDS, Spectral Embedding | https://scikit-learn.org/stable/modules/manifold.html |
| umap-learn | Reference UMAP implementation | https://umap-learn.readthedocs.io/ |
| openTSNE | Fast, extensible t-SNE with `transform` | https://opentsne.readthedocs.io/ |
| Multicore-TSNE | Parallel Barnes-Hut t-SNE | https://github.com/DmitryUlyanov/Multicore-TSNE |
| PyTorch / Keras | Autoencoders and VAEs | https://pytorch.org/ |
| RAPIDS cuML | GPU PCA, t-SNE, UMAP | https://docs.rapids.ai/api/cuml/stable/ |
| statsmodels | PCA, factor analysis, multivariate stats | https://www.statsmodels.org/stable/multivariate.html |
| MDP / scikit-learn ICA, MNE-Python | ICA for signals/EEG | https://mne.tools/stable/ |

## Learning resources

- **An Introduction to Statistical Learning (ISLR/ISLP)** — Ch. 12 unsupervised learning / PCA, free PDF: https://www.statlearning.com/
- **The Elements of Statistical Learning (ESL)** — §14.5 PCA, ICA, multidimensional scaling, free PDF: https://hastie.su.domains/ElemStatLearn/
- **Mathematics for Machine Learning (MML)** — Ch. 10 dimensionality reduction with PCA, free PDF: https://mml-book.github.io/
- **Murphy, *Probabilistic Machine Learning*** — latent linear/nonlinear models, free PDFs: https://probml.github.io/pml-book/
- **Distill — "How to Use t-SNE Effectively"** (interactive pitfalls guide): https://distill.pub/2016/misread-tsne/
- **UMAP documentation** — "How UMAP Works" and parameter guides: https://umap-learn.readthedocs.io/en/latest/how_umap_works.html
- **StatQuest (Josh Starmer)** — PCA and t-SNE explainers: https://www.youtube.com/c/joshstarmer
- **scikit-learn user guides** — decomposition & manifold learning (linked above)

## Key papers

- Schölkopf, Smola & Müller (1998), *Nonlinear Component Analysis as a Kernel Eigenvalue Problem* (Kernel PCA), Neural Computation. DOI: https://doi.org/10.1162/089976698300017467
- Hyvärinen & Oja (2000), *Independent Component Analysis: Algorithms and Applications*, Neural Networks 13(4–5):411–430. DOI: https://doi.org/10.1016/S0893-6080(00)00026-5
- Tenenbaum, de Silva & Langford (2000), *A Global Geometric Framework for Nonlinear Dimensionality Reduction* (Isomap), Science. DOI: https://doi.org/10.1126/science.290.5500.2319
- Roweis & Saul (2000), *Nonlinear Dimensionality Reduction by Locally Linear Embedding* (LLE), Science. DOI: https://doi.org/10.1126/science.290.5500.2323
- van der Maaten & Hinton (2008), *Visualizing Data using t-SNE*, JMLR 9:2579–2605. https://jmlr.org/papers/v9/vandermaaten08a.html
- McInnes, Healy & Melville (2018), *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*, arXiv:1802.03426. https://arxiv.org/abs/1802.03426
- Kingma & Welling (2013), *Auto-Encoding Variational Bayes* (VAE), arXiv:1312.6114. https://arxiv.org/abs/1312.6114

## Cross-references in AIForge

- [Clustering Algorithms](../Clustering_Algorithms/) — often paired with reduction for visualization and structure discovery
- [Feature Engineering](../../Feature_Engineering/) — feature extraction, selection, and scaling pipelines
- [Mathematics for ML](../../Mathematics_for_ML/) — linear algebra, eigendecomposition, and SVD foundations
- [Deep Learning](../../Deep_Learning/) — autoencoders, VAEs, and learned representations

## Sources

- scikit-learn manifold learning: https://scikit-learn.org/stable/modules/manifold.html
- scikit-learn decomposition API: https://scikit-learn.org/stable/api/sklearn.decomposition.html
- UMAP documentation & repo: https://umap-learn.readthedocs.io/ , https://github.com/lmcinnes/umap
- t-SNE (JMLR 2008): https://jmlr.org/papers/v9/vandermaaten08a.html
- UMAP paper (arXiv:1802.03426): https://arxiv.org/abs/1802.03426
- Kernel PCA (Neural Computation 1998): https://doi.org/10.1162/089976698300017467
- ICA (Neural Networks 2000): https://doi.org/10.1016/S0893-6080(00)00026-5
- Distill "How to Use t-SNE Effectively": https://distill.pub/2016/misread-tsne/
- ISLR/ISLP: https://www.statlearning.com/ ; ESL: https://hastie.su.domains/ElemStatLearn/ ; MML: https://mml-book.github.io/
