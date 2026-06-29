# Linear Algebra for ML

> The mathematics of vectors, matrices, and linear transformations — the computational substrate on which virtually every machine learning model is built.

## Why it matters

Data in ML is represented as vectors and matrices (feature matrices, embeddings, weight tensors), and the core operations of training and inference — projections, decompositions, gradient steps — are linear-algebraic at their heart. Understanding eigenvalues, SVD, and matrix calculus turns black-box methods like PCA, least squares, and backpropagation into transparent, derivable procedures. Efficient linear algebra (BLAS/LAPACK, GPU kernels) is also what makes large-scale models tractable.

## Core concepts

- **Vectors & vector spaces.** A vector `x ∈ ℝⁿ` is an element of a vector space closed under addition and scalar multiplication. Span, linear independence, basis, and dimension describe the structure of these spaces. Subspaces (column space, null space, row space, left null space) are the "four fundamental subspaces."
- **Inner products & norms.** The dot product `⟨x, y⟩ = xᵀy` induces the Euclidean norm `‖x‖₂ = √(xᵀx)` and the notion of orthogonality (`xᵀy = 0`). Other norms (`ℓ₁`, `ℓ∞`, Frobenius) appear in regularization and error analysis. Cosine similarity `xᵀy / (‖x‖‖y‖)` underpins retrieval and embeddings.
- **Matrices as linear maps.** A matrix `A ∈ ℝ^{m×n}` represents a linear map `ℝⁿ → ℝᵐ`. Matrix multiplication = composition of maps. Rank = dimension of the column space; rank deficiency signals redundancy/collinearity.
- **Solving linear systems.** `Ax = b` is solved via Gaussian elimination / `LU` factorization; over- or under-determined systems are handled by the **normal equations** `AᵀA x = Aᵀb` or, more stably, by `QR` and SVD. Least squares: `x̂ = (AᵀA)⁻¹Aᵀb = A⁺b` (the Moore–Penrose pseudoinverse).
- **Eigen-decomposition.** For square `A`, `Av = λv` gives eigenvalues/eigenvectors. Symmetric matrices admit the **spectral theorem**: `A = QΛQᵀ` with orthonormal `Q`. Positive (semi-)definiteness (`xᵀA x ≥ 0`) characterizes covariance matrices, Hessians at minima, and valid kernels.
- **Singular Value Decomposition (SVD).** Every matrix factors as `A = UΣVᵀ` with orthonormal `U, V` and non-negative singular values `σᵢ`. SVD generalizes eigen-decomposition to rectangular matrices and underlies PCA, low-rank approximation (Eckart–Young), pseudoinverse, and matrix completion.
- **Matrix calculus.** Gradients of scalar functions of vectors/matrices: `∇ₓ(aᵀx) = a`, `∇ₓ(xᵀA x) = (A + Aᵀ)x`, `∇_X tr(AX) = Aᵀ`. The Jacobian and Hessian generalize derivatives; the chain rule over these objects is exactly what backpropagation computes.
- **Tensors.** Higher-order generalizations (rank-3+ arrays) used throughout deep learning; einsum notation expresses contractions compactly.

## Algorithms / Methods

| Method | What it does | Typical use in ML |
|---|---|---|
| LU decomposition | `A = LU`; solves `Ax = b` | Direct linear solves |
| Cholesky | `A = LLᵀ` for SPD `A` | Gaussian processes, covariance sampling |
| QR decomposition | `A = QR`, orthonormal `Q` | Stable least squares, Gram–Schmidt |
| Eigen-decomposition | `A = QΛQ⁻¹` | PCA, spectral clustering, PageRank |
| SVD | `A = UΣVᵀ` | PCA, LSA/LSI, recommender systems, low-rank approx |
| Pseudoinverse `A⁺` | Generalized inverse | Least squares, minimum-norm solutions |
| PCA | Project onto top eigenvectors of covariance | Dimensionality reduction, whitening |
| Power iteration / Lanczos | Top eigen/singular pairs of large/sparse `A` | Scalable spectral methods |
| Conjugate gradient | Iterative solver for SPD systems | Large sparse linear systems |
| Randomized SVD | Sketch-based low-rank factorization | Fast approximate SVD at scale |

## Tools & libraries

| Library | Language | Role | URL |
|---|---|---|---|
| NumPy | Python | Core arrays + `numpy.linalg` | https://numpy.org/ |
| SciPy | Python | `scipy.linalg`, sparse, decompositions | https://scipy.org/ |
| scikit-learn | Python | PCA, TruncatedSVD, NMF | https://scikit-learn.org/ |
| PyTorch | Python | Tensors, autograd, `torch.linalg` | https://pytorch.org/ |
| JAX | Python | Autodiff + `jax.numpy.linalg`, XLA | https://github.com/google/jax |
| TensorFlow | Python | Tensors, `tf.linalg` | https://www.tensorflow.org/ |
| LAPACK | Fortran/C | Reference dense linear algebra | https://www.netlib.org/lapack/ |
| Eigen | C++ | Header-only matrix library | https://eigen.tuxfamily.org/ |
| cuBLAS / cuSOLVER | CUDA | GPU BLAS & solvers | https://developer.nvidia.com/cublas |
| Julia LinearAlgebra | Julia | Built-in high-performance LA | https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/ |

## Learning resources

**Books**
- *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong (free PDF). Parts I cover linear algebra and matrix calculus specifically for ML. https://mml-book.github.io/ · PDF: https://mml-book.github.io/book/mml-book.pdf
- *Introduction to Linear Algebra* — Gilbert Strang (Wellesley-Cambridge). The classic intuition-first text. https://math.mit.edu/~gs/linearalgebra/
- *Linear Algebra and Learning from Data* — Gilbert Strang. ML-oriented sequel. https://math.mit.edu/~gs/learningfromdata/
- *The Matrix Cookbook* — Petersen & Pedersen (free). Dense reference of matrix identities and derivatives. https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf · Archive: https://archive.org/details/imm3274
- *Numerical Linear Algebra* — Trefethen & Bau. Rigorous on conditioning, QR, SVD.

**Courses**
- MIT 18.06 *Linear Algebra* — Gilbert Strang (full video lectures, problem sets). https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- MIT 18.065 *Matrix Methods in Data Analysis, Signal Processing, and Machine Learning* — Strang. https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/

**Visual / tutorials**
- 3Blue1Brown *Essence of Linear Algebra* — geometric intuition for vectors, transformations, eigenvectors. https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- *The Matrix Calculus You Need For Deep Learning* — Parr & Howard. https://explained.ai/matrix-calculus/
- *Computational Linear Algebra* (fast.ai / Rachel Thomas) — Jupyter-based course. https://github.com/fastai/numerical-linear-algebra

## Key papers

- Eckart, C. & Young, G. (1936). *The approximation of one matrix by another of lower rank.* Psychometrika 1(3). DOI: https://doi.org/10.1007/BF02288367 — foundation of low-rank/SVD approximation.
- Golub, G. & Kahan, W. (1965). *Calculating the singular values and pseudo-inverse of a matrix.* SIAM J. Numer. Anal. DOI: https://doi.org/10.1137/0702016 — practical SVD computation.
- Halko, N., Martinsson, P.-G. & Tropp, J. (2011). *Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions.* SIAM Review. DOI: https://doi.org/10.1137/090771806 · arXiv: https://arxiv.org/abs/0909.4061 — randomized SVD.
- Page, L., Brin, S., Motwani, R. & Winograd, T. (1999). *The PageRank Citation Ranking.* Stanford InfoLab. http://ilpubs.stanford.edu:8090/422/ — eigenvector centrality at web scale.
- Hu, Y., Koren, Y. & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets.* IEEE ICDM. DOI: https://doi.org/10.1109/ICDM.2008.22 — matrix-factorization recommenders.
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv: https://arxiv.org/abs/2106.09685 — low-rank structure for efficient fine-tuning.

## Cross-references in AIForge

- [Calculus for ML
- [Probability & Statistics
- [Optimization Algorithms
- [Machine Learning
- [Deep Learning

## Sources

- Mathematics for Machine Learning — https://mml-book.github.io/
- MIT OpenCourseWare 18.06 — https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- MIT OpenCourseWare 18.065 — https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/
- 3Blue1Brown, Essence of Linear Algebra — https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- The Matrix Cookbook (Internet Archive) — https://archive.org/details/imm3274
- The Matrix Calculus You Need For Deep Learning — https://explained.ai/matrix-calculus/
- NumPy / SciPy / scikit-learn / PyTorch / JAX official documentation
