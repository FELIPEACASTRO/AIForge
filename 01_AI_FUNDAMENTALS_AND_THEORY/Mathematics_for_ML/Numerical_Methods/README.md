# Numerical Methods and Scientific Computing

> The study of algorithms that approximate solutions to continuous mathematical problems using finite-precision arithmetic — covering linear solvers, numerical stability, and the conditioning of computations that underpin all of ML.

## Why it matters

Every ML model is ultimately a numerical computation: training reduces to solving linear systems, factorizing matrices, and iterating gradient updates in floating-point arithmetic. Understanding conditioning, stability, and round-off error is what separates code that silently produces garbage (NaNs, divergence, rank-deficient covariance matrices) from code that is correct and fast. NumPy and SciPy wrap decades of battle-tested numerical linear algebra (LAPACK/BLAS), and knowing what they do under the hood lets you choose the right solver and diagnose failures.

## Core concepts

- **Floating-point representation (IEEE 754).** Reals are stored as `sign × mantissa × 2^exponent`. `float64` (double) has ~15-17 significant decimal digits; machine epsilon ε ≈ 2.22e-16. Not all decimals are representable (`0.1 + 0.2 != 0.3`). Special values: `inf`, `-inf`, `nan`. Subtractive **cancellation** destroys significant digits when subtracting nearly equal numbers.
- **Conditioning.** A problem's sensitivity to input perturbation, independent of algorithm. For a linear system `Ax = b`, the **condition number** κ(A) = ‖A‖·‖A⁻¹‖ = σ_max/σ_min bounds the relative error: a perturbation of size δ in the data can be amplified by κ(A). κ ≈ 10^k means you lose roughly k digits of accuracy. κ = ∞ means singular.
- **Stability.** A property of the *algorithm*. A **backward-stable** algorithm returns the exact solution to a slightly perturbed problem (small backward error). Forward error ≲ condition number × backward error.
- **Direct vs. iterative solvers.** Direct methods (factorize, then back-substitute) give an answer in a fixed number of operations (O(n³) for dense). Iterative methods (Krylov subspace, fixed-point) refine a guess and are preferred for large sparse systems where O(n³) and O(n²) storage are infeasible.
- **Matrix factorizations.** `LU` (general square), `Cholesky` `A = LLᵀ` (symmetric positive-definite, ~2× faster than LU and stable without pivoting), `QR` `A = QR` (least squares, orthogonalization), `SVD` `A = UΣVᵀ` (rank, pseudoinverse, PCA, the most numerically robust). Solving via factorization is preferred over forming `A⁻¹` explicitly — never compute an inverse to solve a system.
- **Least squares.** For overdetermined `Ax ≈ b`, the normal equations `AᵀAx = Aᵀb` square the condition number (κ²); prefer QR or SVD for numerical accuracy.

## Algorithms / Methods

| Method | Problem solved | Matrix type | Cost (dense n×n) | Notes |
|---|---|---|---|---|
| Gaussian elimination / **LU** | `Ax = b` | general nonsingular | O(n³) | Partial pivoting for stability (`scipy.linalg.lu`) |
| **Cholesky** | `Ax = b` | symmetric positive-definite | ~n³/3 | Fastest stable direct solver; fails ⇒ not SPD |
| **QR** (Householder/Givens) | least squares, orthogonalization | general / tall | O(mn²) | Backward stable; basis for `numpy.linalg.lstsq` |
| **SVD** | rank, pseudoinverse, least squares | any (incl. rank-deficient) | O(mn²) | Most robust; truncated SVD ⇒ low-rank approximation |
| **Conjugate Gradient (CG)** | `Ax = b` | sparse SPD | O(nnz) / iter | Krylov; converges in ≤ n steps in exact arithmetic |
| **GMRES** | `Ax = b` | sparse nonsymmetric | O(nnz·m) | Minimizes residual over Krylov subspace; needs restarts |
| **Jacobi / Gauss-Seidel / SOR** | `Ax = b` | diagonally dominant | O(nnz) / iter | Classical stationary iterations; slow, good preconditioners |
| **Power iteration / Lanczos / Arnoldi** | eigenvalues / eigenvectors | large sparse | varies | Lanczos (symmetric), Arnoldi (general) Krylov methods |
| **Newton's method** | nonlinear `f(x)=0` | — | per iter | Quadratic convergence near root; needs Jacobian |

## Tools & libraries

| Tool | What it is | URL |
|---|---|---|
| NumPy | Core N-d array + `numpy.linalg` (LAPACK-backed) | https://numpy.org/doc/stable/reference/routines.linalg.html |
| SciPy `linalg` | Dense linear algebra: LU, Cholesky, QR, SVD, eig | https://docs.scipy.org/doc/scipy/reference/linalg.html |
| SciPy `sparse.linalg` | Sparse direct + iterative solvers (CG, GMRES, spsolve) | https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html |
| LAPACK / BLAS | Reference numerical linear algebra libraries (Fortran) | https://www.netlib.org/lapack/ |
| JAX | Composable autodiff + XLA-accelerated NumPy-like API | https://jax.readthedocs.io/ |
| Numba | JIT compiler for numerical Python (LLVM) | https://numba.pydata.org/ |
| Julia | Language built for high-performance scientific computing | https://julialang.org/ |
| MPMath | Arbitrary-precision floating-point arithmetic | https://mpmath.org/ |

## Learning resources

- **Numerical Linear Algebra** — Trefethen & Bau (SIAM, 1997; 25th-anniversary ed. 2022). The standard graduate text; 40 short lectures, starts from QR and conditioning. https://people.maths.ox.ac.uk/trefethen/text.html
- **What Every Computer Scientist Should Know About Floating-Point Arithmetic** — David Goldberg, ACM Computing Surveys 1991. The definitive floating-point primer. https://dl.acm.org/doi/10.1145/103162.103163 · PDF: https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf
- **SciPy Lecture Notes / Scientific Python lectures** — practical, hands-on. https://lectures.scientific-python.org/
- **NumPy Linear Algebra reference** — official API + examples. https://numpy.org/doc/stable/reference/routines.linalg.html
- **Matrix Computations** — Golub & Van Loan (Johns Hopkins). The encyclopedic reference for numerical linear algebra. https://jhupbooks.press.jhu.edu/title/matrix-computations
- **An Introduction to the Conjugate Gradient Method Without the Agonizing Pain** — Jonathan Shewchuk. Classic intuitive tutorial. https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
- **0.30000000000000004.com** — interactive demonstration of floating-point errors across languages. https://0.30000000000000004.com/

## Key papers

- Hestenes, M. R. & Stiefel, E. (1952). *Methods of Conjugate Gradients for Solving Linear Systems.* J. Res. NBS 49, 409-436. https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf
- Saad, Y. & Schultz, M. H. (1986). *GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems.* SIAM J. Sci. Stat. Comput. 7(3), 856-869. DOI: https://doi.org/10.1137/0907058
- Goldberg, D. (1991). *What Every Computer Scientist Should Know About Floating-Point Arithmetic.* ACM Computing Surveys 23(1), 5-48. https://dl.acm.org/doi/10.1145/103162.103163
- IEEE (2019). *IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019).* https://doi.org/10.1109/IEEESTD.2019.8766229
- Harris, C. R. et al. (2020). *Array Programming with NumPy.* Nature 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2
- Virtanen, P. et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.* Nature Methods 17, 261-272. https://doi.org/10.1038/s41592-019-0686-2

## Cross-references in AIForge

- [Optimization Algorithms — gradient methods and solvers built on this numerical foundation
- [Machine Learning — where linear systems and factorizations power model fitting
- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/README.md) — Cholesky and stable covariance computation in practice
- [Linear Algebra](../Linear_Algebra/README.md) — the theory underlying these numerical methods

## Sources

- Trefethen & Bau — https://people.maths.ox.ac.uk/trefethen/text.html
- SciPy linalg manual — https://docs.scipy.org/doc/scipy/reference/linalg.html
- SciPy sparse.linalg manual — https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
- NumPy linalg reference — https://numpy.org/doc/stable/reference/routines.linalg.html
- Goldberg, ACM Computing Surveys — https://dl.acm.org/doi/10.1145/103162.103163
- Hestenes & Stiefel, NIST — https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf
- Saad & Schultz, GMRES (SIAM) — https://doi.org/10.1137/0907058
- IEEE 754-2019 — https://doi.org/10.1109/IEEESTD.2019.8766229
- NumPy (Nature 2020) — https://doi.org/10.1038/s41586-020-2649-2
- SciPy 1.0 (Nature Methods 2020) — https://doi.org/10.1038/s41592-019-0686-2
