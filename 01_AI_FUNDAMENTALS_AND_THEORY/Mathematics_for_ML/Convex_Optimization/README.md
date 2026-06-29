# Convex Optimization

> The study of minimizing convex functions over convex sets — a class of problems where any local minimum is global and which can be solved reliably, efficiently, and at scale.

## Why it matters

Convex problems are the "tractable core" of mathematical optimization: they admit global optimality guarantees, polynomial-time algorithms, and rich duality theory, which is why so much ML (SVMs, lasso/ridge, logistic regression, portfolio optimization, MAP estimation, many relaxations) is cast in convex form. Recognizing or reformulating a problem as convex turns an unreliable search into a solvable, certifiable computation. Even in deep, non-convex learning, convex analysis underpins the optimizers, regularizers, and theoretical tools used everywhere.

## Core concepts

- **Convex set.** A set $C$ is convex if for all $x,y \in C$ and $\theta \in [0,1]$, $\theta x + (1-\theta)y \in C$. Examples: hyperplanes, halfspaces, norm balls, the positive semidefinite cone $\mathbf{S}^n_+$, polyhedra $\{x : Ax \le b\}$.
- **Convex function.** $f$ is convex if its domain is convex and $f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta) f(y)$. First-order condition: $f(y) \ge f(x) + \nabla f(x)^\top (y-x)$. Second-order: $\nabla^2 f(x) \succeq 0$ (PSD Hessian).
- **Convex optimization problem.** Minimize a convex $f_0(x)$ subject to convex inequality constraints $f_i(x) \le 0$ and affine equalities $Ax = b$. The feasible set is convex, so **every local minimum is global**.
- **Standard forms.** LP (linear), QP (quadratic objective, affine constraints), QCQP, SOCP (second-order cone), SDP (semidefinite), GP (geometric, convex after log change of variables), and the unifying **conic** form $\min c^\top x$ s.t. $Ax = b,\ x \in \mathcal{K}$.
- **Lagrangian & duality.** With $L(x,\lambda,\nu) = f_0(x) + \sum_i \lambda_i f_i(x) + \nu^\top(Ax-b)$, the dual function $g(\lambda,\nu) = \inf_x L$ gives a lower bound. **Weak duality** $d^\star \le p^\star$ always holds; **strong duality** ($d^\star = p^\star$) holds under constraint qualifications such as **Slater's condition**.
- **KKT conditions.** For convex problems with strong duality, the **Karush-Kuhn-Tucker** conditions (stationarity, primal/dual feasibility, complementary slackness $\lambda_i f_i(x)=0$) are *necessary and sufficient* for optimality.
- **Subgradients.** For non-smooth convex $f$, a subgradient $g$ satisfies $f(y) \ge f(x) + g^\top(y-x)$; the subdifferential $\partial f(x)$ generalizes the gradient (enables L1/lasso, hinge loss).
- **Strong convexity & smoothness.** $\mu$-strong convexity and $L$-Lipschitz gradients (condition number $\kappa = L/\mu$) govern convergence rates of first-order methods.
- **Proximal operator.** $\operatorname{prox}_{f}(v) = \arg\min_x f(x) + \tfrac12\|x-v\|_2^2$ — the engine behind proximal/splitting methods for composite objectives $f + g$.
- **DCP (Disciplined Convex Programming).** A ruleset (atoms + composition rules) that lets modeling tools like CVXPY *verify* convexity by construction and canonicalize to a conic solver.

## Algorithms / Methods

| Method | Problem class | Convergence (smooth case) | Notes |
|---|---|---|---|
| Gradient descent | Smooth, unconstrained | $O(1/k)$; linear if strongly convex | Simplest first-order baseline |
| Nesterov accelerated gradient | Smooth | $O(1/k^2)$ | Optimal first-order rate |
| Subgradient method | Non-smooth convex | $O(1/\sqrt{k})$ | Diminishing step sizes; no descent guarantee |
| Proximal gradient / ISTA | Composite $f + g$ ($g$ non-smooth) | $O(1/k)$ | Handles L1; FISTA accelerates to $O(1/k^2)$ |
| Newton's method | Smooth, twice-diff. | Local quadratic | Uses Hessian; affine-invariant |
| Interior-point (barrier / primal-dual) | LP, QP, SOCP, SDP, conic | Polynomial | Self-concordant barriers; high accuracy |
| ADMM | Separable / composite, distributed | Linear under conditions | Splits problem; great for large-scale/parallel |
| Coordinate descent | Separable structure | Problem-dependent | Workhorse for lasso/GLMnet |
| Mirror descent | Constrained, non-Euclidean geometry | $O(1/\sqrt{k})$ | Generalizes (sub)gradient via Bregman divergence |
| Frank-Wolfe (cond. gradient) | Smooth over compact convex set | $O(1/k)$ | Projection-free; sparse/atomic iterates |
| Ellipsoid method | General convex (feasibility) | Polynomial | Theoretically important, slow in practice |

## Tools & libraries

| Tool | What it is | URL |
|---|---|---|
| CVXPY | Python DCP modeling language for convex programs | https://www.cvxpy.org/ |
| CVX | MATLAB DCP modeling system (Boyd group) | https://cvxr.com/cvx/ |
| Convex.jl | Julia DCP modeling library | https://jump.dev/Convex.jl/stable/ |
| JuMP | Julia algebraic modeling for math optimization | https://jump.dev/ |
| Clarabel | Open-source interior-point conic solver (default in CVXPY) | https://clarabel.org/ |
| OSQP | Operator-splitting QP solver | https://osqp.org/ |
| SCS | Splitting Conic Solver for large cone programs | https://www.cvxgrp.org/scs/ |
| CVXOPT | Python software for convex optimization | https://cvxopt.org/ |
| ECOS | Embedded conic solver (SOCP) | https://github.com/embotech/ecos |
| MOSEK | Commercial high-performance conic/IP solver | https://www.mosek.com/ |
| Gurobi | Commercial LP/QP/MILP/MIQP solver | https://www.gurobi.com/ |
| SciPy `optimize` | General-purpose Python optimization (incl. `linprog`) | https://docs.scipy.org/doc/scipy/reference/optimize.html |

## Learning resources

- **Boyd & Vandenberghe, *Convex Optimization*** — the canonical textbook, free PDF: https://stanford.edu/~boyd/cvxbook/
- **Stanford EE364A: Convex Optimization I** — course site with slides/notes: https://web.stanford.edu/class/ee364a/ and 2023 video lectures: https://www.youtube.com/playlist?list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h
- **Boyd & Vandenberghe lecture slides** (current revision): https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf
- **Nesterov, *Lectures on Convex Optimization* (Springer)** — rigorous treatment of complexity & accelerated methods: https://doi.org/10.1007/978-3-319-91578-4
- **Mathematics for Machine Learning (Deisenroth, Faisal, Ong)** — Ch. 7 on continuous optimization, free PDF: https://mml-book.github.io/
- **CVXPY tutorial / examples** — practical DCP modeling: https://www.cvxpy.org/tutorial/index.html and https://www.cvxpy.org/examples/
- **CVX Short Course (Boyd group)** — slides + exercises: https://www.cvxgrp.org/cvx_short_course/
- **Boyd et al., *Distributed Optimization via ADMM* (monograph)** — free: https://web.stanford.edu/~boyd/papers/admm_distr_stats.html

## Key papers

- Karmarkar, N. (1984). *A new polynomial-time algorithm for linear programming.* Combinatorica. https://doi.org/10.1007/BF02579150
- Nesterov, Y. (1983). *A method for solving the convex programming problem with convergence rate $O(1/k^2)$.* (accelerated gradient; foundational).
- Nesterov & Nemirovskii (1994). *Interior-Point Polynomial Algorithms in Convex Programming.* SIAM. https://doi.org/10.1137/1.9781611970791
- Beck & Teboulle (2009). *A Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for linear inverse problems.* SIAM J. Imaging Sciences. https://doi.org/10.1137/080716542
- Boyd, Parikh, Chu, Peleato, Eckstein (2011). *Distributed Optimization and Statistical Learning via the ADMM.* Foundations and Trends in ML. https://doi.org/10.1561/2200000016
- Diamond & Boyd (2016). *CVXPY: A Python-Embedded Modeling Language for Convex Optimization.* JMLR. https://www.jmlr.org/papers/volume17/15-408/15-408.pdf
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.* JRSS-B. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Stellato, Banjac, Goulart, Bemporad, Boyd (2020). *OSQP: an operator splitting solver for quadratic programs.* Math. Prog. Comp. https://doi.org/10.1007/s12532-020-00179-2

## Cross-references in AIForge

- [../Linear_Algebra](../Linear_Algebra) — vectors, matrices, PSD cones, and SDP foundations
- [../Calculus_and_Optimization — gradients, Hessians, Lagrange multipliers
- [../../Optimization_Algorithms](../../Optimization_Algorithms) — SGD, Adam, and non-convex training optimizers
- [../../Machine_Learning](../../Machine_Learning) — SVMs, lasso/ridge, and other convex learning models
- [../../Bayesian_and_Probabilistic_ML](../../Bayesian_and_Probabilistic_ML) — MAP estimation and convex variational objectives

## Sources

- Boyd & Vandenberghe, *Convex Optimization* — https://stanford.edu/~boyd/cvxbook/
- Stanford EE364A — https://web.stanford.edu/class/ee364a/
- CVXPY — https://www.cvxpy.org/ and JMLR paper https://www.jmlr.org/papers/volume17/15-408/15-408.pdf
- ADMM monograph (Boyd et al.) — https://web.stanford.edu/~boyd/papers/admm_distr_stats.html
- Nesterov & Nemirovskii interior-point theory — https://doi.org/10.1137/1.9781611970791
- Karmarkar (1984) — https://doi.org/10.1007/BF02579150
- FISTA (Beck & Teboulle) — https://doi.org/10.1137/080716542
- OSQP — https://osqp.org/ ; SCS — https://www.cvxgrp.org/scs/ ; Clarabel — https://clarabel.org/ ; CVXOPT — https://cvxopt.org/ ; MOSEK — https://www.mosek.com/
- Mathematics for ML book — https://mml-book.github.io/
