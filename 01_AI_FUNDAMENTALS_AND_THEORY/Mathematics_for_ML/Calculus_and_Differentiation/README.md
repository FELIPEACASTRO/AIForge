# Calculus and Automatic Differentiation

> The study of derivatives, gradients, and Jacobians, plus the algorithmic machinery (autodiff / backpropagation) that computes them exactly and efficiently for the functions that define modern ML models.

## Why it matters

Virtually every trainable model is fit by following gradients of a loss function, so calculus is the engine room of machine learning. Automatic differentiation (autodiff) lets us compute those gradients exactly and at machine precision for arbitrarily complex programs, without deriving formulas by hand or suffering the truncation error of finite differences. Frameworks like JAX, PyTorch, and TensorFlow are, at their core, autodiff systems wrapped around array compute.

## Core concepts

- **Derivative.** For a scalar function f: ℝ → ℝ, f'(x) = lim_{h→0} (f(x+h) − f(x)) / h measures local rate of change.
- **Partial derivative & gradient.** For f: ℝⁿ → ℝ, the gradient ∇f = [∂f/∂x₁, …, ∂f/∂xₙ]ᵀ points in the direction of steepest ascent. Gradient descent steps along −∇f.
- **Jacobian.** For a vector field f: ℝⁿ → ℝᵐ, the Jacobian J ∈ ℝ^{m×n} has Jᵢⱼ = ∂fᵢ/∂xⱼ. It linearizes f locally: f(x+δ) ≈ f(x) + Jδ.
- **Hessian.** The matrix of second partials H ∈ ℝ^{n×n}, Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ; encodes curvature, used in Newton/quasi-Newton methods and for analyzing optima (positive-definite H ⇒ local min).
- **Chain rule.** The backbone of autodiff: for y = g(h(x)), dy/dx = g'(h(x)) · h'(x). In multiple dimensions this is matrix multiplication of Jacobians, J_{f∘g} = J_f · J_g.
- **Taylor expansion.** f(x+δ) ≈ f(x) + ∇f·δ + ½ δᵀHδ + …; the basis of most local optimization and linearization arguments.
- **JVP vs VJP.** A *Jacobian–vector product* (Jv, forward mode) propagates a tangent through the program; a *vector–Jacobian product* (vᵀJ, reverse mode) propagates a cotangent (adjoint) backward. Both avoid ever materializing the full Jacobian.
- **Backpropagation.** Reverse-mode autodiff applied to a scalar loss: one forward pass caches intermediate activations, one backward pass accumulates ∂L/∂(·) via the chain rule. Cost is O(1) forward passes, independent of parameter count — why it dominates deep learning.

### Why reverse mode for ML

For f: ℝⁿ → ℝ (many parameters → scalar loss, the deep-learning case), reverse mode computes the full gradient in roughly the cost of one function evaluation. Forward mode costs scale with n inputs, so it wins only when outputs ≫ inputs (e.g., a few directional derivatives, Hessian-vector products via forward-over-reverse).

## Algorithms / Methods

| Method | Cost (relative to f) | Best when | Accuracy | Notes |
|---|---|---|---|---|
| Manual / symbolic differentiation | varies | small closed-form expressions | exact | "expression swell"; impractical for large programs |
| Numerical (finite differences) | O(n) evals for full gradient | quick checks, gradient-free fallback | approximate (truncation + round-off) | mainly used to verify autodiff (`gradcheck`) |
| Forward-mode autodiff (JVP) | O(n) for full Jacobian, O(1) per input direction | outputs ≫ inputs; directional/tangent derivatives | exact | dual numbers; one pass, no tape needed |
| Reverse-mode autodiff (VJP / backprop) | O(1) for scalar-output gradient | inputs ≫ outputs (ML training) | exact | needs to store/recompute intermediates (memory) |
| Mixed mode (forward-over-reverse) | O(1) per HVP | Hessian-vector products, second order | exact | avoids forming the full Hessian |
| Checkpointing / rematerialization | trades compute for memory | very deep / long-sequence models | exact | recompute activations in backward pass |
| Implicit differentiation | depends on solver | differentiating through fixed points / optimizers / ODEs | exact | adjoint method; underlies Neural ODEs, deep equilibrium models |

## Tools & libraries

| Tool | What it offers | URL |
|---|---|---|
| JAX | Composable autodiff (`grad`, `jvp`, `vjp`, `jacfwd`, `jacrev`, `hessian`) + XLA JIT/vmap | https://github.com/jax-ml/jax |
| PyTorch (autograd) | Dynamic reverse-mode autodiff; `torch.autograd`, `torch.func` (functorch) for JVP/VJP/jacobian | https://pytorch.org/docs/stable/autograd.html |
| TensorFlow | `tf.GradientTape` reverse-mode autodiff over graphs/eager | https://www.tensorflow.org/guide/autodiff |
| Autograd | Original NumPy-transparent reverse/forward autodiff (Maclaurin/Duvenaud) | https://github.com/HIPS/autograd |
| SymPy | Symbolic differentiation / exact closed forms | https://www.sympy.org |
| Enzyme | LLVM-level autodiff for compiled languages (C/C++/Rust/Julia) | https://enzyme.mit.edu |
| Zygote.jl | Source-to-source reverse-mode autodiff for Julia | https://fluxml.ai/Zygote.jl |
| Diffrax | Differentiable ODE/SDE solvers (adjoint/implicit diff) in JAX | https://github.com/patrick-kidger/diffrax |

## Learning resources

- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong) — Ch. 5 "Vector Calculus" covers gradients, Jacobians, backprop, and autodiff. Free PDF: https://mml-book.com
- **The Autodiff Cookbook** — JAX's canonical hands-on guide to JVP/VJP, `jacfwd`/`jacrev`, Hessians: https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html
- **PyTorch — "A Gentle Introduction to torch.autograd"**: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- **CS231n notes — "Backpropagation, Intuitions"** (Karpathy/Stanford): https://cs231n.github.io/optimization-2/
- **3Blue1Brown — Essence of Calculus** (visual intuition for derivatives/chain rule): https://www.3blue1brown.com/topics/calculus
- **3Blue1Brown — "What is backpropagation really doing?"**: https://www.3blue1brown.com/lessons/backpropagation
- **MIT 18.S096 — Matrix Calculus for Machine Learning** (Edelman & Johnson): https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/
- **PyTorch tutorial — Jacobians, Hessians, hvp/vhp with function transforms**: https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html

## Key papers

- Baydin, Pearlmutter, Radul, Siskind — *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018): https://arxiv.org/abs/1502.05767
- Rumelhart, Hinton, Williams — *Learning representations by back-propagating errors* (Nature 1986): https://www.nature.com/articles/323533a0
- Maclaurin, Duvenaud, Adams — *Autograd: Effortless Gradients in NumPy* (ICML AutoML Workshop 2015): https://indico.lal.in2p3.fr/event/2914/contributions/6483/subcontributions/180/attachments/6035/7172/automl-short.pdf
- Paszke et al. — *Automatic Differentiation in PyTorch* (NeurIPS Autodiff Workshop 2017): https://openreview.net/forum?id=BJJsrmfCZ
- Bradbury et al. — *JAX: composable transformations of Python+NumPy programs* (2018): https://github.com/jax-ml/jax
- Chen, Rubanova, Bettencourt, Duvenaud — *Neural Ordinary Differential Equations* (NeurIPS 2018; adjoint/implicit diff): https://arxiv.org/abs/1806.07366
- Griewank, Walther — *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (SIAM, 2nd ed.): https://doi.org/10.1137/1.9780898717761

## Cross-references in AIForge

- [Linear Algebra](../Linear_Algebra/) — Jacobians/Hessians are matrices; matrix calculus depends on it.
- [Convex Optimization](../Convex_Optimization/) and [Optimization Algorithms](../../Optimization_Algorithms/) — gradients feed gradient descent, Newton, and quasi-Newton methods.
- [Numerical Methods](../Numerical_Methods/) — finite differences, conditioning, and stability of derivative computation.
- [Deep Learning](../../Deep_Learning/) — backpropagation is reverse-mode autodiff applied to neural nets.

## Sources

- JAX Autodiff Cookbook & JVP/VJP docs: https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html , https://docs.jax.dev/en/latest/jacobian-vector-products.html
- Baydin et al., AD survey (JMLR/arXiv): https://arxiv.org/abs/1502.05767
- Mathematics for Machine Learning (Deisenroth et al.): https://mml-book.com
- PyTorch autograd documentation: https://pytorch.org/docs/stable/autograd.html
- TensorFlow autodiff guide: https://www.tensorflow.org/guide/autodiff
- Neural ODEs (Chen et al.): https://arxiv.org/abs/1806.07366
- Rumelhart, Hinton & Williams (1986): https://www.nature.com/articles/323533a0
