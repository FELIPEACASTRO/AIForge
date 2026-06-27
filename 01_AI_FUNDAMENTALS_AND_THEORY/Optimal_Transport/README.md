# Optimal Transport

> Optimal transport (OT) is the mathematics of moving one probability distribution onto another at minimal cost — yielding Wasserstein distances, Sinkhorn solvers, and couplings that underpin WGANs, domain adaptation, single-cell biology, and flow/diffusion generative models.

## Why it matters

OT gives a geometry-aware way to compare distributions: unlike KL or TV, the Wasserstein distance accounts for the underlying metric, stays meaningful for distributions with disjoint support, and provides a transport *plan* (a soft matching) as a by-product. Entropic regularization (Sinkhorn) makes it differentiable and GPU-parallel, turning OT into a practical loss for deep learning. It now anchors the theory of flow matching, rectified flows, and Schrödinger bridges, and is a workhorse in single-cell trajectory inference, color transfer, and barycenter averaging.

## Taxonomy

| Sub-area | What it solves | Representative methods |
|---|---|---|
| Discrete / Kantorovich OT | LP coupling between weighted point clouds | network simplex, auction algorithm |
| Entropic OT (Sinkhorn) | Regularized OT; fast, differentiable | Sinkhorn iterations, Sinkhorn divergences |
| Wasserstein distance | Metric between distributions (W1, W2) | Kantorovich-Rubinstein duality |
| Sliced / projected OT | 1-D projections to dodge curse of dimension | Sliced-Wasserstein, max-sliced, generalized SW |
| Unbalanced OT | Differing total mass / outliers | KL-relaxed marginals, FUGW |
| Gromov-Wasserstein (GW) | Match spaces with no shared metric | GW, fused GW, entropic GW |
| Neural / continuous OT | Learn Monge maps via networks | ICNN maps, neural entropic OT |
| Dynamic OT / bridges | Geodesics & stochastic couplings over time | Benamou-Brenier, Schrödinger bridge, flow matching |
| Wasserstein barycenters | Averaging distributions | free-support & fixed-support barycenters |

## Key tools and frameworks

| Tool | Focus | Link |
|---|---|---|
| POT (Python Optimal Transport) | Broad reference library: EMD, Sinkhorn, GW, barycenters, DA | https://github.com/PythonOT/POT |
| OTT-JAX | JAX toolbox: jit/vmap/diff Sinkhorn, low-rank, GW, neural maps | https://github.com/ott-jax/ott |
| GeomLoss | Fast Sinkhorn / Hausdorff / energy losses via KeOps | https://github.com/jeanfeydy/geomloss |
| KeOps | Symbolic kernel/GPU backend powering geometric OT | https://github.com/getkeops/keops |
| TorchCFM / flow matching | Conditional flow matching with OT couplings | https://github.com/atong01/conditional-flow-matching |
| moscot | OT for single-cell genomics (built on OTT-JAX) | https://github.com/theislab/moscot |

## Key methods (named building blocks)

| Method | Idea | Reference |
|---|---|---|
| Sinkhorn algorithm | Entropic regularization → matrix scaling, GPU-friendly | https://arxiv.org/abs/1306.0895 |
| Wasserstein GAN | W1 critic via Kantorovich-Rubinstein duality | https://arxiv.org/abs/1701.07875 |
| WGAN-GP | Gradient penalty replaces weight clipping | https://arxiv.org/abs/1704.00028 |
| Sinkhorn divergences | Debiased, positive-definite OT loss for generative models | https://arxiv.org/abs/1706.00292 |
| Sliced-Wasserstein discrepancy | 1-D projections for tractable distribution matching / DA | https://arxiv.org/abs/1903.04064 |
| Neural entropic OT / GW | Replace Sinkhorn with a learned network at scale | https://arxiv.org/abs/2312.07397 |

## Key papers

| Year | Paper | Link |
|---|---|---|
| 2013 | Sinkhorn Distances: Lightspeed Computation of Optimal Transport (Cuturi) | https://arxiv.org/abs/1306.0895 |
| 2017 | Wasserstein GAN (Arjovsky, Chintala, Bottou) | https://arxiv.org/abs/1701.07875 |
| 2017 | Improved Training of Wasserstein GANs / WGAN-GP (Gulrajani et al.) | https://arxiv.org/abs/1704.00028 |
| 2017 | Learning Generative Models with Sinkhorn Divergences (Genevay, Peyré, Cuturi) | https://arxiv.org/abs/1706.00292 |
| 2018 | Computational Optimal Transport (Peyré & Cuturi) — the standard textbook/survey | https://arxiv.org/abs/1803.00567 |
| 2019 | Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation (Lee et al.) | https://arxiv.org/abs/1903.04064 |
| 2022 | Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein | https://arxiv.org/abs/2201.12324 |
| 2023 | Neural Entropic Optimal Transport and Gromov-Wasserstein Alignment | https://arxiv.org/abs/2312.07397 |

## Cross-references in AIForge

- [Generative Models](../Generative_Models/) — WGANs, flow matching, and diffusion all build on OT couplings.
- [Domain Adaptation](../Domain_Adaptation/) — Wasserstein and sliced-Wasserstein alignment of feature distributions.
- [Information Theory](../Information_Theory/) — entropic regularization links OT to entropy and divergences.
- [Optimization Algorithms](../Optimization_Algorithms/) — Sinkhorn, network simplex, and convex duality solvers.

## Sources

- POT library — https://github.com/PythonOT/POT
- OTT-JAX — https://github.com/ott-jax/ott
- GeomLoss — https://github.com/jeanfeydy/geomloss
- Computational Optimal Transport (Peyré & Cuturi) — https://arxiv.org/abs/1803.00567
- Sinkhorn Distances (Cuturi, 2013) — https://arxiv.org/abs/1306.0895
- Wasserstein GAN — https://arxiv.org/abs/1701.07875
- WGAN-GP — https://arxiv.org/abs/1704.00028
- Sinkhorn Divergences (Genevay et al.) — https://arxiv.org/abs/1706.00292
- Sliced Wasserstein Discrepancy for UDA — https://arxiv.org/abs/1903.04064
- OTT JAX Toolbox paper — https://arxiv.org/abs/2201.12324
- Neural Entropic OT and GW Alignment — https://arxiv.org/abs/2312.07397

_Expanded from a verified high-value gap seed. Contributions welcome (see CONTRIBUTING.md)._
