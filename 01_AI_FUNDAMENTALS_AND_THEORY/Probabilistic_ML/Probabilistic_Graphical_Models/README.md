# Probabilistic Graphical Models

> A framework that uses graphs to compactly encode a joint probability distribution over many random variables, exposing conditional-independence structure so that representation, inference, and learning become tractable.

## Why it matters

Probabilistic Graphical Models (PGMs) unify probability theory and graph theory to reason under uncertainty over high-dimensional, structured domains where a full joint table would be exponentially large. By factorizing the joint distribution along a graph, PGMs make explicit *which* variables interact, enabling efficient inference, modular knowledge engineering, and learning from data. They underpin diagnostic systems, sensor fusion, NLP/speech (HMMs, CRFs), computer vision (MRFs), and modern causal inference.

## Core concepts

- **Factorization.** A PGM represents a joint distribution as a product of local factors. For a **Bayesian Network** (directed acyclic graph), `P(X_1,...,X_n) = Π_i P(X_i | Pa(X_i))`, where `Pa(X_i)` are the parents of node `i`. For a **Markov Network / MRF** (undirected), `P(X) = (1/Z) Π_c φ_c(X_c)`, a product of non-negative clique potentials `φ_c` normalized by the **partition function** `Z = Σ_X Π_c φ_c(X_c)`.
- **Conditional independence.** The graph encodes independencies. In Bayesian networks these are read off via **d-separation**; in Markov networks via simple graph separation (the global Markov property). The **local Markov property** states each variable is independent of its non-descendants (BN) or non-neighbors (MRF) given its **Markov blanket** (parents + children + children's co-parents for a BN).
- **CPDs and potentials.** Each BN node carries a **Conditional Probability Distribution** (a CPT for discrete vars, or e.g. linear-Gaussian for continuous). MRF factors are unnormalized potentials, often written as a **log-linear / Gibbs** model `P(X) ∝ exp(Σ_k w_k f_k(X))` with features `f_k`.
- **Inference queries.** Main tasks: **marginal inference** `P(Y | E=e)` (sum over hidden vars), **MAP / MPE** `argmax_X P(X | e)` (most probable assignment), and computing the partition function `Z`. Exact inference is #P-hard in general; complexity scales with the graph's **treewidth**.
- **Moralization & triangulation.** To run exact inference, a BN is **moralized** (drop arrow directions, marry co-parents) to an undirected graph, then **triangulated** (chordal) to build a **junction (clique) tree** on which message passing is exact.
- **Learning.** **Parameter learning** estimates CPDs/potentials via MLE or Bayesian (Dirichlet) priors; with hidden variables, **Expectation-Maximization (EM)**. **Structure learning** discovers the graph via score-based search (BIC/BDeu + hill climbing) or constraint-based tests (PC, Hill-Climb). MRF parameter learning requires the costly gradient of `log Z`.
- **Dynamic & relational extensions.** **HMMs** and **Dynamic Bayesian Networks (DBNs)** model temporal processes; **Conditional Random Fields (CRFs)** are discriminative undirected models `P(Y|X)`; plate/relational models share parameters across repeated structure.

## Algorithms / Methods

| Method | Type | Exact/Approx | Use / Notes |
|---|---|---|---|
| Variable Elimination | Inference | Exact | Sum out vars in an ordering; cost driven by induced width |
| Belief Propagation (sum-product) | Inference | Exact on trees | Pearl's message passing; exact only for singly-connected graphs |
| Junction / Clique Tree (Lauritzen–Spiegelhalter, Hugin, Shenoy–Shafer) | Inference | Exact | Moralize + triangulate + message passing over clique tree |
| Max-product / Viterbi | MAP inference | Exact on chains/trees | Most probable explanation; Viterbi is the HMM special case |
| Loopy Belief Propagation | Inference | Approximate | BP on graphs with cycles; no guarantee but often good |
| Variational Inference (mean-field, BP/EP) | Inference | Approximate | Cast inference as optimization of a tractable surrogate |
| Gibbs / MCMC sampling | Inference | Approximate | Sample from conditionals; asymptotically exact |
| Forward–Backward / Baum–Welch | Inference + learning | Exact (HMM) | Marginals & EM parameter learning for HMMs |
| Maximum Likelihood / Bayesian (Dirichlet) estimation | Parameter learning | — | Closed-form for fully-observed discrete BNs |
| Expectation-Maximization (EM) | Parameter learning | — | Latent variables / missing data |
| Hill-Climb + BIC/BDeu, PC, GES | Structure learning | — | Score-based search vs. constraint-based independence tests |

## Tools & libraries

| Tool | Language | URL | Notes |
|---|---|---|---|
| pgmpy | Python | https://github.com/pgmpy/pgmpy | BN/MN, DBN, SEM; inference, parameter & structure learning, causal tools |
| pomegranate | Python (PyTorch) | https://github.com/jmschrei/pomegranate | Fast BNs, HMMs, mixtures; composable distributions |
| PyMC | Python | https://www.pymc.org/ | Probabilistic programming, MCMC/VI for Bayesian models |
| Stan | C++/R/Python | https://mc-stan.org/ | HMC/NUTS sampling for Bayesian inference |
| Pyro / NumPyro | Python | https://pyro.ai/ | Deep probabilistic programming, stochastic VI |
| bnlearn | R | https://www.bnlearn.com/ | Reference BN structure/parameter learning in R |
| OpenGM / pgmax / inferlib (factor graphs) | Python/C++ | https://github.com/google-deepmind/PGMax | Large-scale factor-graph belief propagation (JAX) |
| Microsoft Infer.NET | .NET | https://dotnet.github.io/infer/ | Message-passing inference engine (EP, VMP, Gibbs) |

## Learning resources

- **Koller & Friedman, *Probabilistic Graphical Models: Principles and Techniques*** (MIT Press, 2009) — the definitive ~1200-page reference. https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/
- **Coursera — Probabilistic Graphical Models Specialization** (Daphne Koller, Stanford): [Representation](https://www.coursera.org/learn/probabilistic-graphical-models) · [Inference](https://www.coursera.org/learn/probabilistic-graphical-models-2-inference) · [Learning](https://www.coursera.org/learn/probabilistic-graphical-models-3-learning)
- **Stanford CS228 — Probabilistic Graphical Models** (Stefano Ermon): course site + excellent free [course notes](https://ermongroup.github.io/cs228-notes/). https://cs.stanford.edu/~ermon/cs228/index.html
- **Murphy, *Probabilistic Machine Learning: An Introduction / Advanced Topics*** — modern, freely available textbooks with strong PGM chapters. https://probml.github.io/pml-book/
- **Bishop, *Pattern Recognition and Machine Learning*** — Ch. 8 (Graphical Models) is a classic introduction. https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
- **Barber, *Bayesian Reasoning and Machine Learning*** — free PDF, PGM-centric. http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/online.pdf
- **pgmpy tutorials & examples** — hands-on notebooks. https://github.com/pgmpy/pgmpy_notebook

## Key papers

- J. Pearl (1986). *Fusion, propagation, and structuring in belief networks.* Artificial Intelligence 29(3):241–288. DOI: [10.1016/0004-3702(86)90072-X](https://doi.org/10.1016/0004-3702(86)90072-X)
- S. L. Lauritzen & D. J. Spiegelhalter (1988). *Local Computations with Probabilities on Graphical Structures and Their Application to Expert Systems.* JRSS-B 50(2):157–224. DOI: [10.1111/j.2517-6161.1988.tb01721.x](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x)
- L. R. Rabiner (1989). *A tutorial on hidden Markov models and selected applications in speech recognition.* Proc. IEEE 77(2):257–286. DOI: [10.1109/5.18626](https://doi.org/10.1109/5.18626)
- J. Lafferty, A. McCallum, F. Pereira (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.* ICML 2001. [PDF](https://repository.upenn.edu/cis_papers/159/)
- J. S. Yedidia, W. T. Freeman, Y. Weiss (2005). *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms.* IEEE Trans. Inf. Theory 51(7):2282–2312. DOI: [10.1109/TIT.2005.850085](https://doi.org/10.1109/TIT.2005.850085)
- M. J. Wainwright & M. I. Jordan (2008). *Graphical Models, Exponential Families, and Variational Inference.* Foundations and Trends in ML 1(1–2):1–305. DOI: [10.1561/2200000001](https://doi.org/10.1561/2200000001)
- A. Ankan & A. Panda (2015). *pgmpy: Probabilistic Graphical Models using Python.* Proc. SciPy 2015. DOI: [10.25080/Majora-7b98e3ed-001](https://doi.org/10.25080/Majora-7b98e3ed-001)

## Cross-references in AIForge

- [../../Bayesian_and_Probabilistic_ML](../../Bayesian_and_Probabilistic_ML) — Bayesian inference, priors, probabilistic programming
- [../../Machine_Learning](../../Machine_Learning) — supervised/unsupervised foundations and classical estimators
- [../../Deep_Learning](../../Deep_Learning) — deep generative & structured-prediction models that extend PGMs
- [../../Optimization_Algorithms](../../Optimization_Algorithms) — EM, variational optimization, and gradient methods used for learning

## Sources

- pgmpy — GitHub & docs: https://github.com/pgmpy/pgmpy · https://pgmpy.org/
- pomegranate — GitHub: https://github.com/jmschrei/pomegranate · JMLR: https://jmlr.org/papers/v18/17-636.html
- Koller & Friedman, *PGM: Principles and Techniques*, MIT Press: https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/
- Coursera PGM Specialization: https://www.coursera.org/specializations/probabilistic-graphical-models
- Stanford CS228 course & notes: https://cs.stanford.edu/~ermon/cs228/index.html
- Pearl (1986), Artificial Intelligence: https://doi.org/10.1016/0004-3702(86)90072-X
- Lauritzen & Spiegelhalter (1988), JRSS-B: https://doi.org/10.1111/j.2517-6161.1988.tb01721.x
- Wainwright & Jordan (2008), FnT-ML: https://doi.org/10.1561/2200000001
