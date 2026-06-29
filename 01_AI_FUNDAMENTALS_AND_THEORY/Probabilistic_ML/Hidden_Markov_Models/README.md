# Hidden Markov Models

> A Hidden Markov Model (HMM) is a doubly stochastic process in which an unobserved (hidden) sequence of discrete states evolves as a Markov chain, and each state emits an observable symbol or vector according to a state-dependent emission distribution.

## Why it matters

HMMs are the canonical tractable model for sequential data with latent structure, powering decades of speech recognition, part-of-speech tagging, bioinformatics (gene finding, sequence alignment), and financial regime detection. They offer exact, efficient inference via dynamic programming (forward-backward, Viterbi) and unsupervised parameter learning via Baum-Welch (EM), making them a foundational stepping stone toward modern sequence models such as CRFs, state-space models, and RNNs/Transformers.

## Core concepts

An HMM with `N` hidden states and observation space is specified by `λ = (A, B, π)`:

- **Transition matrix** `A`, where `a_ij = P(q_{t+1}=j | q_t=i)` — probability of moving from state `i` to state `j`. Rows sum to 1.
- **Emission distribution** `B`, where `b_j(o_t) = P(o_t | q_t=j)` — probability (discrete: a multinomial; continuous: e.g. a Gaussian or Gaussian mixture) of emitting observation `o_t` from state `j`.
- **Initial distribution** `π`, where `π_i = P(q_1=i)`.

**Markov + output independence assumptions.** The next state depends only on the current state (first-order Markov property), and the current observation depends only on the current state. This conditional-independence structure is what makes inference tractable.

The three classic problems (Rabiner 1989):

1. **Evaluation / likelihood** — compute `P(O | λ)` for an observation sequence `O = o_1...o_T`. Solved by the **forward algorithm** using `α_t(i) = P(o_1...o_t, q_t=i | λ)` with recursion `α_{t+1}(j) = [Σ_i α_t(i) a_ij] b_j(o_{t+1})`. Cost is `O(N²T)` instead of the naive `O(N^T)`.
2. **Decoding** — find the most likely hidden state sequence `argmax_Q P(Q | O, λ)`. Solved by the **Viterbi algorithm**, a max-product dynamic program `δ_t(j) = max_i [δ_{t-1}(i) a_ij] b_j(o_t)` with back-pointers.
3. **Learning** — estimate `λ` to maximize `P(O | λ)`. Solved by **Baum-Welch**, an EM instance using forward (`α`) and backward (`β_t(i) = P(o_{t+1}...o_T | q_t=i)`) variables to form posteriors `γ_t(i)` (state occupancy) and `ξ_t(i,j)` (transition occupancy), then re-estimating `A, B, π` in closed form.

**Numerical note.** Products of many small probabilities underflow; implementations work in log-space or use per-time-step scaling factors. Baum-Welch only finds a local optimum, so multiple random restarts and informed initialization matter.

## Algorithms / Methods

| Algorithm | Problem solved | Idea | Complexity |
|---|---|---|---|
| Forward algorithm | Likelihood `P(O\|λ)` | Sum-product DP over forward variable `α` | `O(N²T)` |
| Backward algorithm | Posterior building block | DP over backward variable `β` | `O(N²T)` |
| Forward-backward | Smoothing `P(q_t\|O)` | Combine `α` and `β` into posteriors `γ`, `ξ` | `O(N²T)` |
| Viterbi | Most likely path | Max-product DP with back-pointers | `O(N²T)` |
| Baum-Welch (EM) | Parameter learning | Iterative E-step (forward-backward) + M-step re-estimation | `O(N²T)` per iteration |
| Viterbi training (segmental k-means) | Approx. learning | Hard-assign best path, then re-estimate | Faster, less accurate |
| Scaling / log-space variants | Numerical stability | Avoid underflow in `α`, `β` products | same order |

## Variants

| Variant | What changes | Typical use |
|---|---|---|
| Discrete (multinomial) HMM | Categorical emissions | Text, DNA/protein symbols |
| Gaussian / GMM-HMM | Continuous Gaussian (mixture) emissions | Acoustic features in speech |
| Left-to-right (Bakis) HMM | Constrained transitions (no return) | Speech, gesture, handwriting |
| Input-Output HMM (IOHMM) | Transitions/emissions conditioned on inputs | Controlled sequences |
| Autoregressive HMM (ARHMM) | Emissions depend on previous observations | Time-series with momentum |
| Hidden Semi-Markov (HSMM) | Explicit state-duration distributions | Activity/segment modeling |
| Factorial HMM | Multiple parallel hidden chains | Combinatorial latent structure |
| Hierarchical HMM | States are themselves HMMs | Multi-scale structure |
| Bayesian / HDP-HMM | Priors; infinite states via HDP | Unknown number of states |

## Tools & libraries

| Tool | Language | Notes | Link |
|---|---|---|---|
| hmmlearn | Python | scikit-learn-style API; Gaussian, GMM, categorical, Poisson HMMs; Baum-Welch & Viterbi | https://github.com/hmmlearn/hmmlearn |
| pomegranate | Python (PyTorch backend) | General-purpose probabilistic modeling; dense/sparse HMMs, GMMs, Bayes nets | https://github.com/jmschrei/pomegranate |
| statsmodels | Python | Markov-switching (regime-switching) models for time series | https://www.statsmodels.org/stable/examples/index.html#statespace |
| ssm (Linderman lab) | Python | HMMs, ARHMMs, SLDS, recurrent state-space models | https://github.com/lindermanlab/ssm |
| HMMBase.jl | Julia | Lightweight HMM inference/estimation | https://github.com/maxmouchet/HMMBase.jl |
| depmixS4 | R | Dependent mixture / hidden Markov models | https://cran.r-project.org/package=depmixS4 |
| HMMER | C/CLI | Profile HMMs for biological sequence search | http://hmmer.org/ |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| Rabiner, "A Tutorial on Hidden Markov Models..." (1989) | Foundational tutorial paper | https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf |
| Jurafsky & Martin, *Speech and Language Processing* — Appendix A: HMMs | Textbook chapter (free PDF) | https://web.stanford.edu/~jurafsky/slp3/A.pdf |
| Bishop, *Pattern Recognition and Machine Learning* — Ch. 13 (Sequential Data) | Textbook | https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/ |
| Murphy, *Probabilistic Machine Learning: Advanced Topics* — HMM / SSM chapters | Textbook (free PDF) | https://probml.github.io/pml-book/book2.html |
| Eisner, "An Interactive Spreadsheet for Teaching the Forward-Backward Algorithm" | Hands-on tutorial | https://www.cs.jhu.edu/~jason/papers/eisner.tnlp02.pdf |
| hmmlearn tutorial & user guide | Library docs | https://hmmlearn.readthedocs.io/en/latest/tutorial.html |
| Ghahramani, "An Introduction to Hidden Markov Models and Bayesian Networks" | Survey | https://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf |

## Key papers

| Year | Paper | Venue | Link |
|---|---|---|---|
| 1966 | Baum & Petrie, *Statistical Inference for Probabilistic Functions of Finite State Markov Chains* | Ann. Math. Statist. 37(6) | https://doi.org/10.1214/aoms/1177699147 |
| 1967 | Viterbi, *Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm* | IEEE Trans. Inf. Theory 13(2) | https://doi.org/10.1109/TIT.1967.1054010 |
| 1970 | Baum, Petrie, Soules & Weiss, *A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains* | Ann. Math. Statist. 41(1) | https://doi.org/10.1214/aoms/1177697196 |
| 1989 | Rabiner, *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition* | Proc. IEEE 77(2) | https://doi.org/10.1109/5.18626 |
| 1997 | Ghahramani & Jordan, *Factorial Hidden Markov Models* | Machine Learning 29 | https://doi.org/10.1023/A:1007425814087 |
| 2017 | Schreiber, *pomegranate: Fast and Flexible Probabilistic Modeling in Python* | JMLR 18 | https://jmlr.org/papers/v18/17-636.html |
| 2017 | Yang, Balakrishnan & Wainwright, *Statistical and Computational Guarantees for the Baum-Welch Algorithm* | JMLR 18 | https://jmlr.org/papers/volume18/16-093/16-093.pdf |

## Cross-references in AIForge

- [Mixture Models and EM](../Mixture_Models_and_EM/) — Baum-Welch is the EM algorithm applied to sequential latent variables.
- [Probabilistic Graphical Models](../Probabilistic_Graphical_Models/) — HMMs are the simplest dynamic Bayesian network / chain-structured graphical model.
- [Variational Inference](../Variational_Inference/) — approximate inference for Bayesian and intractable HMM extensions.
- [State Space Models](../../State_Space_Models/) — continuous-state cousins (Kalman filters, SLDS) and modern deep SSMs.

## Sources

- Rabiner (1989), Proc. IEEE — https://doi.org/10.1109/5.18626
- Baum & Petrie (1966), Ann. Math. Statist. — https://doi.org/10.1214/aoms/1177699147
- Viterbi (1967), IEEE Trans. Inf. Theory — https://doi.org/10.1109/TIT.1967.1054010
- hmmlearn — https://github.com/hmmlearn/hmmlearn ; docs https://hmmlearn.readthedocs.io/en/latest/
- pomegranate — https://github.com/jmschrei/pomegranate ; JMLR https://jmlr.org/papers/v18/17-636.html
- Yang, Balakrishnan & Wainwright (2017), JMLR — https://jmlr.org/papers/volume18/16-093/16-093.pdf
- Jurafsky & Martin, *Speech and Language Processing*, Appendix A — https://web.stanford.edu/~jurafsky/slp3/A.pdf
