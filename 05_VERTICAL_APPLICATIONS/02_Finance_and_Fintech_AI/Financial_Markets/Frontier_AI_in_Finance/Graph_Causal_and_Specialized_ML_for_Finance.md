# Graph, Causal & Specialized ML for Finance

> Curated, verified index of under-covered specialized ML subfields applied to financial markets — graph neural networks, causal inference, regime detection, anomaly detection, conformal/Bayesian uncertainty, and explainability — with primary papers, working repos, and an honest read on what actually works. Audience is Brazil-heavy (notes on B3/BDR access where relevant); for the broad exchange/data-API coverage see the sibling pages in this repo (US, B3, NSE/BSE, SSE/SZSE, generic data APIs).

This page deliberately avoids re-listing exchanges and generic data APIs (covered elsewhere). It concentrates on the *methods literature and code* that sit one level deeper than "train an LSTM on OHLCV". Everything below was confirmed against a primary source (arXiv, ACM/IEEE, Cambridge, official GitHub) in 2026.

---

## 0. Orientation — why these subfields

Most public "AI for trading" material stops at supervised price prediction. The high-value, under-indexed frontier is elsewhere: (1) markets are *relational* (stocks co-move through industry, supply chain, ownership, news) → **graphs**; (2) backtested factors are mostly *spurious associations* → **causal inference**; (3) the data-generating process *switches* → **regime models**; (4) the tails contain fraud, manipulation and crashes → **anomaly detection**; (5) point forecasts are dangerous without calibrated **uncertainty**; (6) regulators and risk committees demand **explanations**. The honest theme across all six: these methods rarely beat a well-tuned baseline on *return* prediction, but they are strong on *structure discovery, risk control, and trust* — which is usually where the real money and the real regulation live.

Two excellent meta-resources to anchor the whole field:

| Resource | What it is | Link |
|---|---|---|
| `jwwthu/GNN4Fintech` | Living collection of GNN-for-fintech papers & code | https://github.com/jwwthu/GNN4Fintech |
| `marcuswang6/stock-top-papers` | Top-venue (KDD/WWW/CIKM/AAAI/IJCAI) stock-prediction paper list | https://github.com/marcuswang6/stock-top-papers |
| "From Deep Learning to LLMs: A Survey of AI in Quantitative Investment" (2025) | Recent survey covering GNNs, RL, LLMs for quant | https://arxiv.org/abs/2503.21422 |
| `kyawlin/GNN-finance` | Curated GNN-in-business/finance/banking list | https://github.com/kyawlin/GNN-finance |

---

## 1. Graph Neural Networks (GNNs) for markets

**Idea (PT: *redes neurais em grafos*):** model the cross-section of assets as nodes and their relations (industry/sector, supply chain, shareholding, news co-occurrence, return correlation) as edges, then let message-passing propagate signal between related assets. Two regimes exist: **explicit-graph** methods (you supply the relations, e.g. from Wikidata or sector maps) and **latent-graph / self-attention** methods (the model learns who is connected).

### 1.1 Foundational explicit-relation models

| Method | Core idea | Use | Key paper | Link |
|---|---|---|---|---|
| **RSR / TGC** (Relational Stock Ranking, Temporal Graph Convolution) | LSTM per stock + relation-aware graph conv; trained as a *ranking* loss not regression | Rank stocks by expected return (long-short) | Feng et al., *Temporal Relational Ranking for Stock Prediction*, ACM TOIS 2019 | [arXiv:1809.09441](https://arxiv.org/abs/1809.09441) · code [fulifeng/Temporal_Relational_Stock_Ranking](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) |
| **HATS** | Hierarchical graph attention that *selectively aggregates over multiple relation types* (Wikidata corporate relations) | Movement classification, node/graph tasks | Kim et al. 2019 | [arXiv:1908.07999](https://arxiv.org/abs/1908.07999) · code [dmis-lab/hats](https://github.com/dmis-lab/hats) |
| **FinGAT** | Hierarchical (short/long-term) encoders + *fully-connected* attention graph over stocks AND sectors; no predefined edges; multi-task (rank + movement) | Recommend Top-K profitable stocks | Hsu, Tsai, Li, *FinGAT*, IEEE TKDE 2021 | [arXiv:2106.10159](https://arxiv.org/abs/2106.10159) · code [Roytsai27/Financial-GraphAttention](https://github.com/Roytsai27/Financial-GraphAttention) |

The RSR datasets (NASDAQ/NYSE, sector + Wikidata relations, ~8,000 tickers, 30y) shipped in the repo became a de-facto benchmark — much follow-up work reuses them.

### 1.2 Temporal / dynamic / heterogeneous GNNs (2023–2025 frontier)

The shift is from a *static* relation graph to **a fresh graph per trading day** and from a single edge type to **heterogeneous** edges.

| Method | What's new | Key paper | Link |
|---|---|---|---|
| **THGNN** | Builds a company-relation graph *each day* from recent prices; Transformer temporal encoder + heterogeneous GAT; deployed in a live quant system | Xiang et al., *Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction*, CIKM 2022 | [arXiv:2305.08740](https://arxiv.org/abs/2305.08740) |
| **MDGNN** | Multi-relational *dynamic* GNN for comprehensive stock investment | 2024 | [arXiv:2402.06633](https://arxiv.org/abs/2402.06633) |
| **FinMamba** | Market-aware graph + multi-level Mamba (state-space) for movement prediction | 2025 | [arXiv:2502.06707](https://arxiv.org/abs/2502.06707) |
| **MaGNet** | Mamba dual-*hypergraph* for temporal-causal + global relational learning | 2025 | [arXiv:2511.00085](https://arxiv.org/abs/2511.00085) |
| **STGAT** | Spatial–temporal graph attention for stock prediction | 2025 (Appl. Sci.) | https://www.mdpi.com/2076-3417/15/8/4315 |
| **ChatGPT-informed GNN** | LLM-generated relations as graph edges for movement prediction | 2023 | [arXiv:2306.03763](https://arxiv.org/abs/2306.03763) |
| Survey: Dynamic GNNs | Models/benchmarks/challenges (general, but the canonical map) | 2024 | [arXiv:2405.00476](https://arxiv.org/abs/2405.00476) |

**Knowledge-graph / supply-chain edges.** Beyond price correlation, edges can encode *supply chain*, *shareholding* and *competition* relations (which can carry opposite-signed influence). See the knowledge-injection survey ([arXiv:2308.04947](https://arxiv.org/abs/2308.04947)) and supply-chain-specific GNNs: learning production functions ([arXiv:2407.18772](https://arxiv.org/abs/2407.18772)) and supply-chain risk reasoning with GNNs ([Tandfonline 2022](https://www.tandfonline.com/doi/full/10.1080/00207543.2022.2100841)).

**Honest read.** GNNs reliably help on *ranking / cross-sectional* tasks where relational structure is real (sector momentum spillover, supply-chain shock propagation). Gains over strong temporal baselines are often small and *regime-dependent*, and many papers under-report transaction costs and turnover. The dynamic/per-day-graph methods (THGNN, MDGNN) are the most credible because they don't assume a frozen relation structure. Treat the headline "98% return ratio" style numbers as *upper-bound, cost-free* backtests.

**Brazil note.** B3 has a small liquid universe (~80–100 names with usable depth), so dense relation graphs are easy to build but the cross-section is thin; supply-chain/ownership edges (e.g. Petrobras → suppliers, Vale → steelmakers) are the most economically meaningful. International graphs are accessible via BDRs/ETFs (e.g. IVVB11 for S&P 500 exposure).

---

## 2. Causal inference for markets

**Idea (PT: *inferência causal*):** move from "what co-occurs" (association, the basis of almost all factor backtests) to "what causes what". This is the most intellectually load-bearing section: López de Prado argues most published factors are *false* precisely because they lack a causal graph.

### 2.1 Causal factor investing (the manifesto)

| Work | Thesis | Link |
|---|---|---|
| López de Prado, **Causal Factor Investing: Can Factor Investing Become Scientific?**, Cambridge Univ. Press, *Elements in Quantitative Finance*, 2023 | Factor literature makes associational claims while denying causal content; without a stated causal graph, findings are likely false (backtest overfitting + bad controls/colliders) | [Cambridge](https://www.cambridge.org/core/elements/causal-factor-investing/9AFE270D7099B787B8FD4F4CBADE0C6E) · [SSRN 4205613](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4205613) · free PDF [quantresearch.org](https://www.quantresearch.org/QF_Causal_Factor_Investing.pdf) |
| López de Prado & Zoonekynd, **Causality and Factor Investing: A Primer**, CFA Institute Research Foundation, 2025 | Accessible primer version | [CFA RF landing](https://rpc.cfainstitute.org/research/foundation/2025/causality-factor-investing) · [PDF](https://rpc.cfainstitute.org/sites/default/files/docs/research-reports/rf_lopezdeprado_causalityprimer_online.pdf) · [SSRN 5277078](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5277078) |
| Journal version | Peer-reviewed *Quantitative Finance* article (2024) | [Tandfonline](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2354849) |
| López de Prado, Lipton & Zoonekynd, **The Case for Causal Factor Investing**, 2024 | Short companion arguing the practical case for causal factor models | [SSRN 4774522](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4774522) |

### 2.2 Double / debiased machine learning (econ.EM)

The rigorous way to estimate a *treatment effect* (e.g. effect of a buyback, ESG inclusion, index addition) while controlling for many confounders with ML nuisance models — without the ML regularization bias leaking into the causal estimate. Three ingredients: Neyman orthogonality, cross-fitting (sample splitting), high-quality ML nuisance estimation.

| Tool / paper | Role | Link |
|---|---|---|
| Chernozhukov et al., **Double/Debiased Machine Learning** | Foundational method | [arXiv:1608.00060](https://arxiv.org/abs/1608.00060) |
| **DoubleML** (Python & R) | Object-oriented DML implementation | https://docs.doubleml.org · [Python arXiv:2104.03220](https://arxiv.org/abs/2104.03220) · [R arXiv:2103.09603](https://arxiv.org/abs/2103.09603) |
| **EconML** (PyWhy / Microsoft) | Heterogeneous treatment effects: DML, causal forests, deep IV, meta-learners | https://github.com/py-why/EconML |
| **CausalML** (Uber) | Uplift modeling + causal ML | https://github.com/uber/causalml |
| **DoWhy** (PyWhy) | Causal graph + identification + refutation workflow | https://github.com/py-why/dowhy |

### 2.3 Causal discovery on returns

Learning the causal *graph* directly from return time series (lagged conditional independence, structural causal models).

| Method / tool | Use | Link |
|---|---|---|
| **PCMCI / PCMCI+** (Tigramite) | Time-series causal discovery with FDR control in autocorrelated, high-dim data | https://github.com/jakobrunge/tigramite |
| **causal-learn** (PyWhy) | Python library: PC, GES, LiNGAM, Granger, etc. | https://github.com/py-why/causal-learn · [arXiv:2307.16405](https://arxiv.org/abs/2307.16405) |
| *Causal Discovery in Financial Markets* (nonstationary framework) | Applies discovery to Fama-French factors, sector returns; networks densify in crises | [arXiv:2312.17375](https://arxiv.org/html/2312.17375v2) |
| Helmholtz–Hodge–Kodaira causal hierarchy of the market network | Decomposes the causal network into hierarchy/circulation | [PMC11507571](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11507571/) |

### 2.4 Causal event studies (treatment effects on announcements)

| Work | Contribution | Link |
|---|---|---|
| Goldsmith-Pinkham & Lyu, **Causal Inference in Financial Event Studies**, 2025 | Shows market-model event studies are biased under (near-certain) factor misspecification; recommends diff-in-means, DiD, synthetic control, synthetic DiD | [arXiv:2511.15123](https://arxiv.org/abs/2511.15123) · [PDF](https://paulgp.com/papers/financial_event_studies_nov18.pdf) |
| Synthetic Diff-in-Diff (practitioner) | Hands-on Python | [Causal Inference for the Brave and True, ch.25](https://matheusfacure.github.io/python-causality-handbook/25-Synthetic-Diff-in-Diff.html) |

**Honest read.** Causal methods rarely improve raw return prediction — that's not their job. Their payoff is (a) *killing spurious factors* before you trade them, (b) getting *unbiased* effect estimates for events/policies, and (c) building models that survive distribution shift. The hard part is the *untestable* assumptions (no unobserved confounding, correct graph); on observational market data these are routinely violated, so treat any single causal estimate as a *hypothesis*, not a fact. PT note: the López de Prado primer is the single best Portuguese-accessible entry point conceptually (he is widely read in BR quant circles).

---

## 3. Regime detection & state models

**Idea (PT: *detecção de regimes de mercado*):** the market alternates between latent states (bull/bear, low/high volatility, risk-on/off). Identify the current state and adapt strategy/risk.

| Method | Use | Key paper / resource | Link |
|---|---|---|---|
| **Gaussian HMM** | Classic 2–3 state bull/bear/crisis labeling on returns+vol | Kritzman et al. regime framework; `hmmlearn` | [Kritzman-Regime-Detection](https://github.com/tianyu-z/Kritzman-Regime-Detection) · https://github.com/hmmlearn/hmmlearn |
| **Hidden semi-Markov / HMM regression** | Model state *durations* and regime-switching regressions | Model-based clustering with HMM regression | [arXiv:1312.7024](https://arxiv.org/abs/1312.7024) |
| **Change-point detection** | Detect structural breaks online | `ruptures` (PELT, BinSeg, window) | https://github.com/deepcharles/ruptures |
| **Wasserstein regime clustering** | Non-parametric regime clustering via the Wasserstein *k*-means algorithm (optimal transport on return distributions) | Horvath, Issa & Muguruza, *Clustering Market Regimes Using the Wasserstein Distance*, J. Comp. Finance 28(1) 2024 | [arXiv:2110.11848](https://arxiv.org/abs/2110.11848) · [SSRN 3947905](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3947905) |
| **Signature/MMD regime detection** | Path-dependent *online* regime detection + clustering via signatures and maximum mean discrepancy | Issa & Horvath, *Non-parametric Online Market Regime Detection and Regime Clustering...*, 2023 | [arXiv:2306.15835](https://arxiv.org/abs/2306.15835) |
| **Practitioner pipeline** | HMM regimes + Random Forest signal layer in Python | QuantInsti tutorial | https://blog.quantinsti.com/regime-adaptive-trading-python/ |

**Honest read.** Regime *detection* is genuinely useful for *risk scaling* (cut leverage when the HMM flips to the high-vol state) and works well in-sample. The trap is *latency and look-ahead*: a Gaussian HMM fit on full history "knows" the crisis; in true online use the state-switch is detected *after* much of the move. Treat regimes as a risk overlay, not an alpha. Markov-switching and HMMs label volatility regimes far more reliably than directional (bull/bear *return*) regimes.

---

## 4. Anomaly detection for markets (fraud / manipulation / flash crashes)

**Idea (PT: *detecção de anomalias e manipulação de mercado*):** flag spoofing, layering, quote stuffing, pump-and-dump, wash trading, and flash-crash microstructure — mostly *unsupervised* (labels are scarce) on limit-order-book (LOB) and tick data.

| Method | Target | Key paper / resource | Link |
|---|---|---|---|
| **LSTM autoencoder + GAN** | Sudden return/volume shifts, structural breaks | Robust anomaly detection in financial markets | https://www.opastpublishers.com/open-access-articles/robust-anomaly-detection-in-financial-markets-using-lstm-autoencoders-and-generative-adversarial-networks-9525.html |
| **Deep AD on HFT data** | LOB-level anomalies in high-frequency trading | 2025 | [arXiv:2504.00287](https://arxiv.org/abs/2504.00287) |
| **Cascaded contrastive representation learning** | Multi-level *manipulation* from the LOB | 2025 | [arXiv:2508.17086](https://arxiv.org/abs/2508.17086) |
| **Unsupervised spoofing detection** | Spoofing/layering without labels | DiVA thesis 2024 | https://www.diva-portal.org/smash/get/diva2:1885077/FULLTEXT01.pdf |
| **Isolation Forest + CatBoost** | Market-efficiency anomalies / exploitable inefficiencies | Comp. Economics 2025 | https://link.springer.com/article/10.1007/s10614-025-11274-8 |
| **LLM-augmented surveillance** | Triage of news-driven abuse (LSE uses Amazon Bedrock + Claude) | OxJournal 2025 | https://www.oxjournal.org/fraud-detection-and-market-surveillance-on-the-london-stock-exchange/ |

**Honest read.** Reported true-positive rates (~92–96% vs ~78% legacy) come from vendor/academic settings with curated labels; real surveillance is dominated by *false positives* and the base-rate problem — manipulation is rare, so even a great classifier floods analysts with alerts. Unsupervised AD is the realistic default, and the genuinely hard part is *explaining* a flag well enough to support enforcement (ties directly to §6). For flash crashes specifically, microstructure/LOB features matter far more than daily bars.

**Brazil note.** B3's surveillance (and CVM enforcement) face the same LOB-anomaly problems; the methods transfer directly, and B3 tick/LOB data (via *market data feeds*) is the relevant input — not adjusted close prices.

---

## 5. Uncertainty & probabilistic ML for risk

**Idea (PT: *quantificação de incerteza*):** a point forecast is useless for risk; you need *calibrated* intervals/quantiles. Three families: **conformal prediction** (distribution-free coverage), **Bayesian deep learning** (posterior uncertainty), and **quantile regression** (direct VaR/ES).

### 5.1 Conformal prediction (distribution-free)

| Method / tool | Use | Link |
|---|---|---|
| **MAPIE** (scikit-learn-contrib) | Prediction intervals/sets for regression, classification, *time series*; sklearn-compatible | https://github.com/scikit-learn-contrib/MAPIE · [arXiv:2207.12274](https://arxiv.org/abs/2207.12274) |
| **Adaptive Conformal Inference (ACI)** | Online recalibration under distribution shift (essential for nonstationary returns) | benchmark survey [arXiv:2601.18509](https://arxiv.org/abs/2601.18509) |
| **Conformal Predictive Portfolio Selection** | Aggregate assets into a portfolio return, then conformalize (sidesteps multivariate CP) | [arXiv:2410.16333](https://arxiv.org/abs/2410.16333) |
| **Conformal Risk Control for nonstationary VaR** | Tail-risk control for portfolio VaR | [arXiv:2602.03903](https://arxiv.org/html/2602.03903) |
| CP with change points | CP that survives regime breaks (links to §3) | [OpenReview](https://openreview.net/forum?id=HgLaVgCpCl) |

### 5.2 Bayesian deep learning & quantile/probabilistic risk

| Method | Use | Key paper | Link |
|---|---|---|---|
| **Deep quantile regression for VaR** | Non-linear conditional-quantile VaR; beats linear/MIDAS/CAViaR | Chronopoulos, Raftapostolos, Kapetanios, *J. Financial Econometrics* 2024 | [Oxford Academic](https://academic.oup.com/jfec/article/22/3/636/7163191) |
| **Quantile CNN for VaR** | Convolutional quantile estimator | Sci. of Data 2021 | https://www.sciencedirect.com/science/article/pii/S2666827021000487 |
| **DeepVaR** | Probabilistic deep NN for portfolio risk | Digital Finance 2022 | https://link.springer.com/article/10.1007/s42521-022-00050-0 |
| **Deep distributional forecasting of returns** | *Forecasting Probability Distributions of Financial Returns with Deep Neural Networks* — full predictive return distribution (Normal / Student-t / skewed-t via NLL) | 2025 | [arXiv:2508.18921](https://arxiv.org/abs/2508.18921) |
| **Deep quantile + GANs scenario gen.** | VaR & ES via scenario generation | Financial Innovation 2023 | https://link.springer.com/article/10.1186/s40854-023-00564-5 |

**Honest read.** This is the section where the methods *genuinely* deliver. Conformal prediction gives finite-sample coverage guarantees with almost no assumptions — but the standard guarantee assumes *exchangeability*, which returns violate; you must use the *adaptive/online* variants (ACI) or coverage silently degrades exactly during crises. Deep quantile regression is a solid, deployable VaR upgrade. Bayesian deep nets are theoretically appealing but heavy and the posterior is approximate (MC dropout / variational), so calibrate empirically rather than trusting the credible interval.

---

## 6. Explainability (XAI) for trading & risk models

**Idea (PT: *IA explicável*):** open the black box for risk committees, model validation (SR 11-7 / BR resolutions) and regulators. Dominant techniques in finance: **SHAP**, **attention attribution**, feature importance, LIME, integrated gradients.

| Method | Use | Key resource | Link |
|---|---|---|---|
| **SHAP** | Per-prediction feature attribution for factor/tree/NN models; explains trade executions & risk scores | `shap` library; surveys below | https://github.com/shap/shap |
| **Attention attribution** | Interpret which timesteps/assets/words drove a transformer's call | Survey of XAI in financial time series | [ACM Computing Surveys 2025](https://dl.acm.org/doi/full/10.1145/3729531) |
| **Survey: XAI in finance (systematic)** | Maps SHAP/attention/feature-importance usage 2018–2024 | 2025 | [arXiv:2503.05966](https://arxiv.org/abs/2503.05966) |
| **Comprehensive review: Financial XAI** | Broad reference | 2023 | [arXiv:2309.11960](https://arxiv.org/abs/2309.11960) |
| **CFA Institute: Explainable AI in Finance** | Practitioner/regulatory framing for diverse stakeholders | 2025 | [CFA RF report](https://rpc.cfainstitute.org/research/reports/2025/explainable-ai-in-finance) |
| **Model-agnostic XAI in finance (review)** | Limitations, challenges, future directions | AI Review 2025 | https://link.springer.com/article/10.1007/s10462-025-11215-9 |
| **Interpretability of LLMs in finance** | Beyond-the-black-box for financial LLMs (Barclays) | 2025 | [arXiv:2505.24650](https://arxiv.org/abs/2505.24650) |

**Honest read.** SHAP and attention are *explanations of the model, not of the market* — high attention weight ≠ causal driver (this is where §2 bites). SHAP on correlated factor inputs splits credit arbitrarily across collinear features and can mislead; attention weights are known to be unstable as faithfulness measures. Use XAI for *governance, debugging, and bias detection*, and be skeptical of anyone presenting a SHAP plot as economic insight.

---

## 7. Quick "what actually works" matrix

| Subfield | Best at | Weakest at | Deploy-ready? |
|---|---|---|---|
| GNNs | Cross-sectional ranking, spillover modeling | Stable alpha after costs | Partly (dynamic-graph variants) |
| Causal inference | Killing fake factors, event effects, robustness | Raw return prediction; untestable assumptions | As analysis, yes; as alpha, no |
| Regime models | Risk scaling, vol regimes | Online directional calls (latency) | Yes, as risk overlay |
| Anomaly detection | Surveillance, structural-break flags | Low base rate → false positives | Yes, with human triage |
| Conformal / Bayesian / quantile | Calibrated VaR/intervals | Exchangeability breaks in crises | **Yes — strongest ROI** |
| XAI | Governance, debugging, regulation | Mistaken for causal/economic insight | Yes, with caveats |

---

## 8. Starter repos & libraries (one-line each)

| Repo / lib | Purpose | Link |
|---|---|---|
| `fulifeng/Temporal_Relational_Stock_Ranking` | RSR/TGC reference + NASDAQ/NYSE relation data | https://github.com/fulifeng/Temporal_Relational_Stock_Ranking |
| `dmis-lab/hats` | HATS hierarchical graph attention | https://github.com/dmis-lab/hats |
| `Roytsai27/Financial-GraphAttention` | FinGAT | https://github.com/Roytsai27/Financial-GraphAttention |
| `jwwthu/GNN4Fintech` | GNN-for-fintech paper/code index | https://github.com/jwwthu/GNN4Fintech |
| `py-why/EconML` | Heterogeneous treatment effects / DML | https://github.com/py-why/EconML |
| `py-why/dowhy` | Causal graph → identify → estimate → refute | https://github.com/py-why/dowhy |
| `py-why/causal-learn` | Causal discovery (PC/GES/LiNGAM/Granger) | https://github.com/py-why/causal-learn |
| `jakobrunge/tigramite` | PCMCI time-series causal discovery | https://github.com/jakobrunge/tigramite |
| DoubleML (Python/R) | Double/debiased ML | https://github.com/DoubleML/doubleml-for-py |
| `uber/causalml` | Uplift / causal ML | https://github.com/uber/causalml |
| `deepcharles/ruptures` | Change-point detection | https://github.com/deepcharles/ruptures |
| `hmmlearn/hmmlearn` | HMM regime fitting | https://github.com/hmmlearn/hmmlearn |
| `scikit-learn-contrib/MAPIE` | Conformal prediction intervals | https://github.com/scikit-learn-contrib/MAPIE |
| `shap/shap` | SHAP explanations | https://github.com/shap/shap |

---

## Sources

- Feng et al., *Temporal Relational Ranking for Stock Prediction* — https://arxiv.org/abs/1809.09441 · https://github.com/fulifeng/Temporal_Relational_Stock_Ranking
- Kim et al., *HATS* — https://arxiv.org/abs/1908.07999 · https://github.com/dmis-lab/hats
- Hsu/Tsai/Li, *FinGAT* — https://arxiv.org/abs/2106.10159 · https://github.com/Roytsai27/Financial-GraphAttention
- THGNN — https://arxiv.org/abs/2305.08740 ; MDGNN — https://arxiv.org/abs/2402.06633 ; FinMamba — https://arxiv.org/abs/2502.06707 ; MaGNet — https://arxiv.org/abs/2511.00085
- López de Prado, *Causal Factor Investing* (Cambridge 2023) — https://www.cambridge.org/core/elements/causal-factor-investing/9AFE270D7099B787B8FD4F4CBADE0C6E · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4205613 · https://www.quantresearch.org/QF_Causal_Factor_Investing.pdf
- López de Prado & Zoonekynd, *Causality and Factor Investing: A Primer* (CFA RF 2025) — https://rpc.cfainstitute.org/research/foundation/2025/causality-factor-investing ; *The Case for Causal Factor Investing* (López de Prado, Lipton & Zoonekynd) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4774522
- Chernozhukov et al., *Double/Debiased ML* — https://arxiv.org/abs/1608.00060 ; DoubleML — https://docs.doubleml.org · https://arxiv.org/abs/2104.03220 ; EconML — https://github.com/py-why/EconML
- causal-learn — https://github.com/py-why/causal-learn · https://arxiv.org/abs/2307.16405 ; Tigramite/PCMCI — https://github.com/jakobrunge/tigramite
- Causal Discovery in Financial Markets — https://arxiv.org/html/2312.17375v2
- Goldsmith-Pinkham & Lyu, *Causal Inference in Financial Event Studies* — https://arxiv.org/abs/2511.15123
- Wasserstein regime clustering (Horvath/Issa/Muguruza, J. Comp. Finance 2024) — https://arxiv.org/abs/2110.11848 ; signature/MMD online regime detection (Issa & Horvath) — https://arxiv.org/abs/2306.15835 ; HMM regression clustering — https://arxiv.org/abs/1312.7024 ; ruptures — https://github.com/deepcharles/ruptures
- HFT anomaly detection — https://arxiv.org/abs/2504.00287 ; LOB manipulation (contrastive) — https://arxiv.org/abs/2508.17086 ; LSE AI surveillance — https://www.oxjournal.org/fraud-detection-and-market-surveillance-on-the-london-stock-exchange/
- MAPIE — https://github.com/scikit-learn-contrib/MAPIE · https://arxiv.org/abs/2207.12274 ; Conformal portfolio selection — https://arxiv.org/abs/2410.16333 ; CP benchmark — https://arxiv.org/abs/2601.18509
- Deep quantile VaR (J. Fin. Econometrics 2024) — https://academic.oup.com/jfec/article/22/3/636/7163191 ; DeepVaR — https://link.springer.com/article/10.1007/s42521-022-00050-0
- XAI in financial time series (ACM CSUR 2025) — https://dl.acm.org/doi/full/10.1145/3729531 ; Systematic review of XAI in finance — https://arxiv.org/abs/2503.05966 ; CFA Explainable AI in Finance — https://rpc.cfainstitute.org/research/reports/2025/explainable-ai-in-finance
- Survey: From Deep Learning to LLMs in Quant Investment — https://arxiv.org/abs/2503.21422 ; GNN4Fintech — https://github.com/jwwthu/GNN4Fintech

**Keywords:** graph neural networks finance, GNN stock prediction (redes neurais em grafos), temporal relational ranking, FinGAT, HATS, THGNN, supply-chain graph, knowledge graph stocks, causal inference markets (inferência causal), causal factor investing (fatores causais), double machine learning (aprendizado de máquina duplo/debiased), causal discovery (descoberta causal), PCMCI, EconML, DoWhy, regime detection (detecção de regimes), hidden Markov model (modelo oculto de Markov), change-point detection (detecção de pontos de mudança), market anomaly detection (detecção de anomalias), spoofing/manipulation (manipulação de mercado), flash crash, conformal prediction (predição conformal), Bayesian deep learning, quantile regression VaR (regressão quantílica), value at risk (valor em risco), explainable AI XAI (IA explicável), SHAP, attention attribution, B3, BDR, ETF.
