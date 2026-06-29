# Market Simulation & Synthetic Market Data

> Frontier tools for **simulating financial markets** and **generating synthetic market data** (agent-based simulators, "market generators", GANs, diffusion models, neural SDEs, RL trading gyms) — used to train ML models, stress-test strategies, and backtest without overfitting to a single realised price history. Curated, verified, current to 2025-2026, with Brazil (B3) access notes.

## Why this matters (data scarcity, regime coverage, robustness)

Real markets give you **exactly one realised path of history**. That is poison for data-hungry ML: deep-hedging networks, RL execution agents, and forecasting models need *millions* of interactions and *many* plausible market regimes (crashes, rallies, low-vol grind) that the single observed history rarely contains. Synthetic data and simulators address four concrete pains:

| Problem | What simulation/synthesis provides |
|---|---|
| **Data scarcity** (escassez de dados) | Generate unlimited realistic paths from a learned distribution; augment small datasets (e.g. emerging-market tickers, new B3 listings). |
| **Regime coverage** | Sample tail events, volatility clusters, and stress scenarios under-represented in the realised path — key for risk and stress-testing (*testes de estresse*). |
| **Backtest overfitting** (*sobreajuste de backtest*) | Validating strategies across many synthetic histories reduces over-fitting to one lucky/unlucky realisation; a strategy that only works on the single observed path is suspect. |
| **Counterfactuals & market impact** (*impacto de mercado*) | Agent-based LOB simulators let your agent's orders *move the book*, so you can model market impact and train RL execution agents in a reactive market — impossible on static replayed tape. |

Two evaluation truths to internalise before trusting any generator: (1) generated series must reproduce **stylized facts** (*fatos estilizados*) — fat tails, volatility clustering, leverage effect, near-zero return autocorrelation; (2) "looks realistic in a plot" ≠ "survives a trading backtest". Recent work stresses backtest-aware evaluation: *Can GANs Learn the Stylized Facts of Financial Time Series?* (Kwon & Lee, ICAIF'24, arXiv <https://arxiv.org/abs/2410.09850> · ACM <https://dl.acm.org/doi/10.1145/3677052.3698661>) and *Beyond Visual Realism: Toward Reliable Financial Time Series Generation* (Zhang et al., ICASSP'26, <https://arxiv.org/abs/2601.12990>).

---

## 1. Agent-based market simulators (ABM / discrete-event LOB)

Agent-based models (ABMs) populate a simulated exchange with many heterogeneous trading agents whose interactions *endogenously* produce price dynamics and a reactive limit order book (LOB / *livro de ofertas*). Unlike historical replay, your agent's orders affect the market.

| Tool / paper | Type | Use | Link |
|---|---|---|---|
| **ABIDES** (Agent-Based Interactive Discrete Event Simulation) | ABM / discrete-event LOB | Nanosecond-resolution exchange with NASDAQ-style ITCH/OUCH messaging; tens of thousands of agents; high-fidelity LOB data generation & microstructure experiments. *(Original repo archived June 2025 — use the JPMC fork below.)* | <https://github.com/abides-sim/abides> |
| **ABIDES-JPMC (public)** | ABM (maintained fork) | J.P. Morgan's maintained ABIDES with `abides-core` + `abides-markets` + `abides-gym`; recommended current entry point | <https://github.com/jpmorganchase/abides-jpmc-public> |
| **ABIDES-Gym** (Amrouni et al., ICAIF'21) | ABM → RL gym wrapper | Wraps multi-agent discrete-event sim as OpenAI Gym envs (daily investor, execution agent) | <https://arxiv.org/abs/2110.14771> |
| **ABIDES-MARL** (Cheridito, Dupret, Wu 2025) | Multi-agent RL env | Endogenous price formation & execution in a LOB; extends ABIDES-Gym for synchronized multi-agent RL | <https://arxiv.org/abs/2511.02016> |
| **MarS / Large Market Model (LMM)** (Microsoft Research, ICLR'25) | Order-level generative foundation model | Order-by-order generative "foundation model" for market simulation; models market impact, controllable/realistic scenarios incl. rare events; open-source engine (`mlib`) — model weights gated on HF | paper <https://arxiv.org/abs/2409.07486> · code <https://github.com/microsoft/MarS> |
| **JAX-LOB** (Frey et al., ICAIF'23) | GPU-accelerated LOB sim | GPU LOB simulator; simulates *thousands* of books in parallel via JAX → fast end-to-end RL for execution | paper <https://arxiv.org/abs/2308.13289> · code <https://github.com/KangOxford/jax-lob> |
| **mbt_gym** (Jerome, Sánchez-Betancourt, Savani, Herdegen, ICAIF'23) | Model-based LOB gym | Vectorised gym envs for market-making & optimal execution (Avellaneda-Stoikov-style models); plugs into Stable-Baselines3 | paper <https://arxiv.org/abs/2209.07823> · ACM <https://dl.acm.org/doi/10.1145/3604237.3626873> · code <https://github.com/JJJerome/mbt_gym> |
| **Limit Order Book Simulations: A Review** (Jain et al. 2024) | Survey | Taxonomy of point-process, ABM, and deep-learning LOB simulators — orientation map | <https://arxiv.org/abs/2402.17359> |

**Notes.** ABIDES is the de-facto research standard (originated at Georgia Tech / J.P. Morgan AI Research); the original `abides-sim/abides` repo is archived, so start from `jpmorganchase/abides-jpmc-public`. For GPU-scale RL, JAX-LOB and mbt_gym are the current go-tos. MarS (2024-25) is the highest-profile "foundation model" take on order-level simulation. The 2024 review (arXiv 2402.17359) is the best single map of the landscape.

---

## 2. Generative "market generators" — GANs

The "market generator" idea (learn the distribution of market paths, then sample new ones) was crystallised by **Kondratyev & Schwarz (2019), *The Market Generator*** — a Restricted Boltzmann Machine that replicates the dependence structure of risk factors (SSRN <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3384948>; Risk.net <https://www.risk.net/cutting-edge/7401191/the-market-generator>). GANs then became the dominant family.

| Tool / paper | Type | Use | Link |
|---|---|---|---|
| **Quant GANs** (Wiese, Knobloch, Korn, Kretschmer 2019/2020) | GAN (Temporal Conv Net) | Generate single-asset return series with volatility clusters, leverage effect, fat tails; risk-neutral transition | paper <https://arxiv.org/abs/1907.06673> · impl <https://github.com/JamesSullivan/temporalCN> |
| **Conditional Sig-Wasserstein GAN (SigCWGAN)** (Liao, Ni, Szpruch, et al. 2020) | GAN + path signature | Sig-W₁ metric as discriminator + AR-FNN generator; principled time-series GAN validated on financial data | paper <https://arxiv.org/abs/2006.05421> · code <https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs> |
| **Sig-Wasserstein GANs** (Ni et al. 2021) | GAN + signature | Generalised Sig-WGAN framework | <https://arxiv.org/abs/2111.01207> |
| **Fin-GAN** (Vuletić, Prenzel, Cucuringu, *Quant. Finance* 2024) | GAN (economics-driven loss) | Forecast *and* classify returns; PnL/Sharpe-aware generator loss; beats LSTM/ARIMA on Sharpe | paper <https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2299466> · code <https://github.com/milenavuletic/Fin-GAN> |
| **TimeGAN** (Yoon, Jarrett, van der Schaar, NeurIPS'19) | GAN (general TS) | Embedding + supervised + adversarial losses; widely-used baseline for multivariate TS incl. finance | paper <https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks> · code <https://github.com/jsyoon0823/TimeGAN> |
| **Deep Hedging: Learning to Simulate Equity Option Markets** (Wiese, Bai, Wood, Buehler 2019) | GAN (option surfaces) | GAN market generator for *equity option markets* → training data for deep hedging | arXiv <https://arxiv.org/abs/1911.01700> · SSRN <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3470756> |
| **Robust Hedging GANs** (Limmer & Horvath 2023) | GAN (robustness) | Automated robustification of hedging strategies via adversarial market generators | <https://arxiv.org/abs/2307.02310> |
| **Stylized Facts Alignment GAN (SFAG)** / *Beyond Visual Realism* (Zhang et al. 2026) | GAN (constrained) | Converts stylized facts into differentiable constraints; backtest-aware generation | <https://arxiv.org/abs/2601.12990> |

**Brazil angle.** Quant GANs / SigCWGAN are asset-agnostic — point them at B3 ticker returns (e.g. PETR4, VALE3, BOVA11) to augment scarce intraday history; signature methods are robust to short samples, useful for newly-listed *ações*.

---

## 3. Diffusion models, VAEs & neural SDEs for market data

Diffusion models are the current frontier for both tabular/return-series synthesis and **order-level LOB simulation** — this is the most active 2025-2026 research area.

| Tool / paper | Type | Use | Link |
|---|---|---|---|
| **TSDiff** — *Predict, Refine, Synthesize* (Kollovieh et al., Amazon, NeurIPS'23) | Diffusion (general TS) | Unconditional diffusion + self-guidance for forecasting & synthetic TS generation | paper <https://neurips.cc/virtual/2023/poster/70377> · code <https://github.com/amazon-science/unconditional-time-series-diffusion> |
| **TRADES** — *Generating Realistic Market Simulations with Diffusion Models* (Berti, Prenkaj, Velardi 2025) | Diffusion (order-level LOB) | Order-by-order LOB generation; counterfactual analysis via agent interaction; ships **DeepMarket** framework | paper <https://arxiv.org/abs/2502.07071> · code <https://github.com/LeonardoBerti00/DeepMarket> |
| **LOBS5** (Nagy et al. 2023) | Deep state-space (S5) LOB | Token-level autoregressive message-by-order LOB generator on a deep state-space net; strong LOB-Bench baseline | paper <https://arxiv.org/abs/2309.00638> · code <https://github.com/peernagy/LOBS5> |
| **Painting the Market** (Backhouse et al. 2025) | Diffusion + inpainting | LOB → image format, diffusion inpainting for simulation *and* forecasting; strong on LOB-Bench | <https://arxiv.org/abs/2509.05107> |
| **DiffVolume** (Wang & Ventre 2025) | Diffusion (LOB volumes) | Generates high-dimensional LOB volume snapshots across price levels | <https://arxiv.org/abs/2508.08698> |
| **DiffLOB** (Wang & Ventre 2026) | Diffusion (counterfactuals) | Counterfactual generation in limit order books conditioned on future market conditions | <https://arxiv.org/abs/2602.03776> |
| **CoFinDiff** (Tanaka et al. 2025) | Controllable diffusion | Conditioned financial TS generation (target stylized facts/regimes) | <https://arxiv.org/abs/2503.04164> |
| **Generation of synthetic financial TS by diffusion models** (Takahashi & Mizuno 2024) | Diffusion (returns) | Wavelet-image DDPM on returns/volume/spread; reproduces fat tails, vol clustering | <https://arxiv.org/abs/2410.18897> |
| **Robust pricing & hedging via neural SDEs** (Gierjatowicz et al. 2020) | Neural SDE | Itô process with NN drift/vol; generative model linked to causal optimal transport for robust pricing/hedging | <https://arxiv.org/abs/2007.04154> |
| **Neural SDEs for Conditional TS Generation + Sig-W₁** (Díaz Lozano et al. 2023) | Neural SDE + signature | Conditional generation with Signature-Wasserstein-1 metric | <https://arxiv.org/abs/2301.01315> |
| **Deep Generators on Commodity Markets** (Boursin et al. 2022) | VAE/GAN generators | Market generators applied to commodity markets → deep hedging | <https://arxiv.org/abs/2205.13942> |

**Signature-based generators** deserve a callout: path signatures give a model-free, theoretically-grounded feature map of a price path. See also *Generating drawdown-realistic financial price paths using path signatures* (Lemahieu, Boudt, Wyns; <https://arxiv.org/abs/2309.04507>).

---

## 4. RL trading environments / gyms

Reinforcement-learning environments are where simulators and ML training meet: an agent acts (place/cancel orders, allocate), the env returns reward (PnL, risk-adjusted). Gym/Gymnasium API is the lingua franca.

| Tool | Type | Use | Link |
|---|---|---|---|
| **FinRL-Meta** (AI4Finance, NeurIPS'22 DataCentricAI workshop) | RL env universe | Hundreds of near-real market environments + data engineering layer; GPU multiprocessing; companion to FinRL | paper <https://arxiv.org/abs/2112.06753> · code <https://github.com/AI4Finance-Foundation/FinRL-Meta> |
| **ABIDES-Gym** | ABM-backed RL env | Reactive multi-agent LOB market as Gym (execution + daily investor) | <https://github.com/jpmorganchase/abides-jpmc-public> |
| **mbt_gym** | Model-based RL env | Market-making / optimal execution; vectorised; SB3-ready | <https://github.com/JJJerome/mbt_gym> |
| **Gym-Trading-Env** (Perroud) | Gymnasium env | Fast, customizable single-instrument trading env; good for quick RL prototyping | <https://github.com/ClementPerroud/Gym-Trading-Env> |
| **gym-anytrading** (AminHP) | Gym env | Minimal, flexible trading envs (stocks/forex); popular teaching baseline | <https://github.com/AminHP/gym-anytrading> |
| **TradingGym** (Yvictor) | Gym-style env | Training + backtesting env for RL or rule-based algos | <https://github.com/Yvictor/TradingGym> |
| **gym-trading** (hackthemarket) | Gym env | Historical bar-data single-instrument RL env (classic) | <https://github.com/hackthemarket/gym-trading> |

**Pick guide.** Need a *reactive* market with market impact → ABIDES-Gym, mbt_gym, or MarS. Need breadth of assets + data pipeline → FinRL-Meta. Need a quick toy env → Gym-Trading-Env / gym-anytrading.

---

## 5. Synthetic tabular finance data & privacy

For *tabular* finance (credit, transactions, KYC, risk factors) rather than price series, the SDV / synthcity ecosystems dominate. Relevant for fraud, credit scoring, and privacy-preserving data sharing (LGPD/GDPR — *proteção de dados*).

| Tool | Type | Use | Link |
|---|---|---|---|
| **SDV** (Synthetic Data Vault, DataCebo/MIT) | Tabular framework | One-stop tabular synthesis: GaussianCopula → CTGAN → TVAE; single/multi-table & sequential | <https://github.com/sdv-dev/SDV> |
| **CTGAN** (Xu et al., NeurIPS'19) | Conditional tabular GAN | Mixed-type tabular synthesis (continuous + categorical); workhorse for finance tables | <https://github.com/sdv-dev/CTGAN> |
| **synthcity** (van der Schaar Lab) | Tabular + TS + privacy | Broad generator zoo (static, regular/irregular TS, censored, DP); fairness/privacy/augmentation metrics | paper <https://arxiv.org/abs/2301.07573> · code <https://github.com/vanderschaarlab/synthcity> |
| **TSGM** (Nikitin, Iannucci, Kaski) | TS generative framework | Generative + simulation-based + augmentation methods; multi-angle eval (similarity, downstream, privacy); Keras/TF/Torch/JAX | paper <https://arxiv.org/abs/2305.11567> · code <https://github.com/AlexanderVNikitin/tsgm> |
| **dp_cgans** | DP conditional GAN | Differential-privacy tabular/RDF synthesis | <https://github.com/sunchang0124/dp_cgans> |

**Privacy note.** Synthetic data is *not automatically anonymous* — membership-inference and reconstruction attacks exist. Use synthcity's privacy metrics (or DP-trained generators like `dp_cgans`) and validate before sharing data externally.

---

## 6. Evaluation, benchmarks & surveys

| Resource | What it gives | Link |
|---|---|---|
| **LOB-Bench** (Nagy et al., ICML'25) | Standard benchmark for generative LOB models in LOBSTER format: spread, imbalance, inter-arrival, adversarial & market-impact scores | paper <https://arxiv.org/abs/2502.09172> · site <https://lobbench.github.io/> · code <https://github.com/peernagy/lob_bench> |
| **Can GANs Learn the Stylized Facts...?** (Kwon & Lee, ICAIF'24) | Sobering empirical check of GAN fidelity on stylized facts | arXiv <https://arxiv.org/abs/2410.09850> · ACM <https://dl.acm.org/doi/10.1145/3677052.3698661> |
| **Beyond Visual Realism** (Zhang et al. 2026) | Backtest-aware evaluation + SFAG constrained generator | <https://arxiv.org/abs/2601.12990> |
| **Limit Order Book Simulations: A Review** (Jain et al. 2024) | Survey/taxonomy of LOB simulators | <https://arxiv.org/abs/2402.17359> |
| **GANs in time series: a survey & taxonomy** (Brophy et al. 2021) | Map of TS-GAN architectures (incl. finance) | <https://arxiv.org/abs/2107.11098> |
| **Systematic comparison of deep generative models on multivariate financial TS** (Caulfield & Gleeson 2024) | Head-to-head GAN/VAE/diffusion vs parametric comparison | <https://arxiv.org/abs/2412.06417> |

**Evaluation checklist** before you trust synthetic market data: (1) **stylized facts** — heavy tails, no linear return autocorrelation, volatility clustering, leverage effect, gain/loss asymmetry; (2) **dependence** — cross-asset correlations and tail co-movement; (3) **downstream / TSTR** (Train-on-Synthetic-Test-on-Real) — does a model trained on synthetic data perform on real?; (4) **backtest survival** — does a strategy tuned on synthetic data hold up out-of-sample?; (5) **privacy** if tabular.

---

## Already covered elsewhere in this repo (do not duplicate)

Exchange/data-access pages exist for **US (NYSE/Nasdaq)**, **Brazil (B3)**, **India (NSE/BSE)**, **China (SSE/SZSE)**, **Japan/HK/Korea/Taiwan**, **ASEAN**, generic global market-data APIs, Kaggle/HuggingFace dataset pulls, and the arXiv **q-fin** feed. This page is strictly the *simulation & synthetic-data* frontier layer on top of those.

## Brazil access notes (acesso no Brasil)

- All tools here are open-source Python and run locally on B3 data — no special access needed. Feed them `.SA` tickers (Yahoo) or B3 market data.
- For **deep-hedging / RL on Brazilian assets**, augment scarce intraday/option history (e.g. options on PETR4, VALE3, or the IBOV index) with Quant GANs / SigCWGAN, then validate against real tape.
- Global ETFs/BDRs (e.g. BDRs of US names, BOVA11 for IBOV) widen the asset universe you can simulate while staying tradeable from a Brazilian brokerage account.
- **LGPD**: for synthetic *tabular* customer/transaction data, prefer synthcity/`dp_cgans` with privacy metrics — synthetic ≠ anonymized by default.

---

**Keywords:** market simulation, synthetic market data, agent-based market simulation, limit order book, LOB, ABIDES, MarS, Large Market Model, JAX-LOB, market generator, Quant GANs, Sig-Wasserstein GAN, signatures, TimeGAN, diffusion models, TRADES, LOBS5, LOB-Bench, neural SDE, deep hedging, reinforcement learning trading, FinRL-Meta, trading gym, CTGAN, SDV, synthcity, TSGM, stylized facts, backtest overfitting, market impact, stress testing — simulação de mercado, dados sintéticos de mercado, simulação baseada em agentes, livro de ofertas, gerador de mercado, redes adversárias generativas, modelos de difusão, hedge profundo, aprendizado por reforço, ambiente de negociação, dados tabulares sintéticos, fatos estilizados, sobreajuste de backtest, impacto de mercado, testes de estresse, B3, ações, escassez de dados
