# Models & Innovative Techniques for Options Prediction

> A dense, sourced map of ML/quant methods for forecasting and learning options markets — implied/realized volatility, the IV surface under no-arbitrage, option prices/Greeks, direction from options signals, and (deep) hedging — focused on 2023–2026 work with foundational classics, honest about low signal-to-noise, no-arbitrage constraints, overfitting, and live-vs-backtest gaps. Global scope; Brazil/B3 (opções) noted where relevant.

---

## 0. What "prediction" means here (and why it is hard)

"Predicting the options market" is several distinct problems, each with its own loss and its own failure mode. Conflating them is the most common modelling error.

| Target (what you predict) | Typical use | Honest difficulty |
|---|---|---|
| **Realized volatility** (volatilidade realizada) over next horizon | vol trading, risk, position sizing | The HAR benchmark is brutally strong; ML gains are usually marginal in RMSE/QLIKE ([Pollok 2025](https://arxiv.org/abs/2506.07928)). |
| **Implied volatility surface** level/shape and its **dynamics** | quoting, vega risk, scenario gen | Must stay **arbitrage-free**; small fit errors create fake butterflies/calendars. |
| **Option price / Greeks** (surrogate / calibration) | fast repricing, calibration, XVA | Well-posed and learnable; the win is *speed*, not alpha. |
| **Option returns** (delta-hedged) cross-section | systematic options alpha | Real but thin; transaction costs and shorting frictions eat most of it ([Bali et al. 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3895984)). |
| **Underlying direction** from options signals (GEX, flow, P/C) | timing, regime | Mostly *conditioning* (vol regime), weak unconditional directional edge. |
| **Hedging policy** under frictions | P&L variance reduction | Deep hedging works in sim; live transfer is the open question. |

Cross-cutting hazards: **low signal-to-noise** (especially returns), **overfitting** to a single underlying/regime, **no-arbitrage** soft-constraints that hold in-sample but leak out-of-sample, **data cost** (clean intraday option chains and dealer-positioning data are expensive), and **backtest-vs-live** gaps driven by liquidity, microstructure and 0DTE structural change.

---

## 1. Volatility forecasting: baselines → deep sequence models

**Start with strong baselines.** GARCH/EGARCH live in the Python `arch` library; HAR-RV (Corsi 2009) is just OLS on daily/weekly/monthly lagged realized variance and is shockingly hard to beat.

| Technique | Idea | Key reference / repo | Link |
|---|---|---|---|
| GARCH / EGARCH / GJR-GARCH | conditional variance, vol clustering, leverage | `arch` (Kevin Sheppard) | https://github.com/bashtage/arch |
| **HAR-RV** | OLS on RV at day/week/month horizons; long-memory proxy | Corsi (2009), *J. Financial Econometrics* | https://academic.oup.com/jfec/article/7/2/174/856522 |
| Realized-EGARCH / HAR-GARCH | add realized measures + EGARCH leverage | Hansen–Huang (2016), *J. Business & Economic Statistics* | https://www.tandfonline.com/doi/full/10.1080/07350015.2015.1038543 |
| LSTM / GRU on RV | nonlinear memory; LSTM-HAR hybrids feed HAR terms into a recurrent net | Bucci (2020), *J. Fin. Econometrics* | https://academic.oup.com/jfec/article/18/3/502/5614458 |
| **Temporal Fusion Transformer (TFT)** | multi-horizon, interpretable variable selection + attention | Lim et al. (2019) | https://arxiv.org/abs/1912.09363 |
| **PatchTST** | patch tokens + channel independence for long-horizon TS | Nie et al. (2022) | https://arxiv.org/abs/2211.14730 |
| **Optiver-style order-book RV** | predict 10-min ahead RV from LOB/trade micro-features (WAP, book imbalance) | Kaggle *Optiver Realized Volatility Prediction* | https://www.kaggle.com/c/optiver-realized-volatility-prediction |
| Optiver 1st place | nearest-neighbour aggregation over tabular features + NN/GBM blend | discussion write-up | https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970 |
| Vision-Transformer RV (2025) | treat intraday RV as image; data-efficient ViT | arXiv 2511.03046 | https://arxiv.org/abs/2511.03046 |

**Honest benchmark check.** *Predicting Realized Variance Out of Sample: Can Anything Beat The Benchmark?* (Pollok 2025) runs high-dim ML vs HAR across the entire S&P 500 cross-section and finds only **marginal** forecast-error gains — though even small RMSE improvements can be economically meaningful in a vol-trading portfolio. Treat any paper claiming large RV-forecasting alpha with suspicion until it shows QLIKE vs HAR out-of-sample. Source: https://arxiv.org/abs/2506.07928

---

## 2. IV-surface modelling & no-arbitrage learning

The surface (superfície de volatilidade implícita) must be free of **static arbitrage**: no calendar arbitrage (total variance non-decreasing in maturity) and no butterfly arbitrage (convexity in strike / valid risk-neutral density). ML approaches either *parametrize* a known arbitrage-free form or *penalize* violations softly — the latter is convenient but can leak arbitrage out-of-sample.

| Technique | Idea | Key paper / repo | Link |
|---|---|---|---|
| **SSVI** | Surface SVI; provably arbitrage-free under parameter conditions | Gatheral–Jacquier (2014) | https://arxiv.org/abs/1204.0646 |
| **Deep Smoothing of the IVS** | NN fit with soft no-arb penalties + a prior (SSVI) for shape | Ackerer, Tagasovska, Vatter (2019/20) | https://arxiv.org/abs/1906.05065 |
| **Deep Local Volatility** | NN/GP interpolation of prices yielding the full Dupire local-vol surface | Chataigner, Crépey, Dixon (2020), *Risks* | https://arxiv.org/abs/2007.10462 |
| Local vol via shape constraints | hard shape constraints (monotonicity/convexity) on the NN | Chataigner et al. (2022) | https://arxiv.org/abs/2212.09957 |
| DupireNN (repo) | reference implementation of NN local vol via Dupire | mChataign/DupireNN | https://github.com/mChataign/DupireNN |
| **Two-step arbitrage-free forecast** | PCA/VAE features → LSTM forecast → arb-constrained DNN decoder | W. Zhang, Li, G. Zhang (2021) | https://arxiv.org/abs/2106.07177 |
| Gaussian-process surface | GP regression; provably arbitrage-free variant | Chataigner et al. (GP branch, *Risks* 2020) | https://www.mdpi.com/2227-9091/8/3/82 |
| **VAE for the surface** | latent representation; arbitrage-free generation via SDE-parameter VAE | Bergeron, Singh, Vatter, et al. (2021/22), *SIAM J. Fin. Math.* | https://arxiv.org/abs/2108.04941 |
| Controllable VAE surfaces (2025) | conditioned/controllable IV-surface generation | arXiv 2509.01743 | https://arxiv.org/abs/2509.01743 |
| **Operator deep smoothing** | neural operator (mesh-free) for IVS smoothing | arXiv 2406.11520 | https://arxiv.org/abs/2406.11520 |
| PINN intraday IVS | physics-informed "whack-a-mole" online IVS fitting | arXiv 2411.02375 | https://arxiv.org/abs/2411.02375 |

> Practical rule: a soft no-arbitrage penalty (Deep Smoothing) buys you flexibility but you **must** re-check density positivity and calendar monotonicity on held-out dates; hard-constraint / GP methods are safer for risk but less flexible.

---

## 3. Deep calibration & rough volatility

Calibrating stochastic/rough-vol models to the whole surface is slow; the deep-learning trick is to learn the **pricing map** (params → surface) offline, then invert it in milliseconds.

| Technique | Idea | Key paper | Link |
|---|---|---|---|
| **Deep Learning Volatility** | NN approximates pricing map for (rough) vol models; calibrate in ms | Horvath, Muguruza, Tomas (2019/21), *Quant. Finance* | https://arxiv.org/abs/1901.09647 |
| Deep calibration of (rough) SV | first NN calibration "on the fly", incl. **rough Bergomi** | Bayer, Horvath, Muguruza, Stemper, Tomas (2019) | https://arxiv.org/abs/1908.08806 |
| Rough-vol interpretability (2024) | what the calibration nets actually learn | arXiv 2411.19317 | https://arxiv.org/abs/2411.19317 |
| **Arbitrage-free neural-SDE market models** | NN drift/diffusion for tradable factors with no-arb constraints | Cohen, Reisinger, Wang (2021) | https://arxiv.org/abs/2105.11053 |
| Neural-SDE risk of option books | scenario engine for joint option dynamics | Cohen, Reisinger, Wang (2022) | https://arxiv.org/abs/2202.07148 |
| Neural-SDE hedging | sensitivity- & min-variance hedges from the market model | Cohen, Reisinger, Wang (2022) | https://arxiv.org/abs/2205.15991 |
| Robust pricing/hedging via neural SDEs | worst-case bounds with NN-SDEs | Gierjatowicz et al. (2020) | https://arxiv.org/abs/2007.04154 |
| **Signature-based models** | rough-path signatures as universal features for price/vol dynamics & calibration | Cuchiero, Gazzani, Svaluto-Ferro (2023), *SIAM J. Fin. Math.* | https://epubs.siam.org/doi/10.1137/22M1512338 |

---

## 4. Deep Hedging & RL

**Deep Hedging** (Buehler, Gonon, Teichmann, Wood) reframes hedging as direct optimisation of a convex risk measure of terminal P&L under frictions — no Greeks, no model-implied deltas required. RL variants handle path-dependence and inventory.

| Technique | Idea | Key paper / repo | Link |
|---|---|---|---|
| **Deep Hedging** | NN policy minimises risk measure of hedged P&L under transaction costs | Buehler et al. (2018/19), *Quant. Finance* | https://arxiv.org/abs/1802.03042 |
| Deep Hedging under frictions (RL) | generic market frictions via RL on convex risk measures | Buehler et al. (2019), SSRN | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3355706 |
| Deep Hedging (official repo) | TensorFlow reference engine | hansbuehler/deephedging | https://github.com/hansbuehler/deephedging |
| PFHedge | PyTorch deep-hedging framework | pfnet-research/pfhedge | https://github.com/pfnet-research/pfhedge |
| Continuous RL hedging (multi-risk-aversion) | one policy across portfolios & risk aversions | arXiv 2207.07467 | https://arxiv.org/abs/2207.07467 |
| Deep RL for option hedging (2025) | benchmark of DRL algos for hedging | arXiv 2504.05521 | https://arxiv.org/abs/2504.05521 |
| Dynamic hedging via RL | option dynamic hedging with RL | arXiv 2306.10743 | https://arxiv.org/abs/2306.10743 |
| **Avellaneda–Stoikov + deep RL** | double-DQN tunes AS risk-aversion/skew for market making | Falces Marín, Díaz Pardo de Vera, López Gonzalo (2022), *PLOS ONE* | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277042 |
| MM with deep RL from LOB | end-to-end MM policy from order books | arXiv 2305.15821 | https://arxiv.org/abs/2305.15821 |

> Brutally honest: deep-hedging P&L-variance reductions are largely demonstrated **in-simulation** (Heston, rough Bergomi). Live transfer depends on the simulator matching real cost/impact/liquidity. A policy that beats BS delta in sim can underperform live if the market generator is wrong.

---

## 5. Pricing surrogates & Greeks

Here ML is on solid ground: you are approximating a known function, so the payoff is **speed and differentiability**, not alpha.

| Technique | Idea | Key paper / repo | Link |
|---|---|---|---|
| **Differential ML (twin nets)** | train on payoffs **and** AAD pathwise differentials → fast prices + Greeks | Huge, Savine (2020) | https://arxiv.org/abs/2005.02347 |
| Differential ML (repo) | official TensorFlow notebooks | differential-machine-learning | https://github.com/differential-machine-learning/notebooks |
| **Deep BSDE** | solve high-dim pricing PDEs/BSDEs with NNs | Han, Jentzen, E (2018), *PNAS* | https://www.pnas.org/doi/10.1073/pnas.1718942115 |
| Deep BSDE delta-gamma hedging (2025) | high-dim multi-asset Bermudan pricing + hedging (Negyesi, Oosterlee) | arXiv 2502.11706 | https://arxiv.org/abs/2502.11706 |
| Deep optimal stopping | NN stopping rules for American/Bermudan | Becker, Cheridito, Jentzen (2019), JMLR | https://www.jmlr.org/papers/v20/18-232.html |
| Longstaff–Schwartz (LSM) + ML | regression-based American pricing; replace basis regression with ML | Longstaff–Schwartz (2001), *RFS* | https://escholarship.org/uc/item/43n1k4jb |
| DL vs Black–Scholes on **B3/Petrobras** | ResNet pricer on PETR options, hybrid loss, 2016–2025 B3 data | Gueiros, Chandravamsi, Frankel (2025) | https://arxiv.org/abs/2504.20088 |

---

## 6. Direction / return prediction from options signals

This is where overfitting and data-snooping are most dangerous. The literature finds **real but modest** predictability, mostly from *informed* option order flow rather than crude ratios.

| Signal / method | What it claims | Evidence | Link |
|---|---|---|---|
| **O/S ratio** (option-to-stock volume) | high O/S predicts underlying returns (informed trading) | Johnson–So (2012); Ge, Lin, Pearson (2016), *JFE* | https://www.sciencedirect.com/science/article/abs/pii/S0304405X16000167 |
| Why options info predicts returns | decomposes the options→stock predictability into informed-trading vs non-fundamental channels | Muravyev, Pearson, Pollet (2025), *JFE* | https://www.sciencedirect.com/science/article/pii/S0304405X25001618 |
| **Option return ML (cross-section)** | nonlinear ML on 265 characteristics predicts delta-hedged option returns; survives costs | Bali, Beckmeyer, Moerke, Weigert (2023), *RFS* ("Option Return Predictability with Machine Learning and Big Data") | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3895984 |
| End-to-end DL options trading | learn trading rule directly from option features | Tan, Roberts, Zohren (2024), ACM ICAIF | https://arxiv.org/abs/2407.21791 |
| **Gamma Exposure (GEX)** | dealer-gamma sign conditions vol regime (long-gamma dampens, short-gamma amplifies) | SpotGamma methodology (vendor) | https://spotgamma.com/gamma-exposure-gex/ |
| 0DTE gamma & intraday flow | 0DTE >50% of SPX volume; single-expiry GEX models miss it | MenthorQ / vendor research | https://menthorq.com/guide/understanding-0dte-gamma-exposure/ |

> Be honest with readers: **GEX is conditioning, not a directional crystal ball.** It says *how* moves may behave (compressed vs amplified), not *which way*. Option-return ML alpha (Bali et al.) is real in academic samples but concentrated in illiquid, hard-to-short contracts — the live, net-of-cost edge is far smaller than headline Sharpe ratios suggest.

---

## 7. Generative & frontier (2023–2026)

| Technique | Idea | Key paper / repo | Link |
|---|---|---|---|
| **VolGAN** | GAN for arbitrage-free IV surfaces + joint underlying dynamics; data-driven hedges | Vuletić, Cont (2025), *Applied Math. Finance* | https://www.tandfonline.com/doi/full/10.1080/1350486X.2025.2471317 |
| **FuNVol** | functional PCA + neural SDE multi-asset IV market simulator | Choudhary, Jaimungal, Bergeron (2023) | https://arxiv.org/abs/2303.00859 |
| Diffusion IVS forecast | DDPM forecasting of next-day surface with SNR-weighted arb penalty | Jin, Agarwal (2025) | https://arxiv.org/abs/2511.07571 |
| VolaDiff (repo) | diffusion model for arbitrage-free IVS forecasts | JPNotleks/VolaDiff | https://github.com/JPNotleks/VolaDiff |
| Hybrid conv-VAE crypto surfaces (2026) | VAE for crypto vol surfaces | arXiv 2606.16961 | https://arxiv.org/abs/2606.16961 |
| Meta-learning neural process IVS (2025) | SABR-prior neural process for sparse surfaces | arXiv 2509.11928 | https://arxiv.org/abs/2509.11928 |
| **0DTE differential ML (2026)** | differential ML for 0DTE under SV+jumps, PIDE-residual penalty | arXiv 2603.07600 | https://arxiv.org/abs/2603.07600 |
| LLMs for options sentiment/strategy | financial word embeddings / LLM features feeding vol & flow models | Rahimikia, Zohren, Poon (financial word-embedding RV) | https://arxiv.org/abs/2108.00480 |

---

## 8. Data, tooling & the Brazil/B3 angle

| Resource | What | Link |
|---|---|---|
| `arch` (GARCH/EGARCH/realized) | the standard Python vol library | https://github.com/bashtage/arch |
| Optiver RV dataset | LOB + trades for RV prediction | https://www.kaggle.com/c/optiver-realized-volatility-prediction |
| CBOE / SPX, VIX, 0DTE | exchange data & methodology for the deepest options market | https://www.cboe.com/ |
| **S&P/B3 Ibovespa VIX** | 30-day implied-vol index for Brazil (launched 19 Mar 2024) | https://www.b3.com.br/en_us/market-data-and-indices/indexes/indices-in-partnership-with-s-p-dowjones/volatility-indices.htm |
| Variance premium in low-liquidity (B3) | IVol-BR index + variance premium for Brazil's thin option market (Astorino, Chague, Giovannetti, da Silva) | https://www.scielo.br/j/rbe/a/RjTbzmbfFtJyNSTWgNYmxNQ/ |

**B3 specifics (opções).** Liquidity concentrates in a handful of names (PETR4, VALE3, the big banks, BOVA11), so cross-sectional option-return ML and surface models calibrated on US chains do **not** transfer naively. Brazilian chains are sparser, with wider spreads and stronger expiry-clustering of liquidity (commonly 1–3 month expiries). The March 2024 launch of the S&P/B3 Ibovespa VIX finally gives a clean implied-vol benchmark for IV-surface and vol-forecasting research on Brazilian equities; the Gueiros et al. (2025) Petrobras study is a concrete template for DL pricing on B3 data scraped over Nov 2016–Jan 2025.

---

## 9. A pragmatic recipe (and traps to avoid)

1. **Always beat HAR before believing your deep model.** Report QLIKE and RMSE out-of-sample; if you can't beat HAR-RV, you have a bug or a story, not a model.
2. **Choose the right no-arbitrage tool.** Risk/pricing → hard constraints or GP; flexible quoting/scenarios → soft penalty (Deep Smoothing/VAE/GAN) **plus** an out-of-sample arbitrage audit.
3. **Separate speed from alpha.** Differential ML, deep BSDE, deep calibration are *surrogates* — they make things fast, they do not create edge.
4. **For returns/flow, assume the costs are real.** Bali et al. and O/S-ratio studies are credible, but spreads, borrow and capacity shrink the live edge; 0DTE structurally changed SPX gamma dynamics, so pre-2022 backtests may not generalise.
5. **For hedging, stress the simulator.** Deep-hedging gains are only as trustworthy as the market generator (Heston/rough Bergomi/neural-SDE) you trained on.
6. **Guard against single-underlying overfitting.** Validate across tickers, regimes and (for Brazil) across B3's liquidity cliffs.

---

**Sources:** arXiv ([1802.03042](https://arxiv.org/abs/1802.03042), [1901.09647](https://arxiv.org/abs/1901.09647), [1908.08806](https://arxiv.org/abs/1908.08806), [1906.05065](https://arxiv.org/abs/1906.05065), [2007.10462](https://arxiv.org/abs/2007.10462), [2106.07177](https://arxiv.org/abs/2106.07177), [2108.04941](https://arxiv.org/abs/2108.04941), [2105.11053](https://arxiv.org/abs/2105.11053), [2202.07148](https://arxiv.org/abs/2202.07148), [2205.15991](https://arxiv.org/abs/2205.15991), [2005.02347](https://arxiv.org/abs/2005.02347), [2502.11706](https://arxiv.org/abs/2502.11706), [2303.00859](https://arxiv.org/abs/2303.00859), [2511.07571](https://arxiv.org/abs/2511.07571), [2506.07928](https://arxiv.org/abs/2506.07928), [2504.20088](https://arxiv.org/abs/2504.20088), [2407.21791](https://arxiv.org/abs/2407.21791), [1912.09363](https://arxiv.org/abs/1912.09363), [2211.14730](https://arxiv.org/abs/2211.14730), [2603.07600](https://arxiv.org/abs/2603.07600)); PNAS Han–Jentzen–E; *Review of Financial Studies* Bali et al. (SSRN 3895984); *PLOS ONE* Avellaneda–Stoikov RL; *SIAM J. Fin. Math.* signatures; *J. Financial Econometrics* (Corsi, Bucci); *J. Business & Economic Statistics* (Hansen–Huang); GitHub (hansbuehler/deephedging, pfnet-research/pfhedge, mChataign/DupireNN, JPNotleks/VolaDiff, bashtage/arch); Kaggle Optiver; CBOE; B3 / SciELO; SpotGamma / MenthorQ.

**Keywords:** options market prediction, implied volatility surface, no-arbitrage neural network, realized volatility forecasting, HAR-RV, GARCH/EGARCH, deep hedging, deep calibration, rough Bergomi, neural-SDE market models, signatures, differential machine learning, deep BSDE, Longstaff-Schwartz, VolGAN, diffusion IV surface, gamma exposure (GEX), 0DTE, option return predictability, B3, opções, volatilidade implícita, volatilidade realizada, superfície de volatilidade, ausência de arbitragem, apreçamento de opções, cobertura dinâmica (hedge), gregas (Greeks), aprendizado de máquina, fluxo de opções
