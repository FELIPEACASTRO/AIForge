# ML & Deep-Learning Models for Financial Markets (Global)

> A country-agnostic model zoo (zoológico de modelos) for financial markets: tabular boosting, classical time-series, deep forecasters, time-series foundation models, limit-order-book nets, reinforcement learning, and finance LLMs — every entry below works on **any** market in the world (equities/ações, ETFs, futures/futuros, options/opções, FX/câmbio, crypto, bonds/renda fixa), because the math operates on price/return series and tabular features, not on a particular exchange or jurisdiction.

## How to read this page

A model is **market-agnostic** when it consumes a generic input — a price/return series, a panel of cross-sectional features, or a limit-order-book snapshot — without hard-coded assumptions about country, currency, or trading calendar. Almost everything below is exactly that. You bring B3 (Brazil), NYSE, LSE, Binance, or CME data; the model neither knows nor cares. The few region-specific items are flagged explicitly. None of these tools provide alpha by themselves — they are estimators; edge comes from your features, labels, validation, and execution assumptions.

> **Universal caveat (vale para tudo):** financial series are low signal-to-noise and non-stationary. Always use walk-forward / purged time-series cross-validation, account for transaction costs and slippage, and beware look-ahead and survivorship bias. A model that "works" on a single backtest is not evidence of edge.

---

## 1) Tabular / Gradient Boosting — the quant workhorses

Gradient-boosted decision trees (GBDT) dominate **cross-sectional alpha** (ranking many assets at one point in time on engineered factors/features). They are fast, handle missing data and mixed feature types, and are the default baseline in most quant pipelines. Fully market/country-agnostic — they only see a feature matrix `X` and a target `y` (e.g., forward return).

| Library | What it does | Link |
|---|---|---|
| **XGBoost** | Scalable regularized GBDT (GBRT); industry-standard for tabular alpha and classification of forward returns; runs on CPU/GPU, Dask, Spark | https://github.com/dmlc/xgboost |
| **LightGBM** | Microsoft histogram-based, leaf-wise GBDT; very fast on large panels of stocks × features; native categorical support | https://github.com/microsoft/LightGBM |
| **CatBoost** | Yandex GBDT with ordered boosting and best-in-class native categorical handling (sectors, tickers, exchanges) | https://github.com/catboost/catboost |
| **scikit-learn** `HistGradientBoosting` | LightGBM-style GBDT inside scikit-learn; convenient baseline with the standard sklearn API | https://scikit-learn.org/stable/modules/ensemble.html |

> Typical use: rank a cross-section of assets each day/week by predicted forward return, then long-top / short-bottom. Combine with proper purged CV to avoid leakage.

---

## 2) Classical Time-Series & Statistics

The econometric baseline. Indispensable for **volatility** (GARCH family) and as honest benchmarks that deep nets frequently fail to beat. All operate on a single (or few) series and are inherently market-agnostic.

| Model / Tool | What it does | Link |
|---|---|---|
| **ARIMA / SARIMAX** (statsmodels) | Autoregressive integrated moving-average with seasonal/exogenous terms; classic price/return forecasting baseline | https://www.statsmodels.org/stable/tsa.html |
| **arch** (GARCH family) | Kevin Sheppard's library for ARCH/GARCH/EGARCH/GJR-GARCH/FIGARCH/APARCH — the standard for **volatility (volatilidade)** and risk forecasting | https://github.com/bashtage/arch |
| **Prophet** | Meta/Facebook additive trend+seasonality+holidays model; robust, low-tuning baseline (better for business/seasonal series than noisy intraday prices) | https://github.com/facebook/prophet |
| **ETS / Exponential Smoothing** | Error-Trend-Seasonality state-space smoothing; strong univariate baseline (Holt-Winters) | https://www.statsmodels.org/stable/tsa.html |
| **Kalman Filter** | State-space filtering/smoothing for latent state, dynamic beta/hedge ratios, pairs trading, noise reduction | https://pykalman.github.io/ |
| **StatsForecast** (Nixtla) | Lightning-fast AutoARIMA, ETS, CES, Theta, GARCH at scale over thousands of series | https://github.com/Nixtla/statsforecast |

> GARCH-family models via `arch` are the single most-used classical tool for FX, equity-index, and crypto volatility forecasting — applicable to any liquid instrument anywhere.

---

## 3) Deep Time-Series Forecasting

Neural sequence models for **temporal** forecasting (one series through time, optionally with covariates). Use these when you have long histories, many related series, or known future inputs. The libraries below implement the named architectures so you rarely code them from scratch. All are market-agnostic — they consume numeric series.

### Architectures

| Architecture | What it does | Reference |
|---|---|---|
| **LSTM / GRU** | Recurrent nets for sequential price/return modeling; classic deep baseline | (in NeuralForecast / TSlib below) |
| **Temporal Fusion Transformer (TFT)** | Attention-based multi-horizon forecaster with interpretable variable selection and known-future covariates | https://arxiv.org/abs/1912.09363 |
| **N-BEATS** | Pure deep residual stacks for univariate forecasting; strong, interpretable | https://arxiv.org/abs/1905.10437 |
| **N-HiTS** | Hierarchical interpolation N-BEATS successor; efficient long-horizon forecasting | https://arxiv.org/abs/2201.12886 |
| **DeepAR** | Autoregressive RNN producing **probabilistic** (quantile) forecasts across many related series | https://arxiv.org/abs/1704.04110 |
| **PatchTST** | Patching + channel-independent Transformer; SOTA long-horizon forecasting ("a time series is worth 64 words") | https://arxiv.org/abs/2211.14730 |
| **TiDE** | Time-series Dense Encoder (MLP) — fast, competitive with Transformers on long horizons | https://arxiv.org/abs/2304.08424 |
| **DLinear / NLinear** | Surprisingly strong linear baselines that often rival Transformers — always test these first | https://arxiv.org/abs/2205.13504 |
| **Autoformer** | Decomposition Transformer with Auto-Correlation for long-term forecasting (NeurIPS 2021) | https://arxiv.org/abs/2106.13008 |
| **Informer** | ProbSparse-attention Transformer for long-sequence forecasting (AAAI 2021 best paper) | https://arxiv.org/abs/2012.07436 |
| **FEDformer** | Frequency-enhanced decomposed Transformer for long-term forecasting | https://arxiv.org/abs/2201.12740 |
| **TimesNet** | 2D temporal-variation modeling; strong general-purpose backbone across tasks | https://arxiv.org/abs/2210.02186 |

### Libraries that implement them

| Library | What it does | Link |
|---|---|---|
| **Nixtla NeuralForecast** | 30+ neural models (NHITS, NBEATS, TFT, PatchTST, TiDE, DeepAR, DLinear, iTransformer, TimeLLM…) with one unified API | https://github.com/Nixtla/neuralforecast |
| **GluonTS** (AWS) | Probabilistic forecasting toolkit (DeepAR, MQ-CNN, Transformers) in PyTorch/MXNet | https://github.com/awslabs/gluonts |
| **PyTorch-Forecasting** | High-level PyTorch-Lightning library; reference TFT, N-BEATS, DeepAR implementations | https://github.com/sktime/pytorch-forecasting |
| **Darts** (Unit8) | User-friendly forecasting + anomaly detection; many models incl. Kalman/GP filters and torch nets | https://github.com/unit8co/darts |
| **sktime** | Unified scikit-learn-style time-series framework (forecasting, classification, pipelines) | https://github.com/sktime/sktime |
| **TSlib / Time-Series-Library** (THUML) | Reference benchmark code for PatchTST, TimesNet, Autoformer, Informer, FEDformer, iTransformer, etc. | https://github.com/thuml/Time-Series-Library |

> Reality check: on noisy financial prices, **DLinear and AutoARIMA are essential baselines**. Deep models earn their keep mainly with covariates, many related series, or long horizons.

---

## 4) Time-Series FOUNDATION Models (pretrained, zero/few-shot)

The 2024–2026 wave: large models pretrained on billions of time points that **forecast any new series with zero or few training examples**. Because they were trained on diverse domains, you can point them at a price/return series from any market and get an immediate forecast — then optionally fine-tune. Maximum market/country-agnosticism by design.

| Model | What it does | Link |
|---|---|---|
| **Amazon Chronos / Chronos-2** | Pretrained TSFM; Chronos-2 (encoder-only, ~120M) adds zero-shot univariate, multivariate, and covariate forecasting; the Chronos family is among the most-downloaded TSFMs on Hugging Face (millions of downloads/month) | https://github.com/amazon-science/chronos-forecasting · https://huggingface.co/amazon/chronos-2 |
| **Google TimesFM** | Decoder-only TSFM pretrained on ~100B time points; v2.5 (200M) with zero-shot forecasting and covariate (XReg) support | https://github.com/google-research/timesfm · https://huggingface.co/google/timesfm-2.5-200m-pytorch |
| **Salesforce Moirai** (uni2ts) | Universal masked-encoder TSFM; Moirai 2.0 is decoder-only, faster, with quantile/probabilistic forecasts; trained on LOTSA | https://github.com/SalesforceAIResearch/uni2ts |
| **Lag-Llama** | First open-source decoder-only **probabilistic** TSFM (uses lags as covariates); strong zero-shot, fine-tunable | https://github.com/time-series-foundation-models/lag-llama · https://huggingface.co/time-series-foundation-models/Lag-Llama |
| **IBM Granite TimeSeries — TinyTimeMixer (TTM)** | "Tiny" (1M+ param) pretrained MLP-Mixer models for fast zero/few-shot multivariate forecasting; r2/r2.1 trained on ~1B samples | https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2 |
| **Nixtla TimeGPT** | Production generative pretrained TSFM via API; zero-shot forecasting + anomaly detection (proprietary; open-source SDK) | https://github.com/Nixtla/nixtla |
| **MOMENT** (CMU Auton Lab) | Open family of TSFMs for forecasting, classification, anomaly detection, imputation, embeddings | https://github.com/moment-timeseries-foundation-model/moment · https://huggingface.co/AutonLab/MOMENT-1-large |

> All of the above can be applied to **any price series** — a Petrobras (PETR4) daily close, an S&P 500 minute bar, or a BTC/USDT tick aggregation — with no retraining required for a first forecast. Validate honestly before trusting zero-shot numbers on noisy markets.

---

## 5) Limit-Order-Book / Microstructure

Models that learn **short-horizon price moves from the order book** (depth, imbalance, queue dynamics). Architecture-agnostic across venues, but you must supply LOB data, which is exchange-specific in format. The benchmark dataset FI-2010 is Nasdaq Nordic equities; the *methods* generalize to any LOB market (equities, futures, crypto).

| Model | What it does | Link |
|---|---|---|
| **DeepLOB** | CNN + LSTM that predicts price-movement direction from raw LOB (IEEE Trans. Signal Processing) | https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books · https://arxiv.org/abs/1808.03668 |
| **TLOB** | Transformer with dual (spatial + temporal) attention for LOB price-trend prediction | https://github.com/LeonardoBerti00/TLOB |
| **lob-deep-learning** | Collection implementing DeepLOB, TransLOB, DeepFolio and others on FI-2010 | https://github.com/Jeonghwan-Cheon/lob-deep-learning |

> These target high-frequency / market-making contexts; they need clean, time-synced LOB feeds and realistic latency/cost modeling to mean anything live.

---

## 6) Reinforcement Learning (trading, execution, portfolio)

RL frames trading as sequential decision-making: an **agent** acts (buy/sell/hold/size) in a market **environment** and learns a policy by trial and error. The frameworks are market-agnostic; you define the environment (any instrument, any country) and reward.

| Tool | What it does | Link |
|---|---|---|
| **FinRL** (AI4Finance) | Educational/research DRL library for trading; ships A2C, DDPG, PPO, TD3, SAC agents and full pipelines | https://github.com/AI4Finance-Foundation/FinRL |
| **FinRL-Meta** | Market environments + benchmarks ("universe" of data-driven financial RL environments) | https://github.com/AI4Finance-Foundation/FinRL-Meta |
| **Stable-Baselines3** | Reliable PyTorch implementations of PPO, A2C, DDPG, TD3, SAC, DQN — the standard single-machine RL backbone for custom trading envs | https://github.com/DLR-RM/stable-baselines3 |
| **Ray RLlib** | Scalable/distributed RL (PPO, SAC, IMPALA, multi-agent) for large-scale execution/portfolio training | https://github.com/ray-project/ray |
| **Gymnasium** | Standard RL environment API; wrap any market simulator to plug into SB3/RLlib | https://github.com/Farama-Foundation/Gymnasium |

> RL for execution (minimizing slippage / market impact) and portfolio allocation tends to be more robust than RL for raw directional trading. Beware overfitting to a single historical regime.

---

## 7) LLMs for Finance

Language models for **financial text** — news, filings, earnings calls, social media — producing sentiment, classification, extraction, summarization, and reasoning. Useful as features (sentiment scores) feeding the models above. Multilingual where noted; flag region/language explicitly.

| Model | What it does | Link |
|---|---|---|
| **FinGPT** (AI4Finance) | Open-source financial LLM framework; FinGPT v3 LoRA-tuned for sentiment on news/tweets; data-centric pipelines | https://github.com/AI4Finance-Foundation/FinGPT · https://huggingface.co/FinGPT |
| **BloombergGPT** | 50B-param finance LLM trained on Bloomberg's corpus — **proprietary, not released** (reference/benchmark only) | https://arxiv.org/abs/2303.17564 |
| **PIXIU / FinMA** (The-FinAI) | Open financial LLM (FinMA 7B/30B) + FIT instruction data + FinBen evaluation benchmark | https://github.com/The-FinAI/PIXIU |
| **FinBERT (ProsusAI)** | Widely-used BERT fine-tuned for financial sentiment (positive/negative/neutral) — English | https://huggingface.co/ProsusAI/finbert |
| **FinBERT-tone (yiyanghkust)** | BERT pretrained on financial communications, fine-tuned on analyst-report tone — English | https://huggingface.co/yiyanghkust/finbert-tone |
| **FinBERT-PT-BR (lucas-leme)** | 🇧🇷 **Brazilian Portuguese** financial sentiment (POSITIVE/NEGATIVE/NEUTRAL); trained on 1.4M PT-BR financial news | https://huggingface.co/lucas-leme/FinBERT-PT-BR |
| **FinBertPTBR (turing-usp)** | 🇧🇷 Alternative Brazilian-Portuguese financial BERT from USP's Turing group (BERTimbau fine-tune; now deprecated in favor of lucas-leme/FinBERT-PT-BR) | https://huggingface.co/turing-usp/FinBertPTBR |

> For Brazil-focused (foco no Brasil) text pipelines, prefer **FinBERT-PT-BR** over English FinBERT — language match matters more than model size for sentiment. The numeric/price models in sections 1–6 need no localization.

---

## 8) Specialized Quant Platforms with built-in models

End-to-end platforms that bundle data handling, feature sets, and a **model zoo** so you can benchmark many models on the same pipeline. Market-agnostic (you supply the market's data).

| Platform | What it does | Link |
|---|---|---|
| **Microsoft Qlib** | AI quant platform with **Alpha158/Alpha360** feature sets + a model zoo (XGBoost, LightGBM, CatBoost, LSTM, GRU, ALSTM, GATs, TRA, Transformers); supervised + RL + market-dynamics modeling | https://github.com/microsoft/qlib |
| **AutoGluon-TimeSeries** | AutoML for forecasting; auto-ensembles statistical + deep + **Chronos-2** foundation models behind one `TimeSeriesPredictor` API | https://auto.gluon.ai/stable/tutorials/timeseries/index.html |
| **Nixtla ecosystem** | StatsForecast + NeuralForecast + MLForecast + TimeGPT — a unified, scalable forecasting stack | https://github.com/Nixtla |
| **AutoGluon (Tabular)** | AutoML that auto-stacks XGBoost/LightGBM/CatBoost/NN for cross-sectional alpha features | https://github.com/autogluon/autogluon |

> Qlib's Alpha158 (158 engineered factors) and Alpha360 (raw price/volume) are defined generically and ship configs for US and China markets; the same feature definitions apply to B3, Europe, or crypto once you load the data.

---

## Choosing a model — quick guidance

| If you need… | Start with |
|---|---|
| Cross-sectional ranking of many assets on factors | LightGBM / XGBoost / CatBoost (§1) |
| Volatility / risk forecasting | GARCH via `arch` (§2) |
| A single-series forecast with little data | A TS foundation model: Chronos-2, TimesFM, Moirai (§4) |
| Long-horizon forecasting with covariates | PatchTST / TFT / TiDE via NeuralForecast (§3) — but benchmark vs DLinear & AutoARIMA |
| High-frequency order-book signals | DeepLOB / TLOB (§5) |
| Learned execution or allocation policy | FinRL + Stable-Baselines3 (§6) |
| Sentiment/text features (PT-BR) | FinBERT-PT-BR (§7) |
| One platform to benchmark many models | Qlib or AutoGluon-TimeSeries (§8) |

---

## Sources

- Amazon Chronos: https://github.com/amazon-science/chronos-forecasting · https://huggingface.co/amazon/chronos-2 · https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting
- Google TimesFM: https://github.com/google-research/timesfm · https://huggingface.co/google/timesfm-2.5-200m-pytorch
- Salesforce Moirai / uni2ts: https://github.com/SalesforceAIResearch/uni2ts · https://www.salesforce.com/blog/moirai-2-0/ · https://arxiv.org/abs/2511.11698
- Lag-Llama: https://github.com/time-series-foundation-models/lag-llama · https://arxiv.org/abs/2310.08278
- IBM Granite TTM: https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2
- Nixtla TimeGPT: https://github.com/Nixtla/nixtla
- MOMENT: https://github.com/moment-timeseries-foundation-model/moment · https://arxiv.org/abs/2402.03885
- Nixtla NeuralForecast / StatsForecast: https://github.com/Nixtla/neuralforecast · https://github.com/Nixtla/statsforecast
- GluonTS: https://github.com/awslabs/gluonts · Darts: https://github.com/unit8co/darts · PyTorch-Forecasting: https://github.com/sktime/pytorch-forecasting · sktime: https://github.com/sktime/sktime · TSlib: https://github.com/thuml/Time-Series-Library
- XGBoost: https://github.com/dmlc/xgboost · LightGBM: https://github.com/microsoft/LightGBM · CatBoost: https://github.com/catboost/catboost
- statsmodels (ARIMA/ETS): https://www.statsmodels.org/stable/tsa.html · arch (GARCH): https://github.com/bashtage/arch · Prophet: https://github.com/facebook/prophet
- DeepLOB: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books · https://arxiv.org/abs/1808.03668 · TLOB: https://github.com/LeonardoBerti00/TLOB · lob-deep-learning: https://github.com/Jeonghwan-Cheon/lob-deep-learning
- FinRL: https://github.com/AI4Finance-Foundation/FinRL · FinRL-Meta: https://github.com/AI4Finance-Foundation/FinRL-Meta · Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3 · Ray RLlib: https://github.com/ray-project/ray
- FinGPT: https://github.com/AI4Finance-Foundation/FinGPT · BloombergGPT: https://arxiv.org/abs/2303.17564 · PIXIU/FinMA: https://github.com/The-FinAI/PIXIU · ProsusAI/finbert: https://huggingface.co/ProsusAI/finbert · finbert-tone: https://huggingface.co/yiyanghkust/finbert-tone · FinBERT-PT-BR: https://huggingface.co/lucas-leme/FinBERT-PT-BR · turing-usp FinBertPTBR: https://huggingface.co/turing-usp/FinBertPTBR
- Microsoft Qlib: https://github.com/microsoft/qlib · AutoGluon-TimeSeries: https://auto.gluon.ai/stable/tutorials/timeseries/index.html · AutoGluon: https://github.com/autogluon/autogluon
- Architecture papers: TFT https://arxiv.org/abs/1912.09363 · N-BEATS https://arxiv.org/abs/1905.10437 · N-HiTS https://arxiv.org/abs/2201.12886 · DeepAR https://arxiv.org/abs/1704.04110 · PatchTST https://arxiv.org/abs/2211.14730 · TiDE https://arxiv.org/abs/2304.08424 · DLinear https://arxiv.org/abs/2205.13504 · Autoformer https://arxiv.org/abs/2106.13008 · Informer https://arxiv.org/abs/2012.07436 · FEDformer https://arxiv.org/abs/2201.12740 · TimesNet https://arxiv.org/abs/2210.02186

**Keywords:** financial markets ML, model zoo, time-series foundation models, gradient boosting, GARCH volatility, LSTM, Temporal Fusion Transformer, PatchTST, Chronos, TimesFM, Moirai, Lag-Llama, TinyTimeMixer, DeepLOB, limit order book, reinforcement learning, FinRL, FinGPT, FinBERT, Qlib, AutoGluon, quant; mercado financeiro, aprendizado de máquina, modelos de série temporal, previsão, volatilidade, ações, futuros, opções, câmbio, criptomoedas, renda fixa, aprendizado por reforço, análise de sentimento financeiro, FinBERT-PT-BR
