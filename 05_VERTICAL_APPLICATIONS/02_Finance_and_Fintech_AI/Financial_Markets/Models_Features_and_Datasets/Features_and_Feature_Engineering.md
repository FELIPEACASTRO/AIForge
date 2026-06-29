# Features & Feature Engineering for Markets (Global)

> Country-agnostic catalogue of features and feature-engineering tooling for market machine learning — usable on **any** instrument in **any** market worldwide (equities/ações, ETFs, futures/futuros, options/opções, FX/câmbio, crypto, bonds/títulos). Everything here is general-purpose: the formulas operate on OHLCV, order-book, fundamental, or text inputs and do not assume a specific exchange, currency, or regulator. Region-specific items are flagged explicitly.

This page is the feature-construction companion to the AIForge financial-markets index. The guiding principle throughout: **features are only as good as the discipline behind them** — point-in-time (PIT) correctness, look-ahead avoidance, leakage control, and proper scaling matter more than any individual indicator. Where a transform can leak future information (rolling stats, normalization, target encoding), that risk is called out.

---

## 0. Why this is global (and why discipline dominates)

A momentum feature, an RSI, a realized-volatility estimate, or a FinBERT sentiment score is just an arithmetic transform of inputs that exist on every venue on Earth. None of them encode a country. What changes by market is the **data plumbing** (tick size, trading calendar, settlement, corporate-action conventions, currency) — not the feature math. So you can run the same `pandas-ta`/`tsfresh`/Qlib pipeline on B3 (Brazil), NYSE, Xetra, NSE India, Binance, or CME, provided you wire in the correct calendar and adjusted prices.

| Cross-cutting rule | What it means | Why it matters |
|---|---|---|
| Point-in-time (PIT) | Only use data **known at the timestamp** of the row | Fundamentals/ratings are restated and lagged; using as-reported values leaks the future |
| Look-ahead avoidance | Rolling/expanding windows must end at or before `t`; no centered windows | Centered moving averages and `df.fillna(method='bfill')` import future bars |
| Leakage control | Fit scalers/encoders on **train only**, then transform val/test | Global `StandardScaler.fit(X)` leaks test-set statistics |
| Survivorship-free universe | Include delisted names in the historical universe | Surviving-only universes inflate every backtested factor |
| Scaling/normalization | Cross-sectional (per-date) ranks/z-scores avoid temporal leakage better than time-series scaling | Per-date ranks are naturally PIT-safe |

---

## 1. Price / Return Features (Preços e Retornos)

The backbone. Computable from a single OHLCV series, identical math everywhere.

| Feature | Definition / formula | Notes & PIT |
|---|---|---|
| Simple return | `r_t = P_t / P_{t-1} - 1` | Use **adjusted** prices (splits/dividends) |
| Log return | `ℓ_t = ln(P_t / P_{t-1})` | Time-additive; preferred for vol/aggregation |
| Multi-horizon momentum | `P_t / P_{t-k} - 1` for k ∈ {5,21,63,126,252} | Classic 12-1 momentum = past 12m skipping last 1m |
| Lags | `r_{t-1}, …, r_{t-n}` | Cheap; watch for over-fitting |
| Rolling stats | rolling mean/std/min/max/skew/kurtosis | Window must **end at t** (`min_periods`, no center) |
| Z-score | `(x_t - μ_window) / σ_window` | Rolling z-score is PIT-safe; global z-score leaks |
| Normalization | min-max, robust, quantile | Fit on train only |
| Detrending | subtract rolling mean / linear trend / HP filter | HP filter is two-sided → look-ahead; avoid online |
| Fractional differentiation | `(1-B)^d` with fractional `d` | Stationarity **while preserving memory** (see below) |

**Fractional differentiation (López de Prado).** Integer differencing (`d=1`, i.e. returns) makes a series stationary but erases its memory; using a *fractional* `d` finds the minimum differencing that passes a stationarity test (ADF) while keeping maximal correlation with the original level series. This is Chapter 5 of *Advances in Financial Machine Learning*. Reference implementation `frac_diff_ffd()` (fixed-width window fracdiff) lives in **mlfinlab** (https://github.com/hudson-and-thames/mlfinlab); a lightweight alternative is **fracdiff** (https://github.com/fracdiff/fracdiff), and a maintained re-implementation is **mlfinpy** (https://mlfinpy.readthedocs.io/en/latest/FractionalDifferentiated.html). Source paper context: arXiv mirror discussions and the book by Marcos López de Prado (Wiley, 2018).

---

## 2. Technical Indicators (Análise Técnica)

Deterministic transforms of OHLCV. Group by what they measure. All apply to any instrument with a price/volume series.

| Group | Indicators | Measures |
|---|---|---|
| Trend (tendência) | SMA, EMA, WMA, MACD, ADX, Ichimoku, PSAR | Direction / regime |
| Momentum | RSI, Stochastic %K/%D, ROC, CCI, Williams %R, TSI | Speed of change / overbought-oversold |
| Volatility | Bollinger Bands, ATR, Keltner, Donchian | Dispersion / range |
| Volume | OBV, VWAP, MFI, A/D line, Chaikin, CMF | Participation / conviction |

### Libraries (all general-purpose, market-agnostic)

| Library | What it does | Link |
|---|---|---|
| TA-Lib | C-backed industry standard, 150+ indicators + candlestick patterns; needs C build | https://github.com/TA-Lib/ta-lib-python |
| pandas-ta | Pandas extension, 130+ indicators + 60 candlestick patterns; pure-Python, DataFrame `.ta` accessor | https://github.com/twopirllc/pandas-ta |
| pandas-ta-classic | Community-maintained continuation, 250+ indicators/patterns, no TA-Lib required | https://github.com/xgboosted/pandas-ta-classic |
| ta (bukosabino) | Pandas/Numpy, 40+ indicators, "add all" one-liner for feature engineering | https://github.com/bukosabino/ta |
| tulipy | Python bindings to Tulip Indicators (fast C lib) | https://github.com/cirla/tulipy |
| FinTA | Pure-pandas implementation of common indicators, easy to read | https://github.com/peerchemist/finta |

> Practical note: `pandas-ta` is the most ergonomic for ML feature matrices (`df.ta.strategy("all")`), while TA-Lib is the fastest at scale. Indicators are **highly collinear** — prune with correlation/feature-importance before modeling.

---

## 3. Volatility & Risk Features (Volatilidade e Risco)

Risk descriptors are predictive features and targets in their own right.

| Feature | Formula / idea | Notes |
|---|---|---|
| Realized volatility (RV) | √(Σ rₜ²) over window; intraday sum of squared returns | Annualize by √(periods/yr) |
| Parkinson | uses High–Low range only | ~5× more efficient than close-to-close; ignores gaps |
| Garman–Klass | uses OHLC | More efficient than Parkinson; assumes no drift, no overnight jump |
| Rogers–Satchell | OHLC, drift-independent | Handles trending series |
| Yang–Zhang | combines overnight + Rogers–Satchell | Robust to opening gaps; most efficient of the OHLC family |
| GARCH outputs | conditional variance σ²ₜ, persistence (α+β) | Fit with `arch` |
| Beta | cov(rₐ, r_mkt)/var(r_mkt) over window | Rolling/exponential; PIT via trailing window |
| Drawdown | `P_t / cummax(P) - 1`, max drawdown | `cummax` is trailing → PIT-safe |
| Downside / semi-deviation | std of negative returns only | Feeds Sortino-style features |

| Library | What it does | Link |
|---|---|---|
| arch | GARCH/EGARCH/HAR volatility models, bootstrap, unit-root tests | https://github.com/bashtage/arch |
| volatility estimators | OHLC estimators (Parkinson/GK/RS/Yang-Zhang) tutorials | https://www.pyquantnews.com/the-pyquant-newsletter/how-to-compute-volatility-6-ways |
| Yang–Zhang (paper + code) | Automated YZ estimation from high-frequency data | https://www.sciencedirect.com/science/article/pii/S2665963824000010 |

---

## 4. Market-Microstructure Features (Microestrutura)

Require tick / order-book (LOB) data. Math is venue-agnostic; only the feed format differs. These dominate short-horizon (intraday/HFT) models.

| Feature | What it captures | Reference |
|---|---|---|
| Order-flow imbalance (OFI) | Net signed change in best bid/ask sizes between LOB updates | Cont, Kukanov, Stoikov (2014) — price-impact link |
| Order-book imbalance (OBI) | `(bidVol − askVol)/(bidVol + askVol)` at top-of-book / N levels | Short-horizon direction signal |
| Bid–ask spread | `ask − bid`; relative spread; effective/realized spread | Liquidity / cost proxy |
| Depth | Cumulative volume per side over N levels | Liquidity supply |
| Trade signs | Buyer/seller-initiated via tick rule or Lee–Ready | Input to OFI, VPIN, Kyle's λ |
| Kyle's λ | Price impact per unit signed order flow (Kyle 1985) | Adverse-selection / illiquidity |
| VPIN | Volume-synchronized prob. of informed trading (Easley et al. 2012) | Order-flow **toxicity**; volume-time buckets |
| Amihud illiquidity | `|return| / dollar volume` | Daily-data illiquidity proxy (no LOB needed) |
| Roll spread | From serial covariance of price changes | Spread estimate from trades only |

Microstructure feature recipes and López-de-Prado-style bars (tick/volume/dollar/imbalance bars) are in **mlfinlab** (https://github.com/hudson-and-thames/mlfinlab). VPIN and OFI walkthroughs: https://microalphas.com/vpin/ and the order-flow normalization study arXiv:2512.18648 (https://arxiv.org/abs/2512.18648). Foundational papers: Kyle (1985) *Continuous Auctions and Insider Trading*; Easley, López de Prado, O'Hara (2012) *Flow Toxicity and Liquidity in a High-Frequency World*.

---

## 5. Cross-Sectional / Factor Features (Fatores)

For multi-asset / panel models you score every name **per date** and compare across the universe. This is where most equity alpha lives, and it is intrinsically PIT-friendly because cross-sectional ranks use only same-date data.

| Factor (fator) | Typical proxy | Direction |
|---|---|---|
| Value (valor) | E/P, B/P, CF/P, dividend yield | Cheap > expensive |
| Quality (qualidade) | ROE, ROA, gross profitability, low accruals | High > low |
| Momentum | 12-1 return, residual momentum | Winners > losers |
| Size (tamanho) | market cap (often log) | Small > large (historically) |
| Low volatility | trailing vol / beta | Low > high |
| Investment / growth | asset growth, capex growth | Conservative > aggressive |

**Cross-sectional transforms (per date):** rank → uniform/percentile; winsorize tails (e.g. 1/99%); z-score; **neutralize** vs sector/industry/market-cap/beta (regress factor on dummies, keep residual). These remove unwanted exposures so the signal is pure.

| Resource | What it provides | Link |
|---|---|---|
| 101 Formulaic Alphas (paper) | 101 explicit, code-ready alpha formulas; avg holding 0.6–6.4 days | https://arxiv.org/abs/1601.00991 |
| 101 alphas (reference code) | Python implementation of the WorldQuant/Kakushadze alphas | https://github.com/yli188/WorldQuant_alpha101_code |
| Qlib Alpha158 | 158 human-engineered expression features (price/vol/rolling) | https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py |
| Qlib Alpha360 | 360 raw normalized price/volume lags (low feature-engineering) | https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py |
| Qlib expression engine | Operators like `Ref($close,60)/$close`, `Mean`, `Std`, `Corr` | https://qlib.readthedocs.io/en/latest/component/data.html |
| Alpha Factor Library (Jansen) | Catalogue of factors incl. 101 alphas + Kakushadze GTJA-191 | https://stefan-jansen.github.io/machine-learning-for-trading/24_alpha_factor_library/ |

> Qlib's Alpha158/Alpha360 are **expression-based** and run on any market you ingest into Qlib's data layer — they are not China-specific despite Qlib's origin. The 101 alphas were designed on US equities but the formulas are generic OHLCV/returns expressions and port to any cross-section.

---

## 6. Calendar / Seasonality Features (Calendário e Sazonalidade)

Cheap, often surprisingly useful, fully PIT-safe (the calendar is known in advance).

| Feature | Examples | Use |
|---|---|---|
| Day-of-week | Mon…Fri one-hot or cyclical sin/cos | Weekday effects |
| Month / quarter | month-of-year, turn-of-month, quarter-end | Window dressing, seasonality |
| Time-of-day (intraday) | open/close auctions, lunch lull, U-shaped volume | Intraday volatility/volume patterns |
| Expiry / roll | days-to-expiry, roll dates (futures/options) | Triple-witching, contango/roll yield |
| Holiday proximity | days before/after market holiday | Pre/post-holiday drift; needs **market-specific** calendar |

> The **only region-specific** part here is the holiday/trading calendar. Use `exchange_calendars` (https://github.com/gerrymanoim/exchange_calendars) or `pandas-market-calendars` (https://github.com/rsheftel/pandas_market_calendars) — both ship calendars for B3 (Brazil), NYSE, LSE, NSE, JPX, crypto-24/7, etc. Encode cyclical features as sin/cos to avoid false ordinality.

---

## 7. Fundamental Features (Fundamentos)

Balance-sheet / income-statement derived. The single biggest leakage trap in all of finance.

| Feature | Formula | Category |
|---|---|---|
| P/E, P/B, P/S, EV/EBITDA | price ÷ fundamental | Valuation |
| ROE, ROA, ROIC | earnings ÷ equity/assets/capital | Profitability |
| Gross / operating margin | profit ÷ revenue | Quality |
| Revenue / EPS growth | YoY or QoQ change | Growth |
| Accruals | (NI − CFO) / assets | Earnings quality (low = better) |
| Leverage | debt/equity, net debt/EBITDA | Risk |
| Piotroski F-score, Altman Z | composite quality/bankruptcy scores | Composite |

**Point-in-time discipline (critical).** Fundamentals are (1) reported with a lag, (2) frequently **restated**. Always lag by the actual filing/announcement date — not the fiscal period-end — and prefer a **point-in-time database** that preserves the originally-reported values. Using as-reported-today values for a historical date is the most common silent source of inflated backtests. Map each fundamental to the date it became publicly known, then forward-fill until the next release.

---

## 8. Alternative-Data / NLP Features (Dados Alternativos e NLP)

Text, search, satellite, and card data. NLP sentiment is the most accessible; models below are general-purpose English-finance (multilingual options noted).

| Resource | What it does | Link |
|---|---|---|
| FinBERT (ProsusAI) | BERT fine-tuned for financial sentiment → positive/negative/neutral | https://huggingface.co/ProsusAI/finbert |
| FinBERT-tone (yiyanghkust) | FinBERT trained on 4.9B-token financial corpus; analyst-report tone | https://huggingface.co/yiyanghkust/finbert-tone |
| FinBERT repo | Training/inference code (Prosus) | https://github.com/ProsusAI/finBERT |
| FinGPT | Open financial LLM suite incl. sentiment/forecasting adapters | https://github.com/AI4Finance-Foundation/FinGPT |
| sentence-transformers | General + financial embeddings for news/headlines | https://github.com/UKPLab/sentence-transformers |

**Feature types from text:** sentiment score/polarity (FinBERT softmax), earnings-call **tone** and uncertainty, document embeddings (mean-pooled or [CLS]), novelty (cosine distance vs prior news), event flags (guidance, M&A, downgrade). **Leakage warning:** timestamp news/filings by **publication time in market timezone**, never by the date you scraped them; align to the next tradable bar. For non-English markets (e.g. Portuguese news in Brazil), use multilingual encoders or translate, and validate that FinBERT's English-finance vocabulary transfers.

---

## 9. Automated Feature Engineering & Feature Stores

Generate, evaluate, and serve features at scale. All library-level and market-agnostic.

| Tool | What it does | Link |
|---|---|---|
| tsfresh | Auto-extracts 100s of time-series features + hypothesis-test relevance filtering | https://github.com/blue-yonder/tsfresh |
| Featuretools | Deep Feature Synthesis: stacks aggregation/transform primitives over relational data | https://github.com/alteryx/featuretools |
| TSFEL | Time Series Feature Extraction Library (temporal/statistical/spectral domains) | https://github.com/fraunhoferportugal/tsfel |
| Feature-engine | Sklearn-compatible transformers (encoding, discretization, outliers, creation) | https://github.com/feature-engine/feature_engine |
| alphalens | Performance/IC/turnover analysis of predictive (alpha) factors | https://github.com/quantopian/alphalens |
| alphalens (maintained) | Active fork with fixes/visualization | https://github.com/cloudQuant/alphalens |
| Feast | Open-source feature **store**: offline (training) + online (serving) consistency | https://github.com/feast-dev/feast |

> **alphalens** is the standard "is this factor any good?" tool — it computes the information coefficient (IC), quantile returns, and turnover for a factor against forward returns, on any universe. **tsfresh/Featuretools/TSFEL** can explode feature counts fast; pair them with the tsfresh relevance filter or model-based selection to avoid multiple-testing/over-fitting. **Feast** solves train/serve skew by guaranteeing the same feature logic offline and online.

---

## 10. Compact Starter Feature Set (Conjunto Inicial)

A minimal, robust, mostly-collinearity-aware set that works on any single instrument with daily OHLCV. Start here, then expand.

| # | Feature | Why it earns its place | PIT-safe? |
|---|---|---|---|
| 1 | Log return `ℓ_t` | Base signal, additive | Yes |
| 2 | Momentum 21d & 63d & 252d | Multi-horizon trend | Yes (trailing) |
| 3 | Rolling vol 21d (std of `ℓ`) | Risk regime | Yes |
| 4 | Yang–Zhang vol 21d | Gap-robust vol | Yes |
| 5 | RSI(14) | Mean-reversion / overbought | Yes |
| 6 | MACD line & signal | Trend/momentum | Yes |
| 7 | Bollinger %B (20,2) | Position in vol band | Yes |
| 8 | ATR(14) (relative) | Range/vol | Yes |
| 9 | Rolling z-score of price (20) | Normalized level | Yes |
| 10 | Volume z-score (20) / OBV slope | Participation | Yes |
| 11 | Max drawdown 63d | Tail-risk state | Yes |
| 12 | Day-of-week + month (cyclical) | Seasonality | Yes |
| 13 | Fractional-diff price (`d`≈0.4) | Stationary + memory | Yes (FFD window) |
| 14 | Cross-sectional rank of (2),(3) per date* | Factor exposure | Yes |
| 15 | FinBERT news sentiment (if text)* | Information edge | Yes if timestamped right |

\* Items 14–15 require a universe / external text feed. Everything else is single-series.

**Pipeline discipline checklist:** adjusted prices → trailing windows only (`min_periods`, no `center=True`) → split train/val/test by **time** → fit scalers/encoders on train only → lag fundamentals/news to known-at-`t` → prune collinear indicators → validate with `alphalens` IC / purged walk-forward CV (avoid k-fold on time series; use purged + embargo CV à la López de Prado).

---

## Sources

- mlfinlab (López de Prado fracdiff, microstructure, bars): https://github.com/hudson-and-thames/mlfinlab
- fracdiff: https://github.com/fracdiff/fracdiff · mlfinpy frac-diff docs: https://mlfinpy.readthedocs.io/en/latest/FractionalDifferentiated.html
- TA-Lib: https://github.com/TA-Lib/ta-lib-python · pandas-ta: https://github.com/twopirllc/pandas-ta · pandas-ta-classic: https://github.com/xgboosted/pandas-ta-classic
- ta (bukosabino): https://github.com/bukosabino/ta · tulipy: https://github.com/cirla/tulipy · FinTA: https://github.com/peerchemist/finta
- arch (GARCH): https://github.com/bashtage/arch · Yang–Zhang paper: https://www.sciencedirect.com/science/article/pii/S2665963824000010 · vol estimators: https://www.pyquantnews.com/the-pyquant-newsletter/how-to-compute-volatility-6-ways
- VPIN explainer: https://microalphas.com/vpin/ · order-flow normalization: https://arxiv.org/abs/2512.18648
- 101 Formulaic Alphas (Kakushadze): https://arxiv.org/abs/1601.00991 · code: https://github.com/yli188/WorldQuant_alpha101_code
- Qlib: https://github.com/microsoft/qlib · handler (Alpha158/360): https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py · docs: https://qlib.readthedocs.io/en/latest/component/data.html
- Alpha Factor Library (Jansen): https://stefan-jansen.github.io/machine-learning-for-trading/24_alpha_factor_library/
- exchange_calendars: https://github.com/gerrymanoim/exchange_calendars · pandas-market-calendars: https://github.com/rsheftel/pandas_market_calendars
- FinBERT: https://huggingface.co/ProsusAI/finbert · FinBERT-tone: https://huggingface.co/yiyanghkust/finbert-tone · FinBERT repo: https://github.com/ProsusAI/finBERT · FinGPT: https://github.com/AI4Finance-Foundation/FinGPT
- tsfresh: https://github.com/blue-yonder/tsfresh · Featuretools: https://github.com/alteryx/featuretools · TSFEL: https://github.com/fraunhoferportugal/tsfel · Feature-engine: https://github.com/feature-engine/feature_engine
- alphalens: https://github.com/quantopian/alphalens · maintained fork: https://github.com/cloudQuant/alphalens · Feast: https://github.com/feast-dev/feast

**Keywords:** feature engineering, engenharia de atributos, technical indicators, indicadores técnicos, fractional differentiation, diferenciação fracionária, realized volatility, volatilidade realizada, Yang-Zhang, Garman-Klass, Parkinson, order flow imbalance, fluxo de ordens, VPIN, Kyle's lambda, microestrutura de mercado, factor models, modelos de fatores, 101 alphas, WorldQuant, Qlib, Alpha158, Alpha360, cross-sectional ranking, winsorize, neutralization, neutralização, point-in-time, look-ahead bias, viés de antecipação, data leakage, vazamento de dados, FinBERT, sentiment analysis, análise de sentimento, tsfresh, Featuretools, TSFEL, Feature-engine, alphalens, Feast, feature store, TA-Lib, pandas-ta, seasonality, sazonalidade, fundamentals, fundamentos, mercados financeiros, financial markets, B3, ações, FX, câmbio, crypto, futuros, opções
