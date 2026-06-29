# Open-Source Quant Ecosystem & Awesome Lists

> The meta-layer of quantitative finance: curated "awesome" lists, OSS backtesting engines, portfolio/risk libraries, and alt-data wrappers that each unlock hundreds-to-thousands of further resources. Star counts verified via the GitHub REST API on 2026-06-29 and rounded; treat them as relative popularity signals, not precise live numbers.

This page is deliberately **meta**: instead of re-indexing exchanges already covered elsewhere in this repo (US NYSE/Nasdaq, Brazil B3, India NSE/BSE, China SSE/SZSE, Japan/HK/Korea/Taiwan, ASEAN, generic data APIs, Kaggle/HuggingFace pulls, arXiv q-fin), it points at the **discovery layer** — the curated lists and OSS toolchains that practitioners actually use to find everything else. A single awesome-list (e.g. `awesome-quant`, ~27k stars) catalogs hundreds of libraries across a dozen languages; the libraries below are the high-signal core that recurs across all of them.

**Brazil note (acesso no Brasil):** Almost everything here is pip/cargo/NuGet-installable worldwide and free; no broker or jurisdiction restriction applies to the *code*. Where a tool needs market data, Brazilian users can wire in B3 data via `python-bcb`, `brapi`, or MetaTrader 5 (MT5) wrappers (see the Brazil table). For foreign-market exposure referenced by these tools, Brazil access is via BDRs and ETFs (e.g. IVVB11 for S&P 500) on B3 — but the libraries themselves are unrestricted.

---

## 1. Meta "Awesome" Lists — the discovery layer

These are curated indexes. Each one is a multiplier: follow it and you reach hundreds of additional repos, papers, blogs, and books. Prefer the actively-pushed ones (right-hand "updated" column matters as much as stars for lists).

| List / Repo | What it catalogs | Stars (~) | Last push | Link |
|---|---|---|---|---|
| `wilsonfreitas/awesome-quant` | The canonical index: libraries, packages, data, books across Python/R/C++/Julia/Rust/Matlab. Brazilian-maintained (Wilson Freitas). | 27.2k | 2026-06 (active) | https://github.com/wilsonfreitas/awesome-quant |
| `georgezouq/awesome-ai-in-finance` | LLMs, deep-learning strategies and tools specific to financial markets (RL bots, sentiment, alpha research). | 6.2k | 2026-06 (active) | https://github.com/georgezouq/awesome-ai-in-finance |
| `firmai/financial-machine-learning` | Practical financial ML tools/applications, organized by task (forecasting, portfolio, NLP). By Derek Snow. | 8.7k | 2025-01 | https://github.com/firmai/financial-machine-learning |
| `paperswithbacktest/awesome-systematic-trading` | ~97 libraries + 40+ documented strategies + 55 books for systematic trading; pairs with paperswithbacktest.com. | 8.5k | 2025-01 | https://github.com/paperswithbacktest/awesome-systematic-trading |
| `wangzhe3224/awesome-systematic-trading` | Independent systematic-trading index (crypto, stocks, futures, options, FX); bilingual EN/中文; live site. | 4.4k | 2026-06 (active) | https://github.com/wangzhe3224/awesome-systematic-trading |
| `grananqvist/Awesome-Quant-Machine-Learning-Trading` | Quant/algo trading resources emphasizing ML (courses, papers, code). | 3.8k | 2025-05 | https://github.com/grananqvist/Awesome-Quant-Machine-Learning-Trading |
| `cbailes/awesome-deep-trading` | ML-based algorithmic-trading resources (deep learning, RL papers + code). Older but well-organized. | 2.0k | 2023-08 (stale) | https://github.com/cbailes/awesome-deep-trading |
| `leoncuhk/awesome-quant-ai` | Newer (2025) curated list focused on AI/ML for quant investment, incl. LLM agents & FinRL agents. | 0.4k | 2026-04 (active) | https://github.com/leoncuhk/awesome-quant-ai |

> **How to use:** start from `awesome-quant` for breadth, `wangzhe3224/awesome-systematic-trading` for actively-maintained strategy/library breadth, `georgezouq/awesome-ai-in-finance` + `leoncuhk/awesome-quant-ai` for the LLM/RL frontier, and `firmai/financial-machine-learning` for ML-task organization. Together they index well over a thousand downstream resources.

---

## 2. Backtesting & Live-Trading Engines (OSS)

The execution/backtesting core. Note the 2024-2026 shift toward Rust-native and event-driven designs.

| Library | What | Stars (~) | Lang | License | Link |
|---|---|---|---|---|---|
| `nautechsystems/nautilus_trader` | Production-grade, **Rust-native**, event-driven engine; one codebase for backtest + live. Fast-rising. | 24.3k | Rust/Python | LGPL-3.0 | https://github.com/nautechsystems/nautilus_trader |
| `mementum/backtrader` | Classic Python backtesting/live framework; huge install base though upstream is quiet since 2024. | 22.2k | Python | GPL-3.0 | https://github.com/mementum/backtrader |
| `QuantConnect/Lean` | LEAN engine behind QuantConnect; multi-asset backtest + live, C# core, Python algos. Cloud + local CLI. | 20.3k | C# | Apache-2.0 | https://github.com/QuantConnect/Lean |
| `je-suis-tm/quant-trading` | Strategy/indicator code collection (mean-reversion, momentum, options) — learning-oriented. | 10.2k | Python | Apache-2.0 | https://github.com/je-suis-tm/quant-trading |
| `kernc/backtesting.py` | Minimal, fast single-asset backtester; popular for teaching/prototyping. | 8.6k | Python | AGPL-3.0 | https://github.com/kernc/backtesting.py |
| `polakowo/vectorbt` | Vectorized (Numba) backtesting; test thousands of param combos in seconds. Pro tier = `vectorbt.pro`. | 8.1k | Python | Apache-2.0* | https://github.com/polakowo/vectorbt |
| `tensortrade-org/tensortrade` | RL-first trading framework (Gym-style environments) for training agents. | 6.4k | Python | Apache-2.0 | https://github.com/tensortrade-org/tensortrade |
| `ricequant/rqalpha` | Extensible Python backtest + trading framework, strong China-market support. | 6.5k | Python | Apache-2.0* | https://github.com/ricequant/rqalpha |
| `edtechre/pybroker` | ML-focused backtesting with walk-forward analysis and bootstrap metrics. | 3.4k | Python | (custom) | https://github.com/edtechre/pybroker |
| `mhallsmoore/qstrader` | Event-driven backtesting library (the QuantStart engine). Educational, stable. | 3.4k | Python | MIT | https://github.com/mhallsmoore/qstrader |
| `stefan-jansen/zipline-reloaded` | Maintained fork of Quantopian's `zipline` event-driven backtester (the "reloaded" line). | 1.8k | Python | Apache-2.0 | https://github.com/stefan-jansen/zipline-reloaded |
| `Blankly-Finance/Blankly` | Unified API for backtest→paper→live across stocks/crypto; quiet since 2024. | 2.5k | Python | LGPL-3.0 | https://github.com/Blankly-Finance/Blankly |

\* `vectorbt` and `rqalpha` carry non-standard / source-available license terms — check before commercial use.

---

## 3. Crypto-Native Bots & HFT (under-indexed vs equities)

These are heavily used but rarely appear in equity-centric indexes.

| Library | What | Stars (~) | Lang | License | Link |
|---|---|---|---|---|---|
| `freqtrade/freqtrade` | The dominant free crypto trading bot: backtest, hyperopt, ML (FreqAI), Telegram control. | 51.9k | Python | GPL-3.0 | https://github.com/freqtrade/freqtrade |
| `hummingbot/hummingbot` | Market-making / arbitrage bot framework across 140+ CEX/DEX connectors. | 19.0k | Python | Apache-2.0 | https://github.com/hummingbot/hummingbot |
| `jesse-ai/jesse` | Research-friendly crypto framework; clean strategy DSL + backtest/optimize. | 8.1k | Python/JS | MIT | https://github.com/jesse-ai/jesse |
| `nkaz001/hftbacktest` | **High-frequency** backtester with full order-book / latency / queue modeling; Rust core. | 4.2k | Rust/Python | MIT | https://github.com/nkaz001/hftbacktest |
| `ccxt/ccxt` | Unified API to 100+ crypto exchanges (REST + WS); the de-facto crypto data/exec layer. | 43.1k | Multi | MIT | https://github.com/ccxt/ccxt |

---

## 4. Portfolio Optimization & Risk Analytics

| Library | What | Stars (~) | License | Link |
|---|---|---|---|---|
| `ranaroussi/quantstats` | Tear-sheets + risk/perf metrics (Sharpe, drawdown, etc.) from a returns series. Successor: `quantstats-lumi`. | 7.4k | Apache-2.0 | https://github.com/ranaroussi/quantstats |
| `PyPortfolio/PyPortfolioOpt` | Mean-variance, Black-Litterman, HRP, shrinkage. The teaching standard. (Repo moved to `PyPortfolio` org.) | 5.8k | MIT | https://github.com/PyPortfolio/PyPortfolioOpt |
| `dcajasn/Riskfolio-Lib` | 20+ risk measures, risk-parity, hierarchical/NCO, factor & CVaR models. Built on CVXPY. | 4.3k | BSD-3 | https://github.com/dcajasn/Riskfolio-Lib |
| `pmorissette/bt` | Flexible algo-tree backtester for portfolio strategies (rebalancing logic). | 2.9k | MIT | https://github.com/pmorissette/bt |
| `pmorissette/ffn` | Financial functions: performance stats, drawdowns, monthly tables (`bt`'s companion). | 2.6k | MIT | https://github.com/pmorissette/ffn |
| `skfolio/skfolio` | **scikit-learn-native** portfolio optimization: CV, hyperparam tuning, stress tests. Modern (2024+). | 2.0k | BSD-3 | https://github.com/skfolio/skfolio |
| `quantopian/empyrical` | Common risk/return metrics (used by pyfolio); maintenance-mode but ubiquitous dependency. | 1.5k | Apache-2.0 | https://github.com/quantopian/empyrical |
| `cvxgrp/cvxportfolio` | Stanford CVX group; convex multi-period portfolio optimization + backtesting (paper-backed). | 1.2k | GPL-3.0 | https://github.com/cvxgrp/cvxportfolio |
| `stefan-jansen/pyfolio-reloaded` | Maintained fork of Quantopian `pyfolio` tear-sheets. | 0.6k | Apache-2.0 | https://github.com/stefan-jansen/pyfolio-reloaded |
| `stefan-jansen/alphalens-reloaded` | Maintained fork of `alphalens` for alpha-factor performance analysis (IC, quantile returns). | 0.6k | Apache-2.0 | https://github.com/stefan-jansen/alphalens-reloaded |

---

## 5. Quant Pricing, ML-Finance & Numerical Libraries

| Library | What | Stars (~) | License | Link |
|---|---|---|---|---|
| `goldmansachs/gs-quant` | Goldman Sachs' Python toolkit: derivatives pricing, risk, backtesting (some features need GS Marquee). | 11.0k | Apache-2.0 | https://github.com/goldmansachs/gs-quant |
| `lballabio/QuantLib` | The reference C++ quant library (pricing, curves, vol, fixed income); Python via `QuantLib-Python`. | 7.3k | BSD-style | https://github.com/lballabio/QuantLib |
| `cantaro86/Financial-Models-Numerical-Methods` | Notebook course: option pricing, SDEs, Kalman, jumps — high-signal learning resource. | 6.9k | AGPL-3.0 | https://github.com/cantaro86/Financial-Models-Numerical-Methods |
| `google/tf-quant-finance` | Google's TensorFlow library for high-performance derivatives pricing (GPU/TPU). | 5.4k | Apache-2.0 | https://github.com/google/tf-quant-finance |
| `domokane/FinancePy` | Pure-Python pricing/risk of fixed income, derivatives, bonds; teaching-friendly. | 3.0k | GPL-3.0 | https://github.com/domokane/FinancePy |
| `hudson-and-thames/mlfinlab` | Implements López de Prado's "Advances in Financial ML" (fractional diff, triple-barrier, meta-labeling). Note: **moved to a paid/closed model** — repo is archived/old; community forks exist. | 4.8k | (restricted) | https://github.com/hudson-and-thames/mlfinlab |

> Caveat: `mlfinlab` and Hudson & Thames' `arbitragelab` (~0.7k, https://github.com/hudson-and-thames/arbitragelab) transitioned to a commercial/closed model; the public repos are frozen. Cite them for the algorithms, not as live OSS dependencies.

---

## 6. AI Agents & LLMs for Markets (2024-2026 frontier — heavily under-indexed)

This is the fastest-moving and least-indexed cluster. All verified live.

| Repo | What | Stars (~) | Last push | Link |
|---|---|---|---|---|
| `TauricResearch/TradingAgents` | Multi-agent LLM trading framework (analyst/researcher/trader/risk roles debate). Viral in 2025-2026. | 89.7k | 2026-06 | https://github.com/TauricResearch/TradingAgents |
| `virattt/ai-hedge-fund` | LLM "hedge fund" of agents emulating famous investors (Buffett, Munger, etc.); educational. | 60.7k | 2026-06 | https://github.com/virattt/ai-hedge-fund |
| `AI4Finance-Foundation/FinGPT` | Open financial LLMs + datasets (sentiment, forecasting, RAG) from the AI4Finance group. | 20.8k | 2026-06 | https://github.com/AI4Finance-Foundation/FinGPT |
| `AI4Finance-Foundation/FinRL` | The canonical deep-RL-for-trading framework (A2C/PPO/SAC/TD3 over market envs). | 15.6k | 2026-05 | https://github.com/AI4Finance-Foundation/FinRL |
| `microsoft/qlib` | Microsoft's AI quant platform (supervised + RL + market dynamics); now drives `RD-Agent` auto-R&D. Surged in 2025. | 45.4k | 2026-04 | https://github.com/microsoft/qlib |
| `microsoft/RD-Agent` | Automates the research→development loop (factor/model mining); integrates with Qlib. | 13.7k | 2026-06 | https://github.com/microsoft/RD-Agent |
| `AI4Finance-Foundation/FinRobot` | LLM agent platform for financial analysis (CoT, forecasting, document analysis). | 7.4k | 2026-05 | https://github.com/AI4Finance-Foundation/FinRobot |
| `AI4Finance-Foundation/FinRL-Meta` | Market environments + datasets layer feeding FinRL (the "gym" of finance). | 1.9k | 2026-05 | https://github.com/AI4Finance-Foundation/FinRL-Meta |
| `The-FinAI/PIXIU` | Financial LLM benchmark + instruction data + FinMA models (eval-focused). | 0.9k | 2025-03 | https://github.com/The-FinAI/PIXIU |
| `benstaf/FinRL_DeepSeek` | 2025 experiment combining FinRL with DeepSeek-style LLM signals; niche but current. | 0.3k | 2025-04 | https://github.com/benstaf/FinRL_DeepSeek |

> **OpenBB as the AI-data backbone:** `OpenBB-finance/OpenBB` (69.8k stars, AGPL/custom, https://github.com/OpenBB-finance/OpenBB) is now positioned as a "financial data platform for analysts, quants and **AI agents**," exposing data to Python, Excel, REST, and **MCP servers** for LLM tool-use. It is the most-starred finance-data project on GitHub and a primary connective layer for agentic workflows.

---

## 7. Technical Analysis & Charting

| Library | What | Stars (~) | License | Link / Note |
|---|---|---|---|---|
| `TA-Lib/ta-lib-python` | Python wrapper for the C TA-Lib (150+ indicators); industry standard. | 12.1k | BSD-2 | https://github.com/TA-Lib/ta-lib-python |
| `bukosabino/ta` | Pure-Python pandas TA library (no C build); easy install. | 5.1k | MIT | https://github.com/bukosabino/ta |
| `matplotlib/mplfinance` | Financial candlestick/OHLC plotting on matplotlib. | 4.4k | BSD-style | https://github.com/matplotlib/mplfinance |
| `pandas-ta` (orig. `twopirllc`) | 130+ indicators as a pandas extension. **The original repo was removed (2024) and moved to a paid model.** Use the community fork below. | n/a | — | (original gone; see fork) |
| `xgboosted/pandas-ta-classic` | Actively-maintained community fork preserving the last open `pandas-ta`. | 0.4k | MIT | https://github.com/xgboosted/pandas-ta-classic |

---

## 8. Data & Alt-Data Wrappers (OSS)

Lightweight ingestion layers — the practical on-ramp before any of the above.

| Library | What | Stars (~) | Status / Note | Link |
|---|---|---|---|---|
| `ranaroussi/yfinance` | Yahoo! Finance downloader; the most-used free OHLCV source. Best-effort, ToS-gray. | 24.5k | Active | https://github.com/ranaroussi/yfinance |
| `JerBouma/FinanceDatabase` | 300k+ symbols (equities, ETFs, funds, crypto, indices) as searchable metadata. | 8.0k | Active | https://github.com/JerBouma/FinanceDatabase |
| `JerBouma/FinanceToolkit` | 150+ financial ratios/metrics from raw fundamentals; transparent formulas. | 5.0k | Active | https://github.com/JerBouma/FinanceToolkit |
| `cuemacro/finmarketpy` | Market analysis + backtesting (Cuemacro); pairs with `findatapy`. | 3.8k | Active | https://github.com/cuemacro/finmarketpy |
| `cuemacro/findatapy` | Unified API over Bloomberg/Quandl/Dukascopy/ALFRED and CSVs. | 2.1k | Active | https://github.com/cuemacro/findatapy |
| `alvarobartt/investpy` | Investing.com scraper. **Deprecated/blocked** since 2022 (Cloudflare); kept for reference only. | 1.8k | Broken | https://github.com/alvarobartt/investpy |
| `dpguthrie/yahooquery` | Alternative Yahoo Finance API (more endpoints: fundamentals, options). | 0.9k | Active | https://github.com/dpguthrie/yahooquery |
| `pandas-datareader` | Legacy pandas remote-data access (FRED, Stooq, etc.); maintenance-mode but still useful for FRED. | — | Low activity | https://github.com/pydata/pandas-datareader |

> **investpy status (important):** `investpy` no longer works — Investing.com added Cloudflare protection in 2022 and the maintainer marked it deprecated. Do not recommend it for new pipelines; use `yfinance`/`yahooquery`/OpenBB or official exchange feeds instead.

---

## 9. Brazil-Specific OSS (foco Brasil)

Under-indexed globally but essential for B3/Bacen data. English description, Portuguese terms in parentheses.

| Tool | What | Stars (~) | Access / Note | Link |
|---|---|---|---|---|
| `wilsonfreitas/python-bcb` | Python interface to **Banco Central do Brasil** web services: SGS time series (séries temporais), PTAX FX (câmbio), market expectations (Expectativas/Focus), via OData/SDMX. Returns pandas DataFrames. | 0.1k | `pip install python-bcb`; free public Bacen APIs. | https://github.com/wilsonfreitas/python-bcb |
| `brapi` (brapi.dev) | Free REST API for B3 equities (ações), FIIs, ETFs, dividends and crypto; ~15-min delayed quotes. Multiple community clients. | — | Free tier; key for higher limits. Org: https://github.com/brapi-dev | https://brapi.dev |
| `geovannyAvelar/brapigo` | Go client for brapi.dev (Brazilian stock data). | 0 | Reference client. | https://github.com/geovannyAvelar/brapigo |
| MetaTrader 5 (MT5) Python | Official `MetaTrader5` PyPI package bridges Python to an MT5 terminal — the standard route to **live B3/forex data and order execution** for Brazilian retail brokers. | — | `pip install MetaTrader5`; needs an MT5 terminal + broker account. | https://pypi.org/project/MetaTrader5/ |
| `sgs` (PyPI) | Lightweight pure-Bacen SGS time-series fetcher (alternative to `python-bcb`). | — | `pip install sgs`. | https://pypi.org/project/sgs/ |

> Brazilian quant context: Wilson Freitas (maintainer of `awesome-quant` *and* `python-bcb`) is a hub of the Brazilian OSS-quant scene. Community courses/notebooks like `codigoquant/python_para_investimentos` show end-to-end Bacen+B3 pipelines in Portuguese.

---

## 10. Foreign-broker / data SDKs worth knowing (US-centric, but free)

| Library | What | Stars (~) | License | Link |
|---|---|---|---|---|
| `alpacahq/alpaca-py` | Official SDK for Alpaca commission-free US stocks/crypto trading + market data. | 1.4k | Apache-2.0 | https://github.com/alpacahq/alpaca-py |
| `polygon-io/client-python` | Official Polygon.io client (US equities/options/FX/crypto market data). | — | MIT | https://github.com/polygon-io/client-python |
| `QuantConnect/lean-cli` | Local CLI for the LEAN engine — backtest/live without the cloud UI. | 0.3k | Apache-2.0 | https://github.com/QuantConnect/lean-cli |

---

## Why this "unlocks thousands"

- **One list → hundreds of repos.** `awesome-quant` alone catalogs hundreds of libraries across Python, R, C++, Julia, Rust, and Matlab; `awesome-systematic-trading` adds ~97 libraries + 40+ documented strategies + 55 books.
- **One engine → an ecosystem.** LEAN, NautilusTrader, Qlib, and FinRL each ship dozens of example strategies, data adapters, and community plugins.
- **One agent framework → the LLM frontier.** `TradingAgents`, `ai-hedge-fund`, `FinGPT`, and `RD-Agent` are the current (2025-2026) leading edge and are themselves rapidly spawning forks.
- **Cross-reference, don't trust blindly.** Star counts and "last push" dates both matter — several once-dominant repos (`backtrader`, `mlfinlab`, original `pandas-ta`, `investpy`) are now stale, closed, or broken. Verify activity before depending on any of them.

---

## Sources

- GitHub REST API (`api.github.com/repos/...`), queried 2026-06-29, for all star counts / push dates / licenses.
- https://github.com/wilsonfreitas/awesome-quant • https://github.com/georgezouq/awesome-ai-in-finance • https://github.com/firmai/financial-machine-learning • https://github.com/paperswithbacktest/awesome-systematic-trading • https://github.com/wangzhe3224/awesome-systematic-trading • https://github.com/leoncuhk/awesome-quant-ai
- https://github.com/nautechsystems/nautilus_trader • https://github.com/QuantConnect/Lean • https://github.com/polakowo/vectorbt • https://github.com/kernc/backtesting.py • https://github.com/freqtrade/freqtrade • https://github.com/hummingbot/hummingbot • https://github.com/jesse-ai/jesse • https://github.com/nkaz001/hftbacktest • https://github.com/ccxt/ccxt
- https://github.com/ranaroussi/quantstats • https://github.com/PyPortfolio/PyPortfolioOpt • https://github.com/dcajasn/Riskfolio-Lib • https://github.com/skfolio/skfolio • https://github.com/pmorissette/bt • https://github.com/pmorissette/ffn • https://github.com/stefan-jansen/pyfolio-reloaded • https://github.com/stefan-jansen/alphalens-reloaded • https://github.com/stefan-jansen/zipline-reloaded
- https://github.com/goldmansachs/gs-quant • https://github.com/lballabio/QuantLib • https://github.com/google/tf-quant-finance • https://github.com/hudson-and-thames/mlfinlab
- https://github.com/TauricResearch/TradingAgents • https://github.com/virattt/ai-hedge-fund • https://github.com/AI4Finance-Foundation/FinGPT • https://github.com/AI4Finance-Foundation/FinRL • https://github.com/microsoft/qlib • https://github.com/microsoft/RD-Agent • https://github.com/OpenBB-finance/OpenBB
- https://github.com/TA-Lib/ta-lib-python • https://github.com/xgboosted/pandas-ta-classic • https://github.com/ranaroussi/yfinance • https://github.com/JerBouma/FinanceDatabase • https://github.com/cuemacro/findatapy • https://github.com/alvarobartt/investpy
- https://github.com/wilsonfreitas/python-bcb • https://brapi.dev • https://github.com/brapi-dev • https://pypi.org/project/MetaTrader5/ • https://pypi.org/project/sgs/

**Keywords:** open-source quant, awesome lists, awesome-quant, systematic trading, backtesting engine, portfolio optimization, risk analytics, financial machine learning, RL trading, LLM trading agents, FinGPT, FinRL, Qlib, OpenBB, NautilusTrader, alt-data, B3, Bacen, MetaTrader 5; código aberto quant, listas awesome, negociação sistemática, backtesting, otimização de carteira, análise de risco, aprendizado de máquina em finanças, aprendizado por reforço, agentes LLM de trading, dados alternativos, ações, câmbio, séries temporais
