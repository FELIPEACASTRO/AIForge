# Open-Source Libraries for Options & Derivatives

> Curated, honestly-maintained map of OSS for pricing, Greeks, implied vol, vol surfaces, options data/brokerage, and derivatives backtesting. Every repo owner/name was verified to exist via the GitHub REST API on **2026-06-30**; stars are rounded and last-push dates are live — treat stars as relative popularity, not precise counts. Maintenance is reported honestly via the legend below.

Most of these are pip-installable worldwide and free; no jurisdiction restriction applies to the *code* itself. Where a tool needs live market data, that data (not the library) is the gated part.

## Maintenance legend

| Tag | Rule of thumb |
|---|---|
| 🟢 active | pushed within ~6 months, issues moving |
| 🟡 slow | pushed within ~12–24 months, usable but quiet |
| 🔴 stale | no push in 2+ years; works but unmaintained |
| 🗄️ archived | read-only on GitHub; do not depend on it |

---

## 1. Pricing, Greeks & implied volatility

| Library | Repo | What | Stars (~) | Last push | Status |
|---|---|---|---|---|---|
| QuantLib | `lballabio/QuantLib` | The reference C++ derivatives/quant library; engines for vanilla→exotic, term structures, vol models, day-count/calendars. BSD-3. | 7.3k | 2026-06 | 🟢 |
| QuantLib-SWIG | `lballabio/QuantLib-SWIG` | Official SWIG bindings exposing QuantLib to Python/Java/C#/R etc. (`pip install QuantLib`). | 0.4k | 2026-06 | 🟢 |
| PyQL | `enthought/pyql` | Cython bindings to QuantLib with a more Pythonic API surface. | 1.3k | 2025-08 | 🟡 |
| FinancePy | `domokane/FinancePy` | Pure-Python derivatives library by Dominic O'Kane (EDHEC); broad coverage, still beta-quality API. | 3.0k | 2026-06 | 🟢 |
| tf-quant-finance | `google/tf-quant-finance` | Google's TensorFlow-based high-perf quant finance (pricing, Greeks via autodiff). "Not an officially supported Google product." | 5.4k | 2026-02 | 🟡 |
| py_vollib | `vollib/py_vollib` | De-facto Python Black/Black-Scholes/Black-76 pricing + IV + Greeks. | 0.4k | 2026-05 | 🟢 |
| py_lets_be_rational | `vollib/py_lets_be_rational` | Pure-Python port of Jäckel's "Let's Be Rational" IV solver; dependency of py_vollib. | 0.1k | 2026-05 | 🟢 |
| vollib | `vollib/vollib` | Older SWIG-backed sibling of py_vollib (C-extension). | 1.0k | 2023-06 | 🔴 |
| py_vollib_vectorized | `marcdemers/py_vollib_vectorized` | NumPy/pandas-vectorized drop-in patch for py_vollib (batch IV/Greeks). | 0.2k | 2024-12 | 🟡 |
| fast-vollib | `raeidsaqur/fast-vollib` | 2026 high-perf IV/Greeks lib with PyTorch/JAX/CUDA fused-kernel backends; py_vollib-compatible API. Paper: arXiv:2604.27210. New/small. | <0.1k | 2026-06 | 🟢 |
| blackscholes | `CarloLepelaars/blackscholes` | Small, well-tested pure-Python BS/Black-76 pricing + full Greeks; good for teaching. | 0.1k | 2025-11 | 🟡 |
| mibian | `yassinemaaroufi/MibianLib` | Classic minimal BS/Greeks/IV lib; widely cited but **effectively abandoned** (last push 2021). | 0.3k | 2021-06 | 🔴 |
| optlib | `dbrojas/optlib` | GBS pricing/Greeks + an option-chain fetcher built on the **now-defunct TD Ameritrade API**; pricing core works, the data layer is dead. | 1.6k | 2022-11 | 🔴 |

---

## 2. Volatility surfaces (SVI / SABR / SSVI)

Honest reality: there is **no single dominant, maintained, standalone vol-surface library**. The robust path is the in-engine machinery of QuantLib / FinancePy / tf-quant-finance (SVI, SABR, local vol, Heston). Standalone repos are mostly small personal/research projects — useful as references, not production dependencies.

| Library | Repo | What | Stars (~) | Last push | Status |
|---|---|---|---|---|---|
| pysabr | `ynouri/pysabr` | Clean Hagan-2002 SABR implementation (lognormal & normal); MIT. | 0.6k | 2022-04 | 🔴 |
| ssvi | `arkonique/ssvi` | Arbitrage-free SSVI surface calibration from option chains (notebook-style). Small/personal. | <0.1k | 2025-10 | 🟡 |
| Arbitrage-Free-Volatility-Surface | `XanderRobbins/Arbitrage-Free-Volatility-Surface` | SVI + Heston calibration with static no-arbitrage checks; MIT. Small/personal. | <0.1k | 2026-05 | 🟢 |
| (in-engine) | QuantLib / FinancePy / tf-quant-finance | Production-grade SVI/SABR/local-vol/Heston live inside the general libraries in §1 — prefer these. | — | — | 🟢 |

---

## 3. Options data & brokerage connectivity

| Library | Repo | What | Stars (~) | Last push | Status |
|---|---|---|---|---|---|
| ib_async | `ib-api-reloaded/ib_async` | The **live** Interactive Brokers async client; community continuation after the original author's death. Use this. | 1.7k | 2026-05 | 🟢 |
| ib_insync | `erdewit/ib_insync` | The original IB async lib — **archived** (author Ewald de Wit passed away); migrate to ib_async. | 3.3k | 2024-03 | 🗄️ |
| alpaca-py | `alpacahq/alpaca-py` | Official Alpaca SDK; supports single- and multi-leg options (`OptionLegRequest`) + options market data, with example notebooks. | 1.4k | 2026-06 | 🟢 |
| tastytrade (unofficial) | `tastyware/tastytrade` | Unofficial, typed, async Tastytrade SDK by tastyware; **more popular and more current** than the official one. | 0.2k | 2026-06 | 🟢 |
| tastytrade (official) | `tastytrade/tastytrade-sdk-python` | Official Tastytrade SDK — **archived** (read-only); prefer the tastyware lib above. | 0.1k | 2026-03 | 🗄️ |
| wallstreet | `mcdallas/wallstreet` | Real-time stock/option quotes + Greeks via Yahoo/Google scraping; brittle and only lightly maintained. | 1.7k | 2024-07 | 🟡 |
| ccxt | `ccxt/ccxt` | Unified crypto-exchange API (140+ venues), incl. derivatives/options where exchanges expose them. | 43k | 2026-06 | 🟢 |
| OpenBB Platform | `OpenBB-finance/OpenBB` | Broad open-source investment-research platform; options chains, Greeks, and many data providers. | 70k | 2026-06 | 🟢 |
| Deribit API | — | No first-party Python SDK; integrate via the official **JSON-RPC v2 docs** (docs.deribit.com). Community Python wrappers exist but are tiny/unmaintained — avoid as a dependency. | — | — | 🔴 |

---

## 4. Backtesting with derivatives

| Library | Repo | What | Stars (~) | Lang | Status |
|---|---|---|---|---|---|
| optopsy | `goldspanlabs/optopsy` | The one genuinely **options-native** backtester (DTE/delta-targeted spreads, condors, etc.); now under the Goldspan Labs org (redirects from `michaelchu/optopsy`). | 1.4k | Python | 🟢 |
| NautilusTrader | `nautechsystems/nautilus_trader` | Production-grade, Rust-native, event-driven engine; one codebase for backtest + live, with derivatives support. | 24k | Rust/Py | 🟢 |
| lumibot | `Lumiwealth/lumibot` | Strategy framework with options support and multiple broker/data integrations. | 1.7k | Python | 🟢 |
| vectorbt | `polakowo/vectorbt` | Fast vectorized backtesting; OSS edition is **maintenance-mostly** with active dev in the closed-source **VectorBT PRO** ($-paid). | 8.1k | Python | 🟡 |
| backtrader | `mementum/backtrader` | Classic Python framework, huge install base, but **quiet upstream** and weak native options support. | 22k | Python | 🔴 |

---

## 5. Risk & portfolio (Greeks / sensitivities)

For derivatives risk, reuse the pricing engines: **QuantLib** (analytic + bump-and-revalue Greeks, scenario/curve risk), **FinancePy** (per-instrument Greeks), and **tf-quant-finance** (autodiff Greeks, fast for large books). There is **no single canonical "pyfin" derivatives-risk library** — if a source points you to "pyfin," treat it as unverified; build on the engines above instead.

---

## 6. Brazil (🇧🇷) coverage

- **Day-count reality:** none of the pricing libs above ship correct B3/ANBIMA **dias-úteis-252** conventions out of the box. The robust combo is QuantLib's `Brazil()` calendar + `Business252` day-counter, paired with **python-bizdays** (`wilsonfreitas/python-bizdays`, ~0.1k, active 2026-04) for ANBIMA business-day calendars.
- **Data:** B3 offers **no free retail options-data SDK**, and `yfinance` does **not** return reliable B3 option chains. Source B3 chains via paid/registered providers (e.g. OpLab) or scrape opcoes.net.br at your own risk.

---

## 7. Quick-pick cheatsheet

- **Just price an option + Greeks (Python):** `blackscholes` (clarity) or `py_vollib` (+ `py_vollib_vectorized` for batches).
- **Industrial pricing / curves / exotics:** QuantLib (via `pip install QuantLib`), or FinancePy for pure-Python.
- **GPU / autodiff at scale:** tf-quant-finance or `fast-vollib`.
- **Vol surfaces:** use the in-engine SVI/SABR of QuantLib/FinancePy; treat standalone repos as references.
- **Live US options trading:** `ib_async` (IBKR), `alpaca-py` (Alpaca), `tastyware/tastytrade` (Tastytrade).
- **Backtest options strategies:** `optopsy`; for full event-driven multi-asset, NautilusTrader.

## Avoid as a hard dependency

`ib_insync` (🗄️ use `ib_async`), official `tastytrade/tastytrade-sdk-python` (🗄️ use tastyware), `mibian` / `optlib` data layer / old `vollib` SWIG build (🔴 unmaintained), and any "deribit-py"/"pyfin" repo (no maintained canonical project — integrate via official docs / general engines).

---

**Sources:** [QuantLib](https://github.com/lballabio/QuantLib), [QuantLib-SWIG](https://github.com/lballabio/QuantLib-SWIG), [PyQL](https://github.com/enthought/pyql), [FinancePy](https://github.com/domokane/FinancePy), [tf-quant-finance](https://github.com/google/tf-quant-finance), [py_vollib](https://github.com/vollib/py_vollib), [py_lets_be_rational](https://github.com/vollib/py_lets_be_rational), [vollib](https://github.com/vollib/vollib), [py_vollib_vectorized](https://github.com/marcdemers/py_vollib_vectorized), [fast-vollib](https://github.com/raeidsaqur/fast-vollib) ([arXiv:2604.27210](https://arxiv.org/abs/2604.27210)), [blackscholes](https://github.com/CarloLepelaars/blackscholes), [mibian/MibianLib](https://github.com/yassinemaaroufi/MibianLib), [optlib](https://github.com/dbrojas/optlib), [pysabr](https://github.com/ynouri/pysabr), [ssvi](https://github.com/arkonique/ssvi), [Arbitrage-Free-Volatility-Surface](https://github.com/XanderRobbins/Arbitrage-Free-Volatility-Surface), [ib_async](https://github.com/ib-api-reloaded/ib_async), [ib_insync](https://github.com/erdewit/ib_insync), [alpaca-py](https://github.com/alpacahq/alpaca-py), [tastyware/tastytrade](https://github.com/tastyware/tastytrade), [tastytrade-sdk-python](https://github.com/tastytrade/tastytrade-sdk-python), [wallstreet](https://github.com/mcdallas/wallstreet), [ccxt](https://github.com/ccxt/ccxt), [OpenBB](https://github.com/OpenBB-finance/OpenBB), [Deribit API docs](https://docs.deribit.com/), [optopsy](https://github.com/goldspanlabs/optopsy), [NautilusTrader](https://github.com/nautechsystems/nautilus_trader), [lumibot](https://github.com/Lumiwealth/lumibot), [vectorbt](https://github.com/polakowo/vectorbt) / [VectorBT PRO](https://vectorbt.pro/), [backtrader](https://github.com/mementum/backtrader), [python-bizdays](https://github.com/wilsonfreitas/python-bizdays).

**Keywords:** options pricing library, derivatives pricing, Greeks, implied volatility, Black-Scholes, Black-76, QuantLib, FinancePy, tf-quant-finance, py_vollib, py_lets_be_rational, py_vollib_vectorized, fast-vollib, blackscholes, mibian, optlib, volatility surface, SVI, SSVI, SABR, pysabr, local volatility, Heston, options data, brokerage API, Interactive Brokers, ib_async, ib_insync, Alpaca, alpaca-py, Tastytrade, tastyware, Deribit, ccxt, OpenBB, options backtesting, optopsy, NautilusTrader, lumibot, vectorbt, backtrader, risk, sensitivities; biblioteca de precificação de opções, derivativos, gregas, volatilidade implícita, superfície de volatilidade, backtesting de opções, B3, dias úteis 252, python-bizdays, ANBIMA, calendário Brasil, código aberto, open source.
