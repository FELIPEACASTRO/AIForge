# Backtesting & Frameworks

> A backtest simulates a strategy on historical data. It is the single most abused tool in quant finance — most published/discovered "edges" are backtest artifacts. This page covers how to do it correctly and the open-source engines.

## How backtesting works

- **Vectorized** (fast, research): apply signals over a price matrix (`vectorbt`, `bt`). Great for sweeps; weak on path-dependent logic.
- **Event-driven** (realistic): a loop processes market events → signals → orders → fills, modeling latency, partial fills, and costs (`Zipline`, `Backtrader`, `LEAN`/QuantConnect, `Nautilus Trader`).

## The pitfalls (why backtests lie)

| Pitfall | What goes wrong |
|---|---|
| **Look-ahead bias** | Using data not available at decision time (restated fundamentals, future timestamps) |
| **Survivorship bias** | Universe excludes delisted/bankrupt names → inflated returns |
| **Data snooping / multiple testing** | Try enough strategies and one looks great by chance → use **deflated Sharpe ratio** |
| **Overfitting** | Too many parameters fit to noise; beautiful in-sample, dead live |
| **Unrealistic costs/fills** | Ignoring slippage, impact, borrow costs, capacity |
| **Regime dependence** | Backtest period doesn't represent the future |
| **p-hacking the split** | Repeatedly peeking at the test set |

## Doing it right

- **Point-in-time data**; survivorship-bias-free universes (include delisted).
- **Walk-forward / out-of-sample**; **purged & embargoed cross-validation** (López de Prado) to prevent leakage in overlapping-label time series.
- Realistic **transaction-cost & slippage models**; capacity/liquidity limits.
- **Deflated Sharpe ratio** and the **probability of backtest overfitting (PBO)** to discount multiple trials.
- Paper trade → small live → scale; monitor for decay.

## Open-source engines

| Tool | Type | Link |
|---|---|---|
| QuantConnect / LEAN | Event-driven, multi-asset, cloud + local | https://www.quantconnect.com/ |
| Zipline-reloaded | Event-driven (Quantopian successor) | https://github.com/stefan-jansen/zipline-reloaded |
| Backtrader | Event-driven Python | https://www.backtrader.com/ |
| vectorbt | Vectorized, fast | https://github.com/polakowo/vectorbt |
| bt | Flexible portfolio backtesting | https://github.com/pmorissette/bt |
| Nautilus Trader | High-performance event-driven (Rust/Python) | https://github.com/nautechsystems/nautilus_trader |
| Qlib (Microsoft) | AI-oriented quant platform | https://github.com/microsoft/qlib |
| backtesting.py | Lightweight | https://github.com/kernc/backtesting.py |

## Key references

- López de Prado, *Advances in Financial Machine Learning* (2018) — purged CV, deflated Sharpe, PBO.
- Bailey, Borwein, López de Prado, Zhu, *The Probability of Backtest Overfitting* (2014). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

## Related in AIForge
- [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Tools & Platforms](../Tools_and_Platforms/) · [Datasets & APIs](../Datasets_APIs_and_Data_Vendors/) · [Key Papers & Research](../Key_Papers_and_Research/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Model_Evaluation/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Model_Evaluation/)

**Keywords:** backtesting, walk-forward, purged cross-validation, look-ahead bias, survivorship bias, overfitting, deflated Sharpe ratio, probability of backtest overfitting, Zipline, Backtrader, vectorbt, QuantConnect, Qlib.
