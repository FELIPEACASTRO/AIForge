# Algorithmic & Quantitative Trading

> Systematic trading turns hypotheses into rules, code, and capital. This page covers the quant research workflow, strategy families, execution algorithms, and reinforcement learning for trading.

## The quant research workflow

1. **Idea / hypothesis** — economic rationale (why does this edge exist and persist?).
2. **Data** — clean, point-in-time, survivorship-free; align timestamps; handle corporate actions.
3. **Signal / alpha research** — features → predictive signal; cross-sectional or time-series.
4. **Portfolio construction** — sizing, risk constraints, neutralization (see [Portfolio Management](../Portfolio_Management_and_Optimization/)).
5. **Backtest** — realistic costs/slippage, purged CV, walk-forward (see [Backtesting](../Backtesting_and_Frameworks/)).
6. **Execution** — minimize impact via execution algos.
7. **Live / monitoring** — paper → small live → scale; monitor decay & drift.

## Strategy families

| Family | Edge |
|---|---|
| Momentum / trend | Persistence of returns |
| Mean reversion / stat-arb / pairs | Cointegration, relative value |
| Market making | Capture spread, manage inventory (see [Microstructure](../Market_Microstructure_and_HFT/)) |
| Factor / risk premia | Value, quality, carry, low-vol |
| Event-driven | Earnings, M&A, index rebalances |
| Sentiment / alt-data | News, social, satellite (see [Alternative Data](../Alternative_Data_and_Sentiment_Analysis/)) |

## ML & reinforcement learning

- **Supervised**: GBDT (workhorse for tabular alpha), neural nets for sequences; predict returns/ranks/vol.
- **Reinforcement learning**: frame trading/execution as MDP; **FinRL** library; deep RL for portfolio allocation and optimal execution (Nevmyvaka-Feng-Kearns). Caveats: non-stationary env, reward shaping, sim-to-real gap.
- **Execution algorithms**: VWAP, TWAP, **Implementation Shortfall** (Almgren-Chriss optimal execution), POV, adaptive/learned schedulers.
- **Meta-labeling** (López de Prado): a secondary ML model sizes/filters a primary signal.

Pitfalls: overfitting & multiple testing (the **deflated Sharpe ratio**), look-ahead/survivorship bias, ignoring capacity/costs, p-hacking the backtest.

## Tools

QuantConnect/LEAN, Zipline-reloaded, Backtrader, vectorbt, `bt`, Qlib (Microsoft), FinRL, broker APIs (Alpaca, IBKR). See [Tools & Platforms](../Tools_and_Platforms/) and [Backtesting](../Backtesting_and_Frameworks/).

## Key references

- López de Prado, *Advances in Financial Machine Learning* (2018) — meta-labeling, purged CV, deflated Sharpe.
- Almgren & Chriss, *Optimal Execution of Portfolio Transactions* (2000). https://www.math.nyu.edu/~almgren/papers/optliq.pdf
- Liu et al., *FinRL: Deep Reinforcement Learning Framework* (2020). https://arxiv.org/abs/2011.09607
- Yang, Liu, Zhong, Walid, *Deep RL for Automated Stock Trading* (2020). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996

## Related in AIForge
- [Portfolio Management](../Portfolio_Management_and_Optimization/) · [Backtesting & Frameworks](../Backtesting_and_Frameworks/) · [Market Microstructure & HFT](../Market_Microstructure_and_HFT/) · [Tools & Platforms](../Tools_and_Platforms/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/) · [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/)
- Existing: [`../../Algorithmic_Trading/`](../../Algorithmic_Trading/) · [`../../Market_Prediction/`](../../Market_Prediction/)

**Keywords:** algorithmic trading, quantitative trading, alpha research, statistical arbitrage, pairs trading, FinRL, reinforcement learning trading, optimal execution, Almgren-Chriss, meta-labeling, deflated Sharpe ratio, Qlib.
