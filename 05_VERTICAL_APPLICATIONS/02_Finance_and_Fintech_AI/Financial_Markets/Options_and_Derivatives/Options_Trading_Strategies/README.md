# 🎯 Options Trading Strategies

> A practical, fact-checked playbook for **buying and selling options** — how to choose a strategy, structure it, and manage it — covered at both the **micro** (execution, strike/expiry) and **macro** (hedging, regimes, events) level. Every payoff, breakeven, and Greeks profile was verified; every cited source/tool/book confirmed live.

This subsection is the strategy layer of [Options & Derivatives](../). For forecasting (vol/IV-surface/direction) see the sibling [Options-Market Prediction](../Options_Market_Prediction/). **Not investment advice — research & education only.**

## Core strategy pages

| Page | What's inside |
|---|---|
| [Strategy Selection & Playbooks](./Strategy_Selection_and_Playbooks.md) | The decision matrix (market view × volatility view × time), **when to BUY vs SELL options**, POP/expected value, IV rank, the high-POP-vs-payoff tradeoff; CBOE benchmark indices (BXM/PUT/CNDR/BFLY/CLL). |
| [Directional & Spread Strategies](./Directional_and_Spread_Strategies.md) | Long call/put, vertical spreads (debit/credit), backspreads, ratio spreads, risk reversal, synthetics, LEAPS, PMCC, collars, married put — payoff/breakeven/Greeks each. |
| [Volatility, Income & Neutral Strategies](./Volatility_Income_and_Neutral_Strategies.md) | Straddles/strangles, iron condor/butterfly, calendars/diagonals, dispersion + income: covered call, CSP, the Wheel, credit spreads, jade lizard, 0DTE — with variance-risk-premium rationale & risk warnings. |
| [Position Management, Adjustments & Risk](./Position_Management_Adjustments_and_Risk.md) | Greeks-based management, entry/exit rules (50% profit, 21 DTE), rolling, adjustments, assignment/early-exercise, position sizing, portfolio Greeks, tail hedging. |
| [Systematic & ML-Driven Strategies](./Systematic_and_ML_Options_Strategies.md) | CBOE strategy-benchmark indices, VRP harvesting, vol-targeting, ML/RL for strategy selection & deep hedging, backtesting (optopsy, ib_insync) and its pitfalls. |

> 🔜 More micro/macro pages (execution & strike selection, hedging & tail risk, earnings/event-driven, strategy encyclopedias, B3 Brazil strategies) are being finalized through an additional triple-check pass and will be added here.

## ⚠️ Risk note
No options strategy is "free money." Selling premium has high win-rate but undefined/large tail risk; buying premium fights theta and IV crush. Liquidity, bid-ask spreads, commissions, assignment, and sizing decide real outcomes. Always backtest with realistic costs ([Backtesting & Frameworks](../../Backtesting_and_Frameworks/)) and size for survival.

## Related in AIForge
- Parent: [`../`](../) (Options & Derivatives) · [Options-Market Prediction](../Options_Market_Prediction/) · [Risk Management & Derivatives Pricing](../../Risk_Management_and_Derivatives_Pricing/) · [Books, Courses & Learning Resources](../../Key_Papers_and_Research/Books_Courses_and_Learning_Resources.md)

**Keywords:** options trading strategies, covered call, cash-secured put, the wheel, iron condor, vertical spread, straddle strangle, credit spread, 0DTE, variance risk premium, options Greeks management, estratégias de opções, venda coberta, trava de alta.
