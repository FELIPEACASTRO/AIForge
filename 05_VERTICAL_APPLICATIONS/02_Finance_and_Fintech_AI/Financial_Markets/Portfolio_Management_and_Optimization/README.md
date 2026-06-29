# Portfolio Management & Optimization

> Turning return forecasts (or none) into positions, subject to risk and constraints. This page covers MPT, modern allocation methods, and ML for covariance and allocation.

## Foundations

- **Modern Portfolio Theory** (Markowitz 1952): mean-variance efficient frontier; trade off expected return vs. variance.
- **CAPM** and the role of the market portfolio; the Sharpe ratio.
- **Problems with naive mean-variance**: extreme sensitivity to estimation error in expected returns → unstable, concentrated weights ("error maximization").

## Modern allocation methods

| Method | Idea |
|---|---|
| Mean-variance (Markowitz) | Maximize return per unit variance |
| **Black-Litterman** | Blend market equilibrium with investor views (stabilizes weights) |
| **Risk parity** / equal risk contribution | Allocate by risk, not capital |
| **Hierarchical Risk Parity (HRP)** | López de Prado — clustering + recursive bisection; robust to ill-conditioned covariance, no matrix inversion |
| Kelly criterion | Growth-optimal bet sizing (and fractional Kelly) |
| Robust / resampled optimization | Account for estimation uncertainty |
| CVaR / drawdown optimization | Optimize tail risk instead of variance |

## ML applications

- **Covariance estimation**: shrinkage (Ledoit-Wolf), Random Matrix Theory denoising, factor-model covariance — critical because the sample covariance is noisy.
- **Expected-return inputs**: ML signals feed the optimizer (garbage-in caveat applies).
- **Deep RL for allocation**: end-to-end policies (FinRL, deep portfolios) — bypass the predict-then-optimize split.
- **Clustering / network methods**: detect correlation structure, diversify across regimes.

## Tools

- `PyPortfolioOpt`, `riskfolio-lib`, `cvxpy` (convex optimization), `scikit-portfolio`, `skfolio`, `Riskfolio-Lib`. See [Tools & Platforms](../Tools_and_Platforms/).

## Key references

- Markowitz, *Portfolio Selection* (JF 1952). https://www.jstor.org/stable/2975974
- Black & Litterman, *Global Portfolio Optimization* (FAJ 1992).
- López de Prado, *Building Diversified Portfolios that Outperform Out-of-Sample* (HRP, 2016). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
- Ledoit & Wolf, *Honey, I Shrunk the Sample Covariance Matrix* (2004).

## Related in AIForge
- [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Risk Management & Derivatives Pricing](../Risk_Management_and_Derivatives_Pricing/) · [ETFs, Funds & Indexing](../ETFs_Funds_and_Indexing/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/)

**Keywords:** portfolio optimization, Markowitz, modern portfolio theory, Black-Litterman, risk parity, hierarchical risk parity, HRP, Kelly criterion, Ledoit-Wolf shrinkage, PyPortfolioOpt, CVaR.
