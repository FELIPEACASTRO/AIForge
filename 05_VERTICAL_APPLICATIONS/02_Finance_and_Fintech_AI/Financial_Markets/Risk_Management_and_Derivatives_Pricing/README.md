# Risk Management & Derivatives Pricing

> Measuring and pricing risk: VaR/ES, stress testing, Monte Carlo, and the neural methods now accelerating derivatives valuation and risk computation.

## Risk measurement

- **Value at Risk (VaR)** & **Expected Shortfall (ES/CVaR)**: tail-loss measures (historical, parametric, Monte Carlo); ES is coherent and now the Basel standard (FRTB).
- **Greeks & sensitivities**: portfolio delta/gamma/vega; scenario and stress testing.
- **Volatility models**: GARCH-family, EWMA, realized/HAR, stochastic vol; ML/Transformer vol forecasting.
- **Correlation/copulas & tail dependence**; the role of the **Gaussian copula** in 2008.
- **Backtesting risk models** (Kupiec, Christoffersen tests); **model risk** governance (**SR 11-7**, EU AI Act).

## Derivatives pricing — classical → ML

| Method | Use |
|---|---|
| Closed-form (Black-Scholes), trees, finite-difference (PDE) | Vanilla & simple exotics |
| **Monte Carlo** | Path-dependent / high-dimensional payoffs, XVA |
| Longstaff-Schwartz (LSM) | American options via regression |
| **Neural network pricers / surrogates** | Learn the price/Greeks map → real-time pricing, fast calibration |
| **Deep BSDE** (Han, Jentzen, E, 2018) | Solve high-dimensional PDEs/BSDEs that defeat grids |
| **Differential ML** (Huge & Savine) | Train on pathwise differentials for fast, accurate Greeks |
| **Deep Hedging** | Hedge under frictions (see [Options & Derivatives](../Options_and_Derivatives/)) |

**XVA** (CVA/DVA/FVA/MVA) — counterparty/funding valuation adjustments — is enormously compute-heavy and a prime ML-acceleration target.

## Tools

- QuantLib (pricing/risk engine), `tf-quant-finance` (Google), ORE (Open Source Risk Engine), `PyMC`/`Stan` for Bayesian risk, `arch` (GARCH).

## Key references

- Jorion, *Value at Risk* (3rd ed.).
- Han, Jentzen, E, *Solving high-dimensional PDEs using deep learning* (Deep BSDE, PNAS 2018). https://www.pnas.org/doi/10.1073/pnas.1718942115
- Huge & Savine, *Differential Machine Learning* (2020). https://arxiv.org/abs/2005.02347
- Buehler et al., *Deep Hedging* (2019). https://arxiv.org/abs/1802.03042

## Related in AIForge
- [Options & Derivatives](../Options_and_Derivatives/) · [Portfolio Management](../Portfolio_Management_and_Optimization/) · [Fixed Income & Bonds](../Fixed_Income_and_Bonds/)
- Existing: [`../../Risk_Management/`](../../Risk_Management/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Probabilistic_ML/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Probabilistic_ML/)

**Keywords:** value at risk, expected shortfall, CVaR, Monte Carlo pricing, deep BSDE, differential machine learning, XVA, GARCH, neural network option pricing, model risk SR 11-7, FRTB.
