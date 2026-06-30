# Options & Derivatives

> Options give the right (not obligation) to buy (call) or sell (put) an underlying at a strike by an expiry. Derivatives derive value from an underlying (equity, index, rate, FX, commodity, crypto). This page covers pricing, the Greeks, volatility, and how ML reshapes hedging and valuation.

## Core concepts

- **Options**: calls/puts, American vs. European, moneyness (ITM/ATM/OTM), intrinsic vs. time value, payoff diagrams.
- **The Greeks**: delta (∂price/∂spot), gamma (∂delta), vega (∂vol), theta (∂time), rho (∂rate) — the sensitivities that drive hedging.
- **Other derivatives**: futures/forwards, swaps (IRS, CDS, total-return), swaptions, exotics (barrier, Asian, lookback), structured products.
- **Implied volatility**: the vol that reprices the market quote via Black-Scholes; aggregated into the **volatility surface** (smile/skew across strike & maturity) and term structure (e.g., VIX).

## Pricing — classical → ML

| Method | Use | Notes |
|---|---|---|
| Black-Scholes-Merton | European options, IV extraction | Closed-form; assumptions (constant vol, lognormal) break in practice |
| Binomial / trees, finite-difference (PDE) | American/exotic options | Numerical; flexible but slow |
| Monte Carlo | Path-dependent / high-dim exotics, XVA | Variance reduction; slow → prime target for ML surrogates |
| Stochastic vol (Heston, SABR), rough vol (rBergomi) | Realistic smile/skew | Calibration is expensive |
| **Neural network surrogates / deep calibration** | Fast pricing & calibration | NN learns price/IV map → near-instant calibration of Heston/rough-vol |
| **Deep Hedging** (Buehler et al., 2019) | Hedging under frictions | RL/deep learning hedges P&L directly under transaction costs/constraints, no closed form needed |
| Neural SDEs / signature methods | Market generation, path modeling | Emerging; market simulators, deep pricing |

## ML applications

- **Volatility forecasting**: GARCH-family baselines vs. LSTM/Transformer/HAR-ML; realized-vol nowcasting from high-frequency data.
- **IV-surface modeling & arbitrage-free interpolation**: neural nets / Gaussian processes that respect no-arbitrage constraints.
- **Deep hedging**: learn hedging policies that beat delta-hedging under real costs.
- **Vol/dispersion trading signals**, gamma exposure (GEX) analytics, and options-flow features for equity prediction.

## Data & tools

- QuantLib (open-source pricing/risk), `py_vollib`, `mibian`. Vol surfaces & options chains via OPRA, ORATS, CBOE DataShop, Polygon options.
- Crypto options (Deribit) expose rich, free options data for research.

## Key references

- Black & Scholes (1973); Merton (1973) — foundational.
- Buehler, Gonon, Teichmann, Wood, *Deep Hedging* (2019). https://arxiv.org/abs/1802.03042
- Horvath, Muguruza, Tomas, *Deep Learning Volatility* (2019) — deep calibration of rough-vol. https://arxiv.org/abs/1901.09647
- Gatheral, *The Volatility Surface* (2006).

## 🔮 Prediction toolkit
- [Options-Market Prediction](./Options_Market_Prediction/) — features (Greeks, IV surface, VRP, GEX), datasets (OptionMetrics/ORATS/Deribit/CBOE), and models & frontier techniques (vol forecasting, arbitrage-free IV-surface NN, deep hedging, generative surfaces) for forecasting volatility, the surface, and options-driven signals.

## 🌎 Options & derivatives deep-dives by country
| Page | What's inside |
|---|---|
| [B3 Options & Derivatives (Brazil)](./B3_Options_and_Derivatives_Brazil.md) | Opções sobre ações/Ibovespa, convenção de séries/letras, futuros WIN/IND, WDO/DOL, DI, agro, especificações de contrato, margem, day trade, COE. |
| [US Options & Derivatives](./US_Options_and_Derivatives.md) | Equity/ETF options (100x), SPX/XSP index options, OCC, VIX, weeklys/0DTE, strategies, CME futures (ES/MES, NQ/MNQ), OPRA. |

## 🧩 More on options & derivatives
- [Options Strategies & Analytics Tools](./Options_Strategies_and_Analytics_Tools.md) — strategy catalog (wheel, spreads, condors…) + builders (OptionStrat, OptionsProfitCalculator) and flow/GEX analytics (SpotGamma, Menthor Q, Unusual Whales, SqueezeMetrics); 🇧🇷 opcoes.net.br, OpLab, Opstra.
- [Exotic Options, Swaps & Structured Products](./Exotic_Options_and_Structured_Products.md) — barriers/Asians/autocallables, variance & VIX derivatives, IRS/CDS/TRS/swaptions, structured notes & COE 🇧🇷, warrants/CFDs/ETNs, ISDA/QuantLib.

## Related in AIForge
- [Risk Management & Derivatives Pricing](../Risk_Management_and_Derivatives_Pricing/) · [Futures & Commodities](../Futures_and_Commodities/) · [Crypto & Digital Assets](../Crypto_and_Digital_Assets/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/) · [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/)

**Keywords:** options pricing, Greeks, Black-Scholes, implied volatility surface, deep hedging, deep learning volatility, Heston calibration, rough volatility, QuantLib, derivatives ML.
