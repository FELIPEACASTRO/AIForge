# 🔮 Options-Market Prediction

> The ML/quant toolkit for **predicting the options market** — forecasting implied/realized volatility, modeling the IV surface, hedging, and extracting signals from options flow. Features, datasets, models, and frontier techniques, each fact-checked to confirm the paper/repo/dataset exists.

Options are the hardest, richest prediction problem in markets: the target (volatility, the surface, direction) is forward-looking, constrained by no-arbitrage, low signal-to-noise, and the best historical data is expensive. This subsection maps what actually works — and is honest about what doesn't.

| Page | What's inside |
|---|---|
| [Features for Options Prediction](./Features_for_Options_Prediction.md) | Greeks (delta/gamma/vanna/charm/volga), IV-surface features (ATM IV, skew/smirk, term structure, IV rank, **variance risk premium**, PCA factors), VIX/VVIX/SKEW/MOVE, options-flow & positioning (put/call, OI, **GEX/dealer gamma/max pain**), microstructure; with a starter feature set and leakage/no-arbitrage caveats. |
| [Datasets & Data Sources](./Datasets_and_Data_Sources_for_Options.md) | OptionMetrics IvyDB (1996+), ORATS, CBOE DataShop, **Deribit (crypto, free)**, Tardis/Amberdata/Laevitas, yfinance/Polygon/Tradier/Alpha Vantage chains, NSE F&O, B3 (MT5/OpLab/opcoes.net.br), Optiver Realized Volatility, CBOE VIX/SKEW free history — comparison table + free/paid + APIs. |
| [Models & Innovative Techniques](./Models_and_Innovative_Techniques_for_Options.md) | Vol forecasting (HAR-RV/Realized-EGARCH → LSTM/TFT/PatchTST), arbitrage-free IV-surface NN/GP/VAE & deep smoothing, deep calibration & rough vol, **Deep Hedging + RL**, pricing surrogates (Differential ML, deep BSDE), direction-from-options-flow, generative IV-surface (VolGAN/diffusion, FuNVol), 0DTE & LLMs — honest on overfitting & live-vs-backtest. |

## ⚠️ Honest framing
Implied vol is forward-looking and the surface must stay arbitrage-free, so naive ML easily produces look-ahead or arbitrageable predictions. Clean historical IV surfaces (OptionMetrics/ORATS) are costly; crypto options (Deribit) are the best free proxy. Treat every backtested edge with the [backtesting discipline](../../Backtesting_and_Frameworks/) (purged CV, deflated Sharpe). Research/education only — not investment advice.

## Related in AIForge
- Parent: [`../`](../) (Options & Derivatives) · [Risk Management & Derivatives Pricing](../../Risk_Management_and_Derivatives_Pricing/) · [Market Microstructure & HFT](../../Market_Microstructure_and_HFT/) · [Models, Features & Datasets](../../Models_Features_and_Datasets/) · [Frontier AI in Finance](../../Frontier_AI_in_Finance/)
- Fundamentals: [`../../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/`](../../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/)

**Keywords:** options prediction, implied volatility forecasting, IV surface ML, deep hedging, variance risk premium, gamma exposure GEX, volatility surface neural network, rough volatility, VolGAN, FuNVol, 0DTE, predição de opções, volatilidade implícita.
