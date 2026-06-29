# Equities & Stock Markets

> Stocks (shares) are fractional ownership of a company, traded on exchanges. This page covers how the equity market works and how ML is applied to stock selection, return prediction, and execution.

## How the market works

- **Instruments**: common/preferred stock, ADRs/GDRs, and the order book that prices them.
- **Venues**: exchanges (NYSE, Nasdaq, LSE, Euronext, JPX, B3, NSE/BSE), plus ATSs/dark pools and ECNs. Trading is largely electronic and order-driven.
- **Order types**: market, limit, stop, stop-limit, IOC/FOK, hidden/iceberg, MOO/MOC. Continuous trading plus opening/closing auctions.
- **Participants**: retail, asset managers, hedge funds, market makers, HFTs, pension/sovereign funds.
- **Corporate actions**: dividends, splits, buybacks, M&A — must be handled (adjusted prices) before any modeling.

## What you're predicting (and why it's hard)

Equity returns have very low short-horizon signal-to-noise; the **Efficient Market Hypothesis** is the null. Cross-sectional (which stock beats which) is often more tractable than absolute price level. Targets: next-period return, rank/ordering, volatility, direction, or "alpha" net of factor exposures.

## ML / techniques

| Approach | Use | Notes |
|---|---|---|
| Gradient-boosted trees (XGBoost/LightGBM/CatBoost) | Cross-sectional return ranking on fundamentals + technicals | The quant workhorse for tabular alpha; robust, fast |
| Factor models (Fama-French, Barra) | Risk decomposition, neutralization | ML augments classic value/size/momentum/quality/low-vol factors |
| LSTM/GRU, Temporal Fusion Transformer, N-BEATS | Sequence/return forecasting | Capture temporal structure; prone to overfit on noisy returns |
| Graph neural networks | Supply-chain / correlation / sector linkages | Model inter-stock relationships |
| Empirical asset pricing with ML (Gu, Kelly, Xiu) | Expected-return models | Landmark: ML measurably improves cross-sectional return prediction |
| LLMs / NLP | News, filings (10-K/10-Q), earnings-call sentiment | Feature generation; see [Alternative Data](../Alternative_Data_and_Sentiment_Analysis/) |
| Reinforcement learning | Position sizing, execution | See [Algo & Quant Trading](../Algorithmic_and_Quant_Trading/) |

Critical practices: **point-in-time** data (no look-ahead via restated fundamentals), survivorship-bias-free universes (include delisted names), purged/embargoed CV ([López de Prado](../Key_Papers_and_Research/)), transaction-cost & capacity modeling, and factor neutralization.

## Data & tools

- APIs: `yfinance`, Polygon.io, Alpaca, IEX Cloud, Tiingo, EOD Historical, Nasdaq Data Link. Academic: CRSP, Compustat, WRDS. See [Datasets & APIs](../Datasets_APIs_and_Data_Vendors/).
- Limit-order-book data: LOBSTER, Nasdaq TotalView-ITCH. See [Market Microstructure](../Market_Microstructure_and_HFT/).
- Backtesting: Zipline, Backtrader, vectorbt, QuantConnect. See [Backtesting & Frameworks](../Backtesting_and_Frameworks/).

## Key references

- Gu, Kelly, Xiu, *Empirical Asset Pricing via Machine Learning* (RFS 2020). https://academic.oup.com/rfs/article/33/5/2223/5758276
- López de Prado, *Advances in Financial Machine Learning* (2018). https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- Zhang et al., *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books* (2019). https://arxiv.org/abs/1808.03668

## 🌎 Market deep-dives by country
| Page | What's inside |
|---|---|
| [B3 — Brazilian Stock Market](./B3_Brazilian_Stock_Market.md) | Ações na B3: Novo Mercado, ON/PN/Units, tickers (PETR4/VALE3), Ibovespa, BDRs, lote/fracionário, D+2, tributação, corretoras, dados (brapi/MT5/yfinance .SA). |
| [US Stock Market — NYSE & Nasdaq](./US_Stock_Market_NYSE_Nasdaq.md) | Reg NMS, NBBO, S&P 500/Dow/Nasdaq/Russell, T+1, PFOF, ADRs, how Brazilians invest in the US (BDRs, Avenue/IBKR, W-8BEN). |

## Related in AIForge
- [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Portfolio Management](../Portfolio_Management_and_Optimization/) · [Technical & Fundamental Analysis](../Technical_and_Fundamental_Analysis/) · [Market Microstructure & HFT](../Market_Microstructure_and_HFT/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/) · [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/)

**Keywords:** stock market ML, equity return prediction, cross-sectional momentum, factor investing, empirical asset pricing machine learning, stock selection, DeepLOB.
