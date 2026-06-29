# Technical & Fundamental Analysis

> The two classic schools of security analysis — price/volume patterns (technical) and intrinsic value from financials (fundamental) — and how ML formalizes and tests both.

## Technical analysis

- **Indicators**: moving averages (SMA/EMA), MACD, RSI, Bollinger Bands, stochastics, ATR, OBV, Ichimoku.
- **Patterns**: support/resistance, trend lines, candlestick patterns, chart formations (head-and-shoulders, triangles).
- **ML angle**: indicators become features for GBDT/NN; CNNs on chart *images*; pattern recognition; honest testing shows most retail TA has weak/no out-of-sample edge after costs — treat as features, not gospel.
- Libraries: `TA-Lib`, `pandas-ta`, `tulipy`.

## Fundamental analysis

- **Financial statements**: income statement, balance sheet, cash flow; ratios (P/E, P/B, ROE, ROIC, debt/equity, FCF yield).
- **Valuation**: DCF, comparables/multiples, dividend discount, residual income.
- **Factors**: value, quality, profitability, growth, accruals — the bridge to systematic investing.
- **ML angle**: GBDT/cross-sectional models on fundamentals (see [Equities](../Equities_and_Stock_Markets/)); financial-statement fraud detection (Beneish M-score, ML extensions); earnings/cash-flow forecasting.

## NLP on disclosures (the modern frontier)

- **10-K/10-Q, MD&A, earnings calls, transcripts**: sentiment, readability, tone, topic modeling.
- **FinBERT** and LLMs extract sentiment, risk factors, and guidance; LLMs summarize filings and flag changes year-over-year. See [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/).
- **Loughran-McDonald** finance sentiment lexicon (domain-specific — generic lexicons mislabel finance text).

## Data & tools

- Fundamentals: SEC EDGAR (free filings + XBRL), `sec-edgar-downloader`, Financial Modeling Prep, SimFin, `yfinance`. Compustat/WRDS (academic). TA: `TA-Lib`, `pandas-ta`.

## Key references

- Graham & Dodd, *Security Analysis*; Graham, *The Intelligent Investor* — fundamental canon.
- Lo, Mamaysky, Wang, *Foundations of Technical Analysis* (JF 2000) — rigorous TA test. https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00265
- Loughran & McDonald, *When Is a Liability Not a Liability?* (JF 2011) — finance sentiment. https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2010.01625.x
- Araci, *FinBERT* (2019). https://arxiv.org/abs/1908.10063

## Related in AIForge
- [Equities & Stock Markets](../Equities_and_Stock_Markets/) · [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/) · [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/)
- Fundamentals: [`../../../../02_LLM_AND_AI_MODELS/`](../../../../02_LLM_AND_AI_MODELS/)

**Keywords:** technical analysis, fundamental analysis, RSI MACD Bollinger, valuation DCF, financial ratios, factor investing, FinBERT, SEC EDGAR, Loughran-McDonald, earnings call NLP, TA-Lib.
