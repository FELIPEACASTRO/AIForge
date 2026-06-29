# Foreign Exchange (FX)

> The currency market is the largest and most liquid in the world (~$7.5T/day, BIS 2022). This page covers FX structure, the canonical strategies, and ML for exchange-rate prediction.

## Core concepts

- **Structure**: spot, forwards, FX swaps, NDFs, options; majors (EUR/USD, USD/JPY, GBP/USD), crosses, EM pairs. Decentralized OTC interbank market + ECNs (EBS, Reuters) + retail.
- **Drivers**: interest-rate differentials, **covered/uncovered interest parity**, purchasing-power parity, balance of payments, central-bank policy, risk sentiment (risk-on/off).
- **Conventions**: pips, base/quote, carry, rollover/swap points.

## Canonical strategies (the "FX factors")

| Factor | Idea |
|---|---|
| Carry | Long high-yield / short low-yield currencies (earn rate differential) |
| Momentum | Trend persistence in exchange rates |
| Value | Mean-reversion toward PPP/fair value |
| Dollar / risk factors | Systematic exposure to USD and global risk |

The **forward premium puzzle / UIP failure** is the empirical anomaly carry exploits.

## ML applications

- **Exchange-rate forecasting**: notoriously hard — Meese-Rogoff (1983) showed a random walk beats structural models out-of-sample; ML must clear that bar. LSTM/Transformer/GBDT on rate diffs, macro, and order-flow.
- **High-frequency FX**: order-flow & microstructure prediction (deep LOB-style models).
- **Macro nowcasting & central-bank NLP**: parse statements/speeches (FinBERT/LLMs) for policy surprises.
- **Regime detection**: HMMs / clustering for risk-on/off switching.

## Data & tools

- Free/retail: `yfinance`, OANDA API, Dukascopy tick data, FRED (rates), HistData.com. Institutional: EBS, Refinitiv. BIS Triennial Survey for market structure.

## Key references

- Meese & Rogoff, *Empirical Exchange Rate Models of the Seventies* (1983) — the random-walk benchmark. https://www.sciencedirect.com/science/article/abs/pii/0022199683900179
- Lustig, Roussanov, Verdelhan, *Common Risk Factors in Currency Markets* (RFS 2011). https://academic.oup.com/rfs/article/24/11/3731/1590802
- Menkhoff et al., *Carry Trades and Global FX Volatility* (JF 2012).

## Related in AIForge
- [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Fixed Income & Bonds](../Fixed_Income_and_Bonds/) · [Market Microstructure & HFT](../Market_Microstructure_and_HFT/) · [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/)

**Keywords:** forex, FX trading, exchange rate forecasting, carry trade, currency momentum, Meese-Rogoff, UIP, deep learning FX, high-frequency forex, central bank NLP.
