# Market Microstructure & HFT

> How prices form at the finest scale: the limit order book, market making, price impact, and the deep-learning models built on tick data. This is where latency, queue position, and execution quality live.

## Core concepts

- **Limit order book (LOB)**: bids/asks at each price level; market vs. limit orders; the spread; queue priority (price-time); matching engine.
- **Liquidity & impact**: bid-ask spread, depth, **price impact** (temporary vs. permanent), the square-root law of impact, Kyle's lambda.
- **Market making**: quote both sides, earn the spread, manage **inventory** and **adverse selection** (Avellaneda-Stoikov model).
- **HFT**: latency arbitrage, co-location, tick-to-trade time, order anticipation; market-quality debates (Flash Crash 2010).
- **Optimal execution**: split a large order to minimize impact + risk (Almgren-Chriss; see [Algo Trading](../Algorithmic_and_Quant_Trading/)).

## ML / deep learning on the LOB

| Model | Use |
|---|---|
| **DeepLOB** (CNN+LSTM on LOB) | Mid-price movement prediction from raw order book |
| Transformers / attention on LOB | Longer-range dependencies in order flow |
| RL for market making | Avellaneda-Stoikov-style quoting policies under inventory risk |
| RL for optimal execution | Learn schedules beating VWAP/IS |
| Hawkes / point processes | Model order arrival clustering, self-excitation |

Data is **huge and high-frequency** (nanosecond timestamps, message-level); careful handling of microstructure noise, queue dynamics, and realistic simulation (a backtest can't assume free fills) is essential.

## Data & tools

- **LOBSTER** (reconstructed Nasdaq LOB), Nasdaq TotalView-ITCH, CME MBO, Databento. Simulators: ABIDES (agent-based), `mbt_gym`. See [Datasets & APIs](../Datasets_APIs_and_Data_Vendors/).

## Key references

- Zhang, Zohren, Roberts, *DeepLOB: Deep CNNs for Limit Order Books* (2019). https://arxiv.org/abs/1808.03668
- Avellaneda & Stoikov, *High-frequency trading in a limit order book* (2008). https://www.tandfonline.com/doi/abs/10.1080/14697680701381228
- O'Hara, *Market Microstructure Theory* (1995).
- Bouchaud et al., *Trades, Quotes and Prices* (2018).

## Related in AIForge
- [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Equities & Stock Markets](../Equities_and_Stock_Markets/) · [Crypto & Digital Assets](../Crypto_and_Digital_Assets/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/) · [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Deep_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Deep_Learning/)

**Keywords:** market microstructure, limit order book, HFT, high-frequency trading, DeepLOB, market making, Avellaneda-Stoikov, price impact, optimal execution, LOBSTER, ABIDES.
