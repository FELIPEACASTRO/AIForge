# Crypto & Digital Assets

> Crypto markets trade 24/7 with uniquely transparent **on-chain data**. This page covers crypto market structure, DeFi, and ML on blockchain and market data.

## Core concepts

- **Assets**: BTC, ETH, stablecoins (USDT/USDC), altcoins, tokens (governance/utility), NFTs.
- **Market structure**: centralized exchanges (CEX: Binance, Coinbase) vs. **decentralized exchanges (DEX:** Uniswap, Curve — AMMs with constant-product `x·y=k`); spot, **perpetual futures** (funding rate), options (Deribit).
- **On-chain primitives**: wallets, transactions, smart contracts, gas; **DeFi** (lending: Aave/Compound; AMMs; yield/staking); **MEV** (maximal extractable value, sandwich/arbitrage).
- **Distinctive features**: 24/7, high volatility, reflexive/narrative-driven, transparent ledgers (a data goldmine), but rife with manipulation/wash trading.

## ML / AI applications

| Application | Idea |
|---|---|
| Price/volatility forecasting | LSTM/Transformer/GBDT — same low-SNR caveats, amplified |
| On-chain analytics | Entity clustering, wallet labeling, illicit-flow detection (GNNs — see [Banking AML](../../Banking/Transaction_Monitoring_and_AML/)) |
| DeFi & AMM strategies | LP optimization, impermanent-loss modeling, arbitrage |
| MEV / market making | RL & optimization for execution and searcher strategies |
| Sentiment / social signals | Twitter/X, Reddit, Telegram, Farcaster NLP; narrative detection |
| Anomaly / scam detection | Rug-pull, pump-and-dump, contract-exploit detection |

## Data & tools

- Market: CCXT (unified exchange API), Binance/Coinbase/Kraken APIs, CoinGecko, CoinMarketCap, Kaiko, Amberdata. On-chain: Etherscan, Dune Analytics, The Graph, Glassnode, Nansen, Chainalysis. Options: Deribit (free, rich).

## Key references

- Nakamoto, *Bitcoin: A Peer-to-Peer Electronic Cash System* (2008). https://bitcoin.org/bitcoin.pdf
- Weber et al., *Anti-Money Laundering in Bitcoin: GCNs for Financial Forensics* (2019) — Elliptic. https://arxiv.org/abs/1908.02591
- Daian et al., *Flash Boys 2.0* (MEV, 2019). https://arxiv.org/abs/1904.05234

## 📉 Crypto derivatives
- [Crypto Derivatives & Perpetual Futures](./Crypto_Derivatives_and_Perpetuals.md) — perps & funding rates (Binance/Bybit/OKX; DEX: dYdX/GMX/Hyperliquid), crypto options (Deribit/Aevo/Derive), analytics (Coinglass/Laevitas/Velo), and ML angles (funding/basis/liquidations).

## Related in AIForge
- [Options & Derivatives](../Options_and_Derivatives/) · [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Market Microstructure & HFT](../Market_Microstructure_and_HFT/) · [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/)
- Sibling: [`../../Banking/Transaction_Monitoring_and_AML/`](../../Banking/Transaction_Monitoring_and_AML/) (on-chain AML)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/)

**Keywords:** crypto trading ML, on-chain analytics, DeFi, AMM, perpetual futures, MEV, blockchain machine learning, crypto sentiment, rug pull detection, CCXT.
