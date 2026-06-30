# Crypto Derivatives & Perpetual Futures

> Dedicated, current (2024-2026) deep-dive into crypto perpetual swaps, options, the venues/data/OSS that power them, and the ML angles — cross-linking, not repeating, the one-line perps/Deribit mention in [`../README.md`](../README.md).

## Perpetual mechanism crib (perpétuos)

| Concept | EN | PT-BR | Note |
|---|---|---|---|
| Perpetual swap | Perpetual future with no expiry | Contrato perpétuo | Invented by [BitMEX](https://www.bitmex.com/) in 2016; ~93% of crypto futures volume |
| Funding rate | Periodic long↔short payment tethering perp to spot | Taxa de financiamento | Positive → longs pay shorts; usually 8h (some venues 1h/4h) |
| Basis | Futures price − spot/index | Base | Cash-and-carry / funding arb signal |
| Mark vs Index price | Liquidation reference vs underlying composite | Preço de marca vs índice | Mark uses fair-value to prevent wick liquidations |
| Liquidation | Forced close when margin < maintenance | Liquidação | Insurance fund / ADL backstop |
| Open interest (OI) | Outstanding contract notional | Posições em aberto | Positioning & squeeze signal |

## CEX perpetual venues

| Venue | Notes | API / docs |
|---|---|---|
| Binance Futures | Largest perp OI/volume | [binance-docs.github.io](https://developers.binance.com/docs/derivatives) |
| Bybit | USDT + inverse perps | [bybit-exchange.github.io/docs](https://bybit-exchange.github.io/docs/) |
| OKX | Unified perps/options/spot, v5 API | [okx.com/docs-v5](https://www.okx.com/docs-v5/en/) |
| Deribit (by Coinbase) | #1 options venue; perps + DVOL; **acquired by Coinbase for ~$2.9B, closed Aug 2025** | [docs.deribit.com](https://docs.deribit.com/) |
| BitMEX | Original perpetual-swap inventor; inverse perps | [bitmex.com/api](https://www.bitmex.com/app/apiOverview) |
| Kraken Futures | Regulated-leaning, multi-collateral perps | [docs.kraken.com](https://docs.kraken.com/api/) |

## DEX perpetual venues

| Venue | Model | API / SDK | Maintenance |
|---|---|---|---|
| Hyperliquid | Onchain CLOB L1; dominant perp DEX | [hyperliquid-dex/hyperliquid-python-sdk](https://github.com/hyperliquid-dex/hyperliquid-python-sdk) | Active — v0.24.0 (Jun 2026), MIT |
| dYdX v4 | Cosmos appchain, fully onchain orderbook | [docs.dydx.xyz](https://docs.dydx.xyz/) · [dydxprotocol/v4-chain](https://github.com/dydxprotocol/v4-chain) | Active |
| GMX | Oracle-priced GM/GLV pools; Arbitrum/Avalanche | [docs.gmx.io](https://docs.gmx.io/) | Active |
| Vertex | Hybrid orderbook+AMM on Arbitrum | [docs.vertexprotocol.com](https://docs.vertexprotocol.com/) | Active |
| Aevo | L2 perps + options (ex-Ribbon, see below) | [api-docs.aevo.xyz](https://api-docs.aevo.xyz/) | Active |
| Aster | Privacy-focused perp DEX (Astherus+APX merger, late 2024); up to 1001x; ASTER TGE Sep 2025 | [docs.asterdex.com](https://docs.asterdex.com/) | Active |

All exchange APIs above expose **free** public market data (REST + WebSocket); trading requires keys/account.

## Crypto options venues

| Venue | Notes |
|---|---|
| [Deribit](https://www.deribit.com/) | Dominant BTC/ETH options; free historical data via [docs.deribit.com](https://docs.deribit.com/) |
| [OKX options](https://www.okx.com/trade-option) | Growing on-CEX options |
| [Bybit options](https://www.bybit.com/) | USDC-settled options |
| [Bullish](https://www.bullish.com/) | Institutional venue; options data syndicated to Laevitas / The Block |
| [Derive (ex-Lyra)](https://www.derive.xyz/) | Onchain options/perps; LYRA→DRV token, rebranded 2024-25 |
| [Premia (Blue)](https://www.premia.blue/) | Onchain options AMM; org [Premian-Labs](https://github.com/Premian-Labs), v3 BUSL-1.1 |
| [Aevo](https://www.aevo.xyz/) | Onchain options + perps |
| [Panoptic](https://panoptic.xyz/) | Perpetual, oracle-free options on Uniswap v3 |
| [GammaSwap](https://gammaswap.com/) | Long/short volatility vs AMM LP positions |

### DVOL — the "crypto VIX"
[Deribit DVOL](https://www.deribit.com/statistics/BTC/volatility-index) is a 30-day forward implied-vol index (VIX-style methodology) for BTC and ETH. Deribit launched DVOL **futures (BTC) on 2023-03-27**, giving isolated direction-independent vol exposure. Also surfaced free via [Glassnode](https://studio.glassnode.com/charts/derivatives.DvolOhlc) and Amberdata.

## Data & analytics providers

| Provider | Coverage | Free tier? |
|---|---|---|
| [Deribit API](https://docs.deribit.com/) | Options/perps incl. free history | Yes |
| [Coinglass](https://www.coinglass.com/) | OI, funding, liquidation heatmaps across 30+ venues | Yes (generous) |
| [Tardis.dev](https://tardis.dev/) | Tick-level historical (paid); OSS clients below | Limited free |
| [Laevitas](https://www.laevitas.ch/) | Options/perps analytics, IV/Greeks; MCP server | Yes (limited) |
| [Block Scholes](https://www.blockscholes.com/data) | Institutional vol surfaces (SVI), derivs API | Paid |
| [Amberdata / AD Derivatives (GVOL)](https://www.amberdata.io/ad-derivatives) | Derivs analytics, vol indices | Paid |
| [Kaiko](https://www.kaiko.com/) | Institutional market data/indices; **acquired Amberdata (announced Jun 2026)** | Paid |
| [Velo Data](https://velodata.app/) | Free terminal: OI, funding, basis, liquidations | Yes (free + $49/mo) |
| [Glassnode](https://glassnode.com/) | On-chain + derivatives/DVOL metrics | Limited free |
| [Coinalyze](https://coinalyze.net/) | Futures OI/funding/liquidations; free API | Yes |
| [CoinAPI](https://www.coinapi.io/) | Unified REST/WS + flat-file historical | Limited free |
| [CryptoDataDownload](https://www.cryptodatadownload.com/) | Free CSV OHLCV incl. Deribit options dumps | Yes |

**Suggested free stack:** Coinglass + Velo (positioning/funding) → Coinalyze API (programmatic) → Deribit + CryptoDataDownload (options history) → Glassnode DVOL.

## Structured products & DeFi derivatives

- **DOVs (DeFi Option Vaults):** automated covered-call / put-selling yield. [Ribbon Finance](https://messari.io/project/ribbon-finance/profile) merged into **[Aevo](https://www.aevo.xyz/)** (RBN→AEVO 1:1, 2023-24).
- **Onchain options protocols:** Derive (ex-Lyra), Premia Blue, Panoptic, GammaSwap.
- **Perp-DEX mechanics:** oracle-pool (GMX) vs onchain CLOB (Hyperliquid, dYdX v4) vs hybrid (Vertex).
- **MEV in derivatives:** liquidation sniping, oracle-update front-running, funding-snapshot games.

## ML / quant angles

| Angle | Idea |
|---|---|
| Funding-rate forecasting | Predict next funding to time cash-and-carry / basis trades |
| Vol-surface modeling | SVI/SSVI calibration, DVOL term-structure, IV skew signals |
| Liquidation cascade detection | OI + liquidation-heatmap features for squeeze risk |
| Cross-venue arb | CEX↔DEX funding/basis dispersion (mind costs & spread reversal) |
| RL execution / market-making | Inventory-aware quoting on perps |

### Verified papers (2022-2026)
- **Designing funding rates for perpetual futures in cryptocurrency markets** — [arXiv:2506.08573](https://arxiv.org/abs/2506.08573) (path-dependent BSDEs, replicating portfolios).
- **The Two-Tiered Structure of Cryptocurrency Funding Rate Markets** — MDPI *Mathematics* [14(2):346](https://www.mdpi.com/2227-7390/14/2/346) (CEX dominates price discovery, CEX→DEX info flow).
- **Exploring Risk and Return Profiles of Funding Rate Arbitrage on CEX and DEX** — ScienceDirect [S2096720925000818](https://www.sciencedirect.com/science/article/pii/S2096720925000818).
- **Panoptic: the perpetual, oracle-free options protocol** — [arXiv:2204.14232](https://arxiv.org/abs/2204.14232) (2022 whitepaper; foundational for perpetual options).

### OSS libraries
| Library | Use | Status |
|---|---|---|
| [ccxt/ccxt](https://github.com/ccxt/ccxt) | 100+ exchange unified REST/WS API (Py/JS/etc.) | Active, MIT |
| [hyperliquid-dex/hyperliquid-python-sdk](https://github.com/hyperliquid-dex/hyperliquid-python-sdk) | Hyperliquid trading SDK | Active (v0.24.0, Jun 2026), MIT |
| [tardis-dev/tardis-node](https://github.com/tardis-dev/tardis-node) | Tick-level real-time + historical (Node) | Active |
| [panoptic-labs/panoptic-v1-core](https://github.com/panoptic-labs/panoptic-v1-core) | Panoptic perpetual-options contracts | Active |
| Deribit clients | Official/community REST+WS clients; see [docs.deribit.com](https://docs.deribit.com/) | Varies |

## Brazil / B3 angle (alternativa regulada)

B3 offers regulated, exchange-cleared crypto derivatives as the onshore alternative to offshore perps:
- **BTC futures (BRL)** — contract size cut ~10× in 2025 for domestic retail access.
- **Bitcoin options** — listed alongside BTC futures.
- **USD-settled ETH (0.25 ETH) and SOL (5 SOL) futures** — launched **2025-06-16**, USD-denominated, Nasdaq ETH/SOL index reference, monthly cash settlement, aimed at international investors ([B3 ETH/SOL futures](https://clientes.b3.com.br/en/w/ethereum-and-solana-futures)).
- XRP/BNB/DOGE futures and event contracts under consideration for 2026.

No funding-rate mechanism (dated futures, not perps); no liquidation-fund/ADL idiosyncrasies; CVM oversight.

## Honest risk notes

- **Leverage & liquidation:** high leverage + thin liquidity → cascade liquidations; mark-price wicks.
- **24/7, no halts:** no circuit breakers; weekend gaps and oracle staleness bite.
- **Data quality:** wash trading inflates some CEX/DEX volume — prefer OI/funding over raw volume.
- **Survivorship:** dead/rebranded protocols (Lyra→Derive, Ribbon→Aevo) break links and skew backtests.
- **Regulatory access:** perps restricted in many jurisdictions (incl. US retail); verify eligibility.

## Sources
Deribit/Coinbase acquisition (coinbase.com, cnbc.com); Kaiko–Amberdata (kaiko.com, blog.amberdata.io); B3 ETH/SOL futures (clientes.b3.com.br, coinspeaker.com); Derive/Lyra (insights.derive.xyz); Ribbon→Aevo (blockworks.co); Aster (docs.asterdex.com); GitHub repos (ccxt, hyperliquid-dex, tardis-dev, panoptic-labs, dydxprotocol, Premian-Labs); papers (arXiv 2506.08573, 2204.14232; MDPI 14(2):346; ScienceDirect S2096720925000818). All URLs verified Jun 2026.

**Keywords / Palavras-chave:** perpetual futures, perpetual swap, funding rate, basis trade, crypto options, implied volatility, DVOL, liquidation, open interest, perp DEX, Hyperliquid, dYdX, Deribit, DeFi options vault · contratos perpétuos, taxa de financiamento, opções de cripto, volatilidade implícita, liquidação, posições em aberto, arbitragem de base, B3 futuros de cripto.
