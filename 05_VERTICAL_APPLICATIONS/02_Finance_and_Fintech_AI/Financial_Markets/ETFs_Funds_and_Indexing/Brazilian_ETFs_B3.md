# Brazilian ETFs on B3 (ETFs na Bolsa Brasileira)

> A reference on Exchange-Traded Funds (Fundos de Índice) listed on B3 — the Brazilian exchange — covering tickers, reference indices, fees, trading mechanics, taxation, and the ML/indexing angle for quants and investors.

An **ETF** (Exchange-Traded Fund, *Fundo de Índice* in Brazil) is a fund whose shares (*cotas*) trade intraday on an exchange like a stock. In Brazil they are regulated by the **CVM** under **Resolução CVM 175** — specifically its **Anexo V (Fundos de Índice / ETF)** — which superseded the old **Instrução CVM 359/2002** as part of the broad investment-fund overhaul introduced by Resolução CVM 184/2023. ETFs are listed and traded on **B3 S.A. – Brasil, Bolsa, Balcão** (the merged exchange, ex-BM&FBOVESPA). Each ETF aims to replicate the return of a **reference index** (*índice de referência*) net of fees, using full or sampled physical replication or, for international/fixed-income products, fund-of-fund (investing in a foreign ETF) and futures-based structures. B3 maintains the official lists of listed ETFs ([equity / renda variável](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-variavel/etfs-listados/), [fixed income / renda fixa](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-fixa/etfs-listados/)).

## How ETFs trade on B3

- **Trading venue & system:** B3 spot market (*mercado à vista*), PUMA Trading System. Quotes are in BRL (R$).
- **Regular session (pregão):** as of the 9-March-2026 schedule, pre-opening from 09:45, continuous trading **10:00–16:55**, with a closing call thereafter ([B3 trading hours](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/horario-de-negociacao/acoes/)). Times can shift relative to the U.S. daylight-saving window — confirm the current schedule.
- **After-market:** **17:30–18:00** (Brasília time), and **restricted** to spot-market assets that (a) belong to the theoretical portfolio of the **Ibovespa, IBrX-50, IBrX-100 or IFIX**, and (b) actually traded in the regular session that day; each order is limited to **±2%** of the regular-session close. Fixed-income ETFs, BDRs and most low-liquidity assets cannot trade after-market ([B3 trading hours](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/horario-de-negociacao/acoes/)).
- **Lot size (lote):** for stocks the standard lot is **100 shares**, with a separate fractional book (*mercado fracionário*) for 1–99 shares using a trailing **`F`** ticker. **For ETFs, B3 set the standard lot to 1 unit (since 28-Sep-2020)**, so ETFs trade from **1 share** on the regular book under their normal ticker — no `F` suffix is used ([Traders.com.br](https://www.traders.com.br/blog/posts/lote-padrao-mercado-fracionario)).
- **Ticker convention:** equity/index ETFs (like FIIs) end in **11** (e.g. `BOVA11`). Settlement is **T+2** via the central depository.

## Main equity ETFs (renda variável)

| Ticker | Fund / ETF name | Reference index | Gestor / Administrador | Admin fee (a.a.) |
|---|---|---|---|---|
| BOVA11 | iShares Ibovespa | Ibovespa | BlackRock (admin. BNP Paribas) | ~0.10% (reduced from 0.30%) ([BlackRock](https://www.blackrock.com/br/products/251816/ishares-ibovespa-fundo-de-ndice-fund)) |
| BOVV11 | It Now Ibovespa | Ibovespa | Itaú Asset | ~0.10% |
| PIBB11 | It Now PIBB IBrX-50 | IBrX-50 | Itaú Asset | ~0.059% ([Itaú factsheet](https://assetfront.arquivosparceiros.cloud.itau.com.br/FND/ITAUPIBBIBX843_It_Now_PIBB_IBrX-50.pdf)) |
| SMAL11 | iShares Small Cap | Índice Small Cap (SMLL) | BlackRock (admin. BNP Paribas) | ~0.50% (some 2026 sources up to ~0.69%) ([BlackRock](https://www.blackrock.com/br/products/251752/ishares-bmfbovespa-small-cap-fundo-de-ndice-fund)) |
| DIVO11 | It Now IDIV | IDIV (dividendos) | Itaú Asset | ~0.50% ([It Now](https://www.itnow.com.br/divo11/)) |
| FIND11 | It Now IFNC | IFNC (setor financeiro) | Itaú Asset | ~0.60% |
| IVVB11 | iShares S&P 500 (BRL) | S&P 500, in R$ (não-hedgeado) | BlackRock (admin. BNP Paribas) | ~0.23% ([BlackRock](https://www.blackrock.com/br/products/251902/ishares-sp-500-fi-em-cotas-de-fundo-de-ndice-inv-no-exterior-fund)) |
| GOLD11 | Trend ETF LBMA Ouro | LBMA Gold Price (≥95% via iShares Gold Trust, IAU) | XP Vista Asset (admin. BNP Paribas) | ~0.30% + ~0.25% underlying IAU ([XP Asset](https://www.xpasset.com.br/fundos/gold11/)) |

Notes: `IVVB11` invests predominantly in the **iShares Core S&P 500 ETF (IVV)**; it gives S&P 500 exposure but is **not currency-hedged**, so its return blends U.S. equity performance with USD/BRL (*câmbio*). A cheaper local alternative is `SPXI11` (Itaú, ~0.18%). `SMAL11` tracks small caps (the SMLL, rebalanced quarterly) and is more volatile than the blue-chip Ibovespa basket. The Ibovespa is B3's flagship index.

## Fixed-income ETFs (renda fixa)

These hold federal government bonds (*títulos públicos*) tracking ANBIMA's IMA family, or replicate DI-futures interest-rate indices.

| Ticker | ETF name | Reference index | Issuer | Admin fee (a.a.) |
|---|---|---|---|---|
| IMAB11 | It Now IMA-B | IMA-B (NTN-B / inflation-linked, all maturities) | Itaú Asset | ~0.25% ([Traders.com.br](https://www.traders.com.br/blog/posts/etfs-renda-fixa-b3-imab11-irfm11)) |
| IB5M11 | It Now IMA-B 5+ | IMA-B 5+ (inflation-linked, >5y) | Itaú Asset | ~0.25% |
| B5P211 | It Now IMA-B 5 P2 | IMA-B 5 P2 (inflation-linked, short) | Itaú Asset | ~0.20% |
| IRFM11 | It Now IRF-M | IRF-M (LTN/NTN-F, prefixed / *prefixado*) | Itaú Asset | ~0.20% |
| FIXA11 | BB ETF Renda Fixa Pré S&P/B3 | S&P/B3 Interest Rate Futures Index — DI 3 Years | BB Asset | ~0.30% ([BB Asset](https://bb.com.br/site/bb-asset-management/fundos-de-investimento/etf/b3-futuros-de-taxas-de-juros-fundo-de-indice/)) |

Fixed-income ETFs offer bond/rate exposure with **no *come-cotas*** (the semiannual withholding that erodes traditional bond funds). `IMAB11` is the most liquid of the group ([Traders.com.br](https://www.traders.com.br/blog/posts/etfs-renda-fixa-b3-imab11-irfm11)).

## International, thematic and crypto ETFs

| Ticker | ETF name | Exposure | Issuer | Admin fee (a.a.) |
|---|---|---|---|---|
| NASD11 | Trend ETF Nasdaq-100 | Nasdaq-100 (via Invesco QQQ) | XP Asset (admin. BNP Paribas) | ~0.30% ETF + ~0.20% underlying QQQ (~0.50% total) ([XP Asset](https://www.xpasset.com.br/fundos/nasd11/)) |
| EURP11 | Trend ETF Europa | European equities | XP Asset | see [B3 comparator](https://borainvestir.b3.com.br/comparador-de-etfs/) |
| ASIA11 | Trend ETF Ásia (ex-Japão) | Asian equities ex-Japan | XP Asset | see [B3 comparator](https://borainvestir.b3.com.br/comparador-de-etfs/) |
| HASH11 | Hashdex Nasdaq Crypto Index | Nasdaq Crypto Index (NCI), basket of ~9 cryptos | Hashdex | ~0.30% ([InvestNews](https://investnews.com.br/financas/etfs-de-criptomoedas-lista-completa-b3/)) |
| QBTC11 | QR CME CF Bitcoin Ref. Rate | 100% Bitcoin (CME CF BRR) | QR Asset (admin. Vórtx) | ~0.75% ([QR Asset](https://qrasset.com.br/qbtc11/)) |
| QETH11 | QR CME CF Ether Ref. Rate | 100% Ether (CME CF Ether Ref. Rate) | QR Asset (admin. Vórtx) | ~0.75% ([QR Asset](https://qrasset.com.br/qeth11/)) |

`HASH11` (launched **26 April 2021**) was the first crypto ETF on B3 and tracks the **Nasdaq Crypto Index (NCI)**, developed in partnership between Nasdaq and Hashdex. The NCI is a dynamic basket — as of mid-2025 it held roughly nine constituents (BTC, ETH, XRP, SOL, ADA, LINK, LTC, XLM, UNI). B3 has since listed many additional crypto ETFs ([InvestNews](https://investnews.com.br/financas/etfs-de-criptomoedas-lista-completa-b3/)).

## Largest issuers (gestoras)

| Issuer | Flagship products | Notes |
|---|---|---|
| BlackRock iShares Brasil | BOVA11, SMAL11, IVVB11 | Pioneer; BOVA11 is the most traded equity ETF (admin. typically BNP Paribas) |
| Itaú Asset (It Now) | BOVV11, PIBB11, DIVO11, FIND11, IMAB11, IB5M11, B5P211, IRFM11 | Broadest local lineup, equity + fixed income |
| XP Asset (Trend) | NASD11, EURP11, ASIA11, GOLD11 | International/thematic specialist (admin. BNP Paribas) |
| BB Asset | FIXA11 | Fixed-income / interest-rate ETF |
| Investo | thematic ETFs | Independent ETF specialist |
| Hashdex / QR Asset | HASH11, QBTC11, QETH11 | Crypto-focused issuers |

## Creation / redemption (criação e resgate)

ETF shares are created and destroyed in large blocks via **Authorized Participants (APs)** — typically brokers/banks. To create, an AP delivers the basket of underlying securities (or cash for some products) to the fund and receives new shares; redemption reverses this *in-kind*. This **primary-market arbitrage** keeps the secondary-market price tethered to the fund's intraday net asset value (**iNAV / valor patrimonial**): if the ETF trades at a premium, APs create and sell; at a discount, they buy and redeem. Retail investors transact only in the **secondary market** on B3; minimum creation/redemption block sizes are defined in each prospectus (e.g. lots of 100,000 shares in the primary market for some products).

## Costs and liquidity

- **Taxa de administração** ranges from ~**0.059%** (PIBB11) to ~**0.75%** a.a. (QBTC11/QETH11); broad equity ETFs cluster near 0.10–0.30%.
- **Tracking error** vs. the index, **bid/ask spread**, and **premium/discount to iNAV** are the real frictions beyond the headline fee.
- **Brokerage (corretagem)** is frequently **R$0** at major Brazilian brokers; B3 *emolumentos* still apply.
- Liquidity is concentrated: BOVA11, IVVB11, SMAL11, HASH11 and IMAB11 dominate average daily volume; many niche ETFs are thinly traded — check the order book before sizing.

## Taxation of ETFs in Brazil (tributação)

> **Critical:** unlike individual stocks, **equity ETFs do NOT receive the R$20,000/month exemption** (*isenção de R$20 mil*) on sales — every realized gain is taxable ([B3 / Bora Investir](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/etfs/tributacao-etf-e-bdr/)).

| ETF type | Rate | Mechanics |
|---|---|---|
| Equity / crypto ETF — normal | **15%** on the gain | Self-assessed via DARF (due by the last business day of the following month); no exemption band; losses offset gains in same/future months ([B3](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/etfs/tributacao-etf-e-bdr/)) |
| Equity / crypto ETF — day trade | **20%** on the gain | Same-day buy/sell |
| FII-type ETF | **20%** | Normal and day-trade treated alike |
| Fixed-income ETF | **regressive 25% / 20% / 15%** | By the **portfolio's average duration**, NOT your holding period ([B3](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/etfs/tributacao-etf-e-bdr/)) |

Fixed-income ETF regressive table (by the ETF's portfolio average duration, per B3): up to **6 months → 25%**; **6 months to 2 years → 20%**; **over 2 years → 15%**. For fixed-income ETFs the tax is **withheld at source**, and there is **no come-cotas**. Equity-ETF tax is paid by the investor via DARF. Always confirm current rules — Brazil's investment-tax framework has been under reform; verify against the CVM/Receita Federal and B3 for 2025–2026 specifics. (Note: proposals to introduce withholding (*IR-fonte*) on equity-ETF gains and other changes have circulated in tax-reform bills — none is in force as of 2026; confirm enactment before relying on it.)

## ML / indexing & replication angle

ETFs are a natural laboratory for quantitative and machine-learning methods:

- **Index replication & tracking-error minimization:** sampled replication for illiquid baskets (e.g. small caps) is an optimization problem — minimize expected tracking error subject to turnover/transaction-cost constraints. Approaches: cardinality-constrained portfolio optimization, LASSO/elastic-net on constituent returns, and reinforcement learning for rebalancing schedules.
- **Statistical arbitrage / pairs:** cointegration between an ETF and its iNAV, or between correlated ETFs (BOVA11 vs BOVV11 vs PIBB11), supports mean-reversion strategies; premium/discount-to-iNAV signals feed RL execution agents.
- **Factor & smart-beta construction:** DIVO11 (dividend), FIND11 (sector) are rules-based factor sleeves; ML is used to forecast factor returns and for regime detection (HMMs, change-point models) to rotate between Ibovespa, small-cap, gold and S&P exposures.
- **Crypto-basket weighting:** HASH11-style indices use liquidity/market-cap weighting; ML aids volatility forecasting (GARCH/LSTM) and risk parity across crypto sleeves.
- **Currency overlay:** IVVB11/NASD11 returns decompose into asset + USD/BRL; FX forecasting and hedging models are directly relevant.

### Data & APIs

| Source | What it gives | Access |
|---|---|---|
| [brapi.dev](https://brapi.dev/docs/acoes) | B3 quotes, history, dividends, fundamentals for stocks/ETFs/FIIs/BDRs | Free tier + token |
| `yfinance` (Yahoo Finance) | OHLCV for tickers like `BOVA11.SA`, `IVVB11.SA` | Free Python lib |
| [B3 for Developers](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/b3-for-developers/) | Official market-data APIs, index methodologies, theoretical portfolios | B3 (some paid) |
| [B3 Bora Investir comparator](https://borainvestir.b3.com.br/comparador-de-etfs/) | Official ETF comparator, fees, composition | Free, web |
| Issuer factsheets | Holdings, fees, prospectuses — e.g. [BlackRock BOVA11](https://www.blackrock.com/br/products/251816/ishares-ibovespa-fundo-de-ndice-fund), [Itaú PIBB11](https://assetfront.arquivosparceiros.cloud.itau.com.br/FND/ITAUPIBBIBX843_It_Now_PIBB_IBrX-50.pdf) | Free PDFs |
| ANBIMA | IMA index family methodology & daily values | Free |

For backtesting, pair B3 EOD data (from B3 BDI files or brapi) with index methodologies from B3/ANBIMA; use `pandas`/`numpy`/`statsmodels` for cointegration and `cvxpy`/`PyPortfolioOpt` for replication optimization.

## Caveats

Fees, tickers and tax rules change. Treat the figures here as 2025–2026 reference points and verify each instrument against its current **prospectus/regulamento**, the **B3 listed-ETF pages**, and **Receita Federal/CVM** before transacting or modeling. Past index performance does not guarantee future returns.

**Sources:** [B3 listed equity ETFs](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-variavel/etfs-listados/) · [B3 listed fixed-income ETFs](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/etf/renda-fixa/etfs-listados/) · [B3 trading hours](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/horario-de-negociacao/acoes/) · [B3 ETF/BDR taxation](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/etfs/tributacao-etf-e-bdr/) · [CVM – Fundos de Índice (ETF), Resolução CVM 175 Anexo V](https://www.gov.br/cvm/pt-br/assuntos/noticias/2025/area-tecnica-da-cvm-esclarece-duvidas-sobre-fundos-de-indice-etf) · [BlackRock SMAL11](https://www.blackrock.com/br/products/251752/ishares-bmfbovespa-small-cap-fundo-de-ndice-fund) · [BlackRock BOVA11](https://www.blackrock.com/br/products/251816/ishares-ibovespa-fundo-de-ndice-fund) · [BlackRock IVVB11](https://www.blackrock.com/br/products/251902/ishares-sp-500-fi-em-cotas-de-fundo-de-ndice-inv-no-exterior-fund) · [Itaú PIBB11 factsheet](https://assetfront.arquivosparceiros.cloud.itau.com.br/FND/ITAUPIBBIBX843_It_Now_PIBB_IBrX-50.pdf) · [XP GOLD11](https://www.xpasset.com.br/fundos/gold11/) · [XP NASD11](https://www.xpasset.com.br/fundos/nasd11/) · [BB Asset FIXA11](https://bb.com.br/site/bb-asset-management/fundos-de-investimento/etf/b3-futuros-de-taxas-de-juros-fundo-de-indice/) · [QR Asset QBTC11](https://qrasset.com.br/qbtc11/) · [Hashdex HASH11](https://hashdex.com/pt-BR/products/HASH11) · [InvestNews crypto ETFs](https://investnews.com.br/financas/etfs-de-criptomoedas-lista-completa-b3/) · [brapi.dev docs](https://brapi.dev/docs/acoes) · [B3 for Developers](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/b3-for-developers/)

**Keywords:** ETF B3, Brazilian ETF, fundo de índice, Resolução CVM 175, BOVA11, IVVB11, SMAL11, BOVV11, PIBB11, IBrX-50, DIVO11, FIND11, GOLD11, NASD11, Nasdaq-100, HASH11, QBTC11, QETH11, IMAB11, IB5M11, B5P211, IRFM11, FIXA11, renda variável, renda fixa, tributação ETF, taxa de administração, isenção R$20 mil, iShares, Itaú It Now, XP Trend, BB Asset, Hashdex, índice de referência, replicação de índice, tracking error, criação e resgate, iNAV, smart beta, brapi, yfinance, quant Brasil, machine learning finanças.
