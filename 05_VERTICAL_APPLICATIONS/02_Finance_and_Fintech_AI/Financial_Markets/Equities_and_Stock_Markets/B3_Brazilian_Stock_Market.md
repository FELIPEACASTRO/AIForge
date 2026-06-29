# B3 — Brazilian Stock Market (Ações na Bolsa Brasileira)

> Reference for cash equities (ações) on **B3 S.A. – Brasil, Bolsa, Balcão**, the sole securities exchange in Brazil: instruments, tickers, indices, trading mechanics, taxation, regulation, and the ML/quant + data stack for Brazilian equities.

## What B3 is

B3 (Brasil, Bolsa, Balcão) is Brazil's vertically integrated financial-market infrastructure — it runs the country's stock, derivatives, and OTC markets and provides trading, clearing, settlement, central depository, and registry services. It is itself listed on its own exchange under the ticker **B3SA3** ([Google Finance](https://www.google.com/finance/quote/B3SA3:BVMF?hl=en)).

B3 was formed by the merger of **BM&FBOVESPA** with **Cetip S.A. – Mercados Organizados**, consummated on **29 March 2017**; the corporate name was changed to *B3 S.A. – Brasil, Bolsa, Balcão* later in 2017 ([B3 History (RI)](https://ri.b3.com.br/en/b3/history/)). The lineage: **Bovespa** (Bolsa de Valores de São Paulo, founded **23 August 1890**) and **BM&F** (the mercantile & futures exchange) combined on **8 May 2008** to create BM&FBOVESPA; the 2017 Cetip deal added OTC, depository, registry and financing infrastructure (e.g., auto/real-estate liens), turning the group into a full market-infrastructure company and, at the time, the fifth-largest exchange in the world by market value ([Wikipedia: B3](https://en.wikipedia.org/wiki/B3_(stock_exchange))).

## Corporate governance / listing segments (segmentos de listagem)

B3 maintains special listing tiers with progressively stricter governance, created to lower investors' risk perception (reduce information asymmetry) and improve liquidity/valuation ([B3 — Segmentos de listagem](https://www.b3.com.br/pt_br/produtos-e-servicos/solucoes-para-emissores/segmentos-de-listagem/)).

| Segment | Share types allowed | Min. free float (general) | Tag along | Board |
|---|---|---|---|---|
| **Novo Mercado** | Only ON (voting) | 25% (or 15% if avg. daily traded volume ≥ R$ 25 mn over 12 mo.) | 100% of ON | ≥ 3 members; ≥ 2 or 20% independent |
| **Nível 2** | ON + PN (PN with extra voting rights in key matters) | ~25% | 100% (ON & PN) | enhanced |
| **Nível 1** | ON + PN | ≥ 25% free float | legal minimum (80% ON) | calendar/disclosure duties |
| **Bovespa Mais** | ON | listing without offer; up to 7 yrs to IPO | 100% | growth/SME focus |
| **Bovespa Mais Nível 2** | ON + PN | similar, with PN | 100% | growth/SME |

Sources: [B3 — Novo Mercado](https://www.b3.com.br/pt_br/produtos-e-servicos/solucoes-para-emissores/segmentos-de-listagem/novo-mercado/), [Suno — regras Novo Mercado](https://www.suno.com.br/noticias/entenda-regras-novo-mercado-segmentos-b3/). The special-segment regulations were revised in **2023**, focusing on liquidity parameters (free float, offering and daily trading volume): the alternative liquidity rule was standardized to **15% free float when average daily traded volume is ≥ R$ 20 mn over the prior 12 months**, with provisions allowing certain issuers to reduce the floor toward **20%** ([Pinheiro Guimarães — novas regras de free float](https://www.pinheiroguimaraes.com.br/novo-mercado-nivel-2-e-nivel-1-terao-novos-criterios-de-liquidez-free-float/)). Confirm the parameters in force against the current regulation.

## Share types and ticker conventions

Brazilian listed equity comes in two main classes plus bundled units:

- **ON (ordinárias / ordinary)** — full voting rights. Ticker ends in **3** (e.g., **VALE3**, **PETR3**, **WEGE3**).
- **PN (preferenciais / preferred)** — priority in dividends, generally no/limited vote. Ticker ends in **4** (e.g., **PETR4**, **ITUB4**, **BBDC4**). Subclasses use 5/6/7/8 (PNA/PNB...).
- **UNIT** — a bundle of more than one security class (often ON + PN), carrying both dividend and voting characteristics. Ticker ends in **11** (e.g., **BPAC11**, **SANB11**). Note: ETFs and some FIIs/BDRs also end in 11.

Sources: [B3 — Bora Investir: ON vs PN](https://borainvestir.b3.com.br/tipos-de-investimentos/entenda-a-diferenca-entre-as-acoes-on-e-pn/), [Investidor10 — tickers](https://investidor10.com.br/conteudo/tickers-entenda-o-que-sao-e-como-interpretar-114807/).

### Major names (tickers)

| Company | ON | PN / Unit | Sector |
|---|---|---|---|
| Petrobras | PETR3 | PETR4 (PN) | Oil & gas |
| Vale | VALE3 | — | Mining |
| Itaú Unibanco | ITUB3 | ITUB4 (PN) | Banking |
| Bradesco | BBDC3 | BBDC4 (PN) | Banking |
| Ambev | ABEV3 | — | Beverages |
| WEG | WEGE3 | — | Industrials |
| BTG Pactual | — | BPAC11 (Unit) | Banking |
| Banco do Brasil | BBAS3 | — | Banking |
| B3 (the exchange) | B3SA3 | — | Exchange/FMI |
| Santander Brasil | — | SANB11 (Unit) | Banking |

At the close of **2025**, **Itaú (ITUB4)** overtook **Petrobras** as the most valuable company on B3 (≈ R$ 443 bn vs. ≈ R$ 433 bn on 4 Dec 2025); **BTG Pactual (BPAC11)** climbed to **#3** (≈ R$ 323 bn), ahead of **Vale (VALE3)** at **#4** (≈ R$ 307 bn) ([InfoMoney](https://www.infomoney.com.br/mercados/itau-supera-petrobras-em-valor-de-mercado-e-se-torna-empresa-mais-valiosa-da-b3/), [Investidor10 — notícia](https://investidor10.com.br/noticias/itau-itub4-supera-petrobras-petr4-e-torna-se-a-empresa-mais-valiosa-da-b3-116488/)). Daily market-cap rankings are published by B3 ([B3 — valor de mercado das empresas listadas](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-a-vista/valor-de-mercado-das-empresas-listadas/bolsa-de-valores-diario/)).

## Round lots and the fractional market

- **Standard lot (lote padrão)** = **100 shares** for most equities ([Modalmais](https://ajuda.modalmais.com.br/hc/pt-br/articles/13654333371667-Lote-padr%C3%A3o-dos-principais-ativos)).
- **Fractional market (mercado fracionário)** lets you trade **1–99 shares**; access it by appending **F** to the ticker — e.g., **VALE3F**, **PETR4F** ([B3 — Bora Investir: ação fracionada](https://borainvestir.b3.com.br/glossario/acao-fracionada/), [InfoMoney — mercado fracionário](https://www.infomoney.com.br/guias/mercado-fracionario-de-acoes/)). The fractional book typically has wider spreads and thinner liquidity than the lote-padrão book.

## Trading hours and auctions (horário de pregão)

Per B3's official equities schedule (Brasília time / BRT, "mercado à vista e fracionário") ([B3 — horário de negociação: ações](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/horario-de-negociacao/acoes/)):

| Phase | Time (BRT) |
|---|---|
| Order-cancellation window (cancelamento de ofertas) | 09:30–09:45 |
| Pre-opening (pré-abertura) | 09:45–10:00 |
| Continuous trading (negociação / pregão regular) | 10:00–16:55 |
| Closing call (call / leilão de fechamento) | 16:55–17:00 |
| After-market (negociação) | 17:25–17:30 |
| After-market order-cancellation window | 17:30–18:00 |

After-market rules: trading is restricted to assets that (i) belong to a main theoretical portfolio — **Ibovespa, IBrX-50, IBrX-100 or IFIX**, (ii) were actually traded in the regular session that day, and (iii) are cash-market; prices may not move more than **±2%** from the regular-session close, with a per-CPF cap. Schedules shift with daylight saving and B3 calendar changes, so always confirm the current page before automating. Matching runs on B3's **PUMA Trading System** engine.

## Main indices (índices)

The benchmark is the **Ibovespa (IBOV)**. B3 publishes roughly 70 indices ([XP — além do Ibovespa](https://conteudos.xpi.com.br/aprenda-a-investir/relatorios/alem-do-ibovespa-bolsa-brasileira-indices/)).

| Index | Ticker | What it tracks |
|---|---|---|
| Ibovespa | IBOV | Most-traded, highest-negotiability shares/units; free-float market-cap weighted, single-asset cap (≈20%) |
| Índice Brasil 100 | IBrX-100 | Top 100 assets by negotiability index ([B3 — IBrX 100](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-brasil-100-ibrx-100.htm)) |
| Índice Brasil 50 | IBrX-50 | Top 50 most-negotiable, most-representative assets |
| Small Cap | SMLL | Smaller-capitalization companies (small caps) |
| Dividend Index | IDIV | High dividend / JCP (juros sobre capital próprio) payers |
| Consumer Index | ICON | Consumption-sector companies |

**Ibovespa methodology:** free-float-adjusted market-value weighting, theoretical portfolio (carteira teórica) **rebalanced every four months (quadrimestral)** — taking effect in **January, May and September** (covering Jan–Apr, May–Aug, Sep–Dec). B3 publishes **three previews (prévias)** before each definitive portfolio. Inclusion broadly requires presence in ≥95% of trading sessions, a minimum share of traded volume, and ranking within the top negotiability tier; a single asset is capped at around **20%** ([B3 — metodologia Ibovespa PDF](https://www.b3.com.br/data/files/9C/15/76/F6/3F6947102255C247AC094EA8/IBOV-Metodologia-pt-br__Novo_.pdf)). Confirm exact thresholds in the current methodology PDF, as parameters are periodically updated.

## BDRs — Brazilian Depositary Receipts

BDRs are locally traded certificates backed by shares of foreign companies, held blocked abroad by a custodian; they let Brazilian investors gain exposure to names like Apple, Amazon or Microsoft without sending money overseas. They are governed by CVM regulation, the B3 Issuers Regulation, and the B3 Manual ([B3 — BDRs](https://www.b3.com.br/pt_br/produtos-e-servicos/solucoes-para-emissores/bdrs-brazilian-depositary-receipts/), [B3 — Bora Investir: BDR](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/bdrs/bdr-o-que-e-e-como-funciona/)).

| BDR type | Foreign issuer registered at CVM? | Ticker suffix |
|---|---|---|
| Não Patrocinado Nível I (depositary-bank initiative, most common) | No — issuer not registered at CVM/B3 | **34 / 35** |
| Patrocinado Nível I | No (issuer-sponsored, limited disclosure) | **31** |
| Patrocinado Nível II (Category A disclosure, exchange-traded) | Yes | **32** |
| Patrocinado Nível III (Nível II + public distribution) | Yes | **33** |

Most BDRs on B3 are *não patrocinados Nível I*, where the foreign issuer is not registered with the CVM/B3 and the Brazilian depositary bank is responsible for tracking and disclosing corporate/financial information ([B3 — BDRs Não Patrocinados Nível I](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/brazilian-depositary-receipts-bdrs-nao-patrocinados-nivel-i.htm)). BDRs of foreign **ETFs** also trade on B3.

## Settlement, brokers and access

- **Settlement (liquidação):** cash equities currently settle at **D+2** via B3's integrated clearinghouse / central depository (CSD); short selling uses B3's securities-lending facility (aluguel de ações / BTB). B3 has announced a project to migrate the equities cycle to **D+1 in February 2028** ([B3 — ciclo D+2](https://clientes.b3.com.br/en/w/ciclo-de-liquidacao-d-2), [Bora Investir — projeto D+1](https://borainvestir.b3.com.br/noticias/mercado/b3-anuncia-projeto-para-reducao-do-ciclo-de-liquidacao-de-acoes-para-d1/)).
- **Brokers / home brokers (corretoras):** retail access via XP, Clear, Rico, BTG Pactual, NuInvest, Inter, C6, Itaú, Bradesco, Ágora, Modalmais, among others. Orders route through the broker's home broker / API into B3's PUMA book.

## Taxation of stocks in Brazil (tributação)

For **individuals (pessoa física)** on cash equities ([B3 — Bora Investir: IR renda variável](https://borainvestir.b3.com.br/noticias/imposto-de-renda/renda-variavel-imposto-de-renda/comprou-ou-vendeu-acoes-veja-como-declarar-swing-trade-day-trade-e-proventos-no-imposto-de-renda/), [XP — day trade no IR](https://conteudos.xpi.com.br/aprenda-a-investir/relatorios/day-trade-no-imposto-de-renda/)):

| Operation | Rate on net gain | Exemption | IRRF ("dedo-duro") |
|---|---|---|---|
| Common / swing trade (operações normais) | **15%** | Gains **exempt if total monthly sales ≤ R$ 20,000** | 0.005% withheld on sell value |
| Day trade | **20%** | No exemption | 1% withheld on day-trade profit |

Tax is self-assessed **monthly** and paid by **DARF** by the last business day of the following month; capital losses can offset future gains of the same type (swing-trade losses cannot offset day-trade gains, and vice-versa). IRRF withheld can be deducted from the DARF due. **Reform note:** a provisional measure proposed unifying variable- and fixed-income taxation at a flat **17.5%** effective January 2026 — verify the current status with the Receita Federal / your broker before filing ([InfoMoney — proposta de alíquotas](https://www.infomoney.com.br/mercados/day-trade-pode-ter-queda-de-20-para-175-em-aliquota-mas-swing-trade-deve-ter-alta/)).

## Regulation (CVM)

The market regulator is the **Comissão de Valores Mobiliários (CVM)**, an *autarquia em regime especial* linked to the **Ministério da Fazenda** (Ministry of Finance), created by **Lei nº 6.385 of 7 December 1976** (the "CVM Law") to supervise, regulate, discipline and develop the securities market ([CVM — Lei 6.385/76](https://conteudo.cvm.gov.br/legislacao/leis-decretos/lei6385.html), [CVM — Sobre a CVM](https://www.gov.br/cvm/pt-br/acesso-a-informacao-cvm/institucional/sobre-a-cvm)). The CVM oversees issuance/distribution, trading and intermediation, exchanges, portfolio management, custody, and the audit of public companies, and can impose warnings, fines, suspensions and registration cancellation. The **Banco Central do Brasil** and CADE also reviewed the B3/Cetip combination.

## ML / quant angle for Brazilian equities

Brazilian equities present distinctive modeling challenges: high macro sensitivity (Selic rate, BRL/USD, commodities), a commodity-heavy index (Petrobras/Vale historically dominate IBOV), and lower per-name liquidity than US large caps — making **microstructure, liquidity filters, and survivorship/free-float adjustments** essential.

- **Direction / return forecasting:** LSTM and Bayesian neural nets have been applied to the Ibovespa; reported daily-direction accuracy ranges widely (≈55% for a baseline LSTM up to ≈71–78% mean for Bayesian NN in specific studies) — treat such figures cautiously and strictly out-of-sample ([Springer — deep learning Brazilian market](https://link.springer.com/article/10.1007/s10614-024-10636-y), [ResearchGate — ML time-series at B3](https://www.researchgate.net/publication/376684445_Machine_Learning-Based_Time_Series_Prediction_at_Brazilian_Stocks_Exchange)).
- **Multimodal / sentiment:** combining prices + technical indicators + Portuguese-language financial news (and LLM-derived sentiment) has been reported to improve out-of-sample accuracy and ROI vs. price-only models ([Springer — sentiment + ChatGPT indices](https://link.springer.com/article/10.1007/s10614-024-10835-7)).
- **Tree ensembles:** Gradient Boosting and kNN have outperformed ARIMA on Ibovespa-constituent return forecasting in error and hit rate in several studies.
- **Quant pipelines:** factor models (value/quality/momentum) on the IBrX/SMLL universe, pairs trading on co-integrated banks/utilities, and execution models accounting for the fractional book.

## Data & APIs

| Tool / endpoint | What it provides | Notes |
|---|---|---|
| **brapi.dev** | REST API: quotes, history, dividends, fundamentals for ações, FIIs, ETFs, BDRs, FX, crypto, macro | Free tier; official Python SDK `pip install brapi` (Python 3.8+, sync/async) ([brapi.dev](https://brapi.dev/), [Python SDK docs](https://brapi.dev/docs/sdks/python)) |
| **yfinance** | Yahoo Finance OHLCV; append **`.SA`** suffix for B3 tickers (e.g., `PETR4.SA`, `VALE3.SA`) | Free; lower reliability/SLA for BR names |
| **MetaTrader 5 + Python** | Intraday/tick B3 data via a broker's MT5 feed; good for quant backtesting | Practical free route for granular data ([Asimov Academy](https://hub.asimov.academy/blog/como-baixar-dados-da-b3-com-metatrader-5-e-python/)) |
| **B3 UP2DATA / UP2DATA On Demand** | Official end-of-day + reference data (equities, derivatives, indices, corporate actions); TXT/CSV/XML/JSON | Server/cloud delivery ([B3 — UP2DATA](https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/), [On Demand store](https://www.up2dataondemand.com/)) |
| **B3 for Developers / Market Data** | Real-time market-data feeds and developer APIs | Professional/commercial ([B3 for Developers](https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/b3-for-developers/)) |
| **Economatica** | Commercial fundamentals/quant database widely used by BR asset managers | Paid |
| **Status Invest** | Retail fundamentals, dividends, index composition (e.g., IBOV constituents) | Free web ([statusinvest.com.br](https://statusinvest.com.br/indices/ibovespa)) |

**Quick start:** `yfinance` for prototyping, **brapi.dev** for clean BR fundamentals/dividends, **MT5** for intraday backtests, **UP2DATA** for production-grade official data and corporate actions.

## Sources

- B3 — History (RI): https://ri.b3.com.br/en/b3/history/
- B3 — Listing segments: https://www.b3.com.br/pt_br/produtos-e-servicos/solucoes-para-emissores/segmentos-de-listagem/
- B3 — Novo Mercado: https://www.b3.com.br/pt_br/produtos-e-servicos/solucoes-para-emissores/segmentos-de-listagem/novo-mercado/
- B3 — Trading hours (equities): https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/horario-de-negociacao/acoes/
- B3 — IBrX 100: https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-brasil-100-ibrx-100.htm
- B3 — Ibovespa methodology (PDF): https://www.b3.com.br/data/files/9C/15/76/F6/3F6947102255C247AC094EA8/IBOV-Metodologia-pt-br__Novo_.pdf
- B3 — BDRs: https://www.b3.com.br/pt_br/produtos-e-servicos/solucoes-para-emissores/bdrs-brazilian-depositary-receipts/
- B3 — BDRs Não Patrocinados Nível I: https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/brazilian-depositary-receipts-bdrs-nao-patrocinados-nivel-i.htm
- B3 — Settlement cycle D+2: https://clientes.b3.com.br/en/w/ciclo-de-liquidacao-d-2
- B3 — UP2DATA: https://www.b3.com.br/en_us/market-data-and-indices/data-services/up2data/
- CVM — Lei 6.385/76: https://conteudo.cvm.gov.br/legislacao/leis-decretos/lei6385.html
- CVM — Sobre a CVM: https://www.gov.br/cvm/pt-br/acesso-a-informacao-cvm/institucional/sobre-a-cvm
- brapi.dev: https://brapi.dev/ — Python SDK: https://brapi.dev/docs/sdks/python

**Keywords:** B3, Brasil Bolsa Balcão, BM&FBovespa, Ibovespa, IBOV, IBrX-100, IBrX-50, Small Cap, IDIV, ICON, Novo Mercado, Nível 1, Nível 2, Bovespa Mais, ações ON, ações PN, units, PETR4, VALE3, ITUB4, BPAC11, BDR, Brazilian Depositary Receipts, mercado fracionário, lote padrão, liquidação D+2, D+1, home broker, XP, BTG, CVM, imposto de renda ações, day trade, DARF, dedo-duro, brapi.dev, yfinance, MetaTrader5, UP2DATA, Economatica, Status Invest, machine learning, quant, renda variável, bolsa de valores, pregão, leilão de fechamento, free float, tag along, dividendos, JCP
