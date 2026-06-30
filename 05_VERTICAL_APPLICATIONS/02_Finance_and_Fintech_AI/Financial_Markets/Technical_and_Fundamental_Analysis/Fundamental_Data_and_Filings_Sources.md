# Fundamental Data & Filings Sources

> Where to get company financials, regulatory filings, and earnings transcripts for ML — free government APIs (SEC EDGAR, CVM, EDINET, Companies House), point-in-time vendors (Sharadar, Compustat), AI-era model-ready feeds (Daloopa, Quartr, Fiscal.ai), and retail/research platforms — with honest notes on licensing, point-in-time correctness, and look-ahead/survivorship bias.

This page is about *fundamentals & disclosures* (income statement / balance sheet / cash flow, ratios, 10-K/10-Q/8-K, transcripts), complementary to generic market-data APIs already cataloged in [`../../Datasets_APIs_and_Data_Vendors/`](../Datasets_APIs_and_Data_Vendors/). For NLP on the *text* of these filings see [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/); for the analysis methods see the [parent README](./README.md).

**The one bias that wrecks fundamental backtests:** restated financials. Most free APIs serve the *latest restated* numbers, so a 2015 backtest "sees" a 2018 restatement → look-ahead bias. Only **point-in-time (PIT)** datasets (the value *as originally reported* at each historical date) are safe for serious cross-sectional research. The PIT column below is the most important one in every table.

---

## 1. Free government / regulator primary sources (the ground truth)

These are the authoritative filings themselves — no vendor in between. Free, but you parse XBRL/HTML yourself.

| Source | Content | Coverage | PIT? | API / how | Link |
|---|---|---|---|---|---|
| **SEC EDGAR — Submissions API** | All filing metadata (10-K/10-Q/8-K/13F/Form 4…) per company | US issuers, full history | Yes (as-filed) | `https://data.sec.gov/submissions/CIK##########.json` | https://www.sec.gov/search-filings/edgar-application-programming-interfaces |
| **SEC EDGAR — Company Facts** | Every XBRL fact a company ever filed, one JSON | US, since XBRL (~2009) | Yes | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | same |
| **SEC EDGAR — Company Concept** | One concept's full history (e.g. `Revenues`) | US | Yes | `…/api/xbrl/companyconcept/CIK…/us-gaap/Revenues.json` | same |
| **SEC EDGAR — XBRL Frames** | One fact across *all* filers for a period (cross-section) | US | Yes | `…/api/xbrl/frames/us-gaap/Assets/USD/CY2024Q1I.json` | same |
| **SEC EDGAR — Full-Text Search** | Search filing *text* since 2001 (keywords, boolean, CIK/ticker) | US | n/a | `https://efts.sec.gov/LATEST/search-index?q=…` / UI at sec.gov/edgar/search | https://www.sec.gov/edgar/search/ |
| **SEC Financial Statement Data Sets** | Quarterly ZIPs: `SUB`/`TAG`/`NUM`/`PRE` CSVs of all primary-statement XBRL facts | US, 2009Q1– | Yes (as-filed snapshots) | DERA bulk ZIP download | https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets |
| **SEC bulk `companyfacts.zip` / `submissions.zip`** | Entire Company Facts & submissions corpus in one download | US | Yes | `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` | https://www.sec.gov/about/developer-resources |

Rate limit: **10 req/s per IP**, no key, but you **must** send a descriptive `User-Agent` (e.g. `name email`) or you get 403. All free.

**International primary sources:**

| Source | Country | Content | Free | API / bulk | Link |
|---|---|---|---|---|---|
| **CVM Dados Abertos** 🇧🇷 | Brazil | ITR/DFP/FRE/FCA, capital composition, fund informes | Yes | CKAN datasets, yearly CSV/ZIP | https://dados.cvm.gov.br/ |
| **RAD CVM** 🇧🇷 | Brazil | Consulta de documentos de companhias abertas (filings portal) | Yes | Web (scrape) | https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx |
| **Companies House API** | UK | Company profile, filing history, officers, PSC | Yes | REST + key | https://developer.company-information.service.gov.uk/ |
| **Companies House — Free Accounts Data Product** | UK | XBRL iXBRL accounts (instance docs), daily + monthly | Yes | Bulk ZIP | https://download.companieshouse.gov.uk/en_accountsdata.html |
| **Companies House — Free Company Data Product** | UK | Basic company data snapshot (all live companies), CSV | Yes | Bulk ZIP | https://download.companieshouse.gov.uk/en_output.html |
| **SEDAR+** | Canada | All TSX/TSXV issuer & fund filings (replaced SEDAR/SEDI) | Yes | Web search (no open API) | https://www.sedarplus.ca/ |
| **EDINET** | Japan | Statutory disclosure (yūho), XBRL | Yes | REST API v2 (`api.edinet-fsa.go.jp/api/v2/`) + bulk; portal UI | https://disclosure2.edinet-fsa.go.jp/ |
| **JPX TDnet** | Japan | Timely disclosure (earnings, summaries), JP+EN since Apr 2025 for Prime | Paid feed | TDnet API Service (subscription); also via Snowflake (JPXI) | https://www.jpx.co.jp/english/markets/paid-info-listing/tdnet/ |
| **ESMA Registers / FIRDS** | EU | Reference data for MiFID II instruments (ISIN-level) | Yes | Bulk download + UI | https://registers.esma.europa.eu/ |

> Honest notes: SEDAR+ has **no official open API** — third parties (QuoteMedia, Thomson Reuters) resell. CVM data is XBRL/CSV with idiosyncratic encoding (Latin-1, account codes per CPC standard) — not turnkey. EDINET/TDnet are Japanese-first; English coverage is partial and recent.

---

## 2. Point-in-time / institutional fundamentals (research-grade)

The serious end. PIT correctness + survivorship-bias-free universes.

| Source | Content | Coverage | PIT? | Free/Paid | API / how | Link |
|---|---|---|---|---|---|---|
| **Sharadar SF1** (via Nasdaq Data Link) | US fundamentals from 10-K/10-Q, ~150 indicators, ARQ/ARY/MRQ dims | 16k+ US tickers, active+delisted, ~28yr | **Yes** (as-reported dimensions) | Paid (academic discount) | Nasdaq DL / Quandl API; also QuantRocket | https://data.nasdaq.com/databases/SF1 |
| **Sharadar SEP / SFP** | Daily prices (split/div adjusted), incl. delisted | US, since 2014 | n/a | Paid | Nasdaq DL API | https://data.nasdaq.com/databases/SEP |
| **Sharadar SF3 / SF2 / EVENTS / DAILY** | Institutional holdings (13F), insiders (Form 3/4/5), 8-K events, daily metrics | US | Yes | Paid | Nasdaq DL API | https://www.sharadar.com/ |
| **Compustat** (S&P, via WRDS) | Standardized global fundamentals, 50+ yr history | North America + Global | **Yes** (PIT/Snapshot products; PIT since 1987) | Academic (WRDS) / commercial | WRDS Python/SAS/web; cloud | https://wrds-www.wharton.upenn.edu/ |
| **CRSP** (via WRDS) | Returns, delisting returns, share data (pairs w/ Compustat = CCM) | US, since 1925 | Survivorship-free | Academic | WRDS | https://wrds-www.wharton.upenn.edu/ |
| **Refinitiv / LSEG Fundamentals** | Standardized + as-reported statements, estimates (I/B/E/S) | Global | PIT products available | Commercial | Eikon/Workspace, DSWS, RDP API | https://www.lseg.com/en/data-analytics/financial-data/company-data/fundamentals-data |
| **FactSet Fundamentals** | Global standardized financials, filings, estimates | Global | PIT add-on | Commercial | FactSet API / feeds | https://www.factset.com/ |

> Sharadar is the sweet spot for independent quants who want PIT US fundamentals without a Bloomberg/WRDS budget. Compustat+CRSP via WRDS remains the academic gold standard (the basis of most published factor papers). Note: some universities have **dropped** Compustat PIT/Snapshot (e.g. Dartmouth's access ended 2025-07-31) — confirm your institution's current entitlements.

---

## 3. AI-era "model-ready" feeds & transcripts (2024-2026)

The newest category: data normalized and source-linked specifically for LLM agents / financial-model building. Several now ship **MCP servers** for direct LLM use.

| Source | Content | Coverage | PIT? | Free/Paid | API / how | Link |
|---|---|---|---|---|---|---|
| **Daloopa** | 1,300+ standardized, source-linked metrics from filings/press/IR decks, >99% accuracy; 37 endpoints | 5,500+ global tickers | As-reported, source-linked | Free account; API/Plus sales-quoted | REST API + MCP | https://daloopa.com/products/api |
| **Quartr** | Earnings transcripts (speaker ID + timestamps; archived WER <2%, live WER <5%), audio (HLS/MPEG), slides, filings | Global, live | n/a | Paid (sales-quoted; MCP) | REST API, JSONL, MCP | https://quartr.com/products/quartr-api |
| **Fiscal.ai** (ex-FinChat; merged Stratosphere.io) | Verified financials, KPIs, segments, transcripts, prices | Global | As-reported | Paid (free API trial: 25 cos / 250 calls/day) | REST API + MCP (Claude/ChatGPT) | https://fiscal.ai/ |
| **financialdatasets.ai** | SEC-primary statements, prices, news, insider/institutional — clean REST | US (27k+ active+delisted, 30yr) | As-reported | Paid (free 100 req/day) | REST API + MCP (OAuth) | https://docs.financialdatasets.ai/ |

> These feeds explicitly target RAG / agent pipelines: "source-linked" means each number carries a citation back to the filing line, which is what you want for grounded LLM finance ([see Agent design](../Frontier_AI_in_Finance/)). Daloopa/Quartr are B2B-priced; Fiscal.ai and financialdatasets.ai have usable free trials.

---

## 4. Mid-market & retail fundamentals APIs (good free tiers)

Where individuals and small teams actually start. All have caveats on PIT (most serve restated numbers).

| Source | Content | Coverage | PIT? | Free tier | API | Link |
|---|---|---|---|---|---|---|
| **SimFin** | Standardized statements + ratios, bulk CSV | Global, 10+ yr | No (restated) | **Yes** (12-mo delayed bulk, free key) | Python `simfin`, REST, Excel | https://www.simfin.com/ |
| **Finnhub** | Basic financials (117 metrics), financials-as-reported, statements 30yr | Global, 30k+ co | "as-reported" endpoint = paid | Yes (60 calls/min) | REST, Python | https://finnhub.io/docs/api/company-basic-financials |
| **Financial Modeling Prep (FMP)** | Statements, 70+ ratios, DCF, segments | 70k+ securities, 46 countries, 30yr (paid) | No (restated) | Yes (250 req/day, US only, 5yr) | REST, Python | https://site.financialmodelingprep.com/ |
| **Tiingo Fundamentals** | Statements + daily metrics, `asReported` flag | US, 20+ yr | `asReported=true` option | Limited (paid for full) | REST, Python `tiingo` | https://www.tiingo.com/documentation/fundamentals |
| **Roic.ai** | 30+ yr statements, 90+ ROIC-focused ratios, transcripts | 70+ exchanges incl. B3 🇧🇷 | No (restated) | **Yes** (no card) | REST, RQL, MCP | https://www.roic.ai/api |
| **EODHD** | Fundamentals (BS/IS/CF), ratios, insider (Form 4), institutional, ESG | 60+ exchanges, 150k+ tickers | No (restated) | Yes (20 calls/day, US; fundamentals = 10 calls each) | REST, Python/R, MCP | https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds |
| **Alpha Vantage** | `OVERVIEW`, income/balance/cashflow, earnings | Global | No | Yes (25 req/day) | REST | https://www.alphavantage.co/ |

---

## 5. Research / analyst platforms (UI-first, some export)

Not primarily APIs — but excellent for manual deep-dives, model templates, and filing comparison.

| Source | What it's for | Coverage | Free/Paid | Link |
|---|---|---|---|---|
| **TIKR** | Screener + 100k global stocks, estimates, transcripts | 92 countries, 136 exchanges | Free (US, 5yr) / Plus ~$15/mo | https://www.tikr.com/ |
| **BamSEC** (by Tegus) | Fast SEC filings + transcripts reader, table-to-Excel | US | Free browse / paid search & tools | https://www.bamsec.com/ |
| **Wisesheets** | Spreadsheet add-on: 50k+ equities, 50+ exchanges, API-ready DB | Global | Paid (trial) | https://www.wisesheets.io/ |
| **Stockpup** ⚠️ | Historical free CSV fundamentals (S&P 100/500, 15yr quarterly) | US | Free — **but site defunct**; mirrors on GitHub/Kaggle | (archive only) |

> ⚠️ **Stockpup** is the classic "free fundamentals CSV" everyone references — the original site is **dead**. Use only archived mirrors and treat the data as stale (not maintained). Don't build a live pipeline on it.

---

## 6. Brazil 🇧🇷 fundamentals — practical stack

Brazil-specific guidance, since the audience is BR-heavy. The primary source is **CVM** (above); these wrap it for convenience.

| Source | Content | PIT? | Free/Paid | API / how | Link |
|---|---|---|---|---|---|
| **brapi.dev** | Quotes + fundamentals (BS/IS/CF/DVA from CVM), dividends, FIIs, BDRs | No (latest) | Free tier + paid | REST API | https://brapi.dev/ |
| **Fundamentus** | Web fundamentals/ratios for B3 stocks & FIIs | No | Free (web) | Scrape / unofficial libs | https://www.fundamentus.com.br/ |
| **Status Invest** | Rich fundamentals, dividends, screeners (BR + intl) | No | Free (web) | Scrape (no official API) | https://statusinvest.com.br/ |
| **CVM Dados Abertos** | ITR/DFP raw XBRL/CSV (ground truth) | **Yes** (as-filed) | Free | Bulk CSV/ZIP | https://dados.cvm.gov.br/ |
| **Roic.ai** | Normalized B3 financials in USD (e.g. `VALE3.SA`) | No | Free tier | REST | https://www.roic.ai/api |

> BR honesty: brapi/Fundamentus/Status Invest serve **latest** figures — no point-in-time. For PIT BR research you must reconstruct from CVM ITR/DFP filings by their *original* filing date (`DT_RECEB`). BR accounting follows CPC/IFRS; account codes (`CD_CONTA`) differ from US-GAAP, so US-trained pipelines need remapping. Watch tickers changing (e.g. units `11`, ON/PN), spin-offs, and Ibovespa rebalancing for survivorship.

---

## 7. Open-source parsing libraries (free, do-it-yourself)

To turn raw EDGAR/CVM filings into DataFrames without paying a vendor:

| Library | What it does | Link |
|---|---|---|
| **edgartools** (`edgartools`) | Read/parse EDGAR: 10-K/8-K, XBRL financials → DataFrames, Form 3/4/5, 13F. MIT, no key. | https://github.com/dgunning/edgartools |
| **sec-edgar-downloader** | Bulk-download raw filings by ticker/CIK (you parse) | https://pypi.org/project/sec-edgar-downloader/ |
| **secfsdstools** | Work with SEC Financial Statement Data Sets locally | https://pypi.org/project/secfsdstools/ |
| **sec-cik-mapper** | Ticker ↔ CIK ↔ company-name mapping | https://pypi.org/project/sec-cik-mapper/ |
| **OpenBB Platform** | Aggregates many of the above (FMP, Tiingo, Intrinio, SEC) behind one Python API | https://openbb.co/ |

---

## Choosing — quick guidance

- **Serious factor research / backtests** → Sharadar SF1+SEP (independent) or Compustat+CRSP via WRDS (academic). Insist on **PIT**.
- **LLM / agent pipeline** → Daloopa, Quartr, Fiscal.ai, or financialdatasets.ai (source-linked, MCP-ready) over raw EDGAR.
- **Free & DIY** → SEC EDGAR APIs + `edgartools`; SimFin/FMP/Roic.ai free tiers for quick prototypes.
- **Brazil 🇧🇷** → CVM Dados Abertos for ground truth + PIT; brapi/Status Invest for fast latest-value lookups.
- **Always validate**: survivorship (delisted names included?), point-in-time (restated vs as-filed?), look-ahead (filing date vs period-end date — use *filing/acceptance* date, never period-end, as the data-availability timestamp).

## Related in AIForge
- [Technical & Fundamental Analysis (parent)](./README.md) · [Datasets, APIs & Data Vendors](../Datasets_APIs_and_Data_Vendors/) · [Brazil & B3 Market Data APIs](../Datasets_APIs_and_Data_Vendors/Brazil_B3_Market_Data_APIs.md)
- [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/) · [Equities & Stock Markets](../Equities_and_Stock_Markets/) · [Frontier AI in Finance](../Frontier_AI_in_Finance/)

Sources: https://www.sec.gov/search-filings/edgar-application-programming-interfaces · https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets · https://www.sec.gov/about/developer-resources · https://www.sec.gov/edgar/search/ · https://data.nasdaq.com/databases/SF1 · https://data.nasdaq.com/databases/SEP · https://www.sharadar.com/ · https://wrds-www.wharton.upenn.edu/ · https://www.lseg.com/en/data-analytics/financial-data/company-data/fundamentals-data · https://www.factset.com/ · https://daloopa.com/products/api · https://quartr.com/products/quartr-api · https://fiscal.ai/ · https://docs.financialdatasets.ai/ · https://www.simfin.com/ · https://finnhub.io/docs/api/company-basic-financials · https://site.financialmodelingprep.com/ · https://www.tiingo.com/documentation/fundamentals · https://www.roic.ai/api · https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds · https://www.tikr.com/ · https://www.bamsec.com/ · https://www.wisesheets.io/ · https://dados.cvm.gov.br/ · https://www.rad.cvm.gov.br/ · https://brapi.dev/ · https://www.fundamentus.com.br/ · https://statusinvest.com.br/ · https://developer.company-information.service.gov.uk/ · https://download.companieshouse.gov.uk/en_accountsdata.html · https://www.sedarplus.ca/ · https://disclosure2.edinet-fsa.go.jp/ · https://www.jpx.co.jp/english/markets/paid-info-listing/tdnet/ · https://registers.esma.europa.eu/ · https://github.com/dgunning/edgartools · https://pypi.org/project/secfsdstools/ · https://openbb.co/

**Keywords:** fundamental data, financial statements, SEC EDGAR API, XBRL frames, company facts, 10-K 10-Q filings, point-in-time fundamentals (dados point-in-time), Sharadar SF1, Compustat WRDS, Refinitiv FactSet, Daloopa, Quartr transcripts (transcrições), Fiscal.ai, SimFin, Finnhub, Financial Modeling Prep, Tiingo, Roic.ai, EODHD, TIKR, BamSEC, Wisesheets, CVM dados abertos, ITR DFP, RAD CVM, brapi, Fundamentus, Status Invest, Companies House, SEDAR+, EDINET, JPX TDnet, ESMA FIRDS, edgartools, survivorship bias (viés de sobrevivência), look-ahead bias, demonstrações financeiras, fundamentos de ações.
