# Insider, Institutional & Ownership Data

> A dense, current (2024-2026) field guide to **positioning & ownership signals** for financial markets — who owns, who's buying/selling, who's short, and who in government is trading. Covers SEC Form 4 (insider trades), Form 13F (institutional holdings), Schedule 13D/13G (activist/passive stakes), short interest & securities-lending, fails-to-deliver, congressional/political trading, and ETF/fund flows — with real URLs, free-vs-paid flags, API endpoints, Brazil 🇧🇷 (CVM/B3) equivalents, and honest notes on **reporting lag, point-in-time, look-ahead, and signal decay**. Audience: quants and researchers (Brazil-heavy; English body, termos em português entre parênteses).

This page catalogs **ownership/positioning data** not indexed elsewhere in this repo. Generic price APIs (yfinance/Polygon), options/derivatives, crypto, FRED-basic macro, Kaggle/HuggingFace pulls, arXiv q-fin, and the broader alt-data taxonomy (news/satellite/card/on-chain) live in their own pages and are referenced only in passing. The focus here is the **regulatory-filing exhaust** that reveals smart-money and insider positioning — most of it **free at the primary source** and resold (cleaned, normalized, API-ready) by a layer of vendors.

The unifying mental model: every signal below is a **mandatory disclosure** with a **reporting deadline**. The deadline *is* the signal's half-life. Form 4 → 2 business days (near-real-time). 13D → 5 business days (was 10 calendar days until the rule took effect **Feb 5, 2024**). Short interest → twice monthly, ~4-day lag. 13F → **45 days after quarter-end** (stalest of all). Knowing the lag tells you whether a feed is tradeable or merely descriptive.

---

## 1. Insider transactions — SEC Forms 3/4/5 (operações de insiders)

Under **Section 16** of the Securities Exchange Act of 1934, directors, officers, and >10% beneficial owners ("insiders") must report trades in their own company's stock: **Form 3** (initial ownership), **Form 4** (changes — the workhorse, due within **2 business days**), **Form 5** (annual catch-up). Filed electronically to EDGAR as structured XML, so the data is machine-readable at the source. Academic literature finds insider **buys** (especially routine-cluster purchases by multiple insiders) carry more signal than sells; **opportunistic** insiders outperform **routine** filers (Cohen, Malloy & Pomorski 2012).

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **SEC EDGAR (primary)** | Raw Form 3/4/5 XML + filing index | All US issuers, 2003→ (XML); 2001→ scanned | **Free** | EDGAR full-text search; `data.sec.gov/submissions/`; daily index `Archives/edgar/full-index/`; Form 4 XML in each accession | [sec.gov/edgar](https://www.sec.gov/search-filings) |
| **SEC Insider Transactions Data Sets** | Quarterly bulk flat files (TSV) parsed from Forms 3/4/5 | 2006→, quarterly; 2025 sets added `AFF10B5ONE` (10b5-1 flag) | **Free** | ZIP bulk download | [sec.gov/.../insider-transactions-data-sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets) |
| **OpenInsider** | Cleaned Form 4 screener; cluster buys, CEO/CFO buys, value filters | US, real-time off EDGAR | **Free** (web + HTML-table/CSV scrape) | No official API; scrape `openinsider.com/screener?...` query strings | [openinsider.com](http://openinsider.com/) |
| **SECForm4.com** | Insider trade tracker & analytics, alerts | US | Free + paid alerts | Web/email | [secform4.com](https://www.secform4.com/) |
| **sec-api.io** | Insider Ownership & Trading API; bulk JSONL | **2003→, ~11.4M transactions, 277K insiders, 22.3K issuers; survivorship-bias-free**; ~300 ms post-filing | **Paid** (free trial) | `POST api.sec-api.io/insider-trading` (Lucene) + `/bulk/form-4/YEAR/...jsonl.gz` | [sec-api.io/docs/insider-ownership-trading-api](https://sec-api.io/docs/insider-ownership-trading-api) |
| **Finnhub** | Insider transactions + **Insider Sentiment (MSPR)** monthly buy/sell ratio | US/UK/CA/AU/IN/EU; from Form 3/4/5 + SEDI | **Free tier** (60 req/min); sentiment may need premium | `/stock/insider-transactions`, `/stock/insider-sentiment` | [finnhub.io/docs/api/insider-transactions](https://finnhub.io/docs/api/insider-transactions) |
| **EODHD** | Insider Transactions API (Form 4), daily refresh | US-listed, sourced from EDGAR | **Paid** (cheap tiers) | New `/api/sec-filings/{symbol}/form4`; legacy `/api/insider-transactions` | [eodhd.com/.../insider-transactions-api](https://eodhd.com/financial-apis/insider-transactions-api) |
| **Quiver Quantitative** | "Insider Trades" dataset (Form 4 normalized) | US | Free tier + paid (Premium ~US$25/mo; tiered API) | Quiver API (`api.quiverquant.com`) | [quiverquant.com](https://www.quiverquant.com/) |
| **API Ninjas / Apify** | Lightweight insider-trade endpoints & EDGAR scrapers | US | Free tier / pay-per-run | REST / Apify actors | [api-ninjas.com/api/insidertrading](https://api-ninjas.com/api/insidertrading) |

**Signal notes.** Distinguish **open-market purchases (P)** from **option exercises / 10b5-1 plan sales** — the latter are scheduled and carry little signal (the new `AFF10B5ONE` flag finally tags them). Watch **cluster buys** (multiple insiders, same window) and **CEO/CFO conviction buys**. Decay: the 2-day deadline means the easy alpha is largely arbitraged within hours of the filing hitting EDGAR; the durable edge is in *aggregation* and *classification*, not in being first.

**🇧🇷 Brazil:** CVM **Resolução 44/2021** (replaced Instrução 358) requires administradores, controladores e acionistas relevantes to report trades. Public bulk data lives at the **CVM Portal de Dados Abertos** dataset **"Valores Mobiliários Negociados e Detidos"** (`cia_aberta-doc-vlmo`, CSV, last 5 years) — the closest Brazilian analogue to Form 4. There is also a **15-day blackout (período de vedação)** before quarterly/annual results. Insider trading itself is criminalized (Lei 6.385, art. 27-D; Lei 13.506/2017). Links: [dados.cvm.gov.br/dataset/cia_aberta-doc-vlmo](https://dados.cvm.gov.br/dataset/cia_aberta-doc-vlmo).

---

## 2. Institutional holdings — SEC Form 13F (carteiras institucionais)

Managers with **>US$100M** in 13F-eligible US securities must file **Form 13F-HR** quarterly, listing long positions (shares, market value, CUSIP, call/put flag, voting authority). This is how you track Berkshire, Bridgewater, Citadel, Renaissance, Scion, etc. **Critical limitations:** 13F shows **long US-listed equities + listed options only** — no shorts, no cash, no FX, no foreign-listed names, no bonds. Filed up to **45 days after quarter-end**, so positions can be **up to ~135 days stale** by the time you read them. You also can't tell *when* in the quarter a position was opened/closed.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **SEC EDGAR (primary)** | Raw 13F-HR XML information tables | All filers, 1999→ | **Free** | EDGAR; XML per accession; `data.sec.gov` | [sec.gov/edgar](https://www.sec.gov/search-filings) |
| **13F.info** | Clean free viewer; cross-quarter diff, manager history, "who holds X" | All filers | **Free** | Web (no formal public API) | [13f.info](https://13f.info/) |
| **WhaleWisdom** | 13F + 13D/G + insider; **WhaleScore** manager ranking, backtester, clone tool | 13F back to **2001** | Free tier (2 yrs) → paid (~US$300/yr); **API** | REST API (20 req/min); Std=50 filers/stocks per qtr, Pro=200, Enterprise=unlimited + live feed; Excel add-in | [whalewisdom.com](https://whalewisdom.com/) · [API](https://whalewisdom.com/help/api) |
| **Fintel** | Institutional ownership %, fund holdings, + short & options | Institutions >US$100M AUM | Freemium → paid | Web + subscriber data | [fintel.io](https://fintel.io/) |
| **HedgeFollow** | Holdings of **10,000+** institutions (Buffett, Soros, …); portfolio tracker | US institutions | Freemium | Web | [hedgefollow.com/13f](https://hedgefollow.com/13f) |
| **Unusual Whales — Institutions** | 13F holdings + flow overlays | US | Paid (sub) | Web + platform API | [unusualwhales.com/institutions](https://unusualwhales.com/institutions) |
| **Finnhub** | Institutional ownership (13F-derived) over time | US | Free tier | `/stock/institutional-ownership`, `/institutional-portfolio` | [finnhub.io/docs/api/institutional-portfolio-13f](https://finnhub.io/docs/api/institutional-portfolio-13f) |
| **sec-api.io** | Form 13F Holdings dataset, normalized | **May 2013→** (this dataset; raw 13F-HR XML on EDGAR goes back to 1999) | Paid | REST + bulk; one row per position, CUSIP enriched w/ ticker+CIK | [sec-api.io/datasets/form-13f-holdings](https://sec-api.io/datasets/form-13f-holdings) |
| **Apify 13F actors** | DIY scrapers parsing 13F-HR XML (CUSIP, shares, value, calls/puts) | 5,000+ funds | Pay-per-run / some free | Apify REST | [apify.com/.../sec-13f-holdings](https://apify.com/ryanclinton/sec-13f-holdings/api) |

**Signal notes.** 13F "clone" strategies suffer from the 45-day lag and long-only blindness — they describe *what was held*, not *what is held now*. Useful for: crowding/consensus detection, new-position discovery, sector-rotation among big managers, and **point-in-time discipline studies**. Beware **CUSIP→ticker mapping drift** (corporate actions) and **confidential treatment requests** (managers can delay disclosing sensitive positions, creating a backfill hole). The SEC stamps every filing "not necessarily reviewed… accuracy not determined" — clean it yourself.

**🇧🇷 Brazil:** No exact 13F analogue. Closest signals: the **CVM Formulário de Referência (FRE)** (`cia_aberta-doc-fre`, governed by **Resolução CVM 80/2022**) discloses controlling/relevant shareholders, board, and 5%+ holders; B3's **"Ações em Circulação no Mercado"** publishes free-float and ownership-structure data; fund portfolios are disclosed via CVM fund informes. Foreign-investor *aggregate* flows: B3's **"Investidor Estrangeiro"** participation statistics. Links: [dados.cvm.gov.br/dataset/cia_aberta-doc-fre](https://dados.cvm.gov.br/dataset/cia_aberta-doc-fre); [ri.b3.com.br](https://ri.b3.com.br/).

---

## 3. Activist & 5%+ stakes — Schedule 13D / 13G (participações relevantes)

Anyone crossing **5% beneficial ownership** of a US issuer files within days: **Schedule 13D** = intent to **influence/control** (activist — board fights, M&A pressure), **Schedule 13G** = **passive** accumulation (index funds, long-only). A 13D appearing on a name is a classic event-study trigger (activists historically generate abnormal returns around the filing). **Deadline shortened in 2024:** the SEC's Feb 2023 rule (effective Feb 5, 2024) cut the 13D initial deadline from **10 → 5 business days** and tightened 13G timelines — a material change to the signal's freshness.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **SEC EDGAR (primary)** | Raw SC 13D / 13G / amendments (`SC 13D`, `SC 13G`, `/A`) | All US issuers | **Free** | EDGAR full-text & form-type search; daily index | [sec.gov/edgar](https://www.sec.gov/search-filings) |
| **Fintel — Activists** | Latest 13D activist filings feed | US | Freemium | Web | [fintel.io/activists](https://fintel.io/activists) |
| **WhaleWisdom** | 13D/G processed **daily**; augments 13F holdings between quarters | US | Freemium → paid | Web + Excel add-in + API | [whalewisdom.com](https://whalewisdom.com/) |
| **13D Monitor** | Premier subscription **activist-research** service (Ken Squire, since 2006); analyzes ~2,000 13Ds + ~4,000 amendments/yr, tracks 150–200 live campaigns; powers the 13D Activist Fund (DDDIX) | US | **Paid** (subscription) | Web research reports | [13dmonitor.com](https://www.13dmonitor.com/) |
| **ForcedAlpha — Activist Stakes** | Daily-updated 13D/13G tracker | ~2,800+ active positions | Freemium | Web tool | [forcedalpha.com/tools/activist-stakes](https://forcedalpha.com/tools/activist-stakes/) |
| **13F Insight** | 13D/activist explainers + position data | US | Freemium | Web | [13finsight.com](https://13finsight.com/learn/what-is-13d-filing-activist-investor-positions-guide) |
| **sec-api.io** | 13D/13G in filings/full-text APIs | US, 1994→ | Paid | REST query | [sec-api.io](https://sec-api.io/) |

**Signal notes.** Distinguish 13D (tradeable event) from 13G (mostly noise — it's BlackRock/Vanguard crossing 5%). Track **amendments** (13D/A): stake increases, going to ≥10%, or shift from passive→activist (13G→13D conversion) are the high-information events. With the 5-day deadline, front-running the public filing is largely closed; the edge is in *campaign-outcome prediction* and *which activist* (some funds' filings move stocks more than others). **🇧🇷 Brazil:** CVM **Resolução 44** requires disclosure when crossing **5% multiples** (and changes thereof) of a class of shares — the functional 13D/13G analogue (*comunicado de participação relevante*), filed via CVM/B3 and surfaced in the FRE and company *fatos relevantes*.

---

## 4. Short interest & securities lending (posições vendidas e aluguel de ações)

Two distinct data families: **(a) Short interest** — official, lagged, twice-monthly snapshots of shares sold short; **(b) Securities-lending analytics** — near-real-time *estimates* of borrow fees, utilization, days-to-cover, squeeze risk (vendor-modeled from buy/sell-side lending data). Don't confuse them.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **FINRA Equity Short Interest** | Official short interest per security, **twice monthly** (mid + end of month), due 6pm ET on **T+2** after settlement date | All US equities | **Free** | Bulk download / FINRA Data catalog | [finra.org/.../equity-short-interest/data](https://www.finra.org/finra-data/browse-catalog/equity-short-interest/data) |
| **FINRA Daily/Monthly Short Sale Volume** | Aggregated short-sale *volume* (≠ short interest) per security, daily | US | **Free** | Daily/monthly flat files | [finra.org/.../daily-short-sale-volume-files](https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files) |
| **Ortex** | **Daily short-interest estimates**, borrow fee, utilization, days-to-cover, est. shorts | Global (US + intl) | **Paid** (also via Nasdaq Data Link `ORTX`) | API; Nasdaq Data Link database | [public.ortex.com](https://public.ortex.com/ortex-short-interest-data/) · [Nasdaq ORTX](https://data.nasdaq.com/databases/ORTX) |
| **S3 Partners** | True short interest, float-adjusted utilization, bid/offer rates, **Crowded Score** & **Squeeze Risk** | Global | **Paid** | SFTP, API, Snowflake, **Bloomberg Data License (2025)**, FactSet, GS Marquee, AWS | [s3partners.com](https://www.s3partners.com/) |
| **Fintel — Shorts** | Short interest + **short-squeeze score**, daily est. between FINRA prints, borrow rates | US (NASDAQ/NYSE/Cboe/IEX + FINRA) | Freemium → paid | Web + subscriber | [fintel.io](https://fintel.io/) |
| **Cboe Short Interest Report** | Exchange short-interest dataset | US (Cboe-listed/traded) | Paid | Cboe DataShop | [datashop.cboe.com](https://datashop.cboe.com/cboe-short-interest-report) |
| **Nasdaq Data Link ORTX** | Ortex daily short-interest estimates, packaged | Global | Paid | Nasdaq Data Link API | [data.nasdaq.com/databases/ORTX](https://data.nasdaq.com/databases/ORTX) |

**Signal notes.** FINRA short interest is **official but lagged** (~4 trading days after the settlement reference date, only twice a month) — it tells you the *level*, not the trend. Vendor estimates (Ortex/S3) interpolate daily from lending data and lead the official prints, but are *models* with their own error. **Days-to-cover (DTC)** and **utilization** matter more than raw % float for squeeze risk. Note: a 2026 FINRA proposal (new Rule 4321 + Rule 4560 amendments) would further change short-interest/FTD-allocation reporting — track the final rule. **🇧🇷 Brazil:** B3 publishes daily **aluguel de ações (BTB / securities lending)** open-interest and the **"Posições em Aberto"** of stock lending — the local proxy for short pressure (no FINRA-style consolidated short interest exists). Data via B3 market-data files (`rb3` R package parses B3 public files) and B3's empréstimo de ativos statistics.

---

## 5. Fails-to-deliver (FTD) — settlement failures (falhas de liquidação)

The SEC publishes **fails-to-deliver** data from the NSCC's **Continuous Net Settlement (CNS)** system: for each settlement date, the aggregate shares that failed to deliver per CUSIP/ticker. Heavily used by retail/meme communities as a (contested) proxy for naked-short pressure. **Read carefully:** a day's FTD figure is *cumulative* (all outstanding fails to date + new − settled), not a daily flow.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **SEC Fails-to-Deliver Data** | Date, CUSIP, ticker, issuer, price, total FTD shares (CNS aggregate) | 2004→; **all** securities w/ any balance from Sep 16 2008 (≥10k before) | **Free** | Semi-monthly flat files (`.txt`/`.zip`) | [sec.gov/.../fails-deliver-data](https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data) |
| **gaming-wall-street/cns-fails-sec** | Cleaned, concatenated SEC FTD history (static data repo) | US, 2004→ | **Free** (OSS) | GitHub data repo | [github.com/gaming-wall-street/cns-fails-sec](https://github.com/gaming-wall-street/cns-fails-sec) |
| **juszhan/Fail-To-Deliver** | FTD visualizer over SEC data (2010–2020) | US | **Free** (OSS) | GitHub | [github.com/juszhan/Fail-To-Deliver](https://github.com/juszhan/Fail-To-Deliver) |
| Ortex / Fintel | FTD surfaced alongside short data | US | Freemium/paid | Web/API | [public.ortex.com](https://public.ortex.com/) |

**Caveat:** FTDs arise from many benign causes (operational errors, market-maker timing, bona-fide arbitrage) — high FTD ≠ proof of naked shorting. Treat as a *secondary*, noisy signal; the SEC itself cautions against over-interpretation.

---

## 6. Congressional & political trading (negociações de políticos)

The **STOCK Act (2012)** requires US Members of Congress (and many senior officials) to disclose securities transactions ≥US$1,000 within **45 days** via **Periodic Transaction Reports (PTRs)**. The primary sources are free government portals; vendors parse the (often hand-scanned PDF) filings into clean, queryable feeds and compute returns. This has become a popular retail/alt-data factor (mind the **legal/ethical** caveats and the 45-day staleness).

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **House Clerk Financial Disclosure (primary)** | Official House PTRs & annual reports (PDF) | US House | **Free** | Search portal; bulk annual ZIPs (XML/PDF) | [disclosures-clerk.house.gov](https://disclosures-clerk.house.gov/FinancialDisclosure) |
| **Senate eFD (primary)** | Official Senate PTRs (`efdsearch.senate.gov`) | US Senate | **Free** | Search portal (scrape-only) | [efdsearch.senate.gov](https://efdsearch.senate.gov/search/home/) |
| **Senate/House Stock Watcher** | Community-parsed JSON of Senate (and House) trades | US Congress | **Free** (OSS) | JSON dumps; GitHub | [github.com/.../senate-stock-watcher-data](https://github.com/timothycarambat/senate-stock-watcher-data) |
| **Quiver Quantitative** | Congress (House/Senate) trades, politician net worth, lobbying, gov contracts, patents, donors, **off-exchange/dark-pool**, exec comp | US | Free web + **paid API** (Premium ~US$25/mo; tiered API: Hobbyist/Trader/Commercial) | `/datasets/congress-trades` (REST) | [api.quiverquant.com/datasets/congress-trades](https://api.quiverquant.com/datasets/congress-trades) |
| **Capitol Trades** | Cleanest congress-only UI; member/issuer pages (by 2iQ Research) | US Congress (140 members / 14,451 trades reported in 2025) | **Free** (web; no official public API) | Web (scrape) | [capitoltrades.com](https://www.capitoltrades.com/) |
| **Unusual Whales — Politicians** | Congress trades + **options-flow overlay**; annual Congress Trading Report | US | Paid (sub) | Platform + API | [unusualwhales.com/politics](https://unusualwhales.com/politics) |
| **Apify Congress trackers** | STOCK Act PTR scrapers/feeds | US | Pay-per-run | Apify REST | [apify.com/.../congress-stock-tracker](https://apify.com/ryanclinton/congress-stock-tracker) |

**Signal notes.** Top-traded 2025 names skewed mega-cap tech (NVDA, MSFT, AMZN, AAPL, GOOGL per Capitol Trades / Unusual Whales). Data quality issues: **45-day lag**, **wide dollar ranges** (filings report buckets like "$1,001–$15,000," not exact size), late/amended filings, and family-vs-member attribution. Build on the **official portals** for ground truth; use vendors for parsing + return computation. **🇧🇷 Brazil:** No STOCK-Act analogue. Closest transparency: **TSE** electoral asset declarations (*declaração de bens* de candidatos) and the **Portal da Transparência** — neither is a trade-level feed.

---

## 7. ETF & fund flows (fluxos de ETFs e fundos)

Flows = creations/redemptions (ETFs) and net subscriptions (mutual funds), a **demand/positioning** signal at the asset-class, sector, factor, and single-fund level. Daily ETF flows are estimable from shares-outstanding × NAV deltas (public); broad mutual-fund flows come from industry aggregators.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **ETF.com Fund Flows Tool** | Net flows by ticker/group, date-range, intervals | US-listed ETFs | **Free** (web) | Web tool (input tickers) | [etf.com/etfanalytics/etf-fund-flows-tool](https://www.etf.com/etfanalytics/etf-fund-flows-tool) |
| **VettaFi / ETF Database (etfdb)** | ETF flows, classifications, screeners; owns ETF Trends, ETF Database | US ETFs | Freemium → paid (institutional indexing/data) | Web + enterprise data | [etfdb.com/etf-fund-flows](https://etfdb.com/etf-fund-flows/) · [vettafi.com](https://www.vettafi.com/) |
| **ICI (Investment Company Institute)** | **Weekly** estimated long-term mutual-fund flows + **weekly combined ETF + LT flows**; covers ~98% of MF/ETF assets | US industry totals | **Free** | Weekly releases (CSV/HTML) | [ici.org/research/stats/flows](https://www.ici.org/research/stats/flows) |
| **Morningstar Direct / Fund Flows** | Flows by asset class, category, domicile, monthly | Global funds & ETFs | **Paid** (some free commentary) | Direct platform / API | [morningstar.com/business/insights](https://www.morningstar.com/business/insights/blog/funds/us-fund-flows) |
| **EPFR (ISI Markets)** | As-reported global fund flows & allocation; **155,000+ share classes, US$70T+, since 1995**; daily/weekly/monthly | Global | **Paid** | API / feeds | [isimarkets.com/epfr](https://isimarkets.com/epfr/) |
| **LSEG Lipper** | Fund flows & holdings (legacy Refinitiv) | Global | Paid | Data platform / WRDS | [lseg.com](https://www.lseg.com/en/data-analytics) |
| **Nasdaq Data Link `ETFF`** | US ETF fund-flows database | US ETFs | Paid | Nasdaq Data Link API | [data.nasdaq.com/databases/ETFF](https://data.nasdaq.com/databases/ETFF) |
| **FMP** | ETF flows / holdings / buy-side sentiment endpoints | US | Freemium → paid | REST | [financialmodelingprep.com](https://site.financialmodelingprep.com/insights/enterprise/etf-flows-and-buyside-sentiment-data-apis-capital-flow-and-sentiment) |

**Signal notes.** ETF daily flow estimates are reverse-engineered (Δshares-outstanding × prior NAV) and can be noisy on creation/redemption timing; aggregators reconcile against issuer reporting. Flows are a **sentiment/positioning** read, not a forecast — chasing them is momentum, fading extremes is mean-reversion; both regimes exist. **🇧🇷 Brazil:** ETF flows via B3 (cotas em circulação dos ETFs, e.g., BOVA11) and fund flows via **ANBIMA** (consolidated fund-industry statistics) and **CVM** fund informes (`fi_doc` datasets — captação/resgate de fundos). Links: [anbima.com.br](https://www.anbima.com.br/); [dados.cvm.gov.br](https://dados.cvm.gov.br/).

---

## 8. International ownership/short registers (registros internacionais)

| Source | Content | Coverage | Free/Paid | Link |
|---|---|---|---|---|
| **ESMA short-selling** | Links to all EU national net-short-position registers; SSR framework | EU/EEA | **Free** | [esma.europa.eu/.../short-selling](https://www.esma.europa.eu/esmas-activities/markets-and-infrastructure/short-selling) |
| **UK FCA net short positions** | Public net-short disclosures (≥0.5% threshold); **new SSR 2025 regime from 13 Jul 2026** | UK | **Free** | [fca.org.uk/markets/short-selling](https://www.fca.org.uk/markets/short-selling/notification-disclosure-net-short-positions) |
| **EU "major shareholding" (TR-1)** | Transparency Directive 5%+ holdings notifications (national NSMs) | EU | **Free** | Per-country national storage mechanisms |

EU/UK short disclosure is **named and position-holder-level** (you see *which* fund is short *which* stock above threshold) — richer than US aggregate short interest, but only above the disclosure threshold. The temporary **0.1% net-short reporting threshold** (private to regulators) has been repeatedly extended by ESMA/FCA.

---

## 9. Cross-cutting caveats — lag, point-in-time, look-ahead, legality

- **Reporting lag = signal half-life.** Form 4 (2 bd) → fresh; 13D (5 bd) → fresh-ish; short interest (twice-monthly, ~4-day) → stale-ish; **13F (45 days) → up to ~135 days stale**; STOCK Act PTRs (45 days). Always tag each datapoint with *filing date* vs *as-of/transaction date* and backtest on the **filing date**, never the as-of date (classic look-ahead trap).
- **Point-in-time / restatement.** 13F amendments, 13D/A, confidential-treatment delays, and late STOCK Act filings all **rewrite history**. Insist on **vintage/point-in-time** snapshots; survivorship-bias-free panels (sec-api.io advertises this) matter because delisted issuers and dissolved funds vanish from naive pulls.
- **Coverage holes.** 13F = long US equities + listed options only (no shorts/cash/FX/foreign/bonds). Short interest ≠ short-sale volume ≠ FTDs (three different things). Trade-size buckets in STOCK Act filings preclude precise sizing.
- **CUSIP↔ticker mapping** drifts with corporate actions; CUSIP itself is licensed (CUSIP Global Services) — open mappers approximate it.
- **Legality & ethics.** Trading *on* a public disclosure is legal; the underlying acts may not be (insider trading is criminal in the US and 🇧🇷 Brazil). Congressional-trade copying is legal but politically/ethically fraught; vendor ToS often restrict redistribution.
- **Decay.** As these feeds commoditize (free EDGAR + cheap APIs), naive "follow the filing" alpha decays toward zero. Durable edge: **aggregation, classification (routine vs opportunistic, passive vs activist, 10b5-1 vs discretionary), cross-signal fusion** (insider buys + rising short interest + activist 13D), and disciplined point-in-time backtesting.

---

## Related in AIForge
- [Alternative Data & Sentiment Analysis (README)](./README.md) · [Prediction Markets, ESG & New Alt-Data](./Prediction_Markets_ESG_and_New_Alt_Data.md)
- [Datasets, APIs & Data Vendors](../Datasets_APIs_and_Data_Vendors/) · [Equities & Stock Markets](../Equities_and_Stock_Markets/) · [ETFs, Funds & Indexing](../ETFs_Funds_and_Indexing/) · [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Technical & Fundamental Analysis](../Technical_and_Fundamental_Analysis/)

**Sources:** [SEC EDGAR](https://www.sec.gov/search-filings) · [SEC Insider Data Sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets) · [SEC FTD](https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data) · [OpenInsider](http://openinsider.com/) · [sec-api.io insider](https://sec-api.io/docs/insider-ownership-trading-api) · [Finnhub insider](https://finnhub.io/docs/api/insider-transactions) · [EODHD](https://eodhd.com/financial-apis/insider-transactions-api) · [13f.info](https://13f.info/) · [WhaleWisdom](https://whalewisdom.com/) · [WhaleWisdom API](https://whalewisdom.com/help/api) · [Fintel](https://fintel.io/) · [HedgeFollow](https://hedgefollow.com/13f) · [Unusual Whales institutions](https://unusualwhales.com/institutions) · [sec-api.io 13F](https://sec-api.io/datasets/form-13f-holdings) · [Fintel activists](https://fintel.io/activists) · [13D Monitor](https://www.13dmonitor.com/) · [ForcedAlpha 13D/G](https://forcedalpha.com/tools/activist-stakes/) · [FINRA short interest](https://www.finra.org/finra-data/browse-catalog/equity-short-interest/data) · [FINRA short volume](https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files) · [Ortex](https://public.ortex.com/ortex-short-interest-data/) · [S3 Partners](https://www.s3partners.com/) · [Cboe short interest](https://datashop.cboe.com/cboe-short-interest-report) · [House Clerk disclosures](https://disclosures-clerk.house.gov/FinancialDisclosure) · [Senate eFD](https://efdsearch.senate.gov/search/home/) · [Senate Stock Watcher data](https://github.com/timothycarambat/senate-stock-watcher-data) · [Quiver congress API](https://api.quiverquant.com/datasets/congress-trades) · [Capitol Trades](https://www.capitoltrades.com/) · [ETF.com flows](https://www.etf.com/etfanalytics/etf-fund-flows-tool) · [VettaFi](https://www.vettafi.com/) · [etfdb flows](https://etfdb.com/etf-fund-flows/) · [ICI flows](https://www.ici.org/research/stats/flows) · [EPFR (ISI Markets)](https://isimarkets.com/epfr/) · [Morningstar fund flows](https://www.morningstar.com/business/insights/blog/funds/us-fund-flows) · [ESMA short-selling](https://www.esma.europa.eu/esmas-activities/markets-and-infrastructure/short-selling) · [FCA net short positions](https://www.fca.org.uk/markets/short-selling/notification-disclosure-net-short-positions) · [CVM Dados Abertos VLMO](https://dados.cvm.gov.br/dataset/cia_aberta-doc-vlmo) · [CVM FRE](https://dados.cvm.gov.br/dataset/cia_aberta-doc-fre) · [B3 RI](https://ri.b3.com.br/) · [rb3 R package](https://docs.ropensci.org/rb3/) · [gaming-wall-street/cns-fails-sec](https://github.com/gaming-wall-street/cns-fails-sec)

**Keywords:** insider trading data (operações de insiders), SEC Form 4 EDGAR, OpenInsider, 13F institutional holdings (carteiras institucionais 13F), WhaleWisdom Fintel HedgeFollow 13F.info, Schedule 13D 13G activist stakes (participações relevantes), 13D Monitor activist research, short interest data (posições vendidas), Ortex S3 Partners FINRA short interest, fails-to-deliver FTD (falhas de liquidação), congressional trading (negociações de políticos), Quiver Quantitative Capitol Trades Unusual Whales, STOCK Act PTR, ETF fund flows (fluxos de ETFs), VettaFi ICI EPFR, ownership data (dados de participação acionária), CVM B3 valores mobiliários negociados, securities lending aluguel de ações, point-in-time look-ahead survivorship bias, signal decay.
