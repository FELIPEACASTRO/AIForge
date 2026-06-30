# Analyst Estimates & Sell-Side Data

> Where to get **analyst forecasts, consensus estimates, price targets and broker ratings** (estimativas de analistas, consenso, preço-alvo, recomendações) — the "Street view" that earnings surprises are measured against. Covers institutional incumbents (I/B/E/S, FactSet, Visible Alpha, Capital IQ, Zacks), retail/dev-friendly APIs (Finnhub, FMP, TipRanks, Benzinga, Intrinio), crowdsourced/whisper data, research-aggregation platforms (AlphaSense/Tegus, Daloopa), the **PEAD / earnings-surprise** trading angle, and 🇧🇷 Brazil equivalents. Honest on cost, coverage, point-in-time and look-ahead bias.

This page is **new** to AIForge and complements [Alternative Data & Sentiment Analysis](./README.md) and [Prediction Markets, ESG & New Alt-Data](./Prediction_Markets_ESG_and_New_Alt_Data.md). Generic market-data APIs (yfinance/Polygon/FRED) are covered elsewhere; here the product is the **forecast and the analyst opinion**, not the quote.

---

## 1. Why sell-side estimates are their own data class

A consensus number is the **denominator of every earnings surprise**. `Surprise = Actual − Consensus`; the standardized version `SUE = (Actual − E[EPS]) / σ(estimates)` drives the **Post-Earnings-Announcement Drift (PEAD)** — prices keep drifting in the direction of the surprise for weeks (Ball & Brown 1968; [Wikipedia overview](https://en.wikipedia.org/wiki/Post%E2%80%93earnings-announcement_drift), [Quantpedia strategy](https://quantpedia.com/strategies/post-earnings-announcement-effect)). Estimate **revisions**, **dispersion**, **price-target changes** and **rating changes** are all independent, well-studied signals. Key academic anchors:

| Finding | Paper | Link |
|---|---|---|
| PEAD anomaly exists; surprises drift | Ball & Brown (1968); Bernard & Thomas (1989) | [PEAD review, *JBEF* 2021](https://www.sciencedirect.com/science/article/pii/S2214635020303750) |
| High **forecast dispersion → lower** future returns (differences-of-opinion) | Diether, Malloy, Scherbina (2002), *JF* | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=472362) · [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00490) |
| Consensus **rating levels/changes** profitable only gross of costs | Barber, Lehavy, McNichols, Trueman (2001), *JF* | [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00336) |
| **Whisper numbers** beat published consensus on accuracy | Whisper-number literature | [Wikipedia](https://en.wikipedia.org/wiki/Whisper_number) |
| PEAD from **text** of calls/filings | "PEAD.txt" (Meursault, Liang, Routledge, Scanlon; CMU, *JFQA* 2023) | [CMU page](https://www.cmu.edu/tepper/accounting-lab/publications/pead.txt-post-earnings-announcement-drift-using-textpage.html) |

> **Trap to internalize:** most retail estimate feeds are **last-value (revised), not point-in-time**. Backtesting PEAD/revision strategies on a feed that shows *today's* consensus stitched onto a past date is classic **look-ahead bias**. For research you generally need a **point-in-time / as-was** consensus (I/B/E/S, FactSet PIT, Visible Alpha) or a feed with full revision history timestamps.

---

## 2. Institutional incumbents (the "real" consensus)

These are the datasets buy-side quants and academics actually use; expensive, deep, point-in-time, broad international coverage.

| Source | Content | Coverage | Free/Paid | API / How | Link |
|---|---|---|---|---|---|
| **I/B/E/S** (LSEG/Refinitiv) | The reference consensus + detail file: EPS, revenue, EBITDA, price target, 20+ measures; consensus **and** analyst-level; deep history | 23,000+ active cos, 90+ countries, 19,000+ analysts; US back to **1976**, intl to 1987 | Paid (institutional) | LSEG **Workspace** / DataScope; **WRDS** (academic); Data feeds | [LSEG I/B/E/S](https://www.lseg.com/en/data-analytics/financial-data/company-data/ibes-estimates) · [WRDS linking](https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-ibes-with-thomson-refinitiv/) |
| **FactSet Estimates** | Consensus + broker-supplied detail; **point-in-time consensus** product; segment/KPI metrics | 19,000+ cos, 90+ countries; 800+ contributors / 55 countries; 90% from research reports | Paid | **FactSet Estimates API** (developer portal); DataFeed; AWS/Databricks/Snowflake marketplaces | [API catalog](https://developer.factset.com/api-catalog/factset-estimates-api) · [PIT consensus](https://insight.factset.com/resources/at-a-glance-factset-estimates-point-in-time-consensus) |
| **Visible Alpha** (S&P Global, acq. 2024) | **Granular, line-item** consensus built from *full working sell-side Excel models* — segment revenue, KPIs, drivers, not just EPS | 200M+ data points, 1M+ consensus line items, ~3,800 analyst line items/co, 200+ brokers | Paid; add-on | Web platform, **API/Feeds**; now on **S&P Capital IQ Pro** (Mar 2025) | [Visible Alpha](https://www.spglobal.com/market-intelligence/en/solutions/visible-alpha) · [CIQ Pro launch](https://press.spglobal.com/2025-03-25-S-P-Global-Market-Intelligence-Launches-Visible-Alpha-on-S-P-Capital-IQ-Pro-Platform) |
| **S&P Capital IQ Estimates** | 140+ estimate metrics; consensus + line-item; ChatIQ AI layer | 19,200+ cos, 110+ countries; line-items 7,300+ cos | Paid | **API / API Drive**, Xpressfeed, Snowflake, Databricks Delta Sharing, ClariFI | [CIQ Estimates](https://www.spglobal.com/market-intelligence/en/solutions/products/estimates) · [Marketplace](https://www.marketplace.spglobal.com/en/datasets/s-p-capital-iq-estimates-(1)) |
| **Bloomberg (BEst)** | Bloomberg Estimates: consensus EPS/rev/EBITDA, ratings, targets; `BEST_*` fields | Global | Paid (Terminal/B-PIPE/Data License) | Terminal `EE`/`ANR`, BQL, Data License | (Terminal product; no public API) |

🇧🇷 **Brazil note:** I/B/E/S/LSEG, FactSet and Bloomberg all cover liquid B3 names (e.g. BBAS3, B3SA3, PETR4, VALE3). Domestic broker research (BTG, Itaú BBA, XP, BBI) feeds the global consensus but is also distributed directly via Brazilian platforms (§6).

---

## 3. Mid-tier & "prosumer" platforms (UI-first, some data export)

| Source | Content | Free/Paid | API | Link |
|---|---|---|---|---|
| **Zacks** (Zacks Data) | Consensus EPS/sales/EBITDA, **estimate revisions**, ratings, targets, surprises; the famous **Zacks Rank**; 185+ US/Canada brokers, 2,600+ analysts | Paid (consumer + institutional); some via Nasdaq Data Link / Intrinio | **REST API / bulk feeds**; also Nasdaq Data Link `ZEE/ZREV/ZEEH`, Intrinio | [Zacks Data](https://zacksdata.com/) · [Consensus datasets](https://zacksdata.com/datasets/consensus-data/) · [Nasdaq ZEE](https://data.nasdaq.com/databases/ZEE) |
| **Koyfin** | Actuals & consensus (rev/EPS), price-target range (bull/bear), estimate trends, revisions, transcripts; strong global coverage; Bloomberg-lite | Freemium; paid plans | No public data API (UI/exports) | [Koyfin estimates](https://www.koyfin.com/help/actuals-consensus/) · [data coverage](https://www.koyfin.com/data-coverage/stocks/) |
| **MarketBeat** | Aggregated analyst ratings, consensus price targets, ratings-change feed (US/UK/CA) | Free tier; **All Access $39.99/mo (or $399/yr)** | Limited; mostly web | [MarketBeat ratings](https://www.marketbeat.com/ratings/) |
| **TipRanks** | Analyst ratings/price targets with **analyst-accuracy scores**, Smart Score, consensus; 100+ FIs use it | Consumer subscription; **Enterprise** data product | **Enterprise API** (sales-gated, not self-serve); also via **FMP partnership** | [TipRanks Enterprise](https://enterprise.tipranks.com/) · [FMP×TipRanks](https://site.financialmodelingprep.com/datasets/analyst-ratings-tipranks) |

> Community wrappers exist (e.g. [`janlukasschroeder/tipranks-api-v2`](https://github.com/janlukasschroeder/tipranks-api-v2)) and scraper marketplaces (ScrapingBee, Apify ~$49/mo) — useful but **unofficial, ToS-fragile, and not redistribution-safe**. Prefer the FMP-licensed TipRanks feed for production.

---

## 4. Developer-friendly APIs (estimates you can pull today)

Best entry points for a Brazil-based dev who wants programmatic consensus/ratings without an institutional contract.

| API | Estimates content | Free tier? | Endpoints / notes | Link |
|---|---|---|---|---|
| **Finnhub** | Recommendation **trends** (buy/hold/sell counts), **price target**, **EPS/revenue/EBITDA estimates**, earnings calendar, **earnings surprises** | **Yes** (free retail key; estimates often need a paid stock plan); 25yr history, ~65,000 global cos | `/stock/recommendation`, `/stock/price-target`, `/stock/eps-estimate`, `/stock/revenue-estimate` | [Finnhub docs](https://finnhub.io/docs/api/recommendation-trends) · [pricing](https://finnhub.io/pricing) |
| **Financial Modeling Prep (FMP)** | **Analyst estimates** (rev/EPS/EBITDA), **price-target consensus** (high/low/median/avg), price-target summary/history, grades/upgrades-downgrades, **TipRanks** add-on | Freemium (free key, limited calls); paid for full | `/stable/financial-estimates`, `/stable/price-target-consensus`, `/stable/price-target-summary`, ratings/grades | [FMP estimates docs](https://site.financialmodelingprep.com/developer/docs/stable/financial-estimates) · [PT consensus](https://site.financialmodelingprep.com/developer/docs/stable/price-target-consensus) |
| **Intrinio** | Resells **Zacks**: EPS/sales/EBITDA consensus, EPS growth, analyst ratings; 5,000+ US/CA cos | Paid (trial) | `get_zacks_eps_estimates`, `get_zacks_analyst_ratings`, `get_zacks_sales_estimates` (Python/R/Web SDKs) | [Intrinio Zacks API](https://docs.intrinio.com/documentation/web_api/get_zacks_eps_estimates_v2) |
| **Benzinga** | **Analyst Ratings API** (raw, un-normalized: buy/overweight/etc.), price-target changes, **Analyst Insights**, consensus ratings; real-time streams + history | Paid (enterprise) | REST + WebSocket streams; overnight ratings ~3h pre-open; also resold via Massive | [Benzinga ratings API](https://www.benzinga.com/apis/cloud-product/analyst-ratings-api/) · [docs](https://docs.benzinga.com/api-reference/calendar-api/analyst-insights/overview) |

**Quick code (Finnhub recommendation trends, free key):**
```python
import requests
KEY = "YOUR_FINNHUB_KEY"
r = requests.get("https://finnhub.io/api/v1/stock/recommendation",
                 params={"symbol": "AAPL", "token": KEY}).json()
print(r[0])  # {'buy':.., 'hold':.., 'sell':.., 'strongBuy':.., 'period':'2026-06-01', ...}
```
> Validate fields and `period` timestamps before trusting them as point-in-time; free feeds frequently show only the **latest** snapshot per period and may overwrite history.

---

## 5. Crowdsourced & "whisper" estimates (alpha beyond the Street)

| Source | Content | Free/Paid | API / How | Link |
|---|---|---|---|---|
| **Estimize** (ExtractAlpha) | **Crowdsourced** EPS/revenue + company KPIs from buy-side, sell-side, independents, students; built to beat Wall Street consensus | Paid data; some via WRDS | Data feeds; **WRDS** academic access | [Estimize](https://www.estimize.com/) · [WRDS Estimize](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/vendor-partner-estimize/) |
| **Earnings Whispers** | The best-known **whisper-number** provider (since 1998); Earnings Whisper® number, earnings calendar, surprise grades; explicitly built on PEAD; claims ~70% closer to actual than consensus | Free site + subscription; unofficial endpoints | Web; informal JSON endpoints (`earningswhispers.com/stocks/<ticker>`) — **undocumented/ToS-fragile** | [earningswhispers.com](https://earningswhispers.com/) · [about](https://www.earningswhispers.com/about-whispers) |
| **The Whisper Number / WhisperNumber.com** | Unofficial **whisper EPS** vs consensus; investor-sentiment-driven; thewhispernumber.com covers 2,300+ stocks; whispernumber.com (Market Sentiment LLC) since 2000 | Subscription | Web (limited programmatic) | [thewhispernumber.com](https://thewhispernumber.com/) · [whispernumber.com](https://www.whispernumber.com/) |
| **ExtractAlpha** | Quant signals incl. estimate-revision & crowdsourced factors (parent of Estimize) | Paid | Feeds / partners | [extractalpha.com](https://extractalpha.com/) |

> **Why bother:** consensus is *sticky and lagged* (analysts under-react and herd). Crowdsourced/whisper sets often update faster and reduce the stale-consensus bias that PEAD exploits. Academic and vendor work claims whisper numbers historically missed actual EPS by far less than published consensus — but datasets are **thin, US-skewed, and subject to self-selection** of contributors.

---

## 6. Research-aggregation & modeling-data platforms (the "human" layer)

Not raw estimate feeds but where the **research, expert calls, and pre-built models** live — increasingly LLM/RAG-friendly.

| Platform | What it is | Content scale | Free/Paid | API / AI | Link |
|---|---|---|---|---|---|
| **AlphaSense** (incl. **Tegus**, acq. Jul 2024 for ~$930M) | AI search over filings, transcripts, broker research + **expert-call transcripts**; financial models on public cos | 250,000+ expert transcripts (post-Tegus), models on 4,000+ cos, $4B valuation | Paid (institutional) | Platform + enterprise API; **Tegus** brand folded in | [AlphaSense×Tegus](https://www.prnewswire.com/news-releases/alphasense-completes-acquisition-of-tegus-302190934.html) · [expert calls](https://www.alpha-sense.com/solutions/expert-calls/) |
| **Sentieo** | Document-search/research engine — **merged into AlphaSense (2022)**, brand retired; features live as "AlphaSense for Equity Research" | — | Paid | Via AlphaSense | [merger](https://www.integrity-research.com/alphasense-merges-with-sentieo/) |
| **Daloopa** | AI-extracted **structured fundamentals & KPIs** straight from filings; "financial-modeling copilot"; source-linked, audit-traceable | 1,300+ metrics, 5,500+ tickers, 13yr history, >99% claimed accuracy | Paid (trial) | **HTTP API** (37 endpoints / 15 groups), Excel add-in, **MCP** for LLMs/Claude/Codex | [Daloopa API](https://daloopa.com/products/api) · [Daloopa](https://daloopa.com/) |
| **Tegus** | Expert-call network + financial models (now AlphaSense) | 100,000+ transcripts pre-merger | Paid | Via AlphaSense | [Sacra: Tegus](https://sacra.com/c/tegus/) |

> Use these for **LLM/RAG over disclosures** (summarize calls, extract guidance/KPIs, build SUE inputs). Daloopa's MCP server makes it directly callable from agentic LLM workflows — clean structured fundamentals with citations, which mitigates hallucination on numbers. Cross-ref AIForge [`RAG_and_Retrieval`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/) and the alt-data NLP notes in [`README.md`](./README.md).

---

## 7. 🇧🇷 Brazil consensus & estimate sources

| Source | Content | Free/Paid | How | Link |
|---|---|---|---|---|
| **LSEG / FactSet / Bloomberg** | Global consensus covers liquid B3 names; the "official" Brazil consensus the press cites (e.g. LSEG, Bloomberg consensus for BBAS3/B3SA3) | Paid | Workspace / Terminal / feeds | [LSEG I/B/E/S](https://www.lseg.com/en/data-analytics/financial-data/company-data/ibes-estimates) |
| **TradeMap** | **Refinitiv-powered** consensus for B3: buy/sell recs, price targets, earnings projections | Freemium; **TradeMap Pro** | App / web | [TradeMap Pro](https://trademap.com.br/planos/trademap-pro) |
| **StatusInvest** | Indicators + some consensus/price-target context for B3 tickers | Free + paid | Web | [statusinvest.com.br](https://statusinvest.com.br/) |
| **Investing.com (BR)** | Per-ticker **consensus estimates & price target** pages for Brazilian equities | Free | Web | [BBAS3 consensus](https://www.investing.com/equities/brasil-on-consensus-estimates) |
| **BTG Pactual Content / Comdinheiro** | Broker research portal; Comdinheiro for quant/consensus data feeds (Brazil) | Paid | Platform / API | [BTG research](https://content.btgpactual.com/research/) |
| **Dados de Mercado** | Open Brazilian markets DB (CVM/BCB/ANBIMA/B3) with **token-auth REST API** — quotes, market/company indicators, fundamentals (not full sell-side consensus) | Free web + API (free + paid tiers) | [API docs](https://www.dadosdemercado.com.br/api/docs) (Bearer token) | [dadosdemercado.com.br](https://www.dadosdemercado.com.br/) |

> Brazil-specific caveats: analyst coverage is **thin outside the ~50–80 most liquid names**; consensus often = a handful of brokers (small N → noisy dispersion). Many domestic estimates are in **BRL and IFRS** with adjusted ("recurring"/recorrente) earnings conventions that differ from GAAP — reconcile before computing surprises.

---

## 8. Practitioner checklist — pitfalls & data hygiene

- **Point-in-time vs revised:** for backtests use as-was consensus with revision timestamps (I/B/E/S detail, FactSet PIT, Visible Alpha). Retail/free APIs are usually last-value → **look-ahead**.
- **Survivorship / coverage bias:** estimate panels drop dead/delisted tickers and under-cover small caps and EM; PEAD studies are strongest exactly where data is thinnest.
- **Actual definition mismatch:** consensus EPS is typically **non-GAAP/adjusted**; pairing it with GAAP actuals fabricates fake surprises. Match the basis (and FX for 🇧🇷/EM).
- **Stale/herding consensus:** analysts under-react — that's the *source* of PEAD, but also means raw consensus lags reality; crowdsourced/whisper or fast-revision feeds can lead.
- **Currency & calendar:** fiscal-period alignment (FQ vs CQ), restatements, and split adjustments silently corrupt revision series.
- **Licensing / redistribution:** institutional feeds (I/B/E/S, FactSet, VA, Bloomberg) forbid redistribution; scraper APIs violate ToS. For a public repo, ship **code + schema, not the data**.
- **Alpha decay:** rating/PT-change events are crowded and largely arbitraged gross-of-cost (Barber et al. 2001) — net edge is in **timing, revision momentum, and dispersion**, not headline recs.

---

## Related in AIForge
- [Alternative Data & Sentiment Analysis](./README.md) · [Prediction Markets, ESG & New Alt-Data](./Prediction_Markets_ESG_and_New_Alt_Data.md)
- [Technical & Fundamental Analysis](../Technical_and_Fundamental_Analysis/) · [Equities & Stock Markets](../Equities_and_Stock_Markets/) · [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/)
- LLM/RAG over disclosures: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/)

**Sources:** [LSEG I/B/E/S](https://www.lseg.com/en/data-analytics/financial-data/company-data/ibes-estimates) · [WRDS IBES linking](https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-ibes-with-thomson-refinitiv/) · [FactSet Estimates API](https://developer.factset.com/api-catalog/factset-estimates-api) · [FactSet PIT consensus](https://insight.factset.com/resources/at-a-glance-factset-estimates-point-in-time-consensus) · [Visible Alpha (S&P)](https://www.spglobal.com/market-intelligence/en/solutions/visible-alpha) · [VA on CIQ Pro](https://press.spglobal.com/2025-03-25-S-P-Global-Market-Intelligence-Launches-Visible-Alpha-on-S-P-Capital-IQ-Pro-Platform) · [S&P CIQ Estimates](https://www.spglobal.com/market-intelligence/en/solutions/products/estimates) · [Zacks Data](https://zacksdata.com/) · [Nasdaq Data Link ZEE](https://data.nasdaq.com/databases/ZEE) · [Intrinio Zacks API](https://docs.intrinio.com/documentation/web_api/get_zacks_eps_estimates_v2) · [Finnhub docs](https://finnhub.io/docs/api/recommendation-trends) · [Finnhub pricing](https://finnhub.io/pricing) · [FMP estimates](https://site.financialmodelingprep.com/developer/docs/stable/financial-estimates) · [FMP PT consensus](https://site.financialmodelingprep.com/developer/docs/stable/price-target-consensus) · [FMP×TipRanks](https://site.financialmodelingprep.com/datasets/analyst-ratings-tipranks) · [TipRanks Enterprise](https://enterprise.tipranks.com/) · [Benzinga ratings API](https://www.benzinga.com/apis/cloud-product/analyst-ratings-api/) · [Koyfin estimates](https://www.koyfin.com/help/actuals-consensus/) · [MarketBeat ratings](https://www.marketbeat.com/ratings/) · [Estimize](https://www.estimize.com/) · [WRDS Estimize](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/vendor-partner-estimize/) · [Earnings Whispers](https://earningswhispers.com/) · [The Whisper Number](https://thewhispernumber.com/) · [AlphaSense×Tegus](https://www.prnewswire.com/news-releases/alphasense-completes-acquisition-of-tegus-302190934.html) · [Sentieo/AlphaSense](https://www.integrity-research.com/alphasense-merges-with-sentieo/) · [Daloopa API](https://daloopa.com/products/api) · [TradeMap Pro](https://trademap.com.br/planos/trademap-pro) · [Dados de Mercado](https://www.dadosdemercado.com.br/) · [Investing.com BBAS3 consensus](https://www.investing.com/equities/brasil-on-consensus-estimates) · [PEAD (Wikipedia)](https://en.wikipedia.org/wiki/Post%E2%80%93earnings-announcement_drift) · [Quantpedia PEAD](https://quantpedia.com/strategies/post-earnings-announcement-effect) · [Diether-Malloy-Scherbina 2002](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=472362) · [Barber et al. 2001](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00336) · [Whisper number (Wikipedia)](https://en.wikipedia.org/wiki/Whisper_number) · [PEAD.txt (CMU)](https://www.cmu.edu/tepper/accounting-lab/publications/pead.txt-post-earnings-announcement-drift-using-textpage.html)

**Keywords:** analyst estimates, sell-side data, consensus estimates (consenso de analistas), price targets (preço-alvo), earnings surprise (surpresa de resultados), I/B/E/S, FactSet Estimates, Visible Alpha, S&P Capital IQ, Zacks, TipRanks, Estimize, whisper numbers, Earnings Whispers, Finnhub, Financial Modeling Prep, Benzinga ratings, Koyfin, MarketBeat, AlphaSense, Tegus, Daloopa, Intrinio, post-earnings-announcement drift (PEAD), SUE, estimate revisions (revisões de estimativas), forecast dispersion, point-in-time data, look-ahead bias, recomendações de analistas, TradeMap, B3, Brasil.
