# News & Events Data APIs for Markets

> Authoritative, current (2024-2026) catalog of **financial news, sentiment, earnings-call transcripts, regulatory-filing feeds, and global event databases** for NLP / event-driven ML — the unstructured-data layer for alpha, risk, and LLM agents. Heavy on REST endpoints, free-vs-paid status, point-in-time/look-ahead honesty, and **ML/redistribution licensing** (the part most catalogs skip). Brazil 🇧🇷 sources flagged. Every vendor below was confirmed to exist and be live as of 2026-06.

This page complements the generic data-API pages (yfinance / Polygon / FRED basics) and `Macro_and_Economic_Data.md`. Here the unit is **text + events**, not OHLCV: headlines, body text, entity tagging, sentiment scores, earnings transcripts, SEC/CVM filings, and machine-coded world events. The deciding factors for ML are rarely "does it have news" but rather: **historical depth, point-in-time integrity (was the timestamp the *publish* time or the *crawl* time?), entity-resolution quality, and whether the license lets you train/redistribute.**

---

## 0. How to choose (read this first)

| Need | Pick | Why |
|---|---|---|
| Free experimentation / student | **GDELT**, **Finnhub** free, **Marketaux** free, **Tickertick**, **SEC EDGAR** | $0, real APIs, but small windows or attribution required |
| Ticker-tagged sentiment, cheap | **Marketaux**, **EODHD**, **Alpha Vantage** | entity + score in one call |
| Backtest needing deep, time-stamped history | **Tiingo** (1990s), **RavenPack/Bigdata** (1998+), **sec-api.io** (1993) | PIT-grade archives |
| Institutional alpha / risk | **RavenPack (Bigdata.com)**, **LSEG MarketPsych**, **Bloomberg**, **AlphaSense** | quant-ready analytics, sales-led |
| Earnings calls as data | **Quartr**, **FMP**, **EarningsCall API**, **API Ninjas** | transcripts + speaker tags |
| Real-time event detection | **Dataminr**, **Benzinga**, **GDELT 15-min** | low-latency breaking signals |

> **Look-ahead trap #1:** many "historical news" feeds stamp articles with the *time you crawled them*, not the *original publish time*. For event studies you must verify the timestamp semantics (publish vs index vs revised). RavenPack, LSEG MarketPsych, and Tiingo document point-in-time; most cheap APIs do not.
> **Look-ahead trap #2:** sentiment models are frequently **retrained**, so a vendor's *current* score for a 2019 article is not what was emitted in 2019. Ask for the score-as-of, or freeze the model.

---

## 1. Affordable / developer news + sentiment APIs

The retail-and-quant sweet spot: REST JSON, ticker/entity tagging, sentiment polarity, and (sometimes) a usable free tier.

| Source | Content | Coverage / depth | Free / Paid | API / how | Link |
|---|---|---|---|---|---|
| **Marketaux** | Articles tagged with stock/crypto/FX **entities** + sentiment score (−1…+1); "top entities by sentiment" rollups | 5,000+ sources, 30+ languages, 80+ markets, 200k+ entities | **Free** 100 req/day (3 art./req); tiered paid plans add higher limits, deeper history & extra filters (annual = 2 months free) — see pricing | REST; filter by symbol/exchange/industry/country | [marketaux.com/documentation](https://www.marketaux.com/documentation) · [pricing](https://www.marketaux.com/pricing) |
| **Finnhub** — Company News / Market News | Company & market headlines; **News Sentiment** (buzz, bullish/bearish %, news score) | Global; sentiment endpoint **US-only** | **Free** 60 calls/min (company/market news); **News-Sentiment is premium** | REST `/company-news`, `/news`, `/news-sentiment`; WebSocket | [finnhub.io/docs/api/company-news](https://finnhub.io/docs/api/company-news) · [news-sentiment](https://finnhub.io/docs/api/news-sentiment) |
| **Alpha Vantage** — `NEWS_SENTIMENT` | Live + historical articles with **per-ticker sentiment** (−0.35…0.35 bands) and topic tags | US + crypto; "Alpha Intelligence" suite | Free **25 req/day** (premium endpoints); paid tiers raise limits; **commercial use needs separate license** | REST `function=NEWS_SENTIMENT&tickers=...&topics=...` | [alphavantage.co/documentation#news-sentiment](https://www.alphavantage.co/documentation/#news-sentiment) |
| **EODHD** (EOD Historical Data) | Financial News API (full body by ticker/tag) + **daily sentiment** for stocks/ETFs/FX/crypto (−1…1, news+social) | 60–70+ exchanges, 150k+ tickers | Paid (news in higher plans); broad-coverage value play | REST news + separate Sentiment API; Python lib | [eodhd.com/financial-apis/stock-market-financial-news-api](https://eodhd.com/financial-apis/stock-market-financial-news-api) |
| **Financial Modeling Prep (FMP)** | Stock News, **General News**, **Press Releases**, Search-news, grade/analyst news | Stocks/crypto/FX, real-time | Free tier (limited) → paid; 100+ endpoints | REST + WebSocket + bulk CSV | [site.financialmodelingprep.com/developer/docs/stable/stock-news](https://site.financialmodelingprep.com/developer/docs/stable/stock-news) |
| **Tiingo News** | Institutional news feed across equities/ETFs/mutual funds/FX/crypto | **70M+ articles, history to the 1990s** (institutional, 1-yr commit); 3-month window for non-institutional | Paid (add-on); strong PIT depth | REST; Python `tiingo` | [tiingo.com/products/news-api](https://www.tiingo.com/products/news-api) · [docs](https://www.tiingo.com/documentation/news) |
| **Massive** (formerly **Polygon.io**, rebrand completed Oct-2025; `api.polygon.io` still parallel) Ticker News | Ticker-linked articles with **Insights** (per-ticker sentiment + reasoning); Benzinga partner endpoints (Analyst Insights, Consensus Ratings) | US equities; real-time | Paid (news in stock plans); free dev key limited | REST `/v2/reference/news` (now on `api.massive.com`); MCP server for agents | [massive.com/docs/rest/stocks/news](https://massive.com/docs/rest/stocks/news) · [Benzinga analyst insights](https://massive.com/docs/rest/partners/benzinga/analyst-insights) |
| **Stock News API** (stocknewsapi.com) | Articles + video, sentiment per item (−1.5…+1.5), upgrades/downgrades | CNBC, Zacks, Bloomberg, etc. | Paid; 5-day/100-call free trial | REST JSON GET | [stocknewsapi.com/documentation](https://stocknewsapi.com/documentation) |
| **finlight.me** | Real-time + historical financial/geopolitical news, full article body, **per-article sentiment + confidence** | Reuters/WSJ/Guardian/NYT etc.; **history to 2007** | Paid (Pro 50k req/mo); free trial | REST + **WebSocket + webhooks** | [finlight.me](https://finlight.me/) |
| **Tickertick** | Stock-news query language (and/or/diff operators), ~10k US tickers + startups | ~10k source sites | **Free** (commercial OK), 10 req/min/IP | REST; Python `pytickertick` | [github.com/hczhu/TickerTick-API](https://github.com/hczhu/TickerTick-API) |

> 💡 For supervised NLP, **Marketaux / EODHD / Alpha Vantage** give you `(text, ticker, sentiment)` triples cheaply, but the **sentiment is the vendor's label**, not ground truth — treat it as a noisy teacher, not gold.

---

## 2. General-purpose news APIs (not finance-specific, but heavily used for it)

Broad web/news crawlers. You bring the finance entity-tagging and NLP; they bring the firehose. Watch the **licensing** — several forbid storing/redistributing article bodies.

| Source | Content | Coverage | Free / Paid | License note (ML/redistribution) | Link |
|---|---|---|---|---|---|
| **NewsAPI.org** | Headlines + snippets; search & top-headlines | 150k+ sources | **Free dev-only**; production needs **Business ~$449/mo** | Free tier = non-commercial/dev only; articles delayed | [newsapi.org/pricing](https://newsapi.org/pricing) |
| **NewsCatcher** | Real-time + historical news, NLP enrich, dedup | Global, multilingual | Paid from **~$29/mo**; no free plan | Commercial license available; enrichment add-ons | [newscatcherapi.com](https://www.newscatcherapi.com/) |
| **NewsData.io** | Real-time + archive, finance category, NLP tags | 88k+ sources, 206 countries, 80+ langs | **Free 200 credits/day, commercial allowed** (12-h delay, headline/snippet) | Generous for a free tier; full content + sentiment on paid | [newsdata.io/free-news-api](https://newsdata.io/free-news-api) |
| **Mediastack** | Live + historical headlines | 7,500 sources | Free tier (HTTP only); paid for HTTPS/history | **No NLP/sentiment/entity**; thin metadata | [mediastack.com](https://mediastack.com/) |
| **NewsAPI.ai** (ex-**Event Registry**) | Event-clustered news, entities, 5,000+ topics, sentiment | Global, multilingual | Paid (token-based); trial | Event clustering useful for event-study dedup | [newsapi.ai/plans](https://newsapi.ai/plans) |
| **APITube** | Real-time news, sentiment + entity + topic on all plans, story clustering | Global, multilingual | Freemium (more generous free tier) | Separate vendor from NewsAPI.ai; simpler pricing | [apitube.io](https://apitube.io/) |

> ⚠️ "Number of sources" is a marketing metric. For finance, **entity resolution accuracy** (does it map "Apple" the fruit vs AAPL?) matters far more. Generic APIs usually do this worse than Marketaux/RavenPack.

---

## 3. Global event databases (free, machine-coded world events)

For macro/event-driven research and geopolitical risk, not company headlines.

| Source | Content | Coverage / cadence | Free / Paid | API / how | Link |
|---|---|---|---|---|---|
| **GDELT 2.0 — Events** | Machine-coded events (actors, CAMEO action, geo, **Goldstein scale**, avg tone) | World, 100+ langs, **updates every 15 min**; 1979→ in BigQuery | **Free** | BigQuery (`gdelt-bq.gdeltv2`), raw CSV, web tools | [gdeltproject.org/data.html](https://www.gdeltproject.org/data.html) |
| **GDELT 2.0 — GKG** (Global Knowledge Graph) | People/orgs/themes/emotions/quotes/images per article | World, 65 translated langs, 15-min | **Free** | BigQuery + CSV | [blog.gdeltproject.org](https://blog.gdeltproject.org/) |
| **GDELT DOC 2.0 API** | Full-text article search over **rolling 3-month** window; **TimelineTone**, ArtList, tonechart, word cloud | World | **Free**, no key | `https://api.gdeltproject.org/api/v2/doc/doc` (`mode=`, `tone>`/`tone<` filters); Python `gdelt-doc-api` | [DOC 2.0 debut](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) · [client](https://github.com/alex9smith/gdelt-doc-api) |
| **GDELT Context / GEO / TV 2.0** | Contextual snippets, geographic, US TV (Internet Archive) coverage | World / US TV | **Free** | REST + BigQuery | [gdeltproject.org](https://www.gdeltproject.org/) |

> GDELT is the best **free** unstructured-events source in existence, but it is *noisy*: CAMEO mis-coding, duplicate counting, and tone computed by a fixed lexicon (not a modern model). Use mentions/event counts as features, validate heavily.

---

## 4. Earnings-call transcripts (text + speaker structure)

The single richest periodic corpus per company. Free-to-read ≠ free-to-API.

| Source | Content | Coverage / depth | Free / Paid | API / how | Link |
|---|---|---|---|---|---|
| **Quartr API** | Raw + edited transcripts (JSON), **speaker ID** (name/role/affiliation, new transcripts since ~Apr-2025), chapters, summaries, live audio | Global public co's; events beyond earnings | Enterprise / custom (paying = unlimited calls) | REST datasets: transcripts, chapters | [quartr.com/products/quartr-api](https://quartr.com/products/quartr-api) · [docs](https://quartr.com/docs/datasets/earnings-call-transcripts) |
| **FMP transcripts** | Full earnings-call transcripts, transcript dates by symbol, transcript list | Broad US + intl | Paid (in FMP plans) | REST `earning-call-transcript`, `earning_call_transcript`, list/dates | [site.financialmodelingprep.com/datasets/earnings-call-transcripts](https://site.financialmodelingprep.com/datasets/earnings-call-transcripts) |
| **EarningsCall API** (earningscalls.dev) | Word-for-word transcripts + speaker segments (CEO/CFO/Analyst), full-text search; free to read on web | US-listed (NYSE/NASDAQ) | Web **free to read**; API has a free tier (50 req/mo, previews) → paid plans | REST; MCP support | [earningscalls.dev](https://earningscalls.dev/) |
| **API Ninjas — Earnings Call Transcript** | Transcript text by ticker/quarter | History **from 2005**; Dev tier = last 5 yrs, full archive on Business/Pro | Freemium | REST `/v1/earningstranscript` | [api-ninjas.com/api/earningscalltranscript](https://api-ninjas.com/api/earningscalltranscript) |
| **Alpha Vantage `EARNINGS_CALL_TRANSCRIPT`** | Transcript text via Alpha Intelligence | US | Free 25 req/day / premium | REST `function=EARNINGS_CALL_TRANSCRIPT` | [alphavantage.co/documentation](https://www.alphavantage.co/documentation/) |
| 🇧🇷 **Quartr / TIKR / company IR** | BR large-caps (PETR4, VALE3, ITUB4…) hold English + PT calls; transcripts via Quartr or IR sites | B3 blue chips | Mixed | Quartr API or scrape IR/RI pages | [quartr.com](https://quartr.com/) |

> For research at scale, several **free transcript datasets** exist on HuggingFace/Kaggle (e.g., MAEC, earnings-call corpora) — see `HuggingFace_Finance_Datasets_and_Models.md`. Mind that scraped corpora often **lack reliable timestamps** (look-ahead risk for trading studies).

---

## 5. Regulatory filing feeds (events as primary documents)

The most authoritative, **timestamped, point-in-time** event source. US is free and excellent; map to local regulators elsewhere.

| Source | Content | Coverage | Free / Paid | API / how | Link |
|---|---|---|---|---|---|
| **SEC EDGAR — data.sec.gov** | `submissions/CIK##########.json` (full filing history), **XBRL company-facts/company-concept** JSON | All US registrants | **Free**, no key (fair-access UA + ≤10 req/s) | REST JSON, real-time (<1 s after dissemination) | [sec.gov/search-filings/edgar-application-programming-interfaces](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) |
| **EDGAR Full-Text Search (EFTS)** | Keyword search across filing bodies (2001→) | US filings | **Free**, no key (≤10 req/s) | `https://efts.sec.gov/LATEST/search-index?q=...` (JSON); UI at `sec.gov/edgar/search` | [EDGAR full-text search](https://www.sec.gov/edgar/search/) |
| **EDGAR RSS / index feeds** | Latest-filings RSS; daily/quarterly **index** in html/xml/**json** | US | **Free** | RSS by company/type; `full-index/` | [sec.gov/about/developer-resources](https://www.sec.gov/about/developer-resources) |
| **sec-api.io** | Query API, full-text, **real-time WebSocket stream (~300 ms)**, XBRL-to-JSON, 10-K/Q/8-K section extractors, **Form 3/4/5 insider**, 13F, 13D/G, 144, S-1/424B4, N-PORT/N-PX | 400+ form types, **back to 1993** | Paid (free trial) | REST + WebSocket; Python/Node SDK | [sec-api.io](https://sec-api.io/) · [docs](https://sec-api.io/docs) |
| **Kaleidoscope (kscope.io)** | SEC, 13F, insider data API | US | Paid | REST | [api.kscope.io](https://api.kscope.io/) |
| 🇧🇷 **CVM RAD / ENET** | **Fato Relevante**, comunicados, DFP/ITR, formulário de referência | All BR listed co's | **Free** | Bulk open data + ENET consulta; no clean REST (scrape/portal) | [rad.cvm.gov.br](https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx) · [CVM dados abertos](https://dados.cvm.gov.br/) |
| 🇧🇷 **B3 Plantão de Notícias / developers.b3** | Market news (last 365 days), CVM-doc generation APIs | B3 | Free (portal) / API access | Portal + B3 dev APIs | [developers.b3.com.br/apis](https://developers.b3.com.br/apis) |
| 🇧🇷 **brapi.dev** | BR fundamentals + CVM-derived data; lightweight | B3 | Freemium | REST | [brapi.dev](https://brapi.dev/) |

> 🇧🇷 The CVM equivalent of an "8-K" is the **Fato Relevante** (material fact) and **Comunicado ao Mercado**. They are free and timestamped via CVM/B3 but there is **no first-class REST news API** — you parse the RAD/ENET portal or the `dados.cvm.gov.br` open-data dumps. This is the BR look-ahead-safe event source.

---

## 6. Institutional / enterprise news analytics (sales-led, quant-grade)

Where serious systematic alpha and risk teams shop. Pricing is **sales-led** (expect 5–6 figures/yr); these are listed for completeness and because their *academic/eval* footprints matter for ML benchmarking.

| Source | Content | Coverage / PIT | Free / Paid | API / how | Link |
|---|---|---|---|---|---|
| **RavenPack** → **Bigdata.com** | News analytics (sentiment, relevance, novelty, event taxonomy), knowledge graph, **agentic LLM platform (MCP, Search API)** over news + transcripts + filings | 40,000+ premium sources; **company PIT to 2000** (RavenPack), ESG-style depth | Paid (institutional) | REST + Python SDK; **MCP** integration | [ravenpack.com](https://www.ravenpack.com/) · [bigdata.com/for-developers](https://bigdata.com/for-developers) |
| **LSEG MarketPsych Analytics** (ex-Refinitiv/Thomson Reuters) | Sentiment + theme/emotion scores (Joy, Anger, Inflation, EarningsExpectations…), risks; **ESG analytics PIT to 1998**, 100k+ companies | Equities/FX/commodities/countries, real-time | Paid | **Pull API** (JSON/CSV by asset+dates); feeds | [lseg.com/.../marketpsych-analytics](https://www.lseg.com/en/data-analytics/financial-data/analytics/marketpsych-analytics) |
| **Bloomberg** News + Social Analytics | Story-level sentiment, news heat, social velocity; Event-Driven Feeds (machine-readable releases) | Global | Paid (Terminal / B-PIPE / Enterprise) | BLPAPI / Data License | [bloomberg.com/professional](https://www.bloomberg.com/professional/) |
| **AlphaSense** (+ **Tegus**, acq. 2024) | AI search over earnings calls, broker research, filings, **100k+ expert-call transcripts**; doc-level sentiment | 4,000+ public co's, 50+ sectors | Paid (enterprise) | Platform + some API/integrations | [alpha-sense.com](https://www.alpha-sense.com/) · [Tegus close](https://www.prnewswire.com/news-releases/alphasense-completes-acquisition-of-tegus-302190934.html) |
| **Accern** | No-code generative NLP; theme + sentiment from news/filings, ESG/credit/crypto risk | Broad | Paid; AWS Marketplace | Platform + API | [accern.com](https://www.accern.com/) · [docs](https://documentation.accern.com/) |
| **StockGeist.ai** | Social + news **market-sentiment** signal, 2,000+ US tickers | US-centric | **Free 10k REST credits/mo** (+5 streams); paid tiers | REST | [stockgeist.ai/stock-market-api](https://www.stockgeist.ai/stock-market-api/) |
| **Benzinga** (data/APIs) | Real-time news feed (ticker/category/tags: earnings, M&A, analyst), **Newsquantified** analytics, **StockSnips** ML sentiment signal, calendars, webhook engine | US + crypto | Paid (free trial varies) | REST + **webhook push**; on QuantConnect | [benzinga.com/apis/data](https://www.benzinga.com/apis/data/) · [docs](https://docs.benzinga.com/) |
| **Dataminr — First Alert / Pulse** | Real-time **event detection** from 1M+ public sources (text/image/video/sensor), 150+ langs, 220+ countries | Global breaking events | Paid (enterprise); some free humanitarian access | Platform + integrations (ArcGIS, AWS) | [dataminr.com/products/first-alert](https://www.dataminr.com/products/first-alert/) |

> The MarketPsych and RavenPack **point-in-time** guarantee (score-as-of-publication, frozen model versions) is precisely what the cheap APIs lack and what makes them defensible for backtests — it's most of what you pay for.

---

## 7. Licensing & ML-use checklist (do not skip)

| Question | Why it matters | Typical answer by tier |
|---|---|---|
| Can I **store** article bodies? | Many publishers license headline+snippet only | Cheap APIs: snippet OK, body often not; enterprise: negotiated |
| Can I **train models** on the text? | Derivative-work / TDM rights vary by jurisdiction | Often allowed for *internal* models; redistribution of text restricted |
| Can I **redistribute** scores/derived signals? | Sentiment scores are the vendor's IP | Usually no without a redistribution license (RavenPack/LSEG explicit) |
| Is the **timestamp** publish-time and immutable? | Look-ahead & survivorship | PIT vendors yes; crawlers often no |
| Is the **sentiment model frozen** per-vintage? | Retraining = silent look-ahead | RavenPack/LSEG version; most APIs serve "latest model" |
| **Survivorship**: are delisted/renamed tickers kept? | Event studies need dead names | Enterprise yes; many APIs map only live symbols |
| **Attribution** required? | Free tiers (NewsAPI.org, GDELT, Marketaux free) | Yes on free; check ToS |

> Rule of thumb: **free/cheap APIs are fine for research, prototyping, and feature engineering on a rolling window**; the moment you go to production *or* publish/sell signals, re-read the ToS — body-text storage and signal redistribution are the two clauses that bite.

---

## 8. Open-source tooling & datasets to glue it together

| Tool / dataset | Use | Free | Link |
|---|---|---|---|
| **FinBERT / FinBERT-tone** | Off-the-shelf financial sentiment scorer (label your own news) | Free | [HuggingFace ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) |
| **`gdelt` (PyPI), `gdeltPyR`** | Pull GDELT events/GKG to DataFrame | Free | [pypi.org/project/gdelt](https://pypi.org/project/gdelt/) |
| **`sec-edgar-downloader`, `edgartools`** | Bulk-download EDGAR filings, parse sections | Free | [github.com/dgunning/edgartools](https://github.com/dgunning/edgartools) |
| **`gdeltdoc` / `gdelt-doc-api` (Python)** | Wrapper for DOC 2.0 full-text/tone (`pip install gdeltdoc`) | Free | [github.com/alex9smith/gdelt-doc-api](https://github.com/alex9smith/gdelt-doc-api) |
| **FNSPID** | 29.7M prices + 15.7M time-aligned news, 4,775 S&P500 co's, 1999–2023 | Free | [huggingface.co/datasets/Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) |
| **MAEC** | Multimodal (text+audio) aligned earnings-call corpus, S&P 1500, CIKM'20 | Free | [github.com/Earnings-Call-Dataset](https://github.com/Earnings-Call-Dataset) |

---

### Quick decision matrix (TL;DR)

- **$0, just exploring** → GDELT + Finnhub free + Marketaux free + SEC EDGAR.
- **Cheap ticker sentiment for a model** → Marketaux (free tier, then low-cost paid) or EODHD or Alpha Vantage.
- **Deep timestamped history for a real backtest** → Tiingo (1990s) or sec-api.io (1993); enterprise → RavenPack/LSEG (1998–2000+, PIT).
- **Earnings calls as text** → Quartr (best structure) / FMP / API Ninjas (cheap).
- **Brazil 🇧🇷** → CVM RAD/ENET + `dados.cvm.gov.br` (Fato Relevante, free, PIT) + B3 Plantão de Notícias + Quartr for transcripts.

**Sources:** [Marketaux docs](https://www.marketaux.com/documentation) · [Marketaux pricing](https://www.marketaux.com/pricing) · [Finnhub company-news](https://finnhub.io/docs/api/company-news) · [Finnhub news-sentiment](https://finnhub.io/docs/api/news-sentiment) · [Alpha Vantage NEWS_SENTIMENT](https://www.alphavantage.co/documentation/#news-sentiment) · [EODHD news API](https://eodhd.com/financial-apis/stock-market-financial-news-api) · [FMP stock-news](https://site.financialmodelingprep.com/developer/docs/stable/stock-news) · [FMP transcripts](https://site.financialmodelingprep.com/datasets/earnings-call-transcripts) · [Tiingo news](https://www.tiingo.com/products/news-api) · [Polygon/Massive news](https://massive.com/docs/rest/stocks/news) · [Massive Benzinga analyst insights](https://massive.com/docs/rest/partners/benzinga/analyst-insights) · [Stock News API](https://stocknewsapi.com/documentation) · [finlight.me](https://finlight.me/) · [Tickertick](https://github.com/hczhu/TickerTick-API) · [NewsAPI.org pricing](https://newsapi.org/pricing) · [NewsCatcher](https://www.newscatcherapi.com/) · [NewsData.io free](https://newsdata.io/free-news-api) · [Mediastack](https://mediastack.com/) · [NewsAPI.ai plans](https://newsapi.ai/plans) · [APITube](https://apitube.io/) · [GDELT data](https://www.gdeltproject.org/data.html) · [GDELT DOC 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) · [gdelt-doc-api client](https://github.com/alex9smith/gdelt-doc-api) · [Quartr API](https://quartr.com/products/quartr-api) · [Quartr docs](https://quartr.com/docs/datasets/earnings-call-transcripts) · [EarningsCall API](https://earningscalls.dev/) · [API Ninjas transcript](https://api-ninjas.com/api/earningscalltranscript) · [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) · [SEC developer resources](https://www.sec.gov/about/developer-resources) · [sec-api.io](https://sec-api.io/) · [kscope.io](https://api.kscope.io/) · [CVM RAD](https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx) · [CVM dados abertos](https://dados.cvm.gov.br/) · [B3 developers](https://developers.b3.com.br/apis) · [brapi.dev](https://brapi.dev/) · [RavenPack](https://www.ravenpack.com/) · [Bigdata.com developers](https://bigdata.com/for-developers) · [LSEG MarketPsych](https://www.lseg.com/en/data-analytics/financial-data/analytics/marketpsych-analytics) · [Bloomberg Professional](https://www.bloomberg.com/professional/) · [AlphaSense](https://www.alpha-sense.com/) · [AlphaSense–Tegus close](https://www.prnewswire.com/news-releases/alphasense-completes-acquisition-of-tegus-302190934.html) · [Accern](https://www.accern.com/) · [StockGeist](https://www.stockgeist.ai/stock-market-api/) · [Benzinga APIs](https://www.benzinga.com/apis/data/) · [Dataminr First Alert](https://www.dataminr.com/products/first-alert/) · [FinBERT](https://huggingface.co/ProsusAI/finbert) · [FNSPID dataset](https://huggingface.co/datasets/Zihan1004/FNSPID) · [MAEC dataset](https://github.com/Earnings-Call-Dataset)

**Keywords:** financial news API, news sentiment API, event-driven ML, earnings call transcripts (transcrições de teleconferência de resultados), SEC EDGAR filings, fato relevante (material fact), GDELT global events, RavenPack Bigdata.com, LSEG MarketPsych, Bloomberg news analytics, AlphaSense Tegus, Benzinga, Marketaux, Finnhub, Alpha Vantage NEWS_SENTIMENT, EODHD, finlight.me, Quartr, point-in-time (ponto no tempo), look-ahead bias (viés de antecipação), redistribution license (licença de redistribuição), NLP financeiro, dados alternativos (alternative data), notícias de mercado, análise de sentimento, CVM, B3 Plantão de Notícias, sentiment scoring, knowledge graph, FinBERT
