# Stock Screeners & Research Platforms

> A dense, fact-checked map of the tools that screen, value, and research public equities — general-purpose screeners, fundamental-data terminals, quant/factor backtesters, charting, and the Brazil 🇧🇷 stack — with each entry tagged free/paid and, crucially, whether it exposes an **API or MCP** you can wire into ML pipelines and agents. Verified live (June 2026); maintenance and data-provenance caveats called out honestly.

This page complements the repo's data-API coverage (yfinance, Polygon, ORATS/Deribit), backtesting, options-market prediction, B3 options, and microstructure pages — it does not repeat them. Here the focus is the *research/screening front end*: where humans (and increasingly agents) discover, filter, and value stocks.

---

## 1. General-purpose screeners (Screeners de uso geral)

Consumer/retail-grade screeners and research portals. Most are great for humans but weak for ML: their data is licensed from third parties (S&P Capital IQ, Morningstar, FactSet) and **redistribution / programmatic access is usually contractually blocked**. Treat anything without an official API as scrape-at-your-own-risk for ML.

| Tool | Free? | API for ML? | Notes |
|---|---|---|---|
| **Finviz** ([finviz.com](https://finviz.com/)) | Free + Elite (~$299.50/yr, often discounted to ~$199.50) | No official API; Elite has **CSV export** of screener results | Fast US screener; famous heatmaps/maps. 15-min delayed on free. |
| **TradingView** ([tradingview.com](https://www.tradingview.com/)) | Free + paid tiers | No public market-data API; **Pine Script** for in-platform strategies; charting widgets/embeds | Best-in-class charting + global screener + social. |
| **Stock Rover** ([stockrover.com](https://stockrover.com/)) | Free + Premium/Premium Plus/Ultimate/Ultimate Pro | No public API; deep CSV/export | 800+ metrics, strong screening/portfolio research for US/Canada. |
| **Zacks** ([zacks.com](https://www.zacks.com/)) | Free + Zacks Premium | No public API | Home of the **Zacks Rank** estimate-revision factor. |
| **Finbox** ([finbox.com](https://finbox.com/)) | Free + paid | Spreadsheet add-ins (Excel/Sheets); paid data plans | Modeling + screening across ~6 major markets. |
| **Simply Wall St** ([simplywall.st](https://simplywall.st/)) | Free + paid | No public API | "Snowflake" visual analysis; global + Brazil coverage. |
| **TipRanks** ([tipranks.com](https://www.tipranks.com/)) | Free + Premium | No public retail API | Aggregates analyst/insider/blogger signals. |
| **GuruFocus** ([gurufocus.com](https://www.gurufocus.com/)) | Mostly paid (~$499/yr) | Add-ins; limited API on higher tiers | Value-investing focus; guru 13F tracking, long fundamentals. |
| **Stockopedia** ([stockopedia.com](https://www.stockopedia.com/)) | Paid | No public API | **StockRank** (Quality/Value/Momentum), Piotroski F-Score, Altman Z. |
| **Stock Analysis** ([stockanalysis.com](https://stockanalysis.com/)) | Free + Pro | **No API** — their FAQ states they license data and lack redistribution rights | Clean, fast fundamentals/quotes; do not plan an ML pipeline on it. |
| **Yahoo Finance** ([finance.yahoo.com](https://finance.yahoo.com/)) | Free | No supported public API (the old one was retired); community wrappers like `yfinance` scrape it | Ubiquitous; fragile for production ML. |
| **Google Finance** ([google.com/finance](https://www.google.com/finance/)) | Free | No API; `GOOGLEFINANCE()` only inside Google Sheets | Quotes + portfolios; thin fundamentals. |

> ⚠️ **Atom Finance** is effectively **defunct** for consumers — it discontinued its consumer terminal, was absorbed by Togal.AI (2024), and its site/status page have been down since 2025. Do not rely on it; it is listed here only to flag stale references.

---

## 2. Fundamental-data terminals & research platforms (Terminais de dados fundamentalistas)

"Bloomberg-lite" web terminals and filings/data infrastructure. This tier is where ML-relevant **APIs and MCP servers** are emerging fastest.

| Platform | Free? | API / MCP for ML? | Notes |
|---|---|---|---|
| **Koyfin** ([koyfin.com](https://www.koyfin.com/)) | Free + paid | **No public API.** Fundamentals/estimates are **licensed from S&P Capital IQ** (fund data from Morningstar; FRED/Trading Economics for macro) → redistribution-limited | Excellent custom dashboards/screeners; data-provenance limits ML use. |
| **TIKR** ([tikr.com](https://tikr.com/)) | Free tier + paid | **No public API.** Financials **powered by S&P Global Capital IQ** | 100k+ global tickers, 13F/insider portfolios, valuation models. |
| **Fiscal.ai** ([fiscal.ai](https://fiscal.ai/)) | Free + paid; **API free trial (≈25 cos / 250 calls/day)** | **Yes — REST API + official MCP server** ([docs.fiscal.ai](https://docs.fiscal.ai/)); also an **official ChatGPT app + OpenAI Codex** integration | **Formerly FinChat.io**; FinChat and Stratosphere.io **merged and rebranded to Fiscal.ai in 2025** ($10M Series A). The single survivor of those three names. |
| **Roic.ai** ([roic.ai](https://www.roic.ai/)) | Free + paid | **Yes — financial-data API** ([roic.ai/api](https://www.roic.ai/api)): statements, ratios, multiples, transcripts | Advertises 30+ yr history; genuine free API tier. |
| **Quartr** ([quartr.com](https://quartr.com/)) | Free app + Pro/API | API on paid tiers | Earnings calls, live transcripts, filings, IR decks. |
| **Daloopa** ([daloopa.com](https://daloopa.com/)) | Paid (enterprise) | **Yes — API + remote MCP server** ([docs.daloopa.com](https://docs.daloopa.com/docs/daloopa-mcp), OAuth, read-only); connectors for ChatGPT, Claude, M365 Copilot | Source-linked, QA'd fundamentals/KPIs across 5,500+ tickers — built for LLMs/agents. |
| **BamSEC** ([bamsec.com](https://www.bamsec.com/)) | Free + paid | No public API | Clean reader over SEC filings + transcripts; table/ownership tools. |
| **SEC EDGAR** ([sec.gov/edgar](https://www.sec.gov/edgar/search/)) | **Free** | **Yes — official REST/XBRL APIs** at `data.sec.gov` (`/api/xbrl/companyfacts`, `/companyconcept`, `/frames`, `/submissions`) + **full-text search**; docs: [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | **The authoritative, free, ML-grade US fundamentals source.** No key; ~10 req/s; User-Agent required. |
| **Macrotrends** ([macrotrends.net](https://www.macrotrends.net/)) | Free | No official API | Long-history charts of fundamentals/ratios; good for eyeballing trends. |
| **Wisesheets** ([wisesheets.io](https://wisesheets.io/)) | Paid (trial) | Excel/Google Sheets **add-on** (functions, not a REST API) | Fundamentals, estimates, prices, dividends straight into a spreadsheet. |
| **Morningstar** ([morningstar.com](https://www.morningstar.com/)) | Free + Investor; enterprise data via Morningstar Direct/PitchBook | Enterprise APIs (paid, gated) | Funds/ratings authority; "economic moat" framework. |

---

## 3. Quant / factor screeners & backtesters (Screeners quant e backtesters)

Where screening crosses into ML: factor research, AI-driven stock ranking, no-code strategy automation, and full backtesting engines.

| Platform | Free? | API / OSS for ML? | Notes |
|---|---|---|---|
| **Portfolio123** ([portfolio123.com](https://www.portfolio123.com/)) | Paid (trial) | **Yes — REST API** + Python SDK **`p123api-py`** ([github.com/portfolio-123/p123api-py](https://github.com/portfolio-123/p123api-py), MIT, Python; **low activity**, ~40 commits/10★ — usable but lightly maintained). **AI Factor** module trains ML models (LightGBM, ExtraTrees, etc.) on 1000+ pre-processed features; `aifactor_predict()` returns predictions as DataFrames | Strongest individual-quant factor/ML platform; clean point-in-time data. |
| **Composer** ([composer.trade](https://www.composer.trade/)) | Paid | API for strategy automation | No-code "symphonies" (algorithmic strategies). **Acquired by SoFi; relaunched as "Composer by SoFi" (June 23, 2026)** with AI-assisted, natural-language strategy building. |
| **QuantConnect** ([quantconnect.com](https://www.quantconnect.com/)) | Free tier + paid | **Yes — open-source LEAN engine** ([github.com/QuantConnect/Lean](https://github.com/QuantConnect/Lean), C#/Python, ~20k★, **actively maintained**) + cloud API + `lean` CLI; data marketplace | Full research→backtest→live pipeline; 20+ brokerage integrations. |
| **Fiscal.ai API** (see §2) | — | REST + MCP | Listed again here because its API/MCP make it the easiest way to feed *clean fundamentals* into a quant/agent loop. |

---

## 4. Visualization & charting (Visualização e gráficos)

| Tool | Free? | Notes |
|---|---|---|
| **TradingView** ([tradingview.com](https://www.tradingview.com/)) | Free + paid | The charting standard; Pine Script, alerts, embeddable widgets. |
| **Koyfin** ([koyfin.com](https://www.koyfin.com/)) | Free + paid | Dashboard-style cross-asset charts, yield curves, macro overlays. |
| **Finviz maps** ([finviz.com/map.ashx](https://finviz.com/map.ashx)) | Free | The canonical market/sector heatmap. |

---

## 5. Brazil 🇧🇷 (B3) — research & data stack

Consumer BR research portals are excellent for humans but **none expose an official public API** — community scrapers exist but are fragile and ToS-gray. For ML/agents, the two clean paths are **brapi.dev** (developer REST API) and **Mais Retorno** (REST + native MCP).

| Tool | Free? | API / MCP for ML? | Notes |
|---|---|---|---|
| **Status Invest** ([statusinvest.com.br](https://statusinvest.com.br/)) | Free + paid | **No official API** (community scrapers only) | Most popular BR fundamentals/dividends portal; ações, FIIs, FI-Infra. |
| **Investidor10** ([investidor10.com.br](https://investidor10.com.br/)) | Free + paid | **No official API** | Indicators, dividends, portfolio manager. |
| **Fundamentus** ([fundamentus.com.br](https://www.fundamentus.com.br/)) | Free | **No official API** (widely scraped) | The classic minimalist B3 fundamentals table. |
| **Oceans14** ([oceans14.com.br](https://www.oceans14.com.br/)) | Free | **No official API** | Historical results, dividends, fundamentals for ações/FIIs. |
| **TradeMap** ([trademap.com.br](https://trademap.com.br/)) | Free + paid | No public API | App-first market monitoring/news. |
| **Comdinheiro** ([comdinheiro.com.br](https://www.comdinheiro.com.br/)) | Paid | Has an integration/API layer for clients | Pro-grade analytics/reporting used by advisors and funds. |
| **Mais Retorno** ([maisretorno.com](https://maisretorno.com/)) | Free + paid | **Yes — REST API + native MCP** ([maisretorno.com/mcp](https://maisretorno.com/mcp)); works with **any MCP-compatible client**; **Free plan: 500 req/mo, 1 yr history**, D‑1 EOD, 15 req/s | Cleanest B3 path for agents/LLMs; funds, ações, FIIs, CVM/ANBIMA. |
| **brapi.dev** ([brapi.dev](https://brapi.dev/)) | Free + paid | **Yes — developer REST API** (Free plan ≈ **15,000 req/mo**); Python SDK ([brapi.dev/docs/sdks/python](https://brapi.dev/docs/sdks/python)) | Aggregates B3 OHLCV, fundamentals, dividends, crypto, macro (Selic/IPCA/CDI). The go-to free B3 API. |
| **Simply Wall St (BR)** ([simplywall.st](https://simplywall.st/)) | Free + paid | No public API | Same platform as §1, with B3 coverage. |

---

## 6. Decision guide (Guia de decisão)

| If you need… | Use |
|---|---|
| Fast free US visual screening | **Finviz** (+ heatmaps), **TradingView** screener |
| Deep human research dashboards | **Koyfin**, **TIKR**, **Stock Rover** |
| **Clean fundamentals via API/MCP** for ML/agents | **SEC EDGAR** (free, authoritative US), **Fiscal.ai**, **Daloopa**, **Roic.ai** |
| Factor/ML stock ranking + backtest | **Portfolio123 (AI Factor)**, **QuantConnect (LEAN)** |
| No-code strategy automation | **Composer by SoFi** |
| Value-investing screens | **GuruFocus**, **Stockopedia**, **Simply Wall St** |
| Earnings calls / transcripts | **Quartr**, **BamSEC**, **Fiscal.ai** |
| **B3 data for ML/agents** 🇧🇷 | **brapi.dev** (free REST), **Mais Retorno** (REST + MCP) |
| B3 human research 🇧🇷 | **Status Invest**, **Investidor10**, **Fundamentus**, **Oceans14** |

> **ML/agent rule of thumb:** if a platform has no official API, assume you cannot legally redistribute its data — most license it from S&P Capital IQ / Morningstar / FactSet. Build pipelines on **first-party, redistribution-friendly sources**: SEC EDGAR (US, free), and for Brazil **brapi.dev** / **Mais Retorno MCP**. Use the licensed terminals (Koyfin, TIKR) for human research, not as silent data feeds.

---

**Sources:** finviz.com · tradingview.com · stockrover.com · zacks.com · finbox.com · simplywall.st · tipranks.com · gurufocus.com · stockopedia.com · stockanalysis.com (FAQ: no API) · finance.yahoo.com · google.com/finance · koyfin.com (+ Koyfin FAQ: data from S&P Capital IQ) · tikr.com (powered by S&P Global Capital IQ) · fiscal.ai (FinChat+Stratosphere merger/rebrand, $10M Series A; docs.fiscal.ai MCP) · roic.ai/api · quartr.com · daloopa.com (docs.daloopa.com MCP) · bamsec.com · sec.gov/edgar + data.sec.gov APIs · macrotrends.net · wisesheets.io · morningstar.com · portfolio123.com + github.com/portfolio-123/p123api-py · composer.trade (SoFi acquisition, "Composer by SoFi" launch 23 Jun 2026) · quantconnect.com + github.com/QuantConnect/Lean · statusinvest.com.br · investidor10.com.br · fundamentus.com.br · oceans14.com.br · trademap.com.br · comdinheiro.com.br · maisretorno.com/mcp · brapi.dev. Atom Finance flagged defunct (consumer terminal discontinued; absorbed by Togal.AI 2024; site offline 2025).

**Keywords:** stock screener, screener de ações, research platform, plataforma de pesquisa, fundamental data, dados fundamentalistas, Finviz, TradingView, Stock Rover, Zacks Rank, Finbox, Simply Wall St, TipRanks, GuruFocus, Stockopedia, stockanalysis.com, Koyfin, TIKR, Fiscal.ai, FinChat, Stratosphere, Roic.ai, Quartr, Daloopa, BamSEC, SEC EDGAR, data.sec.gov, XBRL, company facts, full-text search, Macrotrends, Wisesheets, Morningstar, economic moat, Portfolio123, AI Factor, p123api, LightGBM, Composer, Composer by SoFi, QuantConnect, LEAN, backtesting, factor investing, MCP, Model Context Protocol, API, machine learning, quant, Status Invest, Investidor10, Fundamentus, Oceans14, TradeMap, Comdinheiro, Mais Retorno, brapi.dev, B3, Bovespa, ações, FIIs, dividendos, indicadores fundamentalistas, Piotroski F-Score, Altman Z-Score, StockRank, valuation, screening, heatmap, mapa de mercado
