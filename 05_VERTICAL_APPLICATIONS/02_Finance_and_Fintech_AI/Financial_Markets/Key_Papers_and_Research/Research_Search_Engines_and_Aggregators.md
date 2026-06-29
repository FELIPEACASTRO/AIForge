# Research Search Engines & Aggregators for ML-Finance

> Scholarly search engines and metadata aggregators to discover machine-learning + financial-markets research (equities, ETFs, B3/Brazil, US markets, options/derivatives, quant), with their public APIs, concrete query strings, and a reproducible literature-mining pipeline.

This page indexes the *discovery layer* for the AIForge ML/finance index: the search engines and open metadata APIs you use to find papers, code, benchmarks, and datasets on financial markets. It is deliberately API-first. Every platform below exposes a programmatic interface, so you can move from "I read a paper" to "I have a deduplicated, DOI-keyed, full-text-resolved corpus" without manual copy-paste. Audience is Brazil-heavy, so Portuguese/Brazil sources (SciELO, SBFin/RBFin, B3, CVM) are flagged where relevant.

---

## 1. Platform comparison at a glance

| Platform | Best for | Open API? | Auth needed | Cost | Notes |
|---|---|---|---|---|---|
| [OpenAlex](https://openalex.org/) | Broad coverage, filtering, citation graph | Yes (REST) | No (email in `mailto=` for polite pool) | Free | ~250M+ works; successor to Microsoft Academic Graph |
| [Semantic Scholar](https://www.semanticscholar.org/) | AI/CS papers, embeddings, recommendations | Yes (Academic Graph API) | Optional API key (higher limits) | Free | SPECTER2 embeddings, TLDRs, influential citations |
| [Crossref](https://www.crossref.org/) | DOI metadata, deduplication backbone | Yes (REST) | No (polite pool via `mailto`) | Free | ~180M+ records; canonical metadata of record |
| [Papers with Code](https://paperswithcode.com/) | SOTA leaderboards, code links, benchmarks | Archived only | n/a | Free | **Sunset Jul 2025 by Meta; now redirects to HF Papers** |
| [Hugging Face Papers](https://huggingface.co/papers) | Trending papers ↔ code/models/datasets | Yes (HF Hub API) | Optional token | Free | Successor to Papers with Code (Meta + HF, Jul 2025) |
| [DBLP](https://dblp.org/) | CS venue completeness (NeurIPS, ICML, KDD…) | Yes (search API) | No | Free | CC0 metadata; best for venue/author lookups |
| [CORE](https://core.ac.uk/) | Open-access full text aggregation | Yes (API v3) | API key (free) | Free | ~300M+ records, ~40M+ full texts from repositories |
| [BASE](https://www.base-search.net/) | Repository/OA discovery | Yes (interface, registration) | Registration for API | Free | Bielefeld; ~400M+ docs, strong on grey literature |
| [The Lens](https://www.lens.org/) | Scholarly + patent linkage | Yes (paid scholarly API) | API token | Freemium | Unique for patent↔paper (fintech IP) |
| [Dimensions](https://www.dimensions.ai/) | Grants, citations, policy links | Yes (paid; free for some) | Yes | Freemium | Free web search; API mostly commercial |
| [Google Scholar](https://scholar.google.com/) | Recall, grey literature, citation counts | **No official API** | — | Free | Use sparingly; no ToS-compliant bulk API |
| [Internet Archive Scholar](https://scholar.archive.org/) | Preserved/long-tail OA fulltext | Search UI + IA APIs | No | Free | Fatcat catalog; recovers "dead" PDFs |
| [arxiv-sanity-lite](https://arxiv-sanity-lite.com/) | Personalized arXiv recs (tfidf+SVM) | Self-host (open source) | — | Free | Karpathy; tag papers → similar-paper recs |
| [alphaXiv](https://www.alphaxiv.org/) | Discussion + AI reading over arXiv | Search/AI features | Account for posting | Free | Swap `arxiv.org`→`alphaxiv.org` on any preprint |
| [Unpaywall](https://unpaywall.org/) | OA fulltext resolution by DOI | Yes (REST) | Email param | Free | The "last mile" of any pipeline (100k calls/day) |

---

## 2. The big open APIs (with finance query strings)

### 2.1 OpenAlex — `https://api.openalex.org`

The single best free starting point: huge coverage, no key required, rich filters. Add your email as `mailto=` to enter the faster "polite pool".

- Docs: https://developers.openalex.org/ (the old `docs.openalex.org` now 301-redirects here) · Works reference: https://developers.openalex.org/api-reference/works
- Base entity: `/works` (also `/authors`, `/sources`, `/institutions`, `/topics`).
- Full-text + title + abstract search via `search=`; field-scoped search via `filter=title.search:...`.

Example finance queries (paste in a browser or `curl`):

```
# Deep RL trading
https://api.openalex.org/works?search=deep%20reinforcement%20learning%20trading&per-page=25&mailto=you@example.com

# Limit order book modeling, 2022+, sort by citations
https://api.openalex.org/works?search=limit%20order%20book&filter=from_publication_date:2022-01-01&sort=cited_by_count:desc&mailto=you@example.com

# Option pricing with neural networks (title-scoped, boolean)
https://api.openalex.org/works?filter=title.search:(option%20pricing%20AND%20neural%20network)&mailto=you@example.com

# Brazil-affiliated quant-finance works (institution country = BR)
https://api.openalex.org/works?search=stock%20market%20machine%20learning&filter=institutions.country_code:BR&mailto=you@example.com
```

Tips: boolean operators `AND`/`OR`/`NOT` must be uppercase; wrap phrases in quotes. The whole request URL is capped (OpenAlex documents a ~2,048-4,096-character limit; over it you get HTTP 414), so chunk large `OR` lists and union IDs client-side. Basic `page=` paging stops at 10,000 results — use `cursor=*` for stable deep pagination, and `select=id,doi,title,publication_year` to trim payloads.

### 2.2 Semantic Scholar — Academic Graph API

Strongest for AI/CS/quant-ML: TLDR summaries, `influentialCitationCount`, SPECTER2 embeddings, and a paper-recommendations endpoint.

- Docs hub: https://www.semanticscholar.org/product/api · Swagger: https://api.semanticscholar.org/api-docs/
- Base: `https://api.semanticscholar.org/graph/v1`
- Get a free key (request form linked from the API page) for dedicated rate limits; unauthenticated traffic shares a global pool.

```
# Keyword search with selected fields
https://api.semanticscholar.org/graph/v1/paper/search?query=deep+reinforcement+learning+trading&fields=title,year,abstract,citationCount,externalIds&limit=25

# Bulk search (up to 1000/page, supports boolean + year filter)
https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=limit+order+book+deep+learning&year=2020-2026&fields=title,year,externalIds

# Single paper by DOI, with references/citations
https://api.semanticscholar.org/graph/v1/paper/DOI:10.1093/rfs/hhaa009?fields=title,abstract,citationCount,references.title

# "More like this" recommendations
https://api.semanticscholar.org/recommendations/v1/papers/forpaper/ARXIV:1911.10107?fields=title,year
```

Pass the key via header `x-api-key: <KEY>`. `externalIds` gives you DOI/ArXiv/DBLP IDs in one shot — the join keys for deduplication.

### 2.3 Crossref — `https://api.crossref.org`

The metadata-of-record and your deduplication backbone (one DOI = one canonical record). Free, no key; send `mailto=` to use the polite pool.

- Docs: https://www.crossref.org/documentation/retrieve-metadata/rest-api/ · Swagger: https://api.crossref.org/

```
# Bibliographic free-text query
https://api.crossref.org/works?query.bibliographic=neural+network+option+pricing&rows=20&mailto=you@example.com

# Filter by date window + type
https://api.crossref.org/works?query=algorithmic+trading&filter=from-pub-date:2023-01-01,type:journal-article&mailto=you@example.com

# Resolve a single DOI to full metadata
https://api.crossref.org/works/10.1111/jofi.12852
```

Same-filter repeats = OR; different filters = AND. Use `cursor=*` for deep paging.

### 2.4 DBLP — `https://dblp.org/search`

Authoritative for CS venues (NeurIPS, ICML, ICLR, KDD, AAAI, SIGMOD) and exact author disambiguation. CC0 data, no key.

- Publication search (XML/JSON): `https://dblp.org/search/publ/api`
- Author / venue search: `https://dblp.org/search/author/api`, `https://dblp.org/search/venue/api`

```
# JSON publication search
https://dblp.org/search/publ/api?q=reinforcement+learning+portfolio&format=json&h=30

# Find papers from a specific venue token
https://dblp.org/search/publ/api?q=limit+order+book+stream:conf/icaif&format=json
```

Note the finance-AI venue **ICAIF** (ACM Intl. Conf. on AI in Finance) is well indexed here; `h=` sets hits, `f=` sets the offset.

### 2.5 CORE — `https://api.core.ac.uk/v3`

Largest aggregator of open-access *full text* (harvested from institutional repositories worldwide, including Brazilian ones). Free API key required.

- Docs: https://api.core.ac.uk/docs/v3 · Service: https://core.ac.uk/services/api

```
# Full-text + metadata search (POST or GET /search/works)
https://api.core.ac.uk/v3/search/works?q=stock%20price%20prediction%20LSTM&limit=25
# Header: Authorization: Bearer <YOUR_CORE_KEY>
```

CORE exposes `fullText` and `downloadUrl` fields — useful when a paper is OA but Crossref/Unpaywall don't have a clean link. Throughput is modest (single-request bursts per 10s), so batch politely.

### 2.6 Unpaywall — `https://api.unpaywall.org/v2`

The "last mile": given a DOI, return the best legal open-access PDF. Free, just append `email=`.

- Docs: https://unpaywall.org/products/api

```
# Best OA location for a DOI
https://api.unpaywall.org/v2/10.1093/rfs/hhaa009?email=you@example.com
```

Read `best_oa_location.url_for_pdf`. Limit: 100,000 calls/day.

---

## 3. SOTA, benchmarks & code linkage

### Papers with Code → Hugging Face Papers (status note)

**Important (current as of 2025-2026):** Meta **sunset Papers with Code on 24-25 July 2025**, and `paperswithcode.com` now **redirects to Hugging Face**. The historical leaderboard data (benchmark tables, paper↔code links, datasets) is preserved read-only:

- Archived dataset dump: https://github.com/paperswithcode/paperswithcode-data
- Legacy API docs (for the archived schema): https://paperswithcode.com/api/v1/docs/ (base was `https://paperswithcode.com/api/v1/`, endpoints `/papers/`, `/tasks/`, `/datasets/`, with `q=`, `page`, `items_per_page`)
- Python client (works against archived data): https://github.com/paperswithcode/paperswithcode-client

Successor for live SOTA/trending + code/model/dataset linkage:

- **Hugging Face Papers** — https://huggingface.co/papers (Meta + Hugging Face, announced Jul 2025). Query trending and search; each paper links to code, models, and datasets on the Hub. Programmatic access via the HF Hub API (`https://huggingface.co/api/...`) and the `huggingface_hub` Python library.
- Community alternative archiving SOTA tables: **CodeSOTA** — https://www.codesota.com/papers-with-code

Finance-relevant tasks that lived on Papers with Code (still useful as search terms): `stock-market-prediction`, `stock-price-prediction`, `algorithmic-trading`, `portfolio-optimization`, `time-series-forecasting`. Search the archived dump or HF Papers with these.

---

## 4. Personalized arXiv discovery (q-fin and beyond)

| Tool | URL | What it does |
|---|---|---|
| arxiv-sanity-lite | https://arxiv-sanity-lite.com/ · code: https://github.com/karpathy/arxiv-sanity-lite | Tag papers of interest; get tfidf+SVM recommendations and daily email digests. Self-hostable; point it at `q-fin.*` and `cs.LG`. |
| alphaXiv | https://www.alphaxiv.org/ | Open discussion + AI reading over any arXiv preprint. Replace `arxiv.org/abs/XXXX` with `alphaxiv.org/abs/XXXX`. Adds search + "AI Ask" Q&A over papers (seed funding Nov 2025). |
| arXiv itself | https://arxiv.org/list/q-fin/recent · API: https://info.arxiv.org/help/api/ | Primary preprint source. Query the q-fin categories directly (see §6). |

arXiv API example for quant finance:

```
http://export.arxiv.org/api/query?search_query=cat:q-fin.TR+AND+all:reinforcement+learning&start=0&max_results=25
```

arXiv has nine q-fin subcategories: `q-fin.TR` (Trading & Market Microstructure), `q-fin.PR` (Pricing of Securities), `q-fin.CP` (Computational Finance), `q-fin.PM` (Portfolio Management), `q-fin.ST` (Statistical Finance), `q-fin.RM` (Risk Management), `q-fin.MF` (Mathematical Finance), `q-fin.GN` (General Finance), and `q-fin.EC` (Economics).

---

## 5. Brazil / Portuguese-language discovery 🇧🇷

| Source | URL | Use |
|---|---|---|
| SciELO Brasil | https://www.scielo.br/ | OA journals; search Portuguese/English finance articles (e.g. *Revista Contabilidade & Finanças*). |
| SciELO Preprints | https://preprints.scielo.org/ | Latin-American preprints incl. economics/finance; DOI-minted, OAI-PMH harvestable. |
| RBFin / SBFin | https://rbfin.sbfin.org.br/ · https://www.sbfin.org.br/ | *Brazilian Review of Finance* (open access, no fees) — the SBFin journal; flagship Brazilian quant-finance venue. |
| B3 (exchange) | https://www.b3.com.br/ · market data: https://www.b3.com.br/en_us/market-data-and-indices/ | Primary source for Brazilian equities/ETFs/derivatives definitions, index methodologies (Ibovespa), and historical series. |
| CVM | https://www.gov.br/cvm/ · open data: https://dados.cvm.gov.br/ | Brazilian SEC; regulatory filings and **CVM open-data portal** (funds, issuers) for empirical datasets. |

Tip for OpenAlex/CORE: filter to Brazil with `institutions.country_code:BR` (OpenAlex) and harvest SciELO/repository full text via CORE or OAI-PMH endpoints. RBFin and SciELO articles carry DOIs, so they slot directly into the DOI-keyed pipeline below.

---

## 6. Building a finance literature pipeline

A reproducible recipe to turn fuzzy topics into a clean, full-text corpus:

1. **Cast a wide net (recall).** Fire the same query at **OpenAlex** (`/works?search=`), **Semantic Scholar** (`/paper/search/bulk`), and **arXiv** (`cat:q-fin.* AND all:<terms>`). Pull DBLP for any CS-venue gaps (ICAIF, KDD, NeurIPS workshops). Keep title + abstract + all `externalIds`.
2. **Normalize to DOIs (dedupe).** Collapse records on DOI. For DOI-less arXiv preprints, key on the arXiv ID; backfill a DOI later if the work was published. Crossref `query.bibliographic=` resolves messy titles → canonical DOI when an ID is missing.
3. **Enrich metadata.** Hit **Crossref** `/works/{DOI}` for authoritative venue, date, authors, references; hit **Semantic Scholar** for `citationCount`, `influentialCitationCount`, and TLDR.
4. **Resolve full text (last mile).** For each DOI call **Unpaywall** `/v2/{DOI}?email=` → `best_oa_location.url_for_pdf`. Fall back to **CORE** `/v3/search/works` (`downloadUrl`/`fullText`) and **Internet Archive Scholar** (https://scholar.archive.org/) for preserved/dead PDFs.
5. **Rank & expand.** Sort by citations or recency; expand the frontier with Semantic Scholar recommendations (`/recommendations/v1/papers/forpaper/...`) and arxiv-sanity-lite tags.
6. **Be polite.** Send `mailto=`/`email=` on every OpenAlex/Crossref/Unpaywall call, cache responses (15-min+), back off on HTTP 429, and respect each platform's rate limits (Semantic Scholar/CORE keys raise yours). Never scrape Google Scholar — it has no ToS-compliant bulk API; use it only for spot citation checks.

Join keys that make this work: **DOI** (Crossref/Unpaywall/OpenAlex), **arXiv ID** (arXiv/Semantic Scholar/alphaXiv), **DBLP key** (CS venues), and **Semantic Scholar `externalIds`** (which carries all of the above in one object).

---

## 7. Representative finance/ML resources discoverable via these platforms

These are well-known, real works you can locate with the queries above (search by title on OpenAlex/Semantic Scholar to get the live record, DOI, and OA link):

| Resource | How to find it | Why it matters |
|---|---|---|
| López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018) | OpenAlex/Crossref title search | Canonical reference for ML in trading; meta-labeling, purged CV. |
| Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine Learning" (*Rev. Financial Studies*, 2020, DOI:10.1093/rfs/hhaa009) | Crossref `/works/10.1093/rfs/hhaa009` | Landmark ML cross-section-of-returns study. |
| Sirignano & Cont, "Universal features of price formation in financial markets: perspectives from deep learning" (*Quantitative Finance*, 2019, DOI:10.1080/14697688.2019.1622295) | Semantic Scholar title search | Foundational deep-learning LOB result. |
| Deng et al., "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading" (IEEE TNNLS vol. 28, 2017; online 2016) | DBLP / OpenAlex | Early influential deep-RL trading paper. |
| FinRL (Liu et al., 2020; arXiv:2011.09607, NeurIPS Deep RL Workshop) deep-RL-for-finance library | arXiv (`q-fin`/`cs.LG`) + HF Papers; code at github.com/AI4Finance-Foundation/FinRL | Open framework + reproducible RL trading benchmarks. |
| FinBERT (Araci, 2019; arXiv:1908.10063) and FinGPT (https://fingpt.io/) | HF Papers / Semantic Scholar | Domain LLMs for financial NLP; widely benchmarked. |
| Hull, *Options, Futures, and Other Derivatives* | OpenAlex/Crossref | Standard derivatives reference underpinning option-pricing ML work. |
| ICAIF proceedings (ACM Intl. Conf. on AI in Finance) | DBLP venue search `stream:conf/icaif` | The flagship AI-in-finance venue; survey papers and benchmarks. |
| RBFin articles (SBFin) | https://rbfin.sbfin.org.br/ | Representative Brazilian quant-finance empirical work. |

(Confirm exact DOIs/years on the live record before citing — metadata is authoritative at the platform, not memorized here.)

---

## 8. Quick-start cheat sheet

```bash
# One topic across three engines (recall), then resolve OA fulltext
TOPIC="deep reinforcement learning trading"
curl "https://api.openalex.org/works?search=${TOPIC// /%20}&per-page=25&mailto=you@example.com"
curl "https://api.semanticscholar.org/graph/v1/paper/search?query=${TOPIC// /+}&fields=title,year,externalIds,citationCount&limit=25"
curl "http://export.arxiv.org/api/query?search_query=cat:q-fin.TR+AND+all:${TOPIC// /+}&max_results=25"
# Then per DOI:
curl "https://api.unpaywall.org/v2/<DOI>?email=you@example.com"
```

---

**Keywords:** research search engines, scholarly aggregators, OpenAlex API, Semantic Scholar Academic Graph API, Crossref REST API, Papers with Code, Hugging Face Papers, DBLP, CORE API, Unpaywall, BASE, The Lens, Dimensions, Internet Archive Scholar, alphaXiv, arxiv-sanity, q-fin, ML finance, deep reinforcement learning trading, limit order book, option pricing neural network, quantitative finance, literature mining, DOI deduplication; *motores de busca acadêmica, agregadores de pesquisa, finanças quantitativas, aprendizado de máquina em finanças, mercado de ações, B3, CVM, SciELO Preprints, SBFin, RBFin, precificação de opções, derivativos, aprendizado por reforço para trading, mineração de literatura, pipeline de pesquisa.*
