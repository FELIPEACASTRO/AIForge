# Scholarly Search Engines & Aggregators for Football Research

> A verified, query-ready map of scholarly search engines and cross-publisher aggregators for mining **football (soccer) match & betting prediction** research (*previsão de partidas e apostas de futebol*), with concrete search URLs, live APIs, and real example papers. Every platform and paper below was confirmed against a live endpoint in 2026. **Research & education only.**

## ⚠️ Responsible gambling / Jogo Responsável (read first)

This page is for **research and education**, not betting tips. Peer-reviewed evidence is blunt: bookmaker and exchange markets are **highly efficient**, published edges are small, fragile, and usually vanish after fees, margin (*vig/overround*), and market movement; **most bettors lose money over time**. Academic "accuracy" (e.g. beating a naive baseline) is *not* the same as long-run profit after the bookmaker margin. Treat every model as a hypothesis, back-test out-of-sample, and never stake money you cannot lose.

- **GambleAware / BeGambleAware** — <https://www.gambleaware.org> · **GamCare** (UK, free helpline) — <https://www.gamcare.org.uk>
- 🇧🇷 **Jogo Responsável** — apostar é entretenimento adulto (18+), nunca renda. Ajuda: **Jogadores Anônimos Brasil** <https://jogadoresanonimos.com.br> · SUS **CAPS** para transtorno de jogo (*Ludopatia/Transtorno do Jogo*).

---

## Why aggregators (not just one database)?

Football-prediction work is scattered across CS venues (KDD, ECML/PKDD), statistics/sports-science journals, and preprint servers. No single index has everything, and coverage/recall differs sharply. A robust literature survey **queries several aggregators, then deduplicates by DOI and resolves open-access (OA) copies**. Use CS-focused indexes (DBLP, Semantic Scholar) for methods, broad indexes (OpenAlex, BASE, CORE) for recall, and Crossref + Unpaywall as the DOI/OA backbone.

**Which engine for which job:**
- **Method / model papers (GNNs, xG, ML):** DBLP → Semantic Scholar → arXiv-linked Hugging Face Papers.
- **Maximum recall / systematic review:** OpenAlex + BASE + CORE (harvest, then dedupe).
- **Betting / odds / market-efficiency economics:** Crossref bibliographic search + Google Scholar snowballing.
- **Injury & player-availability features:** Europe PMC + PubMed (E-utilities).
- **Legal full-text PDFs of paywalled work:** Unpaywall → CORE → Internet Archive Scholar.
- **LatAm / Portuguese scholarship:** SciELO Preprints + SciELO (also mirrored in OpenAlex/BASE).

## Aggregator quick-reference

| Engine | Football query / API endpoint | Free? | API | Link |
|---|---|:--:|:--:|---|
| **Semantic Scholar** | `api.semanticscholar.org/graph/v1/paper/search?query=soccer+match+prediction&fields=title,year,externalIds,openAccessPdf` | ✅ | ✅ (key optional) | [semanticscholar.org](https://www.semanticscholar.org) |
| **OpenAlex** | `api.openalex.org/works?search=soccer%20match%20prediction` · topic filter `filter=primary_topic.id:...` | ✅ | ✅ (no key) | [openalex.org](https://openalex.org) |
| **Google Scholar** | `scholar.google.com/scholar?q=%22soccer%22+%22match+prediction%22&as_ylo=2024` | ✅ | ❌ (no official API) | [scholar.google.com](https://scholar.google.com) |
| **CORE** | `api.core.ac.uk/v3/search/works?q=soccer%20match%20prediction` | ✅* | ✅ (key) | [core.ac.uk](https://core.ac.uk) |
| **BASE** | `base-search.net/Search/Results?lookfor=soccer+match+prediction` | ✅ | ✅ (`api.base-search.net`) | [base-search.net](https://www.base-search.net) |
| **Dimensions** | `app.dimensions.ai/discover/publication?search_text=soccer%20match%20prediction` | ✅* | 💲 (paid API) | [app.dimensions.ai](https://app.dimensions.ai) |
| **Lens.org** | `lens.org/lens/search/scholar/list?q=soccer%20match%20prediction` | ✅* | 💲 (paid API) | [lens.org](https://www.lens.org) |
| **DBLP** | `dblp.org/search/publ/api?q=soccer+prediction&format=json` | ✅ | ✅ (no key) | [dblp.org](https://dblp.org) |
| **Hugging Face Papers** | `huggingface.co/papers?q=soccer` | ✅ | ❌ | [huggingface.co/papers](https://huggingface.co/papers) |
| **Papers with Code** | ⚠️ sunset 2025 → redirects to HF | — | — | see note below |
| **Europe PMC** | `ebi.ac.uk/europepmc/webservices/rest/search?query=soccer+injury+prediction+machine+learning&format=json` | ✅ | ✅ (no key) | [europepmc.org](https://europepmc.org) |
| **PubMed** | `pubmed.ncbi.nlm.nih.gov/?term=soccer+injury+prediction+machine+learning` | ✅ | ✅ (E-utilities) | [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov) |
| **Internet Archive Scholar** | `scholar.archive.org/search?q=soccer+match+prediction` | ✅ | ✅ (fatcat) | [scholar.archive.org](https://scholar.archive.org) |
| **Crossref** | `api.crossref.org/works?query=football+betting+prediction&mailto=you@example.org` | ✅ | ✅ (no key) | [search.crossref.org](https://search.crossref.org) |
| **Unpaywall** | `api.unpaywall.org/v2/{DOI}?email=you@example.org` | ✅ | ✅ (email) | [unpaywall.org](https://unpaywall.org) |

\* Free web/interactive search; bulk/programmatic API is paid or key-gated (CORE key is free for research). 💲 = commercial API tier.

---

## Platform notes + concrete football queries

### Semantic Scholar (S2)
CS/AI-strong index (~200M+ papers) with the free **Academic Graph API**. Live example — the query above returned the KDD 2025 paper *"Player-Team Heterogeneous Interaction Graph Transformer for Soccer Outcome Prediction"* (S2 `paperId 7bd6af85fa10073a349e62e6c76b4566282bbd5e`): <https://www.semanticscholar.org/paper/7bd6af85fa10073a349e62e6c76b4566282bbd5e>. The API is aggressively rate-limited without a free key (anonymous calls readily return HTTP 429). Ask for `openAccessPdf` and `externalIds` to get direct PDFs and DOIs.

### OpenAlex
Fully open successor to Microsoft Academic Graph: **250M+ works, no key required** (add `?mailto=you@example.org` for the polite pool). The same KDD paper resolves as work `W4412877180` with `primary_topic = "Sports Analytics and Performance"` — filter by that topic to harvest the whole sub-field: `api.openalex.org/works?filter=primary_topic.search:sports%20analytics`. Also supports `title.search:`, `abstract.search:`, and `from_publication_date:` filters. (Anonymous `search=` can be throttled under heavy load — the API returns a "temporarily rate-limited" notice and suggests a free key; DOI/ID look-ups stay reliable.)

### Google Scholar
Highest recall, but **no official API** and heavy bot-blocking. Use operators: `intitle:soccer intitle:prediction`, `allintitle:`, `author:"..."`, `source:` (journal), and the year slider `&as_ylo=2024`. Best for citation-chasing ("Cited by") and grey literature; export to a reference manager, then dedupe elsewhere.

### CORE
World's largest OA aggregator — **400M+ metadata records / 40M+ hosted full texts** (2025), harvested from **15,000+ data providers**. **CORE API v3** (`api.core.ac.uk/v3/search/works`) needs a free research key and returns full-text links. Ideal for finding legal OA PDFs of paywalled football papers.

### BASE (Bielefeld Academic Search Engine)
**400M+ documents** from **10,000+ content providers** (OAI-PMH), ~60% OA. Strong for theses/dissertations and institutional-repository preprints often missing elsewhere. HTTP/OAI interface at `api.base-search.net`.

### Dimensions & Lens.org
Both offer **free interactive web search** across publications, citations, grants, patents and clinical trials (registration for full features); programmatic/bulk access is a **paid** tier. Good for landscape/altmetric views and linking football-analytics papers to funders and patents.

### DBLP
Curated **computer-science bibliography** — the fastest way to find method papers and author profiles at CS venues. The JSON API `dblp.org/search/publ/api?q=soccer+prediction&format=json` returned 60+ hits (64 at time of writing) including KDD, ECML/PKDD (MLSA workshop) and arXiv/CoRR entries, most with a DOI. Use it to build author-centric reading lists.

### Papers with Code — ⚠️ sunset
Meta **shut down Papers with Code on 24 July 2025**; `paperswithcode.com` now redirects to Hugging Face and the SOTA leaderboards are gone. Historical dumps remain on GitHub (`github.com/paperswithcode/paperswithcode-data`). For live code+paper discovery, use **Hugging Face Papers** instead.

### Hugging Face Papers
Trending/searchable arXiv-linked papers with attached models, datasets and demos. `huggingface.co/papers?q=soccer` surfaces football work such as *"MatchTime: Towards Automatic Soccer Game Commentary Generation"* (arXiv `2406.18530`, page `huggingface.co/papers/2406.18530`). Pair with `huggingface.co/datasets?search=soccer` for training data.

### Europe PMC / PubMed
Biomedical coverage — critical for **injury & availability modeling** (*disponibilidade de atletas*), a real feature source for match prediction. The Europe PMC REST query above returned ~600 hits (598 at time of writing), e.g. *"From classical models to attention-based transformers: A comparative study of injury prediction pipelines in female varsity soccer"* (J Biomech 2026, DOI `10.1016/j.jbiomech.2026.113278`) and *"Machine learning applications in sport: a scoping review"* (Front Psychol 2026, DOI `10.3389/fpsyg.2026.1802549`).

### Internet Archive Scholar
Full-text search over **35M+ preserved papers** (incl. long-tail/OA and at-risk journals), backed by the fatcat catalog. Good as a fallback to recover PDFs of dead links. (Public endpoint rate-limits aggressive scraping.)

### Crossref + Unpaywall (the DOI/OA backbone)
**Crossref** is the registration agency behind most DOIs — the canonical source of truth for dedup. `api.crossref.org/works?query=football+betting+prediction` returns betting-specific papers with DOIs (see table below). **Unpaywall** then maps any DOI to a legal OA copy: e.g. DOI `10.3390/app10010046` → `is_oa: true, oa_status: gold`, published PDF at mdpi.com.

---

## Verified example papers (found via these engines)

| Paper (year) | Where / venue | Identifier | Found via |
|---|---|---|---|
| Player-Team Heterogeneous Interaction Graph Transformer for Soccer Outcome Prediction (2025) | KDD '25 | DOI [10.1145/3711896.3737082](https://doi.org/10.1145/3711896.3737082) · arXiv [2507.10626](https://arxiv.org/abs/2507.10626) | DBLP, OpenAlex, S2 |
| Next-Event Prediction in Soccer: Assessing the Impact of Team and Player Information (MLSA 2025 / proc. 2026) | MLSA @ ECML/PKDD (Porto, Sep 2025; CCIS proceedings) | DOI [10.1007/978-3-032-15165-0_5](https://doi.org/10.1007/978-3-032-15165-0_5) | DBLP |
| Prediction-based evaluation of back-four defense with spatial control in soccer (2025) | arXiv CoRR | arXiv [2511.06191](https://arxiv.org/abs/2511.06191) | DBLP |
| Machine Learning in Football Betting: Prediction of Match Results Based on Player Characteristics (2019) | Applied Sciences (MDPI) | DOI [10.3390/app10010046](https://doi.org/10.3390/app10010046) · **OA gold** | Crossref → Unpaywall |
| Prediction of Football Match Outcomes Based on Bookmaker Odds by k-NN (2018) | Int. J. Machine Learning & Computing | DOI [10.18178/ijmlc.2018.8.1.658](https://doi.org/10.18178/ijmlc.2018.8.1.658) | Crossref |
| The Evolution of Football Betting: A Machine Learning Approach to Match Outcome Forecasting and Bookmaker Odds Estimation (2025) | Smart Innovation, Systems & Technologies | DOI [10.1007/978-981-96-6254-8_9](https://doi.org/10.1007/978-981-96-6254-8_9) | Crossref |
| Soccer match outcome prediction with random forest and gradient boosting models (2024) | Applied & Computational Eng. | DOI [10.54254/2755-2721/40/20230634](https://doi.org/10.54254/2755-2721/40/20230634) | Crossref |
| MatchTime: Towards Automatic Soccer Game Commentary Generation (2024) | arXiv / EMNLP 2024 | arXiv [2406.18530](https://arxiv.org/abs/2406.18530) | HF Papers |

> Inclusion = "confirmed to exist," **not** an endorsement of the method's profitability. Odds-based papers in particular often report accuracy without accounting for the bookmaker margin.

## A reproducible literature pipeline

1. **Search wide** — run the *same* query across OpenAlex + Semantic Scholar + DBLP + CORE (methods & recall) and Europe PMC (injury/availability). Prefer JSON APIs so results are machine-readable.
2. **Normalize & dedupe by DOI** — collapse duplicates using the **Crossref** DOI as the primary key (`api.crossref.org/works?query.bibliographic=<title>`); merge arXiv IDs and repository copies onto the same record.
3. **Resolve open access** — for each DOI, call **Unpaywall** (`api.unpaywall.org/v2/{DOI}?email=`) and fall back to **CORE** / **BASE** / **Internet Archive Scholar** for a legal full-text PDF.
4. **Snowball** — expand via Semantic Scholar references/citations and Google Scholar "Cited by"; re-run steps 2-3.
5. **Screen critically** — for betting claims, demand out-of-sample back-tests, closing-odds baselines, and profit *after margin/commission*. Log everything with `mailto`/`email` params to stay in the polite API pools.

Copy-paste starters (all tested live; add your own `email`/`mailto`):

```bash
# OpenAlex – title search, JSON
curl 'https://api.openalex.org/works?filter=title.search:soccer%20prediction&per_page=25&mailto=you@example.org'
# DBLP – CS venues (KDD, ECML/PKDD, arXiv), JSON
curl 'https://dblp.org/search/publ/api?q=soccer+prediction&format=json&h=30'
# Crossref – betting/odds papers, select minimal fields
curl 'https://api.crossref.org/works?query=football+betting+odds+prediction&rows=25&mailto=you@example.org'
# Europe PMC – injury/availability features
curl 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=soccer+injury+prediction+machine+learning&format=json'
# Unpaywall – DOI -> legal OA PDF (dedup key = DOI)
curl 'https://api.unpaywall.org/v2/10.3390/app10010046?email=you@example.org'
```

## 🇧🇷 Brazil & LatAm note (English body / termos em português)
For regional and Portuguese-language scholarship, add **SciELO Preprints** (`preprints.scielo.org/index.php/scielo/search?query=futebol`, live/free) and **SciELO** proper to the pipeline; both are also indexed by OpenAlex, BASE and CORE. Useful PT keywords: *previsão de resultados de futebol*, *modelo de gols (Poisson)*, *mercado de apostas esportivas*, *odds justas*, *disponibilidade de jogadores*. Regulated sports betting in Brazil (*apostas de quota fixa*, Lei 14.790/2023) is 18+ — this material is academic, not a betting service.

---

**Sources:** [OpenAlex](https://openalex.org) · [OpenAlex API](https://api.openalex.org/works?search=soccer%20prediction) · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/search?query=soccer+match+prediction) · [DBLP API](https://dblp.org/search/publ/api?q=soccer+prediction&format=json) · [Crossref API](https://api.crossref.org/works?query=football+betting+prediction) · [Unpaywall](https://unpaywall.org) · [CORE](https://core.ac.uk/about) · [BASE](https://www.base-search.net) · [Dimensions](https://app.dimensions.ai) · [Lens.org](https://www.lens.org) · [Hugging Face Papers](https://huggingface.co/papers?q=soccer) · [Europe PMC API](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=soccer%20injury%20prediction%20machine%20learning&format=json) · [Internet Archive Scholar](https://scholar.archive.org) · [SciELO Preprints](https://preprints.scielo.org) · [Papers with Code sunset](https://github.com/paperswithcode/paperswithcode-data/issues/116) · [GambleAware](https://www.gambleaware.org) · [GamCare](https://www.gamcare.org.uk)

**Keywords:** football/soccer match prediction, betting prediction, bookmaker odds, expected goals (xG), Poisson model, scholarly aggregators, OpenAlex, Semantic Scholar API, DBLP, CORE, BASE, Crossref DOI, Unpaywall open access, Europe PMC, literature review pipeline, responsible gambling · *previsão de partidas de futebol, apostas esportivas, odds, modelo de Poisson, agregadores acadêmicos, acesso aberto, revisão de literatura, jogo responsável, SciELO Preprints*
