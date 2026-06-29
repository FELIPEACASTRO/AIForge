# arXiv q-fin & Preprint Servers for Finance

> Where to find FINANCIAL-MARKETS research (equities, ETFs, B3/US, options & derivatives, quant/ML finance) on arXiv's Quantitative Finance (q-fin) archive and the global preprint ecosystem — with exact category codes, listing URLs, the arXiv API, and Brazil/LatAm servers (SciELO Preprints).

This page is a finding aid. The goal is to get you from "I want recent ML-for-markets work" to a concrete, reproducible URL or API call. Preprint servers are the fastest channel: papers appear here months before (or instead of) journal publication, full text is open, and arXiv numbers are stable, citable identifiers.

---

## 1. arXiv Quantitative Finance (q-fin) — the core archive

- **Archive home:** <https://arxiv.org/archive/q-fin>
- **Recent (rolling ~last 5 days, all subcats):** <https://arxiv.org/list/q-fin/recent>
- **Current month:** <https://arxiv.org/list/q-fin/current>
- **Specific month (YYYY-MM):** e.g. <https://arxiv.org/list/q-fin/2025-12>
- **New submissions today:** <https://arxiv.org/list/q-fin/new>

q-fin launched in December 2008. It is curated and moderated; submissions need endorsement on first use. Most ML-heavy "AI for markets" work is **cross-listed** between q-fin and `cs.LG` / `stat.ML`, so searching only one archive misses papers — always query across categories (see §4).

### 1.1 All q-fin subcategories (exact codes + scope)

| Code | Name | Scope (per arXiv) | Typical markets/AI content |
|------|------|-------------------|----------------------------|
| **q-fin.CP** | Computational Finance | Computational methods — Monte Carlo, PDE, lattice and other numerical methods for financial modeling | Option pricers, deep PDE solvers, neural-network calibration, GPU/ML pricing |
| **q-fin.GN** | General Finance | Development of general quantitative methodologies with applications in finance | Catch-all; fintech, LLMs in finance, sentiment, alt-data |
| **q-fin.MF** | Mathematical Finance | Mathematical/analytical methods — stochastic, probabilistic, functional, algebraic, geometric | Stochastic control, rough volatility, optimal stopping, signatures |
| **q-fin.PM** | Portfolio Management | Security selection & optimization, capital allocation, investment strategies, performance measurement | Mean-variance/RL portfolios, factor models, asset allocation |
| **q-fin.PR** | Pricing of Securities | Valuation & hedging of securities, their derivatives, and structured products | **Options & derivatives**, deep hedging, implied-vol surfaces, exotics |
| **q-fin.RM** | Risk Management | Measurement & management of financial risk in trading, banking, insurance, corporate | VaR/ES, credit risk, ML default models, systemic risk |
| **q-fin.ST** | Statistical Finance | Statistical, econometric and econophysics analysis of financial markets & economic data | Return prediction, stylized facts, volatility forecasting, time-series ML |
| **q-fin.TR** | Trading & Market Microstructure | Microstructure, liquidity, exchange/auction design, automated trading, agent-based modeling, market-making | HFT, limit-order-book ML, execution/RL agents, market impact |
| **q-fin.EC** | Economics | **Alias for `econ.GN`** — micro/macro, international, theory of the firm, labor, other non-finance economics | Empirical/applied economics; overlaps `econ.*` |

> Per-subcategory listing URLs follow the pattern `https://arxiv.org/list/<code>/recent` — e.g. derivatives work at <https://arxiv.org/list/q-fin.PR/recent>, trading/microstructure at <https://arxiv.org/list/q-fin.TR/recent>, statistical finance at <https://arxiv.org/list/q-fin.ST/recent>.

### 1.2 Adjacent archives where ML-for-markets papers actually live

| Code | Archive | Why it matters for markets |
|------|---------|----------------------------|
| **cs.LG** | Machine Learning (CS) | Most deep-learning/RL trading & forecasting papers cross-list here first |
| **stat.ML** | Machine Learning (Stats) | Probabilistic ML, Gaussian processes, time-series models applied to prices |
| **cs.CL** | Computation & Language | Financial NLP, news/earnings-call sentiment, LLM finance agents (e.g. FinGPT, BloombergGPT-style) |
| **cs.CE** | Computational Eng., Finance & Science | Numerical finance, simulation |
| **econ.GN** | General Economics | = q-fin.EC alias; applied economics |
| **econ.EM** | Econometrics | Time-series/panel methods feeding asset pricing |
| **econ.TH** | Theoretic Economics | Market/auction theory |
| **math.OC / math.PR** | Optimization & Control / Probability | Stochastic control, optimal execution, portfolio theory |

A robust market-ML query unions `q-fin.*` with `cs.LG`, `stat.ML`, and `cs.CL`.

---

## 2. arXiv listing URL patterns (no code needed)

| Goal | URL |
|------|-----|
| All q-fin, recent | `https://arxiv.org/list/q-fin/recent` |
| One subcategory, recent | `https://arxiv.org/list/q-fin.TR/recent` |
| A given month | `https://arxiv.org/list/q-fin/2026-01` |
| Page through results | append `?skip=100&show=100` (e.g. `.../recent?skip=100&show=100`) |
| Single paper (abstract) | `https://arxiv.org/abs/2011.09607` |
| Single paper (PDF) | `https://arxiv.org/pdf/2011.09607` |
| Full-text search UI | `https://arxiv.org/search/?searchtype=all&terms-0-field=all&terms-0-term=deep+hedging&classification-q_finance=y` |

The **advanced search** UI (<https://arxiv.org/search/advanced>) lets you tick the *Quantitative Finance* box and add author/title/abstract terms and date ranges — the easiest no-API path.

---

## 3. The arXiv API (programmatic harvesting)

- **Endpoint:** `http://export.arxiv.org/api/query` (HTTPS works too)
- **Manual:** <https://info.arxiv.org/help/api/user-manual.html>
- **Output:** Atom 1.0 XML (title, authors, abstract, categories, DOI, dates)
- **Pagination:** `start` + `max_results` (cap 30,000 per query; page in batches of ≤2,000)
- **Sort:** `sortBy=submittedDate|lastUpdatedDate|relevance`, `sortOrder=ascending|descending`
- **Etiquette:** ≤1 request / 3 s; identify yourself; bulk harvesting should use OAI-PMH instead.

**Example queries (paste into a browser or `curl`):**

```text
# Newest Trading & Market Microstructure papers
http://export.arxiv.org/api/query?search_query=cat:q-fin.TR&sortBy=submittedDate&sortOrder=descending&start=0&max_results=50

# Newest Pricing-of-Securities (options/derivatives) papers
http://export.arxiv.org/api/query?search_query=cat:q-fin.PR&sortBy=submittedDate&sortOrder=descending&max_results=50

# ML-for-markets across q-fin + cs.LG, keyword "reinforcement learning"
http://export.arxiv.org/api/query?search_query=%28cat:q-fin.TR+OR+cat:cs.LG%29+AND+all:%22reinforcement+learning%22&max_results=100

# Author search
http://export.arxiv.org/api/query?search_query=au:Buehler_H+AND+cat:q-fin.PR
```

Query-builder fields: `ti:` (title), `au:` (author), `abs:` (abstract), `cat:` (category), `all:`; combine with `AND`/`OR`/`ANDNOT` (URL-encode spaces as `+` and quotes as `%22`).

- **Bulk / mirroring:** OAI-PMH at `http://export.arxiv.org/oai2` (set `metadataPrefix=arXiv`, `set=q-fin`).
- **Python wrappers:** `arxiv` (<https://pypi.org/project/arxiv/>), `lukasschwab/arxiv.py` (<https://github.com/lukasschwab/arxiv.py>).
- **RSS/Atom feeds:** `http://export.arxiv.org/rss/q-fin.TR` (per-subcategory daily feed for trackers).

---

## 4. Browsing, reading & discovery layers on top of arXiv

| Tool | URL | What it adds | Tip |
|------|-----|--------------|-----|
| **ar5iv** | `https://ar5iv.org/abs/<id>` (or `ar5iv.labs.arxiv.org`) | HTML rendering of any arXiv paper (mobile-friendly, no PDF) | Swap `arxiv.org/abs/2011.09607` → `ar5iv.org/abs/2011.09607` |
| **alphaXiv** | <https://www.alphaxiv.org/> | AI Q&A on papers, comments, related-work; replace `arxiv`→`alphaxiv` in any URL | Good for understanding a dense q-fin proof fast |
| **SciRate** | <https://scirate.com/> | Community "scites" / ranking of arXiv papers, daily feeds per category | Sort q-fin by scites to surface attention |
| **Hugging Face Papers** | <https://huggingface.co/papers> | Daily trending arXiv papers, linked code/models/datasets, search by title/abstract | Best for ML-finance with released code |
| **Connected Papers** | <https://www.connectedpapers.com/> | Citation-graph visualization from any arXiv/DOI seed | Map a subfield (e.g. deep hedging) quickly |
| **Semantic Scholar** | <https://www.semanticscholar.org/> | Citations, influential-citation counts, TLDRs; has an API | Cross-check impact beyond arXiv |
| **arXiv Sanity / replacements** | <https://arxiv-sanity-lite.com/> | Personalized recommendations over arXiv | Tune to q-fin + cs.LG |

For Portuguese-speaking readers, ar5iv + alphaXiv + a translation layer makes dense math-finance preprints far more accessible than raw PDFs.

---

## 5. General preprint & working-paper servers (finance/econ)

| Server | URL | Finance/econ relevance | Notes |
|--------|-----|------------------------|-------|
| **SSRN** | <https://www.ssrn.com/> | The **giant** for finance/econ/law working papers (FEN — Financial Economics Network) | Most empirical asset-pricing & market-microstructure WPs appear here first. **See the dedicated SSRN page in this index.** |
| **RePEc / IDEAS** | <https://ideas.repec.org/> | World's largest economics WP collection; mirrors many arXiv q-fin papers | Author profiles, rankings, MPRA preprints |
| **NBER WPs** | <https://www.nber.org/papers> | Premier US economics/finance working papers | Often paired with SSRN posting (gated abstracts) |
| **OSF Preprints** | <https://osf.io/preprints/> | Multidisciplinary (PKP-based); hosts econ/finance + data/code alongside | ORCID login; DOI on accept |
| **Zenodo** | <https://zenodo.org/> | CERN/OpenAIRE general repository — preprints, datasets, code, slides | Great for B3/CVM-derived datasets + reproducible artifacts; DOI for everything |
| **Research Square** | <https://www.researchsquare.com/> | Multidisciplinary preprints (In Review) | Indexed in Scopus preprint coverage |
| **Authorea (Wiley)** | <https://www.authorea.com/> | Multidisciplinary collaborative preprints | Web-native authoring |
| **Preprints.org (MDPI)** | <https://www.preprints.org/> | Multidisciplinary, incl. business/economics section | Fast DOI, versioning |
| **TechRxiv (IEEE)** | <https://www.techrxiv.org/> | Engineering/tech; fintech, ML-systems, blockchain | Useful for trading-systems/infra papers |
| **EconStor (ZBW)** | <https://www.econstor.eu/> | Open econ/business full-text repository | Strong European coverage |

> Coverage note: Scopus indexes preprints from arXiv, SSRN, TechRxiv, Research Square (and bio/chem/medRxiv) — so a Scopus preprint search is a fast cross-server sweep.

---

## 6. Brazil & Latin America (Portuguese-language resources)

| Resource | URL | Use for markets/finance |
|----------|-----|-------------------------|
| **SciELO Preprints** | <https://preprints.scielo.org/> | LatAm multidisciplinary preprint server (Open Preprint Systems / PKP). Accepts **Applied Social Sciences** incl. economics & finance. ORCID required; DOI + CC-BY on acceptance; PT/EN/ES | Submission guide: <https://preprints.scielo.org/index.php/scielo/about/submissions> |
| **SciELO Brasil** | <https://www.scielo.br/> | Published Brazilian journals (e.g. *RBE*, *RAC*, *BBR*, *Revista Brasileira de Finanças*) | Find peer-reviewed PT-BR finance research |
| **SBFin** | <https://sbfin.org.br/> | Brazilian Finance Society — Brazilian Finance Meeting, *RBFin* journal | Brazil-specific market/asset-pricing scholarship |
| **B3 (exchange)** | <https://www.b3.com.br/> | Market data, indices (Ibovespa, IBrX), options/derivatives specs, microstructure rules | Ground-truth for B3 empirical studies |
| **CVM** | <https://www.gov.br/cvm/> | Brazilian securities regulator — filings, rules, datasets | Disclosure/regulatory data for research |
| **BCB** | <https://www.bcb.gov.br/> | Central Bank of Brazil — macro/financial series, Working Papers series | Macro-finance, rates, FX |
| **Dados Abertos B3 / IPEAData** | <https://www.ipeadata.gov.br/> | Open Brazilian economic & financial time series | Reproducible Brazil datasets (host on Zenodo) |

Tip: many Brazil-focused market-ML papers post the manuscript on **SciELO Preprints or SSRN** and the dataset/code on **Zenodo**, then publish in *RBFin* or a SciELO journal — search all four.

---

## 7. Landmark / representative q-fin & ML-finance papers (with links)

A starting reading list, all discoverable via the channels above. Use these as Connected-Papers / Semantic-Scholar seeds.

| Paper | Authors | Where | Link |
|-------|---------|-------|------|
| **Deep Hedging** | Buehler, Gonon, Teichmann, Wood (2019) | arXiv 1802.03042 (q-fin.PR/CP) | <https://arxiv.org/abs/1802.03042> |
| **Empirical Asset Pricing via Machine Learning** | Gu, Kelly, Xiu (2020, *RFS*) | SSRN 3159577 / NBER w25398 | <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577> · <https://www.nber.org/papers/w25398> |
| **FinRL: A Deep RL Library for Automated Stock Trading** | Liu et al. (2020) | arXiv 2011.09607 (q-fin.TR / cs.LG) | <https://arxiv.org/abs/2011.09607> |
| **FinRL: DRL Framework to Automate Trading** | Liu et al. (2021) | arXiv 2111.09395 | <https://arxiv.org/abs/2111.09395> |
| **Deep Hedging: Learning to Simulate Equity Option Markets** | Wiese, Bai, Wood, Buehler (2019) | arXiv 1911.01700 (q-fin.PR) | <https://arxiv.org/abs/1911.01700> |
| **Deep Hedging: Learning Risk-Neutral Implied Volatility Dynamics** | Buehler et al. (2021) | arXiv 2103.11948 | <https://arxiv.org/abs/2103.11948> |
| **Stock Market Prediction via Deep Learning Techniques: A Survey** | Survey (2022) | arXiv 2212.12717 (q-fin.ST) | <https://arxiv.org/abs/2212.12717> |

> Verify IDs by opening the `/abs/` link; arXiv numbers and versions are stable and citable. Do not cite a paper you have not opened.

---

## 8. Fast recipes

- **"What's new in B3/options ML this month?"** → `https://arxiv.org/list/q-fin.PR/recent` and `q-fin.TR/recent`, plus SSRN FEN and SciELO Preprints (PT-BR).
- **"Track a subfield automatically"** → subscribe to RSS `http://export.arxiv.org/rss/q-fin.TR` and follow a SciRate category feed.
- **"Find code with the paper"** → Hugging Face Papers and the Papers-with-Code links on the abstract page.
- **"Reproducible dataset for a Brazil study"** → publish/find data on Zenodo (DOI), manuscript on SciELO Preprints/SSRN.
- **"Map citations around Deep Hedging"** → seed Connected Papers / Semantic Scholar with arXiv 1802.03042.

---

**Keywords:** quantitative finance, q-fin, arXiv, preprints, computational finance, mathematical finance, portfolio management, pricing of securities, risk management, statistical finance, market microstructure, derivatives, options, deep hedging, machine learning finance, asset pricing, reinforcement learning trading, SSRN, RePEc, NBER, OSF, Zenodo, SciELO Preprints, B3, CVM, SBFin — finanças quantitativas, mercado financeiro, ações, opções, derivativos, precificação, gestão de risco, gestão de carteiras, microestrutura de mercado, aprendizado de máquina, preprints, servidores de preprints, pesquisa em finanças, Brasil, Bolsa B3, Banco Central do Brasil.
