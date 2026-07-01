# Academic Preprints & Working Papers for Football Prediction

> A dense, source-verified map of **preprint servers, working-paper repositories and scholarly aggregators** where you can find peer-review-adjacent research on football (soccer / *futebol*) match and betting-market **prediction** — for **research and education only**, not betting advice. Every platform below was checked against a live URL, and every "example paper" was confirmed to exist (title + authors + ID/DOI). Current for **2024-2026**.

---

## ⚠️ Read this first — responsible gambling (*jogo responsável*)

- **This is a literature-discovery page, not a tipster service.** Reading forecasting papers builds statistics/ML skill; it does **not** give you a betting edge.
- **Markets are efficient and the house has a margin.** Sharp closing odds (*odds de fechamento*) are near-perfectly calibrated; after the *over-round / vig* (*margem*), positive long-run ROI is rare and fragile. **Most bettors lose money over time.** Several papers below (Winkelmann et al.; Angelini & De Angelis; Gross & Rebeggiani) empirically find markets are *mostly* efficient.
- **The literature warns about harm, not just profit.** NBER w33108 finds sports-betting legalization *reduces household savings and raises overdrafts* among constrained households; the SciELO and PsyArXiv items below study problem-gambling behaviour directly.
- **Help / *ajuda*:** 🇬🇧 [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) (0808 8020 133) · 🇧🇷 **Jogo Responsável** — [Jogadores Anônimos Brasil](https://jogadoresanonimos.com.br/) · CVV **188** (apoio emocional) · [Gov.br – Apostas / SPA/MF](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas).

---

## How to use this page

Preprints are **not peer-reviewed** — treat them as working drafts, check for a later journal version, and read critically. The tables give, for each platform: a **concrete search route** (URL/API) tuned for football/betting, **verified example papers**, and cost. All items are **open access** unless noted.

---

## 1. arXiv — the core ML / statistics / econometrics feed

Football forecasting clusters in four arXiv categories: **stat.AP** (Applications), **stat.ME** (Methodology), **stat.ML** & **cs.LG** (Machine Learning), and **econ.EM** (Econometrics). Betting-market efficiency work often appears in **q-fin** and **econ.EM**.

**Search routes**
- Full-text search UI: `https://arxiv.org/search/?searchtype=all&query=soccer+prediction` (also try `football prediction`, `Dixon-Coles`, `expected goals`, `betting market efficiency`).
- Category firehose: `https://arxiv.org/list/stat.AP/recent`.
- **API (free, no key):** `http://export.arxiv.org/api/query?search_query=all:soccer+AND+all:prediction&max_results=50` — machine-readable Atom, ideal for building a watch-list.

| Verified paper | Authors | arXiv ID · category · yr |
|---|---|---|
| Machine Learning for Soccer Match Result Prediction (survey/book chapter) | Bunker, Yeung, Fujii | [2403.07669](https://arxiv.org/abs/2403.07669) · cs.LG · 2024 |
| Evaluating Soccer Match Prediction Models: A Deep Learning Approach and Feature Optimization for Gradient-Boosted Trees | Yeung, Bunker, Umemoto, Fujii | [2309.14807](https://arxiv.org/abs/2309.14807) · cs.LG · 2023 |
| A Systematic Review of Machine Learning in Sports Betting: Techniques, Challenges, and Future Directions | Galekwa, Tshimula, Tajeuna, Kyandoghere | [2410.21484](https://arxiv.org/abs/2410.21484) · cs.LG · 2024 |
| Forecasting football matches by predicting match statistics (GAP ratings) | Wheatcroft | [2001.09097](https://arxiv.org/abs/2001.09097) · stat.AP · 2020 |
| Evaluating probabilistic forecasts of football matches: the case against the RPS | Wheatcroft | [1908.08980](https://arxiv.org/abs/1908.08980) · stat.AP · 2019 |
| Extending the Dixon & Coles model: an application to women's football data | Michels, Ötting, Karlis | [2307.02139](https://arxiv.org/abs/2307.02139) · stat.ME · 2023 |
| A Machine Learning Approach for Player- and Position-Adjusted Expected Goals (xG) | Hewitt, Karakuş | [2301.13052](https://arxiv.org/abs/2301.13052) · cs.LG · 2023 |
| Real-time forecasting within soccer matches through a Bayesian lens (in-play) | Divekar, Deb, Roy | [2303.12401](https://arxiv.org/abs/2303.12401) · stat.AP · 2023 |
| Predicting soccer matches with complex networks and machine learning 🇧🇷 (USP São Carlos) | Baratela, Xavier, Peron, Villas-Boas, Rodrigues | [2409.13098](https://arxiv.org/abs/2409.13098) · cs.SI · 2024 |
| From Players to Champions: Generalizable ML for Match Outcome Prediction (FIFA World Cup) | Al-Bustami, Ghazal | [2505.01902](https://arxiv.org/abs/2505.01902) · cs.LG · 2025 |

**Free?** ✅ Fully open; free API.

---

## 2. SportRxiv — the dedicated sports-science preprint server

[**SportRxiv**](https://sportrxiv.org/) (`sportrxiv.org`) is the first community-led, open-access preprint repository **dedicated to sport, exercise, performance and health research** — hosted by University of Ottawa Libraries with the STORK non-profit, on PKP's Open Preprint Systems.

**Search routes**
- Browse by section: *Methods and Practices · Exercise Science · Sport Science · Clinical Research · Other* at `https://sportrxiv.org/index.php/server/index`.
- Site search box (top-right) — query `football prediction`, `match analysis`, `expected goals`, `analytics`.
- RSS/Atom feeds for new submissions (watch-list).

**Note:** SportRxiv leans toward sports *science* (performance, physiology, methods) rather than betting-market econometrics; use it for match-analysis, xG methodology and performance-modelling preprints, and cross-reference match-prediction items to arXiv/SSRN. **Free?** ✅

---

## 3. Economics & finance working papers — betting-market efficiency

The classic "can you beat the market?" question lives in economics working-paper series. **RePEc** is the umbrella network; **IDEAS**, **EconPapers**, **MPRA**, **SSRN** and **NBER** are its main front-ends/archives.

**Search routes**
- **IDEAS/RePEc** advanced search: `https://ideas.repec.org/search.html` — query `football betting market efficiency`; filter by JEL **Z2** (Sports Economics), **D8** (info/uncertainty), **G14** (market efficiency).
- **EconPapers** full-text: `https://econpapers.repec.org/scripts/search.pl?ft=football+betting+market+efficiency`.
- **MPRA** (Munich Personal RePEc Archive) browse/search: `https://mpra.ub.uni-muenchen.de/cgi/search` — query `football betting`.
- **NBER** working papers: `https://www.nber.org/search?page=1&perPage=50&q=sports%20betting`.
- **SSRN** search: `https://papers.ssrn.com/sol3/results.cfm` — query `football betting market efficiency` (Sports & Behavioural/Financial Economics eJournals).

| Repository | Verified example paper (link) | Free? |
|---|---|---|
| **SSRN** | Efficiency of Online Football Betting Markets — Angelini & De Angelis, [SSRN 3070329](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3070329) | ✅ free download |
| **SSRN** | Betting Market Inefficiencies in European Football — Bookmakers' Mispricing or Pure Chance? — Winkelmann, Oetting, Deutscher, [SSRN 3672233](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3672233) | ✅ |
| **SSRN** | Sports Forecasting: A Comparison of the Forecast Accuracy of Prediction Markets, Betting Odds and Tipsters — Spann & Skiera, [SSRN 2479770](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2479770) | ✅ |
| **MPRA / RePEc** | Chance or Ability? The Efficiency of the Football Betting Market Revisited — Gross & Rebeggiani, [MPRA 87230](https://mpra.ub.uni-muenchen.de/87230/) | ✅ |
| **MPRA / RePEc** | Testing semi-strong efficiency in a fixed odds betting market: evidence from principal European football leagues — Bernardo, Ruberti & Verona, [MPRA 66414](https://mpra.ub.uni-muenchen.de/66414/) | ✅ |
| **MPRA / RePEc** | Comparing Two Methods for Testing the Efficiency of Sports Betting Markets — Hegarty & Whelan, [MPRA 121382](https://mpra.ub.uni-muenchen.de/121382/) | ✅ |
| **NBER** | Gambling Away Stability: Sports Betting's Impact on Vulnerable Households — Baker, Balthrop, Johnson, Kotter, Pisciotta, [NBER w33108](https://www.nber.org/papers/w33108) | ✅ abstract + PDF |
| **NBER** | The Economic Winners and Losers of Legalized Gambling — Kearney, [NBER w11234](https://www.nber.org/papers/w11234) | ✅ |

RePEc indexes MPRA, university series and SSRN metadata, so an IDEAS/EconPapers search surfaces most of the above in one place.

---

## 4. OSF Preprints, SocArXiv & PsyArXiv — open, social-science & behavioural

The [**OSF Preprints**](https://osf.io/preprints/) infrastructure (Center for Open Science) hosts many branded servers: **SocArXiv** (social science), **PsyArXiv** (psychology), and more. Great for gambling *behaviour* and applied ML crossovers, plus the responsible-gambling angle.

**Search routes**
- Global discover: `https://osf.io/preprints/discover?q=football%20prediction` (also `sports betting`, `problem gambling`).
- **API (free):** `https://api.osf.io/v2/preprints/?filter[title]=football` returns JSON metadata (the OSF web pages are JS-rendered, so the API is the reliable route for scripting).
- Per-server: SocArXiv `https://osf.io/preprints/socarxiv/discover`, PsyArXiv `https://osf.io/preprints/psyarxiv/discover`.

| Server | Verified example preprint (link) | Free? |
|---|---|---|
| **SocArXiv** | Predicting Football Match Outcomes Using Large Language Models: A Comparative Study with Traditional ML — [osf.io/preprints/socarxiv/e5wpy](https://osf.io/preprints/socarxiv/e5wpy) (2025) | ✅ |
| **PsyArXiv** | Confidence biases in problem gambling — [osf.io/preprints/psyarxiv/j59ds](https://osf.io/preprints/psyarxiv/j59ds) (2023) | ✅ |

**Free?** ✅ Open; free JSON API.

---

## 5. SciELO Preprints 🇧🇷 — Brazil & Latin America

[**SciELO Preprints**](https://preprints.scielo.org/index.php/scielo) (`preprints.scielo.org`) is the multidisciplinary, multilingual preprint server of the SciELO Network — the natural home for **Brazilian/LatAm** work in Portuguese/Spanish, including the fast-growing literature on *apostas esportivas* (sports betting) after Brazil's Lei 14.790/2023.

**Search routes**
- Search UI: `https://preprints.scielo.org/index.php/scielo/search` — query `futebol`, `apostas esportivas`, `previsão`, `bets`.
- Browse "About the Server": `https://preprints.scielo.org/index.php/scielo/about`.

| Verified example preprint (PT) | Author | Link |
|---|---|---|
| *Diversão ou Armadilha? Um estudo exploratório das Apostas Esportivas (Bets) entre Universitários Brasileiros sob a Lente da TCP* (n=814 estudantes) | Ahmed Sameer El Khatib | [SciELO Preprint 10133](https://preprints.scielo.org/index.php/scielo/preprint/view/10133) (2024) |

**Free?** ✅ Open access; strong for responsible-gambling and behavioural framing in Portuguese.

---

## 6. Multidisciplinary preprint & repository hosts

General-purpose hosts that also carry football ML papers and datasets.

| Platform | How to search for football | Verified example (link) | Free? |
|---|---|---|---|
| **Preprints.org** (MDPI) | `https://www.preprints.org/search?query1=soccer+prediction` | Short Survey in Machine Learning for Soccer Analytics — P. Amadu, [202410.0178](https://www.preprints.org/manuscript/202410.0178/v1) (2024) | ✅ |
| **Zenodo** (CERN) — datasets + preprints, DOIs | `https://zenodo.org/search?q=soccer+prediction` (also `football machine learning`) | Football Player Performance Prediction Using ML — Nair & Francis, DOI [10.5281/zenodo.15363952](https://zenodo.org/records/15363952) (2025); SoccerMon dataset DOI [10.5281/zenodo.10033832](https://zenodo.org/records/10033832) | ✅ + API |
| **Research Square** | `https://www.researchsquare.com/search?q=football%20prediction` | *(search route — verify each hit's title/authors before citing)* | ✅ |
| **Authorea** (Wiley) | `https://www.authorea.com/search?q=soccer%20prediction` | *(search route — verify each hit before citing)* | ✅ |

> Anti-noise note: for Research Square and Authorea I did **not** find a specific football-prediction preprint I could fully verify at write time, so no example is listed — do **not** cite a title from these hosts without opening the record first.

---

## 7. Scholarly aggregators — discover across all of the above at once

Instead of searching servers one-by-one, use cross-index aggregators (all free, most with APIs). These federate arXiv, SSRN, RePEc, OSF, SciELO, journals and more.

| Aggregator | Search route for football prediction | API? |
|---|---|---|
| **Google Scholar** | `https://scholar.google.com/scholar?q=football+match+prediction+betting` | no official API |
| **Semantic Scholar** | `https://www.semanticscholar.org/search?q=soccer%20match%20prediction` | ✅ free [Academic Graph API](https://api.semanticscholar.org/) |
| **OpenAlex** | `https://openalex.org/works?search=soccer%20match%20prediction` | ✅ free [REST API](https://docs.openalex.org/) |
| **CORE** | `https://core.ac.uk/search?q=football+prediction` | ✅ free API (key) |
| **BASE** | `https://www.base-search.net/Search/Results?lookfor=soccer+prediction` | ✅ |
| **Papers with Code** | `https://paperswithcode.com/search?q=soccer+prediction` | ✅ (code + benchmarks) |

Use OpenAlex/Semantic Scholar filters on **concept** and **year (2024-2026)** to build a reproducible, de-duplicated reading list; both expose citations for snowballing from the arXiv seeds in §1.

---

## Master recipe — one-line searches you can paste

| Platform | Paste-ready query |
|---|---|
| arXiv API | `http://export.arxiv.org/api/query?search_query=all:soccer+AND+all:prediction&max_results=50` |
| SSRN | `football betting market efficiency` at [papers.ssrn.com](https://papers.ssrn.com/sol3/results.cfm) |
| IDEAS/RePEc | `football betting` + JEL **Z2/G14** at [ideas.repec.org/search.html](https://ideas.repec.org/search.html) |
| MPRA | `football betting` at [mpra.ub.uni-muenchen.de/cgi/search](https://mpra.ub.uni-muenchen.de/cgi/search) |
| OSF API | `https://api.osf.io/v2/preprints/?filter[title]=football` |
| SciELO Preprints | `apostas esportivas` / `futebol` at [preprints.scielo.org …/search](https://preprints.scielo.org/index.php/scielo/search) |
| SportRxiv | `match prediction` at [sportrxiv.org](https://sportrxiv.org/) |
| OpenAlex API | `https://api.openalex.org/works?search=soccer%20match%20prediction&filter=from_publication_date:2024-01-01` |

---

## Responsible-gambling resources (*jogo responsável*)

- 🇬🇧 **BeGambleAware** — https://www.begambleaware.org/ · **GamCare** helpline 0808 8020 133 — https://www.gamcare.org.uk/
- 🌍 **Gamblers Anonymous** — https://www.gamblersanonymous.org/
- 🇧🇷 **Jogadores Anônimos Brasil** — https://jogadoresanonimos.com.br/ · **CVV 188** (apoio emocional 24h) — https://www.cvv.org.br/ · **Gov.br / Secretaria de Prêmios e Apostas (SPA/MF)** — https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas
- If gambling stops being fun, it is not fun anymore. Research ≠ a system to beat the market. *Se você aposta, aposte só o que pode perder — e pare quando deixar de ser diversão.*

---

**Sources:** [arXiv](https://arxiv.org/) · [arXiv 2403.07669](https://arxiv.org/abs/2403.07669) · [2309.14807](https://arxiv.org/abs/2309.14807) · [2410.21484](https://arxiv.org/abs/2410.21484) · [2001.09097](https://arxiv.org/abs/2001.09097) · [1908.08980](https://arxiv.org/abs/1908.08980) · [2307.02139](https://arxiv.org/abs/2307.02139) · [2301.13052](https://arxiv.org/abs/2301.13052) · [2303.12401](https://arxiv.org/abs/2303.12401) · [2409.13098](https://arxiv.org/abs/2409.13098) · [2505.01902](https://arxiv.org/abs/2505.01902) · [SportRxiv](https://sportrxiv.org/) · [SSRN 3070329](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3070329) · [SSRN 3672233](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3672233) · [SSRN 2479770](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2479770) · [MPRA 87230](https://mpra.ub.uni-muenchen.de/87230/) · [MPRA 66414](https://mpra.ub.uni-muenchen.de/66414/) · [MPRA 121382](https://mpra.ub.uni-muenchen.de/121382/) · [NBER w33108](https://www.nber.org/papers/w33108) · [NBER w11234](https://www.nber.org/papers/w11234) · [OSF SocArXiv e5wpy](https://osf.io/preprints/socarxiv/e5wpy) · [PsyArXiv j59ds](https://osf.io/preprints/psyarxiv/j59ds) · [SciELO Preprint 10133](https://preprints.scielo.org/index.php/scielo/preprint/view/10133) · [Preprints.org 202410.0178](https://www.preprints.org/manuscript/202410.0178/v1) · [Zenodo 15363952](https://zenodo.org/records/15363952) · [Zenodo 10033832](https://zenodo.org/records/10033832) · [OpenAlex](https://openalex.org/) · [Semantic Scholar](https://www.semanticscholar.org/) · [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/)

**Keywords:** football/soccer prediction, betting-market efficiency, preprints, working papers, arXiv stat.AP/cs.LG/econ.EM, SportRxiv, SSRN, RePEc, MPRA, NBER, OSF/SocArXiv, PsyArXiv, SciELO Preprints, Zenodo, expected goals (xG), Dixon-Coles, responsible gambling · *futebol, previsão de partidas, apostas esportivas, eficiência de mercado, artigos de pesquisa, pré-publicações, gols esperados, jogo responsável, mercado eficiente*
