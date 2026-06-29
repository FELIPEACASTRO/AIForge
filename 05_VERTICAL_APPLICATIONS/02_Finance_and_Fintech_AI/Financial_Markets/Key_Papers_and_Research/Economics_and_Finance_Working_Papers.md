# Economics & Finance Working-Paper Repositories (SSRN, NBER, RePEc)

> The preprint / working-paper ecosystem where market-relevant finance and economics research (asset pricing, derivatives, factors, quant) circulates years before journal publication — with real URLs, how-to-search tips, API endpoints, and Brazil-focused sources.

In empirical finance and asset pricing, the **working paper is the unit of record**. Landmark results — factor models, anomaly catalogs, replication audits — are read, cited, traded on, and replicated from preprints long before (or instead of) journal publication. A quant or finance researcher who only reads journals is reading the field 2-4 years late. This page maps the global working-paper ecosystem for **markets, asset pricing, derivatives, and quantitative finance**, with concrete search recipes and, where they exist, machine-readable endpoints.

---

## 1. The big picture: three overlapping networks

| Network | What it is | Home URL | Coverage of finance |
|---|---|---|---|
| **SSRN** | Centralized commercial preprint platform (Elsevier-owned); the de-facto home of finance papers via its **Financial Economics Network (FEN)** | https://www.ssrn.com/ | Dominant — most finance working papers land here first |
| **NBER** | US National Bureau of Economic Research; curated, member-authored series with an **Asset Pricing** program | https://www.nber.org/papers | Elite US academic asset pricing & macro-finance |
| **RePEc** | Decentralized, volunteer-run index aggregating ~5M+ items from 2,000+ archives; surfaced via IDEAS, EconPapers, EconStor, MPRA | http://repec.org/ | Broadest — indexes nearly every series below |

The three are complementary: **search RePEc/IDEAS to discover across everything**, **browse SSRN-FEN for the newest finance preprints**, and **track NBER for top-tier US asset pricing**. Most NBER, CEPR, Fed, and university series are *also* indexed in RePEc, so IDEAS is the best single starting point.

---

## 2. SSRN — Financial Economics Network (FEN)

SSRN's eLibrary holds **700,000+ abstracts** with full text for most. For markets work, the relevant umbrella is the **Financial Economics Network (FEN)**, a cluster of subject eJournals.

- **SSRN home:** https://www.ssrn.com/
- **eLibrary (papers):** https://papers.ssrn.com/
- **FEN landing page:** https://www.ssrn.com/index.cfm/en/fen/
- **FEN about:** https://www.ssrn.com/update/fen/fen_about.html
- **FEN eJournal offerings:** https://www.ssrn.com/index.cfm/en/fen/fen-ejournals/

**FEN eJournals most relevant to markets** (each is a browsable feed of new abstracts): *Asset Pricing & Valuation*, *Capital Markets: Market Efficiency*, *Derivatives*, *Microstructure: Market Structure & Pricing*, *Risk Management & Analysis in Financial Institutions*, *Behavioral & Experimental Finance*, and *Econometric & Statistical Methods*. Subscribing to an eJournal e-mails you every new working paper in that niche — this is how practitioners stay on the bleeding edge of factor/anomaly research.

### How to search SSRN for markets work
- **Direct paper URL pattern:** `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=<ID>` (e.g. `...?abstract_id=2287202` = Fama-French five-factor).
- **Browse a network/eJournal:** open the FEN eJournal listing and click an eJournal title to see all included papers.
- **Search tips:** use the eLibrary search box with author + keyword (e.g. `Kelly machine learning asset pricing`); filter by date to find the newest preprints; sort by **download count** to find the most influential papers in a topic (download rank is SSRN's crude impact proxy that quants actually use to find "what's hot").
- **Why quants use it:** SSRN download counts and the FEN feeds are the fastest way to spot a new factor or anomaly paper before it hits a journal, and to pull the PDF + code links for replication.

---

## 3. NBER Working Papers

NBER papers are authored by affiliated researchers and organized into **programs**. For markets, the key program is **Asset Pricing**.

- **All NBER working papers:** https://www.nber.org/papers
- **Asset Pricing program:** https://www.nber.org/programs-projects/programs-working-groups/asset-pricing
- **Direct paper URL pattern:** `https://www.nber.org/papers/w<NUMBER>` (e.g. https://www.nber.org/papers/w25398).
- **Full-text PDF pattern:** `https://www.nber.org/system/files/working_papers/w<NUMBER>/w<NUMBER>.pdf`
- **Program meetings** (where new asset-pricing work is presented): e.g. https://www.nber.org/conferences/asset-pricing-program-meeting-spring-2026 and the Summer Institute Asset Pricing https://www.nber.org/conferences/si-2026-asset-pricing

**How to search:** the `/papers` page supports keyword, author, JEL, and program filters; you can also browse by program to get a chronological feed of asset-pricing output. NBER PDFs are paywalled for some users but abstracts and metadata are open, and most are mirrored on SSRN and RePEc.

---

## 4. RePEc and its interfaces

RePEc (Research Papers in Economics) is a **decentralized bibliographic database** built from archives maintained by institutions. You never use "RePEc" directly — you use one of its front-ends. RePEc handles (`RePEc:xxx:yyy:zzz`) uniquely identify every paper, series, and author.

| Interface | URL | Best for |
|---|---|---|
| **IDEAS** | https://ideas.repec.org/ | Largest front-end; author profiles, citation/reference graphs, rankings, JEL & series browsing |
| **EconPapers** | https://econpapers.repec.org/ | Clean series/JEL browsing, full-text links, RSS feeds |
| **EconStor** | https://www.econstor.eu/ | ZBW (Leibniz) open-access full-text server, 200,000+ texts |
| **MPRA** | https://mpra.ub.uni-muenchen.de/ | Self-archiving for unaffiliated authors; 50,000+ items |

### Searching RePEc by topic (the quant workflow)
- **Browse by JEL code** — the canonical way to find asset-pricing work:
  - **G12** Asset Pricing; Trading Volume; Bond Interest Rates → https://ideas.repec.org/j/G12.html
  - **G11** Portfolio Choice; Investment Decisions; **G13** Contingent Pricing; Futures Pricing (derivatives/options); **G14** Information & Market Efficiency; **C58** Financial Econometrics.
  - EconPapers JEL pattern: `https://econpapers.repec.org/scripts/search.pf?jel=G12`
- **Browse by series** — IDEAS series pattern `https://ideas.repec.org/s/<handle>.html` (e.g. CEPR DPs at https://ideas.repec.org/s/cpr/ceprdp.html). EconPapers series pattern: `https://econpapers.repec.org/paper/<archive><series>/`.
- **Author tracking** — every registered economist has an IDEAS profile listing all their indexed papers plus citation counts; this is how you follow a factor researcher's full pipeline.
- **Citation/reference graphs** (via CitEc) on IDEAS let you do forward-citation chasing on a seminal factor paper to find every follow-up and replication.
- **Rankings** — IDEAS ranks authors, institutions, and series, useful for gauging a series' weight.

---

## 5. Other major series (mostly RePEc-indexed)

| Series | Home URL | Focus / notes |
|---|---|---|
| **CEPR Discussion Papers** | https://cepr.org/publications/discussion-papers | ~1,000/yr, archive 20,000+; strong European macro-finance. RePEc: https://ideas.repec.org/s/cpr/ceprdp.html |
| **CESifo Working Papers** | https://www.cesifo.org/en/publications | Munich-based; public economics & finance. EconPapers: https://econpapers.repec.org/paper/cesceswps/ |
| **IZA Discussion Papers** | https://www.iza.org/publications/dp | Labor-centric but indexed alongside finance |
| **EconStor** | https://www.econstor.eu/ | Open-access full text (ZBW); searchable by keyword/JEL |
| **MPRA** | https://mpra.ub.uni-muenchen.de/ | Self-archive; useful for emerging-market & non-affiliated authors |
| **Fed FEDS** | https://www.federalreserve.gov/econres/feds/index.htm | Finance & Economics Discussion Series — US markets & financial stability; current year e.g. https://www.federalreserve.gov/econres/feds/2026.htm |
| **Fed IFDP** | https://www.federalreserve.gov/econres/ifdp/index.htm | International Finance Discussion Papers — global finance & capital flows |
| **IMF Working Papers** | https://www.imf.org/en/Publications/WP | Macro-financial, sovereign, FX, financial stability |
| **World Bank Policy Research WP** | https://documents.worldbank.org/ | Emerging-market finance & development; search the Documents & Reports portal |
| **AgEcon Search** | https://ageconsearch.umn.edu/ | Agricultural/applied & **commodity** economics (~150,000 texts) — relevant for commodity derivatives |
| **arXiv q-fin** | https://arxiv.org/archive/q-fin | Quantitative-finance preprints (math/ML-heavy; pricing, microstructure, portfolio management) |

**FEDS direct PDF pattern:** `https://www.federalreserve.gov/econres/feds/files/<YEAR><NUM>pap.pdf`.

---

## 6. Brazilian & Portuguese-language sources (Brazil-focused)

| Source | URL | Notes |
|---|---|---|
| **BCB — Trabalhos para Discussão (Working Paper Series)** | https://www.bcb.gov.br/ | Banco Central do Brasil research series, ISSN **1519-1028**; PDFs at `https://www.bcb.gov.br/content/publicacoes/WorkingPaperSeries/TD<NUM>.pdf`. Monetary policy, banking, financial stability, market microstructure for Brazil |
| **SBFin — Sociedade Brasileira de Finanças** | https://sbfin.org.br/en | Brazilian Finance Society; runs the EBFin meeting and the RBFin journal |
| **RBFin — Brazilian Review of Finance** | https://rbfin.sbfin.org.br/en | Open-access (CC BY), ISSN 1984-5146; PT & EN; finance, financial econometrics. Mirror: https://periodicos.fgv.br/rbfin |
| **SciELO Preprints** | https://preprints.scielo.org/ | Latin-American open preprint server (PT/ES/EN) — economics & finance |
| **CVM — Comissão de Valores Mobiliários** | https://www.gov.br/cvm/ | Brazilian securities regulator; studies, regulatory data, market statistics |
| **B3 — Brasil, Bolsa, Balcão** | https://www.b3.com.br/ | Brazilian exchange; market data, derivatives specs, research notes |
| **IPEA / FGV / Insper** | https://www.ipea.gov.br/ ; https://portal.fgv.br/ | Domestic research institutions; many series indexed in RePEc |

**Tip:** much Brazilian finance research is dual-posted to SSRN-FEN and RePEc/EconPapers; search IDEAS author profiles for B3/CVM/BCB-affiliated economists, and filter EconPapers by the Brazilian archive handles to find Portuguese-language discussion papers.

---

## 7. Landmark / representative papers discoverable via these platforms

A short, real, verifiable list of the kind of markets/asset-pricing work these repositories deliver:

| Paper | Authors | Where to find |
|---|---|---|
| **A Five-Factor Asset Pricing Model** | Fama & French | SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2287202 |
| **Dissecting Anomalies with a Five-Factor Model** | Fama & French | SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2503174 |
| **Dissecting Anomalies** | Fama & French | SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=911960 |
| **International Tests of a Five-Factor Asset Pricing Model** | Fama & French | SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2622782 |
| **Empirical Asset Pricing via Machine Learning** | Gu, Kelly & Xiu | NBER w25398 https://www.nber.org/papers/w25398 · SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577 |

These five trace the modern factor-zoo / ML-asset-pricing arc and are all freely discoverable (abstract + working-paper PDF) through SSRN, NBER, and RePEc/IDEAS.

---

## 8. Quick how-to-search cheat sheet

- **Find every paper on a factor/anomaly:** IDEAS JEL `G12` page → sort recent, then forward-cite the seminal paper via CitEc.
- **Get the newest finance preprints weekly:** subscribe to the relevant SSRN-FEN eJournal feed (Asset Pricing & Valuation, Derivatives, Market Efficiency).
- **Replicate a result:** pull the SSRN/NBER PDF, then check the author's IDEAS profile and personal page for data/code links.
- **Brazil-specific market research:** BCB Trabalhos para Discussão + RBFin + SBFin/EBFin proceedings, cross-checked on EconPapers Brazilian archives.
- **Derivatives/options pricing theory:** JEL **G13** on IDEAS + arXiv **q-fin.PR** (pricing of securities).
- **Commodity derivatives:** AgEcon Search + JEL **Q14/G13**.

---

**Sources:** [SSRN](https://www.ssrn.com/) · [SSRN FEN](https://www.ssrn.com/index.cfm/en/fen/) · [NBER Papers](https://www.nber.org/papers) · [NBER Asset Pricing](https://www.nber.org/programs-projects/programs-working-groups/asset-pricing) · [RePEc](http://repec.org/) · [IDEAS](https://ideas.repec.org/) · [EconPapers](https://econpapers.repec.org/) · [IDEAS JEL G12](https://ideas.repec.org/j/G12.html) · [CEPR DPs](https://cepr.org/publications/discussion-papers) · [EconStor](https://www.econstor.eu/) · [MPRA](https://mpra.ub.uni-muenchen.de/) · [Fed FEDS](https://www.federalreserve.gov/econres/feds/index.htm) · [Fed IFDP](https://www.federalreserve.gov/econres/ifdp/index.htm) · [IMF WP](https://www.imf.org/en/Publications/WP) · [AgEcon Search](https://ageconsearch.umn.edu/) · [BCB](https://www.bcb.gov.br/) · [RBFin](https://rbfin.sbfin.org.br/en) · [SBFin](https://sbfin.org.br/en) · [SciELO Preprints](https://preprints.scielo.org/)

**Keywords:** working papers, preprints, SSRN, Financial Economics Network, FEN, NBER, RePEc, IDEAS, EconPapers, EconStor, MPRA, CEPR, CESifo, FEDS, IFDP, IMF, World Bank, AgEcon Search, asset pricing, factor models, anomalies, derivatives, quantitative finance, JEL G12, B3, CVM, BCB, SBFin, RBFin, SciELO Preprints | papers de trabalho, preprints, precificação de ativos, modelos de fatores, anomalias, derivativos, opções, finanças quantitativas, mercado financeiro, ações, renda variável, Banco Central do Brasil, Trabalhos para Discussão, Comissão de Valores Mobiliários, Bolsa B3, Sociedade Brasileira de Finanças, Revista Brasileira de Finanças
