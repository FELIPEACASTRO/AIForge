# Conferences, Journals & Communities for ML-Finance

> A dense, source-verified map of **where the newest machine-learning-in-finance work appears first** — the flagship conference (ACM ICAIF), the AI/ML workshops that now host the best quant papers (NeurIPS / KDD / ICLR), the journals that matter (academic *and* practitioner), the preprint pipes (SSRN FEN, arXiv q-fin), and the online communities (Reddit, Wilmott, QuantConnect, Numerai, Quantocracy) where practitioners actually gather. Brazil-specific venues (SBFin, QuantBrasil, Trading com Dados, Quantzed) are called out. This is the *discovery layer* most repos under-index: the exchanges and data APIs are covered elsewhere in this AIForge section — this page is about **people, venues, and peer review**.

Audience note (Brazil): body is in English; key Portuguese terms appear in parentheses. Most academic journals here are paywalled — see the **Open-access shortcut** column. arXiv q-fin and SSRN are free.

---

## 1. The flagship: ACM ICAIF

**ICAIF — ACM International Conference on AI in Finance** is the single most important place for peer-reviewed AI-in-finance research. It is the venue where multi-agent LLM trading papers, deep-RL portfolio work, and market-simulation research land first. Sponsored by **ACM SIGAI/SIGKDD**; proceedings go into the **ACM Digital Library**.

| Edition | Dates | Location | Notes |
|---|---|---|---|
| **ICAIF'25** (6th) | **15–18 Nov 2025** | **Singapore** (co-located with Singapore FinTech Festival) | General Chair Tat-Seng Chua (NUS); program chairs from NTU, Bowdoin, J.P. Morgan AI Research. Site: https://icaif25.org/ |
| ICAIF'24 (5th) | Nov 2024 | Brooklyn, NY (NYU Tandon) | Proceedings: https://dl.acm.org/doi/proceedings/10.1145/3677052 |
| ICAIF'23 / '22 / '21 / '20 | 2020–2023 | NY / Online | NYU Tandon's FRE dept ran early editions: https://engineering.nyu.edu/academics/departments/finance-and-risk-engineering/icaif-conference |

- **Submission timing (ICAIF'25, for calibration of the annual cycle):** full-paper deadline **early Aug 2025** (final extended deadline 3 Aug 2025 AoE), notifications **26 Sep 2025**, camera-ready before the November conference. Papers are **≤ 8 pages** in ACM two-column `sigconf` format including references. Call: https://icaif25.org/calls-for-papers/
- **Practical rule of thumb:** the ICAIF CFP opens in **Q1–Q2** with a **summer deadline** and a **November conference**. Watch the ACM page and the `aifin-worldwide` Google Group (below) for the next edition's dates.
- Proceedings DOI series: ICAIF'25 = `10.1145/3768292` — https://dl.acm.org/doi/proceedings/10.1145/3768292

---

## 2. AI/ML conference workshops (where the frontier papers really appear)

The strongest ML-finance papers increasingly debut at **workshops** attached to the big ML conferences — faster turnaround, more methods-heavy, and free preprints on OpenReview. These are heavily under-indexed.

| Workshop / venue | Host conf. | Focus | Link | Timing (typical) |
|---|---|---|---|---|
| **Generative AI in Finance** | **NeurIPS 2025** | Scenario generation (high-dim time series, vol surfaces, LOB dynamics, financial networks), synthetic data, privacy, foundation models under market constraints, uncertainty under regime shift | https://sites.google.com/view/neurips-25-gen-ai-in-finance/home · OpenReview: https://openreview.net/group?id=NeurIPS.cc/2025/Workshop/GenAI_in_Finance | Submissions ~late Aug, conf. **6–7 Dec 2025**, San Diego |
| **KDD-MLF — Machine Learning in Finance** | **ACM SIGKDD** | GenAI on financial tabular/time-series/clickstream data, fraud, predictive analytics; bridges classic ML and GenAI; practitioner short papers welcome | KDD-MLF 2025: https://sites.google.com/view/kdd-mlf-2025/ · KDD-MLF 2026 announced: https://www.aiexpertmagazine.com/acm-sigkdd-workshop-on-machine-learning-in-finance-kdd-mlf-2026-everything-you-need-to-know/ | Submissions ~May, summer workshop |
| **DMO-FinTech — Decision Making & Optimization in FinTech** | **PAKDD 2026** | RL infra, optimization, FinRL-style systems (e.g. FinRL-X presented here) | via PAKDD 2026 program | Spring 2026 |
| **Crypto/agent challenges** (e.g. single-crypto trading) | **COLING 2025** | LLM agents in crypto trading benchmarks | via COLING 2025 workshops | — |
| ICML / ICLR finance & time-series tracks | ICML / ICLR | Time-series foundation models, RL for trading appear in main track + ad-hoc workshops (no permanent finance workshop — search OpenReview each cycle) | https://openreview.net/ | Spring/summer |

> Practical tip: search **OpenReview** for the venue group each year — accepted papers post there before camera-ready, often with code. The NeurIPS GenAI-in-Finance group is the best single feed for 2025–2026 generative-finance work.

---

## 3. Industry & practitioner conferences (the "buy-side" circuit)

Academic venues are only half the picture. Practitioners (banks, hedge funds, asset managers) gather at commercial conferences — pricier, less peer-reviewed, but where deployment reality and recruiting happen.

| Event | Organizer | Focus | Link | Cadence |
|---|---|---|---|---|
| **QuantMinds International** (ex–Global Derivatives) | Informa Connect | The largest/longest-running quant-finance event (28+ yrs); derivatives, vol modelling, portfolio optimization, model risk, ML/AI applications; 150+ speakers, 400+ attendees | https://informaconnect.com/quantminds-international/ | **17–20 Nov 2025** & **16–19 Nov 2026**, InterContinental London – The O2 |
| **CFE-CMStatistics** (CFEnetwork) | CMStatistics / CFEnetwork | Computational & **Financial Econometrics** — academic, methods-heavy; ~1,200+ talks; sponsored by *Econometrics and Statistics* (EcoSta) | https://www.cmstatistics.org/ · 2025: https://www.cmstatistics.org/CFECMStatistics2025/ | **13–15 Dec 2025**, Birkbeck, London |
| **Risk.net Quant / quant-finance coverage** | Risk.net (Infopro) | Industry derivatives/quant news + Quant Congress events; XVA, model risk, ML in pricing | https://www.risk.net/quantitative-finance | Rolling + annual congresses |
| **CQF Institute events** | Fitch Learning / CQF | Talks, lectures, networking for the CQF community; ML-in-finance webinars | https://cqfinstitute.org/ | Year-round, many free |
| **AI4Finance events / FinRL Contest** | AI4Finance Foundation | Open RL-trading contests (FinRL Contest 2025 incl. FinRL-DeepSeek LLM-signal task); ties to ICAIF tutorials | https://github.com/AI4Finance-Foundation/FinRL · group: https://groups.google.com/g/ai4finance | Annual contest + workshops |
| Singapore FinTech Festival (SFF) | MAS | Mega fintech/AI event; ICAIF'25 co-located | https://www.fintechfestival.sg/ | Nov |

> Brazil access: QuantMinds/CFE registration is online for international attendees; **SBFin Meeting** (below) is the main in-country academic finance venue.

---

## 4. Academic journals — top-tier finance & quant

The "big-3" finance journals (**JF, JFE, RFS**) rarely run pure-ML papers but are where ML-driven empirical asset pricing (e.g. "Empirical Asset Pricing via Machine Learning") gets validated. The quant/math journals are where methods live.

| Journal | Publisher | Why it matters for ML-finance | Open-access shortcut | Link |
|---|---|---|---|---|
| **Journal of Finance (JF)** | Wiley / AFA | Top-3; ML asset-pricing & market-efficiency results | SSRN preprints | https://onlinelibrary.wiley.com/journal/15406261 |
| **Journal of Financial Economics (JFE)** | Elsevier | Top-3; empirical methods, microstructure | SSRN | https://www.sciencedirect.com/journal/journal-of-financial-economics |
| **Review of Financial Studies (RFS)** | Oxford / SFS | Top-3; data-science-heavy empirical finance | SSRN | https://academic.oup.com/rfs |
| **Journal of Financial Markets** | Elsevier | Trading mechanisms, order placement, microstructure, price behavior — core for HFT/microstructure ML | SSRN | https://www.sciencedirect.com/journal/journal-of-financial-markets |
| **Quantitative Finance** | Taylor & Francis (since 2001) | Mathematical finance, financial econometrics, agent-based modelling, market microstructure, ML on markets — the central quant journal | Some Open Select | https://www.tandfonline.com/journals/rquf20 |
| **Mathematical Finance** | Wiley | Novel math/stat methods for financial problems; stochastic control, computational finance | arXiv q-fin | https://onlinelibrary.wiley.com/journal/14679965 |
| **Finance Research Letters (FRL)** | Elsevier | Bimonthly **letters** (≤ ~2,500 words) — *fast* outlet for ML/crypto/sentiment results | SSRN | https://www.sciencedirect.com/journal/finance-research-letters |

---

## 5. Academic journals — quant/data-science specialist (the under-indexed gems)

These are the venues most catalogs miss, and exactly where ML-in-finance methods are published.

| Journal | Publisher | Focus | Cost / OA | Link |
|---|---|---|---|---|
| **Journal of Financial Data Science (JFDS)** | Portfolio Management Research (PMR) | The home journal for **AI/ML/data science in investment management**; CIO/PM-facing; **no submission fee** | Paywall; preprints on SSRN | https://jfds.pm-research.com/ |
| **The Journal of Portfolio Management (JPM)** | Portfolio Management Research | Factor investing, portfolio construction, ML overlays; Vol. 52 (2026) | Paywall | https://jpm.pm-research.com/ |
| **Algorithmic Finance** | IOS Press | HFT/algo trading, statistical arbitrage, agent-based finance, computational financial intelligence, **quantum finance**, news analytics; **no publication fee** (eISSN 2157-6203) | Open peer review, no APC | https://www.iospress.nl/journal/algorithmic-finance/ |
| **Digital Finance** | Springer | Blockchain, crypto, fintech, digital banking; empirical + computational | Hybrid OA | https://link.springer.com/journal/42521 |
| **The Journal of Finance and Data Science** | Elsevier (KeAi) | Data-science methods across finance (distinct from JFDS) | **Fully open access** | https://www.sciencedirect.com/journal/the-journal-of-finance-and-data-science |
| **Journal of Investment Strategies** | Risk.net / Infopro | Practitioner strategy research; allocation, risk-managed strategies | Paywall | https://www.risk.net/journal-of-investment-strategies |
| **Applied Mathematical Finance** | Taylor & Francis | Applied stochastic methods, pricing, hedging | Paywall | https://www.tandfonline.com/journals/ramf20 |
| **Frontiers of Mathematical Finance** | AIMS | Math finance, control, computational methods | Hybrid | https://www.aimsciences.org/fmf |
| **Revista Brasileira de Finanças (Brazilian Finance Review)** | SBFin | Brazil's official finance journal (Portuguese/English) | **Open access** | https://www.sbfin.org.br/ |

---

## 6. Preprints & research networks (free, fast, primary)

| Network | What | Why use it | Link |
|---|---|---|---|
| **arXiv q-fin** | Quantitative Finance preprint section (covered in depth elsewhere in this repo) | Fastest route to new methods; sub-areas q-fin.TR (trading), q-fin.CP (computational), q-fin.ST (statistical finance), q-fin.PM (portfolio) | https://arxiv.org/archive/q-fin |
| **SSRN — Financial Economics Network (FEN)** | The dominant working-paper repository for finance/econ; FEN founded/directed historically by Michael C. Jensen | Where JF/RFS/JFE papers circulate *before* publication; topical eJournals (incl. AI/ML in finance) | https://www.ssrn.com/index.cfm/en/fen/ · eJournals: https://www.ssrn.com/index.cfm/en/fen/fen-ejournals/ |
| **OpenReview** | Open review for NeurIPS/ICLR/KDD workshops | Accepted ML-finance workshop papers + reviews + code | https://openreview.net/ |
| **RePEc / IDEAS** | Research Papers in Economics index | Cross-index of finance journals & working papers | https://ideas.repec.org/ |
| **Annual Review of Financial Economics** | Annual Reviews | Authoritative survey articles — good entry points | https://www.annualreviews.org/content/journals/financial |

---

## 7. Communities — global (forums, Reddit, platforms)

Where practitioners actually debate, share code, and recruit. Quality varies; signal-to-noise is highest in the moderated/technical venues.

| Community | Type | Focus | Link |
|---|---|---|---|
| **r/algotrading** | Reddit | Beginner-friendly; bots, Python, broker APIs, strategy talk | https://www.reddit.com/r/algotrading/ |
| **r/quant** | Reddit | Career, interviews, quant methods (more rigorous than r/algotrading) | https://www.reddit.com/r/quant/ |
| **r/quantfinance** | Reddit | Careers/education into quant finance | https://www.reddit.com/r/quantfinance/ |
| **QuantConnect Community** | Forum + platform | 280k+ users; LEAN engine support, shared algorithms, backtests | https://www.quantconnect.com/forum/ |
| **Wilmott Forums** | Forum | Long-standing, mature/technical; derivatives, stochastic calculus, model debates | https://forum.wilmott.com/ |
| **Nuclear Phynance** | Forum | Old-school practitioner forum; derivatives & risk (lower traffic now) | https://www.nuclearphynance.com/ |
| **Elite Trader (Quant section)** | Forum | Execution tactics, slippage/latency, microstructure, real-world trading | https://www.elitetrader.com/ |
| **Quantopian Lecture Series (legacy archive)** | Archive | Quantopian shut down 2020, but its **55 notebook lectures + videos** survive — still a top free curriculum | Gist mirror: https://gist.github.com/ih2502mk/50d8f7feb614c8676383431b056f4291 · forum archive: https://quantopian-archive.netlify.app/ |
| **Numerai / Numerai Signals** | Crowdsourced fund + community | Data-science tournament (encrypted data) + Signals (bring-your-own-signal); active Discord/RocketChat, real stakes | https://numer.ai/ · Signals: https://signals.numer.ai/ · docs: https://docs.numer.ai/ |
| **Hudson & Thames (MlFinLab) community** | Slack | ML-in-finance (López de Prado techniques); members' Slack | https://hudsonthames.org/ · https://github.com/hudson-and-thames/mlfinlab |
| **`aifin-worldwide` Google Group** | Mailing list | The ICAIF / AI-in-finance CFP announcement list — subscribe to catch deadlines | https://groups.google.com/g/aifin-worldwide |

---

## 8. Aggregators, blogs & curated lists (the radar layer)

| Resource | Type | Why | Link |
|---|---|---|---|
| **Quantocracy** | Daily blog aggregator ("Quant Mashup") | Curates the best algo/quant blog posts each day — the single best firehose for practitioner writing | https://quantocracy.com/ |
| **Quantitative Trading (Ernie Chan)** | Blog | Long-running practitioner blog by Ernest P. Chan (QTS Capital, PredictNow.ai); pragmatic ML-trading takes | http://epchan.blogspot.com/ · https://epchan.com/ |
| **awesome-quant-ai** | GitHub list | Curated AI/ML-in-finance resources (2024–2026) | https://github.com/leoncuhk/awesome-quant-ai |
| **EliteQuant** | GitHub list | Broad index of quant modelling/trading/portfolio resources | https://github.com/EliteQuant/EliteQuant |
| **financial-machine-learning topic** | GitHub topic | Live feed of trending ML-finance repos | https://github.com/topics/financial-machine-learning |
| **Quants Hub** | Training library | Paid expert lectures (risk, ML, model validation) | https://quantshub.com/ |

---

## 9. Brazil-focused venues & communities (comunidades brasileiras)

Under-indexed internationally but central for the Brazil-heavy audience. Most operate in Portuguese; several offer APIs and free content.

| Venue / community | Type | Focus | Link |
|---|---|---|---|
| **SBFin — Sociedade Brasileira de Finanças** | Academic society | Founded 2001; runs the **Encontro Brasileiro de Finanças** (annual meeting) and publishes the **Revista Brasileira de Finanças**; SBFin School free extension courses | https://www.sbfin.org.br/ · eventos: https://sbfin.org.br/pt/eventos |
| **QuantBrasil** | Platform + newsletter | Family-run (Quintanilha) quantitative-analysis platform: backtests, cointegração, VaR, factor strategies (Magic Formula, momentum); Substack "Code Capital"; Telegram/YouTube — **no Discord** | https://quantbrasil.com.br/ |
| **Trading com Dados** | School + API + community | Brazil's first data-science school focused on the financial market (20k+ alunos); "Python para o Mercado Financeiro" course; **financial-data API** for B3 assets; networking cohort | https://tradingcomdados.com/ |
| **Quantzed** | EdTech | Quant education for Brazilian investors (math/stats/programming) | https://quantzed.com.br/ |
| **Clube de Finanças (UFSC/ESAG)** | University club | Student finance club; algo-trading & derivatives research nucleus; tutorials (e.g. Quantopian) | https://clubedefinancas.com.br/ |

> Brazil access to global products: many strategies discussed in these communities trade **B3** equities/ETFs directly; US exposure is via **BDRs** (Brazilian Depositary Receipts) and B3-listed ETFs (e.g. IVVB11 for S&P 500). Numerai/QuantConnect are usable from Brazil (USD-denominated, online); academic journals are largely paywalled — use **SSRN** and **arXiv q-fin** for free copies. (B3 = Brasil, Bolsa, Balcão; the local exchange is covered in detail elsewhere in this repo.)

---

## 10. How to use this page (a workflow)

1. **Track deadlines:** subscribe to `aifin-worldwide` (ICAIF) and watch the NeurIPS/KDD finance-workshop OpenReview groups — these are the calendar anchors.
2. **Read the frontier free:** arXiv q-fin + SSRN FEN + OpenReview cover ~90% of new methods before paywalled publication.
3. **Validate, don't trust:** workshop/preprint trading results are frequently inflated by leakage — cross-check against the *Quantitative Finance* / RFS published literature.
4. **Plug into one community:** Quantocracy (radar) + one forum (QuantConnect or Wilmott) + one contest (Numerai) is a high-signal starter stack.
5. **Brazil:** SBFin for academia, Trading com Dados / QuantBrasil for applied B3 work.

---

**Sources:** ICAIF'25 https://icaif25.org/ , CFP https://icaif25.org/calls-for-papers/ , proceedings https://dl.acm.org/doi/proceedings/10.1145/3768292 ; NeurIPS'25 GenAI-in-Finance https://sites.google.com/view/neurips-25-gen-ai-in-finance/home , OpenReview https://openreview.net/group?id=NeurIPS.cc/2025/Workshop/GenAI_in_Finance ; KDD-MLF 2025 https://sites.google.com/view/kdd-mlf-2025/ ; QuantMinds https://informaconnect.com/quantminds-international/ ; CFE-CMStatistics 2025 https://www.cmstatistics.org/CFECMStatistics2025/ ; Risk.net https://www.risk.net/quantitative-finance ; FinRL/AI4Finance https://github.com/AI4Finance-Foundation/FinRL ; JFDS https://jfds.pm-research.com/ ; JPM https://jpm.pm-research.com/ ; Quantitative Finance https://www.tandfonline.com/journals/rquf20 ; Mathematical Finance https://onlinelibrary.wiley.com/journal/14679965 ; Finance Research Letters https://www.sciencedirect.com/journal/finance-research-letters ; Algorithmic Finance https://www.iospress.nl/journal/algorithmic-finance/ ; Digital Finance https://link.springer.com/journal/42521 ; Journal of Financial Markets https://www.sciencedirect.com/journal/journal-of-financial-markets ; RFS https://academic.oup.com/rfs ; J. of Finance https://onlinelibrary.wiley.com/journal/15406261 ; SSRN FEN https://www.ssrn.com/index.cfm/en/fen/ ; Quantocracy https://quantocracy.com/ ; Numerai https://numer.ai/ , Signals https://signals.numer.ai/ ; Quantopian archive https://gist.github.com/ih2502mk/50d8f7feb614c8676383431b056f4291 ; Wilmott https://forum.wilmott.com/ ; QuantConnect https://www.quantconnect.com/forum/ ; Ernie Chan https://epchan.com/ ; MlFinLab https://github.com/hudson-and-thames/mlfinlab ; awesome-quant-ai https://github.com/leoncuhk/awesome-quant-ai ; SBFin https://www.sbfin.org.br/ ; QuantBrasil https://quantbrasil.com.br/ ; Trading com Dados https://tradingcomdados.com/ ; Quantzed https://quantzed.com.br/ .

**Keywords:** ML finance conferences, ICAIF, ACM AI in Finance, NeurIPS finance workshop, KDD-MLF, QuantMinds, CFE-CMStatistics, Journal of Financial Data Science, Quantitative Finance journal, Algorithmic Finance, Finance Research Letters, SSRN FEN, arXiv q-fin, quant community, Quantocracy, Numerai, QuantConnect, Wilmott, r/algotrading, r/quant, Ernie Chan, MlFinLab; (Português) conferências de finanças quantitativas, comunidade quant, periódicos de finanças, aprendizado de máquina em finanças, SBFin, QuantBrasil, Trading com Dados, Quantzed, BDR, ETF, B3, negociação algorítmica, finanças quantitativas.
