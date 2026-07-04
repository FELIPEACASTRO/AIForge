# Omaha Solvers & Software (Solvers e Softwares de Omaha)

A fact-checked, deep catalog of every **Omaha-capable** solver, presolved library, GTO trainer, equity calculator and open-source engine — for **Pot-Limit Omaha (PLO)**, **5-card PLO (PLO5)**, **6-card PLO (PLO6)** and **Omaha Hi-Lo (Omaha-8)**. This is the tooling companion to the sibling [Omaha Strategy & GTO](./Omaha_Strategy_and_GTO.md) page; it goes **much** deeper on the *software landscape* than the brief [Poker Variants](../Poker_Variants_PLO_and_Mixed_Games.md) overview and complements the Hold'em-centric parent [Solvers & Open-Source Tools](../Solvers_and_Open_Source_Tools.md). Every site, price, repository and free tier below was verified live in **July 2026**; vendor pages that could not be confirmed against a primary source had their numbers dropped rather than guessed. For **research, education and off-table study only.**

> **Why Omaha tooling is its own category (por que as ferramentas de Omaha são um caso à parte):** Omaha's game tree is enormous — **270,725** four-card starting hands (C(52,4)), roughly *200× more preflop combinations than Hold'em*, and far more again for PLO5/PLO6. Full, unabstracted solving that is routine in NLHE is still **expensive, RAM-hungry and mostly limited to PLO4** in Omaha. As a result the Omaha software world splits into (1) heavy **desktop CFR solvers** (MonkerSolver), (2) **presolved cloud libraries + trainers** (GTO Wizard PLO, PLO Genius, RangeConverter, Vision, PLO Mastermind), (3) **equity/analysis calculators** (ProPokerTools, PLO.com, Equilab Omaha), and (4) **open-source research code** (PokerKit, phevaluator, PokerRL-Omaha). Coverage of PLO5/PLO6 is *thin everywhere* — treat those solutions as directional.

---

## Responsible Gambling First (Jogo Responsável)

> **Omaha is normally played for real money and its variance is *structurally higher* than No-Limit Hold'em — preflop equities run closer together, pots grow faster under pot-limit, and bankroll swings are larger (PLO standard deviation typically runs well above NLHE at the same stake, and split-pot Hi-Lo adds "getting quartered" on top). After rake, the player pool loses net.** Owning a solver does not make Omaha beatable if you play stakes your bankroll cannot absorb. If gambling stops being fun, seek help — it is free and confidential (se o jogo deixou de ser diversão, procure ajuda gratuita e sigilosa):
>
> | Service | Where | Contact |
> |---|---|---|
> | **CVV — Centro de Valorização da Vida** | Brazil 🇧🇷 | Call **188** (free, 24h) or chat — [cvv.org.br](https://cvv.org.br/) |
> | **Jogo Responsável (SPA / Ministério da Fazenda)** | Brazil 🇧🇷 | Player-protection rules, self-exclusion; licensed sites use the `.bet.br` domain — [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) |
> | **National Gambling Helpline (GamCare)** | UK | **0808 8020 133**, free, 24/7 — [gamcare.org.uk](https://www.gamcare.org.uk/) · consumer info [gambleaware.org](https://www.gambleaware.org/) |
> | **Gambling Therapy (Gordon Moody)** | International | Free multilingual online support, **including Português (Brasil)** — [gamblingtherapy.org/pt-br](https://www.gamblingtherapy.org/pt-br/) |
>
> **Bots and Real-Time Assistance (RTA) are banned by essentially every real-money poker site** — accounts are closed and funds confiscated. Every solver, trainer, range library and evaluator on this page is for **study away from the tables**, never for live decision help. Memorizing solver output is legal; consulting it mid-hand is not.

---

## The Big Comparison Table (tabela comparativa)

Coverage columns show **PLO4 / PLO5 / PLO6** support; ✅ = supported, ➖ = not / not yet, ✳️ = partial or in development. "Free option" = whether a genuinely free tier or free download exists (distinct from a time-limited trial).

| Tool | Type | PLO4 | PLO5 | PLO6 | Cash / MTT | Pricing (verified Jul 2026) | Free option |
|---|---|:--:|:--:|:--:|---|---|:--:|
| **[MonkerSolver](https://monkerware.com/solver.html)** | Desktop CFR solver | ✅ | ✳️ | ✳️ | Cash + MTT trees, **any # of players (multiway)** | **€499** one-time full license | ✅ (free build, limited to **turn & river**) |
| **[GTO Wizard PLO](https://gtowizard.com/plo/)** | Presolved cloud library + trainer | ✅ | ➖ | ➖ | Cash (6-max/7-max, 50/100bb) + MTT (10–100bb) | **Premium $139/mo**, **Elite $179/mo** (early-bird, billed annually; standard $179 / $239) | ✅ (Free plan: ~1 postflop spot/day) |
| **[PLO Genius](https://plogenius.com/)** | Cloud solver + trainer + libraries | ✅ | ✅ | ✅ | Cash, MTT, HU, live | **Edge $59/mo** ($708/yr), **PRO $125/mo** ($1,500/yr) | ✅ (Starter tier, free) |
| **[RangeConverter](https://rangeconverter.com/)** | Presolved ranges + GTO trainer | ✅ | ✅ | ✅ | Cash + MTT + more | **Range Reg $29/mo** ($239/yr); **Range Pro $99/mo** ($799/yr, cash/MTT/PLO/short-deck), **Spins $49/mo** ($399/yr) | ✅ (15bb HU viewer + trainer demo) |
| **[Vision GTO Trainer](https://www.runitonce.com/vision-gto-trainer/)** (Run It Once) | GTO drill trainer | ✅ | ✅ | ➖ | Cash-oriented (preset boards) | Paid standalone subscription (see below) | ➖ |
| **[PLO Mastermind](https://plomastermind.com/)** | Training site + GTO trainer | ✅ | ✅ | ➖ | Cash-focused study | **$199/mo** per format; ~20% off annual | ✅ (free membership tier) |
| **[PLO+ (ThinkGTO)](https://thinkgto.com/apps/plo-plus)** | Mobile preflop range app | ✅ | ➖ | ➖ | Cash + MTT preflop | Mobile app (App Store / Google Play) | ✳️ (app-store listing) |
| **[ProPokerTools / Odds Oracle](https://www.propokertools.com/)** | Equity + PQL calculator | ✅ | ✅ | ➖ | Equity/range analysis (all formats) | Free **online** tools; Odds Oracle **desktop** download | ✅ (free web tools) |
| **[PLO.com](https://plo.com/)** | Equity + browser "Solver Room" | ✅ | ✅ | ✅ | Study/equity | Free core equity tools (freemium) | ✅ |
| **[Equilab Omaha](https://www.pokerstrategy.com/poker-software-tools/equilab-omaha/)** | Equity calculator (Windows) | ✅ | ➖ | ➖ | Equity vs ranges | Free | ✅ |
| **[phevaluator](https://github.com/HenryRLee/PokerHandEvaluator)** | Open-source hand evaluator | ✅ | ✅ | ✅ | (library) | Apache-2.0 (free) | ✅ |
| **[PokerKit](https://github.com/uoftcprg/pokerkit)** | Open-source game engine | ✅ | ✳️ | ✳️ | (simulation) | MIT (free) | ✅ |

> **A note on Flopzilla:** the popular flop-texture/range tool **Flopzilla (and Flopzilla Pro)** is **Texas Hold'em only** — its own site bills it as "analysis software for Hold'em," and the **US$25** one-time purchase (which "includes both FlopzillaPro and Flopzilla v1") does **not** add Omaha. For "Flopzilla-style" Omaha flop-distribution and range-equity analysis, use **ProPokerTools / Odds Oracle**, **PLO.com**, **Equilab Omaha**, or the built-in equity engines in **PLO Genius** and **GTO Wizard PLO** instead.

---

## Desktop Solvers (solvers de mesa)

### MonkerSolver — the multiway/PLO reference engine

**[MonkerSolver](https://monkerware.com/solver.html)** (from MonkerWare) is the long-standing heavyweight for Omaha. Its defining feature is that it can **"Solve Omaha and Hold'em from any street with any number of players"** — i.e. genuine **multiway (potes multiway)** CFR solving, which almost no other solver offers and which matters enormously in Omaha because PLO pots go multiway far more than NLHE.

- **Price:** **€499** one-time for the full version (buy after registering/logging in on the site).
- **Free version:** a limited free build exists, **restricted to the turn and river** only — enough to learn the interface and study late-street spots.
- **System requirements:** Windows 64-bit or macOS, **at least 8 GB RAM**; tree size scales with available RAM and solve speed scales with CPU. Large multiway PLO trees are famously memory-hungry.
- **Companion:** **[MonkerViewer](https://www.monkerware.com/)** lets you "conveniently view your library of preflop ranges" and "purchase preflop ranges from the cloud and have access to them from any computer or android phone." Third-party libraries of Monker-solved **NLHE and PLO** preflop ranges are sold separately by **[MonkerGuy](https://www.monkerguy.com/)** (delivered as MonkerViewer charts, full `.mkr` sims and PioViewer-compatible `.txt` files).

MonkerSolver remains the go-to when you specifically need **multiway** or **mixed-game / 5-card** trees that the presolved cloud libraries do not cover.

---

## Cloud Solvers, Presolved Libraries & Trainers (solvers na nuvem e treinadores)

### GTO Wizard PLO

**[GTO Wizard PLO](https://gtowizard.com/plo/)** is a **presolved** cloud library plus trainer and hand-history analyzer, sold as a **separate subscription** from GTO Wizard's Hold'em product.

- **Coverage:** **PLO4 only** — the site states **PLO5 and PLO6 are "in development with no ETA at this time."** Solutions span **6-max and 7-max cash** (50bb and 100bb) plus **6-max MTT chip-EV from 10bb to 100bb**, across **all 1,755 flop textures** with full runouts.
- **Solve quality:** advertised **River Nash Distance < 0.1%** ("Rivers solved exactly. No abstractions"), average turn EV loss ~0.14%.
- **Pricing:** **Free** plan ($0, no card — roughly 1 postflop spot and 5 trainer hands per day). Paid: **Premium $139/month** and **Elite $179/month** at early-bird rates *billed annually* (standard prices $179 and $239); Elite adds custom solving, custom sizes/ranges, custom ICM and node editing.

### PLO Genius

**[PLO Genius](https://plogenius.com/)** brands itself a "modern Pot Limit Omaha Solver" and is the widest on card counts — it covers **PLO4, PLO5 and PLO6** (PLO6 preflop simulations were recently added), across cash, MTT, heads-up and live formats, with a fast-growing library (10M+ calculations, ~20k added weekly) and a **PLO Trainer**.

- **Starter — free forever:** limited preflop calculations (100bb for HU/6-max PLO and 6-max PLO5).
- **Edge — $59/month** ($708/year): full preflop for all formats (6-max, HU, MTT & Live), PLO5 included.
- **PRO — $125/month** ($1,500/year): everything in Edge **plus PLO6**, on-demand solver, import options and aggregated reports.

### RangeConverter (PLO / PLO5 / PLO6)

**[RangeConverter](https://rangeconverter.com/)** is a GTO trainer + range viewer whose **PLO Pro** product now includes **6-card Omaha solved preflop ranges** in its Range Viewer, alongside **PLO** and **PLO5** (and NLHE cash, MTT, spins, short-deck and straddle).

- **Free:** full Range Viewer for **15bb heads-up** ranges + a GTO Trainer demo.
- **Range Reg — $29/month** ($239/year) — on sale from a $49/mo ($399/yr) list price.
- **Range Pro — $99/month** ($799/year) covers **Cash, MTT, Short Deck and the PLO variants**; **Spins** is the cheaper tier at **$49/month** ($399/year).

### Vision GTO Trainer (Run It Once)

**[Vision GTO Trainer](https://www.runitonce.com/vision-gto-trainer/)**, built by **Phil Galfond** through **[Run It Once](https://www.runitonce.com/)**, is a dedicated PLO **drill trainer** — its product page bills it as a study tool for **4- and 5-card PLO**. At launch (Feb 2020) it shipped with **300+ preset boards and 6,000+ solver simulations**, a customizable **quiz generator**, and mobile-ready study, letting you test GTO decisions without running full sims each time.

- **Pricing (documented launch pricing, Feb 2020):** 6-max PLO or heads-up PLO editions **$129.99/month** each; **combined $199.99/month**; **25% off annually**. It is a **paid standalone subscription** with **no free tier**; the range has since expanded (5-card PLO), so confirm the **current** price on the [official Vision page](https://www.runitonce.com/vision-gto-trainer/). (Run It Once's main training memberships — Essential and Elite — are separate; annual Elite has at times bundled a free month of Vision.)

### PLO Mastermind

**[PLO Mastermind](https://plomastermind.com/)** is a subscription **training platform** (theory + practical strategy videos, quizzes, a **GTO Trainer**, and a coaching community) with dedicated **4-Card and 5-Card** memberships.

- **Paid:** **$199/month** per format — 4-Card PLO and 5-Card PLO are **separate memberships** (a bundle combining both runs **~$299/month**); annual plans run roughly **20% cheaper** (about **$1,910/year** single-format, **$2,870/year** for the combo). *(One third-party review headlines a "$249/month" figure; the vendor's own pricing page lists $199/month per format — the figure used here.)*
- **Free membership:** a genuinely free tier (the **"Preflop Pass"**) unlocks **100bb 6-max preflop charts**, a working **[PLO equity calculator](https://plomastermind.com/plo-equity-calculator/)** and rotating sample video content — email sign-up, no card required.

### PLO+ (ThinkGTO)

**[PLO+](https://thinkgto.com/apps/plo-plus)** by ThinkGTO is a **mobile app** (App Store / Google Play) putting **GTO-optimal PLO preflop ranges** for every position and stack depth in your pocket, plus a **4-card equity calculator** (Monte Carlo) and training drills, with separate cash and MTT ranges. It currently supports **4-card PLO only** ("PLO5 support is on our roadmap for a future update"). App-store pricing was not independently confirmed at the primary source, so no price is asserted here — check the store listing.

---

## Equity & Range-Analysis Calculators (calculadoras de equity e ranges)

### ProPokerTools / Odds Oracle

**[ProPokerTools](https://www.propokertools.com/)** is a cornerstone Omaha analysis tool. The website offers **free, no-download online tools** — an equity calculator and range explorer for **Hold'em, Omaha, Omaha Hi-Lo, 5-card Omaha (hi & hi-lo), 7-card Stud (hi & hi-lo) and Razz** — and a downloadable desktop app, the **Odds Oracle** (Windows, Mac, Linux), which adds all-in equity calculations, equity graphs, a range explorer, reverse/shove-equity tools, a hand-history importer and the **PQL (ProPokerTools Query Language)** interpreter for scripting precise "how often does X happen?" simulations. Documentation lives at the [Odds Oracle PQL docs](https://www.propokertools.com/oracle_help/pql). Its advanced **range syntax** is influential enough that the open-source parser below exists specifically to read it.

### PLO.com

**[PLO.com](https://plo.com/)** offers a **free** equity calculator supporting **PLO4, PLO5 and PLO6** with **board and dead-card control**, a browser-based **"Solver Room"** for pre/postflop study, a daily **"Spot of the Day,"** and a library of strategy studies. It is a clean, no-install starting point for exact Omaha equities across all three card counts.

### Equilab Omaha & other free equity tools

- **[Equilab Omaha](https://www.pokerstrategy.com/poker-software-tools/equilab-omaha/)** (PokerStrategy.com) — a **free** Windows GUI equity calculator for Omaha (a separate build from the Hold'em [Equilab](https://www.pokerstrategy.com/poker-software-tools/equilab-holdem/)); good for equity-vs-range work and drills.
- **[Run It Once — PLO Odds Calculator](https://www.runitonce.com/tools/plocalculator/)** and **[PLO Mastermind — PLO Equity Calculator](https://plomastermind.com/plo-equity-calculator/)** — free, browser-based PLO equity calculators from the respective training brands.

---

## Open-Source Omaha (código aberto no GitHub)

Omaha is under-served in open source compared to Hold'em, but a handful of solid, verified projects exist — evaluators, a game engine, and research CFR agents.

| Repo | License | What it is (verified Jul 2026) |
|---|---|---|
| **[HenryRLee/PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator)** | Apache-2.0 | `phevaluator` — fast perfect-hash evaluator explicitly supporting **PLO4, PLO5 and PLO6** (evaluation "from 5-card hands to 7-card hands, as well as Omaha poker hands, including PLO4, PLO5, and PLO6"). Official C/C++ and Python, plus community ports. The backbone for Omaha equity/enumeration scripts. |
| **[uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit)** | MIT | Pure-Python engine that **simulates Pot-Limit Omaha Hold'em** (and many other variants) plus high-quality hand evaluators. Actively maintained; ideal for building your own Omaha analysis or RL environments. |
| **[diditforlulz273/PokerRL-Omaha](https://github.com/diditforlulz273/PokerRL-Omaha)** | MIT | Research fork of *PokerRL* adding **Omaha functionality** (Pot-Limit Omaha, **2–6 players**) with **Deep CFR / SD-CFR**, GPU-CPU distributed training, preflop bucketing and PokerStars-format hand logging (~73 stars). One of the only open-source Omaha-RL codebases. |
| **[robb17/PLOAI](https://github.com/robb17/PLOAI)** | (no explicit license) | A Pot-Limit Omaha **game-playing agent** via game abstraction + **CFR / Monte-Carlo CFR**, with a fast Monte-Carlo hand evaluator (Python + C). Small research project; study/reference code, not a maintained product. |
| **[tredfern0/pptparser](https://github.com/tredfern0/pptparser)** | MIT | Python utility that **parses ProPokerTools-style PLO range syntax** into card masks (numpy arrays) — glue code for anyone scripting Omaha equity work against PPT-style ranges. |

Deeper Hold'em-focused solver/evaluator coverage (TexasSolver, postflop-solver, OpenSpiel, RLCard, PokerStove, OMPEval, etc.) lives on the parent [Solvers & Open-Source Tools](../Solvers_and_Open_Source_Tools.md) page.

---

## Choosing a Stack (free option first) — escolhendo suas ferramentas

| Goal | Free pick | Paid upgrade |
|---|---|---|
| Exact PLO4 / PLO5 / PLO6 **equities** in-browser | [PLO.com](https://plo.com/), [ProPokerTools online](https://www.propokertools.com/), Equilab Omaha | Odds Oracle desktop; PLO Genius / GTO Wizard PLO engines |
| See what a **presolved PLO4 solution** looks like | [GTO Wizard PLO — Free plan](https://gtowizard.com/plo/) | GTO Wizard PLO Premium/Elite |
| **PLO5 / PLO6** presolved preflop | [PLO Genius Starter](https://plogenius.com/), [RangeConverter free viewer](https://rangeconverter.com/) | PLO Genius Edge/PRO; RangeConverter PLO Pro |
| Solve **custom / multiway** Omaha trees locally | [MonkerSolver free build](https://monkerware.com/solver.html) (turn/river) | MonkerSolver full (€499) |
| **Drill** GTO PLO as a human | [PLO Mastermind free tier](https://plomastermind.com/), [RangeConverter demo](https://rangeconverter.com/) | Vision GTO Trainer; PLO Mastermind; PLO Genius Trainer |
| **Script / build** your own Omaha analysis | [phevaluator](https://github.com/HenryRLee/PokerHandEvaluator), [PokerKit](https://github.com/uoftcprg/pokerkit), [pptparser](https://github.com/tredfern0/pptparser) | — |
| **Research** an Omaha AI | [PokerRL-Omaha](https://github.com/diditforlulz273/PokerRL-Omaha), [PLOAI](https://github.com/robb17/PLOAI) | — |

**A caution on "GTO" in Omaha:** because the tree is so large, most public PLO solutions rely on abstractions, are limited to **PLO4**, or cover only heads-up / limited multiway spots — and PLO5/PLO6 coverage is thinner still. Treat solver outputs as **directional study** (nut-focus, blocker logic, sizing families), not memorized answers, especially multiway and in the bigger variants. And again: these are **off-table study tools** — using any of them for **Real-Time Assistance (RTA)** during real-money play is banned and gets accounts closed and funds seized.

---

**Related:** Sibling Omaha pages → [`Omaha/`](./) · [Omaha Variants & Rules](./Omaha_Variants_and_Rules.md) · [Omaha Strategy & GTO](./Omaha_Strategy_and_GTO.md) · [Omaha Math, Combinatorics & Equity](./Omaha_Math_Combinatorics_and_Equity.md) · Parent section → [Poker AI & Game Theory](../README.md) · Broader context → [Poker Variants — PLO, Short-Deck & Mixed Games](../Poker_Variants_PLO_and_Mixed_Games.md) · Hold'em tooling → [Solvers & Open-Source Tools](../Solvers_and_Open_Source_Tools.md) · Human study → [Strategy, Training & Community](../Strategy_Training_and_Community.md) · Trackers → [Tracking Software, HUDs & Analytics](../Tracking_Software_HUDs_and_Analytics.md)

**Sources:** [monkerware.com — MonkerSolver](https://monkerware.com/solver.html) · [monkerware.com](https://www.monkerware.com/) · [monkerguy.com](https://www.monkerguy.com/) · [GTO Wizard PLO](https://gtowizard.com/plo/) · [PLO Genius](https://plogenius.com/) · [RangeConverter](https://rangeconverter.com/) · [Run It Once — Vision GTO Trainer](https://www.runitonce.com/vision-gto-trainer/) · [PokerNews — Run It Once launches Vision (2020)](https://www.pokernews.com/news/2020/02/run-it-once-launches-vision-pot-limit-omaha-gto-trainer-36649.htm) · [PLO Mastermind](https://plomastermind.com/) · [PLO Mastermind — Pricing](https://plomastermind.com/pricing/) · [PLO Mastermind — PLO Equity Calculator](https://plomastermind.com/plo-equity-calculator/) · [ThinkGTO — PLO+](https://thinkgto.com/apps/plo-plus) · [ProPokerTools](https://www.propokertools.com/) · [Odds Oracle PQL docs](https://www.propokertools.com/oracle_help/pql) · [PLO.com](https://plo.com/) · [Flopzilla — Purchase](https://www.flopzilla.com/purchase/) · [Equilab Omaha (PokerStrategy)](https://www.pokerstrategy.com/poker-software-tools/equilab-omaha/) · [Run It Once — PLO Odds Calculator](https://www.runitonce.com/tools/plocalculator/) · [HenryRLee/PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) · [uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit) · [diditforlulz273/PokerRL-Omaha](https://github.com/diditforlulz273/PokerRL-Omaha) · [robb17/PLOAI](https://github.com/robb17/PLOAI) · [tredfern0/pptparser](https://github.com/tredfern0/pptparser) · [gov.br — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [cvv.org.br](https://cvv.org.br/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [gambleaware.org](https://www.gambleaware.org/) · [gamblingtherapy.org/pt-br](https://www.gamblingtherapy.org/pt-br/)

**Keywords:** Omaha solver, PLO solver, Pot-Limit Omaha software, MonkerSolver, MonkerWare, GTO Wizard PLO, PLO Genius, RangeConverter PLO, Vision GTO Trainer, Run It Once, PLO Mastermind, PLO+ ThinkGTO, ProPokerTools, Odds Oracle, PQL, PLO.com, Equilab Omaha, Flopzilla, PLO4 PLO5 PLO6, 5-card Omaha, 6-card Omaha, Omaha Hi-Lo, presolved ranges, GTO trainer, equity calculator, phevaluator, PokerKit, PokerRL-Omaha, CFR, pricing, free tier, responsible gambling / solver de Omaha, software de Pot-Limit Omaha, calculadora de equity PLO, ranges pré-solvidos, treinador GTO, biblioteca na nuvem, código aberto, avaliador de mãos, preços e planos gratuitos, jogo responsável, apostas regulamentadas bet.br
