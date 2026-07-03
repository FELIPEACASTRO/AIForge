# Poker Solvers & Open-Source Tools (Solvers e Ferramentas Open-Source)

Curated, fact-checked catalog of GTO solvers, research frameworks, equity calculators and hand evaluators — for **research, education and off-table study only**. Every repository, site and paper below was verified live in July 2026.

---

## Responsible Gambling First (Jogo Responsável)

> **If gambling stops being fun, seek help — it is free and confidential (se o jogo deixou de ser diversão, procure ajuda gratuita e sigilosa):**
>
> - **Brazil — CVV (Centro de Valorização da Vida):** call **188**, free, 24h nationwide, plus chat/e-mail — [cvv.org.br](https://cvv.org.br/)
> - **Brazil — Jogo Responsável (Secretaria de Prêmios e Apostas / Ministério da Fazenda):** official guidance, self-exclusion and player-protection rules (Portaria SPA/MF nº 1.231/2024; licensed sites use the `.bet.br` domain) — [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel)
> - **UK — National Gambling Helpline (GamCare):** **0808 8020 133**, free, 24/7 — [gamcare.org.uk](https://www.gamcare.org.uk/). Note: the charity GambleAware closed on 31 March 2026; UK prevention/treatment commissioning moved to the government statutory-levy system, and GamCare has stated it intends to keep the helpline running.
> - **International — Gambling Therapy (Gordon Moody charity):** free multilingual online support in 35+ languages — [gamblingtherapy.org](https://www.gamblingtherapy.org/)

**Ethics & legality (leia antes de usar):**

- These tools are for **study away from the tables**. Real-Time Assistance (RTA — using solvers, charts or advice **during** play) and **bots are banned by essentially every poker site**. PokerStars prohibits RTA, solvers, range calculators and solver-derived charts during play and reports a proactive RTA-detection rate above 95%; penalties include permanent ban and confiscation of funds. GGPoker bans all third-party tools (RTA, bots, solvers, charts, external HUDs) with the same consequences.
- OpenHoldem-style bot frameworks are listed **only** as historical/security-research references. Deploying a bot on a real-money site violates the site's terms and may be illegal in your jurisdiction.
- **AGPL-3.0 warning:** TexasSolver and the postflop-solver family are AGPL-licensed — if you build a network service on them you must open-source your service's code.

---

## Commercial Desktop Solvers

| Tool | Site | Games | Notes | Free? |
|---|---|---|---|---|
| **PioSOLVER** | [piosolver.com](https://piosolver.com/) | NLHE postflop (preflop in higher editions) | Industry-standard GTO solver; current line is PioSOLVER 3.x (v3.10 listed July 2026) | No (paid editions) |
| **GTO+** | [gtoplus.com](https://www.gtoplus.com/) | NLHE heads-up postflop | Budget favorite; converges to 0% Nash distance, compact save files; companion to Flopzilla | No (paid, one-time) |
| **MonkerSolver** | [monkerware.com](https://monkerware.com/) | Hold'em & Omaha, **any number of players**, any street | The reference for multiway and PLO trees; heavy RAM requirements | No (paid) |
| **Simple Postflop / Simple Preflop (Simple Poker)** | [simplepoker.com](https://simplepoker.com/) | NLHE preflop + postflop | Suite of GTO solvers with cloud solving; **Simple Nash** (push/fold Nash calculator for SNG/MTT) is completely free | Simple Nash free; solvers paid |

## Cloud Solvers & Trainers (Freemium)

| Tool | Site | What it is | Free tier |
|---|---|---|---|
| **GTO Wizard** | [gtowizard.com](https://gtowizard.com/) | Largest presolved GTO library + AI solver, trainer, hand-history analyzer | Yes — limited (approx. 1 postflop spot/day, 10 practice hands/day); paid plans from ~US$26/mo (2026 pricing) |
| **DTO Poker** | [dtopoker.com](https://www.dtopoker.com/) | GTO trainers for cash and MTT (separate products) built by pros | "Start for free"; paid subscriptions |
| **Odin** | [odinpoker.io](https://odinpoker.io/) | Browser database of 500k+ presolved postflop sims (founded by pro Rory Young); solutions later repackaged under the "GTO Strategy" brand | Paid; **never use during play** — a 2023 real-time feature drew RTA criticism in industry press |
| **RangeConverter** | [rangeconverter.com](https://rangeconverter.com/) | GTO trainer + range viewer for NLHE cash, MTT, spins, PLO/PLO5, short deck | Yes — 15bb heads-up spots free; Reg/Pro tiers paid |

## Open-Source Solvers

| Repo | License | Status (verified Jul 2026) | Notes |
|---|---|---|---|
| [bupticybee/TexasSolver](https://github.com/bupticybee/TexasSolver) | AGPL-3.0 | Maintained (README updated Mar 2026 pointing to a faster GPU fork, *TexasSolverGPU*); last core release v0.2.0 (Nov 2021) | C++ NLHE/short-deck solver with GUI + console; ~2.5k stars; author benchmarks show flop speed competitive with PioSOLVER |
| [b-inary/postflop-solver](https://github.com/b-inary/postflop-solver) | AGPL-3.0 | **Development suspended** (author's note, Oct 2023 — went commercial) | Rust postflop solver engine; backend of the two projects below |
| [b-inary/wasm-postflop](https://github.com/b-inary/wasm-postflop) | AGPL-3.0 | Suspended, but the free hosted app is still live: [wasm-postflop.pages.dev](https://wasm-postflop.pages.dev/) | GTO solver running entirely in the browser via WebAssembly |
| [b-inary/desktop-postflop](https://github.com/b-inary/desktop-postflop) | AGPL-3.0 | Suspended (last release v0.2.7, Oct 2023); still downloadable and functional | Tauri desktop port of the same engine |

## Research Frameworks (RL / CFR)

| Repo | License | Status | Notes |
|---|---|---|---|
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) | Apache-2.0 | **Active** (v1.6.15, May 2026) | Reference framework for game-theory RL; CFR variants on Kuhn/Leduc poker and many other games |
| [uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit) | MIT | **Active** (v0.7.4, May 2026) | Fine-grained multi-variant poker simulation + hand evaluation in pure Python. Paper: J. Kim, "PokerKit: A Comprehensive Python Library for Fine-Grained Multi-Variant Poker Game Simulations," *IEEE Trans. on Games* 17(1), 2025, DOI [10.1109/TG.2023.3325637](https://doi.org/10.1109/TG.2023.3325637) ([arXiv:2308.07327](https://arxiv.org/abs/2308.07327)) |
| [datamllab/rlcard](https://github.com/datamllab/rlcard) | MIT | Low activity (last commit Jun 2024; last release 2022) | RL toolkit for card games (Leduc/Limit/No-Limit Hold'em, Dou Dizhu, Mahjong, UNO) from Rice/TAMU DATA Lab |
| [EricSteinberger/PokerRL](https://github.com/EricSteinberger/PokerRL) | MIT | Stale (last commit Jul 2020) | Multi-agent deep-RL poker framework (NFSP, Deep CFR) with Ray distributed training |
| [EricSteinberger/Deep-CFR](https://github.com/EricSteinberger/Deep-CFR) | MIT | Stale (last commit Apr 2019) | Implements Deep CFR (Brown et al., ICML 2019, [arXiv:1811.00164](https://arxiv.org/abs/1811.00164)) and Single Deep CFR (Steinberger 2019, [arXiv:1901.07621](https://arxiv.org/abs/1901.07621)) |
| [fedden/poker_ai](https://github.com/fedden/poker_ai) | GPL-3.0 | **Archived** (read-only since 16 Jul 2024) | Open-source Texas Hold'em AI trained with MCCFR; Pluribus-inspired |
| [dickreuter/neuron_poker](https://github.com/dickreuter/neuron_poker) | MIT | Maintained (modern Python 3.11 tooling) | OpenAI-Gym-style Hold'em environment with keras-rl agents and Monte Carlo equity |
| [OpenHoldem/openholdembot](https://github.com/OpenHoldem/openholdembot) | GPL-3.0 | Legacy (last release Dec 2021) | Historical screen-scraping bot framework. **Reference only — running bots on real-money sites is banned and gets accounts closed and funds seized** |

## Equity Calculators

| Tool | Where | Type | Free? |
|---|---|---|---|
| **Equilab (PokerStrategy.com)** | [pokerstrategy.com — Equilab](https://www.pokerstrategy.com/poker-software-tools/equilab-holdem/) | Windows GUI, Hold'em (separate Omaha edition) — equity vs. ranges, scenario analyzer, equity trainer | Yes (free) |
| **Power-Equilab** | [power-equilab.com](https://www.power-equilab.com/) | Advanced successor by the same developer: weighted ranges, multiway, heatmaps | Paid (14-day free trial) |
| **Simple Nash (Simple Poker)** | [simplepoker.com](https://simplepoker.com/) | Push/fold Nash-equilibrium calculator for SNG/MTT | Yes (free) |
| [andrewprock/pokerstove](https://github.com/andrewprock/pokerstove) | GitHub, BSD-3-Clause | Classic C++ equity library + CLI (ps-eval); 14 variants; maintenance release Jan 2024 | Yes (open source) |
| [ihendley/treys](https://github.com/ihendley/treys) | GitHub, MIT | Pure-Python 5/6/7-card evaluator (Deuces port); last commit Mar 2023, stable | Yes (open source) |
| [ktseng/holdem_calc](https://github.com/ktseng/holdem_calc) | GitHub, MIT | Python Hold'em win-probability calculator (exact or Monte Carlo) | Yes (open source) |
| [cookpete/poker-odds](https://github.com/cookpete/poker-odds) | GitHub, MIT | Zero-dependency JavaScript CLI for hand-vs-hand odds | Yes (open source) |

## Hand Evaluators (Low-Level Libraries)

| Repo | License | Notes |
|---|---|---|
| [HenryRLee/PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) | Apache-2.0 | `phevaluator` — perfect-hash evaluator, ~144 KB tables for 7-card; supports PLO4/5/6; C/C++ and Python official, community ports in JS/Go/Rust/C# and more |
| [zekyll/OMPEval](https://github.com/zekyll/OMPEval) | ISC | Fast C++ 7-card evaluator + equity calculator (Monte Carlo and full enumeration, up to 6 players); long dormant but widely reused |
| [tangentforks/XPokerEval](https://github.com/tangentforks/XPokerEval) | (no license file) | Frozen one-commit archive of the codingthewheel.com evaluator roundup: Cactus Kev, TwoPlusTwo lookup, Senzee7, PokerSource and others |
| [chenosaurus/poker-evaluator](https://github.com/chenosaurus/poker-evaluator) | (see repo) | Node.js evaluator using the TwoPlusTwo algorithm and bundled `HandRanks.dat` lookup table (`npm install poker-evaluator`) |
| [uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit) | MIT | Also ships high-quality evaluators for many variants (see Research Frameworks above) |

## Choosing a Stack (free option first)

| Goal | Free pick | Paid upgrade |
|---|---|---|
| Learn what a solver output looks like | [wasm-postflop.pages.dev](https://wasm-postflop.pages.dev/) in the browser | GTO Wizard (presolved library + trainer) |
| Solve custom NLHE postflop trees locally | bupticybee/TexasSolver | PioSOLVER or GTO+ |
| PLO / multiway trees | — (no solid free option) | MonkerSolver |
| Push/fold & preflop tournament spots | Simple Nash | Simple Preflop / GTO Wizard MTT |
| Write a poker AI research paper | OpenSpiel + PokerKit (+ RLCard baselines) | — |
| Equity calculations in your own code | phevaluator, treys, pokerstove | — |
| Drill ranges as a human | RangeConverter free tier, GTO Wizard free tier | DTO Poker, paid tiers |

---

**Sources:** [piosolver.com](https://piosolver.com/) · [gtoplus.com](https://www.gtoplus.com/) · [monkerware.com](https://monkerware.com/) · [simplepoker.com](https://simplepoker.com/) · [gtowizard.com](https://gtowizard.com/) · [GTO Wizard plans (PokerNews, 2026)](https://www.pokernews.com/news/2026/03/gto-wizard-subscription-plans-new-features-pricing-50908.htm) · [dtopoker.com](https://www.dtopoker.com/) · [odinpoker.io](https://odinpoker.io/) · [Odin RTA coverage (VegasSlotsOnline, 2023)](https://www.vegasslotsonline.com/news/2023/03/08/poker-trainer-odin-just-became-an-rta-tool/) · [rangeconverter.com](https://rangeconverter.com/) · [TexasSolver](https://github.com/bupticybee/TexasSolver) · [postflop-solver](https://github.com/b-inary/postflop-solver) · [wasm-postflop](https://github.com/b-inary/wasm-postflop) · [desktop-postflop](https://github.com/b-inary/desktop-postflop) · [wasm-postflop app](https://wasm-postflop.pages.dev/) · [open_spiel](https://github.com/google-deepmind/open_spiel) · [pokerkit](https://github.com/uoftcprg/pokerkit) · [PokerKit paper DOI](https://doi.org/10.1109/TG.2023.3325637) · [arXiv:2308.07327](https://arxiv.org/abs/2308.07327) · [rlcard](https://github.com/datamllab/rlcard) · [PokerRL](https://github.com/EricSteinberger/PokerRL) · [Deep-CFR](https://github.com/EricSteinberger/Deep-CFR) · [arXiv:1811.00164](https://arxiv.org/abs/1811.00164) · [arXiv:1901.07621](https://arxiv.org/abs/1901.07621) · [fedden/poker_ai](https://github.com/fedden/poker_ai) · [neuron_poker](https://github.com/dickreuter/neuron_poker) · [openholdembot](https://github.com/OpenHoldem/openholdembot) · [pokerstove](https://github.com/andrewprock/pokerstove) · [treys](https://github.com/ihendley/treys) · [holdem_calc](https://github.com/ktseng/holdem_calc) · [poker-odds](https://github.com/cookpete/poker-odds) · [PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) · [OMPEval](https://github.com/zekyll/OMPEval) · [XPokerEval](https://github.com/tangentforks/XPokerEval) · [poker-evaluator](https://github.com/chenosaurus/poker-evaluator) · [Equilab (PokerStrategy)](https://www.pokerstrategy.com/poker-software-tools/equilab-holdem/) · [Power-Equilab](https://www.power-equilab.com/) · [PokerStars prohibited tools](https://www.pokerstars.com/poker/room/prohibited/) · [GGPoker security & ecology policy](https://ggpoker.com/network/security-ecology-policy/) · [gov.br Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [cvv.org.br](https://cvv.org.br/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [gamblingtherapy.org](https://www.gamblingtherapy.org/)

**Keywords:** poker solver, GTO solver, open-source poker AI, CFR, counterfactual regret minimization, equity calculator, hand evaluator, poker RL framework, responsible gambling / solver de poker, teoria dos jogos, calculadora de equity, avaliador de mãos, IA de poker, jogo responsável, apostas regulamentadas bet.br
