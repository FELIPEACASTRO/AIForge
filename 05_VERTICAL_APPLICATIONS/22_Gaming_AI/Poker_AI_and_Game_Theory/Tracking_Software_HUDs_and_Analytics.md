# Poker Tracking Software, HUDs & Analytics (Trackers, HUDs e Análise de Dados)

Fact-checked catalog of poker tracking databases, heads-up displays (HUDs), tournament analytics and the data-mining research behind them — for **research, education and off-table study only**. Every tool, site policy and paper below was verified live in July 2026.

---

## Responsible Gambling First (Jogo Responsável)

> **If gambling stops being fun, seek help — it is free and confidential (se o jogo deixou de ser diversão, procure ajuda gratuita e sigilosa):**
>
> - **Brazil — CVV (Centro de Valorização da Vida):** call **188**, free, 24h nationwide, plus chat/e-mail — [cvv.org.br](https://cvv.org.br/)
> - **Brazil — Jogadores Anônimos (Gamblers Anonymous):** free weekly meetings (in-person and Zoom), WhatsApp helpline (21) 99472-1933 — [jogadoresanonimos.com.br](https://jogadoresanonimos.com.br/)
> - **Brazil — Jogo Responsável (Secretaria de Prêmios e Apostas / Ministério da Fazenda):** official player-protection guidance and self-exclusion rules; licensed operators use the `.bet.br` domain — [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel)
> - **UK — National Gambling Helpline (GamCare):** **0808 8020 133**, free, 24/7 — [gamcare.org.uk](https://www.gamcare.org.uk/)
> - **International — Gambling Therapy (Gordon Moody charity):** free multilingual online support — [gamblingtherapy.org](https://www.gamblingtherapy.org/)

**Ethics & legality (leia antes de usar):**

- Trackers and HUDs are **legal on some sites and account-closing offenses on others** — the policy table below is the heart of this page. Always read the current terms of service of the site you play on; policies change without notice.
- **Bots and Real-Time Assistance (RTA) are banned by essentially every real-money poker site**, with penalties up to permanent ban and confiscation of funds. Everything here is for **study away from the tables** and for academic research in sandboxes such as OpenSpiel and RLCard — never for automated or assisted real-money play.
- Data-mining hand histories you did not play in (buying/sharing "mined" hands) violates the terms of most sites even where HUDs are allowed.
- In Brazil, fixed-odds betting and online gaming are regulated by **Lei nº 14.790/2023** (29 Dec 2023), with licensed operation from 1 Jan 2025 under Ministério da Fazenda authorization ([planalto.gov.br](https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2023/lei/L14790.htm)). Check your local rules before playing anywhere.

---

## What Trackers and HUDs Do (O que fazem)

A **tracker** imports your hand histories into a local database and produces winrate reports, leak analysis and hand replays. A **HUD** (heads-up display) overlays live statistics (VPIP, PFR, 3-bet% …) on each opponent at the table, computed from hands you have played against them. Analytically they are applied databases: aggregation pipelines + positional filters over event logs of poker actions.

## Trackers & HUDs (verified July 2026)

| Tool | Site | Platform | Pricing model | Trial |
|---|---|---|---|---|
| **PokerTracker 4** | [pokertracker.com](https://www.pokertracker.com/) | Windows + macOS | Paid license | 14-day free trial |
| **Hold'em Manager 3** | [holdemmanager.com](https://www.holdemmanager.com/hm3/) | Windows only (no native macOS) | Paid license; limited free version after trial | 14-day free trial |
| **DriveHUD 2** | [drivehud.com](https://drivehud.com/dh2-pricing-recommendations/) | Windows | 1-year licenses: US$23.99 (micro), US$45.99 (small stakes), US$74.99 (Pro, all stakes); cheap yearly renewals | 30-day free trial |
| **Hand2Note 4** | [hand2note.com](https://hand2note.com/) | Windows + macOS (Apple Silicon and Intel) | Paid; positional/dynamic stats focus, popular with database-analysis pros | Yes (see site) |
| **Poker Copilot** | [pokercopilot.com](https://pokercopilot.com/) | macOS (10.11+) + Windows — long the reference tracker for Mac | One-time: US$29 (micro) / US$49 (small) / US$99 (all stakes) | 14-day free trial, no card |
| **Jurojin Poker** | [jurojinpoker.com](https://jurojinpoker.com/) | Windows | Multi-tabling overlays + HUD; **free at micro stakes** (up to NL10 cash, $5 MTTs, $2 spins), paid above | Free tier |

**Industry note:** PokerTracker and Hold'em Manager have been owned by the same company since August 2014, when PokerTracker Software and CardRunners Gaming merged to form **Max Value Software, LLC** ([official announcement](https://www.pokertracker.com/announcement/pokertracker-holdemmanager-merger.php), [PokerNews](https://www.pokernews.com/news/2014/08/pokertracker-and-holdem-manager-merge-19100.htm)). DriveHUD, Hand2Note, Poker Copilot and Jurojin are independent of that merger.

## HUD Stats Glossary (Glossário de Estatísticas)

| Stat | Name | What it measures |
|---|---|---|
| **VPIP** | Voluntarily Put $ In Pot | % of hands where the player voluntarily invested chips preflop (call/raise; posting blinds doesn't count). Core looseness measure |
| **PFR** | Preflop Raise | % of hands raised preflop. The VPIP–PFR gap separates raisers from callers |
| **3-Bet%** | Three-bet | How often the player re-raises over an open raise preflop |
| **AF / AFq** | Aggression Factor / Frequency | AF = (bets + raises) ÷ calls postflop; AFq = aggressive actions ÷ all actions. Passive vs. aggressive tendency |
| **C-Bet% / Fold-to-CBet%** | Continuation bet | How often the preflop aggressor bets the flop, and how often a player folds to that bet |
| **WTSD** | Went To Showdown | % of times the player reaches showdown after seeing the flop (calling-station indicator) |
| **W$SD** | Won $ at Showdown | % of showdowns won — read together with WTSD |
| **bb/100** | Big blinds per 100 hands | Standard winrate unit for cash games |
| **All-in Adj. EV** | All-in adjusted EV winnings | Replaces actual results of all-in pots with pot-equity share — a simple variance filter on the winnings graph |
| **Positional splits** | — | Any of the above filtered by seat (UTG…BTN, blinds); positional awareness is the first thing databases expose |

## Where HUDs Are Allowed — Site Policies (2025–2026)

Verified against official policy pages and a May 2026 industry survey ([GipsyTeam, 23 May 2026](https://www.gipsyteam.com/news/23-05-2026/huds-on-poker-sites)). **Policies change — always re-check the site's own terms.**

| Site / network | HUDs & trackers | Details |
|---|---|---|
| **PokerStars** | ✅ Allowed with restrictions | Maintains an approved-tools list (PT4, HM3, Hand2Note included). Bans RTA, seating scripts, and "dynamic" HUDs that change with game state; official list: [pokerstars.com/poker/room/prohibited](https://www.pokerstars.com/poker/room/prohibited/) |
| **GGPoker** | ❌ Banned | All third-party tools prohibited (RTA, bots, solvers, charts, HUDs, mass data analysis); only the built-in **Smart HUD** is allowed; penalties include fund confiscation — [Security & Ecology Policy](https://ggpoker.com/network/security-ecology-policy/) |
| **partypoker** | ❌ Banned since June 2019 | Client stopped writing live hand histories; anonymized histories became downloadable after sessions later that year ([PokerNews](https://www.pokernews.com/news/2019/06/partypoker-bans-huds-34530.htm), [Pokerfuse](https://pokerfuse.com/news/poker-room-news/210556-new-restrictions-huds-and-other-third-party-tools/)) |
| **WPT Global** | ❌ Banned | Terms prohibit "using or attempting to use HUDs" that display player information |
| **888poker** | ✅ Allowed | External HUDs supported, except in SNAP fast-fold games |
| **Winamax** | ✅ Allowed | Works with PT4, Hand2Note, Poker Copilot; also has a basic built-in HUD |
| **iPoker network** | ✅ Allowed | Network-wide compatibility with PT4/HM3/Hand2Note |
| **WPN (Americas Cardroom)** | ✅ Allowed (own hands only) | HUDs may only use data from hands you played |

## Tournament Analytics (Análise de Torneios)

| Tool | Site | What it does | Pricing |
|---|---|---|---|
| **ICMIZER 3** | [icmizer.com](https://www.icmizer.com/) | ICM + Nash push/fold calculator, SNG Coach quizzes, replayer | Subscriptions ~US$79.99/yr (Basic) to US$179.99/yr (Pro); 7-day free trial |
| **HoldemResources Calculator (HRC)** | [holdemresources.net](https://www.holdemresources.net/) | Preflop + postflop tournament/cash solver used by many MTT pros | Paid subscription; free trial |
| **SharkScope** | [sharkscope.com](https://www.sharkscope.com/) | Tournament results database (ROI, profit graphs) claiming 99.9% of online MTT/SNG results; collusion-detection features; permitted by major sites | Freemium — free account with limited searches; paid Gold tiers |

## Research: Opponent Modeling & Poker Data Mining (Pesquisa)

| Work | Venue / link | Contribution |
|---|---|---|
| Teófilo & Reis, *Identifying Players' Strategies in No Limit Texas Hold'em Poker through the Analysis of Individual Moves* (2013) | [arXiv:1301.5943](https://arxiv.org/abs/1301.5943) | Clusters individual moves from a hand-history database to classify players into 7 behavioral types — the academic version of what HUD stats do |
| Bertsimas & Paskov, *World-class interpretable poker* (2022) | *Machine Learning* 111:3063–3083, DOI [10.1007/s10994-022-06179-8](https://doi.org/10.1007/s10994-022-06179-8) | Fully interpretable HUNL agent (optimal decision trees + CFR self-play) that beats Slumbot — analytics you can read |
| Bonjour, Aggarwal & Bhargava, *Information Theoretic Approach to Detect Collusion in Multi-Agent Games* (UAI 2022) | [PMLR v180](https://proceedings.mlr.press/v180/bonjour22a.html) | Mutual information between agents' actions flags collusion, extended to partially observable games like poker — the science behind site security teams |
| Kim, *Recording and Describing Poker Hands* (arXiv 2023; IEEE CoG 2024) | [arXiv:2312.11753](https://arxiv.org/abs/2312.11753) | The **PHH file format** — standardized, machine-readable hand histories (10k+ sample hands, 11 variants); fixes the format chaos trackers wrestle with |
| Kim, *PokerKit* (IEEE Trans. on Games 17(1), 2025) | DOI [10.1109/TG.2023.3325637](https://doi.org/10.1109/TG.2023.3325637) · [GitHub](https://github.com/uoftcprg/pokerkit) | Python library for multi-variant poker simulation and hand parsing/evaluation — a research-grade backend for hand-history analysis |

**Academic sandboxes:** build and test agents in [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) (Apache-2.0, active) or [datamllab/rlcard](https://github.com/datamllab/rlcard) (MIT) — never on real-money sites. The Annual Computer Poker Competition (ACPC) historically served this role; its website was unreachable when checked (July 2026).

See also: [Solvers & Open-Source Tools](./Solvers_and_Open_Source_Tools.md) · [Datasets & Hand Histories](./Datasets_and_Hand_Histories.md) · [Poker AI Milestones & Research](./Poker_AI_Milestones_and_Research.md)

---

**Sources:** [pokertracker.com](https://www.pokertracker.com/) · [holdemmanager.com/hm3](https://www.holdemmanager.com/hm3/) · [drivehud.com pricing](https://drivehud.com/dh2-pricing-recommendations/) · [hand2note.com](https://hand2note.com/) · [pokercopilot.com](https://pokercopilot.com/) · [jurojinpoker.com](https://jurojinpoker.com/) · [Max Value merger announcement](https://www.pokertracker.com/announcement/pokertracker-holdemmanager-merger.php) · [PokerNews on the 2014 merger](https://www.pokernews.com/news/2014/08/pokertracker-and-holdem-manager-merge-19100.htm) · [PokerStars prohibited tools](https://www.pokerstars.com/poker/room/prohibited/) · [GGPoker Security & Ecology Policy](https://ggpoker.com/network/security-ecology-policy/) · [PokerNews — partypoker bans HUDs (2019)](https://www.pokernews.com/news/2019/06/partypoker-bans-huds-34530.htm) · [Pokerfuse — partypoker restrictions (2019)](https://pokerfuse.com/news/poker-room-news/210556-new-restrictions-huds-and-other-third-party-tools/) · [GipsyTeam HUD survey (May 2026)](https://www.gipsyteam.com/news/23-05-2026/huds-on-poker-sites) · [icmizer.com](https://www.icmizer.com/) · [holdemresources.net](https://www.holdemresources.net/) · [sharkscope.com](https://www.sharkscope.com/) · [arXiv:1301.5943](https://arxiv.org/abs/1301.5943) · [DOI 10.1007/s10994-022-06179-8](https://doi.org/10.1007/s10994-022-06179-8) · [Bonjour et al., PMLR v180](https://proceedings.mlr.press/v180/bonjour22a.html) · [arXiv:2312.11753](https://arxiv.org/abs/2312.11753) · [DOI 10.1109/TG.2023.3325637](https://doi.org/10.1109/TG.2023.3325637) · [open_spiel](https://github.com/google-deepmind/open_spiel) · [rlcard](https://github.com/datamllab/rlcard) · [pokerkit](https://github.com/uoftcprg/pokerkit) · [Lei 14.790/2023](https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2023/lei/L14790.htm) · [gov.br Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [cvv.org.br](https://cvv.org.br/) · [jogadoresanonimos.com.br](https://jogadoresanonimos.com.br/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [gamblingtherapy.org](https://www.gamblingtherapy.org/)

**Keywords:** poker tracker, poker HUD, heads-up display, hand history database, VPIP, PFR, opponent modeling, poker analytics, ICM calculator, SharkScope, collusion detection, responsible gambling / rastreador de poker, HUD de poker, histórico de mãos, análise de dados de poker, modelagem de oponentes, calculadora ICM, jogo responsável, apostas regulamentadas bet.br
