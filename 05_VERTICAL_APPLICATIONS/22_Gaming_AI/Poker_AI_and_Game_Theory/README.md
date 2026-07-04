# ♠️ Poker AI & Game Theory

> The complete, fact-checked index of **poker as an AI/ML research domain** — the landmark superhuman AIs (Libratus, Pluribus), CFR game theory, GTO foundations, solvers, datasets, strategy resources, the analytics software stack, game variants, the behavioral-science literature, live-tournament data, and the law & economics of the game. **11 pages + a dedicated 8-page [Omaha deep-dive](./Omaha/), every paper/repo/tool verified to exist.**

> ⚠️ **Research & education only — not gambling advice.** Poker involves real-money risk and extreme variance; after rake, most players lose long-term. **Poker bots violate every real-money site's ToS** (accounts banned, funds confiscated) — agent work belongs in sandboxes (OpenSpiel/RLCard/ACPC). If gambling is a problem: [BeGambleAware](https://www.begambleaware.org/), [GamCare](https://www.gamcare.org.uk/), [Gambling Therapy](https://www.gamblingtherapy.org/pt-br/) (tem Português); 🇧🇷 [Jogo Responsável](https://www.gov.br/fazenda/pt-br/assuntos/loterias/jogo-responsavel), CVV 188.

## 📚 The pages

| Page | What's inside |
|---|---|
| [Poker AI Milestones & Research](./Poker_AI_Milestones_and_Research.md) | Cepheus (2015, HULHE "solved"), **DeepStack**, **Libratus** (2017), **Pluribus** (2019, 6-max superhuman), ReBeL, Student of Games; CFR → CFR+ → MCCFR → Deep CFR papers (verified DOIs); LLM-poker era (PokerGPT, **PokerBench**, GTO Wizard LLM Benchmark); ACPC history. |
| [Game Theory & GTO Foundations](./Game_Theory_and_GTO_Foundations.md) | Nash equilibrium, GTO vs exploitative, **MDF & alpha**, pot odds/equity/EV, ranges & combinatorics, blockers, **ICM** (Malmuth-Harville), toy games; canonical books (Chen & Ankenman, Janda, Tipton, Acevedo) with verified ISBNs; variance & bankroll math. |
| [Solvers & Open-Source Tools](./Solvers_and_Open_Source_Tools.md) | Commercial: PioSOLVER, GTO+, MonkerSolver, **GTO Wizard**; open-source (20 repos verified live): **wasm-postflop/desktop-postflop**, TexasSolver, **OpenSpiel**, **RLCard**, PokerRL, **pokerkit** (U.Toronto); hand evaluators (TwoPlusTwo, treys, OMPEval). |
| [Datasets & Hand Histories](./Datasets_and_Hand_Histories.md) | UCI Poker Hand, **IRC Poker Database** (1995-2001), ACPC match logs, **PHH format** (uoftcprg), Pluribus published logs, parsers & converters — with licensing/ToS caveats. |
| [Kaggle Poker Datasets](./Kaggle_Poker_Datasets.md) | ~20 datasets pulled live via API: hand classification, real hand histories, Spin & Go, heads-up gameplay, card computer-vision. |
| [Strategy, Training & Community](./Strategy_Training_and_Community.md) | GTO Wizard, Upswing, Jonathan Little; books (Harrington, Tendler's Mental Game); TwoPlusTwo, r/poker; 🇧🇷 **cena brasileira** (BSOP, maior base MTT do mundo) + nota regulatória. |
| [Tracking Software, HUDs & Analytics](./Tracking_Software_HUDs_and_Analytics.md) | PokerTracker 4, Hold'em Manager 3, Hand2Note; stats (VPIP/PFR/3-bet/WTSD); **site policies 2025-26** (GGPoker bans all third-party HUDs; PokerStars restricts RTA); ICMIZER, SharkScope; bot/collusion detection. |
| [Poker Variants — PLO, Short-Deck & Mixed Games](./Poker_Variants_PLO_and_Mixed_Games.md) | Beyond NLHE: **Pot-Limit Omaha** (PLO4/5/6, exact combinatorics), **Short-Deck (6+)** rules & Triton history (Ivey 2018), stud/draw/**mixed games** (H.O.R.S.E., 8-game, Badugi, 2-7 TD) anchored to the 2025 WSOP; solvers (MonkerSolver, GTO Wizard PLO, PLO Genius, Vision), **pokerkit** variant coverage — why variants are open AI research. |
| [Behavioral Research & Poker Psychology](./Behavioral_Research_and_Poker_Psychology.md) | Peer-reviewed behavioral science: **tilt** & emotion regulation (Palomäki/Laakasuo/Salmela), loss-chasing, **skill-vs-chance evidence** (DeDonno, Levitt-Miles, Meyer), problem-gambling among poker players (Barrault-Varescon), do RG tools work (Auer-Griffiths); *Thinking in Bets*, *The Biggest Bluff* — the section's responsible-gambling research anchor. |
| [Live Tournament Data & Results](./Live_Tournament_Data_and_Results.md) | The live-results universe (unit = a tournament cash, not a hand): **The Hendon Mob** database, **Global Poker Index** rankings & methodology, WSOP.com / WPT / EPT / **Triton** results, **PGT** leaderboard, the **Levitt-Miles** 2010-WSOP skill study; scraping/ToS & database-rights caveats; 🇧🇷 BSOP circuit. Every figure re-verified (EPT payouts, GPI criteria, Triton totals). |
| [Regulation, Skill-vs-Chance Law & Economics](./Regulation_Skill_vs_Chance_and_Economics.md) | The law & money: **United States v. DiCristina** (skill ruling + reversal), UIGEA / Wire Act / **Black Friday 2011**, state-by-state regulated online poker & the **MSIGA** compact, 🇧🇷 **Lei 14.790/2023** & `.bet.br` (CBTH mind-sport recognition), **rake economics** (why the pool loses net), W-2G / IR taxation — every statute, case citation & date verified against primary sources. |

## ♦️ Deep-dive subsection — [Omaha (PLO & all variants)](./Omaha/)

A dedicated **8-page Omaha deep-dive** goes far beyond the general variants page: [Variants & Rules](./Omaha/Omaha_Variants_and_Rules.md) · [Math, Combinatorics & Equity](./Omaha/Omaha_Math_Combinatorics_and_Equity.md) · [Strategy & GTO](./Omaha/Omaha_Strategy_and_GTO.md) · [Solvers & Software](./Omaha/Omaha_Solvers_and_Software.md) · [AI & Computational Research](./Omaha/Omaha_AI_and_Computational_Research.md) · [Datasets & Hand Histories](./Omaha/Omaha_Datasets_and_Hand_Histories.md) · [Books, Courses & Coaching](./Omaha/Omaha_Books_Courses_and_Coaching.md) · [Where It's Played & Live Scene](./Omaha/Omaha_Where_Its_Played_and_Live_Scene.md). Every page went through a research → fact-check → **adversarial re-verify** pipeline.

## 🤖 Why poker matters for AI

Poker is the canonical **imperfect-information game**: unlike chess/Go, players act without seeing the full state, so classic tree search fails. Solving it required new machinery — **Counterfactual Regret Minimization** (Zinkevich et al., 2007) and its descendants — which now powers negotiation, security, auction, and multi-agent research far beyond cards. The 2015-2019 run (Cepheus → DeepStack → Libratus → Pluribus) is one of AI's landmark achievement arcs, and the 2024-2026 LLM-poker benchmarks opened a new chapter.

## Related in AIForge
- Parent vertical: [`../`](../) (Gaming AI)
- Fundamentals: [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/) · [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Bayesian_and_Probabilistic_ML/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Bayesian_and_Probabilistic_ML/)
- Betting-markets research (different domain, same rigor): [`../../29_Sports_Analytics_AI/Football_Match_and_Betting_Prediction/`](../../29_Sports_Analytics_AI/Football_Match_and_Betting_Prediction/)

**Keywords:** poker AI, Libratus, Pluribus, DeepStack, Cepheus, counterfactual regret minimization, CFR, GTO poker, game theory optimal, poker solver, PioSOLVER, GTO Wizard, OpenSpiel, RLCard, poker dataset, hand history, ICM, poker tracker HUD, IA de poker, teoria dos jogos, solver de poker.
