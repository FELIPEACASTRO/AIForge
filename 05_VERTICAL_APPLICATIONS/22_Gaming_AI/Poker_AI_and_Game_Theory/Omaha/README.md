# ♦️ Omaha Poker — Deep Dive (PLO & All Variants)

> The most complete, fact-checked index of **Omaha poker as a study & research domain** — every variant (PLO4/5/6, Omaha Hi-Lo, Big O, Courchevel), the combinatorics & equity math, GTO strategy, the solver/software stack, the computational-AI research frontier, datasets, the book/course canon, and where Omaha is played. **8 pages, every rule/paper/repo/tool/price verified live** through a research → fact-check → adversarial re-verify pipeline.

> ⚠️ **Research & education only — not gambling advice.** Omaha's variance is **structurally higher than No-Limit Hold'em**: preflop equities run closer, pots grow faster under pot-limit, and bankroll swings are larger. After rake, the player pool loses net. **Bots & real-time assistance (RTA) are banned by every real-money site** — the tools here are for study away from the tables. If gambling is a problem: [BeGambleAware](https://www.gambleaware.org/), [GamCare](https://www.gamcare.org.uk/) (0808 8020 133), [Gambling Therapy](https://www.gamblingtherapy.org/pt-br/) (tem Português); 🇧🇷 [Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) / `.bet.br`, CVV **188**.

## 📚 The pages

| Page | What's inside |
|---|---|
| [Omaha Variants & Rules](./Omaha_Variants_and_Rules.md) | Every Omaha form — **PLO4/5/6**, **Omaha Hi-Lo / Omaha-8** (8-or-better, scoop/quarter, wheel = nut low), **Big O**, **Courchevel** (+Hi-Lo); the defining **"exactly two hole cards + exactly three board cards"** rule and how it breaks Hold'em intuition (four aces ≠ quads, single-suited ace ≠ flush); Limit/Pot-Limit/No-Limit structures + the pot-limit bet math. |
| [Omaha Math, Combinatorics & Equity](./Omaha_Math_Combinatorics_and_Equity.md) | Starting-hand counts (C(52,4)=270,725; C(52,5); C(52,6)), **equity compression** (why hands run close), nut-focus & non-nut danger, blockers, **double-suited** vs rainbow, **rundowns & wraps** (13/17/20-card wraps), preflop hand rankings, running it twice & variance, Hi-Lo low combinatorics. |
| [Omaha Strategy & GTO](./Omaha_Strategy_and_GTO.md) | Preflop hand selection & position, 3-bet/4-bet pot-limit dynamics, postflop **nut-focus** & blocker-based bluffing, board texture, pot-limit bet-sizing, multiway pots, Hi-Lo scooping/quartering/nut-low play, ICM in PLO MTTs — solver-derived where possible. |
| [Omaha Solvers & Software](./Omaha_Solvers_and_Software.md) | The Omaha tool stack with verified pricing & free tiers: **MonkerSolver**, **GTO Wizard PLO**, **PLO Genius**, **Vision GTO Trainer**, PLO Mastermind, RangeConverter, **ProPokerTools Odds Oracle**; a coverage/pricing comparison table (PLO4/5/6, cash/MTT). |
| [Omaha AI & Computational Research](./Omaha_AI_and_Computational_Research.md) | Why Omaha's game tree dwarfs NLHE (abstraction/precompute), **CFR/CFR+ on Omaha**, the Harvard DASH NL Omaha Hi-Lo endgame equilibrium, **pokerkit** (IEEE ToG paper), **phevaluator** PLO4/5/6, OMPEval, and the **OpenSpiel/RLCard Omaha coverage gap** as a research opportunity. |
| [Omaha Datasets & Hand Histories](./Omaha_Datasets_and_Hand_Histories.md) | Omaha data for ML: Kaggle Omaha datasets, the **PHH (Poker Hand History)** format's Omaha support, hand-history converters, ProPokerTools sims — with licensing/ToS caveats. |
| [Omaha Books, Courses & Coaching](./Omaha_Books_Courses_and_Coaching.md) | The verified learning canon: **Jeff Hwang**'s PLO trilogy, **JNandez/Habegger** (D&B), Tri Nguyen, Boca; **PLO Mastermind**, Run It Once PLO, Upswing Omaha; notable coaches — publisher/course pages verified. |
| [Omaha — Where It Is Played & Live Scene](./Omaha_Where_Its_Played_and_Live_Scene.md) | The ecosystem: online sites spreading PLO (GGPoker, PokerStars, partypoker…), high-stakes online & **live high-roller PLO** (Triton PLO, WSOP Omaha/Hi-Lo bracelet events), the modern **PLO boom** — with responsible-gambling framing. |

## ♦️ Why Omaha matters (for players and for AI)

Omaha is the **second-most popular poker game in the world** after No-Limit Hold'em, and it is a genuinely **harder computational problem**: the "use exactly two of your four-plus hole cards" rule explodes the number of information sets and starting-hand combinations (C(52,4)=**270,725** for PLO alone, vs 1,326 in Hold'em), so the CFR-family solvers that "solved" heads-up Hold'em still can't brute-force full PLO — it demands abstraction and massive precomputation. That gap is exactly why Omaha remains **open research territory** and why the mainstream RL poker frameworks (OpenSpiel, RLCard) still lack it.

## Related in AIForge
- Parent section: [`../`](../) (Poker AI & Game Theory) · broader overview: [`../Poker_Variants_PLO_and_Mixed_Games.md`](../Poker_Variants_PLO_and_Mixed_Games.md)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/) · [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Bayesian_and_Probabilistic_ML/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Bayesian_and_Probabilistic_ML/)

**Keywords:** Omaha poker, Pot-Limit Omaha, PLO, PLO4, PLO5, PLO6, Omaha Hi-Lo, Omaha-8, eight or better, Big O, Courchevel, PLO solver, MonkerSolver, GTO Wizard PLO, PLO Genius, pokerkit Omaha, phevaluator, CFR Omaha, Omaha combinatorics, nut-focus, blockers, rundowns, wraps, PLO strategy, responsible gambling — poker Omaha, Omaha Pot-Limit, Omaha Alto-Baixo, solver de PLO, combinatória do Omaha, estratégia de PLO, jogo responsável.
