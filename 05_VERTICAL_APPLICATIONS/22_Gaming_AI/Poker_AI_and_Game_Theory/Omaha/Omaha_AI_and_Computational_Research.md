# Omaha AI & Computational Research (IA e Pesquisa Computacional no Omaha)

> The computer-science side of Omaha: **why the game tree dwarfs No-Limit Hold'em** (por que a árvore do jogo é muito maior que a do Hold'em) and forces abstraction and precompute, how **Counterfactual Regret Minimization (CFR / CFR+)** is applied to a four-card game, the one openly published Omaha-equilibrium thesis (**Harvard DASH — Ho, 2015**), the open-source engines that actually implement Omaha correctly (**PokerKit**, **PokerHandEvaluator / phevaluator**, **OMPEval**), and the conspicuous **coverage gaps in the standard RL research stacks (OpenSpiel, RLCard) — which is a live research opportunity.** **Research and education only.** Every arXiv id, DOI, GitHub repository, license and version number on this page was opened live in July 2026; anything that could not be confirmed against a primary source was deleted rather than guessed.

---

## ⚠️ Responsible Gambling First (Jogo Responsável)

**This is an AI/ML and computational-research reference, NOT gambling advice or an inducement to play.** Before studying any of the algorithms below, keep the money reality in view:

- **Omaha/PLO variance is *structurally* higher than NLHE (a variância do Omaha é estruturalmente maior que a do Hold'em).** Four hole cards mean equities run closer together, draws are bigger, and pot-limit lets pots grow faster — so bankroll swings are larger even for winning players. The math and code on this page explain *why*; they do not make the swings safer.
- **The pool loses net (o bolo perde no agregado).** Poker is zero-sum before rake and negative-sum after **rake (a taxa da casa)**; by construction most players are long-term losers.
- **Bots and Real-Time Assistance (RTA) are banned (bots e assistência em tempo real são proibidos).** The solvers, evaluators and CFR implementations described here are for **off-table research and study only** — running any of them (or a bot built on them) during real-money play violates the terms of service of essentially every poker site and can be prosecuted as fraud.

| Support | Region | Contact | Cost |
|---|---|---|---|
| **CVV — Centro de Valorização da Vida** | 🇧🇷 Brazil, emotional support 24/7 | dial **188** · [cvv.org.br](https://cvv.org.br/) | Free |
| **Jogo Responsável — SPA / Ministério da Fazenda** | 🇧🇷 Brazil official guidance; licensed sites use `.bet.br` | [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) | Free |
| **GambleAware (BeGambleAware)** | 🇬🇧 UK / global info | [begambleaware.org](https://www.begambleaware.org/) | Free |
| **National Gambling Helpline (GamCare)** | 🇬🇧 UK helpline & chat | **0808 8020 133** · [gamcare.org.uk](https://www.gamcare.org.uk/) | Free |
| **Gambling Therapy (Gordon Moody)** | 🌍 Global, **Português-BR support** | [gamblingtherapy.org](https://www.gamblingtherapy.org/) | Free |

If study stops being intellectually motivated and starts feeling compulsive, use the resources above.

---

## Why This Page Exists (o que muda no Omaha do ponto de vista computacional)

Most public "poker AI" — the milestones on the [Poker AI Milestones & Research](../Poker_AI_Milestones_and_Research.md) page (Cepheus, DeepStack, Libratus, Pluribus) — is about **Texas Hold'em**. Omaha is a *harder* computational target that has received far less academic attention, so the literature is thin, the open-source tooling is younger, and the standard reinforcement-learning benchmarks **don't cover it at all**. This page catalogues what *does* exist, with every claim tied to a primary source, and marks the gaps as research openings.

The unit facts that drive everything below are established combinatorially on the sibling [Omaha Math, Combinatorics & Equity](./Omaha_Math_Combinatorics_and_Equity.md) page; here we use them to reason about **tree size, abstraction and solver cost**.

---

## 1 · Why Omaha's Game Tree Dwarfs Hold'em (a explosão da árvore do jogo)

An imperfect-information solver's cost scales with the number of **information sets** and the size of the **strategy vector** it must store and update at each one. Both blow up in Omaha because the private-hand space is ~200× larger *at every decision point*.

| Quantity | No-Limit Hold'em | Pot-Limit Omaha (PLO4) | Ratio |
|---|---:|---:|---:|
| Private starting hands dealt, C(52,k) | **1,326** | **270,725** | **~204×** |
| Distinct (suit-isomorphic) starting hands | **169** | **16,432** | ~97× |
| 5-card hands to evaluate **per player, per board** (exactly-2-of-hole × exactly-3-of-board) | **1** | C(4,2)·C(5,3) = **60** | 60× |
| Canonical (suit-isomorphic) flops | **1,755** | **1,755** | 1× |

*Combinatorics computed from first principles on the sibling [Math page](./Omaha_Math_Combinatorics_and_Equity.md); the 60-way per-player evaluation is the "exactly two hole + exactly three board" rule.*

The board isomorphism count is *the same* (1,755 canonical flops) — the explosion is entirely on the **private** side. Every player's range at an information set is a probability vector over ~16k canonical hands instead of 169, and the evaluator that resolves showdowns does ~60× the work per hand. **GTO Wizard**, which ships presolved PLO4 solutions, frames it exactly this way: *"PLO4 has over 270,000 starting hands, 200x more than Hold'em,"* with cash solutions covering *"all flops + runouts"* ([gtowizard.com/plo](https://gtowizard.com/plo/)).

**Consequences for anyone building an Omaha solver:**

1. **Memory.** Full-tree tabular CFR stores regret and average-strategy accumulators per action per information set; a ~200× wider hand space multiplies that footprint accordingly, which is why public Omaha solving is largely limited to **PLO4** and to **heads-up / small multiway** subtrees.
2. **Precompute & abstraction.** Practical engines lean on **card abstraction** (bucketing strategically similar hands), **action abstraction** (a small menu of bet sizes), and heavy **precompute** — solving spots offline and serving them, rather than solving live.
3. **PLO5 / PLO6 are worse.** Five and six hole cards push the dealt-hand count to C(52,5)=2,598,960 and C(52,6)=20,358,520, with 100 and 150 sub-hands to evaluate per player. Even the best-funded commercial engine lists **PLO5/PLO6 as *"in development with no ETA"*** ([GTO Wizard PLO FAQ](https://gtowizard.com/plo/)).

---

## 2 · CFR & CFR+ Applied to Omaha (o algoritmo por trás do equilíbrio)

**Counterfactual Regret Minimization (CFR)** is the workhorse for approximating **Nash equilibria (equilíbrios de Nash)** in large imperfect-information games; its accelerated variant **CFR+** (regret-matching⁺ with a linear averaging schedule) *"typically outperforms the previously known algorithms by an order of magnitude or more in terms of computation time while also potentially requiring less memory"* — Oskari Tammelin, **[*Solving Large Imperfect Information Games Using CFR+*, arXiv:1407.5042 (2014)](https://arxiv.org/abs/1407.5042)**. CFR+ is the same family of algorithm that essentially solved heads-up limit Hold'em (the Cepheus / *Science* 2015 milestone — see the [Milestones page](../Poker_AI_Milestones_and_Research.md)).

Porting CFR/CFR+ to Omaha inherits the tree-size problem from §1: the algorithm itself is unchanged, but each iteration touches a ~200× larger private-hand space, so the engineering is dominated by **abstraction + parallelism**. In practice, Omaha equilibrium work restricts the tree (a single **endgame**, or heads-up, or a fixed betting stub), buckets hands, and runs on GPUs/clusters. The one openly published academic instance of this exact recipe for Omaha is the Harvard thesis in §3.

> **A note on "GTO" for Omaha.** Because the tree is so large, virtually all public Omaha "solutions" rely on abstraction, are limited to PLO4, or cover only heads-up / small multiway spots. Treat solver output as **directional** (nut-focus, blocker logic, sizing families) rather than an exactly-solved oracle — the caveat is spelled out on the [Omaha Strategy & GTO](./Omaha_Strategy_and_GTO.md) page.

---

## 3 · The Harvard DASH Omaha Equilibrium Thesis (Ho, 2015)

The single most relevant *academic* Omaha-AI document is a Harvard master's thesis, freely available through **DASH (Digital Access to Scholarship at Harvard)**:

| Field | Detail |
|---|---|
| **Title** | *A No-Limit Omaha Hi-Lo Poker Jam/Fold Endgame Equilibrium* |
| **Author** | Kenneth Ho |
| **Year** | 2015 |
| **Type / institution** | Master's thesis, Harvard University (Harvard Extension School), DASH repository |
| **Repository handle** | `1/24078344` — PDF `HO-THESIS-2015.pdf` |
| **Algorithm** | **CFR+** (Counterfactual Regret Minimization Plus) |
| **Compute** | **OpenCL** heterogeneous (GPU) computing + **Amazon Web Services** cloud |
| **Output** | An **ε-Nash equilibrium** for a jam/fold (all-in-or-fold) endgame, plus a **scoring heuristic** that approximates the equilibrium strategy |

**What it does.** Omaha Hi-Lo adds two complications on top of Hold'em: **four hole cards** (the 270,725-combination space of §1) *and* a **split high/low pot** (the 56-rung qualifying-low ladder from the [Math page](./Omaha_Math_Combinatorics_and_Equity.md)). Ho restricts the problem to a tractable **jam/fold endgame** — the short-stack, all-in-or-fold subgame — and runs **CFR+** on it, using **OpenCL** to push the evaluation onto GPUs and **AWS** to scale, to reach an approximate (ε-)Nash equilibrium. He then distills the equilibrium into a **hand-scoring heuristic** so the strategy can be applied without re-solving. It is, in effect, an Omaha Hi-Lo analogue of the classic Hold'em "SAGE"/jam-or-fold push-fold charts, but *derived from a CFR+ solve rather than hand-tuned.*

- **Primary record (Semantic Scholar):** [semanticscholar.org — Ho, 2015](https://www.semanticscholar.org/paper/A-No-Limit-Omaha-Hi-Lo-Poker-Jam-Fold-Endgame-Ho/6ba243f83899c5d0b38b3346260180a5b12c5367)
- **Open-access PDF (DASH):** `dash.harvard.edu/bitstream/handle/1/24078344/HO-THESIS-2015.pdf` *(DASH blocks automated fetchers, but the record — title, author, 2015, open-access PDF, and the CFR+ / OpenCL / AWS / jam-fold ε-Nash / scoring-heuristic details — is confirmed via the Semantic Scholar record and the indexed DASH catalogue entry.)*

**Why it matters here.** It is a rare, citable, end-to-end demonstration that the **CFR+ → abstraction → GPU/cloud** pipeline that solved Hold'em transfers to Omaha — *once the tree is cut down to an endgame.* The very fact that the public literature offers essentially one such thesis, rather than the dozens that exist for Hold'em, is the research gap this page keeps pointing at.

---

## 4 · Open-Source Omaha Engines That Are Actually Correct

Getting Omaha *right* in code is non-trivial: a naïve evaluator that treats hole cards like Hold'em will wrongly score a single suited hole card on a four-suit board as a flush. These libraries enforce the **exactly-two-hole + exactly-three-board** rule.

### 4.1 · PokerKit (uoftcprg) — simulation engine with first-class Omaha

**[uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit)** is an open-source Python library for **fine-grained, multi-variant poker game simulation and hand evaluation**, from the University of Toronto Computer Poker Research Group.

| Attribute | Verified value |
|---|---|
| **Peer-reviewed paper** | J. Kim, *"PokerKit: A Comprehensive Python Library for Fine-Grained Multivariant Poker Game Simulations,"* **IEEE Transactions on Games**, vol. **17**, no. **1**, pp. **32–39**, **2025** |
| **DOI** | **[10.1109/TG.2023.3325637](https://doi.org/10.1109/TG.2023.3325637)** |
| **Preprint** | [arXiv:2308.07327](https://arxiv.org/abs/2308.07327) (Juho Kim, 2023) |
| **License** | **MIT** |
| **Version (July 2026)** | **v0.7.4** (released 2026-05-22) |
| **Omaha variants (built-in classes)** | `PotLimitOmahaHoldem` and `FixedLimitOmahaHoldemHighLowSplitEightOrBetter` (Omaha Hi-Lo 8-or-better) |

The paper's own framing is the Omaha argument in a sentence: existing tools *"typically support only a handful of poker variants and lack flexibility in game state control."* PokerKit models the full state machine (posting, dealing, the pot-limit betting cap, showdown with the 2+3 constraint, hi-lo splitting), so it is the natural substrate for **building your own Omaha CFR/RL experiments** rather than re-implementing rules from scratch. It also underpins the equity/enumeration method box on the [Math page](./Omaha_Math_Combinatorics_and_Equity.md).

### 4.2 · PokerHandEvaluator / phevaluator (HenryRLee) — fast PLO4/5/6 evaluation

**[HenryRLee/PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator)** is a fast, perfect-hash hand evaluator whose Omaha support is explicit: it handles **PLO4, PLO5, and PLO6**, and describes the constraint verbatim as *"picking exactly two cards from four player's cards, and exactly three cards from five community cards."*

| Attribute | Verified value |
|---|---|
| **License** | **Apache-2.0** |
| **Core language** | C/C++ (`cpp/`), with a Python distribution **`phevaluator`** on PyPI |
| **Python package** | `phevaluator` **v0.5.3.1** (2024-03-21), Apache-2.0, author **Henry Lee** |
| **Omaha API (Python)** | **`evaluate_omaha_cards(...)`** (module `phevaluator.evaluator_omaha`) — takes the 4 hole + 5 board cards and internally maximises over the 60 legal 2+3 splits |
| **Community ports** | JavaScript, Dart, Haskell, Go, TypeScript, C#, Rust (per the repo README) |

This is the evaluator used to generate the Monte-Carlo equities on the sibling [Math page](./Omaha_Math_Combinatorics_and_Equity.md); its correct enforcement of the two-card rule is what makes those numbers trustworthy.

### 4.3 · OMPEval (zekyll) — a fast Hold'em evaluator as an Omaha building block

**[zekyll/OMPEval](https://github.com/zekyll/OMPEval)** is a fast C++ hand evaluator and equity calculator. It is important to state its scope **accurately**: its public API is **Texas-Hold'em-oriented**, not a turnkey Omaha calculator.

| Attribute | Verified value |
|---|---|
| **License** | **ISC** |
| **`HandEvaluator`** | Evaluates standard hands of **0–7 cards**, returns a 16-bit rank (bigger = better); **perfect hashing** shrinks the main lookup table from ~36 MB to **~200 kB**; ~10 ms init; SSE2/SSE4 acceleration |
| **`EquityCalculator`** | All-in equities **for Texas Hold'em** — Monte-Carlo simulation *or* full enumeration, with board/dead-card control, up to 6 players |
| **Omaha?** | **No native Omaha calculator** — its documented public API is Hold'em-only. Its value for Omaha is as a **building block**: drive the general `HandEvaluator` over each of the 60 two-hole + three-board 5-card combinations and take the max |
| **Python** | Third-party wrapper [Tylder/OMPEval_py_wrapper](https://github.com/Tylder/OMPEval_py_wrapper) |

OMPEval is listed here precisely because it is *frequently mistaken* for an Omaha tool: it is a superb low-level 5-to-7-card evaluator that Omaha projects wrap, but Omaha logic (the 2+3 enumeration and, for Hi-Lo, the low half) must be supplied by the caller. For a drop-in Omaha evaluator, prefer **phevaluator** (§4.2).

---

## 5 · The Big Gap: OpenSpiel and RLCard Don't Cover Omaha

The two most-used academic frameworks for **reinforcement learning and game-theoretic research on card games** both ship rich poker suites — and **neither includes Omaha**. This is the clearest research opening on the page.

| Framework | Poker/card games it *does* ship | Omaha? | Source |
|---|---|---|---|
| **OpenSpiel** (google-deepmind) | Kuhn poker, Leduc poker, Liar's Dice, **Poker (Hold 'em)** "implemented via ACPC" (2–10 players) | **Not present** — Omaha is not listed anywhere in the game catalogue | [OpenSpiel `games.md`](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md) |
| **RLCard** (datamllab) | Blackjack, Leduc Hold'em, **Limit & No-Limit Texas Hold'em**, Dou Dizhu, Mahjong, UNO, Gin Rummy, etc. | **Not present** — no Omaha environment | [rlcard.org/games](https://rlcard.org/games.html) · [datamllab/rlcard](https://github.com/datamllab/rlcard) · [arXiv:1910.04376](https://arxiv.org/abs/1910.04376) |

OpenSpiel's poker realism comes from the **ACPC (Annual Computer Poker Competition)** protocol, whose events were **Texas Hold'em variants (limit / no-limit) plus toy games (Kuhn, Leduc)** — Omaha was **never** an ACPC event, and that heritage is why the mainstream RL stacks inherited a Hold'em-only poker surface. (OpenSpiel's game catalogue is the stable reference for this Hold'em-only poker heritage.)

**Why this is an opportunity, not just a limitation:**

- A correct **`omaha`/`pot_limit_omaha` game** in OpenSpiel or an **Omaha environment in RLCard** would immediately let the entire CFR / MCCFR / deep-CFR / RL-agent ecosystem target a game that is *combinatorially harder* than the Hold'em everyone benchmarks on — a natural stress test for abstraction and generalization.
- The rules engine already exists and is validated: **PokerKit** (§4.1) implements PLO and Omaha Hi-Lo, and **phevaluator** (§4.2) supplies fast PLO4/5/6 showdown evaluation. A contributor's task is largely *wiring a correct engine into the RL framework's interface*, not inventing the mechanics.
- Multiway PLO (3+ players) is essentially **unsolved in the open literature** and is exactly where variance and equity-overlap (from the [Math page](./Omaha_Math_Combinatorics_and_Equity.md)) are largest — a rich target for large-scale MCCFR / population-based methods.

---

## 6 · Where the Practical (Closed) Omaha Solving Actually Happens

Because the open research stack lacks Omaha, the *working* Omaha equilibrium tooling is commercial or closed-source. These are catalogued for completeness (and cross-referenced on [Solvers & Open-Source Tools](../Solvers_and_Open_Source_Tools.md) and [Omaha Strategy & GTO](./Omaha_Strategy_and_GTO.md)) — all **off-table study only; RTA is banned.**

| Tool | Omaha capability (verified wording) | Note |
|---|---|---|
| **[MonkerSolver / MonkerWare](https://www.monkerware.com/)** | *"Solve Hold'em and Omaha from any street, with any number of players"* — the long-standing **multiway** Omaha solver; **MonkerViewer** stores/shares preflop ranges | The go-to engine for multiway PLO; equilibrium solvers of this kind are built on the CFR family (§2) |
| **[GTO Wizard PLO](https://gtowizard.com/plo/)** | Presolved **PLO4** preflop→river, *"all flops + runouts,"* cash 6-/7-max at 50bb & 100bb and MTT 10–100bb | *"River Nash Distance < 0.1%"*; reports beating *"the 2018 ACPC champion Slumbot by +19.4 bb/100 over 150,000 hands"*; **PLO5/PLO6 in development, no ETA** |
| **[PLO Genius](https://plogenius.com/)** | *"Modern Pot Limit Omaha Solver"* + trainer covering **PLO / PLO5 / PLO6** | Imports MonkerSolver work; free Starter tier |
| **[PLO.com](https://plo.com/)** | Exact **PLO4/PLO5/PLO6** equity calculator with board and dead-card control; browser "Solver Room" | Browser equity calculator + "Solver Room" study views |

The contrast is the whole point of this page: **Hold'em has an open, reproducible research stack; Omaha's equilibrium tooling is mostly proprietary**, so the open-source contributions in §4 and the framework gaps in §5 carry disproportionate research value.

---

## 7 · Consolidated Verified-Resource Table

| Resource | Type | Key facts (all verified July 2026) | License |
|---|---|---|---|
| **[Ho, *No-Limit Omaha Hi-Lo Jam/Fold Endgame Equilibrium*](https://www.semanticscholar.org/paper/A-No-Limit-Omaha-Hi-Lo-Poker-Jam-Fold-Endgame-Ho/6ba243f83899c5d0b38b3346260180a5b12c5367)** | Thesis (Harvard DASH, 2015) | CFR+ · OpenCL · AWS · ε-Nash endgame + scoring heuristic | Open access |
| **[Tammelin, *Solving Large Imperfect Information Games Using CFR+*](https://arxiv.org/abs/1407.5042)** | Paper (arXiv, 2014) | Introduces **CFR+**; the algorithm used for Omaha equilibria | arXiv |
| **[uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit)** | Sim engine (Python) | IEEE ToG 17(1):32–39, 2025 · DOI 10.1109/TG.2023.3325637 · arXiv:2308.07327 · v0.7.4 · PLO + Omaha Hi-Lo classes | MIT |
| **[HenryRLee/PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator)** | Evaluator (C++/Py) | PLO4/5/6 · `evaluate_omaha_cards` · phevaluator v0.5.3.1 | Apache-2.0 |
| **[zekyll/OMPEval](https://github.com/zekyll/OMPEval)** | Evaluator (C++) | 0–7 card, ~200 kB perfect-hash tables; Hold'em equity API; Omaha only as a building block | ISC |
| **[OpenSpiel](https://github.com/google-deepmind/open_spiel)** | RL/game framework | Kuhn, Leduc, Hold'em (ACPC) — **no Omaha** (gap) | Apache-2.0 |
| **[RLCard](https://github.com/datamllab/rlcard)** | RL card-game toolkit | Leduc/Limit/No-Limit Hold'em, Dou Dizhu, etc. — **no Omaha** (gap) | MIT |

---

## 8 · Open Problems & Research Directions (problemas em aberto)

1. **An open, correct Omaha environment** in OpenSpiel and/or RLCard (PLO4 first, then Hi-Lo, then PLO5/PLO6), reusing PokerKit's validated rules and phevaluator's showdown logic.
2. **Multiway PLO equilibria in the open.** Three-plus-handed PLO is where variance and equity-overlap peak and where public solutions barely exist — a target for scalable MCCFR / deep-CFR / population methods.
3. **Abstraction that respects nuttiness.** Hold'em-style bucketing loses PLO's blocker/nut-outs structure; Omaha-specific card abstractions (nut-potential-aware) are underexplored.
4. **PLO5/PLO6 tractability.** With 2.6M and 20.4M dealt hands, even commercial engines have *no ETA*; better abstractions or learned value functions are open.
5. **Reproducible Omaha benchmarks.** There is no open Omaha analogue of the Hold'em ACPC/Slumbot ecosystem — a shared engine, agent API and evaluation suite would unlock comparability.

---

## Related (Relacionados)

- **This Omaha subsection (esta subseção Omaha):** [Omaha Math, Combinatorics & Equity](./Omaha_Math_Combinatorics_and_Equity.md) · [Omaha Strategy & GTO](./Omaha_Strategy_and_GTO.md) · [Omaha Variants & Rules](./Omaha_Variants_and_Rules.md) · [whole Omaha folder](./)
- **Parent overview:** [Poker Variants — PLO, Short-Deck & Mixed Games](../Poker_Variants_PLO_and_Mixed_Games.md)
- **Hold'em AI lineage & CFR history:** [Poker AI Milestones & Research](../Poker_AI_Milestones_and_Research.md) · [Game Theory & GTO Foundations](../Game_Theory_and_GTO_Foundations.md)
- **Tooling & data:** [Solvers & Open-Source Tools](../Solvers_and_Open_Source_Tools.md) · [Datasets & Hand Histories](../Datasets_and_Hand_Histories.md)
- **Section index:** [♠️ Poker AI & Game Theory](../README.md)

**Sources:** [Ho 2015 — Omaha Hi-Lo Jam/Fold Endgame Equilibrium (Semantic Scholar)](https://www.semanticscholar.org/paper/A-No-Limit-Omaha-Hi-Lo-Poker-Jam-Fold-Endgame-Ho/6ba243f83899c5d0b38b3346260180a5b12c5367) · [Tammelin 2014 — CFR+ (arXiv:1407.5042)](https://arxiv.org/abs/1407.5042) · [PokerKit paper — IEEE ToG, DOI 10.1109/TG.2023.3325637](https://doi.org/10.1109/TG.2023.3325637) · [PokerKit preprint — arXiv:2308.07327](https://arxiv.org/abs/2308.07327) · [uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit) · [HenryRLee/PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) · [phevaluator (PyPI)](https://pypi.org/project/phevaluator/) · [zekyll/OMPEval](https://github.com/zekyll/OMPEval) · [OpenSpiel games.md](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md) · [RLCard games](https://rlcard.org/games.html) · [RLCard (arXiv:1910.04376)](https://arxiv.org/abs/1910.04376) · [GTO Wizard PLO](https://gtowizard.com/plo/) · [MonkerWare](https://www.monkerware.com/) · [PLO Genius](https://plogenius.com/) · [PLO.com](https://plo.com/) · Combinatorics C(52,4)=270,725, 16,432 distinct hands, 60-way per-player evaluation from the sibling Math page.

**Keywords:** Omaha AI, PLO computational research, Omaha game tree, abstraction, precompute, CFR, CFR+, counterfactual regret minimization, Tammelin CFR+, epsilon-Nash equilibrium, jam/fold endgame, Harvard DASH Ho 2015, Omaha Hi-Lo equilibrium, OpenCL, AWS, PokerKit, uoftcprg, IEEE Transactions on Games, PokerHandEvaluator, phevaluator, evaluate_omaha_cards, PLO4 PLO5 PLO6, OMPEval, perfect hashing, OpenSpiel, RLCard, ACPC, universal_poker, MonkerSolver, GTO Wizard PLO, 270725 starting hands, 1755 boards, multiway PLO, research opportunity / IA no Omaha, pesquisa computacional PLO, árvore do jogo, abstração, minimização de arrependimento contrafactual, equilíbrio de Nash aproximado, endgame all-in, tese de Harvard, avaliador de mãos, lacuna de cobertura, oportunidade de pesquisa, solver multiway, jogo responsável
