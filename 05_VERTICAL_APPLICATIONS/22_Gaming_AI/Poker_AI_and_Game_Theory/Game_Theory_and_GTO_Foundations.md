# Game Theory & GTO Foundations for Poker

> The mathematical core of poker as a research object: Nash equilibrium, GTO vs exploitative play, the standard poker-math toolkit, toy games, ICM and variance/bankroll theory (teoria dos jogos, GTO, gestão de banca). Curated strictly for **research and education** — every paper, book, tool and article below was verified live before inclusion.

---

## ⚠️ Responsible Gambling First (Jogo Responsável)

**This page is a research/education index, NOT gambling advice or an inducement to play.** Facts to internalize before studying any of the material below:

- **Most players lose long-term.** Poker is zero-sum *before* rake (a taxa da casa); after rake it is negative-sum, so the majority of participants are net losers by construction.
- **Variance (variância) is enormous.** The bankroll math at the bottom of this page exists precisely because even winning players endure downswings lasting tens of thousands of hands.
- **Bots and real-time assistance (RTA) are banned by poker sites.** Everything here is for offline study and AI research only. [GGPoker's security & ecology policy](https://ggpoker.com/network/security-ecology-policy/) prohibits bots, RTA, solvers, charts and all third-party HUDs (only its built-in Smart HUD is allowed), and PokerStars bans RTA and automation with fund confiscation for violations — GGPoker confiscated ~$1.2M from 42 bot-linked accounts in January 2026 alone ([industry overview of 2025 bot/RTA enforcement](https://thepokeroffer.com/poker-bots-2025-detection-threats-strategies/), [per-site HUD rules 2025](https://pokerbotai.com/blog/which-poker-sites-allow-huds-in-2025-full-breakdown/)). Deploying agents on real-money sites violates terms of service and may be criminal fraud in some jurisdictions.
- **Online-poker legality varies by country.** Check local law; see the Brazil note below.

| Support resource | Region | Link | Cost |
|---|---|---|---|
| GambleAware (BeGambleAware) | 🇬🇧 UK / global info | [gambleaware.org](https://www.gambleaware.org/) | Free |
| GamCare | 🇬🇧 UK helpline & chat | [gamcare.org.uk](https://www.gamcare.org.uk/) | Free |
| Gambling Therapy (Gordon Moody) | 🌍 Global, **Portuguese support incl. Brazil** | [gamblingtherapy.org](https://www.gamblingtherapy.org/) | Free |
| Jogo Responsável — SPA / Ministério da Fazenda | 🇧🇷 Brazil official guidance | [gov.br/fazenda — jogo responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) | Free |
| CVV — Centro de Valorização da Vida | 🇧🇷 Brazil emotional support, 24/7, dial **188** | [cvv.org.br](https://cvv.org.br/) | Free |

If study stops being intellectually motivated and starts feeling compulsive, use the resources above.

---

## Core Concepts (Conceitos Fundamentais)

| Concept | One-line definition | Canonical reference |
|---|---|---|
| **Nash equilibrium** (equilíbrio de Nash) | Strategy profile where no player gains by deviating unilaterally; existence proven for finite games | Nash (1950), ["Equilibrium Points in n-Person Games"](https://www.pnas.org/doi/10.1073/pnas.36.1.48), *PNAS* 36(1):48–49, [DOI 10.1073/pnas.36.1.48](https://doi.org/10.1073/pnas.36.1.48) |
| **GTO vs exploitative** | GTO = unexploitable equilibrium baseline; exploitative = maximally profitable deviation vs a specific opponent's mistakes | Chen & Ankenman, *The Mathematics of Poker* (see Books); [Brokos, *Play Optimal Poker*](https://www.pokernews.com/news/2019/06/pokernews-book-review-play-optimal-poker-by-andrew-brokos-34596.htm) |
| **Mixed strategies & indifference** | Equilibrium in poker requires randomizing so the opponent's bluff-catchers/bluffs become indifferent (equal EV) between actions | Chen & Ankenman, *The Mathematics of Poker*, toy-game chapters |
| **MDF & alpha** | Minimum Defense Frequency = how often you must continue so 0%-equity bluffs don't profit; α is its complement | [GTO Wizard: MDF & Alpha](https://blog.gtowizard.com/mdf-alpha/) |
| **Pot odds / implied odds** | Price offered by the pot on a call; implied odds add expected future winnings | Sklansky, *The Theory of Poker* (see Books) |
| **Equity, EV & equity realization** | Equity = showdown pot share if checked down; EV adjusts for position, playability, initiative (EQR = pot share ÷ raw equity) | [GTO Wizard: What is Equity](https://blog.gtowizard.com/what-is-equity-in-poker/) · [Equity Realization](https://blog.gtowizard.com/equity-realization/) |
| **Ranges & combinatorics** | Reasoning over distributions of hands (1,326 preflop combos), not single holdings; blockers shift combo counts | Janda, *Applications of No-Limit Hold 'em* (see Books) |
| **ICM** (Independent Chip Model) | Malmuth–Harville mapping from tournament chip stacks to $ equity; drives risk premium and bubble play | [HoldemResources ICM calculator (Malmuth–Harville)](https://www.holdemresources.net/icmcalculator) · [GTO Wizard: ICM Basics](https://blog.gtowizard.com/icm-basics/) · Gilbert (2009), ["The ICM and Risk Aversion"](https://arxiv.org/abs/0911.3100) |
| **Nash push/fold** | Equilibrium jam/call ranges for short-stack tournament endgames | [HoldemResources HU push/fold Nash equilibrium](https://www.holdemresources.net/hune) · [Nash ICM calculator](https://www.holdemresources.net/nashicm) |

---

## The Math Toolkit (Fórmulas Essenciais)

Facing a bet **B** into a pot **P** (bet in pot-fraction units s = B/P):

| Quantity | Formula | Verified source |
|---|---|---|
| Expected value | EV = Σ p(outcome) × payoff(outcome) | Chen & Ankenman, *The Mathematics of Poker* |
| Pot odds (equity needed to call) | B / (P + 2B) | Sklansky, *The Theory of Poker*; [GTO Wizard math misconceptions](https://blog.gtowizard.com/mathematical-misconceptions-in-poker/) |
| Alpha (fold equity a 0%-equity bluff needs) | α = B / (B + P) | [GTO Wizard: MDF & Alpha](https://blog.gtowizard.com/mdf-alpha/) |
| Minimum Defense Frequency | MDF = 1 − α = P / (P + B) | [GTO Wizard: MDF & Alpha](https://blog.gtowizard.com/mdf-alpha/) |
| River bluff ratio (polarized bettor) | bluffs = B / (P + 2B) of the betting range — e.g. pot-sized bet → 2:1 value:bluff | [GTO Wizard: MDF & Alpha](https://blog.gtowizard.com/mdf-alpha/) |
| Equity realization | EQR = pot share ÷ raw equity | [GTO Wizard: Equity Realization](https://blog.gtowizard.com/equity-realization/) |
| Geometric pot growth | equal pot-fraction bets on every remaining street so the river bet is all-in; maximizes EV of a perfectly polarized range | [GTO Wizard: Pot Geometry](https://blog.gtowizard.com/pot-geometry/) · [larger-than-geometric sizing](https://blog.gtowizard.com/why-so-much-an-exploration-of-larger-than-geometric-bet-sizing/) |
| Multi-street bluff EV | fold-equity compounding across streets — a multi-street bluff needs far less total fold equity than each bet suggests | [GTO Wizard: The Math of Multi-Street Bluffs](https://blog.gtowizard.com/the-math-of-multistreet-bluffs/) |

---

## Toy Games → Solved Games (Jogos-Brinquedo)

The pedagogical ladder used by both poker theorists and game-theory/AI courses:

- **AKQ game, [0,1] game, Clairvoyance game** — minimal models isolating bluffing frequency, bet sizing and indifference; systematically developed in Chen & Ankenman's *The Mathematics of Poker* (the AKQ/[0,1] analyses are the book's core) and modernized with solver outputs in Brokos's *Play Optimal Poker*.
- **Proof at scale — heads-up limit hold'em is (essentially weakly) solved:** Bowling, Burch, Johanson & Tammelin (2015), ["Heads-up limit hold'em poker is solved"](https://www.science.org/doi/10.1126/science.1259433), *Science* 347(6218):145–149, [DOI 10.1126/science.1259433](https://doi.org/10.1126/science.1259433). The **CFR+** algorithm (4,800 CPUs × 68 days, ~10¹⁴ information sets) produced **Cepheus**; the University of Alberta CPRG hosts a [CFR+ page](https://poker.cs.ualberta.ca/cfr_plus.html) and a queryable [Cepheus strategy site](http://poker.srv.ualberta.ca/strategy). This is the bridge from toy-game theory to the CFR-family solvers covered on the sibling agents/solvers pages.

---

## Foundational Books (Livros de Teoria)

All verified in print with publisher and ISBN. **All paid** unless noted.

| Book | Author(s) | Publisher / Year | ISBN-13 | Why it matters |
|---|---|---|---|---|
| *The Mathematics of Poker* | Bill Chen & Jerrod Ankenman | [ConJelCo, 2006](https://www.biblio.com/9781886070257) | 978-1-886070-25-7 | The rigorous foundation: toy games, indifference, risk of ruin; written by a Berkeley math PhD |
| *The Theory of Poker* | David Sklansky | [Two Plus Two Publishing, 4th ed. 1999](https://www.biblio.com/9781880685006) | 978-1-880685-00-6 | Pre-solver classic; Fundamental Theorem of Poker, pot/implied odds |
| *Applications of No-Limit Hold 'em* | Matthew Janda | [Two Plus Two Publishing, 2013](https://www.abebooks.com/9781880685556/Applications-No-Limit-Hold-Guide-Understanding-1880685558/plp) | 978-1-880685-55-6 | First mainstream range-construction/frequency book of the GTO era |
| *Expert Heads Up No Limit Hold'em, Vol. 1* (2012) & *Vol. 2* (2014) | Will Tipton | [D&B Publishing — Vol. 1](https://dandbpoker.com/products/expert-heads-up-no-limit-holdem-volume-1) · [Vol. 2](https://dandbpoker.com/products/expert-heads-up-no-limit-holdem-volume-2) | 978-1-904468-94-3 / 978-1-909457-03-4 | Decision trees, equilibrium computation, multi-street planning — closest to the AI literature |
| *Modern Poker Theory* | Michael Acevedo | [D&B Publishing, 2019](https://dandbpoker.com/products/modern-poker-theory) | 978-1-909457-89-8 | Solver-era synthesis of GTO principles for cash and MTT |
| *Play Optimal Poker* | Andrew Brokos | [Independently published, 2019](https://www.pokernews.com/news/2019/06/pokernews-book-review-play-optimal-poker-by-andrew-brokos-34596.htm) | 978-1-07-098272-4 | Most accessible on-ramp: game theory via toy games, no solver required |

---

## ICM & Tournament Equilibria (Torneios)

| Resource | Type | Free/Paid | Link |
|---|---|---|---|
| Independent Chip Model overview | Encyclopedia article | Free | [Wikipedia: Independent Chip Model](https://en.wikipedia.org/wiki/Independent_Chip_Model) |
| GTO Wizard ICM article series | Risk premium, bubble factor, pay-jump strategy | Free | [ICM Basics](https://blog.gtowizard.com/icm-basics/) · [MDF vs ICM in MTTs](https://blog.gtowizard.com/mdf-vs-icm-rethinking-bluffing-defense-strategies-in-mtts/) |
| Gilbert (2009), "The Independent Chip Model and Risk Aversion" | Peer-reviewable math analysis of ICM's risk-aversion properties | Free | [arXiv:0911.3100](https://arxiv.org/abs/0911.3100) |
| HoldemResources free tools | Nash ICM push/fold calculator, Malmuth–Harville ICM calculator (≤20 players), HU Nash tables | Free | [holdemresources.net/free-tools](https://www.holdemresources.net/free-tools) |
| HoldemResources Calculator (HRC) | Full preflop tournament solver (Win/macOS/Linux) | Paid (trial) | [holdemresources.net](https://www.holdemresources.net/) |
| ICMIZER 3 | ICM/Nash calculator suite (SNG/MTT/PKO) | Paid ([free basic ICM calc](https://www.icmizer.com/icmcalculator/)) | [icmizer.com](https://www.icmizer.com/icmizerapp/) |

---

## Variance, Risk of Ruin & Bankroll (Variância e Gestão de Banca)

This section doubles as the honest warning: the same math that sizes a bankroll proves how brutal the swings are.

- **Risk of ruin:** with win rate *WR*, standard deviation *σ* (both per 100 hands) and bankroll *BR*, the standard Brownian-motion approximation is **RoR ≈ e^(−2·WR·BR/σ²)** (derived in Chen & Ankenman's *The Mathematics of Poker*, risk-of-ruin chapters). It is exponential: doubling your edge or bankroll squares down your ruin probability — and a zero or negative win rate makes ruin certain.
- **Typical magnitudes:** online NLHE cash standard deviations run on the order of ~80–120 bb/100 (higher for loose-aggressive styles and PLO), dwarfing realistic win rates of 0–10 bb/100 — run the numbers yourself in the free [Primedope cash-game variance calculator](https://www.primedope.com/poker-variance-calculator/), which takes WR and σ straight from tracker software and simulates confidence intervals over any sample.
- **MTT downswings are worse:** high field sizes and top-heavy payouts mean even elite ROI players face six-figure-tournament droughts; simulate with the free [Primedope tournament variance calculator](https://www.primedope.com/tournament-variance-calculator/).
- **Kelly criterion and fractional Kelly:** optimal-growth bet sizing f* = edge/odds ([Kelly 1956, Wikipedia overview](https://en.wikipedia.org/wiki/Kelly_criterion)); poker practice uses fractional Kelly because win rates are estimated with error — overbetting Kelly is far costlier than underbetting it.
- **The blunt takeaway (repetindo o aviso):** after rake, the player pool as a whole loses money. Bankroll formulas manage the variance of an edge; they cannot create one. Treat everything on this page as applied game theory, not as an income plan.

---

## 🇧🇷 Brazil Notes (Notas para o Brasil)

- Poker is treated in Brazil as a **skill game (jogo de habilidade)**; peer-to-peer poker was not brought into the fixed-odds regime of the Lei das Bets (Lei nº 14.790/2023), leaving it legal but largely unregulated — see [Chambers Gaming Law – Brazil](https://practiceguides.chambers.com/practice-guides/gaming-law-2025/brazil) and the [ICLG Brazil gambling chapter](https://iclg.com/practice-areas/gambling-laws-and-regulations/brazil/). Regulatory classification by the SPA/MF remains an open 2025–2026 policy question. **Not legal advice.**
- Portuguese-language study terms used above: teoria dos jogos (game theory), equilíbrio de Nash, blefe (bluff), odds do pote (pot odds), valor esperado (EV), gestão de banca (bankroll management), variância (variance).
- Support in Portuguese: [Gambling Therapy](https://www.gamblingtherapy.org/) and [CVV — 188](https://cvv.org.br/) (see the responsible-gambling table above).

---

## How This Fits the Research Index

GTO theory is the human-readable face of the same equilibrium computation that powers CFR-family poker AI: Nash (1950) guarantees the object exists, CFR+ ([Bowling et al. 2015](https://www.science.org/doi/10.1126/science.1259433)) computes it at scale, and the books/articles above compress it into heuristics humans can execute. See the sibling pages in `Poker_AI_and_Game_Theory/` for solvers and agents (Libratus/Pluribus lineage), datasets, and the human training ecosystem. **Reminder: research and education only — bots, RTA and unauthorized HUDs are banned by every major site.**

**Sources:** [Nash 1950, PNAS](https://www.pnas.org/doi/10.1073/pnas.36.1.48) · [Bowling et al. 2015, Science](https://www.science.org/doi/10.1126/science.1259433) · [UAlberta CFR+/Cepheus](https://poker.cs.ualberta.ca/cfr_plus.html) · [Cepheus strategy site](http://poker.srv.ualberta.ca/strategy) · [arXiv:0911.3100](https://arxiv.org/abs/0911.3100) · [GTO Wizard: MDF & Alpha](https://blog.gtowizard.com/mdf-alpha/) · [ICM Basics](https://blog.gtowizard.com/icm-basics/) · [Equity Realization](https://blog.gtowizard.com/equity-realization/) · [Pot Geometry](https://blog.gtowizard.com/pot-geometry/) · [Math of Multi-Street Bluffs](https://blog.gtowizard.com/the-math-of-multistreet-bluffs/) · [Chen & Ankenman (ConJelCo 2006)](https://www.biblio.com/9781886070257) · [Sklansky (2+2 1999)](https://www.biblio.com/9781880685006) · [Janda (2+2 2013)](https://www.abebooks.com/9781880685556/Applications-No-Limit-Hold-Guide-Understanding-1880685558/plp) · [Tipton Vol 1](https://dandbpoker.com/products/expert-heads-up-no-limit-holdem-volume-1) · [Tipton Vol 2](https://dandbpoker.com/products/expert-heads-up-no-limit-holdem-volume-2) · [Acevedo (D&B 2019)](https://dandbpoker.com/products/modern-poker-theory) · [Brokos review (PokerNews 2019)](https://www.pokernews.com/news/2019/06/pokernews-book-review-play-optimal-poker-by-andrew-brokos-34596.htm) · [HoldemResources free tools](https://www.holdemresources.net/free-tools) · [ICMIZER](https://www.icmizer.com/icmizerapp/) · [Primedope variance calculators](https://www.primedope.com/) · [GGPoker security & ecology policy](https://ggpoker.com/network/security-ecology-policy/) · [HUD rules by site 2025](https://pokerbotai.com/blog/which-poker-sites-allow-huds-in-2025-full-breakdown/) · [Bot enforcement 2025–26](https://thepokeroffer.com/poker-bots-2025-detection-threats-strategies/) · [Chambers Gaming Law Brazil](https://practiceguides.chambers.com/practice-guides/gaming-law-2025/brazil) · [gambleaware.org](https://www.gambleaware.org/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [gamblingtherapy.org](https://www.gamblingtherapy.org/) · [cvv.org.br](https://cvv.org.br/)

**Keywords:** game theory, GTO, Nash equilibrium, mixed strategy, indifference, minimum defense frequency, MDF, alpha, pot odds, implied odds, equity realization, range construction, blockers, combinatorics, ICM, Malmuth-Harville, Nash push fold, CFR+, Cepheus, heads-up limit hold'em solved, Mathematics of Poker, Theory of Poker, Applications of No-Limit Hold'em, Modern Poker Theory, Play Optimal Poker, risk of ruin, Kelly criterion, bankroll management, poker variance, responsible gambling | teoria dos jogos, equilíbrio de Nash, odds do pote, valor esperado, gestão de banca, variância, jogo responsável, jogo de habilidade, poker brasileiro
