# Poker Variants — PLO, Short-Deck & Mixed Games (Variantes de Poker — PLO, Short-Deck e Jogos Mistos)

Fact-checked reference on poker beyond No-Limit Hold'em — Pot-Limit Omaha (PLO/PLO5/PLO6), Short-Deck (6+) Hold'em, stud/draw variants and mixed rotations — and why they matter for game-theory and AI research. **Research, education and off-table study only.** Every site, tool, book, paper and number below was verified live in July 2026.

---

## Responsible Gambling First (Jogo Responsável)

> **These games are typically played for money and carry real financial risk. PLO and Short-Deck in particular run much higher variance than NLHE — equities are closer, pots grow faster, and bankroll swings are larger. If gambling stops being fun, seek help — it is free and confidential (se o jogo deixou de ser diversão, procure ajuda gratuita e sigilosa):**
>
> - **Brazil — CVV (Centro de Valorização da Vida):** call **188**, free, 24h nationwide, plus chat/e-mail — [cvv.org.br](https://cvv.org.br/)
> - **Brazil — Jogo Responsável (Secretaria de Prêmios e Apostas / Ministério da Fazenda):** official player-protection guidance and self-exclusion rules (Portaria SPA/MF nº 1.231/2024; licensed sites use the `.bet.br` domain) — [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel)
> - **UK — National Gambling Helpline (GamCare):** **0808 8020 133**, free, 24/7 — [gamcare.org.uk](https://www.gamcare.org.uk/)
> - **International — Gambling Therapy (Gordon Moody charity):** free multilingual online support in 35 languages, **including Português (Brasil)** — [gamblingtherapy.org](https://www.gamblingtherapy.org/)
>
> Solvers and trainers listed here are for **study away from the tables** — real-time assistance (RTA) during play is banned by essentially every poker site.

---

## Why Variants Matter for AI Research

NLHE is close to "solved" at scale by modern CFR-family solvers; the variants below explode the game tree and remain open research territory. Starting-hand combinatorics (exact, C(n,k) of the deck):

| Game | Hole cards | Deck | Starting combos |
|---|---|---|---|
| Texas Hold'em | 2 | 52 | C(52,2) = **1,326** |
| Short-Deck (6+) Hold'em | 2 | 36 | C(36,2) = **630** |
| Pot-Limit Omaha (PLO4) | 4 | 52 | C(52,4) = **270,725** |
| 5-Card PLO (PLO5) | 5 | 52 | C(52,5) = **2,598,960** |
| 6-Card PLO (PLO6) | 6 | 52 | C(52,6) = **20,358,520** |

More hole cards also mean vastly larger information sets postflop — which is why full-tree PLO solving demands abstraction (MonkerSolver) or massive cloud precomputation (GTO Wizard PLO, PLO Genius), and why PLO5/PLO6 tooling is still catching up.

## Pot-Limit Omaha (PLO)

**Rules delta vs NLHE:** four hole cards (five in PLO5, six in PLO6), and you must use **exactly two** of them plus exactly three board cards; betting is pot-limit, not no-limit.

**Strategic consequences:**
- Preflop equities run much closer together than in NLHE, so single-raised and 3-bet pots are contested more often — variance is structurally higher (see the responsible-gambling note above).
- Nut-focus (foco no nuts): with four+ cards per player, non-nut flushes and low straights are frequently dominated; hand values are driven by nuttiness and redraws.
- Blockers matter more: holding nut-flush or nut-straight cards changes bluffing math dramatically.

### Solvers & trainers (verified July 2026)

| Tool | Site | Coverage | Pricing | Free option? |
|---|---|---|---|---|
| **MonkerSolver** | [monkerware.com](https://monkerware.com/) | Omaha & Hold'em, any street, any number of players (the reference desktop solver for PLO trees; heavy RAM) | **€499** full version | Free version limited to turn/river |
| **GTO Wizard PLO** | [gtowizard.com/plo](https://gtowizard.com/plo/) | **PLO4 only** (cash + MTT), preflop-to-river presolved + on-demand solving; per its FAQ, PLO5/PLO6 are *"in development with no ETA"* | Premium US$139/mo and Elite US$179/mo (early-bird, billed annually; regular $179/$239) | Yes — 1 postflop spot/day, 5 trainer hands/day |
| **PLO Genius** | [plogenius.com](https://plogenius.com/) | PLO4 + PLO5 (Edge tier and up), PLO6 (PRO tier); presolved preflop/flop libraries + on-demand turn/river solver + drill trainer | Edge US$59/mo, PRO US$125/mo (both billed annually: $708/$1,500) | Yes — Starter tier: 100bb HU/6-max PLO4 preflop sims |
| **Vision GTO Trainer** | [runitonce.com/vision-gto-trainer](https://www.runitonce.com/vision-gto-trainer/) | Run It Once's browser trainer for **4-card and 5-card PLO** (HU + 6-max, preflop-to-river; 20,000+ solved boards for PLO4, 7,500+ for PLO5) | Paid plans per game type | Trial/paid (see site) |
| **RangeConverter** | [rangeconverter.com](https://rangeconverter.com/) | GTO trainer + range viewer incl. PLO/PLO5 and short deck | Reg/Pro tiers paid | Yes — free 15bb HU spots |

### Books (verified editions)

- Jeff Hwang, *Pot-Limit Omaha Poker: The Big Play Strategy* — Kensington Publishing, **2008** ([Google Books](https://books.google.com/books/about/Pot_Limit_Omaha_Poker.html?id=prizf2Ila04C)). The classic starting text.
- Fernando "JNandez" Habegger, *Mastering Small Stakes Pot-Limit Omaha: How to Crush Modern PLO Games* — D&B Poker, **2020** ([dandbpoker.com](https://dandbpoker.com/products/mastering-small-stakes-pot-limit-omaha-how-to-crush-modern-plo-games)). Modern, solver-era treatment.

## Short-Deck (6+) Hold'em

**Rules delta vs NLHE:** all 2s–5s removed → **36-card deck**; typically played **ante-only** (everyone antes, button posts extra) rather than with blinds. Hand rankings change with the card-removal math: **a flush beats a full house** (flushes are rarer with only nine cards per suit), and in some rule sets three of a kind beats a straight. The ace plays low in the bottom straight, so the lowest straight is **A-6-7-8-9** ([PokerNews hand rankings](https://www.pokernews.com/poker-hands/short-deck.htm), [Upswing rules guide](https://upswingpoker.com/short-deck-six-plus-holdem-strategy/)).

**Strategic consequences:** equities run much closer (more gamble per all-in — again, higher variance), open-ended straight draws are relatively stronger, and flushes become premium holdings under Triton rules.

**History (verified):** the game emerged in Macau high-stakes cash games (mid-2010s), with high-roller organizers **Paul Phua and Richard Yong** credited for popularizing the format ([Triton Poker explainer](https://triton-series.com/understand-short-deck-holdem/)). The **first-ever live Short Deck tournament** ran at the **2018 Triton Super High Roller Series Montenegro**: **Phil Ivey won** the HKD $250,000 Short Deck Ante-Only event (61 entries), beating Dan Cates heads-up for **HK$4,749,200 (~US$604,977)** ([PokerNews report](https://www.pokernews.com/news/2018/05/phil-ivey-triton-poker-short-deck-montenegro-30754.htm), [CardPlayer](https://www.cardplayer.com/poker-news/22786-phil-ivey-wins-triton-poker-montenegro-250-000-hkd-short-deck-event)). USD figures for this prize vary slightly by source because of exchange-rate rounding; the HKD amount is the official one.

## Stud, Draw & Mixed Games

Anchored to the verified **2025 WSOP (56th Annual)** schedule — these championship events are where mixed-game specialists compete ([PokerNews 2025 WSOP schedule](https://www.pokernews.com/tours/wsop/2025-wsop/schedule.htm)):

| Variant | Core rules delta | Strategic flavor | 2025 WSOP anchor |
|---|---|---|---|
| **Seven Card Stud** | No community cards; 7 cards each (3 down/4 up); fixed-limit | Up-card reading, dead-card memory | Event #25: $10,000 Championship |
| **Razz** | Stud low: best A-to-5 low wins; straights/flushes don't count against | Inverted values; steal-heavy early streets | Event #50: $10,000 Championship |
| **Stud Hi-Lo (8 or Better)** | Stud pot split between best high and best qualifying 8-low | Scooping (winning both halves) drives everything | Event #77: $10,000 Championship |
| **2-7 Triple Draw** | 5-card draw, 3 draws, best deuce-to-seven low; aces high, straights/flushes count against | Snowing (standing pat as a bluff), draw-count leverage | Event #71: $10,000 Limit Championship (also NL 2-7 Single Draw, Event #30 $10,000) |
| **Badugi** | 4-card draw low; best hand is four cards, all different suits and ranks (A-2-3-4 rainbow) | Suit/rank removal math unique among variants | Event #23: $1,500 Badugi |
| **H.O.R.S.E.** | Rotation: Hold'em, Omaha Hi-Lo, Razz, Stud, Stud Eight-or-better | Rewards breadth, punishes one-game specialists | Event #39: $1,500 · Event #55: $10,000 Championship |
| **8-Game / 9-Game Mix** | H.O.R.S.E. plus 2-7 Triple Draw, NLHE, PLO (9-game adds more) | The all-rounder's test | Event #82: $10,000 Eight Game Mixed Championship · Event #58: $3,000 Nine Game Mix · Event #66: $50,000 Poker Players Championship |
| **Dealer's Choice** | Each dealer picks the game from a ~20-variant menu | Game selection itself becomes strategy | Event #18: $10,000 Dealer's Choice 6-Handed Championship |

## Open-Source & Academic Support for Variants

| Resource | Variant coverage | Notes |
|---|---|---|
| [uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit) (MIT) | **Best-in-class**: NLHE, short-deck, PLO, fixed-limit 2-7 Triple Draw, fixed-limit Badugi and more, plus custom-game definitions | Paper: J. Kim, "PokerKit: A Comprehensive Python Library for Fine-Grained Multi-Variant Poker Game Simulations," *IEEE Trans. on Games* 17(1):32–39, 2025, DOI [10.1109/TG.2023.3325637](https://doi.org/10.1109/TG.2023.3325637) |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) (Apache-2.0) | Kuhn poker, Leduc poker, ACPC-style Hold'em (`universal_poker`) — **no Omaha, no short-deck** ([games list](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md)) | Reference CFR/RL framework; variant gap = research opportunity |
| [datamllab/rlcard](https://github.com/datamllab/rlcard) (MIT) | Leduc, Limit and No-Limit Hold'em (plus non-poker card games) — **no Omaha** | RL baselines toolkit |
| [Poker-CNN, arXiv:1509.06731](https://arxiv.org/abs/1509.06731) | Yakovenko, Cao, Raffel & Fan, "Poker-CNN: A Pattern Learning Strategy for Making Draws and Bets in Poker Games" (AAAI-16 era) — evaluated on video poker, Limit Hold'em **and 2-7 Triple Draw** | Early demonstration that one architecture can learn multiple variants |
| [K. Ho, "A No-Limit Omaha Hi-Lo Poker Jam/Fold Endgame Equilibrium" (Harvard DASH, 2015)](https://dash.harvard.edu/entities/publication/73120378-e436-6bd4-e053-0100007fdf3b) | CFR+ on OpenCL/AWS computing a jam/fold ε-Nash equilibrium for NL Omaha Hi-Lo endgames | One of the few public academic treatments of Omaha with CFR+ |
| [HenryRLee/PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) (Apache-2.0) | `phevaluator` — evaluates PLO4/PLO5/PLO6 hands | Low-level building block for variant research |

## Master Comparison

| Variant | Rules delta vs NLHE | Key strategic difference | Best tools/refs |
|---|---|---|---|
| PLO4 | 4 hole cards, use exactly 2; pot-limit | Nut-focus, close equities, blocker-driven bluffing | MonkerSolver, GTO Wizard PLO, Vision GTO, Hwang 2008, Habegger 2020 |
| PLO5 / PLO6 | 5–6 hole cards | Even nuttier; solver coverage still partial | PLO Genius (PLO5/PLO6), Vision GTO (PLO5), MonkerSolver |
| Short-Deck | 36 cards, ante-only, flush > full house, A-6-7-8-9 | Compressed equities, draw-heavy aggression | PokerKit (simulation), RangeConverter, Triton/PokerNews rules pages |
| Stud family | Up-cards, no community board, fixed-limit | Card removal memory, up-card leverage | PokerKit; WSOP $10k Championships as the competitive benchmark |
| Draw lowball / Badugi | Hidden hands, multiple draws | Snowing, draw-count information | PokerKit (2-7 TD, Badugi), Poker-CNN paper |
| Mixed (HORSE/8-Game/DC) | Rotating games | Breadth over depth; game selection | WSOP Events #18/#39/#55/#58/#66/#82 (2025) |

---

**Sources:** [monkerware.com](https://monkerware.com/) · [gtowizard.com/plo](https://gtowizard.com/plo/) · [plogenius.com](https://plogenius.com/) · [runitonce.com/vision-gto-trainer](https://www.runitonce.com/vision-gto-trainer/) · [rangeconverter.com](https://rangeconverter.com/) · [Google Books — Hwang 2008](https://books.google.com/books/about/Pot_Limit_Omaha_Poker.html?id=prizf2Ila04C) · [D&B Poker — Habegger](https://dandbpoker.com/products/mastering-small-stakes-pot-limit-omaha-how-to-crush-modern-plo-games) · [PokerNews — short-deck rankings](https://www.pokernews.com/poker-hands/short-deck.htm) · [Upswing — short-deck rules](https://upswingpoker.com/short-deck-six-plus-holdem-strategy/) · [Triton — short-deck explainer](https://triton-series.com/understand-short-deck-holdem/) · [PokerNews — Ivey Triton Montenegro 2018](https://www.pokernews.com/news/2018/05/phil-ivey-triton-poker-short-deck-montenegro-30754.htm) · [CardPlayer — Ivey win](https://www.cardplayer.com/poker-news/22786-phil-ivey-wins-triton-poker-montenegro-250-000-hkd-short-deck-event) · [PokerNews — 2025 WSOP schedule](https://www.pokernews.com/tours/wsop/2025-wsop/schedule.htm) · [pokerkit](https://github.com/uoftcprg/pokerkit) · [PokerKit paper DOI](https://doi.org/10.1109/TG.2023.3325637) · [open_spiel games list](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md) · [rlcard](https://github.com/datamllab/rlcard) · [arXiv:1509.06731](https://arxiv.org/abs/1509.06731) · [Harvard DASH — Ho 2015](https://dash.harvard.edu/entities/publication/73120378-e436-6bd4-e053-0100007fdf3b) · [PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) · [gov.br Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [cvv.org.br](https://cvv.org.br/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [gamblingtherapy.org](https://www.gamblingtherapy.org/)

**Keywords:** Pot-Limit Omaha, PLO, PLO5, PLO6, short deck, 6+ hold'em, mixed games, H.O.R.S.E., 8-game, Razz, Badugi, 2-7 triple draw, stud, PLO solver, GTO trainer, poker variants AI, responsible gambling / Omaha pot-limit, variantes de poker, jogos mistos, baralho curto, solver de PLO, treinamento GTO, jogo responsável, apostas regulamentadas bet.br
