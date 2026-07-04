# Omaha Variants & Rules (Variantes e Regras do Omaha)

Exhaustive, fact-checked reference on every major form of Omaha poker — **PLO (4-card), 5-Card PLO (PLO5), 6-Card PLO (PLO6), Omaha Hi-Lo / Omaha-8 (eight-or-better), Big O, and Courchevel / Courchevel Hi-Lo** — plus the betting structures (Limit / Pot-Limit / No-Limit) and the single rule that defines the whole family: **use exactly two hole cards + exactly three board cards.** For **research, education and off-table study only.** Every rule, source and link below was verified live in July 2026.

> **Omaha in one sentence (Omaha em uma frase):** deal each player **four or more private hole cards** (four in PLO, five in PLO5/Big O/Courchevel, six in PLO6), deal **five shared community cards**, and every player makes their five-card hand from **exactly two of their hole cards plus exactly three of the board** — players "must use exactly two of their four/five/six down hole cards … and three of the five community cards to form their hand" ([WSOP](https://wsoponline.com/learn-to-play-poker/omaha/)). That one constraint changes hand-reading, equity and betting math versus Texas Hold'em far more than the extra cards alone suggest.

---

## Responsible Gambling First (Jogo Responsável)

> **Omaha is normally played for real money and its variance is *structurally higher* than No-Limit Hold'em — preflop equities run closer together, pots grow faster under pot-limit, and bankroll swings are larger (PLO standard deviation typically runs well above NLHE at the same stake). After rake, the player pool loses net. If gambling stops being fun, seek help — it is free and confidential (se o jogo deixou de ser diversão, procure ajuda gratuita e sigilosa):**
>
> - **Brazil — CVV (Centro de Valorização da Vida):** call **188**, free, 24h nationwide, plus chat/e-mail — [cvv.org.br](https://cvv.org.br/)
> - **Brazil — Jogo Responsável (Secretaria de Prêmios e Apostas / Ministério da Fazenda):** official player-protection guidance and self-exclusion rules (Portaria SPA/MF nº 1.231/2024; licensed operators use the `.bet.br` domain) — [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel)
> - **UK — GambleAware / National Gambling Helpline (GamCare):** **0808 8020 133**, free, 24/7 — [gambleaware.org](https://www.gambleaware.org/) · [gamcare.org.uk](https://www.gamcare.org.uk/)
> - **International — Gambling Therapy (Gordon Moody charity):** free multilingual online support, **including Português (Brasil)** — [gamblingtherapy.org/pt-br](https://www.gamblingtherapy.org/pt-br/)
>
> **Bots and real-time assistance (RTA) are banned by essentially every real-money poker site** — accounts are closed and funds confiscated. Solvers, trainers and the evaluators referenced here are for **study away from the tables**, never for live decision help.

---

## The Defining Rule — "Exactly Two + Exactly Three" (A Regra que Define o Omaha)

Every Omaha variant shares one non-negotiable rule that Hold'em does not have:

> "In Omaha, players must use exactly two of their four/five/six down hole cards (according to the Omaha variant the player chooses to play) and three of the five community cards to form their hand." — [WSOP](https://wsoponline.com/learn-to-play-poker/omaha/)

In Hold'em you may use **two, one, or zero** of your hole cards (you can "play the board"). In Omaha you must **always** use **exactly two** from your hand and **exactly three** from the board — for **four/five/six** hole cards alike ([WSOP](https://wsoponline.com/learn-to-play-poker/omaha/); [Upswing](https://upswingpoker.com/poker-rules/pot-limit-omaha-rules/)). This is the rule new players misread most, and even world-class players have misread their Omaha hands.

### How it changes hand-reading vs Hold'em (leitura de mãos)

| Trap on the board | Hold'em result | Omaha result (exactly two + three) |
|---|---|---|
| **Four aces in your hand** (A♠A♥A♦A♣) | impossible to hold | **Not quads.** Only two of your aces can play, so the best you make is a pair/two aces plus three board cards ([WSOP](https://wsoponline.com/learn-to-play-poker/omaha/)) |
| **Four of one suit on the board**, you hold one card of that suit | Flush (one hole card + four board is fine) | **No flush.** You must use *two* of your hole cards, so a single A♥ with four hearts on board makes **no** ace-high flush ([PokerNews](https://www.pokernews.com/poker-rules/omaha-poker.htm)) |
| **Board makes a straight/flush by itself** | Everyone can "play the board" and chop | **You cannot play the board** — you must still contribute exactly two hole cards, so you may not have the hand the board shows |
| **A pair on the board** for a full house | one hole card can pair the board | You need **two** specific hole cards to combine — trips/boats are read differently |

PokerNews' canonical example: holding **A♥Q♣7♦6♦** on a board of **9♥4♥2♣J♥Q♥**, you have no ace-high flush — "The Omaha poker rules **do not allow** you to make a hand using only one hole card (A♥) in combination with four community cards" ([PokerNews](https://www.pokernews.com/poker-rules/omaha-poker.htm)).

**Strategic knock-on effects.** Because each player mixes two of several hole cards with the board, the number of strong hands rises sharply — "hand values tend to be higher in Omaha than in Texas hold'em, with players making 'the nuts' … much more frequently" ([PokerNews](https://www.pokernews.com/poker-rules/omaha-poker.htm)). This is why Omaha strategy is dominated by **nut-focus (foco no nuts)**, **redraws**, and **blockers**, and why preflop equities run much closer than in NLHE.

### Two-card combinations per hand (combinações de duas cartas)

The "exactly two" rule means a hand's real playing power is the set of **two-card combinations** you can form — C(n,2):

| Game | Hole cards (n) | Two-card combos C(n,2) |
|---|---|---|
| Texas Hold'em | 2 | **1** |
| PLO (4-card) | 4 | **6** |
| 5-Card PLO / Big O / Courchevel | 5 | **10** |
| 6-Card PLO | 6 | **15** |

More combinations means most hands connect with most flops, dominance shrinks, and equities compress further as you add cards ([888poker — 5-card](https://www.888poker.com/magazine/strategy/5-card-omaha); [RangeConverter — PLO6](https://rangeconverter.com/articles/what-is-plo6-six-card-pot-limit-omaha-guide)). (Full *starting-hand* combinatorics of the deck — C(52,4)=270,725 for PLO, etc. — are tabulated on the sibling [Poker Variants](../Poker_Variants_PLO_and_Mixed_Games.md) page.)

---

## Betting Structures — Limit / Pot-Limit / No-Limit (Estruturas de Aposta)

The rules of hand construction are identical across every Omaha variant; only the **betting structure** differs ([WSOP](https://wsoponline.com/learn-to-play-poker/omaha/)).

| Structure | How bets are sized | Where you see it in Omaha |
|---|---|---|
| **Pot-Limit (PL)** — *dominant* | Maximum bet or raise = **the current size of the pot** | The default for PLO, PLO5, PLO6 and most Courchevel/Big O online games. "Pot-Limit is the most common form of Omaha poker" ([partypoker](https://www.partypoker.com/en/poker/how-to-play/games/omaha/index)) |
| **Fixed-Limit (FL)** | Bets/raises are **fixed increments** (small-bet / big-bet), capped raises per street | Traditional home for **Omaha Hi-Lo (Omaha-8)** and often **Big O** in card rooms; WSOP spreads Limit Omaha Hi-Lo 8-or-Better events ([Upswing](https://upswingpoker.com/poker-rules/pot-limit-omaha-rules/); [The Lodge — Big O](https://thelodgepokerclub.com/big-o-poker-rules-beginners-guide-to-the-omaha-variant/)) |
| **No-Limit (NL)** | Any amount up to your whole stack | **Rare** for Omaha — spread occasionally, but uncommon because the close equities make it a shove-fest |

### How the pot-limit maximum is computed (o cálculo do pot-limit)

A pot-sized raise = **the amount you must call, plus the size the pot would be after you call**. PokerNews' worked example: with **$10 in the pot** and a **$5 bet** in front of you, "the most that player can bet **is $25** … calculated by adding the $5 to call to the $20 that would be in the pot *after* the call" ([PokerNews](https://www.pokernews.com/poker-rules/omaha-poker.htm)). There is no single all-in shove available on the first action the way No-Limit allows.

### Why pot-limit dominates Omaha (por que o pot-limit domina)

With four+ hole cards, preflop equities are **compressed** — getting all-in "with the best hand" preflop in PLO is frequently close to a coin flip rather than the big favorite you'd be in Hold'em. Under **No-Limit**, that would collapse into early all-ins decided by tiny edges and enormous variance. The **pot-limit cap** forces pots to build **street by street** rather than in one shove, which (a) preserves multi-street postflop play and skill expression, and (b) partially tames the variance that close equities create ([Upswing](https://upswingpoker.com/poker-rules/pot-limit-omaha-rules/); [PokerNews](https://www.pokernews.com/poker-rules/omaha-poker.htm)). It still runs hotter than NLHE — see the responsible-gambling note above.

---

## The Omaha Family — Variant by Variant

All variants below inherit the **exactly two + exactly three** rule, five community cards, and (unless noted) the four betting rounds: preflop, flop, turn, river ([partypoker](https://www.partypoker.com/en/poker/how-to-play/games/omaha/index)).

### Pot-Limit Omaha (PLO / PLO4) — the standard

- **Four hole cards**, five community cards, pot-limit betting. The **second-most popular poker game in the world** after NLHE ([Upswing](https://upswingpoker.com/poker-rules/pot-limit-omaha-rules/)).
- Each player has **6** two-card combinations to work with; nut-focus, redraws and blockers define good play.
- Rules-of-thumb differ sharply from Hold'em: single-suited aces don't make a flush, four aces aren't quads, and you cannot play the board (see the table above).

### 5-Card PLO (PLO5)

- **Five hole cards** — otherwise identical to PLO. "Exactly two of these 5 hole-cards must be used to create a five-card hand" ([888poker](https://www.888poker.com/magazine/strategy/5-card-omaha)).
- **10** two-card combinations per hand → even nuttier boards and still-closer equities than 4-card PLO. Solver/trainer coverage is thinner than for PLO4 (see [Solvers & Open-Source Tools](../Solvers_and_Open_Source_Tools.md)).

### 6-Card PLO (PLO6)

- **Six hole cards**, played pot-limit. "In every form of Omaha, you have to use exactly two of your hole cards and exactly three from the board … That rule is what makes Omaha Omaha, and it doesn't change in PLO6" ([RangeConverter](https://rangeconverter.com/articles/what-is-plo6-six-card-pot-limit-omaha-guide)).
- **15** two-card combinations per hand → "the gap between premium hands and the rest of the field gets meaningfully smaller." Most ranges hit most flops; the highest-variance mainstream Omaha form.

### Omaha Hi-Lo / Omaha-8 (Eight-or-Better) — the split-pot classic

A **split-pot** game: half the pot to the best **high** hand, half to the best qualifying **low** hand ([Upswing](https://upswingpoker.com/poker-rules/omaha-hi-lo/)). Commonly spread **fixed-limit** (Limit O8) or pot-limit (PLO8).

- **The low qualifier ("eight or better"):** to win the low half you need **five different (unpaired) cards, all ranked eight or lower**. "A low hand qualifies for half of the pot when it is an 8-low or better" — worst qualifying low is **8-7-6-5-4** ([Upswing](https://upswingpoker.com/poker-rules/omaha-hi-lo/); [PokerNews](https://www.pokernews.com/poker-rules/omaha-hi-lo.htm)).
- **Ace-to-five ("California") low ranking:** the ace plays low, and **straights and flushes do not count against your low** — so the best possible low is the **wheel, 5-4-3-2-A**, which cannot be beaten ([PokerNews](https://www.pokernews.com/poker-rules/omaha-hi-lo.htm)). Lows are read from the top card down (the lower your highest card, the better).
- **Exactly two + three applies to each half independently:** "The cards that make your low hand don't have to be the same ones used to make your high hand" — though "the same cards **can** be used" too ([PokerNews](https://www.pokernews.com/poker-rules/omaha-hi-lo.htm); [Upswing](https://upswingpoker.com/poker-rules/omaha-hi-lo/)).
- **No qualifying low → high scoops:** if no one makes an 8-or-better low, the best high hand wins the **entire** pot ([Upswing](https://upswingpoker.com/poker-rules/omaha-hi-lo/)).
- **Scooping (scoop):** winning **both** the high and the low halves — the goal that drives Hi-Lo strategy ([Upswing](https://upswingpoker.com/poker-rules/omaha-hi-lo/)).
- **Quartering (ser "quarteado"):** if two players **tie** for one half (e.g. both hold the nut low), they split that half — each taking only a **quarter** of the whole pot. Committing chips with a hand that can only make a non-nut low risks "getting 'quartered' … splitting the high or low half of the pot with another player" ([The Lodge](https://thelodgepokerclub.com/big-o-poker-rules-beginners-guide-to-the-omaha-variant/)). Getting quartered while facing a scoop-threat is how Hi-Lo players bleed chips.

### Big O (5-Card Pot-Limit Omaha Hi-Lo)

- **Big O = 5-card Omaha Hi-Lo.** "In Big-O, all players are dealt **five** hole cards" and "Big-O generally plays using hi-lo (split pot) rules, where the best high hand and best qualifying low hand split each pot" ([The Lodge](https://thelodgepokerclub.com/big-o-poker-rules-beginners-guide-to-the-omaha-variant/)).
- Same **eight-or-better** low qualifier, ace-to-five low, wheel = nut low (5-4-3-2-A), exactly two + three rule. The extra hole card makes qualifying lows and scoops far more frequent than in 4-card O8.
- Frequently spread **fixed-limit** in US card rooms (e.g. The Lodge runs $1/$2/$5 Limit Big O with match-the-stack), and also as Pot-Limit.

### Courchevel & Courchevel Hi-Lo

Named after the French ski resort, Courchevel is **5-card Omaha with the first flop card exposed before preflop betting**:

- Each player gets **five hole cards**; then **the first community card — the "door card" — is dealt face up before any preflop betting** ([888poker](https://www.888poker.com/magazine/strategy/courchevel-poker); [PokerListings](https://www.pokerlistings.com/poker-rules/courchevel-poker)).
- After preflop betting, "**two more community cards are placed face up to complete 'the flop'**," then turn and river as usual ([888poker](https://www.888poker.com/magazine/strategy/courchevel-poker)).
- Same **two-from-hand + three-from-board** rule. "Courchevel Poker has historically been a **pot-limit** game … Players may also play it as fixed limit or no-limit" ([888poker](https://www.888poker.com/magazine/strategy/courchevel-poker)).
- **Courchevel Hi-Lo** is the split-pot version: half the pot to the high hand, half to the best qualifying **8-or-better** low, with scooping available — otherwise identical to 5-card Omaha Hi-Lo but with the exposed door card. Seeing one flop card preflop makes it "a game of big hands" — starting-hand values shift toward holdings that connect with the visible card.

---

## Comparison Table — Every Omaha Form

| Variant | Hole cards | 2-card combos | Board / dealing twist | Pot split? | Typical betting | Defining feature |
|---|---|---|---|---|---|---|
| **Texas Hold'em** *(reference)* | 2 | 1 | standard | No | No-Limit | may use 0/1/2 hole cards |
| **PLO (PLO4)** | 4 | 6 | standard | No | **Pot-Limit** | the baseline "use exactly 2 + 3" game |
| **5-Card PLO (PLO5)** | 5 | 10 | standard | No | Pot-Limit | nuttier boards, closer equities |
| **6-Card PLO (PLO6)** | 6 | 15 | standard | No | Pot-Limit | most ranges hit most flops; highest variance |
| **Omaha Hi-Lo / Omaha-8** | 4 | 6 | standard | **Yes (8-or-better low)** | Limit or Pot-Limit | scoop / quarter; wheel = nut low |
| **Big O** | 5 | 10 | standard | **Yes (8-or-better low)** | Limit or Pot-Limit | 5-card Omaha Hi-Lo; lows qualify often |
| **Courchevel** | 5 | 10 | **first flop card exposed pre-flop** | No | Pot-Limit (also FL/NL) | door card revealed before betting |
| **Courchevel Hi-Lo** | 5 | 10 | **first flop card exposed pre-flop** | **Yes (8-or-better low)** | Pot-Limit | Courchevel + split pot |

Hand rankings for the **high** hand are identical to Texas Hold'em in every variant ([partypoker](https://www.partypoker.com/en/poker/how-to-play/games/omaha/index)); only the low half uses the ace-to-five system.

---

## The Rules, Encoded in Software (as regras em código)

The exactly-two-plus-three constraint is exactly what makes Omaha harder to evaluate than Hold'em, and open-source libraries model it directly:

- **[HenryRLee/PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator)** (Apache-2.0) — evaluates **PLO4, PLO5 and PLO6** hands; its docs note "Omaha poker … requires picking exactly two cards from four player's cards, and exactly three cards from five community cards."
- **[uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit)** (MIT, v0.7.4, May 2026) — multi-variant simulator that includes Pot-Limit Omaha Hold'em and other Omaha forms with correct hand-construction rules.
- **[GTO Wizard PLO](https://gtowizard.com/plo/)** — presolved GTO study for **PLO4** (cash + tournaments); per its FAQ, "PLO5 and PLO6 are in development with no ETA at this time." (Study away from the tables only — RTA is banned.)

Deeper tooling, solver and dataset coverage lives on the sibling pages linked below.

---

## Official Rule Sources (verified to open, July 2026)

| Source | Page | Covers |
|---|---|---|
| **partypoker** | [partypoker.com — Omaha](https://www.partypoker.com/en/poker/how-to-play/games/omaha/index) | Four hole cards, precisely-two rule, pot-limit as the common form |
| **WSOP** | [wsoponline.com — Learn Omaha](https://wsoponline.com/learn-to-play-poker/omaha/) | 4/5/6-card Omaha, two-plus-three rule, the four-aces misconception |
| **PokerNews** | [Omaha (PLO) rules](https://www.pokernews.com/poker-rules/omaha-poker.htm) · [Omaha Hi-Lo rules](https://www.pokernews.com/poker-rules/omaha-hi-lo.htm) | Pot-limit math, the single-suited-ace trap, the 8-or-better low |
| **Upswing Poker** | [PLO rules](https://upswingpoker.com/poker-rules/pot-limit-omaha-rules/) · [Omaha Hi-Lo rules](https://upswingpoker.com/poker-rules/omaha-hi-lo/) | Exactly-two rule; scooping, worst qualifying low 8-7-6-5-4 |
| **888poker** | [5-Card Omaha](https://www.888poker.com/magazine/strategy/5-card-omaha) · [Courchevel](https://www.888poker.com/magazine/strategy/courchevel-poker) | PLO5 combos; Courchevel door card and betting |
| **The Lodge** | [Big O rules](https://thelodgepokerclub.com/big-o-poker-rules-beginners-guide-to-the-omaha-variant/) | Big O = 5-card Hi-Lo; quartering; limit structure |

---

**Related:** Sibling Omaha pages → [`Omaha/`](./) · Parent section → [Poker AI & Game Theory](../README.md) · Broader variant context → [Poker Variants — PLO, Short-Deck & Mixed Games](../Poker_Variants_PLO_and_Mixed_Games.md) · Study tools → [Solvers & Open-Source Tools](../Solvers_and_Open_Source_Tools.md) · Strategy math → [Game Theory & GTO Foundations](../Game_Theory_and_GTO_Foundations.md)

**Sources:** [partypoker — Omaha](https://www.partypoker.com/en/poker/how-to-play/games/omaha/index) · [wsoponline.com — Omaha](https://wsoponline.com/learn-to-play-poker/omaha/) · [PokerNews — Omaha (PLO)](https://www.pokernews.com/poker-rules/omaha-poker.htm) · [PokerNews — Omaha Hi-Lo](https://www.pokernews.com/poker-rules/omaha-hi-lo.htm) · [Upswing — PLO rules](https://upswingpoker.com/poker-rules/pot-limit-omaha-rules/) · [Upswing — Omaha Hi-Lo](https://upswingpoker.com/poker-rules/omaha-hi-lo/) · [888poker — 5-Card Omaha](https://www.888poker.com/magazine/strategy/5-card-omaha) · [888poker — Courchevel](https://www.888poker.com/magazine/strategy/courchevel-poker) · [The Lodge — Big O](https://thelodgepokerclub.com/big-o-poker-rules-beginners-guide-to-the-omaha-variant/) · [PokerListings — Courchevel](https://www.pokerlistings.com/poker-rules/courchevel-poker) · [RangeConverter — PLO6](https://rangeconverter.com/articles/what-is-plo6-six-card-pot-limit-omaha-guide) · [GTO Wizard PLO](https://gtowizard.com/plo/) · [pokerkit](https://github.com/uoftcprg/pokerkit) · [PokerHandEvaluator](https://github.com/HenryRLee/PokerHandEvaluator) · [gov.br — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [cvv.org.br](https://cvv.org.br/) · [gambleaware.org](https://www.gambleaware.org/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [gamblingtherapy.org/pt-br](https://www.gamblingtherapy.org/pt-br/)

**Keywords:** Omaha rules, Pot-Limit Omaha, PLO, PLO4, PLO5, PLO6, 5-card Omaha, 6-card Omaha, Omaha Hi-Lo, Omaha-8, eight or better, Big O, Courchevel, Courchevel Hi-Lo, split pot, low qualifier, wheel, scoop, quartered, exactly two hole cards, ace-to-five low, pot-limit betting, responsible gambling / regras do Omaha, Omaha Pot-Limit, Omaha Alto-Baixo, oito ou melhor, pote dividido, mão baixa, roda, escanteado, apostas pot-limit, jogo responsável, apostas regulamentadas bet.br
