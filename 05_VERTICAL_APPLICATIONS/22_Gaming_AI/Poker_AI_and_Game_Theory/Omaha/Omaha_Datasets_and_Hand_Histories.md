# Omaha Datasets & Hand Histories (Bases de Dados e Históricos de Mãos de Omaha)

A fact-checked, deep index of **Omaha-specific data for machine learning** — public hand corpora that actually contain Omaha, the **PHH (Poker Hand History)** format's Omaha support, simulation engines you can turn into labelled Omaha data, and every hand-history converter/parser verified to read **Pot-Limit Omaha (PLO)**, **5-card PLO (PLO5)** and **Omaha Hi-Lo (Omaha-8)**. It is the *data* companion to the sibling [Omaha Solvers & Software](./Omaha_Solvers_and_Software.md) and [Omaha AI & Computational Research](./Omaha_AI_and_Computational_Research.md) pages, and it goes **much** deeper on Omaha than the Hold'em-centric parent [Poker Datasets & Hand Histories](../Datasets_and_Hand_Histories.md). Every archive, repository, DOI and format code below was opened and verified live on **2026-07-04**; anything that could not be confirmed against a primary source was **dropped rather than guessed**. For **research, education and off-table study only.**

> **The blunt reality of Omaha data (a verdade nua sobre dados de Omaha):** there is a *lot* of Texas Hold'em data and *very little* Omaha data. The famous modern corpus — the **640M-hand PHH archive** — contains **no Omaha hands at all**. The single largest genuinely-public Omaha hand corpus is 25+ years old, play-money, and low-quality by modern standards (the **IRC Poker Database**). In practice, most Omaha ML datasets are **simulated** (ProPokerTools / PQL, PokerKit, PokerRL-Omaha) or built from **your own** hand histories via a converter. This page lists only sources whose page or archive we actually opened, and flags the licensing/ToS traps around real-money hand data.

---

## ⚠️ Responsible Gambling — Read First (Jogo Responsável)

**This page is for AI research and education ONLY. It is not gambling advice and not a profit method.**

> **Omaha/PLO variance is *structurally higher* than No-Limit Hold'em** — four (or five/six) hole cards make preflop equities run *closer together*, pot-limit betting makes pots *grow faster*, and bankroll swings are *larger*; split-pot Omaha Hi-Lo adds "getting quartered" on top. **After rake (a taxa da casa), the player pool loses money net.** Owning a dataset or a model does not make Omaha beatable. If gambling stops being fun, seek help — it is free and confidential (se o jogo deixou de ser diversão, procure ajuda gratuita e sigilosa):

| Service | Where | Contact |
|---|---|---|
| **CVV — Centro de Valorização da Vida** | 🇧🇷 Brazil | Call **188** (free, 24h) or chat — [cvv.org.br](https://cvv.org.br/) |
| **Jogo Responsável (SPA / Ministério da Fazenda)** | 🇧🇷 Brazil | Player-protection rules, self-exclusion (autoexclusão); licensed sites use the **`.bet.br`** domain — [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) |
| **National Gambling Helpline (GamCare)** | 🇬🇧 UK | **0808 8020 133**, free, 24/7 — [gamcare.org.uk](https://www.gamcare.org.uk/) · consumer info [begambleaware.org](https://www.begambleaware.org/) |
| **Gambling Therapy (Gordon Moody)** | 🌍 International | Free multilingual online support, **including Português (Brasil)** — [gamblingtherapy.org/pt-br](https://www.gamblingtherapy.org/pt-br/) |

> **Poker bots and Real-Time Assistance (RTA) are banned by essentially every real-money poker site** — accounts are closed and funds confiscated. Real-money **hand histories contain other players' personal data**; mass scraping or redistributing them can violate site Terms of Service *and* data-protection law (**GDPR** in the EU, **LGPD** in Brazil). Prefer your own hands, the public research corpora, or synthetic/simulated data. Everything below is for **study away from the tables.**

---

## 1. The Data-Availability Map (o mapa da disponibilidade de dados)

| Source type | Contains **real** Omaha hands? | Best pick (verified) | Notes |
|---|:--:|---|---|
| **Public real-hand corpus** | ✅ (play-money, 1995–2001) | **IRC Poker Database** — `omaha`, `omahapot`, `omahahi`, `ohlpot` folders | Only large free Omaha corpus; old & weak population |
| **Modern PHH archive** | ➖ (Hold'em + 1 badugi hand only) | *format* supports Omaha (`PO`, `FO/8`); *corpus* has none | Great **format**, no Omaha data yet |
| **Simulation / query engine** | ✳️ (you generate it) | **ProPokerTools / PQL**, **PokerKit** | Unlimited labelled Omaha equity/outcome data |
| **RL self-play generator** | ✳️ (you generate it) | **PokerRL-Omaha** (Deep CFR/SD-CFR) | Writes close-to-PokerStars PLO hand logs |
| **Your own hands (via converter)** | ✅ (your table only) | FTR converter, HHSmithy parser, alexyz/poker, PokerNow converter | ToS/PII caveats apply |
| **Kaggle / Hugging Face** | ❌ (none Omaha-specific verified) | — | Poker sets there are Hold'em / UCI-classification |

---

## 2. The IRC Poker Database — the one real public Omaha corpus (o único corpus público real)

The **[IRC Poker Database](http://poker.cs.ualberta.ca/irc_poker_database.html)** (University of Alberta **Computer Poker Research Group**, from Michael Maurer's *Observer* logs of Internet Relay Chat poker, **1995–2001**, **>10 million complete hands**, free for research) is the **only large, genuinely public corpus that contains real Omaha play**. The parent [Datasets page](../Datasets_and_Hand_Histories.md) covers its Hold'em side; here we document the **Omaha** side, confirmed by opening the archive itself.

The full archive is a single gzip tarball — **[`poker.cs.ualberta.ca/IRC/IRCdata.tgz`](http://poker.cs.ualberta.ca/IRC/IRCdata.tgz)** (**~1.02 GB**, `Content-Length: 1017569472`, verified 2026-07-04) — holding monthly per-game sub-archives named `IRCdata/<game>.<YYYYMM>.tgz`. Listing the members directly reveals the **Omaha game categories** and how many monthly files each has:

| IRC game folder | Meaning | Monthly archives (observed) |
|---|---|:--:|
| **`omaha`** | Limit Omaha (Omaha Limitado) | **70** |
| **`omahapot`** | **Pot-Limit Omaha (PLO)** | **39** |
| **`omahahi`** | Omaha High | **14** |
| **`ohlpot`** | Omaha Hi-Lo, pot-limit (Omaha Alto-Baixo) | **13** |

*(For context, the same archive also holds Hold'em families `holdem`, `holdem1/2/3`, `holdemii`, `holdempot`, `nolimit`; stud `7stud`, `7studhi`; tournaments `tourney`, `ptourney`; and bot-filtered subsets `h1-nobots`, `botsonly`.)*

**Internal per-hand format (directly inspected).** Each `<game>/<YYYYMM>/` folder uses the classic three-part Maurer schema — identical for the Omaha folders, only the number of hole cards differs:

- **`hdb`** — the *hand database*: one line per hand — `timestamp  game/set#  hand#  #players`, then a `players/pot` pair per betting round, then board cards. Actual line from `omahapot/199703/hdb` (extracted and verified 2026-07-04; note the **five-card Omaha board**):
  ```
  857971299   1  5971  7  3/30    3/30    2/90     2/90     8d Ks 7h Qc Tc
  ```
- **`hroster`** — the *roster*: `timestamp  #players  name1 name2 …` (who was dealt in).
- **`pdb/pdb.<playername>`** — one *action file per player*: name, timestamp, #players, seat, per-round action strings (`c` call, `f` fold, `-` none, `cc`…), bankroll, amount wagered, amount won, and the cards held. **In the Omaha folders each player's card field carries four hole cards** — e.g. a verified line from `pdb.jvegas2` ends `… 229366   40   90 Kc Qs Ts 6h` (four hole cards).

> **Caveats (ressalvas):** the IRC data is **play-money**, **25+ years old**, and its population plays nothing like a modern real-money PLO game; use it for format-parsing, opponent-modelling *methodology*, and sequence-model prototyping — **not** for deriving 2026 strategy. Extraction helpers such as **[allenfrostline/PokerHandsDataset](https://github.com/allenfrostline/PokerHandsDataset)** parse this database to JSON (that particular script currently keeps *only* Hold'em, so the Omaha folders must be parsed with the same approach applied to `omaha*`/`ohlpot`).

---

## 3. The PHH Format & Omaha (o formato PHH e o Omaha)

The **PHH (Poker Hand History)** format — *"Recording and Describing Poker Hands,"* **Juho Kim, IEEE Conference on Games (CoG) 2024** ([arXiv:2312.11753](https://arxiv.org/abs/2312.11753); spec at [phh.readthedocs.io](https://phh.readthedocs.io/en/stable/), std repo [uoftcprg/phh-std](https://github.com/uoftcprg/phh-std)) — is the modern human-readable, machine-friendly hand-history standard, covering **11 variants**. **Two of them are Omaha:**

| PHH variant code | Variant | Portuguese |
|:--:|---|---|
| **`PO`** | Pot-limit Omaha hold'em | Omaha Pot-Limit |
| **`FO/8`** | Fixed-limit Omaha hold'em high/low-split eight-or-better | Omaha Alto-Baixo (8-or-better) |

So the **format fully supports Omaha** — but the flagship **corpus does not yet**. The curated **[uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset)** (MIT) and its **Zenodo archive** *"A Dataset of Poker Hand Histories"* (~**640M+ hands**, **20.3 GB**, **CC BY 4.0**; concept DOI **[10.5281/zenodo.10796885](https://doi.org/10.5281/zenodo.10796885)**, latest **v3** record [zenodo.org/records/17136841](https://zenodo.org/records/17136841), published 2025-09-16) contain **only Hold'em** (ACPC 2009–2017 logs, 21.6M July-2009 online NLHE hands, all 10,000 Pluribus hands, the 83 televised 2023 WSOP Event #43 hands) **plus a single badugi example — no Omaha hands.** The ~10,088-hand PHH *sample* set shipped with the paper likewise includes `PO`/`FO/8` support but is Hold'em-dominated.

**Bottom line:** if you want Omaha in PHH, you currently have to **generate it yourself** — which the same ecosystem makes easy (next section).

---

## 4. Simulation & Generation as a Data Source (simulação como fonte de dados)

Because real Omaha corpora are scarce, the standard research route is to **generate** labelled Omaha data. All three below are Omaha-capable and were verified.

### 4.1 PokerKit — simulate PLO / Omaha-8 and write PHH

**[uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit)** (MIT, **v0.7.4**, May 2026) is a pure-Python engine that both **simulates Omaha** and **reads/writes PHH natively** (`HandHistory.load()` / `HandHistory.dump()`). Verified Omaha surface in the source:

- **`PotLimitOmahaHoldem`** — full PLO game state/simulation (multiway, 2+ players).
- **`OmahaHoldemHand`** — the exactly-**two-of-four** hole + **three-of-five** board evaluator.
- **`OmahaEightOrBetterLowHand`** + **`FixedLimitOmahaHoldemHighLowSplitEightOrBetter`** — Omaha Hi-Lo (8-or-better) high and low evaluation.
- Notation map (in `pokerkit/notation.py`): `'PO' → PotLimitOmahaHoldem`, `'FO/8' → FixedLimitOmahaHoldemHighLowSplitEightOrBetter`.

```python
# Generate your own Omaha hands and serialise them to PHH
from pokerkit import Automation, PotLimitOmahaHoldem, HandHistory

state = PotLimitOmahaHoldem.create_state(
    tuple(Automation),           # automate posting/dealing/showdown
    True, 0, (500, 1000), 1000,  # antes, blinds (500/1000), starting stacks
    # ... deal 4 hole cards/player, drive actions ...
)
# hh = HandHistory.from_game_state(state); hh.dump(open("plo_hand.phh", "wb"))
# PHH variant field is written as "PO" (Pot-Limit Omaha)
```

### 4.2 ProPokerTools / Odds Oracle / PQL — scripted Monte-Carlo Omaha data

**[ProPokerTools](https://www.propokertools.com/)** provides free web equity/range tools and the downloadable **Odds Oracle**, whose **PQL (ProPokerTools Query Language)** — an SQL-flavoured query language — runs **Monte-Carlo simulations** (typically hundreds of thousands of randomized/exhaustive trials) and returns **percentages/counts** for precise "how often does X happen?" questions. Its game identifiers cover the whole Omaha family:

| PQL game id | Variant |
|---|---|
| `omahahi` | Omaha (high) |
| `omaha8` | Omaha Hi-Lo eight-or-better |
| `omahahi5` | **5-card** Omaha (high) |
| `omaha85` | **5-card** Omaha Hi-Lo eight-or-better |

This makes PQL a practical way to **synthesize labelled Omaha equity/outcome datasets** (hand class → win/tie/scoop probability, board-texture features, blocker effects) at any volume. Docs: [propokertools.com/oracle_help/main](https://www.propokertools.com/oracle_help/main) · [PQL reference](https://www.propokertools.com/oracle_help/pql). *(ProPokerTools' own servers were intermittently returning HTTP 503 to automated fetches on 2026-07-04; the four Omaha game-ids above were independently re-confirmed against the published PQL documentation via search on the same date, and cross-checked against the sibling [Omaha Solvers & Software](./Omaha_Solvers_and_Software.md) page.)*

### 4.3 PokerRL-Omaha — RL self-play that logs PLO hands

**[diditforlulz273/PokerRL-Omaha](https://github.com/diditforlulz273/PokerRL-Omaha)** (MIT, ~73★) is a research fork of *PokerRL* adding **Pot-Limit Omaha for 2–6 players** with **Deep CFR / SD-CFR**. Crucially for data work, it ships a **HandHistoryLogger** that *"writes actual hands played in close-to-PokerStars format in a `.txt` file"* (enabled by default in its standalone head-to-head evaluator) — i.e. an out-of-the-box generator of **synthetic PLO hand histories** in a tracker-friendly text layout. Its author notes it fills the gap that *"the Internet lacks any open-source Omaha Poker Reinforcement Learning code."*

---

## 5. Hand-History Converters & Parsers that support Omaha (conversores e parsers)

To build a dataset from **your own** Omaha play (home game, club, or your own online hands), these tools read/convert Omaha hand histories. All were verified to explicitly support Omaha variants.

| Tool | Type / language | Omaha support (verified) | Sites / input | License |
|---|---|---|---|---|
| **[FTR Hand History Converter](https://flopturnriver.com/hand-converter/)** | Free web converter | **PLO + PLO Hi/Lo** | PokerStars, Full Tilt (+ Bodog/Bovada, PartyPoker & others) → readable text | Free |
| **[HHSmithy/PokerHandHistoryParser](https://github.com/HHSmithy/PokerHandHistoryParser)** | C#/.NET library | **Pot-Limit Omaha, Fixed-Limit Omaha, Omaha Eight-or-Better** | ~12 sites (PokerStars, Full Tilt, Party, 888, iPoker, OnGame, Merge, Microgaming, Winamax, Winning, Boss…) | MIT |
| **[alexyz/poker](https://github.com/alexyz/poker)** | Java (Swing) parser + equity + HUD | **Hold'em, Omaha, Omaha Hi-Lo, Draw, Stud, Badugi** | PokerStars & Full Tilt hand histories | MIT |
| **[evolutionsoftswiss/pokernow-handhistory-converter](https://github.com/evolutionsoftswiss/pokernow-handhistory-converter)** | Java converter | **Pot-Limit Omaha High, Pot-Limit Omaha Hi/Lo**, NLHE | **PokerNow.club** CSV logs → PokerStars format (for PokerTracker/HM) | (unspecified) |

> Converting home-game/club logs (e.g. **PokerNow**) to PokerStars format lets standard trackers and parsers ingest your Omaha hands. This is the safest way to obtain *real* Omaha data — it is **your own table**, avoiding the PII/ToS problems of scraping other players. (Tracker/HUD policy and legality are covered on the parent [Tracking Software, HUDs & Analytics](../Tracking_Software_HUDs_and_Analytics.md) page.)

---

## 6. Kaggle & Hugging Face — the honest negative (o resultado honesto)

A search of **Kaggle** and **Hugging Face** on **2026-07-04** surfaced **no Omaha-specific hand-history dataset** whose page could be opened and verified. The poker datasets on those platforms are **Texas Hold'em / UCI-classification** oriented (e.g. the UCI Poker Hand mirror, script-generated Hold'em hands, real online **NLHE** logs, and LLM Hold'em benchmarks — all catalogued on the parent [Poker Datasets](../Datasets_and_Hand_Histories.md) and [Kaggle Poker Datasets](../Kaggle_Poker_Datasets.md) pages). The nearest real-hand pipeline, **[murilogmamaral/pokerdf](https://github.com/murilogmamaral/pokerdf)** (MIT; PokerStars text → Parquet), is **Hold'em only** — it does not parse Omaha. Per this index's anti-hallucination rule, **no Kaggle/HF dataset is listed here as "Omaha"** because none could be confirmed; if you find one, verify the actual dataset page before trusting the label.

---

## 7. Which source for which Omaha task? (qual fonte para qual tarefa?)

| Research task | Best starting source |
|---|---|
| Parse/opponent-model **real** Omaha play (methodology) | IRC Poker Database — `omaha` / `omahapot` / `ohlpot` folders |
| Build a labelled **PLO equity** dataset (hand → win/tie/scoop %) | ProPokerTools **PQL** (`omahahi`, `omaha8`, `omahahi5`, `omaha85`) |
| Standardise/serialise Omaha hands for tooling | **PHH** format via **PokerKit** (`PO`, `FO/8`) |
| Train an **Omaha RL** agent / self-play trajectories | **PokerRL-Omaha** (Deep CFR/SD-CFR, PLO logs) |
| Dataset from **your own** Omaha hands | FTR / HHSmithy / alexyz / PokerNow converter |
| Large modern **real** Omaha corpus | *(does not exist publicly — simulate instead)* |

---

**Related:** Sibling Omaha pages → [`Omaha/`](./) · [Omaha Variants & Rules](./Omaha_Variants_and_Rules.md) · [Omaha Strategy & GTO](./Omaha_Strategy_and_GTO.md) · [Omaha Math, Combinatorics & Equity](./Omaha_Math_Combinatorics_and_Equity.md) · [Omaha Solvers & Software](./Omaha_Solvers_and_Software.md) · [Omaha AI & Computational Research](./Omaha_AI_and_Computational_Research.md) · Parent section → [Poker AI & Game Theory](../README.md) · Hold'em data → [Poker Datasets & Hand Histories](../Datasets_and_Hand_Histories.md) · [Kaggle Poker Datasets](../Kaggle_Poker_Datasets.md) · Trackers → [Tracking Software, HUDs & Analytics](../Tracking_Software_HUDs_and_Analytics.md) · Broader context → [Poker Variants — PLO, Short-Deck & Mixed Games](../Poker_Variants_PLO_and_Mixed_Games.md)

**Sources:** [IRC Poker Database (CPRG)](http://poker.cs.ualberta.ca/irc_poker_database.html) · [IRCdata.tgz archive](http://poker.cs.ualberta.ca/IRC/IRCdata.tgz) · [allenfrostline/PokerHandsDataset](https://github.com/allenfrostline/PokerHandsDataset) · [arXiv:2312.11753 — Recording and Describing Poker Hands](https://arxiv.org/abs/2312.11753) · [PHH spec (readthedocs)](https://phh.readthedocs.io/en/stable/) · [uoftcprg/phh-std](https://github.com/uoftcprg/phh-std) · [uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset) · [Zenodo concept DOI 10.5281/zenodo.10796885](https://doi.org/10.5281/zenodo.10796885) · [Zenodo v3 record 17136841](https://zenodo.org/records/17136841) · [uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit) · [PokerKit notation docs](https://pokerkit.readthedocs.io/en/stable/notation.html) · [ProPokerTools](https://www.propokertools.com/) · [Odds Oracle help](https://www.propokertools.com/oracle_help/main) · [PQL reference](https://www.propokertools.com/oracle_help/pql) · [diditforlulz273/PokerRL-Omaha](https://github.com/diditforlulz273/PokerRL-Omaha) · [FTR Hand History Converter](https://flopturnriver.com/hand-converter/) · [HHSmithy/PokerHandHistoryParser](https://github.com/HHSmithy/PokerHandHistoryParser) · [alexyz/poker](https://github.com/alexyz/poker) · [evolutionsoftswiss/pokernow-handhistory-converter](https://github.com/evolutionsoftswiss/pokernow-handhistory-converter) · [murilogmamaral/pokerdf](https://github.com/murilogmamaral/pokerdf) · [gov.br — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [cvv.org.br](https://cvv.org.br/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [begambleaware.org](https://www.begambleaware.org/) · [gamblingtherapy.org/pt-br](https://www.gamblingtherapy.org/pt-br/)

**Keywords:** Omaha dataset, PLO dataset, Omaha hand history, Pot-Limit Omaha data, Omaha Hi-Lo data, IRC Poker Database, omahapot, ohlpot, PHH format, PHH Omaha PO FO/8, poker hand history format, PokerKit, PotLimitOmahaHoldem, OmahaEightOrBetter, phh-dataset, Zenodo poker hands, ProPokerTools, PQL, Odds Oracle, omahahi omaha8 omahahi5 omaha85, Monte Carlo simulation, PokerRL-Omaha, Deep CFR, hand history converter, PokerNow converter, HHSmithy parser, alexyz poker, Kaggle poker, Hugging Face poker, machine learning, responsible gambling / conjunto de dados de Omaha, histórico de mãos, base de dados PLO, Omaha Alto-Baixo, formato PHH, simulação Monte Carlo, dados sintéticos, aprendizado de máquina, conversor de históricos, jogo responsável, autoexclusão, LGPD, GDPR, rake, variância, apostas regulamentadas bet.br
