# Poker Datasets & Hand Histories

> Verified index of public **poker datasets and hand-history corpora** (bases de dados e históricos de mãos de pôquer) for machine-learning and game-theory research: the classic UCI and IRC corpora, the modern PHH-format archive (600M+ hands), Pluribus's published hands, curated Kaggle datasets, parsing/simulation tools, and the legal/ethical caveats around HUDs and data scraping. Every link on this page was live-verified on 2026-07-03.

---

## ⚠️ Responsible Gambling — Read First (Jogo Responsável)

**This page is for AI research and education ONLY. It is not gambling advice and not a profit method.**

- Poker for money is **gambling with real financial risk**: variance (variância) is enormous and, after rake (taxa da casa), **most players lose money long-term**. Datasets here exist to study algorithms, not to build a bankroll.
- **Poker bots and real-time assistance are banned by real-money sites.** Running an AI on a live platform violates the terms of service of virtually every operator and leads to account bans and fund confiscation. Even *passive* tools are restricted: PokerStars only permits specific approved HUDs on its desktop client, while GGPoker prohibits third-party HUDs and trackers entirely (offering only its built-in Smart HUD / PokerCraft) — policies verified as of 2025–2026 and subject to change.
- **Online-poker legality varies by country** (a legalidade varia por país). 🇧🇷 In Brazil, licensed betting operates under the Secretaria de Prêmios e Apostas rules (Portaria SPA/MF nº 1.231/2024), including mandatory self-exclusion (autoexclusão) and limit-setting tools — see the official [Jogo Responsável page (gov.br)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel).

**If gambling is harming you or someone you know (free & confidential):**

| Resource | Coverage | Access |
|---|---|---|
| [GambleAware](https://www.gambleaware.org/) | 🇬🇧 UK | Advice, tools, treatment referrals (begambleaware.org now 301-redirects here) |
| [GamCare — National Gambling Helpline](https://www.gamcare.org.uk/) | 🇬🇧 UK | **0808 8020 133**, 24/7, phone/chat/WhatsApp |
| [Gambling Therapy (Gordon Moody)](https://www.gamblingtherapy.org/) | 🌍 Global — **tem Português (Brasil)** ([/pt-br](https://www.gamblingtherapy.org/pt-br/)) | Free online emotional support & groups |
| [CVV — Centro de Valorização da Vida](https://cvv.org.br/) | 🇧🇷 Brazil | **Ligue 188**, 24h, gratuito |
| [Jogo Responsável — gov.br](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) | 🇧🇷 Brazil | Official guidance + operator self-exclusion obligations |

---

## 1. Classic Research Corpora (Corpora Clássicos)

| Dataset | Content | Size | License / access | Link |
|---|---|---|---|---|
| **UCI Poker Hand Dataset** | Five-card hand classification (10 classes, suit+rank features); Cattral & Oppacher, created 2002, donated 2006 | **1,025,010 instances** | CC BY 4.0 | [archive.ics.uci.edu/dataset/158](https://archive.ics.uci.edu/dataset/158/poker+hand) |
| **IRC Poker Database** | Real play-money-era hands logged 1995–2001 from IRC poker channels by Michael Maurer's *Observer* program; hosted by U. Alberta CPRG | **10M+ complete hands** | Free for research | [poker.cs.ualberta.ca/irc_poker_database.html](http://poker.cs.ualberta.ca/irc_poker_database.html) · [extraction scripts](https://github.com/allenfrostline/PokerHandsDataset) |
| **ACPC match logs** (Annual Computer Poker Competition, 2009–2017) | Bot-vs-bot logs from the field's standard competition | ~341.2M fixed-limit + ~278.8M no-limit hold'em hands | Original site **down** (HTTP 503 on 2026-07-03); logs survive converted inside the PHH Zenodo archive below | [phh-dataset](https://github.com/uoftcprg/phh-dataset) (see §2) |

> Caveats: the UCI set is *synthetic classification*, not gameplay; IRC hands are play-money and 25+ years old (population differs radically from modern games); ACPC logs are bot-vs-bot, not human play.

## 2. The Modern Standard: PHH Format & Dataset (2024–2025)

| Item | Details | Link |
|---|---|---|
| **PHH file format** — "Recording and Describing Poker Hands" | Juho Kim, **IEEE Conference on Games (CoG) 2024**, Milan; concise human-readable, machine-friendly hand-history standard covering 11 variants | [arXiv:2312.11753](https://arxiv.org/abs/2312.11753) |
| **uoftcprg/phh-dataset** (GitHub) | Curated PHH-format collection, 11 variants: ACPC 2009–2017 logs, 21.6M real online NLHE hands (July 2009), **all 10,000 Pluribus hands**, 83 televised 2023 WSOP Event #43 final-table hands, historic hand selections; ACPC bulk is Zenodo-only (repo itself holds ~21.7M hands) | [github.com/uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset) (MIT) |
| **Full Zenodo archive** — "A Dataset of Poker Hand Histories" | **~640M hands total**, 20.3 GB download, **v3 published 2025-09-16**, CC BY 4.0 | [doi.org/10.5281/zenodo.10796885](https://doi.org/10.5281/zenodo.10796885) |
| **Pluribus hands** (source publication) | Brown & Sandholm, "Superhuman AI for multiplayer poker", *Science* 365:885–890 (2019) — the 10,000 6-max hands vs elite pros from its evaluation, republished in PHH format above | [DOI 10.1126/science.aay2400](https://www.science.org/doi/10.1126/science.aay2400) · [free PDF (NSF public access)](https://par.nsf.gov/servlets/purl/10119653) |

```bash
# Load PHH hands programmatically (PokerKit has native PHH reader/writer)
pip install pokerkit
python -c "from pokerkit import HandHistory; hh = HandHistory.load(open('hand.phh', 'rb'))"
```

## 3. Kaggle Datasets (each link live-checked)

| Dataset | Content | Link |
|---|---|---|
| UCI Poker Hand (mirror) | The 1M+ hand-classification benchmark on Kaggle | [rasvob/uci-poker-hand-dataset](https://www.kaggle.com/datasets/rasvob/uci-poker-hand-dataset) |
| Poker Hands Dataset | ~1M **script-generated** hold'em hands with board runouts and best-combination labels (synthetic, good for hand-strength models) | [joogollucci/poker-hands-dataset](https://www.kaggle.com/datasets/joogollucci/poker-hands-dataset) |
| Poker Hold'Em Games | Real hold'em game logs | [smeilz/poker-holdem-games](https://www.kaggle.com/datasets/smeilz/poker-holdem-games) |
| Online Poker Games | Real online hand histories, built with the author's open pipeline ([datasetbuilding](https://github.com/murilogmamaral/datasetbuilding), [pokerdf](https://github.com/murilogmamaral/pokerdf) — PokerStars text → Pandas/parquet) | [murilogmamaral/online-poker-games](https://www.kaggle.com/datasets/murilogmamaral/online-poker-games) |
| Poker Spin & Go (1€) | Real Spin & Go tournament hands | [bowaka/poker-spin-and-go-1-euro](https://www.kaggle.com/datasets/bowaka/poker-spin-and-go-1-euro) |
| **Heads Up Poker (Kaggle Game Arena)** | Official Kaggle **LLM benchmark**: frontier models play heads-up no-limit hold'em; leaderboard + gameplay data | [kaggle.com/benchmarks/kaggle/poker-heads-up](https://www.kaggle.com/benchmarks/kaggle/poker-heads-up) · [dataset](https://www.kaggle.com/datasets/kaggle/poker-heads-up) · [blog](https://www.kaggle.com/blog/game-arena-poker) |

More Kaggle poker data (classification, CV, behavior analytics): see the sibling page [Kaggle_Poker_Datasets.md](./Kaggle_Poker_Datasets.md).

## 4. Parsing, Simulation & Analysis Tools (Ferramentas)

| Tool | What it does | Status (verified 2026-07-03) | License / cost |
|---|---|---|---|
| [uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit) | Pure-Python game simulation, hand evaluation, statistical analysis; **native PHH reader/writer** ([docs](https://pokerkit.readthedocs.io/en/stable/notation.html)) | **Active** — v0.7.4 released May 2026 | MIT, free |
| [pokerregion/poker](https://github.com/pokerregion/poker) | Python framework: cards, ranges, hand-history parsing | **Archived read-only since 2025-02-01**; README warns the PokerStars parser is broken by a format change | MIT, free |
| [HHSmithy/PokerHandHistoryParser](https://github.com/HHSmithy/PokerHandHistoryParser) | C#/.NET hand-history parser for **13 sites** (PokerStars .com/.fr/.it/.es, Full Tilt, Party, 888, iPoker, Merge, OnGame, Microgaming, Winamax, Winning, Boss, Entraction) | Inactive/maintenance mode (no recent releases) | MIT, free |
| [PokerTracker 4](https://www.pokertracker.com/) | Industry-standard hand-history database + HUD + leak analysis (your own hands) | Actively sold; 14-day free trial | **Paid** |
| [SharkScope](https://www.sharkscope.com/) | Tournament-results statistics service ("99.9% of all online tournaments") | Live | **Freemium** |

> ⚖️ **Legal/ethical caveats:** hand histories contain other players' data — mass scraping or sharing them can violate site ToS and data-protection law (GDPR in the EU, **LGPD** in Brazil). Several sites ban SharkScope-style services or any third-party tracking outright (see the policy note in the warning section above). Use your own hands, the public research corpora above, or synthetic data.

## 5. Generate Your Own (Synthetic) Data — Zero Risk

The safest dataset is the one you simulate. Both frameworks below are the standard research route and require no real-money site at all:

| Framework | Poker environments | License | Link |
|---|---|---|---|
| **RLCard** | Leduc Hold'em, Limit Texas Hold'em, No-Limit Texas Hold'em (plus Blackjack, UNO, Dou Dizhu…) | MIT | [github.com/datamllab/rlcard](https://github.com/datamllab/rlcard) |
| **OpenSpiel** | Kuhn poker, Leduc poker, and full ACPC-based Texas Hold'em (`universal_poker`) — confirmed in [games list](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md); actively maintained | Apache-2.0 | [github.com/google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) |

```bash
pip install rlcard          # env.run() yields unlimited labeled trajectories
pip install open_spiel      # CFR on kuhn_poker/leduc_poker = the classic study path
```

## 6. Which Dataset for Which Task? (Qual base para qual tarefa?)

| Research task | Best starting dataset |
|---|---|
| Hand-strength classification (classic ML benchmark) | UCI Poker Hand |
| Opponent modeling on real human play | PHH Zenodo archive (July-2009 online NLHE) or IRC Database |
| Studying superhuman AI decisions | Pluribus 10,000 hands (in phh-dataset) |
| Bot-vs-bot strategy analysis | ACPC logs (via PHH Zenodo archive) |
| LLM strategic-reasoning evaluation | Kaggle Heads Up Poker benchmark |
| RL training at scale | Synthetic: RLCard / OpenSpiel self-play |

**Not included:** the "Mandine's Real Poker Hands (MRPH)" Kaggle dataset sometimes cited in older lists could not be re-verified as live and is omitted here.

## Related in AIForge
- [Poker AI Milestones & Research](./Poker_AI_Milestones_and_Research.md) · [Kaggle Poker Datasets](./Kaggle_Poker_Datasets.md) · [Game Theory & GTO Foundations](./Game_Theory_and_GTO_Foundations.md)
- Parent vertical: [`../`](../) (Gaming AI)

---

**Sources:** [UCI dataset 158](https://archive.ics.uci.edu/dataset/158/poker+hand) · [IRC Poker Database (CPRG)](http://poker.cs.ualberta.ca/irc_poker_database.html) · [arXiv:2312.11753](https://arxiv.org/abs/2312.11753) · [uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset) · [Zenodo 10.5281/zenodo.10796885](https://doi.org/10.5281/zenodo.10796885) · [Science 10.1126/science.aay2400](https://www.science.org/doi/10.1126/science.aay2400) · [NSF public-access PDF](https://par.nsf.gov/servlets/purl/10119653) · [uoftcprg/pokerkit](https://github.com/uoftcprg/pokerkit) · [PokerKit PHH docs](https://pokerkit.readthedocs.io/en/stable/notation.html) · [pokerregion/poker](https://github.com/pokerregion/poker) · [HHSmithy/PokerHandHistoryParser](https://github.com/HHSmithy/PokerHandHistoryParser) · [pokertracker.com](https://www.pokertracker.com/) · [sharkscope.com](https://www.sharkscope.com/) · [datamllab/rlcard](https://github.com/datamllab/rlcard) · [open_spiel games.md](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md) · [Kaggle poker-heads-up benchmark](https://www.kaggle.com/benchmarks/kaggle/poker-heads-up) · [Kaggle Game Arena poker blog](https://www.kaggle.com/blog/game-arena-poker) · [gambleaware.org](https://www.gambleaware.org/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [gamblingtherapy.org](https://www.gamblingtherapy.org/) · [cvv.org.br](https://cvv.org.br/) · [gov.br Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel)

**Keywords:** poker dataset, hand history, hand histories, PHH format, poker hand history dataset, UCI poker hand, IRC poker database, ACPC logs, Pluribus hands, PokerKit, hand history parser, RLCard, OpenSpiel, Kaggle poker, LLM poker benchmark, HUD policy, responsible gambling; base de dados de pôquer, histórico de mãos, conjunto de dados, análise de mãos, pôquer Texas hold'em, aprendizado de máquina, jogo responsável, autoexclusão, LGPD, rake, variância
