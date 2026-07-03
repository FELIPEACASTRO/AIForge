# Poker AI Milestones & Research

> Poker is the grand challenge of **imperfect-information games** (jogos de informação imperfeita): unlike chess or Go, players act without seeing the full state, so bluffing, deception and probabilistic reasoning must be handled mathematically. This page indexes the verified landmark systems (Polaris → Cepheus → DeepStack → Libratus → Pluribus → ReBeL → Student of Games → LLM agents), the core algorithm family (CFR and its descendants), the research groups behind them, and the Annual Computer Poker Competition — strictly as **research and education** material.

---

## ⚠️ Responsible Gambling — Read First (Jogo Responsável)

**This page is for AI research and education ONLY. It is not gambling advice, not a profit method, and not an endorsement of real-money play.**

- Poker played for money is **gambling with real financial risk**: variance (variância) is enormous, and after rake (taxa da casa) **most players lose money long-term**. Superhuman AIs described here required millions of dollars of compute — they are science milestones, not a personal edge.
- Using bots or real-time assistance software on poker sites **violates the terms of service of virtually every platform** and gets accounts banned and funds confiscated.
- **Online-poker legality varies by country** (a legalidade do poker online varia por país). 🇧🇷 In Brazil, fixed-odds betting is regulated by the Secretaria de Prêmios e Apostas (Ministério da Fazenda) — see the official [Jogo Responsável page (gov.br)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel), which explains the responsible-gambling rules (Portaria SPA/MF nº 1.231/2024), including the self-exclusion mechanisms (autoexclusão) that licensed platforms are required to offer. Check your local law before playing anywhere.

**If gambling is harming you or someone you know (free & confidential):**

| Resource | Coverage | Access | Cost |
|---|---|---|---|
| [GambleAware](https://www.gambleaware.org/) | 🇬🇧 UK (info in English) | Advice, tools, treatment referrals | Free |
| [GamCare — National Gambling Helpline](https://www.gamcare.org.uk/) | 🇬🇧 UK | 0808 8020 133, 24/7, chat/WhatsApp | Free |
| [Gambling Therapy (Gordon Moody)](https://www.gamblingtherapy.org/) | 🌍 Global — **has Português (Brasil)** | Online emotional support & groups | Free |
| [CVV — Centro de Valorização da Vida](https://cvv.org.br/) | 🇧🇷 Brazil (emotional crisis support) | **Ligue 188**, 24h, todos os dias | Free |
| [Jogo Responsável — gov.br](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) | 🇧🇷 Brazil | Official guidance + platform self-exclusion rules | Free |

---

## 1. Why Poker Is an AI Grand Challenge

Perfect-information techniques (minimax, MCTS, AlphaZero-style self-play) break down when hidden information exists: the value of a state depends on *beliefs* (distributions over hidden cards), and optimal play requires **mixed strategies** — bluffing at mathematically precise frequencies. The field's core solution concept is the **Nash equilibrium** (equilíbrio de Nash), approximated at scale by **Counterfactual Regret Minimization (CFR)** and, since 2017, by **depth-limited search with learned value functions**. Heads-up limit hold'em has ~10¹⁴ decision points; heads-up no-limit has ~10¹⁶¹ — hence the central roles of **abstraction** (abstração), **subgame solving** (resolução de subjogos) and **exploitability** (explorabilidade) as the standard distance-to-equilibrium metric.

## 2. Landmark Systems (Marcos Históricos)

All entries below verified against live paper/lab pages. HULHE = heads-up limit hold'em; HUNL = heads-up no-limit.

| System | Year | Group | Achievement | Paper / link |
|---|---|---|---|---|
| **Polaris** | 2007–2008 | U. Alberta CPRG | First Man-Machine Poker matches; **beat human pros in the 2008 Las Vegas rematch** (HULHE, duplicate format) | [CPRG Man-vs-Machine page](https://poker.cs.ualberta.ca/man-machine/) |
| **Cepheus** | 2015 | U. Alberta CPRG | **HULHE "essentially weakly solved"** — first competitively played imperfect-info game solved; proved the dealer's advantage; introduced CFR+ | Bowling, Burch, Johanson & Tammelin, *Science* 347:145–149, [DOI 10.1126/science.1259433](https://www.science.org/doi/10.1126/science.1259433) · [free author PDF](http://johanson.ca/publications/poker/2015-science-hulhe/2015-science-hulhe.html) |
| **DeepStack** | 2017 | U. Alberta CPRG + Charles U. + CTU Prague | First AI to beat professional players in **HUNL** (44,000 hands); continual re-solving + deep-learned "intuition" value nets | Moravčík et al., *Science* (2017), [arXiv:1701.01724](https://arxiv.org/abs/1701.01724) |
| **Libratus** | 2017/2018 | CMU (Brown & Sandholm) | **Decisively beat 4 top HUNL pros over 120,000 hands** ("Brains vs. AI"); blueprint strategy + nested safe subgame solving + self-improvement | *Science* 359:418–424, [DOI 10.1126/science.aao1733](https://www.science.org/doi/10.1126/science.aao1733) |
| **Pluribus** | 2019 | CMU + Facebook AI (Brown & Sandholm) | **First superhuman multiplayer (6-max) no-limit hold'em AI**; beat elite pros over 10,000 hands at remarkably low compute cost | *Science* 365:885–890, [DOI 10.1126/science.aay2400](https://www.science.org/doi/10.1126/science.aay2400) |
| **ReBeL** | 2020 | Facebook/Meta AI | General **RL + search** framework for imperfect-info games; provably converges to Nash in 2p0s games; superhuman HUNL with minimal domain knowledge | [arXiv:2007.13544](https://arxiv.org/abs/2007.13544) · [code (Liar's Dice release)](https://github.com/facebookresearch/rebel) |
| **Student of Games** (announced as *Player of Games*) | 2021→2023 | DeepMind / EquiLibre | **One unified algorithm** (guided search + self-play + game-theoretic reasoning) strong at chess, Go, poker and Scotland Yard; published in *Science Advances* 9, eadg3256 (2023) | [arXiv:2112.03178](https://arxiv.org/abs/2112.03178) |

## 3. Core Algorithms & Theory Papers (Algoritmos Fundamentais)

| Algorithm / idea | Paper | Venue / year | Link | Why it matters |
|---|---|---|---|---|
| **CFR** — Counterfactual Regret Minimization | Zinkevich, Johanson, Bowling & Piccione, "Regret Minimization in Games with Incomplete Information" | NIPS 2007 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html) | Foundation of essentially all modern poker AI; iterative self-play converging to Nash |
| **MCCFR** — Monte Carlo CFR | Lanctot, Waugh, Zinkevich & Bowling, "Monte Carlo Sampling for Regret Minimization in Extensive Games" | NIPS 2009 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html) | Sampling variants that scale CFR to huge games |
| **CFR+** | Tammelin, "Solving Large Imperfect Information Games Using CFR+" | 2014 | [arXiv:1407.5042](https://arxiv.org/abs/1407.5042) | Order-of-magnitude speedup; the algorithm behind Cepheus |
| **Discounted CFR (DCFR)** | Brown & Sandholm, "Solving Imperfect-Information Games via Discounted Regret Minimization" | 2018/2019 | [arXiv:1809.04040](https://arxiv.org/abs/1809.04040) | Discounts early bad iterations; beat CFR+ across tested games |
| **Deep CFR** | Brown, Lerer, Gross & Sandholm, "Deep Counterfactual Regret Minimization" | ICML 2019 | [arXiv:1811.00164](https://arxiv.org/abs/1811.00164) | Neural nets replace hand-crafted abstraction — CFR in the full game |
| **Safe & nested subgame solving** | Brown & Sandholm, "Safe and Nested Subgame Solving for Imperfect-Information Games" | 2017 | [arXiv:1705.02955](https://arxiv.org/abs/1705.02955) | Real-time endgame refinement without becoming exploitable; core of Libratus |
| **RL + search unification** | Brown, Bakhtin, Lerer & Gong (ReBeL) | 2020 | [arXiv:2007.13544](https://arxiv.org/abs/2007.13544) | Extends AlphaZero-style self-play+search to imperfect information via public belief states |

Related standard concepts you will meet in these papers: **abstraction** (grouping strategically similar hands/bets to shrink the game), **blueprint strategy** (coarse offline equilibrium later refined online), **exploitability / mbb·hand⁻¹** (milli-big-blinds per hand, the standard win-rate and distance-to-Nash unit), and **AIVAT** (variance-reduction for evaluating agents — used e.g. by the GTO Wizard leaderboard below).

## 4. LLMs × Poker (2024–2026)

A new wave asks: can general language models reason strategically under uncertainty? Current verified answer: **out of the box they play far below equilibrium**, but fine-tuning and structured scaffolds close part of the gap.

| Work | Year | Type | Link | Verified takeaway |
|---|---|---|---|---|
| **PokerGPT** | 2024 | LLM agent (fine-tuned, RLHF-style) | [arXiv:2401.06781](https://arxiv.org/abs/2401.06781) | Lightweight end-to-end multi-player hold'em solver built on an LLM from textual game records |
| **PokerBench** | 2025 | Benchmark (11,000 pre/post-flop spots) | [arXiv:2501.08328](https://arxiv.org/abs/2501.08328) | GPT-4/Llama/Gemma-class models underperform optimal play; fine-tuning helps; scores correlate with real win-rates |
| **GTO Wizard AI Poker Leaderboard** | 2025–2026 | Live benchmark (AIVAT-evaluated, luck-adjusted, real-time) | [gtowizard.com/benchmark](https://gtowizard.com/benchmark) · [arXiv:2603.23660](https://arxiv.org/abs/2603.23660) | Frontier LLMs (GPT, Claude, Gemini, Grok, Kimi…) play HUNL vs GTO Wizard AI and still underperform it — **free to view**; model submission by API-key approval |
| **PokerSkill** | 2026 | LLM agent (rule-structured skills, no training/solver) | [arXiv:2605.30094](https://arxiv.org/abs/2605.30094) | Structured poker-skill scaffolding cuts LLM losses by 49–61% vs default-prompt baselines — expert-level play without training or solver access |
| **Poker Arena** | 2026 | Evaluation platform (9-axis cognitive profile) | [arXiv:2606.13815](https://arxiv.org/abs/2606.13815) | Tournament rank ≠ capability profile; multi-axis evaluation surfaces structure that scalar leaderboards misrank |

## 5. Research Groups (Grupos de Pesquisa)

| Group | Institution | Signature systems | Link |
|---|---|---|---|
| **Computer Poker Research Group (CPRG)** | University of Alberta 🇨🇦 | Loki, Poki, Hyperborean, **Polaris, Cepheus, DeepStack** | [poker.cs.ualberta.ca](https://poker.cs.ualberta.ca/) |
| **Sandholm lab / Noam Brown** | Carnegie Mellon University 🇺🇸 | **Libratus, Pluribus**, subgame-solving & DCFR theory | Papers above (Science 2018/2019, ICML 2019) |
| **Meta (Facebook) AI Research** | Meta 🇺🇸 | **ReBeL**, Pluribus co-development | [github.com/facebookresearch/rebel](https://github.com/facebookresearch/rebel) |
| **DeepMind** | Google DeepMind 🇬🇧 | **Student of Games / Player of Games**, OpenSpiel | [github.com/google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) |

## 6. Annual Computer Poker Competition (ACPC)

The **ACPC** ran from **2006 to 2018**, typically co-located with AAAI/IJCAI, and was the field's standard proving ground: server-based matches between submitted agents, beginning with heads-up limit hold'em in 2006, later adding heads-up no-limit and 3-player events, and closing in 2018 with heads-up no-limit hold'em plus the competition's first 6-player no-limit event. Its benchmark role has since been partially taken over by open frameworks (OpenSpiel) and live leaderboards (GTO Wizard benchmark). References: [Bard, Hawkin, Rubin & Zinkevich, "The Annual Computer Poker Competition", *AI Magazine* 34(2), 2013](https://onlinelibrary.wiley.com/doi/abs/10.1609/aimag.v34i2.2474) · [IEEE Spectrum coverage](https://spectrum.ieee.org/a-texas-hold-em-tournament-for-ais) · [CPRG news archive](https://poker.cs.ualberta.ca/news.html). (The original competition site, computerpokercompetition.org, was intermittently offline as of mid-2026.)

## 7. Open-Source Code for Study (Código Aberto)

| Repo | What it gives you | License | Cost |
|---|---|---|---|
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) | C++/Python framework; CFR family implementations; Kuhn & Leduc poker environments — the standard classroom entry point; actively maintained (v1.6.x releases in 2026) | Apache-2.0 | Free |
| [facebookresearch/rebel](https://github.com/facebookresearch/rebel) | Official ReBeL implementation (released for Liar's Dice) + pretrained checkpoints; **archived read-only since 1 Nov 2024** | Apache-2.0 | Free |
| [CFR+ solver code (Cepheus authors)](http://johanson.ca/publications/poker/2015-science-hulhe/2015-science-hulhe.html) | Author page for the Science 2015 paper with accepted-version PDF, supplement and source-code link | — | Free |

**Suggested study path (trilha de estudo):** Kuhn poker by hand → CFR (NIPS 2007) on Kuhn/Leduc in OpenSpiel → MCCFR & CFR+ → Cepheus paper → subgame solving (arXiv:1705.02955) → DeepStack/Libratus Science papers → Deep CFR → ReBeL → Student of Games → the 2024–2026 LLM wave.

## 8. How Poker AIs Are Evaluated (Metodologia de Avaliação)

Poker results are noisy, so landmark claims rest on careful statistics rather than raw session results:

- **mbb/hand (milli-big-blinds per hand)** — the standard win-rate unit; also used to express *exploitability*, the distance of a strategy from Nash equilibrium. Cepheus's "essentially weakly solved" claim means its exploitability is below what a lifetime of human play could statistically detect ([Science 2015](https://www.science.org/doi/10.1126/science.1259433)).
- **Duplicate poker (formato duplicado)** — the same cards are dealt to opposite sides in mirrored matches to cancel card luck; used in the Polaris Man-vs-Machine matches ([CPRG](https://poker.cs.ualberta.ca/man-machine/)).
- **Variance-reduction estimators (AIVAT)** — statistical techniques that produce luck-adjusted win rates from far fewer hands; used by DeepStack's evaluation lineage and today by the [GTO Wizard AI leaderboard](https://gtowizard.com/benchmark).
- **Sample sizes** — DeepStack: 44,000 hands vs pros ([arXiv:1701.01724](https://arxiv.org/abs/1701.01724)); Libratus: 120,000 hands vs 4 top pros ([Science 2018](https://www.science.org/doi/10.1126/science.aao1733)); Pluribus: 10,000 hands in 6-max vs elite pros ([Science 2019](https://www.science.org/doi/10.1126/science.aay2400)).

## 9. Mini-Glossary (Mini-Glossário EN → PT)

| English term | Português | Meaning |
|---|---|---|
| Imperfect-information game | Jogo de informação imperfeita | Players cannot observe the full game state (hidden cards) |
| Nash equilibrium | Equilíbrio de Nash | Strategy profile where no player gains by deviating unilaterally |
| Mixed strategy | Estratégia mista | Randomizing between actions at fixed frequencies (mathematical bluffing) |
| Regret | Arrependimento | How much better an alternative action would have done in hindsight; CFR minimizes it |
| Abstraction | Abstração | Merging strategically similar hands/bet sizes to shrink the game tree |
| Blueprint strategy | Estratégia-mapa (blueprint) | Coarse offline equilibrium, refined online by subgame solving |
| Subgame solving | Resolução de subjogos | Re-solving the remainder of the hand in real time without losing safety guarantees |
| Exploitability | Explorabilidade | Worst-case loss vs a perfect counter-strategy; 0 at Nash equilibrium |
| Heads-up / 6-max | Um-contra-um / mesa de seis | Two-player format / six-player format |
| Rake | Rake (taxa da casa) | The house fee that makes most real-money players net losers |

## 10. FAQ

**Is poker "solved"?** Only **heads-up *limit*** hold'em is *essentially weakly solved* (Cepheus, 2015). Heads-up **no-limit** and multiplayer games are **not solved** — Libratus and Pluribus are *superhuman*, which is a weaker claim than solved, and for 6+ players Nash equilibrium itself loses its safety guarantees (discussed in the [Pluribus paper](https://www.science.org/doi/10.1126/science.aay2400)).

**Can I run these systems on real poker sites?** No. Beyond the legal risk, bots and real-time assistance violate essentially all platforms' terms of service; this index exists for reproducing *research*, e.g. CFR on Kuhn/Leduc poker in [OpenSpiel](https://github.com/google-deepmind/open_spiel).

**Do LLMs play good poker (2026)?** Verified benchmarks ([PokerBench](https://arxiv.org/abs/2501.08328), [GTO Wizard leaderboard](https://gtowizard.com/benchmark), [PokerSkill](https://arxiv.org/abs/2605.30094)) show frontier LLMs still lose to solver-based GTO play, though fine-tuning and structured skill scaffolds cut the losses substantially. LLM-poker is currently a *reasoning benchmark*, not a playing-strength frontier.

**Why does this research matter outside poker?** The same machinery — belief states, regret minimization, search with learned values — generalizes to negotiation, security, auctions and any strategic setting with hidden information; that generalization is the explicit thesis of [ReBeL](https://arxiv.org/abs/2007.13544) and [Student of Games](https://arxiv.org/abs/2112.03178).

## 11. Brazil Note (Cenário Brasileiro) 🇧🇷

Brazil has one of the world's largest and most active poker communities, and Brazilian audiences are a major consumer of GTO (game-theory-optimal / teoria dos jogos) study content. For AI researchers this means abundant Portuguese-language study demand — but the same responsible-gambling rules apply: study the math for free with open-source tools; treat real-money play as regulated gambling under the Secretaria de Prêmios e Apostas rules, use the [official jogo-responsável guidance](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) including the self-exclusion tools licensed platforms must offer, and seek help early ([Gambling Therapy em português](https://www.gamblingtherapy.org/), [CVV 188](https://cvv.org.br/)).

---

**Sources:** [Science 10.1126/science.1259433](https://www.science.org/doi/10.1126/science.1259433) · [Science 10.1126/science.aao1733](https://www.science.org/doi/10.1126/science.aao1733) · [Science 10.1126/science.aay2400](https://www.science.org/doi/10.1126/science.aay2400) · [arXiv:1701.01724](https://arxiv.org/abs/1701.01724) · [arXiv:2007.13544](https://arxiv.org/abs/2007.13544) · [arXiv:2112.03178](https://arxiv.org/abs/2112.03178) · [NIPS 2007 CFR](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html) · [NIPS 2009 MCCFR](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html) · [arXiv:1407.5042](https://arxiv.org/abs/1407.5042) · [arXiv:1809.04040](https://arxiv.org/abs/1809.04040) · [arXiv:1811.00164](https://arxiv.org/abs/1811.00164) · [arXiv:1705.02955](https://arxiv.org/abs/1705.02955) · [arXiv:2401.06781](https://arxiv.org/abs/2401.06781) · [arXiv:2501.08328](https://arxiv.org/abs/2501.08328) · [arXiv:2603.23660](https://arxiv.org/abs/2603.23660) · [arXiv:2605.30094](https://arxiv.org/abs/2605.30094) · [arXiv:2606.13815](https://arxiv.org/abs/2606.13815) · [poker.cs.ualberta.ca](https://poker.cs.ualberta.ca/) · [CPRG Man-vs-Machine](https://poker.cs.ualberta.ca/man-machine/) · [CPRG news archive](https://poker.cs.ualberta.ca/news.html) · [johanson.ca Cepheus page](http://johanson.ca/publications/poker/2015-science-hulhe/2015-science-hulhe.html) · [gtowizard.com/benchmark](https://gtowizard.com/benchmark) · [open_spiel](https://github.com/google-deepmind/open_spiel) · [rebel](https://github.com/facebookresearch/rebel) · [ACPC — AI Magazine 34(2) 2013](https://onlinelibrary.wiley.com/doi/abs/10.1609/aimag.v34i2.2474) · [IEEE Spectrum](https://spectrum.ieee.org/a-texas-hold-em-tournament-for-ais) · [gambleaware.org](https://www.gambleaware.org/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [gamblingtherapy.org](https://www.gamblingtherapy.org/) · [cvv.org.br](https://cvv.org.br/) · [gov.br Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel)

**Keywords:** poker AI, computer poker, imperfect-information games, game theory, Nash equilibrium, counterfactual regret minimization, CFR, CFR+, MCCFR, Deep CFR, discounted CFR, subgame solving, abstraction, exploitability, Cepheus, DeepStack, Libratus, Pluribus, ReBeL, Player of Games, Student of Games, Annual Computer Poker Competition, ACPC, University of Alberta CPRG, CMU, Noam Brown, Tuomas Sandholm, Michael Bowling, LLM poker, PokerBench, PokerGPT, GTO, responsible gambling, no-limit hold'em; IA de pôquer, jogos de informação imperfeita, teoria dos jogos, equilíbrio de Nash, minimização de arrependimento contrafactual, resolução de subjogos, explorabilidade, pôquer Texas hold'em, jogo responsável, apostas, autoexclusão, variância, rake
