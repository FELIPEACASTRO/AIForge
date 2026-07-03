# Live Tournament Data & Results (Dados e Resultados de Torneios Presenciais)

Fact-checked reference on **live (in-person) poker tournament results databases, ranking systems and results aggregators** — where researchers, students and analysts find who cashed, for how much, and where. This is distinct from the online hand-history datasets covered on the *Datasets & Hand Histories* page: here the unit of record is a **tournament result** (player, event, buy-in, finish, payout), not a played hand. **Research, education and off-table study only.** Every site, figure, ranking rule, paper and number below was verified live in July 2026 against primary sources.

---

## Responsible Gambling First (Jogo Responsável)

> **Live tournaments are played for money and carry real financial risk. Fields are large, most entrants min-cash or bust, and the fee/rake (taxa) plus variance mean the pool as a whole loses money. The results databases below record the tiny top slice who cashed — survivorship bias is baked in. If gambling stops being fun, seek help — it is free and confidential (se o jogo deixou de ser diversão, procure ajuda gratuita e sigilosa):**
>
> - **Brazil — CVV (Centro de Valorização da Vida):** call **188**, free, 24h nationwide, plus chat/e-mail — [cvv.org.br](https://cvv.org.br/)
> - **Brazil — Jogo Responsável (Secretaria de Prêmios e Apostas / Ministério da Fazenda):** official player-protection guidance and self-exclusion rules (Portaria SPA/MF nº 1.231/2024; licensed sites use the `.bet.br` domain) — [gov.br/fazenda — Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel)
> - **UK — National Gambling Helpline (GamCare):** **0808 8020 133**, free, 24/7 — [gamcare.org.uk](https://www.gamcare.org.uk/)
> - **UK — BeGambleAware:** free confidential advice and self-assessment — [begambleaware.org](https://www.begambleaware.org/)
> - **International — Gambling Therapy (Gordon Moody charity):** free multilingual online support, **including Português (Brasil)** — [gamblingtherapy.org](https://www.gamblingtherapy.org/)
>
> These databases are for **study and record-keeping**, not for identifying "beatable" opponents or scouting live games. Real-time assistance (RTA) is banned by every reputable poker site and tour.

---

## Live Results vs. Hand-History Data — Know the Difference

| Dimension | **Live results DBs** (this page) | Online hand histories (*Datasets* page) |
|---|---|---|
| Unit of record | One tournament result per player (finish + payout) | One dealt hand (hole cards, actions, board) |
| Granularity | Coarse: outcome only, no in-hand decisions | Fine: every bet/check/fold |
| Typical use | Player earnings, field sizes, prize-pool economics, skill-vs-luck studies | Strategy modeling, CFR/RL training, exploit detection |
| Public reuse | Almost always **ToS-restricted / scraping-forbidden** | Some open corpora (PHH, IRC, ACPC); many licensed |
| Bias to watch | **Survivorship** (only cashes recorded), self-reported staking | Site/stakes selection, bot contamination |

> ICM (Independent Chip Model) and payout-structure math govern the *value* of these finishes — that theory is covered on the *Game Theory & GTO Foundations* page and is only pointed to here, not duplicated.

---

## The Hendon Mob Poker Database — the global live-results DB

[**thehendonmob.com**](https://www.thehendonmob.com/) (database front-end: [pokerdb.thehendonmob.com](https://pokerdb.thehendonmob.com/)) is the de-facto global registry of live tournament results, self-described as **"The Largest Live Poker Database."** It catalogues an enormous, continually growing archive of registered players, events and individual tournament results; its homepage displays live running totals that update as new festivals are reported.

It powers all-time money lists, per-country money lists, annual Player-of-the-Year races, festival/circuit result pages, and the Global Poker Index (GPI) rankings. Individual player pages ("Mob profiles") aggregate a player's lifetime live cashes; **MyPokerDiary** lets players claim and annotate their own results. Under GDPR, since 2018 European players can request account deletion ([PokerNews, 2018](https://www.pokernews.com/news/2018/09/european-law-forces-hendon-mob-offer-account-deletion-32010.htm)).

**Ownership:** The Hendon Mob and GPI are operated under Alex Dreyfus's **Mediarex** group of poker companies. Dreyfus's Malta-based **Zokay Entertainment** acquired GPI from **Pinnacle Entertainment** in **August 2012** ([PokerNews](https://www.pokernews.com/news/2012/08/pinnacle-entertainment-sells-global-poker-index-to-zokay-13192.htm)); **GPI then acquired The Hendon Mob in July 2013** ([Pokerfuse](https://pokerfuse.com/news/live-and-online/global-poker-index-acquires-the-hendon-mob-22-07/), [Flushdraw](https://www.flushdraw.net/news/global-poker-index-acquires-hendon-mob-site-database/)).

### Access & ToS reality — read before you touch it

> **Scraping The Hendon Mob is prohibited.** Its [Terms & Conditions](https://www.thehendonmob.com/terms_and_conditions.html) forbid trawling, downloading, reproducing or reusing the Database in whole or part; the data is protected by **UK/EU copyright and database rights**, and unauthorized data-taking "carries the risk of prosecution" for breach of those laws. There is **no documented public REST API** — the only sanctioned bulk channel is a private, licensed **Data Feed** obtained by contacting the operator directly. For research, treat it as a *look-up* source and cite it; do **not** build a scraper against it.

## Global Poker Index (GPI) — rankings & methodology

The [**Global Poker Index**](https://www.globalpokerindex.com/) ranks the world's live tournament players **weekly**, scoring cashes over a rolling **36-month window**. Its published methodology multiplies three factors per result:

| Factor | What it measures |
|---|---|
| **Finishing Factor** | How high a player finished relative to the field size (deeper run in a bigger field = higher), on a logarithmic scale |
| **Buy-In Factor** | Buy-in relative to a **$1,000 baseline**, on a **logarithmic** curve (skill gains diminish as buy-ins climb); buy-ins are treated as capped at **$20,000** for larger events |
| **Aging Factor** | Recency weighting across six-month periods; recent results weigh more |

**Result caps:** the best **5** results per six-month period count for the most recent 18 months, then the best **4** per six-month period for the prior 18 months. **Qualifying events** must have **≥ 32 entrants** and be open to the public ([GPI About](https://www.globalpokerindex.com/about/), [PokerNews explainer](https://www.pokernews.com/poker-players/global-poker-index-rankings/), [Wikipedia](https://en.wikipedia.org/wiki/Global_Poker_Index)). GPI also runs derivative races (Player of the Year, GPI 300). For skill-vs-luck and player-tracking studies, GPI is a ready-made, transparent skill proxy — but note it rewards **volume** and access to high buy-ins, not pure win-rate.

## WSOP.com — official World Series of Poker results

[**WSOP.com**](https://www.wsop.com/) is the authoritative primary source for World Series of Poker outcomes — bracelet events since 1970 are archived.

| Resource | URL |
|---|---|
| Past Tournaments archive (results & winners by year) | [wsop.com/past-tournaments](https://www.wsop.com/past-tournaments/) |
| All-Time Bracelet Winners standings | [wsop.com/player-standings/all-time-bracelets](https://www.wsop.com/player-standings/all-time-bracelets/) |
| Bracelet Legacy / history | [wsop.com/about/bracelet-legacy](https://www.wsop.com/about/bracelet-legacy/) |

Because the WSOP is the largest, longest-running live series, it is the anchor dataset for most academic poker-skill work (see below). Full payout tables per event are published on the official pages and mirrored on Hendon Mob.

## World Poker Tour (WPT) — official results & Champions Club

[**worldpokertour.com**](https://www.worldpokertour.com/) has run since 2002. Winners of WPT Main Tour stops (buy-in range roughly $3,500–$25,000) join the **WPT Champions Club** and are engraved on the **Mike Sexton WPT Champions Cup** (renamed July 21, 2020 in Sexton's honor).

| Resource | URL |
|---|---|
| Event results | [worldpokertour.com/event/results](https://www.worldpokertour.com/event/results) |
| Champions Club roster | [worldpokertour.com/players/champions-club](https://www.worldpokertour.com/players/champions-club) |
| WPT Prime champions | [worldpokertour.com/players/wpt-prime-champions](https://www.worldpokertour.com/players/wpt-prime-champions) |

Reference: [WPT Champions Club — Wikipedia](https://en.wikipedia.org/wiki/World_Poker_Tour_Champions_Club).

## European Poker Tour (EPT) / PokerStars Live

The [**European Poker Tour**](https://www.pokerstarslive.com/ept/) is PokerStars' flagship live circuit; results also flow to Hendon Mob's [EPT circuit page](https://pokerdb.thehendonmob.com/circuit.php?a=e&n=EPT). A verified recent benchmark of live-field scale:

- **EPT Barcelona 2025** — €5,300 Main Event: **2,045 entries**, **€9,918,250** prize pool; won by **Thomas Eychenne** for **€1,217,175** ([PokerNews payouts](https://www.pokernews.com/tours/ept/2025-pokerstars-ept-barcelona/5-300-main-event/payouts.htm)).

EPT stops (Barcelona, Monte Carlo, Prague, Paris, etc.) publish live reporting and payouts through the PokerStars Blog and PokerStars Live site.

## Triton Poker Series — high-roller results

[**triton-series.com**](https://triton-series.com/) (events portal: [tritonpokerseries.com](https://tritonpokerseries.com/en-US/events)) is the premier super-high-roller series, founded **2016**, with **minimum buy-ins of US$15,000** and frequent six-/seven-figure events. Per [Wikipedia](https://en.wikipedia.org/wiki/Triton_Poker_Series), across 2016–2026 Triton staged **300+ events awarding roughly US$1.88 billion** in prize money. Landmark record: the **Triton Million for Charity** (August 2019) carried a **£1,050,000 buy-in** and a **£54,000,000 prize pool**; **Bryn Kenney's £16,890,509** runner-up prize remains the largest single-tournament payout in poker history (unsurpassed as of 2026).

| Resource | URL |
|---|---|
| Official event results | [triton-series.com/event-results](https://triton-series.com/event-results/) |
| Hendon Mob Triton circuit | [pokerdb.thehendonmob.com — TRITON](https://pokerdb.thehendonmob.com/circuit.php?a=e&n=TRITON) |
| Triton Poker Plus (streams/stats app) | [tritonpoker.plus](https://tritonpoker.plus/) |

## PokerGO Tour (PGT) — points & results

The [**PokerGO Tour**](https://www.pgt.com/) tracks U.S. high-roller results and runs its own points leaderboard. Points are awarded for in-the-money finishes, **scaled to prize money and tiered by buy-in**: higher buy-in events award a smaller fraction of points per dollar cashed, and cashes **over $1,000,000** use a separate fixed points schedule. The exact per-tier point values are published on the official points page and are revised between seasons (the 2026 season, for example, added higher point rates for events with buy-ins of $5,000 or less).

Source: [pgt.com/points-system](https://www.pgt.com/points-system) (see also [pgt.com/points](https://www.pgt.com/points)). Leaderboard and reporting live at [pgt.com/leaderboard](https://www.pgt.com/leaderboard) and [pgt.com/live-reporting](https://www.pgt.com/live-reporting) — e.g., **Yuri Dzivielevski** led the 2026 PGT leaderboard with **1,504 points** as of the June 21, 2026 update.

## Brazil — BSOP results (Brasil)

The **Brazilian Series of Poker (BSOP)** is Latin America's largest live circuit; its festivals are fully catalogued on Hendon Mob's [BSOP circuit page](https://pokerdb.thehendonmob.com/circuit.php?a=e&n=BSOP), including stops in **São Paulo, Rio de Janeiro, Brasília, Foz do Iguaçu, Fortaleza**, plus **BSOP Millions** and **BSOP Winter Millions**. Live reporting and payout tables for Brazilian events therefore reach the same global registry researchers use for WSOP/EPT/WPT — making BSOP the natural entry point for **Brazil-specific live-poker analytics (análise de pôquer ao vivo no Brasil)**. For responsible-gambling context in Brazil, see the `.bet.br` licensing and Jogo Responsável links at the top of this page.

## Academic uses of live-tournament data

Live-results DBs (especially the WSOP) are the empirical backbone of the **poker skill-vs-luck** literature used in legal and economics debates:

| Study | Source | Data used |
|---|---|---|
| Levitt & Miles, *The Role of Skill Versus Luck in Poker: Evidence from the World Series of Poker* | NBER Working Paper **17023**, May 2011, DOI [10.3386/w17023](https://doi.org/10.3386/w17023); published in *Journal of Sports Economics*, DOI [10.1177/1527002512449471](https://doi.org/10.1177/1527002512449471) | All **57 events** of the **2010 WSOP** (~32,000 entrants); **720** pre-identified "highly skilled" players earned **+30.5% ROI** vs. **−15.6%** for the rest — cited here as reference, discussed on the *Behavioral Research* page, **not** duplicated |

The methodological template is reusable: take a public results registry (WSOP archive, Hendon Mob money lists, GPI/PGT points), pre-classify skill, and test whether prior skill predicts future ROI. **Caveat:** these datasets record only cashes (survivorship bias), rarely capture staking/backing arrangements, and any bulk reuse of Hendon Mob-style data must respect the ToS above — prefer official primary-source archives (WSOP.com, tour sites) for reproducible academic work.

## Aggregators & tooling — reality check

- **Official tour sites** (WSOP.com, worldpokertour.com, triton-series.com, pgt.com, pokerstarslive.com) are the correct **primary** sources — cite these first.
- **Hendon Mob / GPI** is the global aggregator but is **look-up only** (no public API; scraping forbidden; licensed Data Feed only).
- **News/coverage** (PokerNews tour pages, PokerStars Blog) provide payout tables and entry counts that corroborate the databases.
- There is **no free, sanctioned, machine-readable bulk feed** of the full global live-results corpus. Any project needing structured live data at scale should (a) license a Data Feed, (b) restrict to a single tour's official archive with permission, or (c) use the open online hand-history corpora on the *Datasets* page instead — which are actually redistributable.

---

**Sources:** [thehendonmob.com](https://www.thehendonmob.com/) · [pokerdb.thehendonmob.com](https://pokerdb.thehendonmob.com/) · [Hendon Mob Terms & Conditions](https://www.thehendonmob.com/terms_and_conditions.html) · [PokerNews — GDPR account deletion](https://www.pokernews.com/news/2018/09/european-law-forces-hendon-mob-offer-account-deletion-32010.htm) · [PokerNews — Pinnacle sells GPI to Zokay](https://www.pokernews.com/news/2012/08/pinnacle-entertainment-sells-global-poker-index-to-zokay-13192.htm) · [Pokerfuse — GPI acquires Hendon Mob](https://pokerfuse.com/news/live-and-online/global-poker-index-acquires-the-hendon-mob-22-07/) · [Flushdraw — GPI acquires Hendon Mob](https://www.flushdraw.net/news/global-poker-index-acquires-hendon-mob-site-database/) · [globalpokerindex.com/about](https://www.globalpokerindex.com/about/) · [PokerNews — GPI rankings](https://www.pokernews.com/poker-players/global-poker-index-rankings/) · [Global Poker Index — Wikipedia](https://en.wikipedia.org/wiki/Global_Poker_Index) · [wsop.com/past-tournaments](https://www.wsop.com/past-tournaments/) · [wsop.com — all-time bracelets](https://www.wsop.com/player-standings/all-time-bracelets/) · [worldpokertour.com/event/results](https://www.worldpokertour.com/event/results) · [worldpokertour.com — Champions Club](https://www.worldpokertour.com/players/champions-club) · [WPT Champions Club — Wikipedia](https://en.wikipedia.org/wiki/World_Poker_Tour_Champions_Club) · [pokerstarslive.com/ept](https://www.pokerstarslive.com/ept/) · [PokerNews — EPT Barcelona 2025 payouts](https://www.pokernews.com/tours/ept/2025-pokerstars-ept-barcelona/5-300-main-event/payouts.htm) · [Hendon Mob — EPT circuit](https://pokerdb.thehendonmob.com/circuit.php?a=e&n=EPT) · [triton-series.com/event-results](https://triton-series.com/event-results/) · [Triton Poker Series — Wikipedia](https://en.wikipedia.org/wiki/Triton_Poker_Series) · [Hendon Mob — Triton circuit](https://pokerdb.thehendonmob.com/circuit.php?a=e&n=TRITON) · [pgt.com/points-system](https://www.pgt.com/points-system) · [pgt.com/leaderboard](https://www.pgt.com/leaderboard) · [Hendon Mob — BSOP circuit](https://pokerdb.thehendonmob.com/circuit.php?a=e&n=BSOP) · [NBER w17023](https://doi.org/10.3386/w17023) · [Journal of Sports Economics — Levitt & Miles](https://doi.org/10.1177/1527002512449471) · [gov.br Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas/jogo-responsavel) · [cvv.org.br](https://cvv.org.br/) · [gamcare.org.uk](https://www.gamcare.org.uk/) · [begambleaware.org](https://www.begambleaware.org/) · [gamblingtherapy.org](https://www.gamblingtherapy.org/)

**Keywords:** live tournament results, Hendon Mob poker database, Global Poker Index, GPI rankings, GPI methodology, WSOP results, World Poker Tour results, WPT Champions Club, European Poker Tour, EPT, Triton Poker Series, PokerGO Tour, PGT leaderboard, BSOP results, poker earnings database, skill versus luck poker, live poker analytics, scraping ToS, database rights / resultados de torneios ao vivo, banco de dados Hendon Mob, ranking mundial de pôquer, resultados WSOP, Triton super high roller, BSOP resultados, análise de pôquer ao vivo, jogo responsável, apostas regulamentadas bet.br
