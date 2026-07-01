# Community, Forums, Newsletters & Open-Source Projects

> Authoritative map of **where the football (soccer / futebol) modelling community actually lives** — the forums (fóruns), Reddit communities (comunidades), Q&A sites, analytics newsletters (boletins) & blogs, podcasts, YouTube channels, notable **open-source (código aberto) prediction/data projects on GitHub** (each verified to exist, with star counts), and the core books. This is the "people & projects" page for the section; the *how* (models, data, odds, staking) lives on the sibling pages. **Research & education only (pesquisa e educação — data science / ML), current 2024–2026.**

> ⚠️ **Research & education only — NOT betting advice (NÃO é dica de aposta).** Football betting markets are **highly efficient**: sharp books (e.g. Pinnacle) and deep exchanges price near the true probability and the **closing line** is very hard to beat. **Only a small minority of bettors are profitable long-term; the large majority lose money.** No forum tip, newsletter, Discord, or GitHub repo is a money machine — treat every "system" with deep skepticism (survivorship bias, cherry-picked records, paid-tout scams are rampant). If gambling stops being fun or you cannot stop, get help now: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) (multilingual) · [Gamblers Anonymous](https://www.gamblersanonymous.org/) · 🇧🇷 [Jogo Responsável — SPA/MF](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188** · [cvv.org.br](https://www.cvv.org.br/).

**Sibling pages (do not duplicate):** models → [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · data/APIs → [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Datasets](./Kaggle_Football_Datasets_and_Competitions.md) · core libraries → [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · odds/value/**CLV**/staking → [Bet Selection, Staking & High-Odds Analysis](./Bet_Selection_Staking_and_High_Odds_Analysis.md) · exchanges/trading → [Betting Exchanges, Trading & Microstructure](./Betting_Exchanges_Trading_and_Microstructure.md) · regions → [Europe](./Europe_Football_Data_and_Markets.md) · [Asia & India](./Asia_and_India_Football_Data_and_Markets.md) · [Africa, Middle East & Oceania](./Africa_MiddleEast_and_Oceania_Football_Data.md).

---

## 0) The landscape in one picture (o mapa da comunidade)

Football modelling has **three overlapping communities** that rarely talk to each other — know which one a resource belongs to:

| Community | Focus | Culture | Watch out for |
|---|---|---|---|
| **Football analytics (analítica)** | xG, tracking, player valuation, tactics — mostly *club/media*, not betting | Open, academic, publishes code | Rarely about beating the market |
| **Algorithmic / quant betting (aposta algorítmica)** | Beating the closing line, CLV, staking, automation | Secretive, evidence-driven, sceptical | Real edges are hidden; public "tips" are noise |
| **Fantasy / FPL (fantasy)** | Points optimization, not odds | Huge, friendly, tool-rich | Not a betting edge |

> **Golden rule of every betting forum:** anyone *selling* picks, a "guaranteed" system, or a Telegram/WhatsApp VIP group is almost certainly a **tout/scam (charlatão)**. Genuine edges are not sold retail. The useful communities discuss **method and back-testing**, not tips.

---

## 1) Reddit communities (comunidades no Reddit)

Reddit hosts the most active *public* discussion. Member counts are approximate and change constantly (public-source estimates, mid-2026, cross-checked via [GummySearch](https://gummysearch.com/) subreddit stats). All free.

| Subreddit | ~Members | What it's for | Link |
|---|---:|---|---|
| **r/algobetting** | ~24k | The serious one: sports **modelling, statistics, programming, automation**, back-testing, constructive critique. Best signal-to-noise for quants. | [r/algobetting](https://www.reddit.com/r/algobetting/) · [stats](https://gummysearch.com/r/algobetting/) |
| **r/footballanalytics** | ~90k | xG, event/tracking data, viz, model-building; club/media analytics crowd (not betting). | [r/footballanalytics](https://www.reddit.com/r/footballanalytics/) |
| **r/SoccerBetting** | ~140k | Football-specific betting: daily picks threads, discussion, some model talk. High tip-noise. | [r/SoccerBetting](https://www.reddit.com/r/SoccerBetting/) · [stats](https://gummysearch.com/r/SoccerBetting/) |
| **r/sportsbook** | ~600k | Largest US-centric sportsbook community; sticky daily threads, book/limits talk. | [r/sportsbook](https://www.reddit.com/r/sportsbook/) |
| **r/sportsbetting** | ~550k | General sports betting; strategy, book reviews, RG discussion. | [r/sportsbetting](https://www.reddit.com/r/sportsbetting/) |
| **r/FantasyPL** | ~500k+ | Fantasy Premier League — transfers, captaincy, model/tool sharing (see FPL repos §8). | [r/FantasyPL](https://www.reddit.com/r/FantasyPL/) |
| **r/dfsports** | ~60k | Daily Fantasy Sports optimization / lineup modelling. | [r/dfsports](https://www.reddit.com/r/dfsports/) |

> Reddit subreddit names are **case-insensitive** (`r/soccerbetting` ≡ `r/SoccerBetting`). Start at **r/algobetting** for method; treat picks-threads elsewhere as entertainment.

---

## 2) Forums & Discords (fóruns e Discords)

| Venue | Type | Focus | Free/Paid | Link |
|---|---|---|---|---|
| **Bet Angel Forum** | Forum | The busiest **Betfair exchange trading (negociação)** forum — hundreds of thousands of posts; scalping, automation, Guardian/Bet Angel tooling | Free | [forum.betangel.com](https://forum.betangel.com/) |
| **Betfair Community** (official) | Forum | Official Betfair boards by sport (Football, Horse Racing, Tennis…) | Free | [community.betfair.com](https://community.betfair.com/) |
| **Geeks Toy Forum** | Forum | Betfair trading software + strategy; long-standing trader crowd | Free | [geekstoy.com/forum](https://www.geekstoy.com/forum/) |
| **Betfair Trading Community (BTC)** | Membership + forum | Structured trading education/mentoring across football, racing, tennis | Paid (membership; price shown at signup) | [betfairtradingcommunity.com](https://betfairtradingcommunity.com/) |
| **r/algobetting Discord** (linked from sub) | Discord | Live chat for the modelling crowd — code, data, APIs | Free | via [r/algobetting](https://www.reddit.com/r/algobetting/) sidebar |
| **Twelve / Soccermatics course Discord** | Discord | Learners of David Sumpter's course (see §9) | Free w/ course | [twelve.football/courses](https://twelve.football/courses) |

> ⚠️ Many "private VIP" Discords/Telegram channels charge for "sure bets" — this is the **tout economy**. Anything with a subscription for *picks* (not education/tooling) should be assumed to be selling variance and survivorship bias.

---

## 3) Q&A / Stack Exchange (perguntas e respostas)

Rigorous, citable answers on the maths — better than forums for *why* a method works.

| Site | Use it for | Link |
|---|---|---|
| **Cross Validated** (Stats SE) | Poisson/Dixon-Coles, Elo, calibration, Brier/log-loss, betting-EV probability questions | [stats.stackexchange.com](https://stats.stackexchange.com/) |
| **Sports Stack Exchange** | Rules, data sources, sport-specific modelling questions | [sports.stackexchange.com](https://sports.stackexchange.com/) |
| **Quantitative Finance SE** | Kelly criterion, bankroll as portfolio, risk of ruin (transfers cleanly to staking) | [quant.stackexchange.com](https://quant.stackexchange.com/) |
| **Mathematics / Math Overflow** | Probability derivations underpinning models | [math.stackexchange.com](https://math.stackexchange.com/) |
| **Cross Validated `[sports]` / `[poisson-distribution]` tags** | Directly relevant threads | [tag: sports](https://stats.stackexchange.com/questions/tagged/sports) |

---

## 4) Analytics newsletters & blogs (boletins e blogs de analítica)

The public "football analytics" writing scene — mostly *not* betting, but the source of the ideas (xG, xT, VAEP) that feed prediction models.

| Resource | Author/Org | What you get | Free/Paid | Link |
|---|---|---|---|---|
| **StatsBomb Blog / Articles** | StatsBomb (Hudl) | Metrics & explainers, set-piece/360 analysis; huge public archive (the main blog now lives on Hudl since the StatsBomb→Hudl acquisition) | Free | [StatsBomb @ Hudl](https://www.hudl.com/blog/elite/statsbomb) · [blog archive 2013–2025](https://blogarchive.statsbomb.com/) |
| **Opta Analyst ("The Analyst")** | Opta / Stats Perform | Data storytelling + the **Opta supercomputer** predictions; weekly *Stat, Viz, Quiz* newsletter | Free | [theanalyst.com](https://theanalyst.com/) · [newsletter](https://theanalyst.com/sign-up) · [predictions](https://theanalyst.com/articles/opta-football-predictions) |
| **American Soccer Analysis** | ASA | MLS/NWSL analytics with **Goals Added (g+)**, interactive tools, the *American Soccer Analysis* podcast | Free | [americansocceranalysis.com](https://www.americansocceranalysis.com/) |
| **Karun Singh** | Karun Singh | Home of **Expected Threat (xT)**; interactive explainers | Free | [karun.in](https://karun.in/) · [xT post](https://karun.in/blog/expected-threat.html) |
| **John Muller** — *space space space* | John Muller (ex-Athletic) | Tactics+analytics essays, "How Football Works"; now building *Futi* | Free (Substack) | [johnspacemuller.com](https://johnspacemuller.com/) · [Substack](https://johnspacemuller.substack.com/) |
| **Tony's Blog** | Tony ElHabr | **R** how-tos: xG, meta-analytics, worldfootballR/socceraction workflows | Free | [tonyelhabr.rbind.io](https://tonyelhabr.rbind.io/) · [GitHub](https://github.com/tonyelhabr) |
| **Soccermatics (Medium)** | David Sumpter | Maths-of-football essays; explainers of xG/xT | Free | [soccermatics.medium.com](https://soccermatics.medium.com/) · [david-sumpter.com](https://www.david-sumpter.com/) |
| **Get Goalside** | Get Goalside | Football-analytics blog + "handy resources" list | Free | [getgoalsideanalytics.com](https://www.getgoalsideanalytics.com/) |
| **Twenty First Group** (ex-21st Club) | Twenty First Group | B2B sports-intelligence insight pieces (player/team value models) | Free (blog) | [twentyfirstgroup.com](https://www.twentyfirstgroup.com/) |
| **R-bloggers** (aggregator) | community | Aggregated R posts incl. lots of soccer/xG modelling | Free | [r-bloggers.com](https://www.r-bloggers.com/) |
| **The xG Philosophy** (X/Twitter) | @xGPhilosophy | Post-match xG timelines for most games; huge reach | Free | [x.com/xGPhilosophy](https://x.com/xGPhilosophy) |

---

## 5) Betting-model blogs & educational resources (educação em apostas)

The *market-facing* writing — efficiency, margin removal, CLV, staking. This is where "can a model actually beat the book?" is treated honestly.

| Resource | Author/Org | Signature content | Free/Paid | Link |
|---|---|---|---|---|
| **Football-Data.co.uk** | Joseph Buchdahl | Free historical CSVs **and** the essays: *The Wisdom of the Crowd* (margin-free "true" odds), book list | Free | [football-data.co.uk](https://www.football-data.co.uk/) · [WoC PDF](https://www.football-data.co.uk/The_Wisdom_of_the_Crowd_updated.pdf) · [WoC bet tracker](https://www.football-data.co.uk/wisdom_of_crowd_bets.php) |
| **Pinnacle "Betting Resources"** | Pinnacle + guest authors | Deep, non-salesy educational library (Poisson, xG-for-betting, CLV, Kelly, Bayes-factor skill tests) | Free | [pinnacle.com/betting-resources](https://www.pinnacle.com/betting-resources/en/category/educational) · [Buchdahl author page](https://www.pinnacle.com/betting-resources/en/author/joseph-buchdahl) |
| **Football-Data ↔ Pinnacle article index** | Buchdahl | Page collecting his Pinnacle articles (samples + link to his full Pinnacle author page) | Free | [pinnaclesports_articles.php](https://www.football-data.co.uk/blog/pinnaclesports_articles.php) |
| **BettingIsCool** | "BettingIsCool" | Research posts + a Streamlit app demoing the *Wisdom of the Crowd* strategy | Free | [bettingiscool.com/category/research](https://bettingiscool.com/category/research/) |
| **Caan Berry** | Caan Berry | Betfair *trading* education (exchange, not tipping) | Free/Paid | [caanberry.com](https://caanberry.com/) |

> The single most valuable lesson from this literature: **beating the no-vig closing line (CLV) is the only durable evidence of skill.** If a resource never mentions closing-line value, margin removal, or sample size, it is entertainment, not analysis. See the [Bet Selection & Staking](./Bet_Selection_Staking_and_High_Odds_Analysis.md) page.
>
> Note: Football-Data's *Wisdom of the Crowd* live bet tracker stopped being updated in mid-2025 after Pinnacle discontinued its free API odds feed — the methodology write-up and historical archive remain online.

---

## 6) Podcasts (podcasts)

| Podcast | Angle | Link |
|---|---|---|
| **The Double Pivot** (Mike L. Goodman & Michael Caley — independent) | Soccer analytics, xG, tactics & analytics-in-clubs interviews | [Spotify](https://open.spotify.com/show/4lBU3spHZaQWJyUcCUbkY8) · [Apple](https://podcasts.apple.com/us/podcast/the-double-pivot-soccer-analysis-analytics-and-commentary/id1121866859) |
| **American Soccer Analysis Podcast** | MLS/NWSL analytics from the ASA crew | [ASA podcasts](https://www.americansocceranalysis.com/podcasts) |
| **Analytics FC Podcast** | Data in scouting/recruitment; guests from clubs & media | [analyticsfc.co.uk/podcast](https://analyticsfc.co.uk/podcast/) · [Spotify](https://open.spotify.com/show/3xSnkVzcLpwbLuUgWmgTiS) |
| **Wharton Moneyball** | Academic sports analytics (Bradlow/Jensen/Massey/Wyner) | [Knowledge@Wharton](https://knowledge.wharton.upenn.edu/shows/moneyball/) · [Apple](https://podcasts.apple.com/us/podcast/wharton-moneyball/id1159695411) |
| **Bettor Thinking** | Betting-skill / psychology; Ep.01 is Joseph Buchdahl on *why most bettors fail* | [Apple ep.01](https://podcasts.apple.com/gb/podcast/ep-01-joseph-buchdahl-why-most-sports-bettors-fail/id1785666445) |

---

## 7) YouTube channels (canais no YouTube)

| Channel | What it covers | Link |
|---|---|---|
| **Tifo Football / The Athletic FC** | Animated tactics + analytics explainers (1M+ subs) | [youtube @TifoIRL](https://www.youtube.com/c/TifoIRL) |
| **StatsBomb** | Product + analytics education, conference talks | [statsbomb.com](https://statsbomb.com/) → YouTube |
| **Friends of Tracking** | Free lecture series (Sumpter, Laurie Shaw, etc.) on tracking-data modelling in Python | [Friends of Tracking](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w) |
| **McKay Johns** | Practical Python football-data tutorials (scraping, xG, viz) | [youtube.com/c/McKayJohns](https://www.youtube.com/c/McKayJohns) |

---

## 8) Open-source prediction & data projects on GitHub (projetos de código aberto)

Verified to exist with **star counts checked mid-2026** (rounded). These are *community showcase / prediction / betting-automation* projects — the **core modelling libraries** (`penaltyblog`, `socceraction`, `soccerdata`, `kloppy`, `mplsoccer`, `elote`, `footBayes`, `goalmodel`) are catalogued on the [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) page; here we index what the *community* builds and gathers around.

### 8a) Open data hubs & scrapers the community runs on

| Repo | ★ (2026) | What it is | Link |
|---|---:|---|---|
| **statsbomb/open-data** | ~3.4k | Free event + 360 data (World Cups, WSL, historic La Liga) — the reference open dataset | [github](https://github.com/statsbomb/open-data) |
| **probberechts/soccerdata** | ~1.8k | Scrapes FBref, Understat, ClubElo, Sofascore, WhoScored, Football-Data.co.uk into tidy frames | [github](https://github.com/probberechts/soccerdata) |
| **openfootball/football.json** | ~970 | Public-domain results/fixtures JSON (EPL, Bundesliga, Serie A…), no API key | [github](https://github.com/openfootball/football.json) · [worldcup.json ~450★](https://github.com/openfootball/worldcup.json) |
| **JaseZiv/worldfootballR** | ~600 | R wrapper for FBref/Transfermarkt/Understat (⚠️ **archived** Sep 2025 — still usable, no new dev) | [github](https://github.com/JaseZiv/worldfootballR) |
| **statsbomb/StatsBombR** | ~300 | R package to stream StatsBomb open/API data | [github](https://github.com/statsbomb/StatsBombR) |

### 8b) Match-prediction / betting projects

| Repo | ★ (2026) | What it is | Link |
|---|---:|---|---|
| **martineastwood/penaltyblog** | ~190 | End-to-end: Dixon-Coles + ML + ratings + odds decoding + back-test (also in sibling page) | [github](https://github.com/martineastwood/penaltyblog) |
| **aziztitu/football-match-predictor** | ~135 | Classic-ML 1X2 predictor (LR/NB/RF) over top-5 leagues — good teaching repo | [github](https://github.com/aziztitu/football-match-predictor) |
| **opisthokonta/goalmodel** | ~115 | R Dixon-Coles / bivariate-Poisson goals model | [github](https://github.com/opisthokonta/goalmodel) |
| **octosport/octopy** | ~76 | Poisson, **Shin** margin-removal, Elo, ML prediction (companion to the octosport blog) | [github](https://github.com/octosport/octopy) |
| **RyanSCodes/Dixon-Coles-Football-Predictor** | small (~12) | Minimal, readable Dixon-Coles implementation for learning | [github](https://github.com/RyanSCodes/Dixon-Coles-Football-Predictor) |
| **BettingIsCool/woc-streamlit** | small | Streamlit demo of Buchdahl's *Wisdom of the Crowd* margin-free strategy | [github](https://github.com/BettingIsCool/woc-streamlit) |

### 8c) Exchange & betting-automation frameworks

| Repo | ★ (2026) | What it is | Link |
|---|---:|---|---|
| **betcode-org/flumine** | ~240 | Betfair / Betdaq / BetConnect **trading framework** (back-test → live); the standard for exchange bots | [github](https://github.com/betcode-org/flumine) |
| **betcode-org/betfair** (`betfairlightweight`) | ~510 | Betfair API-NG Python wrapper + streaming — the base layer of most exchange projects | [github](https://github.com/betcode-org/betfair) |

> ⚠️ Automation frameworks are neutral tooling: they place *your* decisions faster, they do **not** create an edge. Backtests overfit; live commission, latency and adverse selection erode "paper" profit. See [Betting Exchanges, Trading & Microstructure](./Betting_Exchanges_Trading_and_Microstructure.md).

### 8d) Fantasy (FPL) modelling projects

| Repo | ★ (2026) | What it is | Link |
|---|---:|---|---|
| **vaastav/Fantasy-Premier-League** | ~1.7k | The community's canonical **FPL dataset** (per-GW player CSVs, all seasons). ⚠️ Weekly updates ended after 2024/25 — now refreshed ~3× per season (season start, January window, season end) | [github](https://github.com/vaastav/Fantasy-Premier-League) |
| **sertalpbilal — FPL Optimization** (+ YouTube "FPL Optimization") | — | Linear-programming squad optimizers popular in r/FantasyPL | [github/sertalpbilal](https://github.com/sertalpbilal) |
| **olbauday/FPL-Core-Insights** | ~160 | FPL API fused with match stats + ClubElo, aligned by FPL IDs (2025/26), refreshed twice daily | [github](https://github.com/olbauday/FPL-Core-Insights) |

---

## 9) "Awesome" lists & meta-resources (listas curadas)

The fastest way to discover new repos/blogs — start here, then verify each link yourself (some entries rot).

| List | ★ (2026) | Scope | Link |
|---|---:|---|---|
| **matiasmascioto/awesome-soccer-analytics** | ~610 | Broadest curated soccer-analytics list (EN/ES): data, tools, blogs, courses | [github](https://github.com/matiasmascioto/awesome-soccer-analytics) |
| **openfootball/awesome-football** | ~235 | Datasets: national teams, clubs, schedules, players, stadiums | [github](https://github.com/openfootball/awesome-football) |
| **diegopastor/awesome-football-analytics** | ~200 | Articles, books, tools | [github](https://github.com/diegopastor/awesome-football-analytics) |
| **MLonSoccer/awesome-machine-learning-on-soccer** | ~18 | ML papers + code specifically | [github](https://github.com/MLonSoccer/awesome-machine-learning-on-soccer) |
| **wywyWang/Awesome-Sports-Analytics** | ~22 | Cross-sport papers/code/sites (incl. soccer) | [github](https://github.com/wywyWang/Awesome-Sports-Analytics) |
| **Soccermatics course (readthedocs)** | — | David Sumpter's free open course: xG/xT/Poisson in Python, with a Discord & Twelve companion | [soccermatics.readthedocs.io](https://soccermatics.readthedocs.io/) · [twelve.football/courses](https://twelve.football/courses) |

---

## 10) Books (livros)

The canon — split into *analytics* (understanding the game) and *betting* (understanding the market). All in print/e-book.

| Book | Author | Year | Angle | Link |
|---|---|---:|---|---|
| **Soccermatics: Mathematical Adventures in the Beautiful Game** | David Sumpter | 2016 (rev. eds.) | Accessible maths of football — the gateway book | [Bloomsbury](https://www.bloomsbury.com/uk/soccermatics-9781472924124/) |
| **The Expected Goals Philosophy** | James Tippett | 2019 | Popular intro to **xG** and data-driven recruitment (Brentford story) | [Amazon](https://www.amazon.com/Expected-Goals-Philosophy-Game-Changing-Analysing/dp/1089883188) · [Goodreads](https://www.goodreads.com/book/show/48734223-the-expected-goals-philosophy) |
| **Squares & Sharps, Suckers & Sharks** | Joseph Buchdahl | 2016 | The science, psychology & philosophy of gambling — market efficiency, why most lose | [High Stakes](https://www.highstakespublishing.co.uk/authorpage.php?id=231) |
| **Monte Carlo or Bust: Simple Simulations for Aspiring Sports Bettors** | Joseph Buchdahl | 2022 | Hands-on **Monte-Carlo** simulation of edges, variance, staking, risk of ruin | [Amazon](https://www.amazon.com/Monte-Carlo-Bust-Simulations-Aspiring/dp/0857304852) · [Pinnacle review](https://www.pinnacle.com/betting-resources/en/educational/book-review-monte-carlo-or-bust/sha274ccu45t7n7b) |
| **Fixed Odds Sports Betting: Statistical Forecasting & Risk Management** | Joseph Buchdahl | 2003 | The older classic on modelling + bankroll | [High Stakes](https://www.highstakespublishing.co.uk/authorpage.php?id=231) |
| **How to Find a Black Cat in a Coal Cellar** | Joseph Buchdahl | 2013 | The hard truth about **tipsters** (touts) — read before ever paying for picks | [High Stakes](https://www.highstakespublishing.co.uk/authorpage.php?id=231) |

> Reading order for a Brazilian newcomer (leitor iniciante): **Soccermatics** → **The Expected Goals Philosophy** (understand the game) → **Squares & Sharps** → **Monte Carlo or Bust** (understand the market). The two Buchdahl "market" books will do more to protect your bankroll than any tip service ever could.

---

## 11) How to plug in without getting burned (como participar com segurança)

- **Learn in public, bet in private.** Post *methods* and *back-tests* on r/algobetting / Cross Validated; never buy picks.
- **Demand CLV.** Judge any claim against the **no-vig closing line**, not "units won". Screenshots of winning weekends prove nothing (survivorship).
- **Verify every repo yourself.** Stars ≠ correctness; check the last commit, open issues, and whether it leaks future data into features (look-ahead bias).
- **Portuguese-speaking readers (🇧🇷):** the same efficiency applies to Brasileirão and to the newly-regulated market (mercado regulamentado, Lei 14.790/2023). Bet only with regulated operators, set deposit limits, and use **self-exclusion** if needed.

> ⚠️ **Final reminder / lembrete final:** this page indexes *people, communities and code for study* — it is **not** a strategy, tip, or endorsement of gambling. Markets are efficient; the expected value of betting for entertainment is **negative**. If it stops being fun, stop. **Ajuda / help:** [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

---

**Sources:** [r/algobetting](https://www.reddit.com/r/algobetting/) · [GummySearch r/algobetting](https://gummysearch.com/r/algobetting/) · [GummySearch r/SoccerBetting](https://gummysearch.com/r/SoccerBetting/) · [r/footballanalytics](https://www.reddit.com/r/footballanalytics/) · [Bet Angel Forum](https://forum.betangel.com/) · [Betfair Community](https://community.betfair.com/) · [Geeks Toy Forum](https://www.geekstoy.com/forum/) · [Betfair Trading Community](https://betfairtradingcommunity.com/) · [Cross Validated](https://stats.stackexchange.com/) · [StatsBomb @ Hudl](https://www.hudl.com/blog/elite/statsbomb) · [StatsBomb blog archive](https://blogarchive.statsbomb.com/) · [Opta Analyst](https://theanalyst.com/) · [American Soccer Analysis](https://www.americansocceranalysis.com/) · [Karun Singh xT](https://karun.in/blog/expected-threat.html) · [John Muller](https://johnspacemuller.com/) · [Tony ElHabr](https://tonyelhabr.rbind.io/) · [Soccermatics/Medium](https://soccermatics.medium.com/) · [Get Goalside](https://www.getgoalsideanalytics.com/) · [Twenty First Group](https://www.twentyfirstgroup.com/) · [Football-Data.co.uk](https://www.football-data.co.uk/) · [Wisdom of the Crowd PDF](https://www.football-data.co.uk/The_Wisdom_of_the_Crowd_updated.pdf) · [Pinnacle Betting Resources](https://www.pinnacle.com/betting-resources/en/category/educational) · [Buchdahl on Pinnacle](https://www.pinnacle.com/betting-resources/en/author/joseph-buchdahl) · [The Double Pivot](https://open.spotify.com/show/4lBU3spHZaQWJyUcCUbkY8) · [Analytics FC Podcast](https://analyticsfc.co.uk/podcast/) · [Wharton Moneyball](https://knowledge.wharton.upenn.edu/shows/moneyball/) · [Bettor Thinking ep.01](https://podcasts.apple.com/gb/podcast/ep-01-joseph-buchdahl-why-most-sports-bettors-fail/id1785666445) · [Tifo Football](https://www.youtube.com/c/TifoIRL) · [Friends of Tracking](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w) · [McKay Johns](https://www.youtube.com/c/McKayJohns) · [statsbomb/open-data](https://github.com/statsbomb/open-data) · [probberechts/soccerdata](https://github.com/probberechts/soccerdata) · [openfootball/football.json](https://github.com/openfootball/football.json) · [JaseZiv/worldfootballR](https://github.com/JaseZiv/worldfootballR) · [martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog) · [aziztitu/football-match-predictor](https://github.com/aziztitu/football-match-predictor) · [opisthokonta/goalmodel](https://github.com/opisthokonta/goalmodel) · [octosport/octopy](https://github.com/octosport/octopy) · [betcode-org/flumine](https://github.com/betcode-org/flumine) · [betcode-org/betfair](https://github.com/betcode-org/betfair) · [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) · [olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) · [awesome-soccer-analytics](https://github.com/matiasmascioto/awesome-soccer-analytics) · [Soccermatics course](https://soccermatics.readthedocs.io/) · [Expected Goals Philosophy](https://www.amazon.com/Expected-Goals-Philosophy-Game-Changing-Analysing/dp/1089883188) · [Monte Carlo or Bust](https://www.amazon.com/Monte-Carlo-Bust-Simulations-Aspiring/dp/0857304852) · [High Stakes / Buchdahl](https://www.highstakespublishing.co.uk/authorpage.php?id=231)

**Keywords:** football prediction community, soccer analytics forums, r/algobetting, r/footballanalytics, betting model blogs, StatsBomb blog, Opta Analyst, American Soccer Analysis, Expected Threat, Karun Singh, Tony ElHabr, David Sumpter Soccermatics, Joseph Buchdahl, Wisdom of the Crowd, Pinnacle Betting Resources, closing line value, CLV, open-source football prediction GitHub, penaltyblog, soccerdata, socceraction, flumine, betfairlightweight, Fantasy Premier League FPL, awesome soccer analytics, Monte Carlo or Bust, Squares and Sharps, The Expected Goals Philosophy — *(PT)* comunidade de previsão de futebol, fóruns de analítica de futebol, blogs de modelos de apostas, projetos de código aberto, aposta algorítmica, valor da linha de fechamento, jogo responsável, futebol, apostas esportivas, previsão de partidas, livros de apostas, boletins e podcasts de análise de dados.
