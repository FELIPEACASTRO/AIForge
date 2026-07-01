# Odds, Betting Markets & Value Betting (Research)

> Authoritative, worldwide reference on the **betting side** of football (soccer / futebol) prediction — how bookmaker odds work, how to strip the margin (removing the vig), why markets are **highly efficient**, what **Closing Line Value (CLV)** is, where to get **odds data / APIs**, and the theory of **value betting, Kelly staking (gestão de banca), arbitrage and honest backtesting** — with real URLs, free-vs-paid marks and multi-region coverage. Built for **data science / ML research & education**, current 2024–2026. Sibling page: [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md).

> ⚠️ **Research & education only — NOT betting advice, NOT a system, NOT a tip.** Betting markets are **highly efficient**; **most bettors lose money over time**, and **beating the closing line consistently is extremely hard**. Across ~397,935 Pinnacle football games the closing line tracked real outcomes with **r² ≈ 0.997** ([Trademate Sports](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458); see also [football-data.co.uk — efficiency of the Pinnacle closing line](https://www.football-data.co.uk/blog/pinnacle_efficiency.php)). Nothing here is a recommendation to bet. If gambling stops being fun, get help — see [§11](#11-responsible-gambling-jogo-responsável--mandatory). 🇧🇷 **CVV 188** · [Autoexclusão SIGAP](https://www.gov.br/pt-br/servicos/plataforma-centralizada-de-autoexclusao-apostas).

---

## 1) Reading odds — formats & implied probability (probabilidade implícita)

Odds are just a price. Every price implies a probability; the sum of a market's implied probabilities exceeds 100% because of the bookmaker's margin. Convert everything to **decimal** for modelling.

| Format | Example | Decimal (`d`) | Implied prob `p` | Net profit per 1 unit staked |
|---|---|---|---|---|
| **Decimal** (EU / 🇧🇷 "odd") | `2.50` | `2.50` | `1/d` = **0.40** | `d − 1` = 1.50 |
| **Fractional** (UK) | `6/4` | `a/b + 1` = 2.50 | `b/(a+b)` = 0.40 | `a/b` = 1.50 |
| **American** (+) | `+150` | `m/100 + 1` = 2.50 | `100/(m+100)` = 0.40 | `m/100` = 1.50 |
| **American** (−) | `−200` | `100/m + 1` = 1.50 | `m/(m+100)` = 0.667 | `100/m` = 0.50 |
| **Hong Kong** | `1.50` | `hk + 1` = 2.50 | `1/(hk+1)` = 0.40 | `hk` = 1.50 |

- **Implied probability** from decimal odds: **`p = 1 / d`**. A quoted 2.50 "says" ~40%.
- **Fair (true) probability** ≠ implied probability until you remove the margin (§2).
- Reference calculators: [Pinnacle odds/margin tools](https://www.pinnacle.com/betting-resources/en/betting-tools/margin-calculator) · Python conversions in [penaltyblog](https://github.com/martineastwood/penaltyblog).

---

## 2) The bookmaker margin (overround / vig / juice / *margem*) & how to remove it

The **overround** (a.k.a. margin, vig, juice; 🇧🇷 *margem/comissão embutida*) is the sum of implied probabilities minus 1. It is the structural reason a break-even bettor still loses.

**Overround** = `Σ (1/dᵢ) − 1`. Example 1X2 market at `2.10 / 3.40 / 3.50` → `0.476 + 0.294 + 0.286 = 1.056` → **5.6% margin**. A two-way `1.91 / 1.91` market → `1.047` → **4.7% margin** ([worked example](https://www.pinnacleoddsdropper.com/blog/overround)). Sharp books (Pinnacle) run **~2%** on liquid markets vs ~5–8% at soft/retail books ([Pinnacle low-margin model](https://www.completesports.com/pinnacles-low-margin-business-model/)).

**Removing the margin ("de-vig" / *tirar a margem*)** — several competing theories of *how* books apply the margin. The open-source `penaltyblog` implements seven; the [`implied`](https://cran.r-project.org/web/packages/implied/vignettes/introduction.html) R package (Jonas C. Lindstrøm) covers the same family.

| Method (`penaltyblog`) | Idea | Notes |
|---|---|---|
| **Multiplicative / basic** | Divide each `1/dᵢ` by the overround sum (normalisation) | Default; ignores favourite–longshot bias |
| **Additive** | Subtract an equal share of the margin from each outcome | Can give negative probs on longshots |
| **Power** | Raise implied probs to a power `k` so they sum to 1 | Bends more margin onto longshots |
| **Shin** | Solve Shin's `z` = share of "insider"/informed money | Academic standard for **favourite–longshot bias** |
| **Differential margin weighting** | Weight margin unequally across outcomes | Flexible, more parameters |
| **Odds ratio** | Fair odds via a constant odds-ratio transform | Cheng/others |
| **Logarithmic** | Log transform of implied probs | Alternative shape |

- Docs: [penaltyblog — implied](https://penaltyblog.readthedocs.io/en/latest/implied/implied.html) · tutorial [From Biased Odds to Fair Probabilities](https://pena.lt/y/2025/09/14/from-biased-odds-to-fair-probabilities/).
- Peer-reviewed comparison: **Clarke, Kovalchik & Ingram (2017)**, *Adjusting Bookmaker's Odds to Allow for Overround*, *American Journal of Sports Science* 5(6):45–49 ([journal](https://www.sciencepublishinggroup.com/article/10.11648/j.ajss.20170506.12) · [PDF](https://outlier.bet/wp-content/uploads/2023/08/2017-clarke-adjusting_bookmakers_odds.pdf)) — finds the **power method** generally outperforms multiplicative and is comparable to Shin.
- **Shin (1991/1993)** insider-trading model underlies the Shin method; it de-biases longshots better than plain normalisation.

| Concept | Formula / definition | Source |
|---|---|---|
| Implied probability | `p = 1/d` | [Pinnacle Odds Dropper](https://www.pinnacleoddsdropper.com/blog/implied-odds) |
| Overround / margin | `Σ 1/dᵢ − 1` | [Pinnacle Odds Dropper](https://www.pinnacleoddsdropper.com/blog/overround) |
| Fair prob (basic) | `pᵢ = (1/dᵢ) / Σ(1/dⱼ)` | [penaltyblog](https://pena.lt/y/2025/09/14/from-biased-odds-to-fair-probabilities/) |
| Fair (no-vig) odds | `d_fair = 1 / p_fair` | [Clarke et al. 2017](https://outlier.bet/wp-content/uploads/2023/08/2017-clarke-adjusting_bookmakers_odds.pdf) |

---

## 3) Market efficiency & the closing line — the real benchmark (CLV)

The **closing line** is the last price before kickoff. It has absorbed all sharp money, team news and late information, so it is the market's best estimate of true probability. This is the crux of every honest betting-research project.

- **Closing Line Value (CLV)** = did you bet at odds **better than the eventual close**? Consistently positive CLV is the single most reliable evidence of a real edge; it is the metric syndicates track ([Pinnacle: What is CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting) · [ProbWin](https://en.probwin.com/guides/closing-line-value-clv-ultimate-metric-measure-your-edge/)).
- **Pinnacle as the "sharp" reference**: low margin, high limits, welcomes winners → its closing odds are treated as ground truth in academic and practitioner work ([why Pinnacle is sharp](https://www.completesports.com/how-pinnacle-sets-the-sharpest-lines/)). Efficiency is strong: across ~397,935 Pinnacle games, closing prices ≈ outcomes with **r² ≈ 0.997** ([Trademate Sports](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458) · [football-data.co.uk closing-line efficiency](https://www.football-data.co.uk/blog/pinnacle_efficiency.php)).
- **The honest test**: a model that "profits" against opening/soft odds but **cannot beat the close** has found no edge — it has been front-running information the market already priced. Beating the closing line is the bar, and it is very hard.
- **Known residual inefficiency**: the **favourite–longshot bias** (longshots over-bet, favourites under-bet) is documented across football markets, but margins usually swallow it ([Cain, Law & Peel, UK football](https://www.researchgate.net/publication/4920285_The_Favourite-Longshot_Bias_and_Market_Efficiency_in_UK_Football_Betting) · [Whelan 2024, *Economica*](https://onlinelibrary.wiley.com/doi/10.1111/ecca.12500)).

| Idea | What it means | Source |
|---|---|---|
| Closing line | Final pre-kickoff price = most efficient estimate | [Pinnacle CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting) |
| CLV | `your_odds / close_odds − 1` (positive = you beat the close) | [ProbWin](https://en.probwin.com/guides/closing-line-value-clv-ultimate-metric-measure-your-edge/) |
| Efficient-market benchmark | Pinnacle close, r² ≈ 0.997 vs results (~397,935 games) | [Trademate Sports](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458) |
| Real-world limit-risk | Winning accounts get limited/closed (Kaunitz 2017) | [arXiv 1710.02824](https://arxiv.org/abs/1710.02824) |

---

## 4) Odds data — sources & APIs (dados de odds)

Odds columns let you back-test against the market and *measure* efficiency. Free CSV history (football-data.co.uk, in the sibling page) is the cheapest start; live/aggregated odds need an API.

| Source | Type | Coverage | Odds formats | Free / Paid | URL |
|---|---|---|---|---|---|
| **The Odds API** | Live + historical (since 2020) aggregator, **40+** books | 70+ sports; soccer EPL/UCL/La Liga/Serie A/Bundesliga/Ligue 1 + many | decimal & American | **Free 500 credits/mo**; paid $30 (20k), $59 (100k), $119 (5M), $249 (15M) | [the-odds-api.com](https://the-odds-api.com/) |
| **Betfair Exchange API** | **Exchange** back/lay prices + traded volume (real market probabilities) | Global exchange markets, in-play | decimal | **Free** for personal dev use (JSON-RPC; live betting needs funded acct) | [developer.betfair.com](https://developer.betfair.com/exchange-api/) · [samples](https://github.com/betfair-datascientists/API) |
| **Pinnacle API** | Sharp bookmaker odds/lines feed | Global, all bet types | decimal/American | **Restricted** — closed to public since 23 Jul 2025; academic/commercial only via `api@pinnacle.com` | [docs (GitHub)](https://github.com/pinnacleapi/pinnacleapi-documentation) |
| **OddsJam** | Real-time aggregator, **100+** books, props/alt markets | US-centric + global | decimal/American | **Paid** (enterprise; contact-for-pricing, no public free tier) | [oddsjam.com/odds-api](https://oddsjam.com/odds-api) |
| **OddsPortal** | Odds-comparison site + historical odds/movement (opening+closing) | Hundreds of books, worldwide leagues | decimal/etc. | **Free site** (scrape ⚠️ ToS/JS-rendered) | [oddsportal.com](https://www.oddsportal.com/) · scraper [OddsHarvester](https://github.com/jordantete/OddsHarvester) |
| **SportsGameOdds / OddsPapi / SharpAPI** | Aggregators bundling Pinnacle + soft books via one endpoint | Global | decimal/American | Freemium → paid | [sportsgameodds.com](https://sportsgameodds.com/) · [oddspapi.io](https://oddspapi.io/) · [sharpapi.io](https://sharpapi.io/) |
| **SportsDataIO / SportMonks Odds** | Odds inside broader stats APIs | US + global leagues incl. 🇧🇷 Série A | decimal | Freemium → paid | [sportsdata.io](https://sportsdata.io/live-odds-api) · [sportmonks.com](https://www.sportmonks.com/football-api/) |
| **football-data.co.uk** (see sibling) | Historical CSV: up to 10 books, **opening+closing** since 2019/20, O/U + Asian handicap | 22 EU divisions + 16 extra (🇧🇷🇦🇷🇺🇸🇯🇵🇨🇳🇲🇽…) | decimal | **Free** | [data.php](https://www.football-data.co.uk/data.php) |

> **Note on "free odds APIs":** many free feeds are *scraped* from books and may breach ToS or vanish. The exchange feed (Betfair) is the closest thing to a **market-implied probability** you can get openly, because traded prices are set by bettors, not a bookmaker.

---

## 5) Value betting — edge, expected value & where an "edge" could come from

A **value bet (aposta de valor)** exists when **your** estimated probability beats the market's fair probability, i.e. the bet has **positive expected value (EV)**.

| Concept | Formula / definition | Meaning |
|---|---|---|
| Edge / EV per unit | `EV = p·d − 1` | Positive → +EV over many bets |
| Value condition | `p > 1/d` (your prob beats implied) | Necessary for a value bet |
| Model vs market | compare model `p` to **de-vigged** market `p_fair` | Beating *fair* odds, not headline odds |
| Yield / ROI | `profit / total_staked` | Long-run return, dominated by variance |

- The edge must come from **information the market has not priced** (rare) or from a **soft book** slower than the sharp consensus. Against the Pinnacle **close**, edges are near zero by construction (§3).
- Model → probability pipelines (Poisson, **Dixon–Coles (1997)**, bivariate Poisson, xG-based, gradient boosting, Elo/rating) live in the sibling datasets page and in [penaltyblog](https://github.com/martineastwood/penaltyblog) / [Dixon–Coles docs](https://docs.pena.lt/y/models/dixon_coles.html). A model is only useful for betting if its calibrated probabilities beat de-vigged market probabilities **out of sample**.
- **Classic scoring models:** Maher (1982), *Modelling Association Football Scores*, *Statistica Neerlandica* 36:109–118; Dixon & Coles (1997), *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, *JRSS Series C (Applied Statistics)* 46(2):265–280 ([DOI 10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065)).
- **Reality:** "most betting edges vanish when you backtest properly" — the default assumption should be *no edge* until CLV proves otherwise.

---

## 6) Staking & bankroll management (gestão de banca) — Kelly & fractional Kelly

Even with a genuine edge, **bet sizing** decides whether you survive variance. The **Kelly criterion** (Kelly, 1956) maximises the long-run growth rate of the log of the bankroll.

| Staking plan | Rule | Trade-off |
|---|---|---|
| **Flat** (*aposta fixa*) | Same stake every bet | Simplest; ignores edge size |
| **Percentage** | Fixed % of current bankroll | Auto-scales up/down with results |
| **Kelly (full)** | `f* = (p·d − 1)/(d − 1) = edge/(d−1)` | Growth-optimal **if `p` is exact**; brutal drawdowns |
| **Fractional Kelly** | Bet `¼–½ · f*` | Much lower variance; standard among pros |

- Full Kelly assumes you **know the true probability** — you don't. Overestimating `p` oversizes bets and risks ruin, so practitioners use **fractional Kelly (¼–½)** ([Punter2Pro staking](https://punter2pro.com/flat-percentage-kelly-staking-plans/)).
- **Football-Data's own warning**: Kelly staking's pitfalls (estimation error, variance, non-independent bets) can wreck a bankroll ([football-data.co.uk/blog/kelly_staking](https://www.football-data.co.uk/blog/kelly_staking.php)).
- **Variance ≠ ruin only in theory:** with Half-Kelly a 5-bet losing streak is a ~22% drawdown; full Kelly is far worse. Bankroll size, minimum stakes and correlated bets all matter.

---

## 7) Arbitrage & sure bets (arbitragem / *surebets*) — concept only

**Arbitrage ("arbing")** backs *every* outcome across different books so the payout exceeds the total stake regardless of result. A pure arb exists when the **combined implied probability is below 100%**.

| Concept | Formula / definition | Source |
|---|---|---|
| Arb condition (2-way) | `1/d_A + 1/d_B < 1` | [Oddspedia surebets](https://oddspedia.com/surebets) |
| Arb % (margin) | `arb% = Σ 1/dᵢ` → profit if `< 1` | [RebelBetting formula](https://www.rebelbetting.com/en-us/arbitrage-betting/formula) |
| Stake split | `stakeᵢ ∝ 1/dᵢ` to equalise payout | [Oddspedia](https://oddspedia.com/surebets) |

- The word "sure" refers to the **maths, not the execution**: odds move, stakes get voided, and **books limit/close accounts** that arb or beat the close — exactly what happened to Kaunitz et al. ([arXiv 1710.02824](https://arxiv.org/abs/1710.02824) · [code: BeatTheBookie](https://github.com/Lisandro79/BeatTheBookie)). Included here as a **market-mechanics concept**, not a strategy to run.

---

## 8) Backtesting a betting strategy honestly (avoid fooling yourself)

Most "profitable" backtests are artefacts of leakage, survivorship and unrealistic execution. A credible test looks like this.

| Pitfall | Why it inflates results | Honest fix |
|---|---|---|
| **Ignoring the vig** | Break-even skill still loses to the margin | Bet against **de-vigged** fair odds; report net of margin |
| **Backtesting vs opening/soft odds** | You're front-running info the market later prices | Grade against the **closing line**; report **CLV** |
| **Look-ahead / leakage** | Uses info unknown at bet time (lineups, later odds) | **Walk-forward** / time-respecting splits only |
| **Overfitting** | Many parameters fit historical noise | Few, justified parameters; out-of-sample + rolling windows |
| **Assuming you can stake freely** | Winners get **limited/closed**; big stakes move the line | Model stake caps, min odds, limits, slippage |
| **Ignoring variance** | Short samples look "profitable" by luck | Thousands of bets; report yield **with** confidence bands |

- Practical walk-throughs: [Systematic Sports — Backtesting a Sports Betting Strategy](https://medium.com/systematic-sports/backtesting-a-sports-betting-strategy-283833a5eca3) · [Betfair Data Scientists — backtesting with JSON stream data](https://betfair-datascientists.github.io/tutorials/backtestingRatingsTutorial/).
- **Golden rule:** if the strategy cannot demonstrate **positive CLV** on out-of-sample, closing-line-graded data, treat the "profit" as noise.

---

## 9) Regional odds-market notes (multi-country)

| Region | Odds-data reality | Notes |
|---|---|---|
| 🇪🇺 **Europe** (EPL, La Liga, Serie A, Bundesliga, Ligue 1) | Deepest, most efficient markets; football-data.co.uk closing odds; Betfair exchange liquidity high | Hardest to beat the close |
| 🇧🇷 **Brazil** (Brasileirão Série A/B, Libertadores) | Regulated since **1 Jan 2025** (Lei das Bets); odds via The Odds API / SportMonks; results+odds CSV in football-data.co.uk "extra" | See regulation in §11 |
| 🌎 **South America** (CONMEBOL) | Thinner than Europe; some soft-book inefficiency but low limits | Argentina/Brazil odds in football-data.co.uk |
| 🌍 **Africa** (CAF/AFCON) | Sparse odds coverage; mostly via global aggregators | Data gaps; verify samples |
| 🌏 **Asia** (J-League, K-League, ACL) | **Asian handicap** markets deep & liquid (Pinnacle, exchanges); 🇯🇵 Japan odds in football-data.co.uk | AH is the sharpest soccer market |
| 🇺🇸 **North America** (MLS, Liga MX) | US odds in American format; DraftKings/FanDuel via The Odds API | Newer, sometimes softer lines |
| 🏆 **International / World Cup** | Huge liquidity around tournaments; efficient closes | Public attention ≠ inefficiency |

---

## 10) Reality check — read before trusting any odds model

- **The market is the benchmark, not the enemy.** The closing line (esp. Pinnacle) is ~efficient (r² ≈ 0.997); your model's job in research is usually to **approximate** it, not beat it.
- **CLV is the only honest scoreboard.** Profit without positive CLV is variance; positive CLV without profit still signals skill.
- **Winning is punished operationally.** Even a real edge triggers **account limits/closure** at most books — a structural reason retail "systems" don't compound ([Kaunitz 2017](https://arxiv.org/abs/1710.02824)).
- **Most bettors lose.** The margin guarantees the average bettor is negative-EV. Treat all of the above as **modelling practice and market microstructure study**, never as an income plan.

---

## 11) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious financial and mental-health harm. This page exists for **data-science / ML research and education only** and is **not** betting advice. Set limits, never chase losses (*não persiga o prejuízo*), and get help if betting stops being fun.

| Region | Resource | Contact / tool |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) (Gordon Moody) | Multilingual (40+ langs) online chat & forum |
| 🇬🇧 UK | [GambleAware](https://www.gambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇺🇸 USA | [NCPG](https://www.ncpgambling.org/) | **1-800-MY-RESET** — call/text/chat · [1800myreset.org](https://www.1800myreset.org/) (24/7) |
| 🇧🇷 Brazil | [Jogadores Anônimos](https://jogadoresanonimos.com.br/) · CAPS/SUS · [SPA/MF — Apostas de Quota Fixa](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) | Apoio emocional **CVV 188** (24h) |
| 🇧🇷 Brazil — self-exclusion | [Autoexclusão centralizada (SIGAP)](https://www.gov.br/pt-br/servicos/plataforma-centralizada-de-autoexclusao-apostas) | Bloqueia o CPF em **todas** as casas autorizadas (prazo determinado ou indeterminado) |

**🇧🇷 Regulation (Lei das Bets):** fixed-odds betting was legalised by **Lei nº 14.790/2023** (building on Lei nº 13.756/2018) and regulated by the **Secretaria de Prêmios e Apostas (SPA/MF)**; only SPA-authorised operators may run nationally **since 1 Jan 2025**, monitored via **SIGAP** (Serpro). Responsible-gambling tools — deposit limits, alerts, self-exclusion — are **mandatory** ([Lei das Bets overview](https://pt.wikipedia.org/wiki/Lei_das_Bets) · [Portaria SPA/MF nº 1.231/2024 — jogo responsável](https://www.legisweb.com.br/legislacao/?id=462714)). Minors and self-excluded CPFs are blocked.

---

## 12) Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** the-odds-api.com · developer.betfair.com · github.com/pinnacleapi/pinnacleapi-documentation · oddsjam.com · oddsportal.com · github.com/jordantete/OddsHarvester · sportsgameodds.com · oddspapi.io · sharpapi.io · sportsdata.io · sportmonks.com · football-data.co.uk (data.php · blog/kelly_staking · blog/pinnacle_efficiency) · penaltyblog (github.com/martineastwood/penaltyblog · readthedocs · pena.lt/y · docs.pena.lt/y) · cran.r-project.org/package=implied · sciencepublishinggroup.com (Clarke, Kovalchik & Ingram 2017) · outlier.bet (Clarke PDF) · pinnacle.com (CLV · margin-calculator) · pinnacleoddsdropper.com · completesports.com · tradematesports.medium.com · arxiv.org/abs/1710.02824 (Kaunitz et al.) · github.com/Lisandro79/BeatTheBookie · researchgate.net (Cain, Law & Peel) · onlinelibrary.wiley.com (Whelan 2024) · doi.org/10.1111/1467-9876.00065 (Dixon–Coles 1997) · oddspedia.com/surebets · rebelbetting.com · punter2pro.com · medium.com/systematic-sports · betfair-datascientists.github.io · gambleaware.org · gamcare.org.uk · gamblingtherapy.org · ncpgambling.org · 1800myreset.org · gov.br/fazenda (SPA) · gov.br (autoexclusão SIGAP) · jogadoresanonimos.com.br · pt.wikipedia.org/wiki/Lei_das_Bets · legisweb.com.br (Portaria SPA/MF 1.231/2024)

**Keywords:** betting odds, decimal fractional American odds, implied probability, bookmaker margin, overround, vig, juice, remove the vig, de-vig, Shin method, power method, no-vig fair odds, market efficiency, closing line value, CLV, Pinnacle sharp odds, favourite-longshot bias, value betting, expected value, edge, Kelly criterion, fractional Kelly, bankroll management, staking, arbitrage, sure bets, backtesting, The Odds API, Betfair Exchange API, OddsPortal, OddsJam, football prediction, soccer betting research, jogo responsável, odds de apostas, probabilidade implícita, margem da casa, tirar a margem, valor esperado, gestão de banca, critério de Kelly, mercado eficiente, linha de fechamento, aposta de valor, arbitragem, surebets, autoexclusão, Lei das Bets, SIGAP, CVV 188.
