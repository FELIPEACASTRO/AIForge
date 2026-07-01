# Bet Selection, Staking & High-Odds ("Apostas Bomba") Analysis

> Authoritative, evidence-based reference on **how bets are selected and sized** (seleção e dimensionamento de apostas) and an **honest, mathematical debunk of high-odds/accumulator ("apostas bomba", múltiplas) betting** for football (soccer / futebol) — value/edge detection, expected value, closing-line value (CLV), devigging, probability calibration, staking (flat / Kelly / fractional Kelly), market types & where soft spots historically appear, and why long-odds parlays and "tipster bomb tips" are usually −EV marketing. **Research & education only (pesquisa e educação — data science / ML), current 2024–2026.**

> ⚠️ **NOT betting advice (NÃO é dica de aposta). Read this first.** Football betting markets are **highly efficient**: from **397,935** Pinnacle games the closing line predicted outcomes with **r² ≈ 0.997** ([Trademate Sports](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458)), and sharp books run ~**2–3%** margins on main markets ([Pinnacle AH](https://www.pinnacle.com/betting-resources/en/betting-strategy/how-good-are-pinnacles-asian-handicap-markets/zm62p4gdr62ngx8n)). **Only ~2–5% of bettors are profitable long-term; ~95–97% lose** ([SportBotAI stats](https://www.sportbotai.com/stats/sports-betting-profitability), [BettorEdge](https://www.bettoredge.com/post/what-percentage-of-sports-bettors-are-profitable)). Sustained edges are **rare, small, and hard to keep** (you get limited/banned — see §10). Nothing here is a tip or a system. Help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

**Sibling pages:** [Global Datasets & APIs](./Global_Datasets_and_Data_APIs.md) · [Open-Source Tools](./Open_Source_Tools_and_Libraries.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · [Kaggle Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

---

## 1) The single number that matters: Closing Line Value (CLV)

Before any staking maths, understand what "having an edge" actually means. **CLV = beating the closing line.** If you bet a price better (higher) than the market's final pre-kickoff odds at a sharp book, you have positive CLV. Because the **closing line is the most accurate public estimate of true probability**, consistent positive CLV is the **strongest single predictor of long-term profit** — and its *absence* is proof you have no edge, however good your P&L looks over a small sample ([Pinnacle](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting), [Buchdahl via Pinnacle Odds Dropper](https://www.pinnacleoddsdropper.com/blog/closing-line-value--clv-demystified-by-expert-joseph-buchdahl), [VSiN](https://vsin.com/how-to-bet/the-importance-of-closing-line-value/)).

| Concept | Definition | Caveat |
|---|---|---|
| **Closing line** | Final odds at kickoff at a sharp book (Pinnacle/Betfair) after all info + sharp money | The market, not you, is usually right; r²≈0.997 vs outcomes |
| **CLV (positive)** | Your taken price > devigged closing price | Needs a large sample; one bet proves nothing |
| **CLV as proof** | Long-run +CLV ⇒ likely +EV; −CLV ⇒ almost certainly −EV | You can win with −CLV (luck) or lose with +CLV (variance) short-term |
| **Beat-the-close %** | Fraction of bets with +CLV | Track this *before* trusting ROI |

**Bottom line:** if you cannot beat the closing line, you do not have an edge. Track CLV first; treat ROI over <500–1,000 bets as noise.

---

## 2) Value detection — edge, implied probability, expected value

A bet is "value" only when your **model probability exceeds the fair (margin-free) market probability** by more than your uncertainty. The market's quoted odds already bake in the bookmaker margin (*overround / vigorish / "juice"* — pt: *margem*), so you must **devig** before comparing.

| Quantity | Formula (decimal odds `o`, model prob `p`) | Notes |
|---|---|---|
| **Implied prob (raw)** | `1 / o` | Sums to >1 across a market = the overround |
| **Overround / margin** | `Σ(1/oᵢ) − 1` | e.g. 1X2 sum 1.05 ⇒ 5% margin ([EV/devig](https://oddsindex.com/guides/expected-value-betting-guide)) |
| **Fair prob (devigged)** | raw prob normalized to sum = 1 (see §2.1) | Compare *this*, not raw |
| **Edge** | `p − p_fair` | Only bet if edge > threshold (after margin + error) |
| **Expected value (per unit)** | `EV = p·o − 1 = p·(o−1) − (1−p)` | +EV needed for long-run profit ([BettorEdge](https://www.bettoredge.com/post/how-to-calculate-expected-value-in-betting), [Smarkets](https://help.smarkets.com/hc/en-gb/articles/214554985)) |
| **Break-even prob** | `1 / o` | At −110 (o≈1.91) you need **52.38%** just to break even ([Betmana](https://betmana.co.uk/guide/why-bettors-lose-money/)) |

**Worked example.** Model says home win `p = 0.55`; best price `o = 2.00`. `EV = 0.55×2.00 − 1 = +0.10` (+10% per unit) — *if* `p` is correct. If the fair (devigged) market probability is 0.52, your edge is only 3 points, well inside typical model error, so the "value" may be illusion.

### 2.1 Devigging (removing the margin) — pick your method

The overround is **not** shared equally across outcomes (longshots are taxed more — see §7), so the method matters.

| Method | How it strips the vig | Caveat / when to use |
|---|---|---|
| **Multiplicative / proportional** | Divide each raw prob by the overround (scale to 1) | Simplest, most common; assumes vig is proportional — ignores favourite-longshot bias ([Bet Hero](https://betherosports.com/blog/devigging-methods-explained)) |
| **Additive** | Subtract equal share of overround from each outcome | Can produce negative probs on longshots; rarely best |
| **Power** | Solve `p_fair = raw^k` so Σ = 1 | Handles lopsided odds; often the pro default ([Sharkbetting](https://www.sharkbetting.com/blog/devig-explained)) |
| **Shin (1993)** | Models margin as insider-information "z" | Best for 3-way (1X2) markets; academic basis. Empirical comparison of devig methods: [Clarke, Kovalchik & Ingram (2017)](https://www.researchgate.net/publication/326510904_Adjusting_Bookmakers_Odds_to_Allow_for_Overround) |

Empirically the **power** method matches or beats Shin and clearly beats multiplicative for skewed prices. Open-source `penaltyblog` implements multiplicative, additive, power and Shin ([docs](https://penaltyblog.readthedocs.io/en/latest/)).

---

## 3) Model vs. market — sharp closing odds as ground truth, and calibration

You are not competing against "the bookmaker's opinion"; you are competing against **all sharp money aggregated into the closing price**. The disciplined workflow treats **devigged sharp closing odds as the label** and asks where your model *systematically* disagrees.

| Step | What to do | Caveat |
|---|---|---|
| **Anchor to sharps** | Use Pinnacle/Betfair devigged closing prob as best estimate of truth | Soft-book prices are noisier and higher-margin |
| **Calibrate** | Ensure "p=0.60" events happen ~60% of the time (isotonic/Platt) | A model can rank well (high AUC) yet be miscalibrated ([arXiv 2106.14345](https://arxiv.org/pdf/2106.14345)) |
| **Score probabilistically** | Brier score / log loss / RPS, not accuracy | Miscalibration distorts edge, stake size and CLV ([StatsTest](https://www.statstest.com/calibration-checks-brier-score-reliability-diagrams)) |
| **Compare to market** | Does your model beat the *closing* Brier/RPS? | If not, the market already knows what you know |

The Wilkens (2026) Bundesliga study is a clean, honest example: a simple **xG → Skellam** model with **isotonic calibration** over 11 seasons found bookmaker odds were *better calibrated*, yet the model still captured a home-win signal worth ~10% ROI at average odds (≈15% at best prices) — while **away-win bets consistently lost** ([Journal of Sports Analytics](https://journals.sagepub.com/doi/10.1177/22150218261416681), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5381388)). Treat such ROI figures as *upper bounds* that shrink after margin, limits and slippage.

---

## 4) Staking — how much to bet (dimensionamento / gestão de banca)

Selection decides *what*; staking decides *how much*, and it governs **variance, drawdown, and risk of ruin** far more than most bettors realise. All methods below assume you *already* have +EV; on −EV bets, staking only changes how fast you lose.

| Method | Definition | Caveat |
|---|---|---|
| **Flat** | Same stake every bet (e.g. 1 unit) | Simple, low-variance, robust to bad probability estimates; ignores edge size — good baseline/back-test control |
| **Percentage** | Fixed % of current bankroll | Auto-compounds and shrinks in drawdown; still ignores edge/odds |
| **Kelly (full)** | `f* = (b·p − q)/b = (p·o − 1)/(o − 1)`, `b=o−1`, `q=1−p` | Maximizes long-run growth ([Kelly 1956 / Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)); **very** volatile; assumes `p` is exact |
| **Fractional Kelly** | Stake ½ or ¼ of `f*` | Standard pro choice — cuts variance/drawdown sharply for small growth loss; buffers estimation error ([SportsTrade](https://www.sportstrade.io/blog-detail/141/the-fractional-kelly-bankroll-management-system.html), [Downey sims](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html)) |
| **Kelly cap** | Cap any single stake (e.g. ≤2–5% bankroll) | Guards against outlier `f*` from a mispriced input ([Bet Hero](https://betherosports.com/blog/kelly-criterion-sports-betting)) |

**Kelly's fragility.** If you *think* `p=0.55` but truth is `0.51`, full Kelly stakes **≈2× too much** ([Quant Matter](https://quantmatter.com/kelly-criterion-formula/)); over-betting Kelly raises drawdown *and* long-run ruin risk. Because model probabilities are always estimates, **fractional Kelly (¼–½) is the norm**, and a common rule caps any bet at ≤5% of bankroll.

**Simultaneous & correlated bets.** Naïve Kelly assumes one bet at a time. Real slates have **overlapping/correlated** selections (same match, same league round, same driver of variance). Summed independent Kelly fractions **over-stake**; the correct approach solves a **portfolio Kelly** (maximize expected log-growth of the *combined* outcome, accounting for covariance) or, in practice, scales down and hard-caps total exposure per round. Framing a book of bets as a **risk-managed portfolio** is exactly the direction flagged by the ML-in-betting systematic review ([arXiv 2410.21484](https://arxiv.org/abs/2410.21484)).

| Risk term | Meaning | Practical guard |
|---|---|---|
| **Drawdown** | Peak-to-trough bankroll fall | Even +EV Kelly bettors routinely see 30–50%+ drawdowns |
| **Risk of ruin** | Prob. of losing the bankroll | Rises fast above full Kelly; near-0 only with small fractions + caps |
| **Variance** | Spread of outcomes | Explodes with high odds and parlays (see §7) |

---

## 5) Market & bet types — and where soft spots historically appear

Not all markets are equally efficient. The **most-bet, high-liquidity markets are the sharpest**; obscure/derivative markets are where inefficiency has historically survived — but they are also where you get limited fastest.

| Market (pt) | What it is | Efficiency / soft-spot note |
|---|---|---|
| **1X2** (resultado / mata-mata) | Home / Draw / Away | Core market, very efficient at sharp books (~2–3% margin) |
| **Double chance** (dupla chance) | Two of three 1X2 outcomes | Derived from 1X2; no free value, just lower variance |
| **DNB** (empate anula) | Draw ⇒ stake refunded | Equivalent to Asian 0 handicap |
| **Over/Under goals** (mais/menos gols) | Total goals vs line | Very liquid; Pinnacle specializes; efficient ([Pinnacle](https://www.pinnacle.com/betting-resources/en/betting-strategy/how-good-are-pinnacles-asian-handicap-markets/zm62p4gdr62ngx8n)) |
| **BTTS** (ambas marcam) | Both teams to score | Popular/recreational; occasionally softer than O/U |
| **Asian handicap** (handicap asiático) | Fractional handicap, no draw | Sharpest market at Pinnacle; low margin; loss rates ~3.5–4% with **little favourite–longshot bias** ([Hegarty & Whelan, *Rev. Behav. Finance*](https://www.karlwhelan.com/Papers/RBF.pdf)) |
| **Correct score** (placar exato) | Exact final score | High-margin, high-variance; Dixon-Coles territory; softer but heavily marginated |
| **HT/FT** (intervalo/final) | Half-time & full-time combo | Derivative, higher margin, longshot-heavy |
| **Player props** (mercados de jogador) | Shots, cards, goalscorer, etc. | Historically **softest** (thin models, less sharp money) — and first to be limited |
| **Lower leagues / women's / youth** | Less-covered competitions | Historically softer pricing, less liquidity ([Kaunitz et al.](https://arxiv.org/abs/1710.02824)) |

**Reality check:** soft spots exist but are **small, transient, and self-defeating** — the moment you exploit them profitably, books limit you (§10), and prices sharpen as more models pile in.

---

## 6) Betting exchanges — the other "market truth"

Exchanges (Betfair, Smarkets, Matchbook) let punters back *and* lay, so prices track the sharpest bookmakers almost instantly; any bookmaker-vs-exchange gap is arbitraged away fast ([Caan Berry](https://caanberry.com/why-is-betfair-exchange-pricing-so-accurate/)).

| Venue | Margin / cost | Note |
|---|---|---|
| **Betfair Exchange** | ~2–5% commission on net winnings (down to ~2% on football) | Very efficient; a valid CLV benchmark alongside Pinnacle |
| **Smarkets / Matchbook** | ~2% commission | Lower headline commission; thinner liquidity on minor markets |
| **Pinnacle** | ~2–3% margin, "winners welcome" | Sharpest bookmaker; standard CLV reference |

Research still finds exchanges are **not** fully semi-strong efficient, and cross-market arbitrage (bookmaker vs exchange) yielded a guaranteed positive return in **~19% of top-5-league matches** in one study ([Inter-Market Arbitrage — Franck, Verbeek & Nüesch 2013](https://www.researchgate.net/publication/228309038_Inter-Market_Arbitrage_in_Betting)) — but such gaps are tiny, fleeting, and limit-prone.

---

## 7) HIGH-ODDS & ACCUMULATORS ("apostas bomba", múltiplas) — the honest math

This is the section the marketing hype ignores. Accumulators (parlays / *múltiplas*) and long-odds singles are **structurally worse bets** for three compounding reasons. This is a **debunk, not a promotion.**

### 7.1 The bookmaker margin compounds multiplicatively

For independent legs, expected returns multiply. If each leg returns on average `(1 − h)` per unit (where `h` is the per-leg margin/hold), an `n`-leg acca returns `(1 − h)ⁿ`, so the **combined hold is `1 − (1 − h)ⁿ`** — it grows fast with legs.

| Per-leg margin `h` | 2 legs | 3 legs | 5 legs | 8 legs |
|---|---|---|---|---|
| **3%** (sharp book) | 5.9% | 8.7% | 14.1% | 21.6% |
| **5%** (typical) | 9.8% | 14.3% | 22.6% | 33.7% |
| **7%** (soft/retail) | 13.5% | 19.6% | 30.4% | 44.0% |

An 8-fold acca on a 5%-margin book gives the house **~34%** expected hold vs **5%** on a single. Real-world state-regulator data agrees: sportsbook **parlay hold runs ~15–25%** (e.g. Illinois 2023: **18.2%** on parlays vs **4.9%** on straights; New Jersey Sept 2024: **24.2%**) vs **~4–5%** on straight singles ([BettorEdge](https://www.bettoredge.com/post/parlay-betting-vs-single-bet-which-is-better), [OddsShopper](https://www.oddsshopper.com/articles/betting-101/same-game-parlay-strategy)).

### 7.2 Favourite–longshot bias makes long odds *worse*, not better

Longshots are **overbet and overpriced**: they return systematically less than favourites. A ~even-money favourite returns ≈85¢/$1 on average, a 30/1 longshot only ≈63¢/$1 ([Pinnacle](https://www.pinnacle.com/betting-resources/en/betting-strategy/what-is-the-favourite-longshot-bias/vun2u32r85ppf4yp), [Sestović — academic](https://www.researchgate.net/publication/320072261_Bookmaker_Margins_and_Favourite-Longshot_Bias_in_Football_Prediction_Markets)). Stacking longshots into an acca compounds this bias on top of the compounding margin.

### 7.3 Variance explodes (the real cost of "big odds")

A 5-leg acca at ~2.0/leg is ~32/1: you lose **~97% of the time** and rely on rare wins. Even a genuine +EV edge is buried under such variance for thousands of bets, so **your bankroll can be ruined long before an edge (if any) shows** — the opposite of what "one apostas bomba changes your life" marketing implies.

### 7.4 Same-game parlays (SGP) & correlated legs

Legs *within one match* are **not independent** (a team covering the handicap is likelier to hit the over). Books price this **correlation tax**; the same SGP pays very different odds across books, and most are −EV by construction ([OddsIndex](https://oddsindex.com/guides/same-game-parlay-correlation), [Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)).

| SGP example (Wizard of Odds) | Value |
|---|---|
| True prob (with correlation) | 18.9% |
| Fair odds | +429 |
| Odds actually offered | +350 (implies 22.2%) |
| **House edge** | **~14.9%** (vs 4–5% single) |
| EV of 3 singles vs 1 SGP (same $30) | −0.7% vs **−15%** ⇒ SGP costs ~7× more |

**Honest verdict on "apostas bomba":** margin compounds, longshots are overpriced, variance is brutal, and SGP correlation is taxed. High-odds multiples are **entertainment with a large expected loss**, not a strategy. The rare pro exception (modelling correlation better than the book) is exactly the market that gets limited first.

---

## 8) "Tipster bomb tips" — why they look good and usually aren't

Paid/free "bomb" tipsters are mostly **survivorship bias and marketing**, not skill.

| Red flag | Why it misleads |
|---|---|
| **Survivorship bias** | Losers quietly delete accounts; only lucky survivors are shown. A Monte-Carlo of *zero-edge* coin-flip tipsters (quit if yield <−5% after 100 tips) left ~56 "survivors" averaging **+5.4% yield** by pure chance ([football-data.co.uk](https://www.football-data.co.uk/blog/survivorship_bias.php)) |
| **Small samples** | "70% strike rate" over 50 tips is noise; edges need 1,000s of bets ([Pinnacle](https://www.pinnacle.com/betting-resources/en/educational/the-tipster-marketplace-and-the-impact-of-survivorship-bias/hgz26fmjeq29gn9g)) |
| **No CLV shown** | Real edge shows as +CLV, not cherry-picked wins |
| **Deleted losers / edited screenshots** | Records faked or curated ([RebelBetting](https://www.rebelbetting.com/your-tipster-is-a-scam), [Honest Betting Reviews](https://www.honestbettingreviews.com/sports-tipster-scams/)) |
| **FOMO + "guaranteed"** | No bet is risk-free; "sure/100%" = scam tell |
| **Multiple picks, sell the winner** | Send opposite tips to different groups; advertise whoever won |

If a tipster truly beat the closing line at scale, they would bet it themselves (and get limited) — not sell it cheaply on Telegram.

---

## 9) Back-testing a selection strategy honestly

Most "profitable systems" are **overfit** artefacts. A back-test only means something if it survives these controls.

| Pitfall | Honest control |
|---|---|
| **Look-ahead / closing-odds leak** | Select on info available *pre-bet*; measure edge vs the **closing** line, don't *trigger* on it ([Predictology](https://www.predictology.co/blog/how-to-avoid-the-biggest-backtesting-pitfalls-in-football-betting/), [Great Bets](https://www.greatbets.co.uk/how-to-backtest-a-sports-betting-strategy-without-overfitting/)) |
| **Overfitting** | Out-of-sample/walk-forward; suspect any >8–10% ROI over a big sample — pros target **~3–8%** ([Great Bets](https://www.greatbets.co.uk/how-to-backtest-a-sports-betting-strategy-without-overfitting/)) |
| **Ignoring margin & the price you'd *get*** | Use realistically *available* odds, not best-of-market you couldn't have taken |
| **No stake caps / infinite bankroll** | Apply flat or fractional-Kelly caps; model drawdown & ruin |
| **Assuming unlimited access** | Model **getting limited/banned** and reduced max stakes (§10) |
| **Survivorship in data** | Include relegated/defunct teams, dead leagues |
| **Multiple-testing** | Testing 100 "systems" guarantees false winners; correct for it |

The Kaunitz et al. strategy is the cautionary tale: profitable in a 10-year closing-odds simulation *and* with real money — until **bookmakers limited their accounts within weeks** and the strategy died ([arXiv 1710.02824](https://arxiv.org/abs/1710.02824), [code](https://github.com/Lisandro79/BeatTheBookie)).

---

## 10) The part nobody advertises: you get limited or banned

A winning method is not enough; you must be *allowed to bet it*. Soft books **limit or "gub"** winners to pennies; estimates suggest tens of thousands of accounts (as many as ~50,000) have been closed/restricted ([ESPN Chalk 2018](https://www.espn.com/chalk/story/_/id/24425026/gambling-bookmakers-growing-us-legal-betting-market-allowed-ban-bettors), [ESPN 2024 — books defend limiting sharps](https://www.espn.com/sports-betting/story/_/id/41231266/espn-sports-betting-news-sportsbooks-defend-practice-limiting-sharp-customers), [Soccerwidow — gubbing](https://www.soccerwidow.com/football-gambling/betting-knowledge/arbitrage/gubbing-explained/)). Pinnacle/exchanges are the exception ("winners welcome") but run the sharpest, hardest-to-beat prices. **This is the structural reason retail edges rarely translate into sustained profit.**

---

## 11) Open-source tools for selection & staking (free)

| Tool | What it does for selection/staking | Free? | URL |
|---|---|---|---|
| **penaltyblog** (Python) | Devig (multiplicative/additive/power/Shin), Dixon-Coles probs, **Kelly** helper | ✅ | [github](https://github.com/martineastwood/penaltyblog) · [docs](https://penaltyblog.readthedocs.io/) |
| **BeatTheBookie** (Kaunitz) | Reference code + odds dataset for the mispricing study | ✅ | [github](https://github.com/Lisandro79/BeatTheBookie) |
| **football-data.co.uk** | Free CSVs w/ opening+closing odds & multiple books — back-test vs the close | ✅ | [data](https://www.football-data.co.uk/data.php) |
| **soccerdata / ScraperFC** | Pull odds, xG, results for feature building | ✅ | [soccerdata](https://github.com/probberechts/soccerdata) · [ScraperFC](https://github.com/oseymour/ScraperFC) |
| **No-vig / EV / Kelly calculators** | Quick devig, EV and stake sizing checks | ✅ | [devig](https://www.sharkbetting.com/blog/devig-explained) · [EV guide](https://oddsindex.com/guides/expected-value-betting-guide) |

---

## 12) Key papers & references (verified)

| Work | Topic | Link |
|---|---|---|
| Kaunitz, Zhong, Kreiner (2017) | Beating bookies with their own odds; then **limited** | [arXiv 1710.02824](https://arxiv.org/abs/1710.02824) |
| Wilkens (2026), *J. Sports Analytics* | Simple xG/Skellam model, calibration, Bundesliga | [SagePub](https://journals.sagepub.com/doi/10.1177/22150218261416681) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5381388) |
| Galekwa et al. (2024) | Systematic review: ML in sports betting; portfolio risk | [arXiv 2410.21484](https://arxiv.org/abs/2410.21484) |
| Hegarty & Whelan (2024), *Rev. Behav. Finance* — *Returns on Complex Bets: Evidence from Asian Handicap Betting on Soccer* | AH returns; finds **little** favourite-longshot bias in AH | [PDF](https://www.karlwhelan.com/Papers/RBF.pdf) |
| Dixon & Coles (1997) | Correlated-Poisson goals model (correct-score/O-U) | [via penaltyblog](https://docs.pena.lt/y/models/dixon_coles.html) |
| Constantinou, Fenton & Neil (*pi-football*, 2012); Constantinou (*Dolores*, 2019) | Bayesian-network + dynamic-ratings match forecasting | [pi-football](https://dl.acm.org/doi/abs/10.1016/j.knosys.2012.07.008) · [Dolores](https://link.springer.com/article/10.1007/s10994-018-5703-7) |
| Foulley (2021) | Verification of football probability forecasts (reliability & discrimination) | [arXiv 2106.14345](https://arxiv.org/pdf/2106.14345) |
| Kelly (1956) / Shin (1993) | Optimal growth staking / devig model | [Kelly](https://en.wikipedia.org/wiki/Kelly_criterion) · [devig-method comparison — Clarke et al. 2017](https://www.researchgate.net/publication/326510904_Adjusting_Bookmakers_Odds_to_Allow_for_Overround) |

---

## 13) Responsible gambling (Jogo Responsável) — MANDATORY

**This page is research & education only, NOT betting advice.** The honest, evidence-based summary:

- **Markets are highly efficient.** Sharp closing odds predict outcomes with r²≈0.997; margins are ~2–3% at sharp books, more at soft ones.
- **Most people lose.** Only **~2–5%** of bettors are profitable long-term; **~95–97% lose money** over time ([SportBotAI](https://www.sportbotai.com/stats/sports-betting-profitability), [Betmana](https://betmana.co.uk/guide/why-bettors-lose-money/)). At −110 you must win **52.38%** just to break even.
- **Edges are rare, small, and perishable.** If real, they get **limited/banned** (§10). "Apostas bomba" / long-odds multiples are **worse**, not better (§7).
- **Never chase, never bet borrowed money, never stake rent/food money.** No system, tip, or AI model guarantees profit. Treat any wagering as paid entertainment with an expected loss — not income.

**Get help / Onde buscar ajuda:**

| Resource | Region | Contact |
|---|---|---|
| **BeGambleAware** | 🌍/UK | [begambleaware.org](https://www.begambleaware.org/) |
| **GamCare** | 🌍/UK | [gamcare.org.uk](https://www.gamcare.org.uk/) |
| **Gambling Therapy** | 🌍 multilingual | [gamblingtherapy.org](https://www.gamblingtherapy.org/) |
| **Gamblers Anonymous** | 🌍 | [gamblersanonymous.org](https://www.gamblersanonymous.org/) |
| 🇧🇷 **Jogo Responsável (SPA/MF)** | Brazil (regulator, Lei 14.790/2023) | [gov.br/fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) |
| 🇧🇷 **CVV** (apoio emocional 24h, gratuito) | Brazil | ligue **188** · [cvv.org.br](https://www.cvv.org.br/) |
| 🇧🇷 **Jogadores Anônimos** | Brazil (grupos 12 passos) | [jogadoresanonimos.com.br](https://jogadoresanonimos.com.br/) |
| 🇧🇷 **CAPS (SUS)** | Brazil (saúde pública) | procure a UBS/CAPS mais próxima |

Brazil's Lei **14.790/2023** (regulated from Jan 2025 by the **SPA/MF**) mandates deposit limits, self-exclusion (**SIGAP**), and prominent responsible-gambling information ([Chambers Gaming Law 2025](https://practiceguides.chambers.com/practice-guides/gaming-law-2025/brazil), [ICLG 2026](https://iclg.com/practice-areas/gambling-laws-and-regulations/brazil/)). **Ludopatia (transtorno do jogo)** is a recognized health condition — if betting stops being fun or you cannot stop, seek help now.

---

**Keywords:** football/soccer betting prediction, value betting, expected value (EV), edge, closing line value (CLV), devig/overround, favourite-longshot bias, Kelly criterion, fractional Kelly, risk of ruin, bankroll management, Asian handicap, correct score, same-game parlay correlation, accumulator/parlay math, tipster survivorship bias, account limiting/gubbing, market efficiency, backtesting/overfitting, responsible gambling · **Português:** previsão de futebol, aposta de valor, valor esperado, vantagem/edge, linha de fechamento, margem/overround, viés favorito-azarão, critério de Kelly, Kelly fracionado, risco de ruína, gestão de banca, handicap asiático, placar exato, múltiplas/"apostas bomba", correlação em same-game, viés de sobrevivência de tipsters, limitação de conta, eficiência de mercado, backtesting/sobreajuste, jogo responsável, ludopatia, Lei 14.790, SPA/MF, CVV 188.
