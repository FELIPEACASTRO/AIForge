# Corners, Cards & Secondary-Markets Modeling

> Evidence-based reference on modeling football (soccer / futebol) markets **beyond the match result** — corners (escanteios), cards/discipline (cartões, bookings), fouls (faltas), shots & shots-on-target (chutes / no alvo), offsides (impedimentos), throw-ins (laterais), Both-Teams-To-Score (ambas marcam / BTTS), total-goals bands (faixas de gols) and player props (mercados de jogador). This is a **statistically distinct domain**: the drivers, distributions and data differ from 1X2. **Research & education only (pesquisa e educação — data science / ML), current 2024–2026.**

> ⚠️ **NOT betting advice (NÃO é dica de aposta). Read this first.** Betting markets are highly efficient and **the vast majority of long-term bettors lose money**; secondary markets add their *own* traps — **higher margins, lower stake limits, faster suspension and void/push rules** (see §2). Any "softness" is small, fragile and constrained by liquidity. Nothing here is a tip or a system. Help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) (multilingual) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188** ([cvv.org.br](https://www.cvv.org.br/)).

**Sibling pages (shared machinery — not repeated here):** [Match Prediction Models](./Match_Prediction_Models_and_Techniques.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) (devig, CLV, EV) · [Global Datasets & APIs](./Global_Datasets_and_Data_APIs.md) · [Open-Source Tools](./Open_Source_Tools_and_Libraries.md).

---

## 1) Why secondary markets are a different problem

Match-result (1X2), Asian handicap and Over/Under 2.5 goals attract the most bookmaker modeling, the sharpest lines and the deepest liquidity. Corners, cards, fouls and player props are **derived / lower-attention markets**. That cuts both ways.

| Property | Corners / cards / props | Implication |
|---|---|---|
| **Bookmaker attention** | Lower than 1X2/AH; often algorithmic off team + referee averages | Soft spots can persist longer (specific referee, derby, tactical mismatch) |
| **Count distribution** | Corners ~10/match, cards ~3–5/match: **low counts, over-dispersed** | Poisson is a starting point but usually under-fits the variance (see §8) |
| **Correlation with game state** | A red card, an early goal or a chase for an equaliser reshapes corner/card/foul rates | In-play lines shift on **regime breaks**, not smoothly |
| **Liquidity / limits** | **Lower max stakes**, later posting, faster suspension, higher hold | An edge you can't stake at size is not a business |
| **Settlement rules** | Whole-number lines can **push/void** (e.g. exactly 10 corners on "10.0"); card = booking, red = specific points | Model the *rule*, not just the count |

**Bottom line:** less-watched ≠ inefficient in expectation. Treat any measured edge as an upper bound that shrinks after margin, limits and slippage — the discipline of the [Odds & Value-Betting page](./Odds_Betting_Markets_and_Value_Betting.md) (devig → fair prob → EV → CLV) applies unchanged.

---

## 2) Data — who exposes corners, cards, fouls, shots, offsides

The single biggest blocker for secondary markets is **event-level data**. This table lists **verified** sources and exactly which fields they expose.

| Source | Corners | Cards | Fouls | Shots / SoT | Offsides | Cost | Access |
|---|:--:|:--:|:--:|:--:|:--:|---|---|
| [**Football-Data.co.uk**](https://www.football-data.co.uk/) ([field notes](https://www.football-data.co.uk/notes.txt)) | ✅ HC/AC | ✅ HY/AY/HR/AR | ✅ HF/AF | ✅ HS/AS, **HST/AST** | ✗ | **Free** | Historical CSV, many leagues + closing odds |
| [**API-Football v3**](https://www.api-football.com/documentation-v3) (fixtures *statistics*) | ✅ | ✅ | ✅ | ✅ | ✅ | **Freemium** (~100 req/day free) | JSON REST, live + historical |
| [**StatsBomb Open Data**](https://github.com/statsbomb/open-data) ([statsbombpy](https://github.com/statsbomb/statsbombpy)) | ~ (set-piece play-pattern) | ✅ (Foul Committed / Bad Behaviour) | ✅ | ✅ (event xG) | ✅ | **Free** (open subset) | Event JSON, spatial |
| [**Understat**](https://understat.com/) ([understatapi](https://pypi.org/project/understatapi/)) | ✗ | ~ (player-season) | ✗ | ✅ **shot-level xG** | ✗ | **Free** | 6 leagues since 2014/15 |
| [**Sofascore**](https://scraperfc.readthedocs.io/) / WhoScored via [ScraperFC](https://github.com/oseymour/ScraperFC) / [soccerdata](https://github.com/probberechts/soccerdata) | ✅ | ✅ | ✅ | ✅ | ✅ | **Free lib** | Scrape — **respect each site's ToS** |
| **Opta / Stats Perform, StatsBomb API** | ✅ | ✅ | ✅ | ✅ | ✅ | **Paid** / licensed | Enterprise feeds (most complete) |

Notes: `soccerdata` (Apache-2.0, [probberechts](https://github.com/probberechts/soccerdata)) unifies Club Elo, ESPN, FBref, Football-Data.co.uk, Sofascore, SoFIFA, Understat and WhoScored behind one pandas API. **FBref is no longer usable for these markets:** on 20 Jan 2026 its provider Stats Perform (Opta) terminated the feed and all advanced / *Misc* / *Shooting* stats were removed — FBref now serves only basic results, schedules and squads ([Sports Reference notice](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/)). The R package worldfootballR was also **archived (Sept 2025)** and is unmaintained. Always check licence and local law before scraping.

---

## 3) Corners (escanteios)

Corners correlate with **attacking volume, territorial dominance, wide play and shot count** — signals that pure 1X2 markets discard, which is the analytical appeal. The standard workflow estimates a team's **corners-for** and **corners-against** rate, forms a match expectation, then prices Over/Under, team totals and handicaps from a count distribution.

| Market | Model approach | Data | Note |
|---|---|---|---|
| Total corners O/U | Estimate λ = f(team corners-for, opponent corners-against, home edge); price with Poisson/NB | Football-Data.co.uk (HC/AC), API-Football, Sofascore | Whole lines (10.0) can **push**; half-lines (9.5/10.5) do not |
| Team corners O/U | Separate λ per side; Poisson/NB tail | Same | Favourite chasing the game inflates its corners |
| Corner handicap (Asian) | Difference-of-counts (Skellam-like) on corner λ | Same | Sensitive to game-state assumptions |
| Corners as a *goals* signal | **GAP ratings** using shots **and corners** as inputs | Shot/corner counts | See Wheatcroft below |

- **Wheatcroft (2020), *Int. J. Forecasting* 36(3):916–932** — "A profitable model for predicting the over/under market in football" builds *Generalised Attacking Performance (GAP)* ratings from **shots and corners** (not goals) and reports an average profit of **~0.8% per bet** in the **goals** Over/Under 2.5 market across ten leagues over twelve years ([open PDF, LSE](https://researchonline.lse.ac.uk/103712/) · [RePEc](https://ideas.repec.org/a/eee/intfor/v36y2020i3p916-932.html), DOI 10.1016/j.ijforecast.2019.11.001). It validates corners as a predictive feature, not a get-rich system.
- **Over-dispersion:** empirically corner counts vary more than Poisson allows (bursts in open, end-to-end games). Prefer **Negative Binomial** or a Conway–Maxwell–Poisson tail — [penaltyblog](https://github.com/martineastwood/penaltyblog) and R [goalmodel](https://github.com/opisthokonta/goalmodel) both fit NB / CMP variants.

---

## 4) Cards & discipline (cartões, bookings)

Card totals are driven **more by the referee than by either team**, plus fixture heat (derbies), competition and home advantage. This is the market where a specific, checkable prior (this referee, these teams) is most defensible — and also where bookmakers apply the **tightest limits**.

| Market | Model approach | Data | Note |
|---|---|---|---|
| Total cards O/U | λ from team card rates × **referee multiplier**; Poisson/NB | Football-Data.co.uk (HY/AY/HR/AR), API-Football, Sofascore | Referee shifts card expectation materially — use *per-referee* history |
| Booking points O/U | UK settlement: **10 pts/yellow, 25 pts/red (max 35/player)**; Monte-Carlo the per-player caps | Same + lineups | Second yellow → red counts as 35 total (10+25), not additive beyond |
| Home/away card split | **Bivariate** count model (home & away cards correlated) | Match-level cards | Home teams get **~12% fewer** yellows (referee/crowd effect) |
| Player to be carded | Player foul/booking rate × role (full-back vs winger) × referee | Event data (StatsBomb, Sofascore) | Thin, high-variance, often small max stakes |

- **Philipson (2026), *JRSS-A***: "Yellow fever" fits a **bivariate mean-parameterized Conway–Maxwell–Poisson copula** to **7,203 Big-5 matches (2018/19–2021/22, 171 referees, 129 teams)**, handling under-/over-dispersion and home/away correlation. It finds home teams receive **~12% fewer** yellow cards — an effect that **disappeared when matches were played without crowds** (COVID lockdowns) ([Oxford Academic](https://academic.oup.com/jrsssa/advance-article/doi/10.1093/jrsssa/qnag014/8488960), DOI 10.1093/jrsssa/qnag014). A rigorous template for card-count modeling.
- **Azmat & Yi (2024), arXiv 2401.08718** — *Expected Booking (xB)*: an ensemble ML model estimating the probability a foul results in a yellow card, developed and evaluated on 2022 World Cup data ([arXiv](https://arxiv.org/abs/2401.08718)). Connects fouling *style* to card risk — a player-prop input.
- **Practical priors:** referee card averages circulate widely on tipster/odds sites — treat those narratives skeptically and **rebuild the rates yourself from raw match data** (Football-Data.co.uk HY/AY/HR/AR joined to referee name).

---

## 5) Fouls, shots, offsides, throw-ins ("exotic" props)

The same low-count / over-dispersed logic extends to the thinnest markets. Fouls feed cards; shots feed both goals and SoT props; offsides and throw-ins are almost pure style/tempo signals.

| Market | Model approach | Data | Note |
|---|---|---|---|
| Total fouls O/U | Team fouls-committed / fouls-drawn rates; Poisson/NB; referee leniency | Football-Data.co.uk (HF/AF), API-Football, Sofascore | Strongly correlated with cards — do not treat as independent |
| Total shots / SoT O/U | Team shot-generation & concession rates; NB | Understat (shot-level), API-Football, StatsBomb | SoT ≈ fraction of shots; model the ratio, not just the count |
| Offsides O/U | High line / pressing style; low mean, very over-dispersed | API-Football, Sofascore, StatsBomb | Sparse data, wide relative margins |
| Throw-ins O/U | Possession-in-wide-areas proxy; scarce free public data | Provider/scrape-dependent | Verify data exists before modeling |

Because these markets are so thin, **posting is late, limits are tiny and hold is high**. Model quality rarely converts to realizable profit; treat them primarily as *research on football style*.

---

## 6) BTTS & total-goals bands (ambas marcam / faixas de gols)

BTTS and multi-band totals (0–1, 2–3, 4+) are **functions of the joint goal distribution**, so they inherit everything from the goals model — but with a correlation trap.

| Market | Model approach | Data | Note |
|---|---|---|---|
| BTTS Yes/No | From joint scoreline matrix: `P = 1 − P(H=0) − P(A=0) + P(H=0,A=0)` | Any goals dataset + team λ | **Independent Poisson under-states BTTS** — goals are positively correlated |
| Total-goals bands | Sum scoreline-matrix cells over each band | Same | Bands are just partitions of the same matrix — price them jointly |
| Correct score / O/U 2.5 | Bivariate Poisson / Dixon–Coles / diagonal-inflated | Same | Shared engine; keep one model, derive all markets |

- **Fix the correlation:** use **Dixon–Coles** (low-score adjustment) or **bivariate / diagonal-inflated Poisson**, not two independent Poissons. R [goalmodel](https://github.com/opisthokonta/goalmodel) ships `predict_btts()` / `pbtts()` ([worked example, R-bloggers](https://www.r-bloggers.com/2022/02/calculating-the-probability-for-both-teams-to-score-in-r/)); [footBayes](https://github.com/leoegidi/footBayes) (CRAN, Egidi et al.) fits double/bivariate Poisson, Skellam, diagonal-inflated bivariate Poisson and zero-inflated Skellam in Stan; [penaltyblog](https://github.com/martineastwood/penaltyblog) provides Poisson, bivariate Poisson and Dixon–Coles plus devigging utilities.
- **Consistency check:** BTTS, O/U and correct-score prices must all fall out of *one* scoreline matrix. If your BTTS and O/U 2.5 imply contradictory joint probabilities, the model — not the market — is wrong.

---

## 7) Player props (mercados de jogador)

Player markets are among the least efficient — but also the most **lineup-sensitive** (rotation, minutes, set-piece role) and the **hardest to stake at size**.

| Market | Model approach | Data | Note |
|---|---|---|---|
| Anytime / to-score | Player **xG per 90 × expected minutes**; `P(scores) = 1 − exp(−xG_expected)` | Understat (shot-level xG), StatsBomb | Confirm starting XI & minutes; a benched striker is 0 |
| Shots on target O/U | Player shot & SoT rate per 90 × minutes × opponent; NB | Understat, Sofascore, StatsBomb | **Volume market → more predictable** than finishing/goals |
| Assists / to-assist | Key-pass & xA rate; conditional on team scoring | Understat (xA, key passes), StatsBomb | Very high variance; small samples deceive |
| Player to be carded | Foul/booking rate × referee × role (see §4) | Event data (StatsBomb, Sofascore) | Thin; combine with xB-style foul-risk |

- **Minutes are the dominant uncertainty.** A great per-90 rate is worthless if the player is rotated or subbed at 60'. Model expected minutes explicitly and mark props unstable until lineups confirm.
- Shots-on-target props are typically **more tractable than goalscorer props** because they measure *volume* rather than finishing luck.

---

## 8) Modeling pitfalls specific to secondary markets

- **Over-dispersion:** corners/cards/fouls/offsides almost always vary more than Poisson → use **Negative Binomial** or **Conway–Maxwell–Poisson**; validate with a variance-to-mean check, not just log-loss.
- **Regime breaks, not smooth drift:** red cards, early goals and score-chasing structurally change corner/foul/card rates mid-match — pre-match λ is not valid in-play after such events.
- **Cross-market correlation:** fouls↔cards, shots↔corners↔goals move together; never combine them in parlays as if independent.
- **Settlement mechanics:** whole-line push/void, booking-point caps (35/player), TV-only bookings and dead-heat rules change EV before any model does — encode the exact rule.
- **Thin liquidity:** low limits, late posting and fast suspension mean a "10% edge" may be unstakeable. Track **realizable** stake, not paper edge.
- **Calibration first:** as everywhere, verify that "p=0.6" props occur ~60% of the time (see the calibration section of the [Match Prediction page](./Match_Prediction_Models_and_Techniques.md)) — a well-ranked but miscalibrated model mis-sizes every stake.

---

## 9) Responsible gambling (Jogo Responsável) — read this again

| Resource | Region | Link |
|---|---|---|
| **BeGambleAware** | 🌍/UK | [begambleaware.org](https://www.begambleaware.org/) |
| **GamCare** | 🌍/UK | [gamcare.org.uk](https://www.gamcare.org.uk/) |
| **Gambling Therapy** (multilingual) | 🌍 | [gamblingtherapy.org](https://www.gamblingtherapy.org/) |
| 🇧🇷 **Jogo Responsável (SPA/MF)** — regulador, Lei 14.790/2023 | Brazil | [gov.br/fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) |
| 🇧🇷 **CVV** — apoio emocional 24h, gratuito | Brazil | ligue **188** · [cvv.org.br](https://www.cvv.org.br/) |

Secondary markets look "beatable" because they are less watched — but their **higher margins, tiny limits and push/void rules** claw back most modeled edge, and the variance of props is punishing. Markets are efficient; the vast majority of bettors lose. **Ludopatia (transtorno do jogo)** is a recognised health condition. Use these pages as **market education**, never as encouragement to bet. **This is not betting advice.**

**Sources:** [Football-Data.co.uk notes](https://www.football-data.co.uk/notes.txt) · [API-Football docs](https://www.api-football.com/documentation-v3) · [soccerdata](https://github.com/probberechts/soccerdata) · [StatsBomb open-data](https://github.com/statsbomb/open-data) · [statsbombpy](https://github.com/statsbomb/statsbombpy) · [understatapi](https://pypi.org/project/understatapi/) · [ScraperFC](https://github.com/oseymour/ScraperFC) · [penaltyblog](https://github.com/martineastwood/penaltyblog) · [goalmodel](https://github.com/opisthokonta/goalmodel) · [footBayes](https://github.com/leoegidi/footBayes) · [BTTS in R](https://www.r-bloggers.com/2022/02/calculating-the-probability-for-both-teams-to-score-in-r/) · Wheatcroft 2020 IJF [LSE PDF](https://researchonline.lse.ac.uk/103712/) / [RePEc](https://ideas.repec.org/a/eee/intfor/v36y2020i3p916-932.html) · Philipson 2026 JRSS-A [Oxford](https://academic.oup.com/jrsssa/advance-article/doi/10.1093/jrsssa/qnag014/8488960) · Azmat & Yi 2024 [arXiv 2401.08718](https://arxiv.org/abs/2401.08718) · FBref advanced-stats removal [Sports Reference notice](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/).

**Keywords:** corners modeling, over/under corners, team corners-for/against, cards model, yellow/red cards, booking points, referee card rate, fouls, shots on target, offsides, throw-ins, Poisson, negative binomial, Conway–Maxwell–Poisson, bivariate Poisson, Dixon–Coles, over-dispersion, BTTS both teams to score, total goals bands, player props, anytime goalscorer, expected goals xG, expected booking xB, secondary markets, market efficiency, betting limits, responsible gambling · **Português:** modelagem de escanteios, mais/menos escanteios, cartões, pontos de cartão (booking points), taxa de cartões do árbitro, faltas, chutes no alvo, impedimentos, laterais, Poisson, binomial negativa, superdispersão, Poisson bivariada, Dixon–Coles, ambas marcam (BTTS), faixas de gols, mercados de jogador, marcar a qualquer momento, gols esperados (xG), mercados secundários, eficiência de mercado, limites de aposta, jogo responsável, ludopatia, CVV 188, Lei das Bets 14.790.
