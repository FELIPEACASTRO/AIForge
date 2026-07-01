# Referee Analytics & Discipline Data

> Evidence-based reference on the **match official (árbitro)** as a *potential predictive factor* in football (soccer / futebol): where to get referee data, which tendencies are actually measurable (cards / penalties / home bias / added time), how they relate to **discipline markets** (cartões, pênaltis), and what the peer-reviewed literature really shows about referee bias and VAR. **Research & education only (pesquisa e educação — data science / ML), current 2024–2026.**

> ⚠️ **NOT betting advice (NÃO é dica de aposta). Read this first.** Betting markets are highly efficient and **the vast majority of long-term bettors lose money**. Any referee "edge" is **small, late-arriving and largely priced in** — bookmakers already build card lines partly off referee averages, so residual signal is thin, and card/penalty markets carry **higher margins and tighter stake limits** than 1X2 (see the [Corners, Cards & Secondary-Markets page](./Corners_Cards_and_Secondary_Markets_Modeling.md)). Nothing here is a tip or a system. Help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) (multilingual) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188** ([cvv.org.br](https://www.cvv.org.br/)).

**Sibling pages (shared machinery — not repeated here):** [Corners, Cards & Secondary Markets](./Corners_Cards_and_Secondary_Markets_Modeling.md) (card/booking-point modeling) · [Match Prediction Models](./Match_Prediction_Models_and_Techniques.md) · [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) (devig, EV, CLV) · [Global Datasets & APIs](./Global_Datasets_and_Data_APIs.md) · [Features & Feature Engineering](./Features_and_Feature_Engineering.md).

---

## 1) Why the referee is a distinct — but modest and late — feature

The referee matters most for **discipline** (cards, fouls, added time) and marginally for **penalties**, far less for the match result itself. Three properties make it a weak feature to build a strategy on:

- **Late-arriving.** The officiating referee is usually confirmed only 1–3 days before kickoff (via the appointment sources in §3), so the signal cannot be used until close to the market.
- **Largely priced in.** Referee card/booking averages are public and bookmakers already fold them into card and booking-point lines. Residual mispricing, if any, is small.
- **Confounded.** A raw "home teams get fewer cards" gap conflates referee behaviour with *within-game events* — teams that are losing chase the game and foul more. Failing to control for score-state overstates "bias" (see Buraimo et al. in §5).

Treat referee data as **one modest, well-documented feature** — useful for discipline-market research and calibration, not an edge on its own.

---

## 2) Referee data sources (verified)

| Source | Referee field | Discipline data | Cost | Access |
|---|:--:|---|---|---|
| [**Football-Data.co.uk**](https://www.football-data.co.uk/) ([field notes](https://www.football-data.co.uk/notes.txt)) | ✅ `Referee` (name) | ✅ `HY/AY/HR/AR` cards, `HBP/ABP` booking points, `HF/AF` fouls | **Free** | Historical CSV, many leagues, closing odds |
| [**API-Football v3**](https://www.api-football.com/documentation-v3) ([docs](https://api-sports.io/documentation/football/v3)) | ✅ referee name in `fixtures` | ✅ cards/fouls via fixture *statistics* & *events* | **Freemium** (~100 req/day free) | JSON REST, live + historical |
| [**Transfermarkt — Referees**](https://www.transfermarkt.com/premier-league/schiedsrichter/pokalwettbewerb/GB1) | ✅ per-referee profiles | ✅ matches, yellows, second yellows, reds, **penalties** per referee, sortable by competition | **Free** (web) | Human-readable; check ToS before scraping |
| [**WorldReferee.com**](https://worldreferee.com/) | ✅ 9,000+ referee profiles | ✅ individual match records, yellow/red data, career stats across all six FIFA confederations | **Free** (web) | Reference DB since 2005 |
| [**soccerdata**](https://github.com/probberechts/soccerdata) / [**ScraperFC**](https://github.com/oseymour/ScraperFC) | ✅ (Sofascore / WhoScored referee fields) | ✅ match-level cards/fouls | **Free lib** | Scrape — **respect each site's ToS** and local law |

Notes:
- **Football-Data.co.uk** is the simplest reproducible starting point: the `Referee` column plus `HY/AY/HR/AR` (cards) and `HF/AF` (fouls) let you rebuild per-referee card and foul rates yourself instead of trusting tipster averages ([field key](https://www.football-data.co.uk/notes.txt): `Referee = Match Referee`; `HBP/ABP = Booking Points, 10 = yellow, 25 = red`).
- **FBref** match pages have historically listed the officiating referee, but on **20 Jan 2026** its provider Stats Perform (Opta) terminated the advanced-data feed and all advanced/*Misc*/*Shooting* stats were removed; FBref now serves only basic results, schedules and squads ([Sports Reference notice](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/)). Treat it as basic-only.

---

## 3) Where to get the *appointment* (who referees a given match, in advance)

| League / body | Official appointment source | Notes |
|---|---|---|
| England (PL) | [**Premier League — Referees & PGMOL**](https://www.premierleague.com/en/referees) | Weekly match-official lists (referee, ARs, 4th, VAR) |
| England (EFL) | [**EFL — Match Officials**](https://www.efl.com/match-officials/) | Championship / League One / League Two appointments |
| Brazil (CBF) | [**CBF — Relação de Árbitros**](https://www.cbf.com.br/a-cbf/arbitragem/relacao-arbitros) · [**Notícias / Escala de Arbitragem**](https://www.cbf.com.br/a-cbf/arbitragem/noticias-arbitragem) | CBF publishes the round-by-round *escala de arbitragem* per competition |

Appointments typically land **1–3 days before** the fixture — after most pre-match modeling is done — which is exactly why the referee is a late feature (§1).

---

## 4) Measurable tendencies → discipline markets (research framing, not tips)

| Referee tendency (measurable) | Where it shows up | Modeling note |
|---|---|---|
| **Cards per match** (strictness) | Total cards O/U, booking points | Rebuild per-referee from raw `HY/AY/HR/AR`; low counts → use Negative-Binomial / CMP, not plain Poisson (see [Corners, Cards page §8](./Corners_Cards_and_Secondary_Markets_Modeling.md)) |
| **Home/away card split** | Team cards, booking-point split | Home teams generally receive *fewer* cards — but control for score-state before calling it bias (Buraimo et al., §5) |
| **Fouls tolerated before booking** | Fouls O/U, player-to-be-carded | Foul-to-card ratio varies by referee; see *Expected Booking (xB)* in [Corners, Cards page §4](./Corners_Cards_and_Secondary_Markets_Modeling.md) |
| **Penalties awarded rate** | Penalty markets, anytime-penalty | Sparse per-referee samples; several studies find home sides awarded more penalties (§5) — high variance, easy to over-fit |
| **Added-time discretion** | In-play totals near full time | Injury-time favoritism is documented (Garicano et al., §5) but small and hard to monetize |

**Caution:** these are *style descriptions*, not a betting method. Per-referee samples are small (a referee officiates only tens of matches per season), so estimated rates are noisy and shrink toward league means. Always regularize / partially pool referee effects.

---

## 5) What the peer-reviewed literature actually shows

| Study | Finding (as reported) | Reference |
|---|---|---|
| **Garicano, Palacios-Huerta & Prendergast (2005)** | Referees systematically **shorten** close games when the home team is ahead and **lengthen** them when the home team is behind — favoritism driven by crowd pressure | *Rev. Econ. & Statistics* 87(2):208–216 · [MIT Press](https://direct.mit.edu/rest/article-abstract/87/2/208/57539) · [NBER w8376](https://www.nber.org/papers/w8376) |
| **Nevill, Balmer & Williams (2002)** | Qualified referees shown televised challenges **with** crowd noise awarded **fewer fouls against the home team** than those viewing in silence | *Psych. Sport & Exercise* 3:261–272 · DOI [10.1016/S1469-0292(01)00033-4](https://doi.org/10.1016/S1469-0292(01)00033-4) |
| **Buraimo, Forrest & Simmons (2010)** | Home teams get fewer cards, but a **minute-by-minute** model shows much of the raw gap reflects **losing-team aggression / score-state**, not pure bias | *JRSS-A* 173(2):431–449 · DOI [10.1111/j.1467-985X.2009.00604.x](https://doi.org/10.1111/j.1467-985X.2009.00604.x) |
| **Pettersson-Lidbom & Priks (2010)** | Natural experiment on Italian matches forced behind closed doors (**n ≈ 21**): home bias in fouls/cards **weakened** without spectators — suggestive given the small sample | *Economics Letters* 108(2):212–214 · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165176510001497) |
| **Endrich & Gesche (2020)** | Bundesliga COVID "ghost matches": pre-COVID referees gave home teams fewer fouls/yellows; that advantage **shrank** once crowds were absent | *Economics Letters* 197:109621 · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0165176520303815) |
| **Philipson (2026), "Yellow fever"** | Bivariate mean-parameterized **Conway–Maxwell–Poisson copula** on **7,203 Big-5 matches (2018–2022, 171 referees, 129 teams)**: away teams get more yellows; a **~13% reduction in yellow-card rate** in matches behind closed doors essentially removed the home effect | *JRSS-A* (2026) · DOI [10.1093/jrsssa/qnag014](https://academic.oup.com/jrsssa/advance-article/doi/10.1093/jrsssa/qnag014/8488960) |
| **Dohmen & Sauermann (2016)** | Survey of the referee-bias literature: crowd-driven home favoritism (added time, cards, penalties) is a robust, repeatedly documented effect | *J. Economic Surveys* 30(4):679–695 · DOI [10.1111/joes.12106](https://onlinelibrary.wiley.com/doi/abs/10.1111/joes.12106) |

**Takeaway:** crowd-driven home favoritism in discipline is one of the most replicated findings in sports economics — and the COVID ghost-game experiments (Pettersson-Lidbom & Priks; Endrich & Gesche; Philipson) are the cleanest evidence that the *crowd*, not the referee alone, drives much of it. The effects are **statistically real but small**, which is why they are hard to convert into betting profit.

---

## 6) VAR impact

- **Spitz, Wagemans, Memmert, Williams & Helsen (2021)** — reviewing decisions in a professional league, referee **decision accuracy rose from 92.1% to 98.3%** after VAR review; VAR corrects clear errors on reviewable incidents (goals, penalties, red cards, mistaken identity). *J. Sports Sciences* 39(2):147–153 · DOI [10.1080/02640414.2020.1809163](https://doi.org/10.1080/02640414.2020.1809163).
- **Işın & Yi (2024)** — proof-of-concept on the Turkish Super League (**1,838 matches**, with/without VAR): the **only** variable that changed significantly was **fouls** (down for both sides); home advantage in points/goals and away-side disadvantage in yellows/penalties **persisted** with VAR. *BMC Sports Sci. Med. Rehabil.* · DOI [10.1186/s13102-024-00813-9](https://link.springer.com/article/10.1186/s13102-024-00813-9).

Net: VAR improves accuracy on the specific incidents it reviews but does **not** eliminate crowd-driven home advantage in discipline. For modeling, VAR mainly adds noise/stoppages and shifts penalty conversion timing — treat pre-/post-VAR seasons as different regimes.

---

## 7) Modeling pitfalls specific to referee features

- **Small per-referee samples.** Tens of matches per season → noisy rates. **Partially pool** (hierarchical/shrinkage) referee effects toward league means; never trust a raw single-season average.
- **Confounding with score-state.** The home/away card gap is partly losing-team aggression, not bias (Buraimo et al.). Include game-state controls before attributing anything to the referee.
- **Over-dispersion / under-dispersion.** Card counts rarely fit plain Poisson — Philipson finds **under-dispersion** best handled by Conway–Maxwell–Poisson; corners/fouls are usually **over-dispersed** (Negative Binomial). Check variance-to-mean, not just log-loss.
- **Regime breaks.** VAR introduction, empty-stadium periods, and rule changes (added-time crackdowns) shift referee behaviour discontinuously — don't pool across them.
- **Late + priced-in + limited.** Even a correct referee signal arrives late, is partly in the line already, and card/penalty markets have **tight limits and high margins** — realizable edge is far smaller than paper edge.

---

## 8) Responsible gambling (Jogo Responsável) — read this again

| Resource | Region | Link |
|---|---|---|
| **BeGambleAware** | 🌍/UK | [begambleaware.org](https://www.begambleaware.org/) |
| **GamCare** | 🌍/UK | [gamcare.org.uk](https://www.gamcare.org.uk/) |
| **Gambling Therapy** (multilingual) | 🌍 | [gamblingtherapy.org](https://www.gamblingtherapy.org/) |
| 🇧🇷 **Jogo Responsável (SPA/MF)** — regulador, Lei 14.790/2023 | Brazil | [gov.br/fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) |
| 🇧🇷 **CVV** — apoio emocional 24h, gratuito | Brazil | ligue **188** · [cvv.org.br](https://www.cvv.org.br/) |

The referee "angle" looks attractive because the bias is real and well-published — but it is **small, late-arriving, largely priced in**, and lives in **high-margin, tightly-limited** card/penalty markets. Markets are efficient; the vast majority of bettors lose. **Ludopatia (transtorno do jogo)** is a recognised health condition. Use this page as **market education**, never as encouragement to bet. **This is not betting advice.**

**Sources:** [Football-Data.co.uk field notes](https://www.football-data.co.uk/notes.txt) · [API-Football docs](https://www.api-football.com/documentation-v3) · [Transfermarkt referees](https://www.transfermarkt.com/premier-league/schiedsrichter/pokalwettbewerb/GB1) · [WorldReferee](https://worldreferee.com/) · [soccerdata](https://github.com/probberechts/soccerdata) · [ScraperFC](https://github.com/oseymour/ScraperFC) · [FBref advanced-stats removal notice](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/) · [Premier League / PGMOL referees](https://www.premierleague.com/en/referees) · [EFL match officials](https://www.efl.com/match-officials/) · [CBF Relação de Árbitros](https://www.cbf.com.br/a-cbf/arbitragem/relacao-arbitros) · Garicano et al. 2005 [MIT Press](https://direct.mit.edu/rest/article-abstract/87/2/208/57539) · Nevill et al. 2002 [DOI](https://doi.org/10.1016/S1469-0292(01)00033-4) · Buraimo et al. 2010 [DOI](https://doi.org/10.1111/j.1467-985X.2009.00604.x) · Pettersson-Lidbom & Priks 2010 [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165176510001497) · Endrich & Gesche 2020 [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0165176520303815) · Philipson 2026 [JRSS-A](https://academic.oup.com/jrsssa/advance-article/doi/10.1093/jrsssa/qnag014/8488960) · Dohmen & Sauermann 2016 [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/joes.12106) · Spitz et al. 2021 [DOI](https://doi.org/10.1080/02640414.2020.1809163) · Işın & Yi 2024 [Springer](https://link.springer.com/article/10.1186/s13102-024-00813-9).

**Keywords:** referee analytics, match official, referee bias, home advantage, card rate, yellow/red cards, booking points, penalties, added time / injury time, VAR video assistant referee, crowd noise, ghost games, Conway–Maxwell–Poisson, negative binomial, over/under-dispersion, hierarchical shrinkage, discipline markets, football data, PGMOL, CBF escala de arbitragem, responsible gambling · **Português:** análise de arbitragem, árbitro, viés do árbitro, mando de campo, taxa de cartões, cartões amarelos/vermelhos, pontos de cartão, pênaltis, acréscimos, VAR árbitro assistente de vídeo, torcida, jogos com portões fechados, Conway–Maxwell–Poisson, binomial negativa, superdispersão/subdispersão, mercados de disciplina, dados de futebol, escala de arbitragem CBF, jogo responsável, ludopatia, CVV 188, Lei das Bets 14.790.
