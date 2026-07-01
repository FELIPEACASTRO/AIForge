# Women's & Lower-Tier Football Data — Less-Efficient Markets (Honest View)

> Where to get **women's football (futebol feminino)** and **men's lower/reserve-tier** data for match & betting-prediction research — the segments historically treated as *less-efficient markets*. Every dataset, repo and URL below was verified live. Framed around an **honest tradeoff**, not an easy-money pitch. Free-vs-paid marked. Current for 2026.

> ⚠️ **Research & education only — not betting advice (não é aconselhamento de apostas).** "Less efficient" does **not** mean beatable: the [Kaunitz, Zhong & Kreiner study (arXiv 1710.02824)](https://arxiv.org/abs/1710.02824) beat *published* odds in a five-month live test, yet bookmakers shut it down by **limiting and closing the winning account**. Softer lines in these markets come bundled with **tiny stake limits, high margins, thin/late/patchy data and short samples** — the edge is usually illusory and hard to realise. Most bettors lose. Nothing here is a tip. Help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

---

## 0) Read first — the honest tradeoff (this is the whole point)

Women's and lower/reserve divisions are *less scrutinised* than the men's top five, so lines can be softer. But every apparent advantage carries a matching cost. Model this table before you model a match:

| What looks appealing | What you actually pay for it |
|---|---|
| Softer, less-sharp lines | **Low stake limits**; winning accounts get limited/closed fast (Kaunitz et al.) |
| Fewer pro modellers competing | **Higher margins / overround (margem da casa)** on obscure markets |
| "Untapped" data edge | **Patchier, later, error-prone data** — lineups, injuries and results feeds are unreliable |
| Rapidly improving talent | **Short samples & roster churn** → unstable parameters, wide confidence intervals |

> **The 2026 data shock:** on **20 Jan 2026** FBref lost its Opta/Stats Perform licence, and **expected goals (xG), progressive passes and shot-creating actions were removed site-wide** — described by [The IX Sports](https://www.theixsports.com/the-ix-soccer/fbrefs-loss-advanced-stats-womens-soccer-data-accessibility/) as a *"disaster for women's soccer data"* because FBref had been the only free large-scale xG source for ~10 women's competitions ([Sports-Reference statement](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/)). Treat FBref pages below as **results / basic-stats** sources now. For free *advanced* women's data, StatsBomb open data (§1) is effectively the last one standing.

---

## 1) Women's football — FREE event data (the one genuine bright spot)

[StatsBomb (Hudl) Open Data](https://github.com/statsbomb/open-data) is the highest-value free resource on this page: full **event-level** data (every pass, shot, carry) with heavy women's coverage — the [competitions.json](https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json) confirms **FA Women's Super League, NWSL, UEFA Women's Euro, Frauen-Bundesliga, Liga F, Serie A Women and the Women's World Cup**.

The 2024/25 free release ([Hudl announcement](https://www.hudl.com/blog/statsbomb-free-womens-data-wsl-ligaf-bundesliga-seriea-nwsl)) added **five complete domestic seasons — 771 matches, 62 teams, ~1,500 players**:

| League (local name) | Season released | Free/Paid |
|---|---|---|
| FA Women's Super League (WSL, England) | 2023/24 (+ earlier WSL seasons in the archive) | **Free** |
| Serie A Women (Serie A Femminile, Italy) | 2023/24 | **Free** |
| Frauen-Bundesliga (Germany) | 2023/24 | **Free** |
| Liga F (Spain) | 2023/24 | **Free** |
| NWSL (USA) | 2023 | **Free** |

Pull it with the official packages: [statsbombpy](https://github.com/statsbomb/statsbombpy) (Python) · [StatsBombR](https://github.com/statsbomb/StatsBombR) (R). Attribution to StatsBomb is required.

---

## 2) Women's leagues — results, tables & basic stats

Post-Opta, use these for scores/schedules/basic stats (no free xG). Verified live comp pages:

| Competition | Best free source | Also | Free/Paid |
|---|---|---|---|
| 🏴 FA Women's Super League | [FBref comps/189](https://fbref.com/en/comps/189/Womens-Super-League-Stats) | [Sofascore](https://www.sofascore.com/) | **Free** |
| 🇺🇸 NWSL | [FBref comps/182](https://fbref.com/en/comps/182/NWSL-Stats) | [nwslR ecosystem](https://github.com/nwslR) (nwsldata / nwslR / nwslpy) | **Free** |
| 🇪🇺 UEFA Women's Champions League | [FBref comps/181](https://fbref.com/en/comps/181/UEFA-Womens-Champions-League-Stats) | Sofascore | **Free** |
| 🇩🇪 Frauen-Bundesliga | [FBref comps/183](https://fbref.com/en/comps/183/Frauen-Bundesliga-Stats) | Sofascore | **Free** |
| 🇪🇸 Liga F | [FBref comps/230](https://fbref.com/en/comps/230/Liga-F-Stats) | Sofascore | **Free** |
| 🇧🇷 Brasileiro Feminino Série A1 | [FBref comps/206](https://fbref.com/en/comps/206/Serie-A1-Stats) | [Sofascore (tournament 10257)](https://www.sofascore.com/tournament/football/brazil/brasileiro-women-serie-a1/10257) · [CBF official tables](https://www.cbf.com.br/futebol-brasileiro/tabelas/campeonato-brasileiro/feminino-a1) | **Free** |

> 🇧🇷 **Brazil relevance:** the [CBF](https://www.cbf.com.br/futebol-brasileiro/tabelas/campeonato-brasileiro/feminino-a1) is the ground-truth for Série A1 fixtures/tables (18 clubs in 2026), and Brazil hosts the **FIFA Women's World Cup 2027** — expect coverage and interest to keep rising.

---

## 3) Men's lower, reserve & development tiers

Softer markets, but data thins fast below the professional top divisions.

| Competition | Best free source | Notes | Free/Paid |
|---|---|---|---|
| 🏴 National League (England, 5th tier) | [FBref comps/34](https://fbref.com/en/comps/34/National-League-Stats) | Results/basic stats; odds history in §4 | **Free** |
| 🏴 Premier League 2 (England U21 reserve league) | [FBref comps/852](https://fbref.com/en/comps/852/Premier-League-2-Stats) | Academy/reserve development data | **Free** |
| 🇺🇸 MLS NEXT Pro (MLS reserve/development league) | [official mlsnextpro.com/stats](https://www.mlsnextpro.com/stats/) · [standings](https://www.mlsnextpro.com/standings/) | **No free event dataset** — FBref covers senior [MLS (comps/22)](https://fbref.com/en/comps/22/Major-League-Soccer-Stats), not NEXT Pro; the official site is the source | **Free (official)** |

> **Honest gap:** below the National League (English 6th tier and regional divisions) there is **no free event/xG data** and often no reliable odds history — results-only via aggregators. Do not invent xG you cannot source; build on results + ratings instead.

---

## 4) Odds & results history — football-data.co.uk

The standard free CSV source covers **English tiers down to the National League (5th)** with betting odds:

| Tier | File | Odds included | Free/Paid |
|---|---|---|---|
| Premier League → League Two (E0–E3) | [englandm.php](https://www.football-data.co.uk/englandm.php) | Full set incl. **closing odds** (Bet365 `B365C*`, Pinnacle `PSC*`, market Avg/Max close, closing Asian handicap) + shots/corners/fouls | **Free** |
| National League (5th, EC.csv) | [englandm.php](https://www.football-data.co.uk/englandm.php) | Same **closing-odds** columns; **fewer match stats** (cards only — no shots/corners) | **Free** |

> Closing columns carry a `C` suffix per the [site notes](https://www.football-data.co.uk/notes.txt) (e.g. `B365CH`). No women's or reserve-league CSVs are provided here — for those, scrape Sofascore or use §1 event data.

---

## 5) Free tooling (works for these segments)

| Tool | Lang | Relevant coverage | URL |
|---|---|---|---|
| **statsbombpy** | Python | StatsBomb open + API event data (all women's comps in §1) | [github/statsbomb](https://github.com/statsbomb/statsbombpy) |
| **StatsBombR** | R | Same, for R | [github/statsbomb](https://github.com/statsbomb/StatsBombR) |
| **soccerdata** | Python | FBref, Sofascore, Understat, ClubElo, football-data.co.uk, WhoScored (v1.9.0, 2026) | [github/probberechts](https://github.com/probberechts/soccerdata) |
| **nwslR / nwslpy** | R / Python | NWSL datasets + access functions | [github/nwslR](https://github.com/nwslR) |
| **penaltyblog** | Python | Poisson / Dixon-Coles / Elo, implied-prob & odds tools (v1.11.0, 2026) | [github/martineastwood](https://github.com/martineastwood/penaltyblog) |
| **worldfootballR** ⚠️ *archived 18 Sep 2025 (read-only, unmaintained)* | R | FBref / Transfermarkt / Understat | [github/JaseZiv](https://github.com/JaseZiv/worldfootballR) |

> ⚠️ Scraping ethics/ToS: Sofascore has no official public API — scrape gently, cache, respect each site's Terms. FBref advanced-stat functions in these tools now return basic data only (post-20-Jan-2026).

---

## 6) Reality check — why "less efficient" is a trap, not a shortcut

- **Soft lines ≠ beatable.** The [Kaunitz et al. experiment](https://arxiv.org/abs/1710.02824) beat published odds in back-test *and* live play, then got limited/closed — the standard fate of winners in thin markets.
- **Data is the bottleneck.** With free women's xG gone (post-FBref/Opta split) and lower-tier feeds sparse and late, model error is largest exactly where you hoped for edge.
- **Small samples lie.** Short women's-league histories and reserve-team roster churn produce unstable estimates — expect wide confidence intervals.
- **Use these datasets to *learn*.** Fit Elo + Poisson/Dixon-Coles baselines on results, validate walk-forward against **Closing Line Value**, and treat it as skill-building, **not income**.

---

## 7) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious harm. This page exists for **data-science / ML research and education only**. Set limits, never chase losses (*nunca persiga prejuízos*), and seek help if betting stops being fun.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) | Apoio emocional **CVV 188** |

---

## Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Africa, Middle East & Oceania Data](./Africa_MiddleEast_and_Oceania_Football_Data.md) · [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · Parent: [`../`](../) (Sports Analytics AI)

**Sources:** github.com/statsbomb/open-data · raw.githubusercontent.com/statsbomb/open-data (competitions.json) · hudl.com/blog (free women's data) · fbref.com (comps 189/182/181/183/230/206/34/852/22) · theixsports.com · sports-reference.com/blog · sofascore.com (tournament 10257) · cbf.com.br (feminino-a1) · mlsnextpro.com/stats · football-data.co.uk (englandm.php, notes.txt, EC.csv) · github.com/statsbomb/statsbombpy · github.com/statsbomb/StatsBombR · github.com/probberechts/soccerdata · github.com/nwslR · github.com/martineastwood/penaltyblog · github.com/JaseZiv/worldfootballR · arxiv.org/abs/1710.02824 · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · gov.br/fazenda (SPA)

**Keywords:** women's football data, futebol feminino dados, StatsBomb open data, FA WSL dataset, NWSL data, Liga F, Frauen-Bundesliga, Serie A Women, Brasileiro Feminino Série A1, UEFA Women's Champions League, lower-tier football data, National League fifth tier, Premier League 2 U21, MLS NEXT Pro, reserve league data, football-data.co.uk closing odds, FBref Opta 2026, less-efficient betting markets, mercados menos eficientes, previsão de partidas, valor esperado, jogo responsável, não é aconselhamento de apostas.
