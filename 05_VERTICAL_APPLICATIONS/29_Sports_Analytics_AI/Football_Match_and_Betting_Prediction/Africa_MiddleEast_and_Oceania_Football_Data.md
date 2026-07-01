# Africa, Middle East & Oceania — Football Data & Markets

> Regional index of **where to get football (futebol) data** for match & betting prediction across the three most under-documented confederations — **CAF (Africa)**, **AFC/West Asia (Middle East)** and **OFC (Oceania)** plus Australia's AFC-affiliated A-League. Real URLs, free-vs-paid marks, honest notes on **data scarcity (escassez de dados)**, current 2024–2026. Completes the global set (see sibling pages for 🇪🇺 Europe, 🌎 Americas, 🌏 Asia+India).

> ⚠️ **Research & education only — not betting advice (não é aconselhamento de apostas).** Betting markets are **highly efficient**: the **closing line** (final pre-match odds) is the single best public predictor of match outcomes, and beating it consistently is extremely hard ([closing-line value analysis](https://joesaumarez.co.uk/sports-betting-market-efficiency-and-the-closing-line)). Thin regional markets (many leagues here) are *less* liquid, so lines can be softer **but** limits are low, margins high, and data is sparse/late — edges are illusory and hard to realise. Most bettors lose. Nothing here is a tip. Help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

---

## 0) Read first — the 2026 data landscape has shifted

Two structural changes reshape this whole region's data supply:

- **FBref lost its Opta/Stats Perform advanced-stats licence on 20 Jan 2026.** Expected goals (xG), progressive passes, shot-creating actions and other Opta-derived metrics were **removed** site-wide; only basic stats (scores, goals, cards, minutes, historical archive) remain. This hits secondary regions hardest, since FBref was often the *only* free xG source for leagues like the Saudi Pro League or A-League ([The IX Sports](https://www.theixsports.com/the-ix-soccer/fbrefs-loss-advanced-stats-womens-soccer-data-accessibility/) · [Liam Henshaw — Where to find football data 2026](https://www.liamhenshaw.com/writing/where-to-find-football-data)). Treat FBref regional pages as **results/schedule** sources now, not xG sources.
- **Gulf money is buying data maturity.** Saudi Arabia's PIF poured billions into the **Saudi Pro League** (SPL); it is now among the most valuable leagues outside Europe's big five (Transfermarkt squad value **≈ €1.08 bn** — second only to Brazil's Série A at **≈ €1.81 bn** among non-European leagues), pulling in official Opta coverage, live APIs and Kaggle interest ([Win Sports — most valuable leagues 2026](https://www.winsportsonline.com/research/football/the-most-valuable-football-leagues-around-the-world-in-2026/)).

**Rule of thumb for this region:** results + odds are widely available; **event/tracking/xG data is scarce or paid** for almost everything except the SPL and the A-League. Build baselines on results + Elo + market values, not on advanced metrics you cannot source.

---

## 1) 🌍 Africa (CAF — Confederation of African Football)

### 1a) Domestic leagues — coverage & sources

| Country | League (local name) | Best free source | Paid/API | Notes |
|---|---|---|---|---|
| 🇪🇬 Egypt | **Egyptian Premier League** (الدوري المصري) | [FootyStats](https://footystats.org/egypt/egyptian-premier-league) · [Sofascore](https://www.sofascore.com/) · [Transfermarkt EGY1](https://www.transfermarkt.com/egyptian-premier-league/startseite/wettbewerb/EGY1) | API-Football, SoccersAPI, [Statorium](https://statorium.com/egypt-football-premier-league-api) | Africa's most valuable league (**≈ €174 m**); Al Ahly & Zamalek dominate |
| 🇿🇦 South Africa | **Betway Premiership** (ex-PSL / DStv Prem) | [FBref comps/52](https://fbref.com/en/comps/52/South-African-Premiership-Stats) · [Sofascore](https://www.sofascore.com/football/tournament/south-africa/premiership/358) · [official PSL](https://www.psl.co.za/) | API-Football, SportMonks | 2nd-most-valuable in Africa (**≈ €156 m**); rebranded Betway 2024/25 |
| 🇲🇦 Morocco | **Botola Pro 1** (البطولة الاحترافية) | [OpenFootball world](https://github.com/openfootball/world) · Sofascore · Transfermarkt | API-Football, SoccersAPI | AFCON 2025 host; rising investment |
| 🇳🇬 Nigeria | **NPFL** (Nigeria Premier Football League) | [OpenFootball world](https://github.com/openfootball/world) · Flashscore · Sofascore | API-Football, SoccersAPI | ⚠️ patchy stats; results reliable, events rare |
| 🇩🇿 Algeria | **Ligue Professionnelle 1** | [OpenFootball world](https://github.com/openfootball/world) · [Sofascore](https://www.sofascore.com/) · Transfermarkt | API-Football | 16 clubs; strong CAF pedigree |
| 🇹🇳 Tunisia | **Ligue Professionnelle 1** | Sofascore · Flashscore · Transfermarkt | API-Football, SoccersAPI | Espérance a CAF regular |
| 🇬🇭 Ghana | **Ghana Premier League** | Sofascore · Flashscore · Transfermarkt | API-Football | ⚠️ scarce advanced data |
| 🇨🇩 DR Congo | **Linafoot Ligue 1** | Sofascore · [Flashscore](https://www.flashscore.com/) · Transfermarkt | API-Football (partial) | TP Mazembe a CAF power; data thin |

**Other CAF leagues** (results-only, via API-Football + Sofascore + Transfermarkt; **no free xG**): 🇦🇴 Angola *Girabola*, 🇿🇲 Zambia *Super League*, 🇿🇼 Zimbabwe *Premier Soccer League*, 🇰🇪 Kenya *Premier League*, 🇹🇿 Tanzania *Ligi Kuu Bara*, 🇨🇮 Côte d'Ivoire *Ligue 1*, 🇸🇳 Senegal *Ligue 1*. Coverage depth drops sharply below the eight leagues tabled above.

> **Data-scarcity reality (realidade da escassez):** football-data.co.uk (the standard free odds/results CSV) covers **none** of these leagues — its "extra" set stops at Argentina, Brazil, China, Japan, Mexico, USA, etc. For African domestic betting/results history you rely on **Sofascore / Transfermarkt scrapes** and **paid APIs**; event/tracking/xG data is essentially unavailable for free.

### 1b) Continental & national-team competitions (well-suited to modelling)

International/tournament data is **denser and cleaner** than club data here — use it.

| Competition | Free datasets / sources | Free/Paid |
|---|---|---|
| **AFCON** (Africa Cup of Nations / CAN) | Kaggle [mohammedessam97/africa-cup-of-nations](https://www.kaggle.com/datasets/mohammedessam97/africa-cup-of-nations) · [martj42 international results 1872–2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | **Free** |
| **African national teams 2010–2024** (8,000+ results) | Kaggle [oussamalariouch/african-national-football-from-2010-2024](https://www.kaggle.com/datasets/oussamalariouch/african-national-football-from-2010-2024) | **Free** |
| **CAF Champions League / Confed Cup** | [CAF official](https://www.cafonline.com/) · Sofascore · Transfermarkt · API-Football | Free site / Paid API |
| **AFCON live/odds pipeline** | [SportMonks AFCON API](https://www.sportmonks.com/football-api/africa-cup-api/) · [Statorium AFCON API](https://statorium.com/afcon-api) · API-Football | **Paid / freemium** |

> 🇧🇷 **Brazil relevance:** AFCON 2025 (hosted by Morocco, played Dec 2025–Jan 2026) and CAF sides increasingly overlap with the Brasileirão transfer market — many African players pass through 🇧🇷 clubs, so Transfermarkt cross-league value data (valores de mercado) is a useful bridge feature.

---

## 2) 🕌 Middle East (AFC — West Asia) + Gulf investment story

The Gulf is the region's **data bright spot**: heavy investment has pulled in official providers and public interest, so the SPL now rivals mid-tier European leagues for availability.

| Country | League (local name) | Best free source | Paid/API | Data maturity |
|---|---|---|---|---|
| 🇸🇦 Saudi Arabia | **Saudi Pro League / Roshn SPL** (دوري روشن) | [FBref comps/70](https://fbref.com/en/comps/70/Saudi-Pro-League-Stats) (results, post-Opta) · [FotMob](https://www.fotmob.com/leagues/536/stats/saudi-pro-league) · [Transfermarkt SA1](https://www.transfermarkt.com/saudi-pro-league/startseite/wettbewerb/SA1) | API-Football, SportMonks, Sportradar/Opta | **High** — €1.08 bn squad value; widest coverage in region |
| 🇶🇦 Qatar | **Qatar Stars League** (دوري نجوم قطر) | [official QSL](https://www.qsl.qa/en) · Sofascore · Transfermarkt | API-Football, SoccersAPI | Medium; post-2022 WC infra |
| 🇦🇪 UAE | **ADNOC Pro League** (دوري أدنوك) | [official UAE Pro League](https://www.uaeproleague.ae/en) · Sofascore · Transfermarkt | API-Football, SoccersAPI | Medium |
| 🇮🇷 Iran | **Persian Gulf Pro League** (لیگ برتر) | [Sofascore](https://www.sofascore.com/football/tournament/iran/persian-gulf-pro-league/915) · [Global Sports Archive](https://globalsportsarchive.com/) · Transfermarkt | API-Football | Medium; big crowds, thin advanced data |
| 🇮🇱 Israel | **Ligat ha'Al** (ליגת העל) | [OpenFootball world](https://github.com/openfootball/world) · [OddsPortal odds](https://www.oddsportal.com/football/israel/ligat-ha-al/) · [BetExplorer](https://www.betexplorer.com/football/israel/ligat-ha-al/) | API-Football | Medium; **odds history via OddsPortal/BetExplorer** |
| 🇪🇬 Egypt | *(see Africa §1 — CAF member, overlaps West-Asian club markets)* | — | — | — |

> **Gulf investment & data growth (crescimento de dados):** the SPL's official media/data partner ecosystem (Opta/Stats Perform, Sportradar) means live event feeds and pre-match odds exist commercially; free xG is now gone with the FBref–Opta split. For the SPL specifically there is unusual **open interest on Kaggle** (below) — a rare bright spot for a non-European league.

**Saudi Pro League open datasets (rare for the region):**

| Dataset / repo | Content | Free/Paid | URL |
|---|---|---|---|
| **SPL Stats — Last 4 Seasons** | Match/team stats | **Free** | [kaggle/saudidata2030](https://www.kaggle.com/datasets/saudidata2030/spl-stats-last-4-years) |
| **Saudi Professional League Datasets** | Results since 2000 (home/away, scores) | **Free (OSS)** | [github/alioh](https://github.com/alioh/Saudi-Professional-League-Datasets) |
| **Saudi Pro League Transfers** | Transfers in/out since 2000 | **Free** | [kaggle/rossi14](https://www.kaggle.com/datasets/rossi14/saudi-pro-league-transfers) |

---

## 3) 🌊 Oceania (OFC) + 🇦🇺 Australia (AFC A-League — the region's data flagship)

Australia's **A-League Men** is AFC-affiliated but geographically Oceania and is **by far the best-documented competition covered on this page** — even featuring in open tracking data. True OFC (Pacific) football, by contrast, is the most data-scarce football on Earth.

| Competition | Best free source | Paid/API | Data maturity |
|---|---|---|---|
| 🇦🇺 **A-League Men** (Isuzu UTE A-League) | [FBref comps/65](https://fbref.com/en/comps/65/A-League-Men-Stats) · [official aleagues.com.au/stats](https://aleagues.com.au/stats/) · Sofascore · [FootyStats](https://footystats.org/australia/a-league) | API-Football, SportMonks, Opta/Stats Perform | **High** — official Opta partner (see below) |
| 🇦🇺 **A-League — open tracking data** ⭐ | [SkillCorner Open Data](https://github.com/SkillCorner/opendata): **10 matches** broadcast tracking (2024/25) + season-level physical/off-ball-run/passing aggregates | **Free (attribution)** | **Rare** — only open tracking set for the region |
| 🇳🇿 **New Zealand National League** | Sofascore · Transfermarkt · [NZ Football](https://www.nzfootball.co.nz/) | API-Football (partial) | Low–medium |
| 🌏 **OFC Professional League** (NEW, inaugural **2026**) | [OFC / Oceania Football](https://www.oceaniafootball.com/ofc-pro-league-welcome-to-new-zealand/) · [Wikipedia 2026 OFC Pro League](https://en.wikipedia.org/wiki/2026_OFC_Professional_League) · Sofascore (partial) | — | ⚠️ brand-new; data forming |
| 🌏 **OFC Champions League** (club) | [OFC official](https://www.oceaniafootball.com/) · Sofascore (partial) | — | ⚠️ very scarce; **no reliable free odds history** |
| 🌏 **OFC Nations Cup** (national teams) | [martj42 international results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | **Free** | Low |

> **A-League official data (Stats Perform / Opta):** in Oct 2024 the A-Leagues **extended** their official media-data deal with Stats Perform, adding *Opta Stream* and *Opta Search* ([Stats Perform](https://www.statsperform.com/resource/a-leagues-extends-with-stats-perform/) · [Inside World Football](https://www.insideworldfootball.com/2024/10/15/aussie-league-adds-opta-search-opta-stream-stats-perform-data-deal/)). Combined with the SkillCorner open set, the A-League is the one competition here where **advanced/tracking modelling** is genuinely feasible.

> **OFC scarcity note:** the 2026 **OFC Professional League** is Oceania's first pro league — 8 clubs from 7 nations (incl. Auckland FC), a circuit format across Auckland, Port Moresby, Melbourne, Honiara & Fiji. It is *so new* that structured datasets barely exist; expect only fixtures/results from official + Sofascore for now.

---

## 4) Cross-region data APIs & providers (build pipelines)

The realistic backbone for Africa/ME/OFC coverage. Free tiers are tiny; broad regional coverage is a paid feature.

| Provider | Regional coverage | Free tier | Paid from | URL |
|---|---|---|---|---|
| **API-Football** (api-sports.io) | 1,200+ comps incl. SPL, Qatar, UAE, Iran, A-League, Egypt, South Africa, Ghana, Algeria + AFCON/CAF | ✅ 100 req/day | $19/mo | [api-football.com/coverage](https://www.api-football.com/coverage) |
| **SportMonks** | A-League, AFCON, SPL, many CAF/AFC comps; xG add-on, Predictions API | 🆓 trial (2 leagues) | €29+/mo | [sportmonks.com](https://www.sportmonks.com/football-api/coverage/) |
| **SoccersAPI** | Egypt, Morocco Botola, Nigeria NPFL, Tunisia, Qatar, UAE + more | 🆓 trial | Paid tiers | [soccersapi.com/coverage](https://soccersapi.com/page/coverage) |
| **Statorium** | Egypt PL, AFCON, regional livescore APIs | 🆓 trial | Paid | [statorium.com](https://statorium.com/) |
| **Sportradar** | Official-grade global incl. SPL/A-League (XY events) | 🆓 trial keys | Enterprise | [developer.sportradar.com](https://developer.sportradar.com/soccer/reference/soccer-api-overview) |
| **Stats Perform / Opta** | Official partner of A-League + SPL ecosystem | ➖ | Enterprise | [statsperform.com/opta-data](https://www.statsperform.com/opta-data/) |
| **football-data.org** | ⚠️ Mostly EU + 🇧🇷 Brasileirão + a few global comps; **little/no** Africa/ME/OFC *domestic* coverage | ✅ 10 calls/min | €49+/mo | [football-data.org/coverage](https://www.football-data.org/coverage) |

---

## 5) Free scraping toolchain (works for these regions via Sofascore/Transfermarkt)

Because FBref advanced stats are gone and football-data.co.uk skips the region, **Sofascore** (global live/results/basic stats) and **Transfermarkt** (values/transfers) are the practical free backbones — reach them with:

| Tool | Lang | Sources it unifies (regional-relevant) | URL |
|---|---|---|---|
| **soccerdata** | Python | Sofascore, FBref, ClubElo, SoFIFA, Understat, WhoScored, football-data.co.uk (⚠️ no Transfermarkt) | [github/probberechts](https://github.com/probberechts/soccerdata) |
| **ScraperFC** | Python | Sofascore, FBref, Transfermarkt, ClubElo, Understat | [github/oseymour](https://github.com/oseymour/ScraperFC) |
| **worldfootballR** ⚠️ *archived Sep 2025 (read-only, unmaintained)* | R | FBref, Transfermarkt, Understat | [github/JaseZiv](https://github.com/JaseZiv/worldfootballR) |
| **OpenFootball (world)** | data | Public-domain results: Egypt, Morocco, Algeria, Nigeria, Israel, Australia + more | [github/openfootball/world](https://github.com/openfootball/world) |

> ⚠️ **Scraping ethics/ToS:** Sofascore & Transfermarkt have no official public API; scrape gently, cache, and respect each site's Terms. For public-domain data with **no key and no ToS friction**, prefer OpenFootball.

---

## 6) Reference / meta features that DO cover the region

When league-level advanced data is missing, lean on globally-computed features:

| Source | What | Regional value | Free/Paid | URL |
|---|---|---|---|---|
| **World Football Elo Ratings** | National-team Elo (all CAF/AFC/OFC sides) | Strong prior for AFCON, Asian Cup, OFC Nations Cup | **Free** | [eloratings.net](https://eloratings.net/) |
| **Transfermarkt** | Squad **market values (valores de mercado)**, transfers | Best cross-league strength proxy where xG is absent (Egypt EGY1, SPL SA1, etc.) | **Free** (scrape) | [transfermarkt.com](https://www.transfermarkt.com/) |
| **FIFA World Ranking** | Official national-team ranking | Baseline for internationals | **Free** | [FIFA rankings](https://inside.fifa.com/fifa-world-ranking) |
| **martj42 international results** | 47k+ internationals 1872–2026 (incl. AFCON, Asian Cup, OFC) | The canonical free international dataset | **Free** | [kaggle/martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |

---

## 7) Official governing bodies & national FAs (primary sources)

Confederation and federation sites are the **ground-truth** for fixtures, results and rule/format changes — invaluable when third-party feeds are patchy.

| Body | Scope | URL |
|---|---|---|
| **CAF** | Africa — AFCON, CAF Champions League, Confederation Cup | [cafonline.com](https://www.cafonline.com/) |
| **AFC** | Asia/West Asia — Asian Cup, AFC Champions League Elite | [the-afc.com](https://www.the-afc.com/) |
| **OFC** | Oceania — OFC Pro League, Champions League, Nations Cup | [oceaniafootball.com](https://www.oceaniafootball.com/) |
| 🇿🇦 PSL | South Africa Betway Premiership fixtures/logs | [psl.co.za](https://www.psl.co.za/) |
| 🇶🇦 QSL | Qatar Stars League | [qsl.qa](https://www.qsl.qa/en) |
| 🇦🇪 UAEPL | ADNOC Pro League | [uaeproleague.ae](https://www.uaeproleague.ae/en) |
| 🇦🇺 A-Leagues | Australia official stats hub | [aleagues.com.au/stats](https://aleagues.com.au/stats/) |

---

## 8) A pragmatic modelling recipe for data-scarce leagues

When advanced data is missing (most of this page), a defensible baseline stack is:

1. **Ratings prior** — team/national Elo (ClubElo where available; [eloratings.net](https://eloratings.net/) for internationals) as the strongest single feature.
2. **Goals model** — Poisson / Dixon-Coles on results history (see the models & OSS pages in this section — penaltyblog, soccerdata).
3. **Strength proxy** — Transfermarkt **market values** (valores de mercado) to substitute for absent xG/quality signals.
4. **Home advantage** — fit per-league; it varies widely across CAF/OFC venues and travel-heavy circuits (e.g. the 2026 OFC Pro League).
5. **Validate honestly** — walk-forward back-test and measure **Closing Line Value**, not just accuracy; expect wide confidence intervals given small samples.

> Do **not** import feature pipelines built for the EPL/top-5 (xG, xT, pressing) here — those inputs simply do not exist for most of these leagues, and imputing them invents signal.

---

## 9) Reality check — why thin regional markets are traps, not opportunities

- **"Soft" lines ≠ beatable.** Lower-liquidity leagues (most of this page) look inefficient, but sharp books **cap stakes low** and **limit/close winning accounts fast** (the Kaunitz et al. lesson — a model beat *published* odds in back-test yet was killed by limits: [arXiv 1710.02824](https://arxiv.org/abs/1710.02824)).
- **Data is sparse, late and error-prone.** Missing xG (post-FBref/Opta split), delayed lineups, unreliable injury news and inconsistent results feeds inject model error precisely where you'd hope for edge.
- **Margins are higher** on obscure leagues, so the overround (over-round / *margem da casa*) you must overcome is larger than in the EPL.
- Use these datasets to **practise modelling and validation** (Elo baselines, Poisson on results, market-value features), measure yourself against **Closing Line Value** ([Pinnacle CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)), and treat it as **learning, not income**.

---

## 10) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious harm. This index exists for **data science / ML research and education only**. Set limits, never chase losses (*nunca persiga prejuízos*), and seek help if betting stops being fun.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇿🇦 South Africa | [SA Responsible Gambling Foundation](https://www.responsiblegambling.org.za/) | Counselling line **0800 006 008** |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) (Portaria SPA/MF nº 1.231/2024) | Apoio emocional **CVV 188** |

---

## Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets](./Kaggle_Football_Datasets_and_Competitions.md) · [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · Parent: [`../`](../) (Sports Analytics AI)

**Sources:** fbref.com (comps 52/65/70) · theixsports.com · liamhenshaw.com · winsportsonline.com · Kaggle (mohammedessam97, oussamalariouch, martj42, saudidata2030, rossi14) · github.com/alioh/Saudi-Professional-League-Datasets · github.com/openfootball/world · github.com/SkillCorner/opendata · statsperform.com · insideworldfootball.com · transfermarkt.com (EGY1, SA1) · footystats.org · sofascore.com · oddsportal.com · betexplorer.com · api-football.com/coverage · sportmonks.com · soccersapi.com · statorium.com · developer.sportradar.com · football-data.org · psl.co.za · qsl.qa · uaeproleague.ae · aleagues.com.au · oceaniafootball.com · en.wikipedia.org/wiki/2026_OFC_Professional_League · cafonline.com · eloratings.net · github.com/probberechts/soccerdata · github.com/oseymour/ScraperFC · github.com/JaseZiv/worldfootballR · arxiv.org/abs/1710.02824 · pinnacle.com · joesaumarez.co.uk · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · responsiblegambling.org.za · gov.br/fazenda (SPA)

**Keywords:** African football data, CAF datasets, AFCON dataset (Copa Africana de Nações), Saudi Pro League data, Gulf football analytics, Qatar Stars League, UAE Pro League, Persian Gulf Pro League, Israeli Ligat ha'Al odds, A-League data, SkillCorner tracking data, OFC Professional League, Oceania football, FBref Opta 2026, Transfermarkt market values, API-Football coverage, dados de futebol africano, escassez de dados, previsão de partidas, mercado de apostas eficiente, valores de mercado, jogo responsável, não é aconselhamento de apostas.
