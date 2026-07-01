# Odds Aggregators & Historical Odds Archives

> Worldwide reference on **where to actually get football (soccer / futebol) odds** — both **live odds-comparison aggregators** (comparadores de odds) and the **historical odds archives** that make honest **backtesting** (*backtesting / retroteste*) possible. This is the *sourcing* page: real sites, real CSV/SQLite archives, real scraping repos, and the **data quirks** (opening vs closing, which bookmaker, margin, timestamp alignment) that decide whether a backtest means anything. Odds *theory* (margin/de-vig/CLV/value/Kelly) lives on the sibling page below. **Research & education only (pesquisa e educação — data science / ML), current 2024–2026.**

> ⚠️ **NOT betting advice — NOT a tip, system, or income plan (NÃO é dica, sistema, nem plano de renda).** Odds data exists here so you can *study* markets and *test* models, not to encourage betting. Betting markets are **highly efficient**: from a sample of **397,935** Pinnacle football games the closing line correlated with observed outcomes at **r² ≈ 0.997** ([Trademate Sports](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458)); football-data.co.uk's own study of **87,960** Pinnacle odds pairs found expected vs observed yields on an almost perfect 1:1 line ([football-data.co.uk](https://www.football-data.co.uk/blog/pinnacle_efficiency.php)). **Most bettors lose over time**, and a backtest that "profits" against soft/opening odds usually collapses against the close. If gambling stops being fun or you cannot stop, get help now: [GambleAware](https://www.gambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) (multilingual) · 🇧🇷 **CVV 188** ([cvv.org.br](https://www.cvv.org.br/)) · [Autoexclusão SIGAP](https://www.gov.br/pt-br/servicos/plataforma-centralizada-de-autoexclusao-apostas). See [§6](#6-responsible-gambling-jogo-responsável--mandatory).

**Sibling pages (do not duplicate):** odds theory / de-vig / CLV / value / Kelly → [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) · exchange order-book & historic Betfair → [Betting Exchanges, Trading & Microstructure](./Betting_Exchanges_Trading_and_Microstructure.md) · datasets/APIs → [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

---

## 1) Live odds-comparison aggregators (comparadores de odds)

These show **the same match priced across many books at once** — useful to see line dispersion, dropping odds and where a soft book lags the consensus. Most are **free to read** but scraping them may breach ToS and pages are JS-rendered; treat the free-site ones accordingly.

| Aggregator | Books compared | Coverage | Historical odds? | Free / Paid | URL |
|---|---|---|---|---|---|
| **OddsPortal** | Hundreds, worldwide | Global leagues; dropping-odds, sure-bets & blocked-odds tools | ✅ results + odds archive (opening/closing on match pages) | **Free site** (scrape ⚠️ ToS/JS) | [oddsportal.com](https://www.oddsportal.com/) |
| **Oddspedia** | **176** bookmakers | **70** countries; pre-match refreshed ~15 s, live ~5 s; sure-bets, dropping odds | ➖ (live-focused) | **Free** (incl. widgets) | [oddspedia.com](https://oddspedia.com/) |
| **BetExplorer** | Multiple per region | **1,000+** competitions; results + odds-movement per match | ✅ odds movement on match pages | **Free site** (scrape ⚠️) | [betexplorer.com](https://www.betexplorer.com/) |
| **Oddschecker** | **25+** UK books | UK-centric; football, correct-score/bet-builder markets (since 1999) | ➖ | **Free site** (UK-centric) | [oddschecker.com](https://www.oddschecker.com/) |
| **The Odds API** | **40+** books | 70+ sports; EPL/UCL/La Liga/Serie A/Bundesliga/Ligue 1 + many | ✅ snapshots since **2020** | **Free 500 credits/mo**; paid $30 / $59 / $119 / $249 | [the-odds-api.com](https://the-odds-api.com/) |
| **OddsJam** | **100+** sportsbooks | US-centric + global; props/alt markets | ✅ full historical DB (opening + closing) | **Paid / enterprise** (free trial) | [oddsjam.com/odds-api](https://oddsjam.com/odds-api) |

---

## 2) Historical odds archives (for honest backtesting)

The material that lets you grade a model **against the market**. Prefer archives that carry **both opening and closing** odds and a **timestamp** — without the close you cannot compute CLV (see sibling theory page).

| Archive | What you get | Coverage | Free / Paid | URL |
|---|---|---|---|---|
| **football-data.co.uk** | CSV; up to **10** bookmakers, average & max prices, 1X2 + O/U 2.5 + Asian handicap. **Opening + closing** odds since **2019/20**; Pinnacle closing 1X2 back to **2012/13** | 11 main countries (England, Scotland, Germany, Italy, Spain, France, Netherlands, Belgium, Portugal, Turkey, Greece) + 16 "extra" (🇧🇷 Brazil, 🇦🇷 Argentina, 🇯🇵 Japan, 🇲🇽 Mexico, 🇺🇸 USA, China, Nordics, Russia…); odds back to ~1999/2000 | **Free** | [data.php](https://www.football-data.co.uk/data.php) |
| **Beat The Bookie — Odds Series** (Kaunitz et al.) | Hourly-sampled odds time series from **32 bookmakers**, from **72 h before** kickoff, + final score | **500,000+** matches across **1,005** leagues, **2005-01-01 → 2015-07-30** | **Free** | [Kaggle](https://www.kaggle.com/datasets/austro/beat-the-bookie-worldwide-football-dataset) · [paper arXiv:1710.02824](https://arxiv.org/abs/1710.02824) · [code](https://github.com/Lisandro79/BeatTheBookie) |
| **European Soccer Database** (hugomathien) | SQLite: matches, events, team/player FIFA attributes, **odds from up to 10 providers** | **25,000+** matches, **11** EU countries, seasons **2008–2016** | **Free** | [Kaggle](https://www.kaggle.com/datasets/hugomathien/soccer) |
| **Betfair Historical Data** | Time-stamped **exchange** data: prices, traded volume, BSP, winning status. Free **BASIC** tier = last-traded price/minute, **no volume**; paid **ADVANCED/PRO** add the full price ladder / higher frequency | Global exchange markets (football + more) | **Freemium** (registered Betfair acct) | [historicdata.betfair.com](https://historicdata.betfair.com/) |
| **OddsPortal archive** (via scraper) | Opening/closing 1X2, O/U, AH + odds movement per historical match | Worldwide leagues, many books | **Free site** (scrape ⚠️) | [oddsportal.com](https://www.oddsportal.com/) → §3 |

> **Sharp reference:** treat **Pinnacle** closing odds (low margin, high limits) or the **exchange** last-traded price as the "true probability" benchmark; soft/retail closing odds still carry a wider margin.

---

## 3) Scraping repos & loaders (open-source)

| Tool | Lang / license | What it pulls | URL |
|---|---|---|---|
| **OddsHarvester** | Python + Playwright · MIT | Scrapes **OddsPortal**: upcoming + **historical** odds/results by date/league/market, optional `--odds-history` movement, JSON/CSV out | [github.com/jordantete/OddsHarvester](https://github.com/jordantete/OddsHarvester) |
| **soccerdata** | Python · Apache-2.0 | Loader/scraper for **Football-Data.co.uk**, ClubElo, ESPN, FBref, Sofascore, SoFIFA, Understat, WhoScored | [github.com/probberechts/soccerdata](https://github.com/probberechts/soccerdata) |
| **BeatTheBookie** | MATLAB/PHP/Python | Kaunitz et al. code + SQL odds databases (~880k matches, 912 leagues) to reproduce the paper | [github.com/Lisandro79/BeatTheBookie](https://github.com/Lisandro79/BeatTheBookie) |

> Scraping aggregators may violate their **Terms of Service** and pages change/JS-render; use official APIs (The Odds API, Betfair) or the free CSV/SQLite archives above when you can, and scrape responsibly.

---

## 4) Data quirks that decide whether a backtest is real

| Quirk | Why it matters | What to do |
|---|---|---|
| **Opening vs closing** | Beating *opening/soft* odds usually means you front-ran info the market later priced. The **close** is the efficient benchmark | Store both; grade & report against the **closing** line (CLV) |
| **Which bookmaker** | Sharp (Pinnacle) / exchange prices ≈ true probability; soft books carry a bigger margin and lag | Label the source book; don't mix "max across books" with a single-book close |
| **Margin / overround baked in** | Raw odds are not probabilities — the vig makes implied probs sum to >100% | De-vig before using odds as probabilities (see sibling theory page) |
| **Timestamp alignment** | An "odd" is only meaningful with *when* it was captured relative to kickoff | Keep the capture timestamp; align features to *pre-match* info only (no leakage) |
| **Average vs max odds** | football-data.co.uk carries both `Avg` and `Max`; backtesting on `Max` overstates achievable price | Decide up front which column represents a *realistically obtainable* price |
| **Void / non-runner / limits** | Winning accounts get limited/closed; big stakes move the line | Model stake caps, min odds and slippage — see [Kaunitz et al.](https://arxiv.org/abs/1710.02824) |

---

## 5) Quick guide — which source for what

- **Fastest free baseline:** `football-data.co.uk` CSV (opening + closing, 10 books) — start here.
- **Time-series / line-movement research:** Beat The Bookie (hourly, 72 h pre-game) or scrape OddsPortal via OddsHarvester.
- **Market-implied probability:** Betfair Historical Data (exchange last-traded price / BSP).
- **Live pipelines / many books at once:** The Odds API (free tier) or OddsJam (paid).
- **Reading dispersion by eye:** OddsPortal / Oddspedia / BetExplorer / Oddschecker.

---

## 6) Responsible gambling (Jogo Responsável) — mandatory

Gambling can cause serious financial and mental-health harm. This page exists for **data-science / ML research and education only** and is **not** betting advice, a system, or a tip. Set limits, never chase losses (*não persiga o prejuízo*), and get help if betting stops being fun.

| Region | Resource | Contact / tool |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) (Gordon Moody, charity) | Free online chat & forum, **30+ languages** |
| 🇬🇧 UK | [GambleAware](https://www.gambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇺🇸 USA | [NCPG](https://www.ncpgambling.org/) | Helpline & chat |
| 🇧🇷 Brazil | [SPA/MF — Secretaria de Prêmios e Apostas](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · apoio emocional **CVV 188** ([cvv.org.br](https://www.cvv.org.br/)) | Regulator + 24h emotional-support line |
| 🇧🇷 Brazil — self-exclusion | [Autoexclusão centralizada (SIGAP)](https://www.gov.br/pt-br/servicos/plataforma-centralizada-de-autoexclusao-apostas) | Voluntary; blocks the CPF across **all** SPA-authorised betting sites (fixed or indefinite term) |

**🇧🇷 Regulation:** fixed-odds betting is regulated by the **Secretaria de Prêmios e Apostas (SPA/MF)** under **Leis nº 13.756/2018 e 14.790/2023**; only SPA-authorised operators may run nationally, and responsible-gambling tools (deposit limits, self-exclusion) are mandatory.

---

## 7) Related in AIForge
- [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) · [Betting Exchanges, Trading & Microstructure](./Betting_Exchanges_Trading_and_Microstructure.md) · [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Kaggle Football Datasets & Competitions](./Kaggle_Football_Datasets_and_Competitions.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** oddsportal.com · oddspedia.com · betexplorer.com · oddschecker.com · the-odds-api.com · oddsjam.com/odds-api · football-data.co.uk (data.php · blog/pinnacle_efficiency) · kaggle.com/datasets/austro/beat-the-bookie-worldwide-football-dataset · kaggle.com/datasets/hugomathien/soccer · historicdata.betfair.com · github.com/jordantete/OddsHarvester · github.com/probberechts/soccerdata · github.com/Lisandro79/BeatTheBookie · arxiv.org/abs/1710.02824 (Kaunitz, Zhong & Kreiner 2017) · tradematesports.medium.com · gambleaware.org · gamcare.org.uk · gamblingtherapy.org · ncpgambling.org · cvv.org.br · gov.br/fazenda (SPA/MF) · gov.br (autoexclusão SIGAP)

**Keywords:** odds aggregator, comparador de odds, odds comparison, historical odds, arquivo histórico de odds, opening odds, closing odds, closing line, CLV, odds movement, dropping odds, OddsPortal, Oddspedia, BetExplorer, Oddschecker, The Odds API, OddsJam, football-data.co.uk, Beat The Bookie, Kaunitz, European Soccer Database, Betfair historical data, exchange BSP, OddsHarvester, soccerdata, backtesting, retroteste, Pinnacle sharp odds, bookmaker margin, overround, timestamp alignment, jogo responsável, autoexclusão, SIGAP, CVV 188, odds de apostas, mercado eficiente.
