# News, Sentiment & Alternative-Data Signals

> Verified index of **NLP / social / alternative-data signals** for football (soccer / futebol) prediction — the layer *outside* the standard results/xG/odds pipeline: news & social text, public-attention and "fade-the-public" proxies, open-source sentiment models, weather, and structural factors (altitude, empty stadiums). Every API, model, paper and site below was checked for existence via live fetch/search this pass; anything that could not be confirmed was removed. Built for **research & education (pesquisa e educação — data science / ML), current 2024–2026.**

> ⚠️ **NOT betting advice (NÃO é dica de aposta). Read this first.** Betting markets are highly efficient and **the vast majority of long-term bettors lose money.** Alternative-data "signals" are the *most* over-hyped and *least* reliable inputs in football modelling: text and attention arrive continuously (easy to leak future information) and are largely **already in the price.** Nothing here is a tip or a system. If gambling stops being fun, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) (multilingual) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 **CVV 188** ([cvv.org.br](https://www.cvv.org.br/)).

**Sibling pages (shared machinery — not repeated here):** [Match Prediction Models](./Match_Prediction_Models_and_Techniques.md) · [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) (devig, EV, CLV) · [Injury, Lineup & Availability](./Injury_Lineup_and_Availability_Data.md) (team-news layer, leakage) · [Referee Analytics & Discipline](./Referee_Analytics_and_Discipline_Data.md) (crowd/ghost-game evidence) · [Global Datasets & APIs](./Global_Datasets_and_Data_APIs.md) · [Features & Feature Engineering](./Features_and_Feature_Engineering.md).

---

## 0) Three honest priors (signal vs noise vs priced-in)

1. **Most of it is priced in.** News, sentiment and public attention are watched by every trader. Measure a signal's value **against the closing line**, never against a naive baseline — if it doesn't beat the close, it isn't an edge.
2. **Text leaks the future if you're careless.** Tweets, articles and weather updates stream continuously. Freeze every feature at a fixed pre-market timestamp `T−k` or your back-test is fantasy (§7).
3. **Event-content ≠ mood.** A tweet spike around a goal describes the *past*; treating fan emotion as forward-looking is the classic alternative-data trap.

---

## 1) News & social text sources

| Source | What it gives | Cost | Access |
|---|---|---|---|
| [**GDELT DOC 2.0 API**](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) | Global news **volume + tone (sentiment)** timelines across 65+ languages | **Free, no key** | `api.gdeltproject.org/api/v2/doc/doc` (JSON) |
| [**Reddit via PRAW**](https://praw.readthedocs.io/) | Subreddit/comment text (e.g. r/soccer, club subs) within Reddit API limits | **Free** (OSS wrapper) | Python lib |
| [**NewsAPI — `/v2/everything`**](https://newsapi.org/docs/endpoints/everything) | Article search over 150k+ sources | **Freemium** (free Developer tier: non-commercial, delayed) | JSON REST |
| [**X / Twitter API v2**](https://developer.x.com/) | Posts for team/match sentiment | **Paid** — pay-per-use for new developers; legacy **Basic ~$200/mo** (≈15k reads/mo) closed to new signups **Feb 2026** | JSON REST |
| [**Google Trends**](https://trends.google.com/trends/) (+ [pytrends](https://github.com/GeneralMills/pytrends)) | Search-attention index per club/fixture | **Free** | Web; **pytrends is unofficial and archived (read-only since Apr 2025)** |
| Official club/league verified channels | Team-news posts (often the *source* the market reacts to) | Free | Web / social |

> **Reality:** X/Twitter access got expensive and unstable after 2023–2026 pricing changes, so **GDELT (free, tone+volume) and Reddit** are the most reproducible starting points for a research pipeline.

---

## 2) Public-attention & "fade-the-public" proxies

- **Google Trends attention** — a cheap pre-match "buzz" proxy (§1). Useful for detecting hype spikes, but hype is not direction.
- **"Fade the public" / reverse line movement (RLM)** — the idea that when tickets pile on one side while the *line* moves the other way, sharp money is on the other side. Soccer split data:
  - [**Action Network — soccer public betting**](https://www.actionnetwork.com/soccer/public-betting): percentage of **bets (tickets)** vs **money (dollars)** with a **DIFF** column — the standard bets-vs-money split view.
  - [**OddsShark — soccer consensus**](https://www.oddsshark.com/soccer/epl/consensus-picks): public-consensus percentages across EPL, Champions League, La Liga, Ligue 1, MLS, World Cup and [**Brazil Série A**](https://www.oddsshark.com/soccer/bra-serie-a/consensus-picks).

> **Honest caveat:** soccer bets-vs-money data is **thinner and more US-sportsbook-centric** than for NFL/NBA, three-way markets (the draw) complicate two-way "splits", and the contrarian edge is **mixed and largely priced**. Treat RLM as a hypothesis to test against the closing line, not a rule.

---

## 3) Open-source sentiment / NLP models

| Model / tool | Task | License | Link |
|---|---|---|---|
| [**cardiffnlp/twitter-roberta-base-sentiment-latest**](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | English tweet sentiment (RoBERTa, ~124M tweets) | OSS | HF |
| [**cardiffnlp/twitter-xlm-roberta-base-sentiment**](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | **Multilingual** tweet sentiment — 8 languages incl. **Portuguese** | OSS | HF |
| [**TweetNLP**](https://github.com/cardiffnlp/tweetnlp) ([tweetnlp.org](https://tweetnlp.org/)) | Python lib: sentiment / emotion / NER for social media (Cardiff NLP) | MIT | GitHub |
| [**ProsusAI/finbert**](https://huggingface.co/ProsusAI/finbert) | Financial-domain sentiment — useful *transfer template* for odds/markets text | OSS | HF |
| [**VADER** (cjhutto/vaderSentiment)](https://github.com/cjhutto/vaderSentiment) | Lexicon/rule-based social sentiment (Hutto & Gilbert, 2014) | MIT | GitHub |
| [**liamhbyrne/twitter-football-prediction**](https://github.com/liamhbyrne/twitter-football-prediction) | Worked example: tweet sentiment → match events; author's own conclusion is that sentiment alone **cannot** predict final outcomes | OSS | GitHub |

> For Portuguese/Spanish-language football audiences, the **XLM-R multilingual** model is the right default; the English-only RoBERTa model silently mis-scores non-English tweets.

---

## 4) What the academic evidence actually shows

| Study | Finding (as reported) | Reference |
|---|---|---|
| **Kampakis & Adamides (2014)** | Twitter **volume + sentiment + user picks** over EPL beat chance and were *comparable* to simple historical-stats models; combining the two helped modestly | *"Using Twitter to predict football outcomes"* · [arXiv:1411.1243](https://arxiv.org/abs/1411.1243) |
| **Godin, Zuallaert et al. (2014)** | Fusing statistics with Twitter reportedly returned **~30% profit** on the 2nd half of EPL 2013/14 — **caution: a single-season backtest, not out-of-sample** | *"Beating the Bookmakers…"* · KDD Workshop · [Semantic Scholar](https://www.semanticscholar.org/paper/Beating-the-bookmakers:-leveraging-statistics-and-Godin-Zuallaert/9a82fc5842dea2afb343e12fa34d5892bf4b19d7) |
| **Schumaker, Jarmoszko & Labedz (2016)** | Tweet-sentiment wagering gave **higher payout but lower accuracy** than betting odds-favourites — a non-favourite trade-off, not free money | *Decision Support Systems* 88:76–84 · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167923616300835) |
| **Wunderlich & Memmert (2021)** | ~2M tweets over 400+ PL matches: **in-play tweet information does NOT improve goal forecasting** (negative result), confirmed on 30k+ matches | *Soc. Netw. Anal. Min.* · DOI [10.1007/s13278-021-00842-z](https://link.springer.com/article/10.1007/s13278-021-00842-z) · [PMC8714875](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8714875/) |

**Verdict:** pre-match text is a **weak, mostly-priced-in** feature; in-play tweet sentiment adds little over the score and clock. Positive backtests tend to be single-season and fragile. Treat these as features to *test against the close*, not as proven edges.

---

## 5) Weather APIs (join by kickoff time + venue coordinates)

| API | Data | Cost / key | Link |
|---|---|---|---|
| [**Open-Meteo — Historical (ERA5)**](https://open-meteo.com/en/docs/historical-weather-api) | Hourly reanalysis **from 1940**; forecast API too | **Free, no key** (non-commercial) | open-meteo.com |
| [**OpenWeather**](https://openweathermap.org/api) | Current / forecast / history | **Freemium** — free 60 calls/min & 1M/mo; [One Call 3.0](https://openweathermap.org/api/one-call-3) = 1,000 calls/day free | openweathermap.org |
| [**Meteostat**](https://meteostat.net/) | Historical station + gridded climate; Python lib + JSON API | **Free**, [CC BY 4.0](https://dev.meteostat.net/license.html) | dev.meteostat.net |
| [**NWS — api.weather.gov**](https://www.weather.gov/documentation/services-web-api) | US forecasts + station observations | **Free, no key** (User-Agent header); **US-only** | api.weather.gov |
| [**NOAA NCEI — CDO v2**](https://www.ncdc.noaa.gov/cdo-web/webservices/v2) | Global historical climate records | **Free** (free token; 10k req/day) | ncdc.noaa.gov/cdo-web |

> Weather matters mainly for **totals / BTTS (ambos marcam)** in *extreme* conditions (heavy rain, high wind); typical stadium climate is already priced. **Leakage rule:** use only the forecast that existed *before* the market you are pricing, not the realised weather.

---

## 6) Structural alternative-data (slow-moving, well-evidenced)

- **Altitude (altitude).** [**McSharry (2007), BMJ 335(7633):1278–81**](https://pubmed.ncbi.nlm.nih.gov/18156225/) — 1,460 international matches across 10 countries: each **+1,000 m** of altitude difference raises goal difference by **≈ ½ a goal**; home-win probability rises from **0.537** (equal altitude) to **0.825** for a **+3,695 m** gap (e.g. Bolivia at La Paz vs a sea-level side such as Brazil); effect significant at **P < 0.001**. Relevant mainly to Andean qualifiers and continental club ties.
- **Crowd / empty stadiums (jogos com portões fechados).** COVID "ghost games" reduced home advantage: [**Bilalić, Gula & Vaci (2021), *Sci. Reports* 11:21558**](https://www.nature.com/articles/s41598-021-00784-8) attribute the drop to **reduced referee bias + lost crowd support**, and [**Leitner & Richlan (2021), *Frontiers Sports Act. Living***](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2021.720488/full) find fewer fouls/cards without fans. Crowd presence is a **regime variable**, not a live edge — see the [Referee page](./Referee_Analytics_and_Discipline_Data.md).
- **Other structural priors (weak, contextual — no assumed magnitudes):** long-haul travel / time-zone changes, **fixture congestion** (raises rotation + injury risk — see [Injury page](./Injury_Lineup_and_Availability_Data.md)), and pitch/surface or new-stadium effects. Test them; don't assume them.

---

## 7) Not fooling yourself

- **Timestamp everything (vazamento / data leakage).** Every text/attention/weather feature must carry a retrieval time; at prediction time use only what was knowable then. This is the #1 source of fake alternative-data "edges".
- **Assume redundancy with the price.** Sentiment and news are public; benchmark incremental value **vs the closing line**, not vs a naive model.
- **Separate event-content from mood.** Distinguish "a goal happened" (past information) from "fans feel positive" (not predictive).
- **Multilingual + bot filtering.** Use the **XLM-R** model for non-English tweets; filter bots/spam/coordinated posting or the sentiment is noise.

---

## 8) Reality check & responsible gambling (Jogo Responsável)

Alternative-data signals are **research features to falsify, not money-printers.** They are the noisiest, most leakage-prone inputs in football modelling and are largely already in the odds. Markets are efficient; **most people lose money gambling.** **Ludopatia (transtorno do jogo)** is a recognised health condition. Use this page as **market education**, never as encouragement to bet. **This is not betting advice.**

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) | Apoio emocional **CVV 188** ([cvv.org.br](https://www.cvv.org.br/)) |

---

## Related in AIForge
- [Match Prediction Models & Techniques](./Match_Prediction_Models_and_Techniques.md) · [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) · [Injury, Lineup & Availability](./Injury_Lineup_and_Availability_Data.md) · [Referee Analytics & Discipline](./Referee_Analytics_and_Discipline_Data.md) · [Features & Feature Engineering](./Features_and_Feature_Engineering.md) · [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources:** [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) · [PRAW](https://praw.readthedocs.io/) · [NewsAPI /v2/everything](https://newsapi.org/docs/endpoints/everything) · [X/Twitter developer](https://developer.x.com/) · [Google Trends](https://trends.google.com/trends/) · [pytrends](https://github.com/GeneralMills/pytrends) · [Action Network soccer public betting](https://www.actionnetwork.com/soccer/public-betting) · [OddsShark soccer consensus](https://www.oddsshark.com/soccer/epl/consensus-picks) · [cardiffnlp twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) · [cardiffnlp twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) · [TweetNLP](https://github.com/cardiffnlp/tweetnlp) · [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) · [VADER](https://github.com/cjhutto/vaderSentiment) · [liamhbyrne/twitter-football-prediction](https://github.com/liamhbyrne/twitter-football-prediction) · [Kampakis & Adamides 2014 (arXiv:1411.1243)](https://arxiv.org/abs/1411.1243) · [Godin et al. 2014](https://www.semanticscholar.org/paper/Beating-the-bookmakers:-leveraging-statistics-and-Godin-Zuallaert/9a82fc5842dea2afb343e12fa34d5892bf4b19d7) · [Schumaker et al. 2016 (DSS 88:76–84)](https://www.sciencedirect.com/science/article/abs/pii/S0167923616300835) · [Wunderlich & Memmert 2021 (SNAM)](https://link.springer.com/article/10.1007/s13278-021-00842-z) · [Open-Meteo Historical](https://open-meteo.com/en/docs/historical-weather-api) · [OpenWeather](https://openweathermap.org/api) · [Meteostat](https://dev.meteostat.net/) · [NWS api.weather.gov](https://www.weather.gov/documentation/services-web-api) · [NOAA NCEI CDO v2](https://www.ncdc.noaa.gov/cdo-web/webservices/v2) · [McSharry 2007 (PubMed 18156225)](https://pubmed.ncbi.nlm.nih.gov/18156225/) · [Bilalić et al. 2021 (Sci. Reports)](https://www.nature.com/articles/s41598-021-00784-8) · [Leitner & Richlan 2021 (Frontiers)](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2021.720488/full).

**Keywords:** football sentiment analysis, alternative data, news signals, Twitter/X sentiment, GDELT tone volume, Reddit PRAW, Google Trends pytrends, fade the public, reverse line movement, bets vs money splits, Action Network, OddsShark consensus, twitter-roberta sentiment, XLM-RoBERTa multilingual, TweetNLP, FinBERT, VADER, in-play tweets, weather API, Open-Meteo, Meteostat, OpenWeather, NWS, NOAA NCEI, altitude McSharry BMJ, ghost games home advantage, data leakage, closing line, responsible gambling · **Português:** análise de sentimento no futebol, dados alternativos, sinais de notícias, sentimento no Twitter/X, tom e volume GDELT, Reddit PRAW, Google Trends, contrariar o público, movimento reverso de linha, divisão apostas vs dinheiro, consenso público, RoBERTa multilíngue (português), clima/tempo, altitude (La Paz/Bolívia), jogos com portões fechados, mando de campo, vazamento de dados, linha de fechamento, jogo responsável, ludopatia, CVV 188, não é dica de aposta.
