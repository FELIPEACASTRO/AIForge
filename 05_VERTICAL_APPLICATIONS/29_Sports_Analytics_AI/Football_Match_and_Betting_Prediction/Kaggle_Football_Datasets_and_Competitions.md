# Kaggle — Football (Soccer) Datasets & Competitions

> Curated index of **football/soccer** datasets and competitions on Kaggle for match & betting prediction (pulled live via the Kaggle API). Great starting points for match results, odds, xG, player values, and World-Cup modeling — worldwide.

> ⚠️ **Research & education only.** Betting markets are highly efficient and most bettors lose money over time. Nothing here is betting advice. If gambling is a problem, seek help: [BeGambleAware](https://www.begambleaware.org/), [GamCare](https://www.gamcare.org.uk/), [Gambling Therapy](https://www.gamblingtherapy.org/); 🇧🇷 [Jogo Responsável](https://www.gov.br/fazenda/pt-br/assuntos/loterias/jogo-responsavel).

## 🗄️ Core match & results datasets

| Dataset | Content | Link |
|---|---|---|
| **European Soccer Database** (hugomathien) | The canonical Kaggle DB: 25k+ matches (2008–2016), 11 EU countries, team/player attributes (FIFA), betting odds — SQLite | https://www.kaggle.com/datasets/hugomathien/soccer |
| Club Football Match Data (2000–2025) | Large multi-league club match dataset | https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025 |
| Football Matches 2024/2025 — Top-5 leagues | Recent top-5 league matches | https://www.kaggle.com/datasets/tarekmasryo/football-matches-20242025-top-5-leagues |
| Most Comprehensive Football Dataset (5.7M+ records) | Very large multi-source records | https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets |
| Football Events | Event-level data (~9k games, granular events) | https://www.kaggle.com/datasets/secareanualin/football-events |
| English Premier League Results | EPL historical results | https://www.kaggle.com/datasets/irkaal/english-premier-league-results |
| All Premier League Matches 2010–2021 | EPL match history | https://www.kaggle.com/datasets/pablohfreitas/all-premier-league-matches-20102021 |
| LaLiga Matches (2019–2025, FBref) | Spanish league w/ FBref stats | https://www.kaggle.com/datasets/marcelbiezunski/laliga-matches-dataset-2019-2025-fbref |
| Football Matches of Spanish League | LaLiga history | https://www.kaggle.com/datasets/ricardomoya/football-matches-of-spanish-league |
| Football Data — European Top-5 Leagues | Top-5 leagues | https://www.kaggle.com/datasets/kamrangayibov/football-data-european-top-5-leagues |
| Major League Soccer Dataset | 🇺🇸 MLS data | https://www.kaggle.com/datasets/josephvm/major-league-soccer-dataset |
| 🇧🇷 **Brazilian Soccer Database** (Brasileirão) | Jogos do Campeonato Brasileiro | https://www.kaggle.com/datasets/ricardomattos05/jogos-do-campeonato-brasileiro |

## 💰 Odds / betting datasets

| Dataset | Content | Link |
|---|---|---|
| **Beat The Bookie: Worldwide Football Odds** | Odds time series across bookmakers, many leagues | https://www.kaggle.com/datasets/austro/beat-the-bookie-worldwide-football-dataset |
| World Soccer DB: archive of odds (2021) | Historical odds archive | https://www.kaggle.com/datasets/sashchernuh/european-football |
| Bets Strategy | Betting-strategy oriented data | https://www.kaggle.com/datasets/caesarlupum/betsstrategy |

## 🌍 International / World Cup / rankings

| Dataset | Content | Link |
|---|---|---|
| Football — FIFA World Cup, 1930–2026 | All World Cups | https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup |
| Football — UEFA EURO, 1960–2024 | All Euros | https://www.kaggle.com/datasets/piterfm/football-soccer-uefa-euro-1960-2024 |
| FIFA World Cup 2022 (matches/players/teams) | WC2022 detail (swaptr) | https://www.kaggle.com/datasets/swaptr/fifa-world-cup-2022-match-data |
| AFCON 2025–26 Complete Match Statistics | 🌍 Africa Cup of Nations | https://www.kaggle.com/datasets/dhrubangtalukdar/afcon-202526-complete-match-statistics |
| FIFA International Men's Ranking (1993–now) | FIFA ranking history | https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now |
| Football/Soccer Clubs Ranking | Club rankings | https://www.kaggle.com/datasets/ramjasmaurya/footballsoccer-clubs-ranking |

## 👤 Players, values & squads

| Dataset | Content | Link |
|---|---|---|
| Football Data from Transfermarkt | Players, clubs, values, appearances | https://www.kaggle.com/datasets/davidcariboo/player-scores |
| Football Players Transfer Fee Prediction | Transfer-fee ML dataset | https://www.kaggle.com/datasets/khanghunhnguyntrng/football-players-transfer-fee-prediction-dataset |
| FIFA Player Performance & Market Value | Player value modeling | https://www.kaggle.com/datasets/itszubi/fifa-player-performance-and-market-value |
| All-Time Premier League Player Statistics | EPL player stats | https://www.kaggle.com/datasets/rishikeshkanabar/premier-league-player-statistics-updated-daily |

## 🏆 Prediction competitions & challenges
- European Soccer Database challenge — https://www.kaggle.com/competitions/jvm000001
- Football Players Value Prediction (1056lab) — https://www.kaggle.com/competitions/1056lab-football-players-value-prediction
- Soccer World Cup 2022 Prediction (community) — https://www.kaggle.com/datasets/shilongzhuang/soccer-world-cup-challenge
- WC2026 Match Probability Baseline — https://www.kaggle.com/datasets/sarazahran1/wc2026-match-probability-baseline-dataset

## 🔌 Pull it yourself (API)
```bash
pip install kaggle   # kaggle.json in ~/.kaggle/
kaggle datasets list -s "football soccer" --sort-by votes
kaggle datasets download -d hugomathien/soccer
```

## Related in AIForge
- [Match Prediction Models & Techniques](./Match_Prediction_Models_and_Techniques.md) · [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Odds, Betting Markets & Value Betting](./Odds_Betting_Markets_and_Value_Betting.md) · [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md)
- Parent vertical: [`../`](../) (Sports Analytics AI)

**Keywords:** Kaggle football, soccer dataset, European Soccer Database, football odds dataset, Beat the Bookie, Brasileirão dataset, World Cup dataset, match prediction data, dataset de futebol, previsão de partidas.
