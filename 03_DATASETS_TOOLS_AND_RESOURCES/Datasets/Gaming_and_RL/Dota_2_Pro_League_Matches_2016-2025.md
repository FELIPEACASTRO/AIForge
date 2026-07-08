# Dota 2 Pro League Matches 2016-2025

## Description
This is a comprehensive dataset of Dota 2 Professional League matches, covering the period from 2016 to 2025. The dataset is updated weekly and is derived from the OpenDota API. It is ideal for data science researchers and enthusiasts interested in esports analytics, machine learning for match outcome prediction, and analysis of the behavior of professional players and teams. The dataset consists of 12 main CSV files per month/year, in addition to constants and image folders.

## Statistics
*   **Total Size:** 43.75 GB
*   **Number of Files:** 1567
*   **Number of Columns:** 5237 (across all files)
*   **Coverage:** Dota 2 Professional League matches from 2016 to November 2025.
*   **Update Frequency:** Weekly.

## Features
The dataset consists of 12 main CSV files that provide detailed data about the matches:
*   **main\_metadata.csv**: Main match metadata.
*   **players.csv**: Details about the heroes and actions of the players.
*   **picks\_bans.csv**: Information about the hero picking and banning phase.
*   **objectives.csv**: Milestones and objectives achieved in the match (e.g., FIRSTBLOOD, BARRACKS down).
*   **teamfights.csv**: Details about the team fights.
*   **radiant\_gold\_adv.csv** and **radiant\_exp\_adv.csv**: Gold and experience advantage per minute for the Radiant team.
*   **chat.csv** and **all\_word\_counts.csv**: Chat data and token counts.
*   **cosmetics.csv**, **draft\_timings.csv**, and **teams.csv**: Additional information about cosmetics, draft timings, and teams.
It also includes `Constants` folders (IDs and statistics of abilities, heroes, and items) and `Images`.

## Use Cases
*   **Match Outcome Prediction:** Use of draft data, gold/experience advantages, and player statistics to predict the winner of matches.
*   **Esports Strategy Analysis:** Study of hero picking and banning patterns (picks/bans), analysis of team fights, and the impact of objectives on the outcome.
*   **Player Behavior Modeling:** Analysis of individual and team performance over time.
*   **AI Agent Development:** Use of the data as a basis for training machine learning and reinforcement learning models to play Dota 2.

## Integration
The dataset is hosted on Kaggle and can be downloaded directly through the Kaggle web interface. For Python users, integration can be done using the Kaggle API, which allows programmatic download of the dataset.

**Steps to download via the Kaggle API:**
1.  Install the Kaggle library: `pip install kaggle`
2.  Configure your Kaggle API credentials.
3.  Download the dataset using the command: `kaggle datasets download -d bwandowando/dota-2-pro-league-matches-2023`
4.  Unzip the downloaded file to access the CSV data.

The primary source of the data is the OpenDota API. Detailed documentation of the fields can be found in the OpenDota documentation.

## URL
[https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023](https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023)
