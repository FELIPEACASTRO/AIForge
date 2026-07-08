# Lichess/standard-chess-games

## Description
The **Lichess/standard-chess-games** dataset is a massive, monthly-updated collection of more than **7.1 billion rows** (games) of rated standard chess played on the Lichess.org platform. The data is provided in Parquet format, partitioned by year and month, which facilitates the analysis and processing of large volumes of data. The collection spans games from 2013 onward and is continuously updated, including 2025 data (in progress). It is a primary source for research and development in artificial intelligence and data science in the chess domain.

## Statistics
**Size:** 4.94 TB (in Parquet files).
**Samples:** More than **7,136,017,970** (7.1 billion) chess games.
**Versions:** Updated monthly, covering data from **2013 to 2025** (the year 2025 is in progress). The focus on 2023-2025 is ensured by continuous updates.

## Features
The dataset includes detailed fields for each game, such as: Event, game URL (Site), Player names (White/Black), Result, player Titles, **Glicko2 Elo** (White/Black) and Elo change, UTC Date/Time, ECO Code, Opening, Termination Type, and Time Control. The `movetext` field contains the moves in PGN format. A notable feature is that about 6% of the games contain **Stockfish analysis evaluations** (advantage in centipawns or mate), essential for training AI models.

## Use Cases
- **Training AI Models:** Ideal for training chess AI models, such as those for Next Move Prediction and position evaluation.
- **Data Science Research:** Used for analyzing large volumes of data, such as studies of player psychology and behavior, and opening trends.
- **Tool Development:** Basis for building Opening Explorers, analysis engines, and other chess assistance tools.

## Integration
The dataset is easily accessible through the **Hugging Face Datasets** library.
1.  **Installation:** `pip install datasets`
2.  **Usage in Python:**
    ```python
    from datasets import load_dataset
    # Load the complete dataset
    dataset = load_dataset("Lichess/standard-chess-games", split="train")
    # To load a specific subset (e.g., 2024 games)
    # dataset_2024 = load_dataset("Lichess/standard-chess-games", split="train", year=2024)
    ```
    The original, complete PGN files can be downloaded directly from the Lichess database dumps (database.lichess.org).

## URL
[https://huggingface.co/datasets/Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games)
