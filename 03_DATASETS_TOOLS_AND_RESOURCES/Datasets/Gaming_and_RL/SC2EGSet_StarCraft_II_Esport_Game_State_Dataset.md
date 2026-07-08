# SC2EGSet: StarCraft II Esport Game State Dataset

## Description
The **SC2EGSet: StarCraft II Esport Game State Dataset** is a comprehensive, up-to-date dataset (Version 2.0.1, March 2025) that provides detailed information about game state and replays from StarCraft II esports tournaments since 2016. This dataset is an evolution of the original work by DeepMind and of SC2ReSet, and is maintained by independent researchers. It was designed to facilitate research in Artificial Intelligence (AI), Machine Learning (ML), and human-computer interaction (HCI) and esports studies. The data is processed from raw replays, offering a rich, structured view of the strategic and tactical decisions of professional players.

## Statistics
**Version:** 2.0.1 (Published in March 2025).
**Processed Replays:** Data processed from 55 tournament "replaypacks".
**Final Files:** 17,895 processed game-state files.
**Size:** The example file `2016_IEM_10_Taipei.zip` is 12.6 GB. The complete dataset is significantly larger, distributed across multiple files.
**Period:** Tournament replays since 2016.

## Features
Game-state data and high-level esports replays. Includes detailed information such as game version histogram, match dates, server information, chosen races, match duration, detected units, and a race-versus-playtime histogram. The dataset is compatible with the PyTorch and PyTorch Lightning APIs through the `sc2_datasets` library, facilitating data loading and modeling. Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

## Use Cases
**Artificial Intelligence and Reinforcement Learning (RL):** Training AI agents to play StarCraft II, such as DeepMind's AlphaStar, and developing offline RL models.
**Esports Analysis:** Studying professional players' strategies, detecting play patterns, and analyzing performance.
**HCI Research:** Investigating human decision-making in complex, real-time environments.
**Predictive Modeling:** Building models to predict match outcomes or players' future actions.

## Integration
The dataset can be accessed and used through the `sc2_datasets` Python library.
1.  **Library Installation:** `pip install sc2_datasets`
2.  **Usage with PyTorch/PyTorch Lightning:** The library provides classes such as `SC2EGSetDataset` (PyTorch) and `SC2EGSetDataModule` (PyTorch Lightning) that manage the download, extraction, and access of the data.
3.  **Download:** The raw files (replays) can be downloaded directly from Zenodo. The file `2016_IEM_10_Taipei.zip` is 12.6 GB, and the complete dataset is composed of multiple files, totaling a significant volume of data. Programmatic access via API is the recommended method.

## URL
[https://zenodo.org/records/15073637](https://zenodo.org/records/15073637)
