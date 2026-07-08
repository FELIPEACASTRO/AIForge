# DQN Replay Dataset (Atari 2600 Games RL)

## Description
The **DQN Replay Dataset** is an *offline* Reinforcement Learning (RL) dataset based on Atari 2600 games. It was collected by training a Deep Q-Network (DQN) agent on 60 different Atari 2600 games, with sticky actions enabled, for 200 million *frames* per game. The dataset stores all experience tuples (*observation, action, reward, next observation*) encountered during training. It is a crucial *benchmark* for research in *offline* RL (also known as *Batch RL*), allowing algorithms to be evaluated without the need for additional interaction with the environment. The dataset is associated with the *paper* "An Optimistic Perspective on Offline Reinforcement Learning" (ICML 2020) [1].

## Statistics
- **Games:** 60 Atari 2600 games.
- **Training Frames:** 200 million *frames* per game.
- **Experience Tuples:** Approximately 50 million experience tuples (*observation, action, reward, next observation*) per *run* of 200 million *frames*.
- **Versions:** The collection process was repeated 5 times for each game, resulting in 5 data *runs* per game.
- **Estimated Size:** Each data *run* is approximately 3.5 times larger than ImageNet (which suggests a massive size; although the exact size in GB is not provided, the scale is in the terabytes) [3].

## Features
- **Game Diversity:** Contains data from 60 different Atari 2600 games.
- **Complete Replay Experience:** Stores the entire *replay* experience of the DQN agent, totaling approximately 50 million experience tuples per *run* of 200 million *frames*.
- **Standard Configuration:** Collected under the standard protocol of 200 million *frames* and with *sticky actions* (25% probability of repeating the previous action).
- **Offline RL Benchmark:** Serves as a high-quality *benchmark* for the development and testing of *Offline* Reinforcement Learning (Batch RL) algorithms.
- **Access via GCP:** The data is hosted in a public Google Cloud Platform (GCP) *bucket*.

## Use Cases
- **Offline Reinforcement Learning (Batch RL):** Training and evaluation of RL algorithms that learn from a fixed dataset, without additional interaction with the environment.
- **Generalization Evaluation:** Testing the ability of *offline* RL algorithms to generalize to high-quality policies from suboptimal data or prior policies.
- **DQN Research:** Analysis of the behavior and *replay* experience of DQN agents.
- **Algorithm Comparison:** Serves as a *benchmark* for comparing the performance of different *offline* RL algorithms (such as REM, CQL, etc.) [1].

## Integration
The dataset is hosted in the public Google Cloud Platform (GCP) *bucket* `gs://atari-replay-datasets`. Access and *download* are performed using the `gsutil` command-line tool.

**Download Instructions:**
1. **Install `gsutil`:** Follow the `gsutil` installation instructions [2].
2. **Full Download:** To download the complete dataset (all 60 games):
   ```bash
   gsutil -m cp -R gs://atari-replay-datasets/dqn ./
   ```
3. **Download per Game:** To download the dataset for a specific game (replace `[NOME_DO_JOGO]` with the name of the game, e.g., `Pong`):
   ```bash
   gsutil -m cp -R gs://atari-replay-datasets/dqn/[NOME_DO_JOGO] ./
   ```
**Important Note:** The dataset was generated using a legacy version of the Atari ROMs (`atari-py<=0.2.5`). To avoid incompatibilities, it is recommended to use `atari-py<=0.2.5` and `gym<=0.19.0`, or to follow the instructions on the official *site* for use with newer versions of `ale-py` [1]. The dataset is also available in `tfds` (TensorFlow Datasets) format as part of **RL Unplugged** [1].

## URL
[https://offline-rl.github.io/](https://offline-rl.github.io/)
