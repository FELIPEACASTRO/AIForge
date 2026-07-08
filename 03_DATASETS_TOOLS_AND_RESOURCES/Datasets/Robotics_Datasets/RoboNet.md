# RoboNet

## Description
RoboNet is an open and diverse database for sharing robotic experiences, focused on large-scale, multi-robot learning. It is designed to enable high-capacity models, such as deep neural networks, to generalize effectively across a wide range of real-world environments. The dataset consists of robot-object interactions captured from multiple viewpoints and robotic platforms. The main goal is to pre-train reinforcement learning models on a diverse dataset and then transfer the knowledge to new robots and tasks with far less task-specific data.

## Statistics
- **Dataset Size (Full):** 36.20 GiB download, 144.90 GiB on disk (128x128 configuration).
- **Samples (Trajectories):** Approximately 162,417 training trajectories (in the full version).
- **Video Frames:** More than 15 million video frames of robot-object interactions.
- **Robotic Platforms:** 7 different robotic platforms.
- **Viewpoints:** 113 unique camera viewpoints.
- **Version (TFDS):** `4.0.1` (default).
- **Available Resolutions (TFDS):** 64x64 and 128x128.

## Features
- **Large-Scale Multi-Robot Learning:** The dataset enables training of generalizable models for vision-based robotic manipulation across multiple platforms.
- **Platform Diversity:** Includes data from 7 different robotic platforms, ranging from Kuka industrial arms to low-cost WidowX arms.
- **Multiple Viewpoints:** Interactions are captured from 113 unique camera viewpoints, increasing the visual robustness of the trained models.
- **Action and State Data:** In addition to video frames, the dataset includes actions (end-effector position and rotation deltas, plus the gripper joint) and states (Cartesian control action space of the end-effector and gripper joint).
- **Knowledge Transfer:** Demonstrates the ability to pre-train on RoboNet and fine-tune on data from a specific robot (such as Franka or Kuka) to outperform robot-specific training with 4x to 20x more data.

## Use Cases
- **Pre-training for Reinforcement Learning (RL):** Used to pre-train RL models on a diverse dataset before fine-tuning for specific tasks.
- **Video Prediction:** Training video prediction models to anticipate robotic interactions.
- **Supervised Inverse Models:** Training models to infer actions from visual observations.
- **Skill Generalization:** Studying the ability to generalize robotic controllers to new objects, tasks, scenes, camera viewpoints, grippers, or even entirely new robots.
- **Computer Vision Research for Robotics:** Provides a benchmark for developing vision algorithms for robotic manipulation.

## Integration
The RoboNet dataset can be accessed and used in several ways:

1.  **TensorFlow Datasets (TFDS):** The easiest way to use the dataset, especially for models in TensorFlow.
    *   **Installation:** `pip install tensorflow-datasets`
    *   **Usage:** The dataset can be loaded directly in Python code:
        ```python
        import tensorflow_datasets as tfds
        ds = tfds.load('robonet/robonet_128', split='train', shuffle_files=True)
        ```
    *   **Configurations:** Different configurations are available (e.g., `robonet_sample_64`, `robonet_128`) that vary in size and resolution.

2.  **Direct Download (HDF5):** The full dataset (36 GB) or a sample (~100 MB) can be downloaded directly using the `gdown` tool and extracted.
    *   **Installation:** `pip install gdown`
    *   **Full Download (36 GB):**
        ```bash
        gdown https://drive.google.com/a/andrew.cmu.edu/uc?id=1BkqHzfRkfzgzCfc73NbNnPMK_rg3i1n9&export=download
        tar -xzvf robonet_v3.tar.gz
        ```
    *   **Sample Download (~100 MB):**
        ```bash
        gdown https://drive.google.com/uc?id=1YX2TgT8IKSn9V4wGCwdzbRnS53yicV2P&export=download
        tar -xvzf robonet_sampler.tar.gz
        ```

3.  **Source Code and Utilities:** The official GitHub repository provides code to load, manipulate, and train models (such as supervised inverse models and video prediction models) on the dataset.
    *   **Repository:** `git clone https://github.com/SudeepDasari/RoboNet.git`
    *   **Installation:** `pip install -r requirements.txt` and `python setup.py develop`

## URL
[https://www.robonet.wiki/](https://www.robonet.wiki/)
