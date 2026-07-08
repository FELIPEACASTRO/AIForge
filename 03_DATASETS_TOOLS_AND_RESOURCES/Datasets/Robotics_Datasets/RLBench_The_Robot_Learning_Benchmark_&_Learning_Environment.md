# RLBench: The Robot Learning Benchmark & Learning Environment

## Description
RLBench is an ambitious large-scale learning benchmark and environment, designed to facilitate research across diverse areas of vision-guided robotic manipulation. The environment features **100 unique, manually designed tasks**, varying in difficulty from simple reaching tasks to more complex manipulations. It is a fundamental resource for the development and evaluation of **Reinforcement Learning (RL)**, **Imitation Learning (IL)**, **Multi-Task Learning**, and **Few-Shot Learning** algorithms in robotics. The environment uses the CoppeliaSim simulator to provide a realistic and flexible simulation environment.

## Statistics
The benchmark is composed of **100 unique robotic manipulation tasks**. The demonstration dataset is generated on demand, but pre-generated subsets exist. For example, the "rlbench-18-tasks" subset (18 tasks) with 100 demonstrations per task has a total size of approximately **~116GB**. The most recent software version mentioned in the GitHub repository is **v1.2.0**, released on February 18, 2022. The original paper is from 2019/2020.

## Features
RLBench stands out for its collection of **100 manipulation tasks** with varying difficulty. It supports various action modes for the robotic arm (such as JointVelocity) and the gripper (such as Discrete). It is optimized for research in **few-shot learning** and **meta-learning** due to its wide range of tasks. It includes support for **Imitation Learning** with the ability to load pre-saved demonstrations and facilitates **Sim-to-Real** experiments through visual domain randomization functionality. The environment also features integration with the popular **Gym** ecosystem (RLBench Gym). The default image observations are 128x128.

## Use Cases
RLBench is widely used for: 1) **Evaluation of RL Algorithms** on complex robotic manipulation tasks. 2) **Research in Imitation Learning and Multi-Task Learning**, leveraging the large number of tasks and demonstrations. 3) **Few-Shot Learning and Meta-Learning Studies**, testing the generalization ability of models. 4) **Sim-to-Real Transfer Experiments**, using domain randomization to increase model robustness.

## Integration
RLBench is built on top of the **CoppeliaSim v4.1.0** simulator and the **PyRep** library. Integration requires installing CoppeliaSim and configuring environment variables. The Python package can be installed via pip directly from the GitHub repository: `pip install git+https://github.com/stepjam/RLBench.git`. To use pre-saved demonstrations (required for Imitation Learning), the dataset path must be specified when initializing the environment: `env = Environment(action_mode, DATASET)`. The demonstration dataset must be generated or downloaded separately.

## URL
[https://sites.google.com/view/rlbench](https://sites.google.com/view/rlbench)
