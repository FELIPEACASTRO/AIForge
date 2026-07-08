# Gymnasium (Successor to OpenAI Gym Environments)

## Description
Gymnasium is the maintained successor to the **OpenAI Gym** library, establishing itself as the de facto standard for developing and comparing Reinforcement Learning (RL) algorithms. It provides a standardized API and a diverse collection of benchmark environments, such as Classic Control, Box2D, MuJoCo, and Atari, enabling researchers and developers to test and evaluate their RL agents in controlled simulations. The project is actively maintained by the Farama Foundation, ensuring compatibility with the latest Python versions and RL best practices.

## Statistics
*   **Current Version:** v1.2.2 (November 2025).
*   **Downloads:** More than 18 million total downloads, with over 1 million monthly downloads in 2025 (referring to the 2024 paper).
*   **Environments:** Hundreds of benchmark environments distributed across core packages and extensions (e.g., `gymnasium-robotics`, `gymnasium-atari`).
*   **"Dataset" Size:** Not a static dataset, but a collection of simulation environments. The base package size is small, but the dependencies (such as MuJoCo) can be large.
*   **Python Support:** 3.9, 3.10, 3.11, 3.12, 3.13.

## Features
*   **Standardized API:** Unified interface for all RL environments, making it easy to swap agents between different tasks.
*   **Environment Diversity:** Includes categories such as Classic Control (e.g., CartPole), Box2D (e.g., LunarLander), MuJoCo (robotics simulations), and Atari (classic games).
*   **Wrapper Support:** Enables easy modification and augmentation of environments (e.g., observation normalization, adding time limits).
*   **Compatibility:** Active support for Python 3.9+ and integration with the leading RL libraries.
*   **Maintenance Focus:** Addresses maintenance issues and inconsistencies present in the original version of OpenAI Gym.

## Use Cases
*   **RL Agent Development:** Primary platform for creating, training, and testing new Reinforcement Learning algorithms.
*   **Benchmarking:** Standard use for comparing the performance of different RL algorithms on known tasks.
*   **Robotics and Control:** Simulation of robotic control and manipulation tasks (via MuJoCo environments and extensions).
*   **Education and Research:** Essential tool for teaching and research in artificial intelligence and machine learning.
*   **Robustness Testing:** Evaluation of agents' generalization ability across different environment configurations.

## Integration
Installation is done through the `pip` package manager. Specific environments may require additional dependencies (e.g., `\[classic-control\]`, `\[box2d\]`, `\[mujoco\]`).

**Basic Installation:**
```bash
pip install gymnasium
```

**Installation with Common Environments:**
```bash
pip install "gymnasium[classic-control, box2d]"
```

**Usage Example (Python):**
```python
import gymnasium as gym

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment
observation, info = env.reset(seed=42)

for _ in range(1000):
    # Select a random action
    action = env.action_space.sample()

    # Execute the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Check whether the episode has ended or was truncated
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## URL
[https://gymnasium.farama.org/](https://gymnasium.farama.org/)
