# Meta-World (Multi-task RL)

## Description
Meta-World is an open-source simulated *benchmark* for the development and evaluation of multi-task reinforcement learning (*Multi-task RL*) and meta-reinforcement learning (*Meta-RL*) algorithms. The *benchmark* consists of 50 distinct robotic manipulation environments in a simulator (MuJoCo), designed to be diverse yet share a common structure. The main goal is to enable the development of algorithms that can generalize and accelerate the acquisition of new skills, evaluating the ability to learn a set of skills simultaneously or to adapt quickly to new tasks.

## Statistics
- **Number of Tasks:** 50 distinct robotic manipulation tasks.
- **Versions:** The most recent version is **v3.0.0** (released in June 2025, according to the GitHub repository).
- **Benchmarks:** 6 evaluation modes (MT1, MT10, MT50, ML1, ML10, ML45).
- **Sample Count/Size:** There is no fixed dataset size in gigabytes, as it is a *benchmark* of simulated environments. The number of "samples" (interactions) is generated dynamically during training and evaluation, potentially reaching millions of *timesteps* per task.

## Features
- **50 Distinct Tasks:** Includes tasks such as reaching, pushing, pick-and-place, opening and closing doors/drawers, and pressing buttons, all in a robotic manipulation environment.
- **Multiple Benchmarks:** Offers six evaluation modes with different difficulty levels:
    - **Multi-Task (MT1, MT10, MT50):** For simultaneous learning of 1, 10, or 50 tasks.
    - **Meta-Learning (ML1, ML10, ML45):** For evaluating the ability to adapt to goal variations (ML1) or to new tasks (ML10, ML45) with few examples.
- **Simulation Environment:** Uses the MuJoCo simulator for robotic interactions.
- **Gymnasium-Compatible API:** The API follows the Gymnasium standard (formerly Gym), facilitating integration with existing RL frameworks.

## Use Cases
- **Evaluation of Meta-Reinforcement Learning (Meta-RL) Algorithms:** Testing the ability of agents to learn to learn and adapt quickly to new tasks with few examples.
- **Evaluation of Multi-task Reinforcement Learning (Multi-task RL) Algorithms:** Developing policies that can solve a diverse set of tasks simultaneously.
- **Research in Generalization and Knowledge Transfer:** Studying how knowledge acquired on a set of tasks can be transferred to accelerate learning on unseen tasks.
- **Development of Robotic Manipulation Agents:** Serving as a standardized testbed for RL agents focused on fine manipulation skills.

## Integration
Meta-World is distributed as a Python package and can be installed via `pip`.

**Installation:**
The most recent and recommended installation is done directly from the Farama Foundation GitHub repository:
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

**Basic Usage (MT1 Example):**
The *benchmark* is accessed through the `gymnasium.make` API.
```python
import gymnasium as gym
import metaworld

# Cria um ambiente MT1 (tarefa única) para 'reach-v3'
env = gym.make("Meta-World/MT1", env_name="reach-v3")

observation, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```
For multi-task benchmarks (MT10, MT50) or meta-learning (ML10, ML45), it is necessary to use `gym.make_vec` to create vectorized environments.

## URL
[https://meta-world.github.io/](https://meta-world.github.io/)
