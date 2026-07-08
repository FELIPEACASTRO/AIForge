# AI2-THOR (The House Of inteRactions)

## Description
AI2-THOR (The House Of inteRactions) is a photorealistic 3D simulation framework and environment developed by the Allen Institute for AI (AI2) for research in Visual and Embodied AI. It provides an interactive environment where AI agents can navigate and interact with objects in indoor scenes (kitchens, bedrooms, bathrooms, living rooms) to perform complex tasks. The environment is built with Unity 3D, which enables physical simulation for objects and scenes, including visual object states (open/closed, on/off, hot/cold). The project has evolved to include more complex environments such as ProcTHOR, which uses procedural generation to create a massive number of environments.

## Statistics
- **iTHOR:** 120 rooms (kitchens, bedrooms, bathrooms, living rooms), more than 2000 unique objects.
- **RoboTHOR:** 89 apartments with more than 600 objects, with physical and simulated counterparts for 14 apartments.
- **ProcTHOR (most recent version):** 10,000 procedurally generated houses, offering a massive volume of data for training.
- **Binary size:** The 3D environment (Unity) is downloaded on first run and is approximately 500MB.
- **Versions:** The most recent publicly available documentation is for version 2.1.0, but the project is actively maintained on GitHub, with updates and new *frameworks* such as ProcTHOR (announced in 2022).

## Features
- **Photorealistic Simulation:** High-quality 3D scenes based on Unity 3D.
- **Physical Interaction:** Support for physical simulation of objects, allowing agents to interact with them (push, pick up, open, etc.).
- **Object States:** Objects with mutable visual states (e.g., toaster on/off, door open/closed).
- **Multiple Agents:** Support for multiple agents in the same scene and different types of agents (humanoids, drones).
- **Sub-environments:** Includes iTHOR (navigation and interaction), RoboTHOR (sim-to-real transfer with LoCoBot robots), and ManipulaTHOR (object manipulation with a robotic arm).
- **Scalability (ProcTHOR):** The most recent version, ProcTHOR, enables procedural generation of 10,000 houses, offering a massive volume of data for training Embodied AI models.

## Use Cases
- **Embodied AI:** Training and evaluation of agents that need to interact with the physical world.
- **Visual Navigation:** Development of models for navigation in complex indoor environments.
- **Object Manipulation:** Research into fine manipulation tasks and long-horizon planning with robotic arms (ManipulaTHOR).
- **Sim-to-Real Transfer (Sim2Real):** Use of RoboTHOR to test the generalization of models trained in simulation to physical robots (LoCoBot).
- **Reinforcement Learning (RL):** Platform for developing RL agents on tasks that require reasoning and complex interaction.
- **Multi-step Task Solving:** Creation of agents capable of following natural language instructions to complete sequences of actions (e.g., cooking, tidying up a room).

## Integration
AI2-THOR is installed as a Python library via `pip`.
1. **Installation:** A Python virtual environment (3.5+) is recommended.
   ```bash
   pip install ai2thor
   ```
2. **Basic Usage (Python):** The 3D environment is downloaded automatically (approx. 500MB) on first run.
   ```python
   import ai2thor.controller
   
   controller = ai2thor.controller.Controller()
   # Inicia o ambiente 3D
   controller.start() 
   
   # Exemplo de ação: mover o agente para frente
   event = controller.step(action="MoveAhead")
   
   # Para a simulação
   controller.stop()
   ```
3. **Documentation:** The official documentation provides details about the available actions (`MoveAhead`, `RotateLeft`, `PickupObject`, etc.) and the structure of the returned metadata.
4. **Requirements:** Requires an X server with OpenGL (for Linux users) and a graphics card supporting DX9 (shader model 3.0) or DX11. *Headless* rendering is supported for compute clusters.

## URL
[https://ai2thor.allenai.org/](https://ai2thor.allenai.org/)
