# Robotics Control Prompts

## Description
The **Robotics Control Prompts** technique refers to the use of **Large Language Models (LLMs)** and **Vision-Language Models (VLMs)** to generate high-level plans, code, or action sequences for robots, allowing humans to interact with robotic systems using natural language [1] [2]. Instead of programming each movement, the user provides an instruction in natural language (the *prompt*), and the LLM acts as a high-level planner, decomposing the task into subtasks and, frequently, generating executable code (such as Python) or low-level commands that the robot can process [3]. Advanced systems, such as **ELLMER** and **ARRC**, separate the LLM's high-level planning from the robot's low-level control, ensuring safety and reactivity. The prompt is crucial for defining the context, the available tools (robot APIs), and the safety constraints, turning the LLM into a reasoning and planning agent for robotics [4].

## Examples
```
**1. Task Planning Prompt (Code-as-Policies):**

```
You are a robot planner. Your task is to generate Python code that uses the functions 'move_to(x, y, z)', 'grasp(object_name)', and 'release()' to complete the task.

Instruction: "Pick up the red block from the table and place it in the blue basket."

Available tools:
- get_coordinates(object_name): Returns the coordinates (x, y, z) of an object.
- move_to(x, y, z): Moves the end effector to the position.
- grasp(object_name): Closes the gripper on the object.
- release(): Opens the gripper.

Generate only the Python code.
```

**2. Reasoning Prompt with Ambiguity (ELLMER):**

```
Context: The user said "I'm tired, please make me a hot drink and decorate the plate with an animal."
Current State: The robot is at the home position. The camera detects a mug, a jar of instant coffee, and a kettle.

Task: Decompose the user's instruction into a logical sequence of 5 to 7 high-level steps. Consider that "hot drink" implies coffee and "decorate the plate" is a secondary task.
```

**3. Error Correction Prompt (Feedback Loop):**

```
Original Plan: move_to(x_caneca, y_caneca, z_caneca)
Sensor Feedback: Grasp failure. The force sensor indicates that the gripper slipped.
Instruction: Adapt the plan to retry grasping the mug. Increase the grasping force by 10% and try a slightly different approach angle (z+5mm). Generate the new Python code.
```

**4. Retrieval-Augmented Generation Prompt (RAG/ARRC):**

```
You are the ARRC control system. Use the retrieved knowledge (below) to generate a safe JSON action plan for the task.

Retrieved Knowledge:
- Safety Heuristic: 'Never exceed 50% of maximum speed when handling hot liquids.'
- Task Template: 'Pick-and-Place: [approach, grip, lift, move, release, retract].'

Instruction: "Move the water bottle from position A to position B."

Generate the action plan in JSON format.
```

**5. State Query Prompt (Debugging):**

```
Instruction: "Describe the current state of the environment and the robot in terms of detected objects, their coordinates, and the position of the gripper."

Expected Response: A formatted list of objects and the position of the end effector.
```
```

## Best Practices
**1. Prompt Structuring (Code-as-Policies):** Always ask the LLM to generate executable code (e.g., Python) instead of just descriptive text. Clearly define the available robotic functions and tools (APIs) in the system prompt. **2. Separation of Responsibilities (ELLMER/ARRC):** Use the LLM only for high-level planning and reasoning. Low-level control (kinematics, safety, collision detection) should be handled by robust local control modules on the robot. **3. Retrieval-Augmented Generation (RAG):** For complex or specific tasks, use RAG to provide the LLM with contextualized knowledge, such as operation manuals, safety heuristics, or examples of previous tasks, improving the validity and adaptability of the plan [2]. **4. Feedback and Iteration:** Include visual (VLM) and force/touch feedback mechanisms so that the LLM can iterate and correct the plan in real time if the state of the world does not match what was expected. **5. Safety Constraints:** Incorporate safety constraints (workspace, speed, and force limits) into the prompt and the execution system to avoid dangerous movements or catastrophic failures.

## Use Cases
**1. Manipulation and Assembly:** Robots that perform complex assembly or object manipulation tasks in warehouses or production lines, receiving instructions in natural language (e.g., "Assemble the chair using parts A, B, and C"). **2. Service Robotics:** Robots in home or office environments that respond to ambiguous or high-level commands (e.g., "Clean the table and bring me a glass of water") [3]. **3. Mission Planning:** Autonomous robots (drones, rovers) that plan routes and action sequences in unstructured environments based on mission objectives expressed in natural language (e.g., "Explore the southern area and collect soil samples"). **4. Collaborative Robotics (Cobots):** Robotic agents that work alongside humans and need to quickly adapt their plans based on verbal commands or gestures. **5. Control Policy Generation:** Use of LLMs to generate control policies (action sequences) that can be used to train Reinforcement Learning models more efficiently.

## Pitfalls
**1. Hallucinations and Unsafety:** The LLM may generate plans or code that are physically impossible, unsafe, or that violate safety constraints. **Mitigation:** Implement "safety guards" (guarded execution) that validate the code or plan before execution and control low-level movement [2]. **2. Lack of Grounding:** The LLM may fail to map abstract natural language concepts to the real state of the world (e.g., not knowing where the "red block" is). **Mitigation:** Use VLMs and perception systems to provide the LLM with metric coordinates and real-time visual feedback. **3. Over-Reliance:** Relying on the LLM for low-level control results in latency and a lack of reactivity to real-time events (e.g., an unexpected collision). **Mitigation:** Use the LLM only for high-level planning and leave reactive, low-latency control to the robot's local control system. **4. Prompt Complexity:** Long and excessively detailed prompts can confuse the LLM or reach the context limit. **Mitigation:** Use the *chain-of-thought* technique to decompose the task and provide the tools (APIs) clearly and concisely in the system prompt. **5. Debugging Difficulty:** The code or plan generated by the LLM can be difficult to debug, since the logic is generated dynamically. **Mitigation:** Require the LLM to generate well-commented code and use a robust *logging* system to trace the plan's execution.

## URL
[https://arxiv.org/html/2510.05547v1](https://arxiv.org/html/2510.05547v1)
