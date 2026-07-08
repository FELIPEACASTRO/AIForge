# Physics Problem Solving Prompts

## Description
Prompt Engineering for Physics Problem Solving is a specialized technique that uses large language models (LLMs) to assist in the analysis, solution, and explanation of complex problems across various areas of physics, such as mechanics, thermodynamics, electromagnetism, and quantum physics. It involves the careful structuring of the *prompt* to provide context, define the model's role (e.g., tutor, researcher, problem solver), specify the output format (e.g., step by step, conceptual analysis, simulation code), and incorporate advanced techniques such as *Chain-of-Thought* (CoT) or *Tree-of-Thought* (ToT) to improve accuracy and reasoning. The goal is to transform the LLM from a simple text generator into a technical and scientific reasoning tool, overcoming the limitations of models that tend to "hallucinate" or skip logical steps in complex calculations. Recent research (2023-2025) indicates that combining structured *prompting* with *Reinforcement Learning with Human-AI Feedback* (RLHF/RLAIF) is the key to improving LLM performance in this domain [1].

## Examples
```
**1. Structured Solution (CoT):**
"You are a physicist. Solve the following kinematics problem using the *Chain-of-Thought* (CoT) method. Provide the final answer in meters per second.
Problem: A car accelerates from 10 m/s to 30 m/s in 5 seconds. What is the average acceleration and the distance traveled?
Steps: 1. Known/unknown variables. 2. Formulas. 3. Acceleration calculation. 4. Distance calculation. 5. Final answer."

**2. Conceptual and Comparative Analysis:**
"Explain the concept of wave-particle duality to a high school student. Then, compare the approaches of Bohr and de Broglie, highlighting the conceptual differences. Use everyday analogies to aid understanding."

**3. Simulation and Code Generation:**
"Generate Python code (using the `numpy` or `scipy` library) to simulate the motion of a projectile launched at 45 degrees with an initial velocity of 20 m/s. The code should calculate and plot the trajectory (position vs. time) and determine the maximum range. Do not include explanations in the code, only the functional code."

**4. Solution Review and Critique:**
"Analyze the following solution to an RC circuit problem and identify whether there are conceptual or calculation errors. If there are, correct them and provide the correct step-by-step solution.
Incorrect Solution: [Insert here a solution with an error, e.g., forgetting to convert units or using the wrong formula]."

**5. Experiment Planning:**
"Propose a detailed experimental *setup* to demonstrate Faraday's Law of Induction in an undergraduate physics laboratory. The *prompt* should include: 1. List of materials. 2. Step-by-step procedure. 3. Variables to be measured. 4. Expected graph of the results."

**6. Formula Derivation:**
"Derive Bernoulli's equation from the principles of energy conservation and the work-energy theorem. Present the derivation in LaTeX format for easy visualization and include a brief explanation of each simplification step."

**7. Symbolic Solution:**
"Solve the following one-dimensional inelastic collision problem. Provide the final velocity of the system in terms of the masses ($m_1$, $m_2$) and the initial velocities ($v_{1i}$, $v_{2i}$). Do not use numerical values, only symbolic manipulation."
```

## Best Practices
The best practices for creating effective *prompts* in physics problem solving focus on maximizing reasoning accuracy and solution clarity.

1.  **Define the Role and Knowledge Level:** Begin the *prompt* by instructing the LLM to assume a specific role (e.g., "You are a university-level physics professor") and to adapt the language and depth of the explanation to the target audience (e.g., "Explain as if to a high school student").
2.  **Problem-Solving Structure (CoT/ToT):** Require the model to follow a logical, sequential structure. The use of techniques such as *Chain-of-Thought* (CoT) or *Tree-of-Thought* (ToT) is crucial. Explicitly ask: "First, list the known and unknown variables. Second, state the relevant physical principle. Third, present the formula. Fourth, substitute the values and calculate. Fifth, provide the final answer with units."
3.  **Specify Units and Format:** Always include the desired units of measurement in the answer (e.g., "Provide the answer in Newtons and the energy in Joules"). If necessary, request the result in a specific format, such as Python code for simulation or a data table.
4.  **Provide Context and Data:** Include all numerical data, constants, and any constraints of the problem. For complex problems, provide examples of solved problems (*Few-Shot Prompting*) or reference internal documents (if the LLM has that capability).
5.  **Validation and Critique:** Ask the LLM to validate its own answer. For example: "After solving, review the solution and identify a common mistake a student might make when trying to solve this problem."

## Use Cases
*Physics Problem Solving Prompts* are applicable in various educational, research, and development scenarios:

*   **Education and Tutoring:** Generating step-by-step solutions to homework problems, creating detailed lesson plans, and developing conceptual explanations adapted to different learning levels (from elementary school to graduate level).
*   **Research and Development (R&D):** Assisting with literature review, summarizing complex scientific articles (e.g., on computational physics or astrophysics), and identifying knowledge gaps in a specific field.
*   **Experimental Design:** Creating lists of materials, laboratory procedures, and data-validation *checklists* for physics experiments, ensuring compliance with educational and safety standards.
*   **Assessment and Content Creation:** Generating multiple-choice or essay questions for exams, and analyzing student responses to identify error patterns and misunderstood concepts.
*   **Simulation and Modeling:** Generating code *scripts* (Python, MATLAB, etc.) to model physical phenomena, such as motion of bodies, fluid dynamics, or quantum simulations, accelerating the prototyping process.

## Pitfalls
Common errors when using LLMs for physics problems generally stem from a lack of rigor and overreliance on the model's computational ability.

*   **Hallucination of Formulas and Constants:** The LLM may cite incorrect or nonexistent formulas or values of physical constants. **Pitfall:** Not verifying the cited formulas.
*   **Unit and Conversion Errors:** The model may mix units (e.g., using centimeters instead of meters) or fail to convert units consistently throughout the calculation. **Pitfall:** Not specifying the input and output units in the *prompt*.
*   **Skipping Logical Steps (Superficial Reasoning):** Instead of solving the problem, the LLM may provide the final answer (correct or incorrect) without the reasoning process, which is useless for teaching purposes. **Pitfall:** Not explicitly requiring the *Chain-of-Thought* (CoT) method.
*   **Incorrect Interpretation of Context:** The model may fail to correctly interpret the boundary conditions or the physical constraints implicit in the problem (e.g., zero friction, elastic vs. inelastic collision). **Pitfall:** Not providing a complete and clear physical context.
*   **Limitations in Complex Symbolic Calculations:** Although LLMs are good at algebra, problems involving vector calculus or complex partial differential equations can lead to symbolic manipulation errors. **Pitfall:** Blindly trusting long derivations without validation.

## URL
[https://clickup.com/p/ai-prompts/physics-problem-solving](https://clickup.com/p/ai-prompts/physics-problem-solving)
