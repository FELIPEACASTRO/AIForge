# Reflexion

## Description
The **Reflexion** technique is an innovative *framework* that enhances the ability of Language Agents (LLM Agents) to solve complex tasks through a process of **self-reflection** and **verbal reinforcement**. Instead of adjusting the model's weights (as in traditional reinforcement learning), Reflexion uses the LLM itself to generate *linguistic feedback* on its previous attempts, transforming that feedback into a dynamic memory and a guide for the next iteration.

The process is typically iterative and involves:
1. **Action Attempt:** The agent attempts to solve the task (e.g., generate code, answer a complex question, plan a sequence of actions).
2. **Observation and Evaluation:** The agent receives a result (e.g., a compilation error, an incorrect answer, feedback from the environment).
3. **Reflection:** The agent uses the *feedback* and the history of the attempt to generate a critical **reflection**. This reflection is a textual analysis that identifies the failures, the areas for improvement, and the new strategies to be adopted.
4. **New Attempt:** The generated reflection is added to the agent's context memory and used to guide the next action attempt, resulting in progressively improved performance.

Reflexion can be applied together with other techniques, such as ReAct (Reasoning and Acting), to create more robust and autonomous agents.

## Examples
```
1.  **Example 1: Code Generation (Initial Attempt)**
    *   **Prompt (User):** "Generate a Python function that computes the n-th Fibonacci number recursively and optimized with memoization."
    *   **LLM Response (Attempt 1):** (Code with a logic error or without memoization)
    *   **Prompt (Reflection):** "The previous attempt failed to include memoization, resulting in O(2^n) time complexity. Analyze the code, identify where memoization should be applied (e.g., using a dictionary or `functools.lru_cache`), and generate a new function that meets the optimization requirement."

2.  **Example 2: Problem Solving (Initial Attempt)**
    *   **Prompt (User):** "A train travels at 60 km/h. If it covers 300 km, how long will it take? Show the step-by-step reasoning."
    *   **LLM Response (Attempt 1):** (Correct reasoning, but a final calculation error, e.g., 300/60 = 6 hours)
    *   **Prompt (Reflection):** "Your previous answer contained a calculation error: 300 divided by 60 is 5, not 6. Review the step-by-step reasoning, correct the calculation, and provide the correct final answer. The reflection should be: 'The error was in the final division. 300/60 = 5. The correct answer is 5 hours.'"

3.  **Example 3: Environment Interaction (API Agent)**
    *   **Prompt (User):** "Fetch the current price of Bitcoin from the CoinGecko API and convert it to Brazilian Reais (BRL)."
    *   **LLM Response (Attempt 1):** (Failure in the API call due to an incorrect parameter, e.g., `moeda=Real` instead of `moeda=brl`)
    *   **Prompt (Reflection):** "The API call in the previous attempt returned a 400 error (Bad Request) with the message 'Invalid currency parameter'. The 'Real' parameter is not accepted. Consult the API documentation and adjust the parameter to the correct ISO code for the Brazilian currency (BRL). Repeat the call and provide the result."

4.  **Example 4: Creative and Stylistic Writing**
    *   **Prompt (User):** "Write an opening paragraph for a science fiction short story in the style of H.P. Lovecraft."
    *   **LLM Response (Attempt 1):** (Generic paragraph, without Lovecraft's vocabulary or cosmic atmosphere)
    *   **Prompt (Reflection):** "The generated paragraph is too direct and lacks Lovecraft's dense adjectives and tone of cosmic horror. Reflect on the use of words like 'indescribable', 'eldritch', 'abyssal', and 'insanity'. Rewrite the paragraph, focusing on evoking a sense of dread and unnameable mystery."

5.  **Example 5: Prompt Optimization (Meta-Reflection)**
    *   **Prompt (User):** "Create an engineered prompt for an LLM that instructs it to act as a cybersecurity expert and analyze a code snippet for XSS vulnerabilities."
    *   **LLM Response (Attempt 1):** (Simple prompt, without defining persona, output format, or analysis steps)
    *   **Prompt (Reflection):** "The engineered prompt created is ineffective because it establishes neither a clear persona nor a structured output format. Reflect on prompt engineering best practices (e.g., CoT, persona, JSON format) and rewrite the prompt to be more robust. The new prompt should include: 1) The persona of 'Level 5 Security Expert', 2) The instruction to use 'Chain-of-Thought' reasoning, and 3) The requirement for output in Markdown format with a vulnerability table."

6.  **Example 6: Refining an Argument (Debate)**
    *   **Prompt (User):** "Write a 200-word argument in favor of implementing a 4-day work week."
    *   **LLM Response (Attempt 1):** (Argument focused only on well-being, ignoring productivity and costs)
    *   **Prompt (Reflection):** "The previous argument is weak because it does not address economic and productivity concerns. Reflect on how to integrate data on increased efficiency and reduced operational costs (e.g., energy, office space) to strengthen the thesis. Rewrite the argument to be more balanced and persuasive for a corporate audience."

7.  **Example 7: Data Correction (Factual)**
    *   **Prompt (User):** "List the 5 longest rivers in the world in order of length."
    *   **LLM Response (Attempt 1):** (Incorrect list, placing the Nile River first, when the Amazon is generally accepted as the longest)
    *   **Prompt (Reflection):** "The previous list is outdated or incorrect. Modern geographic research sources indicate that the Amazon River is the longest in the world, surpassing the Nile. Review the list based on the most accepted measurements and correct the order. The reflection should be: 'The factual information about the Amazon and Nile Rivers has been reviewed. The Amazon is the longest. The list will be corrected.'"
```

## Best Practices
*   **Clear Instruction for the Reflection:** The reflection prompt should be specific, asking the model to identify the root cause of the failure, suggest a correction, and formulate a new strategy.
*   **Dynamic Memory:** Keep a concise history of previous attempts and reflections in the prompt context for the new attempt. The reflection should be the most relevant part of the memory.
*   **Controlled Iteration:** Limit the number of iterations to avoid infinite loops or excessive resource consumption. Three to five iterations are usually sufficient.
*   **Combination with ReAct:** Use Reflexion to enhance the "Reasoning" component of ReAct, allowing the agent to learn from its interactions with the environment.
*   **Focus on the Cause of the Error:** Guide the model to reflect on *why* the attempt failed, and not just *what* failed.

## Use Cases
*   **Code Generation:** Agents that attempt to write code, receive compilation errors or unit test failures, reflect on the error, and correct the code.
*   **Solving Complex Problems (e.g., Mathematics, Logic):** The agent attempts to solve a problem, evaluates the answer, identifies a reasoning error, and refines the chain of thought.
*   **Navigation and Interaction in Virtual Environments:** Agents that interact with APIs or game environments, receive *feedback* from the environment (e.g., "invalid action"), reflect on the strategy, and adjust the action plan.
*   **Iterative Content Creation:** Refining an article, script, or creative piece based on internal or external evaluation criteria.

## Pitfalls
*   **Superficial Reflections:** The model may generate generic or superficial reflections that do not lead to real improvements in the next attempt.
*   **Context Accumulation (Context Window Bloat):** The history of attempts and reflections can quickly exceed the LLM's context limit, requiring summarization or memory pruning strategies.
*   **Reflection Loops:** In rare cases, the agent may enter a cycle where the reflection fails to break the error pattern, leading to repetitive and fruitless attempts.
*   **Computational Cost:** The iterative process and the need for multiple LLM calls (attempt + reflection + new attempt) significantly increase cost and response time.

## URL
[https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
