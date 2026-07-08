# ReAct (Reasoning + Acting)

## Description
**ReAct (Reasoning + Acting)** is a *prompting* paradigm that combines chain-of-thought reasoning (*Chain-of-Thought* - CoT) with the ability to interact with external tools (*Acting*). The large language model (LLM) is prompted to generate verbal reasoning traces (`Thought`) and task-specific actions (`Action`) in an interleaved manner. The reasoning helps the model to induce, track, and update action plans, as well as handle exceptions, while the actions allow it to interact with external sources (such as search APIs or knowledge bases) to gather additional information. This creates an iterative cycle of **Thought -> Action -> Observation -> Thought**, which improves interpretability, factual fidelity (reducing hallucinations), and the ability to solve complex decision-making and reasoning tasks [1] [2]. ReAct is the foundation for building **AI Agents** capable of planning and executing tasks in dynamic environments.

## Examples
```
**Example 1: Question Answering with Web Search**

```
Question: Who was the first human to set foot on the Moon and what is their date of birth?

Thought: I need to find the name of the first human to set foot on the Moon.
Action: Search[first human to set foot on the Moon]
Observation: The first human to set foot on the Moon was Neil Armstrong.
Thought: Now that I have the name, I need to find Neil Armstrong's date of birth.
Action: Search[Neil Armstrong date of birth]
Observation: Neil Armstrong was born on August 5, 1930.
Thought: I have all the information needed to answer the question.
Final Answer: The first human to set foot on the Moon was Neil Armstrong, born on August 5, 1930.
```

**Example 2: Solving Complex Math Problems**

```
Question: What is the result of (15 * 3) + (45 / 5)?

Thought: I first need to compute the multiplication (15 * 3) and the division (45 / 5) and then add the results.
Action: Calculator[15 * 3]
Observation: 45
Thought: The result of the multiplication is 45. Now I will compute the division.
Action: Calculator[45 / 5]
Observation: 9
Thought: The result of the division is 9. Now I will add the two results.
Action: Calculator[45 + 9]
Observation: 54
Thought: The final result is 54.
Final Answer: 54
```

**Example 3: Fact Verification**

```
Question: Is the capital of Australia Sydney?

Thought: I need to verify what the official capital of Australia is.
Action: Search[capital of Australia]
Observation: The capital of Australia is Canberra.
Thought: The initial information is incorrect. The capital is Canberra, not Sydney.
Final Answer: No, the capital of Australia is Canberra.
```

**Example 4: Trip Planning (with Weather Forecast Tool)**

```
Question: What is the weather forecast for Paris tomorrow and should I bring an umbrella?

Thought: I need to get the weather forecast for Paris for tomorrow.
Action: Weather_API[Paris, tomorrow]
Observation: Forecast: 15°C, light rain.
Thought: The forecast indicates light rain. I should recommend an umbrella.
Final Answer: The weather forecast for Paris tomorrow is 15°C with light rain. Yes, you should bring an umbrella.
```

**Example 5: Database Interaction (Database Query)**

```
Question: How many customers in the database are over 60 years old?

Thought: I need to run a SQL query to count the customers older than 60.
Action: SQL_Query[SELECT COUNT(*) FROM clientes WHERE idade > 60]
Observation: 1245
Thought: The query returned 1245 customers.
Final Answer: There are 1245 customers over 60 years old in the database.
```

**Example 6: Solving Logic Problems (with Search Tool)**

```
Question: If the price of Bitcoin rises 10% and Ethereum rises 5%, what will be the total value of a portfolio with 1 BTC (R$ 300,000) and 10 ETH (R$ 15,000 each)?

Thought: I need to compute the new value of Bitcoin and Ethereum.
Action: Calculator[300000 * 1.10]
Observation: 330000
Thought: The new value of Bitcoin is R$ 330,000. Now, the new value of Ethereum.
Action: Calculator[15000 * 1.05]
Observation: 15750
Thought: The new value of 1 ETH is R$ 15,750. The portfolio has 10 ETH.
Action: Calculator[330000 + (10 * 15750)]
Observation: 487500
Thought: The total value of the portfolio will be R$ 487,500.
Final Answer: The total value of the portfolio will be R$ 487,500.
```
```

## Best Practices
**1. Define the Tools:** The prompt should clearly list the external tools available to the model (e.g., `Search`, `Calculator`, `Wikipedia`). **2. Structure the ReAct Cycle:** Use a few-shot prompt format that demonstrates the `Thought`, `Action`, `Observation`, and `Final Answer` cycle. **3. Be Explicit in the Reasoning (`Thought`):** The model should be instructed to articulate its reasoning before each action, explaining why the action is necessary and how it relates to the final goal. **4. Use Observations for Iteration:** The tool's output (`Observation`) should be used as input for the next `Thought`, allowing the model to correct errors, gather more information, or change strategy. **5. Use a Stop Token (`Final Answer`):** The model should be instructed to use a clear stop token (e.g., `Final Answer:`) to indicate that the task is complete and the final answer is ready.

## Use Cases
**1. AI Agents:** Building autonomous agents capable of planning, executing, and monitoring complex tasks in digital environments (e.g., web navigation, software automation). **2. Grounded Question Answering (Grounded QA):** Answering factual questions using external sources (e.g., Wikipedia, databases) to ensure the answer is accurate and free of hallucinations. **3. Fact Checking:** Automating the process of verifying claims, using search tools to cross-reference information and validate the truthfulness of a statement. **4. Reasoning Problem Solving:** Solving problems that require multiple logical steps and the use of specific tools (e.g., mathematics, programming, logic). **5. Interaction with APIs and Systems:** Allowing the LLM to interact with external systems (e.g., weather APIs, SQL databases, e-commerce tools) to perform real-world actions. **6. Games and Decision Environments:** In environments such as ALFWorld and WebShop, ReAct allows the agent to make sequential decisions and interact with the environment to achieve a goal (e.g., buying a product online).

## Pitfalls
**1. Inadequate Tool Definition:** Failing to clearly define the available tools or their signatures (how to call the tool and what to expect in return) can lead the model to generate invalid or ineffective actions. **2. Action Hallucination:** The model may "hallucinate" tools that do not exist or actions that are not supported by the environment, resulting in execution failures. **3. Insufficient Reasoning:** A `Thought` that is too brief or generic may fail to guide the model effectively, leading to an inefficient Action-Observation cycle or a deviation from the main goal. **4. Excessive Dependence on Tools:** Relying on tools for every step, even for simple calculations or information the LLM already possesses, can increase latency and cost. **5. Error Propagation:** If an initial `Observation` is incorrect or ambiguous, the subsequent reasoning may build on that flawed information, propagating the error to the `Final Answer`. **6. Context Limitation:** In complex tasks with many ReAct cycles, the history of `Thought`/`Action`/`Observation` may exceed the LLM's context window, causing it to "forget" the previous reasoning or the goal.

## URL
[https://react-lm.github.io/](https://react-lm.github.io/)
