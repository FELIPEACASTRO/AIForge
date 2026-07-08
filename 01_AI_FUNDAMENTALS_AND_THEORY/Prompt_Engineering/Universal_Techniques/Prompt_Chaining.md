# Prompt Chaining

## Description
**Prompt Chaining** is an advanced Prompt Engineering technique that consists of decomposing a complex task into a sequence of simpler, more manageable subtasks. The output of one prompt (or step) is used as the input for the subsequent prompt, creating a sequential workflow. This approach is fundamental to improving the **quality**, **reliability**, and **traceability** (observability) of Large Language Model (LLM) responses on multifaceted tasks. By isolating the model's cognitive focus at each step, Prompt Chaining reduces cognitive load, minimizes reasoning errors (hallucinations), and enables the application of iterative refinement mechanisms, mirroring human workflows of drafting, critique, and revision [1] [2]. It is the foundation for strategies such as *Least-to-Most Prompting* and is crucial for production systems that require high accuracy and detailed monitoring.

## Examples
```
**Example 1: Research Synthesis and Article Generation**
1. **Prompt 1 (Extraction):** "Analyze the following text and extract the 5 most important key points, formatting them as a JSON list with the title 'KeyPoints'."
2. **Prompt 2 (Analysis):** "Based on the extracted 'KeyPoints' (INPUT), identify the main thesis and the ideal target audience for an article on the topic. Respond in JSON format with the keys 'MainThesis' and 'TargetAudience'."
3. **Prompt 3 (Generation):** "Using the 'MainThesis' and the 'TargetAudience' (INPUT), write the introduction of a 500-word article in a professional and persuasive tone."

**Example 2: Email Classification and Summarization**
1. **Prompt 1 (Classification):** "Classify the following email (INPUT) into one of the categories: 'Urgent', 'Informational', 'Financial', 'Marketing'. Respond with the category only."
2. **Prompt 2 (Conditional Summary):** "If the email category is 'Urgent' (INPUT), generate a one-sentence summary with the required action. Otherwise, generate a one-paragraph summary."

**Example 3: Code Refactoring**
1. **Prompt 1 (Code Analysis):** "Analyze the following Python code snippet (INPUT) and list 3 areas for performance or readability improvement. Respond as a numbered list."
2. **Prompt 2 (Refactoring):** "Based on the listed improvement areas (INPUT), refactor the original code to incorporate these changes. Preserve the original functionality."

**Example 4: Story Character Creation**
1. **Prompt 1 (Concept):** "Generate a character concept for a fantasy story, including 'Name', 'Race', and 'Occupation'. Respond in JSON."
2. **Prompt 2 (Details):** "Using the 'Name' and the 'Race' (INPUT), write a paragraph detailing the character's background and main motivation."

**Example 5: Sentiment Analysis and Response**
1. **Prompt 1 (Sentiment):** "Analyze the following customer comment (INPUT) and classify the sentiment as 'Positive', 'Neutral', or 'Negative'. Respond with the sentiment only."
2. **Prompt 2 (Response):** "Based on the sentiment ('Negative') (INPUT), write an empathetic customer service response and propose a solution. If the sentiment is 'Positive', write a brief thank-you."
```

## Best Practices
**1. Modular Decomposition:** Break the task into logical, discrete steps. Each prompt in the chain should have a single, well-defined objective. **2. Explicit Data Contracts:** Define a strict output format (schema) for each prompt (preferably JSON or XML) to ensure that the output is a clean, predictable input for the next step. **3. Minimize Context:** Pass only the essential information to the next prompt. Excessive context increases cost (tokens) and can introduce noise or focus drift. **4. Intermediate Validation:** Implement checks (deterministic or LLM-as-a-judge-based) after each step to ensure the quality and compliance of the output before proceeding. **5. Iteration and Refinement:** Use the chain to simulate a cycle of *drafting*, *critique*, and *revision*, where one prompt evaluates the output of the previous one and the following prompt refines it.

## Use Cases
**1. Multi-Instruction Tasks:** Decompose tasks that combine data extraction, transformation, analysis, and visualization into ordered steps. **2. Agent Workflows:** It is the backbone of autonomous agent architectures, where the model plans, executes, and reflects on actions in an iterative cycle. **3. Document Synthesis and Review:** Create a draft, critique the draft against a set of rules, and then refine the final draft. **4. Complex Reasoning (Enhanced Chain-of-Thought):** Use the output of one prompt to force the model to generate an explicit reasoning step (e.g., "Think step by step") and use that reasoning as input for the final answer. **5. Structured Content Generation:** Create an outline, generate the content of each section, and finally review and format the complete document.

## Pitfalls
**1. Token Sprawl (Cost Explosion):** Chaining increases the total number of tokens processed, since the output of each step is reintroduced as input. This can lead to a significant increase in cost and latency. **2. Error Propagation (Cascading Error):** An error or hallucination in an initial prompt is passed to subsequent prompts, contaminating the entire chain and leading to an incorrect final result. **3. Excessive Latency:** The sequential execution of multiple prompts increases the total response latency. For real-time tasks, this can be unacceptable. **4. Insufficient Context:** Attempting to minimize context to save tokens can result in the loss of crucial information needed for reasoning in later steps. **5. Rigid Schema Dependency:** If a prompt's output format (schema) fails, the chain can break. It is crucial to have robust validation and *retry* mechanisms to handle formatting failures.

## URL
[https://www.getmaxim.ai/articles/prompt-chaining-for-ai-engineers-a-practical-guide-to-improving-llm-output-quality/](https://www.getmaxim.ai/articles/prompt-chaining-for-ai-engineers-a-practical-guide-to-improving-llm-output-quality/)
