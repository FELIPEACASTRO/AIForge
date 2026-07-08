# Meta-Prompting

## Description
**Meta-Prompting** is an advanced Prompt Engineering technique that focuses on the structural and syntactic aspects of tasks, rather than content-specific details. In its formal definition, a Meta Prompt is a structured, example-agnostic prompt that provides a scaffold for capturing the reasoning structure of a category of tasks, focusing on *how* the problem should be solved, not on *what* [1]. In a practical context, the technique involves using a Large Language Model (LLM), usually a more capable one (the Meta-LLM), to generate, refine, or optimize prompts for another LLM (the target LLM), ensuring that the final prompt is more effective and forces a high-quality output with a consistent structure [2]. Its main characteristics include being structure-oriented, syntax-focused, using abstract examples, and having a categorical approach, which makes it more token-efficient and fairer for model comparison than Few-Shot Prompting [3].

## Examples
```
**Example 1: Prompt Optimization (General)**

**Meta-Prompt (Input for the Meta-LLM):**
```
"""
You are a senior-level Prompt Engineer. Your task is to analyze the 'INITIAL PROMPT' below and rewrite it to maximize the quality and creativity of an LLM's response.

Instructions for the rewrite:
1. Add a detailed persona (e.g., 'award-winning science fiction writer').
2. Define the output format (e.g., 'a 500-word short story').
3. Include style constraints (e.g., 'melancholic tone, with a twist at the end').
4. The final prompt must be self-contained and must not contain these instructions.

INITIAL PROMPT:
"Write about a robot that feels lonely."

Return only the optimized prompt.
"""
```

**Example 2: Output Structure (JSON Schema)**

**Meta-Prompt (Input for the Meta-LLM):**
```
"""
Create a prompt for an LLM that ensures the output is a to-do list for a software development project, strictly in JSON format.

The prompt should include:
1. The instruction to use the JSON format.
2. The mandatory JSON schema: an array of objects, where each object has the keys 'id' (integer), 'task' (string), 'priority' (string: 'High', 'Medium', 'Low'), and 'status' (string: 'Pending', 'In Progress', 'Done').
3. The task to be processed: "Plan the launch of a new productivity app."

Return only the final prompt, including the JSON schema as part of the instructions.
"""
```

**Example 3: Task Decomposition (Forced Chain-of-Thought)**

**Meta-Prompt (Input for the Meta-LLM):**
```
"""
Create a prompt that instructs the LLM to solve the following logic problem, mandatorily using the Chain-of-Thought (CoT) technique before providing the final answer.

Problem: "If a train travels at 60 km/h and covers 300 km, and a second train travels at 90 km/h and covers 450 km, which train arrived first if both departed at the same time?"

The prompt should have two clear sections: 'Reasoning Steps' and 'Final Answer'.
"""
```

**Example 4: Persona Creation and Language Constraints**

**Meta-Prompt (Input for the Meta-LLM):**
```
"""
Generate a prompt for an LLM that instructs it to act as a 'Senior Cybersecurity Consultant'.

The prompt should impose the following constraints:
1. The tone should be extremely formal and technical.
2. The vocabulary should be specialized in information security (e.g., 'cryptography', 'attack vector', 'zero-day').
3. The task is: "Explain the security risks of a public Wi-Fi network."

Return only the persona and task prompt.
"""
```

**Example 5: Meta-Prompting for Multilingual Prompt Generation**

**Meta-Prompt (Input for the Meta-LLM):**
```
"""
Translate and optimize the 'INITIAL PROMPT' from English into Portuguese.

Mandatory optimization:
1. The Portuguese prompt should instruct the LLM to format the final response in a Markdown table.
2. The table should have columns for 'Concept' and 'Definition'.

INITIAL PROMPT:
"Explain the concept of quantum entanglement in simple terms."

Return only the optimized prompt in Portuguese.
"""
```
```

## Best Practices
**Focus on Structure and Syntax**: Prioritize defining the format, persona, and output constraints (JSON, tables, CoT) in the Meta-Prompt, rather than just the content. **Use a Superior Meta-LLM**: Whenever possible, use a more advanced and capable language model (the Meta-LLM) to generate or refine prompts for less powerful models, ensuring higher quality and complexity in the final prompt. **Clarity and Specificity**: The Meta-Prompt should be extremely clear about the optimization objective and the constraints that the generated prompt should impose on the target LLM. **Task Decomposition**: Use Meta-Prompting to force the decomposition of complex problems into clear reasoning steps (such as Chain-of-Thought), improving the accuracy of the response.

## Use Cases
**Prompt Optimization**: Generating more effective and detailed prompts from simple or vague ones. **Output Standardization**: Forcing the target LLM to produce responses in strict and consistent formats, such as JSON, XML, or Markdown tables, essential for integration with software systems. **Decomposition of Complex Problems**: Structuring the prompt to require the LLM to follow a step-by-step reasoning process (Chain-of-Thought), improving accuracy in logic, mathematics, and coding tasks. **Creating Prompts for Multiple Models**: Generating optimized prompts for different LLM models (e.g., one prompt for a fast model and another for a slower but more accurate model). **Persona and Style Refinement**: Defining and imposing a specific persona, tone, and vocabulary for the target LLM's response.

## Pitfalls
**Vague Meta-Prompt**: If the Meta-Prompt is not specific about the optimization objective or the output constraints, the generated prompt may not be significantly better than the original. **High Cost and Latency**: Using a more capable LLM (Meta-LLM) to generate the prompt increases the cost and latency of the overall call, since two API calls are required (one to generate the prompt and another to execute it). **Dependence on the Meta-LLM**: The quality of the final prompt is highly dependent on the capability and performance of the LLM used to generate the Meta-Prompt. **Presumed Innate Knowledge**: The technique assumes that the target LLM possesses the innate knowledge required for the task; performance may deteriorate on very unique or novel tasks, resembling Zero-Shot Prompting [3].

## URL
[https://arxiv.org/html/2311.11482v7](https://arxiv.org/html/2311.11482v7)
