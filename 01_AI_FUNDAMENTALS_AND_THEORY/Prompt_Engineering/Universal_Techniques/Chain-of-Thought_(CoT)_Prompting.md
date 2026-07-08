# Chain-of-Thought (CoT) Prompting

## Description

**Chain-of-Thought (CoT) Prompting** is a prompt engineering technique that enables Large Language Models (LLMs) to perform complex reasoning by guiding them to generate a sequence of intermediate reasoning steps before providing the final answer. This approach mimics the human thought process, breaking complex problems down into manageable steps. CoT is particularly effective for tasks that require multi-step reasoning, such as mathematics, logic, and common sense. The original technique (Few-Shot CoT) requires input/output examples with the explicit "chain of thought," but variations such as **Zero-Shot CoT** (adding "Let's think step by step" or "Think step by step") have made it accessible without the need for examples. Recent developments (2025) include **Layered CoT** (multi-layer reasoning with review), **Trace-of-Thought** (for smaller models), and **LongRePS** (for reasoning over long contexts). However, CoT can increase latency and cost due to the larger number of *tokens* generated, and its effectiveness is most pronounced in models with more than 100 billion parameters.

## Statistics

- **Performance Improvement (PaLM 540B):**
    - **GSM8K (Mathematics):** Accuracy improved from 55% to **74%** (+19%).
    - **SVAMP (Mathematics):** Accuracy improved from 57% to **81%** (+24%).
    - **Symbolic Reasoning:** Accuracy improved from ~60% to **~95%** (+35%).
- **Scale Limitation:** The CoT technique only produces significant performance gains when used with models of **~100 Billion parameters** or more. Smaller models may see reduced accuracy.
- **Citations:** The original paper, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022), is one of the most cited in the LLM field, with more than **21,800 citations** (as of 2025).
- **Cost/Latency:** CoT increases the number of *tokens* generated, resulting in higher latency and inference cost in production. In more recent reasoning models, the accuracy gains may be marginal (2-3%), making the cost/benefit trade-off a critical factor.

## Features

- **Multi-Step Reasoning:** Allows LLMs to break complex problems down into sequential logical steps.
- **Transparency and Auditability:** The generated chain of thought offers visibility into how the model arrived at its answer, increasing trust and enabling debugging.
- **Performance Improvement:** Significantly increases accuracy in reasoning tasks, especially in large models.
- **Advanced Variations:** Includes Zero-Shot CoT (without examples), Automatic CoT (Auto-CoT, for sampling demonstrations), and more recent techniques such as Layered CoT, Trace-of-Thought, and LongRePS.
- **Capability Emergence:** It is an emergent capability that manifests in large-scale models (typically >100B parameters).

## Use Cases

- **Mathematical Problem Solving:** Complex arithmetic and algebraic reasoning tasks (e.g., the GSM8K and SVAMP benchmarks).
- **Common Sense Reasoning:** Solving questions that require inference and multi-step logic (e.g., the CSQA benchmark).
- **Symbolic Reasoning:** Tasks involving the manipulation of symbols and logical rules.
- **Planning and Decision Making:** Simulating planning steps for AI agents and decision-making systems.
- **Customer Service (Gen-AI Backed):** Chatbots that need to analyze user intent, consult multiple data sources, and formulate a structured response.
- **High-Risk Applications (Layered CoT):** Use in healthcare or finance, where reviewing and adjusting multi-layer reasoning is crucial.

## Integration

CoT can be implemented in two main ways:

**1. Few-Shot CoT (CoT with Examples):**
Include in the input prompt one or more examples of question/answer pairs where the answer contains the explicit "chain of thought."

*Prompt Example (Mathematics):*
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. He bought 2 cans of 3 tennis balls each, which is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: The cafeteria had 23 apples originally. They used 20 for lunch, so they were left with 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9.
```

**2. Zero-Shot CoT (Zero-Example CoT):**
Add a simple phrase to the end of the prompt to instruct the model to reason.

*Prompt Example (Common Sense/Logic):*
```
Q: I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. Then I bought 5 more apples and ate 1. How many apples did I have left?

Think step by step.
```
*Best Practices:*
- **Large Models:** Use CoT primarily with large-scale LLMs (typically >100B parameters), as smaller models may generate illogical reasoning.
- **Complex Tasks:** Reserve CoT for tasks that require complex reasoning (mathematics, logic, planning) and avoid it for simple tasks, where it adds unnecessary latency and cost.
- **Variations:** Try Zero-Shot CoT first for its simplicity. For critical tasks, consider using techniques such as **Self-Consistency** (generating multiple chains of thought and choosing the most common answer) to increase robustness.

## URL

https://arxiv.org/abs/2201.11903