# Self-Consistency

## Description
Self-Consistency (SC) is an advanced Prompt Engineering technique that enhances the reasoning capability of Large Language Models (LLMs), especially in complex arithmetic and common-sense reasoning tasks. Proposed as a decoding strategy that replaces naive greedy decoding, SC works by generating multiple and diverse 'Chain-of-Thought' (CoT) reasoning paths for a single question. Instead of accepting the first generated answer, the technique aggregates the results of all sampled reasoning paths and selects the final answer through a majority vote (or another consistency metric). This capitalizes on the idea that, although a complex problem may have several valid reasoning paths, they should all lead to a single correct answer, significantly increasing the accuracy and robustness of the solution. The most advanced variant, Universal Self-Consistency (USC), uses the LLM itself to act as a judge, selecting the most consistent answer among the candidates.

## Examples
```
## Prompt Examples (Self-Consistency)

**General Instruction:** To apply Self-Consistency, the prompt must be sent to the LLM *N* times (where N is the desired number of samples, typically 5 to 10), and the final answer is determined by the majority vote among the *N* answers.

### Example 1: Arithmetic Reasoning (GSM8K)

**Prompt (to be repeated N times):**

```
Q: John has 15 apples. He eats 3 and gives half of the remaining ones to Maria. How many apples did Maria receive?

Let's think step by step:

[Space for the LLM to generate the reasoning and the final answer. The prompt should be designed to force CoT reasoning.]
```

**Process:**
1. Send the prompt 10 times.
2. Collect the 10 final answers (e.g.: 7, 6, 6, 6, 7, 6, 6, 7, 6, 6).
3. The final answer is 6 (majority vote).

### Example 2: Common-Sense Reasoning (StrategyQA)

**Prompt (to be repeated N times):**

```
Q: Is it possible for an adult to regularly sleep more than 12 hours a day without having a medical condition?

Let's think step by step:

[Space for the LLM to generate the reasoning and the final answer (Yes/No).] 
```

**Process:**
1. Send the prompt 5 times.
2. Collect the 5 final answers (e.g.: No, Yes, No, No, No).
3. The final answer is No.

### Example 3: Universal Self-Consistency (USC) - Judgment Stage

**Judgment Prompt (sent to a 'Judge' LLM after generating N answers):**

```
You are a consistency judge. Analyze the following N answers to the question 'What is the capital of Bhutan?' and select the most consistent and correct one. Justify your choice.

Candidate Answers:
1. Thimphu, as it is the political and economic center.
2. Paro, due to its international airport.
3. Thimphu, the largest city and official capital.

Final Answer and Justification:
```

### Example 4: Complex Classification Task

**Prompt (to be repeated N times):**

```
Q: Classify the following text as 'News', 'Opinion' or 'Advertisement', and justify your choice:

[TEXT: 'The new smartphone X is the fastest on the market, with a camera that redefines mobile photography. Available now at an unbeatable price.']

Let's think step by step:

[Space for the LLM to generate the reasoning and the final classification.]
```

### Example 5: Solving Logic Puzzles

**Prompt (to be repeated N times):**

```
Q: If the code for 'SOL' is 191512 and the code for 'LUA' is 122101, what is the code for 'MAR'?

Let's think step by step:

[Space for the LLM to generate the reasoning and the final code.]
```
```

## Best Practices
1. **Increase Diversity:** Use a higher temperature parameter (e.g.: 0.7 to 1.0) when generating the reasoning paths to ensure that the samples are diverse.
2. **Sufficient Sampling:** The number of samples (N) should be large enough (typically N=5 to N=10) for the majority vote to be statistically significant.
3. **CoT is Fundamental:** Self-Consistency must be combined with Chain-of-Thought (CoT) to force the LLM to generate the reasoning steps that lead to the answer.
4. **Focus on the Final Answer:** Voting should be applied only to the final answer extracted from each reasoning path, and not to the reasoning itself.
5. **Use USC for Greater Accuracy:** For critical tasks, use Universal Self-Consistency (USC), where a second LLM is used to judge and select the best answer, instead of a simple majority vote.

## Use Cases
1. **Arithmetic and Mathematical Reasoning:** Solving complex word problems (such as the GSM8K benchmark).
2. **Common-Sense Reasoning:** Answering questions that require inference and world knowledge (such as the StrategyQA benchmark).
3. **Fact and Data Verification:** Increasing confidence in factual answers, especially in domains with high ambiguity.
4. **Classification and Sentiment Analysis:** Improving accuracy in complex classification tasks, where context can lead to different interpretations.
5. **AI Agent Applications:** As a robustness verification mechanism for decision-making in autonomous agents.

## Pitfalls
1. **High Computational Cost:** Requires N times more LLM API calls, significantly increasing cost and latency.
2. **Dependence on CoT:** The effectiveness of SC is intrinsically linked to the quality of the reasoning paths generated by CoT. If CoT fails, SC will also fail.
3. **Flawed Majority Vote:** In cases of highly dispersed answers, the majority vote may not be clear or may converge to an incorrect answer if the majority of reasoning paths share the same subtle error.
4. **Implementation Difficulty:** Requires a post-processing step to extract the final answer from each output and perform the voting, which is more complex than simple greedy decoding.
5. **Not Suitable for Creative Tasks:** Not recommended for tasks that value output diversity (e.g.: poetry generation, brainstorming), since its goal is to converge to a single correct answer.

## URL
[https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
