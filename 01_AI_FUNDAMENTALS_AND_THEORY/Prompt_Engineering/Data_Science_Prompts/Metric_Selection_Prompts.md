# Metric Selection Prompts

## Description
"Metric Selection Prompts" is an advanced Prompt Engineering technique that focuses on **explicitly integrating the evaluation criteria and success metrics** within the prompt itself. Instead of merely instructing the Large Language Model (LLM) about the task, the prompt also instructs it on **how** the output should be measured and evaluated. This technique is fundamental in the field of **LLM Evaluation** and in the development of **Model-as-a-Judge** approaches, where an LLM is used to evaluate the quality of another LLM's output based on specific metrics such as **Relevance**, **Coherence**, **Fluency**, **Groundedness** (Adherence to Source), and **Similarity** [1] [2]. The goal is to ensure that the model's output not only completes the task but also meets a measurable, predefined quality standard, aligning the model's performance with business or technical objectives [3].

## Examples
```
**Example 1: Groundedness Evaluation (Adherence to Source)**
```
**Instruction:** You are an LLM evaluator. Your task is to determine the "Groundedness" of the Generated Response relative to the Provided Context.
**Context:** [Excerpt from a source document]
**Question:** [Original user question]
**Generated Response:** [LLM output to be evaluated]
**Evaluation Metric (Groundedness):** Is the response verifiable and supported *only* by the Provided Context?
**Scale:** 1 (No part is supported) to 5 (The entire response is directly supported).
**Output (JSON):** {"Groundedness_Score": [1-5], "Justification": "..."}
```

**Example 2: Relevance Evaluation**
```
**Instruction:** Evaluate the "Relevance" of the Generated Response to the Original Question.
**Question:** What is the impact of generative AI on the financial sector?
**Generated Response:** Generative AI is transforming the financial sector by automating fraud detection and personalizing customer service.
**Evaluation Metric (Relevance):** Does the response directly address the topic of the question and capture the key points?
**Scale:** 1 (Irrelevant) to 5 (Perfectly Relevant).
**Output:** Relevance Score: [1-5].
```

**Example 3: Optimization for Coherence and Fluency**
```
**Instruction:** Rewrite the following text to maximize "Coherence" and "Fluency".
**Original Text:** "The healthcare sector, AI, is changing. Diagnoses are now faster, and patients, treatment, is personalized."
**Optimization Metric:** The output should have smooth flow, impeccable grammar, and logical transitions, as if written by a human expert.
**Output:** [Rewritten text optimized for Coherence and Fluency]
```

**Example 4: Metric Selection for a Summarization Task**
```
**Instruction:** Summarize the article below in 3 sentences. The success metric for this summary is **Completeness**.
**Article:** [Body of the article]
**Success Criterion (Completeness):** The summary must cover the 3 main points of the article.
**Output:** [3-sentence summary]
```

**Example 5: Prompt for Format Adherence Metric**
```
**Instruction:** Generate a list of 5 title ideas for a blog post about "Prompt Engineering".
**Evaluation Metric (Format Adherence):** Each title must be at most 60 characters and formatted as a numbered list.
**Output:**
1. [Title 1]
2. [Title 2]
3. [Title 3]
4. [Title 4]
5. [Title 5]
```

**Example 6: Prompt for Tone Metric (Subjective)**
```
**Instruction:** Respond to the customer complaint with an apology message.
**Evaluation Metric (Tone):** The tone of the response should be **Empathetic** (score 5/5) and **Professional** (score 5/5). Avoid overly formal or robotic language.
**Output:** [Response to the customer]
```

**Example 7: Prompt for Similarity Metric (Paraphrasing)**
```
**Instruction:** Paraphrase the following sentence.
**Original Sentence:** "Artificial intelligence is revolutionizing the way we interact with technology."
**Evaluation Metric (Semantic Similarity):** The output should retain the core meaning of the original sentence, but use completely different vocabulary and sentence structure.
**Output:** [Paraphrased sentence]
```
```

## Best Practices
**1. Define the Metric Before the Prompt:** Before writing the prompt, determine the success metric (e.g., "Accuracy", "Relevance", "Coherence"). The prompt should be designed to optimize that metric.
**2. Incorporate the Evaluation Criterion:** Include in the prompt a clear section that defines how the output will be evaluated. For example, "The response should be evaluated on a scale of 1 to 5 for 'Groundedness' (Adherence to Source)".
**3. Use Evaluator Models (Model-as-a-Judge):** Instead of merely generating the response, use the prompt to instruct an LLM to act as an evaluator, comparing another model's output against the success criterion.
**4. Provide Context and Ground Truth:** For metrics such as "Groundedness" and "Relevance", the prompt should include the reference context (documents, excerpts) so that the model can verify the fidelity of the information.
**5. Standardize the Output for Evaluation:** Ask the model to format the output in a structured way (JSON, XML) that includes the generated response and the metric score, facilitating automated analysis.

## Use Cases
**1. Automated Prompt Evaluation:** Using an LLM as a judge to evaluate the quality of another LLM's output at scale, replacing or complementing human evaluation (Human-in-the-Loop) [2].
**2. RAG Development (Retrieval-Augmented Generation):** Optimizing prompts for metrics such as **Groundedness** (Adherence to Source) and **Relevance** to ensure that responses are factually correct and based on the retrieved documents [1].
**3. A/B Testing of Prompt Versions:** Comparing different prompt versions (V1 vs. V2) using objective metrics (e.g., Accuracy, Latency) to determine which version offers the best performance for a specific objective [3].
**4. Monitoring Models in Production:** Integrating evaluation metrics into the monitoring workflow to detect quality deviations (drift) in the LLM output over time (e.g., a drop in Coherence or Relevance) [4].
**5. Generation of High-Quality Synthetic Data:** Using prompts that require high scores on metrics such as **Fluency** and **Coherence** to generate synthetic training data that mimics the quality of human language.

## Pitfalls
**1. Poorly Defined Metric:** Using vague terms such as "good quality" or "best response" instead of quantifiable metrics (e.g., "Factual Accuracy > 90%").
**2. Metric Conflict:** Asking the model to optimize metrics that cancel each other out, such as requiring "Maximum Creativity" and "Maximum Adherence to Strict Rules" in the same prompt.
**3. Failure to Standardize the Output:** Not instructing the model to format the metric score in a structured way (JSON, XML), making the automated collection and analysis of evaluation results difficult.
**4. Absence of Context (Ground Truth):** Attempting to evaluate fact-based metrics (such as Groundedness) without providing the source context to the evaluator model.
**5. Excessive Reliance on Subjective Metrics:** Relying solely on subjective metrics (such as "Coherence" or "Fluency") without human validation or without a very detailed and robust evaluation prompt.
**6. Ignoring Computational Cost:** The use of LLMs as judges (Model-as-a-Judge) to evaluate metrics significantly increases the cost and latency of the application, a factor to consider in the prompt design.

## URL
[https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-model-monitoring-generative-ai-evaluation-metrics?view=azureml-api-2](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-model-monitoring-generative-ai-evaluation-metrics?view=azureml-api-2)
