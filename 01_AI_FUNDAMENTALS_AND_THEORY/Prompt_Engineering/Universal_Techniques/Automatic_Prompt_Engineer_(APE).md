# Automatic Prompt Engineering (APE)

## Description
Automatic Prompt Engineering (APE - Automatic Prompt Engineer) is a framework proposed by Zhou et al. (2022) that automates the process of generating and selecting instructions (prompts) for Large Language Models (LLMs). APE frames the instruction generation problem as a natural language synthesis task, treating it as a black-box optimization problem. Instead of a human engineer creating prompts manually by trial and error, APE uses an LLM (the "Inference Model") to generate a set of candidate prompts from provided input and output examples (demonstrations). Then, a second LLM (the "Scoring Model") evaluates the log probability of each candidate prompt producing the desired outputs for the demonstration inputs. The prompt with the highest score (that is, the one that best explains the demonstrations) is selected as the optimized prompt. APE has been shown to be capable of discovering Chain-of-Thought (CoT) reasoning prompts that outperform even human-crafted prompts, such as the famous "Let's think step by step".

## Examples
```
**1. Optimized Prompt Generation for Sentiment Classification**
**Demonstrations (Input/Output):**
- Input: "The movie was spectacular, I loved every minute." Output: "Positive"
- Input: "Delay in delivery and damaged product." Output: "Negative"
- Input: "It is neither good nor bad, just average." Output: "Neutral"
**Inference Prompt for the LLM:** "Generate 5 different prompt instructions that, when given to an LLM, would result in the provided outputs for the corresponding inputs."
**Optimized Candidate Prompt (Example):** "Classify the sentiment of the provided text as 'Positive', 'Negative', or 'Neutral'. Think carefully about the tone before answering."

**2. Optimized Prompt Generation for Mathematical Problem Solving (CoT)**
**Demonstrations (Input/Output):**
- Input: "If a train travels at 60 km/h for 3 hours, what is the distance traveled?" Output: "180 km"
- Input: "What is the result of 15 * 7?" Output: "105"
**Inference Prompt for the LLM:** "Generate 3 prompts that induce step-by-step reasoning to solve math problems."
**Optimized Candidate Prompt (Example):** "Solve the following math problem. To ensure accuracy, work out the solution in a step-by-step approach before providing the final answer."

**3. Optimized Prompt Generation for Summarization**
**Demonstrations (Input/Output):**
- Input: [Long article about AI] Output: [Concise 3-sentence summary]
**Inference Prompt for the LLM:** "Generate 4 prompt instructions to summarize long texts, focusing on conciseness and retention of key information."
**Optimized Candidate Prompt (Example):** "Summarize the following text in no more than 50 words, ensuring that the three main points are preserved. Start with 'In summary,'."

**4. Optimized Prompt Generation for Context-Aware Translation**
**Demonstrations (Input/Output):**
- Input: "The bank is on the river." Output: "O banco está na margem do rio."
- Input: "I need to go to the bank." Output: "Eu preciso ir ao banco (instituição financeira)."
**Inference Prompt for the LLM:** "Generate 2 prompts that instruct the LLM to consider context for word disambiguation in translations."
**Optimized Candidate Prompt (Example):** "Translate the following text from English to Portuguese. Analyze the context of the sentence to choose the most accurate translation for words with multiple meanings."

**5. Optimized Prompt Generation for Entity Extraction**
**Demonstrations (Input/Output):**
- Input: "The meeting with Dr. Silva will be in São Paulo, at the TechCorp headquarters." Output: "Person: Dr. Silva; Location: São Paulo; Organization: TechCorp"
**Inference Prompt for the LLM:** "Generate 3 prompts to extract named entities (Person, Location, Organization) from a text."
**Optimized Candidate Prompt (Example):** "Extract all named entities from the text. Present the result in JSON format with the keys 'Person', 'Location', and 'Organization'. If an entity is not present, use an empty string."
```

## Best Practices
**Focus on the Quality of the Demonstration Data:** The quality and diversity of the input/output examples (demonstrations) provided to the inference LLM are crucial. Poor or inconsistent demonstrations will lead to ineffective candidate prompts.
**Iteration and Refinement:** APE is an iterative process. Do not settle for the first optimized prompt. Use the best-performing prompt as a new starting point to generate and test more variations.
**Inference Model vs. Target Model:** Ideally, use a more powerful LLM (e.g., GPT-4, Claude 3 Opus) as the "Inference Model" to generate candidate prompts and a lighter model or the actual target model for the "Selection" and evaluation, saving costs and time.
**Robust Evaluation Metrics:** Use evaluation metrics that align directly with the task objective (e.g., accuracy for classification, F1-score for entity extraction, ROUGE for summarization). Avoid generic metrics that may not capture the real quality of the output.
**Leverage CoT (Chain-of-Thought):** APE has been shown to be effective in discovering enhanced CoT prompts (such as "Let's work this out in a step by step way..."). Always include the search for prompts that induce step-by-step reasoning.

## Use Cases
**Reasoning Prompt Optimization (CoT):** The most notable use case is the discovery of more effective Chain-of-Thought (CoT) prompts, such as the enhancement of the prompt "Let's think step by step" to "Let's work this out in a step by step way to be sure we have the right answer."
**Prompt Generation for Specific Tasks:** Automating the creation of prompts for Natural Language Processing (NLP) tasks where manual engineering is tedious, such as text classification, named entity recognition (NER), and summarization.
**Adapting Prompts to Different LLMs:** Optimizing prompts for a specific language model (the "Target Model") without needing access to its gradients, treating it as a black box. This is useful for adapting prompts across different LLM APIs.
**Performance Improvement on Benchmarks:** Used in research to reach or surpass the performance of human-crafted prompts on benchmarks such as MultiArith and GSM8K.
**Prompt Optimization in Production Environments:** In production environments, APE can be used to continuously refine prompts based on user feedback data, ensuring that the LLM maintains optimal performance as task requirements evolve.

## Pitfalls
**Dependence on the Quality of the Demonstrations:** APE is highly dependent on the quality and representativeness of the input/output examples provided. Insufficient or noisy demonstrations can lead to optimization for a suboptimal or incorrect prompt.
**High Computational Cost:** The process of generating multiple candidate prompts and then scoring each of them using an LLM (the Scoring Model) can be computationally expensive and time-consuming, especially for large models.
**Risk of Overfitting:** There is a risk that the optimized prompt overfits the demonstration data, resulting in poor performance on unseen test data (weak generalization).
**Implementation Complexity:** Implementing APE requires the orchestration of two LLMs (Inference and Scoring) and the management of a search and evaluation process, which is more complex than manual prompt engineering.
**Limitation to Text Generation Tasks:** Although effective for tasks such as classification and CoT, APE may be less applicable or less efficient for tasks that require complex interactions or the use of external tools.

## URL
[https://arxiv.org/abs/2211.01910](https://arxiv.org/abs/2211.01910)
