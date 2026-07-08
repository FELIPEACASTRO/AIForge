# Models-Vote Prompting (MVP)

## Description

**Models-Vote Prompting (MVP)** is an advanced *prompt engineering* technique based on **ensemble** and **Few-Shot Learning (FSL)**, designed specifically to improve accuracy in the **Identification and Classification of Rare Diseases** from clinical notes (EHRs). The core methodology involves submitting the same task to multiple Large Language Models (LLMs) and then performing a **majority vote** over the generated responses to determine the final diagnosis. This approach is particularly effective in FSL scenarios, where the scarcity of training data for rare diseases is a significant challenge. MVP also incorporates the use of structured output formats, such as **JSON**, to facilitate automated evaluation and integration with clinical systems.

## Statistics

MVP consistently demonstrated the best overall performance in Rare Disease Identification and Classification tasks, outperforming individual models and Self-Consistency Prompting (SC).

**Best Metrics (F-score) in a 64-word Context:**
- **Rare Disease Identification (F-score):** MVP achieved **0.69**, outperforming the best individual model (Llama 2: 0.58) and SC (Llama 2 + SC: 0.49).
- **Rare Disease Classification (F-score):** MVP achieved **0.69**, outperforming the best individual model (Vicuna: 0.67) and tying with SC (Llama 2 + SC: 0.70).

**LLM Models Used in the Ensemble:** Llama 2, MedAlpaca, Stable Platypus 2, and Vicuna.
**Citation:** Oniani, D. et al. (2024). *Large Language Models Vote: Prompting for Rare Disease Identification*. arXiv:2308.12890v3.

## Features

- **Ensemble Prompting:** Uses a set of models (e.g., Llama 2, MedAlpaca, Vicuna) to increase robustness and reduce the bias of a single model.
- **Majority Vote:** The final decision is based on the consensus of the models, outperforming the performance of any individual model.
- **Few-Shot Learning (FSL):** Optimized for tasks with limited data, such as the diagnosis of rare diseases.
- **CoT-Augmented:** Can be combined with techniques like Chain-of-Thought (CoT) to improve reasoning and explainability.
- **JSON-Augmented:** Uses JSON to ensure a parseable output format and facilitate automated evaluation.

## Use Cases

- **Differential Diagnosis:** Assist physicians in the triage and differential diagnosis of patients with atypical symptoms suggesting rare diseases.
- **EHR Analysis:** Extraction and classification of rare disease mentions from large volumes of unstructured clinical notes.
- **FSL Research:** Application in any medical or biological domain where data annotation is expensive and the availability of examples is limited.

## Integration

MVP uses a prompt template that combines the task description, a Chain-of-Thought (CoT) example for *in-context* instruction, and the actual question based on the clinical note. The output is structured in JSON to facilitate voting and analysis.

**Example of Suggested JSON Output Structure:**
```json
{
  "disease_identified": [
    "Babesiosis",
    "Giant Cell Arteritis",
    "Graft Versus Host Disease",
    "Cryptogenic Organizing Pneumonia"
  ],
  "task_disease": "None"
}
```
**Integration Guide:**
1.  Define a set of LLMs (e.g., 4 models).
2.  Create a CoT-Augmented prompt with the desired JSON output format.
3.  Submit the clinical note to the prompt across all LLMs.
4.  Collect the JSON outputs and perform a majority vote on the `"task_disease"` field or on the elements of the `"disease_identified"` list.
5.  The disease with the most votes is the final diagnosis of the ensemble.

## URL

https://arxiv.org/abs/2308.12890
