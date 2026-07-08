# TrialGPT (Clinical Trial Matching Framework)

## Description

TrialGPT is an end-to-end, LLM-agnostic framework (primarily using GPT-4 and GPT-3.5) for zero-shot matching of patients to clinical trials. It automates eligibility screening by dividing the task into three modules: 1) **TrialGPT-Retrieval** for large-scale filtering and keyword generation; 2) **TrialGPT-Matching** for criterion-level eligibility prediction with faithful explanations; and 3) **TrialGPT-Ranking** for scoring and final ranking of trials.

## Statistics

Evaluated on three cohorts (SIGIR 2016, TREC 2021/2022) with 183 synthetic patients and more than 75,000 annotations. **TrialGPT-Retrieval** achieves more than 90% recall using less than 6% of the initial collection. **TrialGPT-Matching** reaches 87.3% accuracy on 1,015 patient-criterion pairs, comparable to experts. The use of TrialGPT demonstrated a 42.6% reduction in patient screening time.

## Features

Zero-shot matching (without trial-specific fine-tuning). Explainability (generates faithful explanations for eligibility decisions). Scalability for large collections of clinical trials (up to 23,000 active trials). LLM-agnostic (can be adapted to different LLMs).

## Use Cases

Automated screening and recruitment of patients for clinical trials. Reduction of the manual workload of screening. Prioritization of the most relevant clinical trials for a specific patient.

## Integration

The framework uses Chain-of-Thought (CoT) prompts for the Matching and Ranking tasks. The prompts are structured to include: 1) Task description; 2) Clinical background information; 3) Inclusion and exclusion criteria. The goal is to generate a relevance explanation (R), a list of relevant sentence IDs (S), and the eligibility prediction (E) for each criterion. Example prompt structure (adapted from the article): 'You are an expert in clinical trial eligibility. Given the patient note and the eligibility criterion [CRITERION], determine whether the patient is 'included', 'not included', 'insufficient information', or 'not applicable'. Think step by step and provide the explanation, the evidence sentences, and the final decision in JSON format.'

## URL

https://www.nature.com/articles/s41467-024-53081-z
