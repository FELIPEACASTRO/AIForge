# AutoMedPrompt and Structured Clinical Reasoning Prompt

## Description

AutoMedPrompt is a new framework that uses 'textual gradients' (TextGrad) to optimize system prompts in general-purpose Large Language Models (LLMs) in order to elicit clinically relevant reasoning. Instead of relying on fine-tuning or prompting methods such as Chain-of-Thought (CoT) that may be inadequate for subspecialties, AutoMedPrompt adjusts the system prompt to improve the LLM's medical reasoning capability, outperforming state-of-the-art proprietary models.

The Structured Clinical Reasoning Prompt is a two-step prompt engineering technique that aims to improve the diagnostic capability of LLMs by forcing the model to follow a standardized clinical reasoning methodology. The first step involves the systematic summarization and categorization of clinical information (history, symptoms, exams) into a structured format. The second step uses this structured information to perform diagnostic reasoning, mimicking a physician's process.

## Statistics

**AutoMedPrompt:** Achieved a new State of the Art (SOTA) on the PubMedQA benchmark with an accuracy of 82.6%. It surpassed the performance of proprietary models such as GPT-4, Claude 3 Opus, and Med-PaLM 2. It also achieved 77.7% accuracy on MedQA and 63.8% on NephSAP (nephrology subspecialty) using the open-source Llama 3.

**Structured Clinical Reasoning Prompt:** Significantly increased primary diagnosis accuracy to 60.6% (vs. 56.5% baseline) and top-three diagnosis accuracy to 70.5% (vs. 66.5% baseline) across 322 diagnostic quiz cases (_Radiology's Diagnosis Please_). The study used the Claude 3.5 Sonnet model.

## Features

**AutoMedPrompt:** System prompt optimization; Uses textual gradients (TextGrad); Does not require extensive fine-tuning; Improves medical reasoning in general-purpose LLMs; Focused on medical QA benchmarks.

**Structured Clinical Reasoning Prompt:** Two-step approach (Summarization + Reasoning); Mimics the clinical workflow; Reduces performance variability; Improves diagnostic capability in complex cases.

## Use Cases

**AutoMedPrompt:** Improving diagnostic accuracy in clinical decision support systems; Optimizing open-source LLMs for specific medical tasks; Research and development of advanced prompts for medical education and knowledge testing.

**Structured Clinical Reasoning Prompt:** Training LLMs to simulate clinical reasoning; Triage and differential diagnosis systems in clinical settings; Decision support tool for resident physicians and students.

## Integration

**AutoMedPrompt:** The technique involves optimizing the LLM's system prompt (initial instruction). Although the final optimized prompt is not explicitly detailed in the abstract, the principle is: 'Optimize the system prompt to elicit the most relevant medical reasoning, using TextGrad to guide the optimization'.

**Structured Clinical Reasoning Prompt:**
*   **Step 1 (Summarization):** 'You are an experienced Diagnostic Radiologist. Your task is to summarize the following clinical case, categorizing the information into: patient information, history of present illness, medical history, symptoms, physical examination findings, laboratory results, imaging findings, etc.'
*   **Step 2 (Reasoning):** 'As a physician, use the structured summary to guide me through the differential diagnosis process to the most likely diagnosis and the next two most likely differential diagnoses, step by step.'

## URL

https://arxiv.org/abs/2502.15944; https://pmc.ncbi.nlm.nih.gov/articles/PMC11953165/
