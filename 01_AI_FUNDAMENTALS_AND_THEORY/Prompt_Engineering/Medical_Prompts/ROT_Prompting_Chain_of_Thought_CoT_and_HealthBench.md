# ROT Prompting, Chain-of-Thought (CoT), and HealthBench

## Description

Comprehensive research on prompt engineering techniques and the evaluation of Large Language Models (LLMs) in the context of **Treatment Recommendation** in healthcare, with a focus on developments from 2023 to 2025. Two main prompting approaches were identified and detailed: **ROT Prompting** (Role-Playing, Output-Constrained, and Thought-Process) and **Chain-of-Thought (CoT)**, in addition to the **HealthBench** benchmark for evaluating medical LLMs.

**ROT Prompting** proved effective at increasing LLM adherence (such as GPT-4) to evidence-based clinical guidelines, achieving up to 77.5% consistency for strong recommendations in a study on osteoarthritis.

**Chain-of-Thought (CoT)** is crucial for the transparency and traceability of clinical reasoning, with models like o1-mini reaching high accuracy (88.4%) in medical discharge summarization tasks.

**HealthBench** serves as a robust benchmark, with 5,000 multi-turn conversations evaluated by physicians, measuring the performance and safety of LLMs across seven health themes, including accuracy and communication quality. Recent frontier models (such as o3 and GPT-4.1) demonstrated significant improvements.

## Statistics

**ROT Prompting:** In the study by Wang et al. (2024) on the AAOS osteoarthritis guidelines, the combination of gpt-4-Web with ROT prompting achieved the highest overall consistency (62.9%) and a consistency of 77.5% for strong recommendations.

**CoT Prompting:** The study by Jeon et al. (2025) showed that the o1-mini model achieved 88.4% accuracy in medical discharge summarization tasks (EHRNoteQA) and 83.5% in clinical notes using CoT.

**HealthBench:** Performance ranges from 0.16 (GPT-3.5 Turbo) to 0.60 (o3). Performance improved 28% across OpenAI's frontier models. The 'Accuracy' axis represents 33% of all rubric criteria.

## Features

**ROT Prompting:** Improves adherence to evidence-based clinical guidelines; Increases the consistency and reliability of responses in complex medical tasks; Combines elements of Role-Playing and Chain-of-Thought (CoT); Most effective for high-evidence (strong level) treatment recommendations.

**Chain-of-Thought (CoT):** Increases the transparency and interpretability of the LLM's reasoning; Improves accuracy in Question Answering (QA) tasks and the summarization of clinical notes; Allows healthcare professionals to intervene at any point in the reasoning chain (Interactive CoT).

**HealthBench:** Evaluation based on rubric and medical consensus; 5,000 realistic multi-turn conversations; Seven themes (including health data tasks and emergency referrals) and five axes (including accuracy and completeness); Two variations: HealthBench Consensus (higher accuracy) and HealthBench Hard (more difficult).

## Use Cases

**General:** Clinical Decision Support Systems (CDSS); Evaluation of LLM compliance with established medical protocols; Summarization and analysis of electronic health records (EHR); Training and evaluation of medical students.

**Specific:** Generation of treatment recommendations for medical conditions (e.g., osteoarthritis); Measurement of the safety and reliability of LLMs in healthcare scenarios; Identification of weaknesses in models for improvement (e.g., accuracy in health data tasks).

## Integration

**ROT Prompting (Example Structure):** 'You are a physician specializing in [Specialty]. Analyze the following clinical case: [Clinical Case]. Think step by step about the clinical evidence and the established guidelines. What is the treatment recommendation most consistent with evidence-based guidelines? Present the recommendation and the level of evidence.'

**CoT Prompting (Example Template):** 'You are a [Medical Specialty]. Clinical Case: [Patient Details, Symptoms, Test Results]. Think step by step (Chain-of-Thought): 1. What is the most likely diagnosis? 2. What are the evidence-based treatment options for this diagnosis? 3. Which treatment do you recommend and why? Final Answer: [Treatment Recommendation].'

**HealthBench (Usage):** Practical use involves submitting an LLM's responses to HealthBench conversations for scoring, comparing performance across themes such as emergency referrals and communication tailored to the user's experience. Models like o3 and GPT-4.1 demonstrated superior performance over other models.

## URL

https://www.nature.com/articles/s41746-024-01029-4
