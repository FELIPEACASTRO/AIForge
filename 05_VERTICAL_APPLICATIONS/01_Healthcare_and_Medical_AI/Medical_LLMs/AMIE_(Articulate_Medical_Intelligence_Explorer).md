# AMIE (Articulate Medical Intelligence Explorer)

## Description

AMIE (Articulate Medical Intelligence Explorer) is a Large Language Model (LLM) optimized specifically for clinical diagnostic reasoning. It was developed to generate an accurate Differential Diagnosis (DDx), both autonomously and as an aid to clinicians. The model was fine-tuned from PaLM 2 (large version) using a mixture of medical tasks, including question answering, clinical dialogue generation, and summarization of Electronic Health Record (EHR) notes. Its optimization focuses on long contexts to enhance long-range reasoning capabilities and contextual understanding, which are essential for the complexity of medical cases. The primary goal is to improve diagnostic accuracy in challenging cases and to broaden access to specialized expertise.

## Statistics

**Top-10 Accuracy (Standalone):** 59.1% (vs. 33.6% for unassisted clinicians; P=0.04). **Top-1 Accuracy:** 29%. **Clinical Assistance:** The DDx quality score was higher for clinicians assisted by AMIE (top-10 accuracy of 51.7%) compared to clinicians without assistance (36.1%) and with traditional search (44.4%; P=0.03). **Base:** PaLM 2 (large version). **Citation:** McDuff et al. (2025). Towards accurate differential diagnosis with large language models. Nature. [1]

## Features

Optimization for Clinical Reasoning: Fine-tuned with specific medical tasks (questions, dialogues, EHR summarization) to improve diagnostic reasoning. DDx Generation: Ability to generate comprehensive and accurate Differential Diagnosis (DDx) lists. Superior Performance: Outperforms unassisted clinicians and traditional search tools in DDx accuracy. Clinical Assistance: Integrated into interactive interfaces to assist physicians in formulating the DDx. Long-Context Use: Trained to handle long clinical histories and contextual information.

## Use Cases

**Clinical Decision Support:** Assist physicians in formulating Differential Diagnoses (DDx) for complex and challenging cases. **Medical Education:** Training students and residents by providing structured diagnostic reasoning. **Triage and Remote Consultation:** Generation of initial diagnostic hypotheses in telemedicine settings or emergency triage. **Research:** Evaluation and comparison of different clinical reasoning approaches.

## Integration

The most effective prompting technique for AMIE and similar LLMs is the **Structured Clinical Reasoning Prompt**, which mimics a physician's thought process.

**Example Prompt (Symptom-to-Diagnosis):**

```
You are a medical diagnosis specialist. Your task is to analyze the patient's history and provide a detailed Differential Diagnosis (DDx), followed by the most likely diagnosis.

**Reasoning Steps:**
1. **Data Collection:** List the symptoms, relevant medical history, and examination findings (if any).
2. **Pattern Analysis:** Identify patterns and correlate the findings with possible disease categories.
3. **DDx:** Generate a list of 3 to 5 plausible differential diagnoses, briefly justifying each one.
4. **Most Likely Diagnosis:** Indicate the most likely diagnosis and the rationale for your choice.

**Patient History:**
Patient: Male, 50 years old.
Symptoms: Fatigue, unexplained weight loss (5 kg in 3 months), frequent urination (polyuria), and excessive thirst (polydipsia).
Medical History: No relevant history, except for controlled hypertension.
```

**Integration Guide:** The AMIE model was integrated into an interactive interface to measure its impact as a clinical assistant, suggesting that the ideal integration is via API in Clinical Decision Support Systems (CDSS) or EHRs, using multi-step prompts (such as CoT) to ensure transparency of reasoning.

## URL

https://www.nature.com/articles/s41586-025-08869-4