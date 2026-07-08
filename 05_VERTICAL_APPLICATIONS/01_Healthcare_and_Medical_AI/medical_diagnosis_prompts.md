# Medical Diagnosis Prompts

## Description
Prompt Engineering for Medical Diagnosis is the discipline of creating structured, calibrated instructions for Large Language Models (LLMs) with the goal of assisting in clinical reasoning, the generation of differential diagnoses, and the analysis of patient data. Recent studies (2024-2025) demonstrate that the diagnostic accuracy of LLMs, such as Claude 3.5 Sonnet and GPT-4, can be significantly improved through prompting techniques that mimic the human clinical reasoning process, such as the two-step approach (Two-Step Clinical Reasoning) [1]. This approach involves first the summarization and structured organization of the patient's data by the LLM, followed by the diagnosis request based on this summary. Clarity, precision, and the inclusion of detailed context are crucial to mitigate biases and the occurrence of "hallucinations" (fabricated information) [3]. The use of diagnostic prompts is a promising tool for optimizing the clinical routine, but it requires caution and rigorous human validation.

## Examples
```
1. **Structured Clinical Reasoning Prompt (Two-Step):**
   *   **Step 1 (Summarization):** "You are an experienced radiologist. Your task is to summarize the following clinical case, organizing the information into the following categories with concise bullet points: Patient Information (age, sex), History of Present Illness, Past Medical History, Key Symptoms, Imaging Findings. If a category has no information, write 'No information'. [INSERT RAW PATIENT DATA]"
   *   **Step 2 (Diagnosis):** "Based on the structured summary you generated, act as a physician. Present the step-by-step reasoning, the most likely diagnosis, and the two most likely differential diagnoses. Include the justification for each one."

2. **Differential Diagnosis Prompt with Format Constraint:**
   "Act as a general practitioner. A 45-year-old male patient presents with persistent fever (38.5°C), dry cough, and progressive dyspnea for 10 days. History of recent travel to Southeast Asia. Chest X-ray shows bilateral interstitial infiltrate.
   List the 5 most likely differential diagnoses in a table. For each diagnosis, include: 1) Probability (High, Medium, Low), 2) Clinical Justification, and 3) Next-Step Complementary Exam."

3. **Prompt for Analysis of Laboratory Results:**
   "You are a hematologist. Analyze the following complete blood count results and suggest the most likely condition and the initial management.
   *   Hemoglobin: 9.5 g/dL (Low)
   *   MCV: 75 fL (Low)
   *   Leukocytes: 12,000/mm³ (High)
   *   Platelets: 450,000/mm³ (High)
   *   Ferritin: 10 ng/mL (Very Low)
   Most likely condition: [Answer]
   Reasoning: [Answer]
   Initial management: [Answer]"

4. **Prompt for Virtual Patient Simulation (Role-Play):**
   "Act as a 68-year-old patient, named João, with a history of smoking. You are in a consultation with the physician. Your symptoms are chest pain when coughing and unintentional weight loss over the past 3 months. Answer the physician's questions concisely and with a worried tone of voice. Do not reveal the lung cancer diagnosis until the physician requests an imaging exam."

5. **Prompt for Review of Clinical Guidelines:**
   "Based on the most recent guidelines of the Sociedade Brasileira de Cardiologia (SBC) (2023-2025), summarize the first-line treatment protocol for Heart Failure with Reduced Ejection Fraction (HFrEF). Include the recommended drug classes and the ideal target dose for each one."
```

## Best Practices
**Two-Step Clinical Reasoning Structure:** Divide the process into two steps: 1) Ask the LLM to **summarize and organize** the raw clinical data (history, symptoms, exams) into structured categories. 2) Use the **structured summary** as input for the diagnosis step, requesting the differential diagnosis and reasoning. This has been proven to increase accuracy [1]. **Role-Play Prompting:** Start the prompt by defining the LLM's role (e.g.: "You are an experienced radiologist", "Act as a general practitioner"). This aligns the model's response with the necessary clinical context [1]. **Inclusion of Context and Constraints:** Provide as much context as possible (age, sex, history, exam results). Constrain the output format (e.g.: "List the 3 most likely diagnoses in table format, including the probability and the next investigation step"). **Transparency and Reasoning (Chain-of-Thought - CoT):** Ask the LLM to detail its reasoning process (CoT) before presenting the final diagnosis. This increases interpretability and allows human validation of the process [2]. **Mandatory Human Validation:** The LLM's output should always be treated as **decision support** and never as a final diagnosis. Review and validation by a qualified healthcare professional is essential [3].

## Use Cases
**Clinical Decision Support:** Generation of differential diagnosis lists for complex or atypical cases, ensuring that no rare condition is overlooked. **Medical Education and Simulation:** Creation of detailed clinical scenarios, scripts for simulated patients (SPs), and Objective Structured Clinical Examination (OSCE) stations, optimizing the training of students and residents [3]. **Analysis of Unstructured Data:** Extraction and summarization of relevant clinical information from electronic health records in free-text format, radiology reports, or progress notes [1]. **Literature and Guideline Review:** Generation of rapid and comparative summaries of treatment protocols and updated clinical guidelines, assisting in evidence-based decision-making. **Exam Interpretation:** Assistance in interpreting laboratory results or imaging findings, suggesting clinical correlations and next investigation steps.

## Pitfalls
**Hallucinations (Fabrication of Facts):** The LLM may generate false medical information that appears plausible. **Mitigation:** Always verify the output with reliable medical sources and clinical guidelines. **Data Biases:** The model may reflect biases present in the training data (e.g., underrepresentation of certain ethnicities or rare conditions), leading to inaccurate or incomplete diagnoses for these groups. **Mitigation:** Explicitly include demographic information and request that the model consider the differential diagnosis in diverse populations. **Lack of Transparency (Black Box):** Without requesting step-by-step reasoning (CoT), the diagnosis is a "black box". **Mitigation:** Always use the CoT technique to understand the model's logic. **Use of Sensitive Data:** Never enter personally identifiable information (PII) or protected health information (PHI) into general-purpose LLMs, due to privacy and compliance concerns (e.g., LGPD, HIPAA) [3]. **Mitigation:** Use only anonymized data or specialized LLM models certified for healthcare (e.g., MedPaLM, Azure Health).

## URL
[https://www.medrxiv.org/content/10.1101/2024.09.01.24312894v1.full-text](https://www.medrxiv.org/content/10.1101/2024.09.01.24312894v1.full-text)
