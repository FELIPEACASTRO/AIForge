# Oncology Prompts

## Description
Prompt engineering in oncology refers to the art and science of formulating effective inputs (prompts) for **Large Language Models (LLMs)**, with the goal of optimizing their performance on clinical, research, and administrative tasks related to cancer. This field is crucial for the safe and effective adoption of Artificial Intelligence (AI) in oncology practice, as highlighted by the **ESMO Guidance on the Use of Large Language Models in Clinical Practice (ELCAP)** [1].

LLMs are used to enhance, not replace, clinical workflows and decision-making, acting as assistive tools. The effectiveness of "Oncology Prompts" depends on the clarity of the instruction, the context provided (patient data, clinical guidelines), and the prompt engineering technique used (e.g., Few-shot, RAG, CoT) [2].

## Examples
```
### Example 1: Adjuvant Treatment Decision Support (CoT + RAG)

**Objective:** Suggest the adjuvant treatment for Breast Cancer, following the provided ESMO guideline.

**Prompt:**
\`\`\`
You are an AI assistant specialized in oncology, focused on clinical decision support. Your task is to analyze the patient's profile and the provided clinical guideline to suggest the most appropriate adjuvant treatment, using step-by-step reasoning (Chain-of-Thought).

**CLINICAL GUIDELINE (RAG Context):**
[INSERT EXCERPT FROM THE ESMO GUIDELINE ON EARLY BREAST CANCER HERE, FOR EXAMPLE: "For patients with early breast cancer, HR+/HER2-, with 1-3 positive lymph nodes and an Oncotype DX score of 26 or higher, chemotherapy followed by endocrine therapy is recommended."]

**PATIENT DATA:**
- Age: 55 years
- Diagnosis: Invasive ductal carcinoma, stage IIA
- Receptor Status: HR+/HER2-
- Lymph Nodes: 2/15 positive
- Risk Score: Oncotype DX = 31

**INSTRUCTIONS:**
1.  **Analysis (CoT):** Compare the PATIENT DATA with the CLINICAL GUIDELINE. Identify the criteria that align and those that do not align.
2.  **Conclusion:** Based on the analysis, what is the adjuvant treatment recommendation?
3.  **Justification:** Cite the specific part of the CLINICAL GUIDELINE that supports your recommendation.
\`\`\`

### Example 2: Structured Extraction from a Pathology Report (Few-shot)

**Objective:** Extract key information from an unstructured pathology report to populate an Electronic Medical Record (EMR).

**Prompt:**
\`\`\`
You are a clinical data extractor. Your output MUST be a JSON object.

**FEW-SHOT EXAMPLE (Input):**
"The prostate biopsy specimen revealed acinar adenocarcinoma. Gleason 4+3=7. Negative margins. Perineural invasion present. Staging pT2c."

**FEW-SHOT EXAMPLE (Output):**
\`\`\`json
{
  "Primary Diagnosis": "Acinar adenocarcinoma of the prostate",
  "Gleason Score": "7 (4+3)",
  "Margin Status": "Negative",
  "Perineural Invasion": "Present",
  "Pathological Staging": "pT2c"
}
\`\`\`

**PATHOLOGY REPORT (Input):**
"The histopathological examination of the surgical lung specimen confirms the presence of Adenocarcinoma. The tumor measures 3.5 cm. Clear surgical margins (distance of 1.2 cm). No vascular or lymphatic invasion was observed. Pathological staging T2a N0 M0."

**INSTRUCTIONS:**
Extract the information from the PATHOLOGY REPORT and return the result in JSON format, following the structure of the FEW-SHOT EXAMPLE.
\`\`\`

### Example 3: Patient Communication (Zero-shot + Persona)

**Objective:** Explain what "immunotherapy" is to a newly diagnosed patient, using simple and empathetic language.

**Prompt:**
\`\`\`
**PERSONA:** You are an oncology nurse with 10 years of experience, known for your ability to explain complex medical concepts in a clear and reassuring way.

**AUDIENCE:** 68-year-old patient, newly diagnosed with melanoma, with a high school education level.

**INSTRUCTION:** Explain what **immunotherapy** for cancer is. Use simple analogies (zero-shot) and avoid medical jargon. The tone should be supportive and informative.
\`\`\`

### Example 4: Clinical Trial Screening (RAG)

**Objective:** Determine whether a patient is eligible for a specific clinical trial.

**Prompt:**
\`\`\`
You are a clinical trial eligibility specialist. Your task is to compare the patient's data with the inclusion and exclusion criteria of the provided Clinical Trial.

**CLINICAL TRIAL CRITERIA (RAG Context):**
- **Inclusion:** Patients with metastatic Renal Cell Carcinoma (RCC), who have received at most one prior line of therapy. Age ≥ 18 years. ECOG Performance Status of 0 or 1.
- **Exclusion:** History of symptomatic brain metastases. Active autoimmune disease.

**PATIENT DATA:**
- Diagnosis: Metastatic RCC
- Prior Therapies: Sunitinib (1 line)
- Age: 62 years
- ECOG Status: 1
- History: Asymptomatic brain metastases treated 3 years ago.

**INSTRUCTIONS:**
1.  List the Inclusion and Exclusion criteria.
2.  For each criterion, indicate whether the patient meets it ("YES" or "NO").
3.  **Conclusion:** Is the patient eligible for the trial? Justify the answer.
\`\`\`

### Example 5: Consultation Summary Generation (Structured Output)

**Objective:** Generate a structured summary of the consultation for the medical record.

**Prompt:**
\`\`\`
**FUNCTION:** Generate a structured consultation summary.

**AUDIO INPUT (Transcription):**
"The patient, Mr. João, 72 years old, came for follow-up. He is in cycle 4 of chemotherapy for colon cancer. He reports mild nausea, controlled with ondansetron. The physical exam is stable. We requested new CEA tests and an abdominal and pelvic CT for reassessment. Next appointment in 3 weeks."

**INSTRUCTIONS:**
Generate a summary in Markdown format, with the following sections:
1.  **Patient Data:** (Name, Age, Diagnosis)
2.  **Treatment Status:** (Current cycle, Regimen)
3.  **Symptoms/Toxicity:** (Description and management)
4.  **Plan:** (Requested tests, Next appointment)
\`\`\`

### Example 6: Risk of Bias Analysis in Research (Permissive Prompt)

**Objective:** Screen research articles for a systematic review study, applying a "soft" exclusion criterion.

**Prompt:**
\`\`\`
You are a literature reviewer for a systematic review on LLMs in oncology.

**PERMISSIVE EXCLUSION CRITERION:** Exclude the article ONLY if it does not explicitly mention the use of Large Language Models (LLMs) or language models in its abstract.

**ARTICLE ABSTRACT (Input):**
"We evaluated the effectiveness of a new machine learning algorithm for predicting response to radiotherapy in lung cancer patients. The model was trained on image and text data from electronic medical records. The results show high accuracy in risk stratification."

**INSTRUCTIONS:**
1.  **Analysis:** Does the abstract explicitly mention LLMs or language models?
2.  **Decision:** Should the article be included or excluded? Justify the decision based on the PERMISSIVE EXCLUSION CRITERION.
\`\`\`
```

## Best Practices
The best practices for "Oncology Prompts" are strongly guided by the principles of safety, transparency, and accountability, as established by ELCAP [1].

| Principle | Description | Related Prompt Engineering Technique |
| :--- | :--- | :--- |
| **Explicit Human Accountability** | For tools aimed at healthcare professionals (Type 2), final responsibility for clinical decisions must remain with the oncologist. The prompt should request the source and justification for the output. | **Chain-of-Thought (CoT):** Ask the LLM to present step-by-step reasoning before the final conclusion. |
| **Data Grounding (RAG)** | Integrate the retrieval of information from reliable sources (clinical guidelines, EHRs, validated literature) into the prompt. This mitigates hallucinations and ensures that the response is evidence-based. | **Retrieval-Augmented Generation (RAG):** Insert excerpts from relevant clinical documents into the prompt before the question. |
| **Validation and Transparency** | For decision support tasks, the prompt must be formally validated and its limitations made transparent. Use **Few-shot** prompts with examples of known and validated clinical cases. | **Few-shot Prompting:** Provide examples of input/output pairs for specific tasks (e.g., stage classification, treatment suggestion). |
| **Data Protection** | In patient-facing applications (Type 1) and institutional systems (Type 3), the prompt should be formulated to protect privacy, avoiding the inclusion of unnecessary Protected Health Information (PHI). | **Filtering/Anonymization Prompt:** Instruct the LLM to anonymize or summarize sensitive data before processing it. |
| **Continuous Monitoring** | In institutional systems, prompts should be designed to facilitate continuous monitoring of bias and performance, ensuring that the system is revalidated when processes or data sources change. | **Structured Prompt:** Use a rigid prompt format (e.g., JSON, XML) to ensure consistent inputs and easily auditable outputs. |

## Use Cases
The use of "Oncology Prompts" spans various areas of oncology [1] [2]:

1.  **Clinical Decision Support:** Generation of adjuvant or neoadjuvant treatment suggestions based on guidelines and patient data (e.g., early breast cancer).
2.  **Data Extraction and Summarization:** Extraction of cancer information from unstructured text (clinical notes, pathology reports) to populate Electronic Medical Records (EMRs) or for research.
3.  **Patient Communication and Education:** Creation of chatbots to answer patient questions about their condition, treatment, and symptom support, operating under clinical supervision.
4.  **Literature and Research Screening:** Use of prompts to screen biomedical articles, identify relevant studies, or evaluate the synergy of cell lines in drug research.
5.  **Clinical Trial Matching:** Automation of matching patient profiles with clinical trial eligibility criteria.
6.  **Question-Answering (QA) Systems:** Improving the usability of complex clinical guidelines by transforming them into actionable QA systems.

## Pitfalls
The adoption of LLMs in oncology presents significant risks that prompt engineering must mitigate [1]:

*   **Hallucinations and Inaccuracy:** The risk of the LLM generating clinically incorrect or fabricated information is high, especially if the prompt is vague or based on incomplete input data.
*   **Bias and Inequity:** The LLM may perpetuate or amplify biases present in the training data, leading to disparities in care or inadequate suggestions for patient subpopulations.
*   **Lack of Transparency:** If the prompt does not require reasoning (CoT) or the source (RAG), the LLM's output may be a "black box", hindering auditing and clinical accountability.
*   **Over-reliance:** Healthcare professionals may over-rely on the LLM's suggestions, neglecting explicit human accountability and clinical judgment.
*   **Privacy Violation:** Poorly formulated prompts may inadvertently expose Protected Health Information (PHI) or lead to data leaks in institutional systems.

## URL
[https://www.esmo.org/society-updates/esmo-publishes-first-guidance-on-the-safe-use-of-large-language-models-in-oncology-practice](https://www.esmo.org/society-updates/esmo-publishes-first-guidance-on-the-safe-use-of-large-language-models-in-oncology-practice)
