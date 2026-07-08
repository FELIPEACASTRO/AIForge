# Radiology Interpretation Prompts

## Description
**Radiology Interpretation Prompts** are prompt engineering instructions specifically designed to guide Large Language Models (LLMs), such as GPT-4 or Gemini, in the analysis and generation of content related to medical imaging exams (radiographs, CT scans, MRIs, etc.). The main objective is to transform raw image data or descriptive findings into **structured reports, patient summaries, diagnostic assistance, or teaching tools** [1] [2].

The technique is based on providing the LLM with a **role** (e.g., radiologist), the **clinical context**, and the **image findings**, requesting an output in a **structured and professional format** [2]. Recent studies (2024-2025) demonstrate that well-designed prompts, which include step-by-step reasoning and confidence limits, can **significantly improve the diagnostic accuracy** of LLMs in complex cases, such as in neuroradiology [3].

Effective radiology prompts are characterized by four key elements: **Instructions/Questions**, **Context**, **Data Input**, and **Output Format** [2]. The correct application of this technique is crucial to integrating LLMs safely and efficiently into the clinical workflow.

## Examples
```
**1. Structured Report Generation (Chest X-Ray)**

```
**Role:** You are an experienced thoracic radiologist.
**Instruction:** Analyze the descriptive findings of the Chest X-Ray and generate a structured and concise report.
**Findings:** "Alveolar opacity in the right lower lobe, associated with air bronchogram. Cardiac silhouette and mediastinum without alterations. Costophrenic angles clear."
**Output Format:**
**Findings:** [Detailed findings]
**Impression:** [Most likely diagnosis and differential diagnosis]
**Recommendation:** [Suggestion for follow-up or complementary exam]
```

**2. Report Summary for Patient (Layperson Language)**

```
**Role:** You are a medical communication assistant.
**Instruction:** Summarize the radiology report below in simple, accessible language for a patient, focusing only on the most important findings and their implications. Maintain a reassuring and informative tone.
**Report:** "Brain MRI: Intra-axial expansive lesion, with ring enhancement and central necrosis, located in the left temporal lobe. Suggestive of glioblastoma multiforme (GBM). Correlation with biopsy and oncological follow-up required."
**Output Format:** Title (What the exam showed), Summary (Simple explanation), Next Steps (What to do next).
```

**3. Differential Diagnosis Assistance (Abdominal CT)**

```
**Role:** You are a diagnostic imaging consultant.
**Instruction:** Based on the Abdominal CT findings, provide a list of the 3 most likely differential diagnoses, along with the confidence level (High, Medium, Low) for each and the clinical reasoning that supports them.
**Findings:** "Solid, heterogeneous mass, with coarse calcifications and delayed enhancement in the upper pole of the right kidney. No retroperitoneal lymphadenopathy."
**Output Format:** Table with columns: Diagnosis, Confidence, Reasoning.
```

**4. Generation of Discussion Text for a Scientific Article**

```
**Role:** You are a radiology researcher.
**Instruction:** Write the Discussion section of a scientific article about the use of AI in the early detection of pulmonary nodules. Include references to recent studies (2023-2025) and discuss the limitations and future of the technology.
**Key Topics:** AI performance (AUC > 0.95), Reduction of false positives, Generalization challenge, PACS integration.
**Output Format:** Academic text, well-structured paragraphs, with citations in [N] format.
```

**5. Exam Protocol Optimization (Magnetic Resonance Imaging)**

```
**Role:** You are a senior MRI technologist.
**Instruction:** Suggest optimizations for the Knee MRI protocol for better visualization of the articular cartilage, considering a 3T scanner.
**Current Protocol:** T1, T2, PD-FS.
**Desired Optimization:** Improve spatial resolution and cartilage contrast.
**Output Format:** List of suggested sequences (e.g., 3D DESS, T2 Mapping) and technical justification for each.
```
```

## Best Practices
**1. Clarity and Precision:** The prompt should be **clear, concise, and unambiguous**, defining the task, the target audience (e.g., radiologist, patient), and the desired output format (e.g., structured report, layperson summary) [1] [2].
**2. Role Adoption:** Begin the prompt by instructing the LLM to assume a specific role, such as "You are a radiologist specialized in neuroradiology" or "You are a medical transcription assistant" [3].
**3. Relevant Context:** Provide as much clinical and imaging context as possible. This includes the type of exam (CT, MRI, X-ray), the anatomical area, the patient's clinical history, and the raw image findings [1].
**4. Step-by-Step Reasoning:** For complex tasks, such as differential diagnosis, instruct the LLM to use the "Chain-of-Thought" (CoT) method or "Structured Clinical Reasoning" [3]. Ask it to list the findings, correlate them with pathologies, and finally provide the diagnostic impression.
**5. Output Constraint:** Specify the output format (e.g., JSON, bulleted list, running text) and the structure (e.g., "Findings", "Impression", "Recommendation"). This ensures consistency and facilitates integration with reporting systems [2].
**6. Confidence Threshold:** In diagnostic assistance prompts, ask the LLM to provide a confidence level for each suggested diagnosis. This helps filter out low-probability suggestions and increases safety [3].
**7. Iteration and Refinement:** Prompt design should be an iterative process. Test and adjust the prompt based on the accuracy and relevance of the LLM's responses [1].

## Use Cases
nan

## Pitfalls
**1. Hallucination and Invention of Findings:** The LLM may "hallucinate" (invent) clinical findings or diagnoses that are not present in the input data. **Mitigation:** Always instruct the model to base its interpretation **only** on the provided data and to indicate when the information is insufficient [1].
**2. Lack of Clinical Context:** Prompts that are too short or lack adequate clinical context (age, sex, symptoms, history) lead to generic or incorrect diagnoses. **Mitigation:** Always include a "Clinical History" block in the prompt [2].
**3. Bias and Generalization:** LLMs trained on data from a specific population may exhibit bias when interpreting exams from other populations. **Mitigation:** There is no direct solution via the prompt, but the radiologist should be aware of this and use the LLM only as assistance, not as the final decision.
**4. Format Inconsistency:** Failing to specify a rigorous output format (e.g., JSON or table) can result in inconsistent responses that are difficult to integrate into hospital information systems (HIS/RIS). **Mitigation:** Use the "Output Format" element explicitly and with examples [2].
**5. Overconfidence:** Relying on the LLM's output, especially in complex cases, can lead to diagnostic errors. **Mitigation:** Use the LLM as a "second opinion" or draft assistant, not as the final author of the report. Human review is mandatory [3].

## URL
[https://www.jacr.org/article/S1546-1440(25)00156-5/fulltext](https://www.jacr.org/article/S1546-1440(25)00156-5/fulltext)