# Neurology Prompts

## Description
Prompt Engineering in Neurology refers to the art and science of creating optimized instructions (prompts) for Large Language Models (LLMs) with the goal of assisting in clinical, research, and administrative tasks within the field of Neurology. It is not a single prompting technique, but rather the application of advanced techniques such as **RAG (Retrieval-Augmented Generation)**, **Role Definition**, and **Output Schema** to ensure that LLMs provide clinically relevant, accurate, and safe responses. Recent studies (2025) demonstrate its effectiveness in generating consultation summaries and predicting risk in emergency settings, although they warn of the need for rigorous validation and human oversight due to limitations in subtle clinical reasoning [1] [2] [3].

## Examples
```
1. **Structured Clinical Summary Generation (RAG-like)**

```
**Role:** You are a neurological documentation assistant. Your task is to generate a concise, high-fidelity summary for the electronic health record.

**Context (RAG):** [INSERT HERE: Patient history, imaging and laboratory results, nursing note, and initial neurological exam.]

**Instruction:** Analyze the context and generate a summary with the following sections and constraints:
1. **Chief Complaint (CC):** Maximum 1 sentence.
2. **Key Exam Findings:** 3 to 5 main points.
3. **Differential Diagnosis (DD):** List of the 3 most likely DDs.
4. **Suggested Management Plan:** List of 3 immediate actions.

**Output Format:** JSON, with the keys 'CC', 'Findings', 'DD', and 'Management'.
```

2. **Differential Diagnosis Support for Headache**

```
**Role:** You are a consultant neurologist specialized in headaches. The patient presents with [INSERT: Age, Sex, Duration, Location, Intensity (scale 1-10), Associated Symptoms (nausea, photophobia, aura)].

**Instruction:** Based on the data, provide:
1. The most likely primary diagnosis.
2. Two differential diagnoses that cannot be ruled out.
3. A list of 3 'Red Flags' (warning signs) that would require immediate investigation.

**Constraint:** The response must be didactic and justify each diagnosis based on the symptoms provided.
```

3. **Emergency Admission Risk Prediction**

```
**Role:** You are a neurological risk triage system (Neuro-Copilot AI).

**Input Data:** [INSERT: NIHSS Score, Age, Blood Pressure, Blood Glucose, Symptom Duration, Presence of Comorbidities (e.g., Atrial Fibrillation)].

**Instruction:** Calculate the probability of:
1. Need for hospital admission (Low, Medium, High).
2. Risk of mortality within 48 hours (%).

**Constraint:** If the probability of admission is 'High', add the sentence: 'REQUIRES IMMEDIATE EVALUATION BY A NEUROLOGIST'. If the data is incomplete, respond: 'INSUFFICIENT DATA FOR PREDICTION'.
```

4. **Simplified Neuroimaging Interpretation (For Patient)**

```
**Role:** You are a medical communicator. Your task is to translate the MRI report into language that a patient with a high school education can understand.

**Original Report:** [INSERT: Excerpt from the report, e.g., 'Multiple hyperintense lesions on T2 and FLAIR, periventricular and juxtacortical, consistent with demyelinating disease.']

**Instruction:** Explain the finding in 3 short paragraphs. Use analogies if necessary. Avoid technical jargon. Maintain a reassuring and informative tone.
```

5. **Structured Data Extraction from Progress Note**

```
**Role:** You are a data extractor for a quality system.

**Progress Note:** [INSERT: Daily progress note, e.g., 'Patient with Parkinson's, no change in levodopa dose. Presents mild bradykinesia, but resting tremor is controlled. No falls. Next appointment in 3 months.']

**Instruction:** Extract the following fields and format them in JSON:
- **Disease:**
- **Key Medication:**
- **Dose Changed (Yes/No):**
- **Dominant Symptom:**
- **Next Follow-up (Months):**
```

6. **Investigation Protocol Suggestion for Neuropathy**

```
**Role:** You are a specialist in the investigation of peripheral neuropathies.

**Data:** [INSERT: Clinical history (DM, alcoholism, chemotherapy), Physical Exam Findings (Pattern of sensory/motor deficit), EMG Result (e.g., Chronic sensorimotor axonal neuropathy).]\n\n**Instruction:** Suggest a second-line laboratory investigation protocol (after basic exams) in numbered list format, prioritizing the most likely causes.
```

7. **Creation of a Clinical Simulation Scenario for Training**

```
**Role:** You are a medical curriculum designer.

**Topic:** Generalized Tonic-Clonic Seizure.

**Instruction:** Create a clinical simulation scenario for neurology residents, including:
1. **Patient Presentation:** (Age, Sex, Brief History).
2. **Initial Resident Actions (Checklist):** (5 items).
3. **Critical Decision Point:** (E.g., When to administer the second dose of benzodiazepine).
4. **Simulation Outcome:** (Brief description).
```

8. **Literature Review Focused on Mechanism of Action**

```
**Role:** You are a senior researcher.

**Drug:** [INSERT: Drug name, e.g., Fingolimod]

**Instruction:** Review the literature from the last 5 years and describe the drug's mechanism of action in the context of Multiple Sclerosis. Focus on:
1. Primary molecular targets.
2. Immunological and non-immunological effects.
3. Impact on the blood-brain barrier.

**Constraint:** Use references (cite the author and year) and limit the response to 300 words.
```
```

## Best Practices
The effectiveness of Neurology Prompts depends on their structuring and the integration of contextual data. Best practices include:

*   **Role and Persona Definition:** Always instruct the LLM to act as a specific healthcare professional (e.g., 'You are a consultant neurologist'), raising the quality and tone of the response.
*   **RAG (Contextualization):** For clinical tasks, the LLM should be augmented with patient data (history, exams, notes). This mitigates hallucination and ensures clinical relevance [1].
*   **Rigid Output Structure:** Require structured formats (JSON, tables, numbered lists) to facilitate analysis, integration with Electronic Health Records (EMRs), and workflow automation.
*   **Safety Constraints:** Include instructions for the LLM to refuse or request more information if the input data is insufficient, ambiguous, or if the task exceeds the scope of an AI assistant (e.g., 'Do not provide a final diagnosis, only suggestions').
*   **Tone and Language:** Specify the tone (clinical, concise, didactic) and the target audience (fellow physician, patient, researcher) to optimize communication.

## Use Cases
The primary use cases for Prompt Engineering in Neurology focus on increasing clinical efficiency and accuracy:

*   **Documentation Optimization:** Automatic generation of consultation summaries, progress notes, and discharge letters, reducing administrative burden [1].
*   **Clinical Decision Support (CDS):** Assistance in formulating differential diagnoses, suggesting investigation and treatment plans, and interpreting complex data (e.g., EEG, EMG) [2].
*   **Triage and Risk Prediction:** Use of LLM-based models to predict critical outcomes, such as the need for admission or short-term mortality risk, especially in emergency settings [3].
*   **Education and Training:** Creation of clinical simulation scenarios, generation of multiple-choice questions, and translation of medical jargon for patient and student education.
*   **Research:** Analysis and summarization of large volumes of scientific literature and extraction of structured data from articles for meta-analyses.

## Pitfalls
The application of LLMs in neurology presents significant risks that must be mitigated by Prompt Engineering:

*   **Clinical Hallucination (Inaccuracy):** The risk of the LLM generating clinically false information or inventing references is high, which can lead to diagnostic or treatment errors. This is aggravated by the lack of subtle reasoning in complex cases [2].
*   **Over-ordering of Tests:** LLMs without specific training tend to suggest more diagnostic tests than necessary, increasing costs and healthcare system burden [2].
*   **Bias and Inequity:** If the training data is biased (e.g., underrepresentation of certain populations), the prompts may perpetuate disparities in care, leading to inadequate recommendations for minority groups.
*   **Over-reliance:** Blind trust in the LLM's outputs, without validation by a human professional, is the greatest risk. The LLM should be seen as an assistant, not as a substitute for clinical judgment.
*   **Sensitive Data Leakage:** The inclusion of patient data in the prompt (even if anonymized) requires rigorous security protocols to prevent the exposure of Protected Health Information (PHI).

## URL
[https://www.nature.com/articles/s41598-025-22769-7](https://www.nature.com/articles/s41598-025-22769-7)
