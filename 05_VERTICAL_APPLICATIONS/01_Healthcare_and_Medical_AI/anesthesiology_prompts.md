# Anesthesiology Prompts

## Description
Prompt Engineering in Anesthesiology refers to the creation and optimization of instructions (prompts) for Large Language Models (LLMs) with the goal of assisting in clinical, administrative, and educational tasks in the perioperative context. This includes risk stratification (such as the ASA-PS classification), the generation of preliminary anesthetic plans, the simplification of educational materials for patients, and support for research and analysis of medical literature. The technique aims to mitigate the risks of hallucinations and biases, ensuring that responses are clinically relevant, accurate, and safe. The use of complex prompts, such as Chain-of-Thought (CoT) and the inclusion of unstructured clinical data (pre-operative notes) via RAG (Retrieval-Augmented Generation), has been shown to significantly improve the model's accuracy in risk classification and outcome prediction tasks.

## Examples
```
**1. ASA-PS Classification with RAG (Few-Shot):**
`You are a Senior Anesthesiologist. Your task is to classify the patient according to the ASA-PS Scale (I to VI) and justify your decision based on the information provided.
**Patient Information:** [Insert unstructured clinical notes and medical history].
**Reference Information (RAG):** [Insert relevant excerpts from the ASA-PS guidelines].
**Few-Shot Example:** [Patient 1: History, Classification, Justification].
**Output Format:** {"ASA_PS": "Class [I-VI]", "Justification": "[Justification text]"}`

**2. Preliminary Anesthetic Plan Generation (Zero-Shot):**
`Generate a complete pre-operative anesthetic plan for the following clinical scenario.
**Scenario:** 68-year-old male patient, ASA II, with a history of controlled hypertension, scheduled for elective laparoscopic cholecystectomy.
**Include:** Cardiovascular and pulmonary risk assessment (Normal/Intermediate/High), Suggested anesthetic technique (General/Regional/Combined), Monitoring plan (Invasive/Non-invasive), and Post-operative pain management plan (Suggested drugs and doses).`

**3. Simplification of Educational Material for Patients:**
`Rewrite the following text about the risk of Malignant Hyperthermia to a 6th-grade reading level, using simple language and avoiding medical jargon.
**Original Text:** [Insert complex text about Malignant Hyperthermia].
**Additional Instruction:** Maintain clinical accuracy and include three clear action points for the patient.`

**4. Clinical Decision Support (Chain-of-Thought):**
`Analyze the following case and provide a differential diagnosis for intraoperative hypotension. Use the Chain-of-Thought (CoT) method to list the reasoning steps that lead to your conclusion.
**Data:** Patient in spine surgery, 45 minutes into the procedure, HR 55 bpm, BP 80/45 mmHg, CVP 8 mmHg, EtCO2 32 mmHg. General anesthesia with Sevoflurane 1.5 MAC.
**Output Format:** 1. Data Analysis. 2. Hypotheses (CoT). 3. Most Likely Diagnosis. 4. Immediate Action Plan.`

**5. Literature Summary for Research:**
`Act as an Anesthesiology researcher. Review the 5 most recent articles (last 2 years) on the use of Dexmedetomidine in pediatric cardiac surgery.
**Task:** Generate a concise summary (max. 300 words) highlighting the main conclusions, the average dosage used, and the most common adverse effects. Provide the references in Vancouver format.`
```

## Best Practices
**1. Role and Context Specification (Role Assignment):** Start the prompt by defining the LLM as a "Senior Anesthesiologist", "Anesthesiology Resident", or "Perioperative Risk Specialist" to ensure a response with the appropriate perspective and vocabulary.
**2. Few-Shot Prompting and RAG (Retrieval-Augmented Generation):** For critical tasks such as ASA-PS classification or mortality prediction, include examples of previous cases (Few-Shot) and use Retrieval-Augmented Generation (RAG) to ground the response in specific clinical data or guidelines, reducing hallucinations.
**3. Structured Output Formatting:** Require the output to be in a structured format (e.g., JSON, Markdown table) to facilitate integration with Electronic Health Record (EHR) systems and rapid analysis by the clinician.
**4. Emphasis on Safety and Risk:** Include in the prompt the explicit instruction to "prioritize patient safety" and "highlight any non-standard or high-risk recommendations" to force the model into critical analysis.
**5. Mandatory Human Validation:** Always treat the LLM's output as **decision support** and not as a final decision. Validation by a qualified healthcare professional is the fundamental best practice.

## Use Cases
**1. Clinical Decision Support:** Assistance in pre-operative risk stratification (e.g., ASA-PS classification), prediction of post-operative outcomes (e.g., 30-day mortality), and suggestion of preliminary anesthetic plans.
**2. Patient Education:** Generation of personalized and simplified educational materials about anesthetic procedures, risks, and post-operative care, adapted to the patient's literacy level.
**3. Administrative Automation:** Efficient generation of billing codes, summarization of medical records, and optimization of operating room scheduling and resource allocation.
**4. Research and Literature Analysis:** Rapid summarization of scientific articles, identification of trends across large volumes of medical literature, and assistance in drafting manuscripts and residency exam questions.
**5. Patient Support Chatbots:** Providing 24/7 support to answer common patient questions about scheduling, preparation, and recovery, reducing anxiety and calls to the clinical team.

## Pitfalls
**1. Hallucinations:** The model may generate plausible but clinically incorrect or fabricated information (e.g., wrong medication doses, inappropriate procedures). This is especially dangerous in a high-risk field like Anesthesiology.
**2. Training Bias:** LLMs may reflect biases present in the training data, leading to recommendations that may be less appropriate for specific populations (e.g., racial minorities, patients with rare comorbidities).
**3. Lack of Genuine Clinical Reasoning:** The model may be excellent at generating coherent text, but the output is not the result of complex, contextualized clinical reasoning like that of a human specialist.
**4. Inconsistency (Stochasticity):** The same prompt may generate different responses, which is unacceptable for clinical protocols. The lack of reproducibility demands caution.
**5. Ignoring Non-Verbal Context:** Text-only LLMs cannot interpret visual data (such as ultrasound images or monitoring charts), limiting their usefulness in scenarios that require multimodal analysis.
**6. Data Privacy Risk:** Entering patient data (even if de-identified) into general-purpose models may violate privacy regulations (such as HIPAA or LGPD), requiring the use of local models or models with strong security guarantees.

## URL
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12228656/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12228656/)
