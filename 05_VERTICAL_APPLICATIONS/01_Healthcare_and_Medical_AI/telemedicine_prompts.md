# Telemedicine Prompts

## Description
**Telemedicine Prompts** refer to structured commands and instructions, developed by healthcare professionals or patients, to interact with Large Language Models (LLMs) and Generative AI in the context of delivering remote healthcare. Prompt Engineering in Telemedicine is crucial to optimize the accuracy, relevance, and safety of AI responses, turning it into a tool for clinical decision support, administrative automation, and patient communication [1] [2]. These prompts are designed to handle the complexity and sensitivity of health data, requiring high specificity and contextualization to ensure patient safety and regulatory compliance (such as HIPAA and LGPD) [3]. The effective use of prompts allows AI to assist with tasks such as initial triage, generating consultation summaries, drafting clinical notes, and creating personalized educational materials for the patient [1].

## Examples
```
**1. Triage and Prioritization (Zero-Shot Prompting)**
`**Prompt:** "You are a telemedicine triage assistant. Analyze the following symptoms and determine the urgency level (Emergency, Urgent, Routine) and the most appropriate medical specialty. **Symptoms:** 45-year-old female patient reports sudden, severe headache (worst pain of her life), accompanied by neck stiffness and vomiting for 2 hours. History of controlled hypertension. **Output Format:** Table with Urgency, Specialty, and Rationale."`

**2. Consultation Summary Generation (One-Shot Prompting)**
`**Prompt:** "Generate a concise summary of the teleconsultation for the electronic medical record. **Format Example:** [Date], [Patient Name], [Reason for Visit], [Main Findings], [Treatment Plan]. **Consultation Data:** 11/08/2025, João Silva, 68 years old. Reason: Follow-up for Type 2 Diabetes Mellitus. Findings: Average fasting glucose of 150 mg/dL over the past 2 weeks. Plan: Adjust Metformin to 1000mg twice daily and order HbA1c."`

**3. Clinical Decision Support (Chain-of-Thought Prompting)**
`**Prompt:** "You are an AI clinical consultant. For the following case, list the 3 most likely differential diagnoses and then justify the reasoning for each one, citing the source (guideline or article). **Case:** 72-year-old male patient with progressive dyspnea and lower limb edema. Physical exam: crackles at the lung bases and jugular venous distension. History of myocardial infarction 5 years ago. **Instruction:** First, reason about the pathophysiology. Then, list the differentials."`

**4. Patient Communication (Layperson Language)**
`**Prompt:** "Explain the medical condition 'Atrial Fibrillation' to a 55-year-old patient with a high school education, using simple analogies and avoiding medical jargon. Include the importance of anticoagulant medication and the warning signs. **Tone of Voice:** Empathetic and educational."`

**5. Creating a Consultation Script (Meta-Prompting)**
`**Prompt:** "Create a structured consultation script for a follow-up teleconsultation with patients who have generalized anxiety. The script should include: 1. Initial triage questions (GAD-7 scale). 2. Assessment of treatment adherence. 3. Discussion of coping strategies (e.g., breathing technique). 4. Setting goals for the next session. **Format:** Numbered and detailed list."`

**6. Remote Monitoring Data Analysis**
`**Prompt:** "Analyze the remote monitoring data of a patient with Congestive Heart Failure (CHF) and identify any concerning trends. **Data:** Daily weight (2kg increase over 3 days), Blood Pressure (average 135/85 mmHg), Heart Rate (average 88 bpm). **Instruction:** Focus on the relationship between weight gain and fluid retention, suggesting an immediate action for the responsible physician."`

**7. Generating Personalized Educational Material**
`**Prompt:** "Generate a 7-day weekly meal plan for a 40-year-old vegetarian patient with a recent diagnosis of hypercholesterolemia (high LDL cholesterol). The plan should be rich in soluble fiber and include the approximate daily calorie count (maximum 2000 kcal). **Format:** Table with breakfast, lunch, and dinner."`
```

## Best Practices
**1. Be Specific and Contextualized:** Include the AI's role (e.g., "You are a triage assistant"), the clinical context (e.g., "65-year-old patient with a history of CHF"), and the clear objective (e.g., "generate a discharge summary"). **2. Use a Reasoning Structure (Chain-of-Thought):** Ask the AI to detail its reasoning process before the final answer, especially for complex diagnoses or treatment plans. This increases transparency and reliability. **3. Reference Guidelines:** Whenever possible, ask the AI to base its response on up-to-date clinical guidelines (e.g., "according to the AHA 2023 guidelines"). **4. Define the Output Format:** Specify the desired format (e.g., "bulleted list", "table", "layperson-language text") to ensure usability. **5. Prioritize Safety and Ethics:** Remind the AI about data confidentiality (HIPAA/LGPD) and the need for all suggestions to be reviewed by a human professional.

## Use Cases
**1. Remote Clinical Decision Support:** Assistance in formulating differential diagnoses, suggesting treatment plans, and interpreting laboratory or imaging exams (teleradiology) [2]. **2. Automation of Administrative Tasks:** Generating discharge summaries, progress notes, referral letters, and completing insurance forms, freeing up the healthcare professional's time [1]. **3. Patient Communication and Education:** Creating personalized educational materials, answering frequently asked questions (FAQ), and drafting empathetic and clear messages for patients, overcoming health literacy barriers [1]. **4. Patient Triage and Prioritization:** Using chatbots and AI assistants to collect symptoms, assess urgency, and direct the patient to the appropriate level of care (teleconsultation, emergency room, etc.) [4]. **5. Workflow Optimization:** Creating detailed flowcharts of the patient journey in telemedicine, from scheduling to post-consultation follow-up.

## Pitfalls
**1. Clinical Hallucinations:** AI can generate false, clinically incorrect, or outdated information (hallucinations), which is dangerous in a healthcare context. **Mitigation:** Require citations from reliable sources and always have a final review by a human professional. **2. Bias and Inequity:** If the AI's training data is biased, prompts can lead to recommendations that perpetuate health disparities in minority groups. **Mitigation:** Include detailed demographic and clinical data in the prompt to ensure relevance and equity. **3. Privacy Violation (HIPAA/LGPD):** Using sensitive patient data in prompts can violate privacy regulations. **Mitigation:** Use only anonymized data or prompts that handle generic or synthetic information, and never enter Protected Health Information (PHI) into AI models not validated for such use. **4. Lack of Human Context:** AI does not replace empathy and clinical judgment. Prompts that seek a "final diagnosis" without human interaction ignore the complexity of medicine. **Mitigation:** Use AI as an **assistant** (support tool), not as a final decision-maker. **5. Vague Prompts:** Generic prompts lead to generic and useless responses. **Mitigation:** Be as specific as possible, defining the AI's role, target audience, output format, and content constraints.

## URL
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12439060/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12439060/)
