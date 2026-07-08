# Treatment Planning Prompts

## Description
**Treatment Planning Prompts** are prompt engineering instructions designed to assist healthcare professionals (physicians, psychologists, therapists, dentists) in creating, refining, and documenting comprehensive and personalized treatment plans. This technique leverages the ability of Large Language Models (LLMs) to synthesize complex clinical information, evidence-based guidelines, and patient-specific data to generate objectives, interventions, and expected outcomes in a structured way. The main focus is to improve efficiency, regulatory compliance (such as HIPAA/GDPR), and quality of care, transforming clinical documentation from a time-consuming administrative task into an AI-assisted process. The effectiveness of these prompts depends on including precise clinical details, specifying the desired output format (e.g., SOAP, DAP, SMART goals), and adhering to ethical principles, such as data anonymization. This technique is fundamental to the adoption of AI in modern clinical practice, especially in areas such as mental health and specialized medicine.

## Examples
```
1. **Generating a Structured Treatment Plan (Mental Health):**
```
Act as a clinical psychologist specialized in Cognitive Behavioral Therapy (CBT).
**Patient:** [Fictional Name], 32 years old, female.
**Diagnosis:** Generalized Anxiety Disorder (GAD).
**Target Symptoms:** Excessive daily worry, insomnia, muscle tension.
**History:** Prior treatment attempt with medication (SSRI) with side effects.
**Task:** Create a 12-session treatment plan. The plan must include:
1. One **Long-Term Goal** (SMART).
2. Three **Short-Term Goals** (SMART) focused on psychoeducation, cognitive restructuring, and relaxation.
3. Specific CBT **Interventions** for each goal.
4. **Discharge Criteria** (measurable).
```

2. **Refining Medical Interventions (Cardiology):**
```
Act as a consulting cardiologist.
**Scenario:** 68-year-old patient with Heart Failure with Reduced Ejection Fraction (HFrEF, EF 35%), Hypertension, and Type 2 Diabetes.
**Current Medication:** Enalapril and Metoprolol.
**Task:** Recommend the next step in optimizing the pharmacological treatment, in accordance with the 2022 ACC/AHA guidelines.
**Output:** List the drug classes (e.g., SGLT2i, MRA) that should be added, justifying the choice based on mortality and hospitalization benefits.
```

3. **Creating SMART Goals (Occupational Therapy):**
```
Act as an occupational therapist.
**Patient:** 7-year-old child with Autism Spectrum Disorder (ASD).
**Problem:** Difficulty dressing independently (buttoning shirts).
**Task:** Generate 3 SMART Goals for the next 8 weeks of therapy.
**Format:** Goal (Specific, Measurable, Achievable, Relevant, Time-bound) and Associated Intervention.
```

4. **Drafting Documentation for Audit (Dentistry):**
```
Act as a dental coding and billing specialist.
**Procedure:** Placement of a crown on a molar (code D2740).
**Clinical Justification:** Extensive cuspal fracture, failed previous restoration.
**Task:** Draft a concise and professional "Statement of Medical Necessity" for the insurer, explaining why the crown is the necessary treatment and not a simple restoration.
```

5. **Generating Patient Educational Material:**
```
Act as a health educator.
**Topic:** New insulin regimen (use of a pen and blood glucose monitoring).
**Audience:** Newly diagnosed diabetic patient, 55 years old, with low health literacy.
**Task:** Create a step-by-step guide, using simple language (5th-grade reading level), for the correct use of the insulin pen. Include 3 safety alert points.
```

6. **Risk Analysis and Crisis Planning (Psychiatry):**
```
Act as a psychiatrist.
**Scenario:** Patient with Bipolar Disorder, a history of medication non-adherence, and a recent increase in passive suicidal ideation.
**Task:** Develop a crisis "Safety Plan." The plan must include:
1. Three crisis warning signs.
2. Three immediate coping strategies.
3. Two emergency contacts (fictional).
4. The next treatment step if the strategies fail.
```
```

## Best Practices
**Explicitness and Clinical Specificity:** Always include relevant clinical details, such as age, comorbidities, disease stage, and specific guidelines (e.g., "per 2023 ADA guidelines"). This reduces ambiguity and increases the clinical validity of the response.
**Contextual Relevance:** Provide as much context as possible. In mental health, this includes the diagnosis, target symptoms, treatment history, and the patient's environment.
**Iterative Refinement:** Do not accept the first response. Use clinical feedback to refine the prompt (e.g., "Refine the treatment goal to be more measurable, focusing on reducing the frequency of panic attacks from 5 to 2 per week").
**Ethical and Privacy Considerations:** Never enter personally identifiable information (PII). Use anonymized data or hypothetical scenarios.
**Evidence-Based Practices:** Ask the model to cite sources or align recommendations with up-to-date clinical guidelines (e.g., "What are the evidence-based interventions for CBT in treating GAD, citing recent studies?").
**Role Definition (Role-Playing):** Begin the prompt by defining the AI's role (e.g., "Act as a clinical psychologist specialized in Cognitive Behavioral Therapy (CBT)").

## Use Cases
nan

## Pitfalls
**Privacy Violation (HIPAA/GDPR):** Entering personally identifiable information (PII) or protected health information (PHI) into consumer LLMs.
**Clinical Hallucinations:** The AI may generate treatment recommendations that are factually incorrect, outdated, or dangerous. Human verification is mandatory.
**Over-Generalization:** Using vague prompts (e.g., "Treatment plan for depression") that result in generic plans not tailored to the individual patient's complexity.
**Bias and Inequity:** The model may perpetuate biases from training data, leading to suboptimal or unfair treatment plans for certain demographic groups.
**Lack of Context:** Failing to provide crucial information (comorbidities, allergies, treatment failure history) that is essential for a safe and effective plan.
**Over-Reliance:** Blindly trusting the AI's output without applying clinical judgment and professional experience.

## URL
[https://www.jmir.org/2025/1/e72644](https://www.jmir.org/2025/1/e72644)
