# Emergency Medicine Prompts (GRACE Framework)

## Description
**Emergency Medicine (EM) Prompts** refer to the application of *Prompt Engineering* techniques to optimize interaction with Large Language Models (LLMs) in the acute care environment. The goal is to ensure that AI responses are **reliable, clinically relevant, accurate, and actionable** for physicians, residents, and other healthcare professionals (APCs) in high-pressure, time-limited situations.

The main advance in this area is the **GRACE Framework** (Ground Rules, Roles, Ask, Chain of Thought, Expectations), developed by emergency clinicians to structure prompts systematically. This framework aims to mitigate the risks inherent in using LLMs in the clinic, such as "hallucinations" (factually incorrect information) and the amplification of biases, by forcing the model to adhere to evidence standards and justify its reasoning. The technique is crucial for transforming LLMs from general information tools into safe and effective clinical decision assistants.

## Examples
```
**1. Evidence Appraisal Prompt (Full GRACE)**

```
You are a senior emergency medicine researcher, skeptical and harm-focused, with experience in toxicology and critical appraisal of literature. Your role is to guide board-certified physicians in the appraisal of clinical evidence.

**G (Ground Rules):** Base all responses exclusively on randomized clinical trials, systematic reviews, and ACEP/Cochrane guidelines (post-2015). Cite verifiable sources for each statement; if the evidence is inconclusive, state so explicitly.
**R (Role):** You are the specialist, I am the emergency physician.
**A (Ask):** What is the current evidence and recommended protocol for the use of tranexamic acid (TXA) in patients with mild to moderate traumatic brain injury (TBI)?
**C (Chain of Thought):** Proceed step by step: 1) Identify the main trials (e.g. CRASH-3). 2) Assess the methodology and risk of bias. 3) Synthesize the findings. 4) Apply to the emergency department context.
**E (Expectations):** Structure the output as: 1. Evidence Summary (Bullet points). 2. Clinical Rationale. 3. Practical Recommendation. Use formal language and include full citations at the end.
```

**2. Differential Diagnosis Prompt (R+A+E)**

```
**R (Role):** You are an emergency attending physician with 20 years of experience.
**A (Ask):** A 65-year-old diabetic patient presents with diffuse abdominal pain and vomiting. Stable vital signs. Physical exam with a mildly distended abdomen that is tender to palpation. What is the most likely differential diagnosis?
**E (Expectations):** Provide a list of 5 diagnoses, ranked by probability, and a brief 1-sentence justification for each one.
```

**3. Discharge Instructions Prompt (R+A+E)**

```
**R (Role):** You are a patient educator.
**A (Ask):** Create discharge instructions for a 30-year-old patient diagnosed with non-purulent cellulitis of the leg, treated with cephalexin.
**E (Expectations):** Use 6th-grade-level language. Include: 1. Warning signs to return to the ED. 2. Wound care. 3. Instructions about the antibiotic (dose, duration, side effects). Format as a numbered list.
```

**4. Case Summary Prompt for Handoff (A+C+E)**

```
**A (Ask):** Summarize the following case for a quick handoff to the admitting team: [Insert the case summary, including HPI, Physical Exam, Labs/Imaging, and Plan].
**C (Chain of Thought):** Follow the SBAR structure (Situation, Background, Assessment, Recommendation).
**E (Expectations):** The output should be at most 4 sentences, one for each SBAR section.
```

**5. Scenario Simulation Prompt (R+A+C)**

```
**R (Role):** You are the examiner of the American Board of Emergency Medicine (ABEM). I am the candidate.
**A (Ask):** I present a 45-year-old patient with atypical chest pain and an ECG with ST depression in V4-V6. What is your immediate next step?
**C (Chain of Thought):** Justify each decision with the relevant pathophysiology or guideline before proceeding to the next step. Keep the simulation interactive.
```

**6. Rapid Topic Review Prompt (G+A+E)**

```
**G (Ground Rules):** Use only the most recent guidelines from the American Heart Association (AHA) and the American College of Cardiology (ACC).
**A (Ask):** What are the absolute indications and contraindications for thrombolysis in a patient with acute ischemic stroke who arrives within the 3-hour window?
**E (Expectations):** Respond in a table with two columns: "Absolute Indications" and "Absolute Contraindications".
```
```

## Best Practices
**Adopt the GRACE Framework (Ground Rules, Roles, Ask, Chain of Thought, Expectations):** This is the main set of recommended practices for EM prompts.
*   **G (Ground Rules):** Explicitly define the constraints and evidence standards (e.g.: "Base yourself only on peer-reviewed literature. Provide citations. Do not invent sources.").
*   **R (Roles):** Assign specific roles to the user and the LLM (e.g.: "You are an experienced EM attending, I am a resident.").
*   **A (Ask):** Be explicit and focused on the core task (e.g.: "Provide the differential diagnosis for this case...").
*   **C (Chain of Thought):** Ask the LLM to "explain its reasoning step by step" to expose the decision-making process and reduce the "black box" risk.
*   **E (Expectations):** Define the output format and style for usability (e.g.: "Respond concisely in a bulleted list.").
**Prioritize Reliable Sources:** Use LLMs that are "grounded" in peer-reviewed medical literature and clinical guidelines, such as OpenEvidence, instead of general-purpose models for critical decisions.
**Include Relevant Clinical Data:** For case prompts, include crucial information such as age, sex, comorbidities, vital signs, exam findings, and current medications to avoid "hallucinations" due to lack of data.

## Use Cases
**Clinical Decision Support:**
*   **Differential Diagnosis:** Generation of DDX lists for complex or atypical cases, based on patient presentation data.
*   **Treatment Recommendation:** Consultation of guidelines and evidence for the management of acute conditions (e.g.: sepsis, MI, stroke).
*   **Exam Interpretation:** Assistance in interpreting unusual laboratory or imaging findings.
**Education and Training:**
*   **Case Simulation:** Creation of interactive clinical scenarios for training residents and students, simulating the role of an examiner or patient.
*   **Rapid Topic Review:** Synthesis of complex information from toxicology, pharmacology, or procedures into easily digestible formats.
**Operational Efficiency:**
*   **Personalized Discharge Instructions:** Generation of clear discharge instructions, in plain language and adapted to the patient's conditions and literacy level.
*   **Handoff Communication (SBAR):** Rapid and structured summarization of cases for care transitions between teams (e.g.: from the ED to the ICU or ward).
*   **Clinical Documentation:** Creation of drafts of clinical notes or justifications for procedures.

## Pitfalls
**Hallucinations and False Sources:** The greatest risk. General-purpose LLMs may invent studies, authors, or citations. **Mitigation:** Use the ground rule "Provide verifiable citations and do not invent sources."
**Data Biases:** The models may perpetuate existing health biases (e.g.: underestimating pain in certain populations). **Mitigation:** Ask the LLM to consider equity and health disparities in its analysis (e.g.: "Consider how management may differ in populations with limited healthcare access.").
**Lack of Clinical Context:** The AI does not have access to the patient, the monitor, or the environment. **Mitigation:** Be as detailed as possible in the prompt, providing all relevant vital signs and exam data.
**Non-Actionable Responses:** Responses that are too long, dense, or academic for an emergency environment. **Mitigation:** Use the Expectations (E) rule to require concise formats (e.g.: "Bottom line up front", bulleted list).
**Over-Reliance:** Using AI as a substitute for clinical judgment. **Mitigation:** AI should be used as a reasoning assistant, not as a final authority. The GRACE Framework encourages critical review of the AI's reasoning.

## URL
[https://www.acepnow.com/article/search-with-grace-artificial-intelligence-prompts-for-clinically-related-queries/](https://www.acepnow.com/article/search-with-grace-artificial-intelligence-prompts-for-clinically-related-queries/)
