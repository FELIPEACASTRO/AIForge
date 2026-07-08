# Geriatrics Prompts

## Description
**Geriatrics Prompts** refers to the specialized application of Prompt Engineering in the domain of Geriatrics and elderly healthcare. It is not a prompting technique in itself, but an **application category** aimed at optimizing interaction with Large Language Models (LLMs) to address the unique complexities of aging. The focus is on creating prompts that account for factors such as polypharmacy, frailty, multiple comorbidities, cognitive decline, and the need for adapted communication. The goal is to use AI as a tool for clinical decision support, education of professionals and patients, and the personalization of care plans, as demonstrated by the development of specific models such as the *Geriatric Care LLaMA*.

## Examples
```
## 1. Clinical Decision Support (Polypharmacy)
**Prompt:**
"You are a clinical pharmacist specializing in geriatrics. Analyze the following list of medications for an 82-year-old patient with heart failure (HF) and Parkinson's disease: [Medication A, Dose], [Medication B, Dose], [Medication C, Dose], [Medication D, Dose]. Identify potentially dangerous drug interactions, assess whether any medication is on the BEERS list, and suggest a safer alternative for the highest-risk medication, justifying the choice based on evidence."

## 2. Education and Training (Case Simulation)
**Prompt:**
"Create a high-fidelity simulation scenario for an anesthesiology resident. The patient is a 78-year-old man, frail (frailty index 5/9), undergoing surgery for a femur fracture. The scenario should begin intraoperatively, with a sudden drop in blood pressure (BP) and an emerging picture of delirium. Include the initial vital signs, the relevant medical history (including polypharmacy), and the first three questions the resident should ask to stabilize the patient and manage the delirium."

## 3. Adapted Communication (Information Simplification)
**Prompt:**
"Simplify the following excerpt from a medical report on 'Atrial Fibrillation' for a 90-year-old patient with low health literacy. The simplified text should have a reassuring tone, use short sentences, avoid medical jargon, and explain what the condition is and why the medication [Anticoagulant Name] is important. The reading level should be equivalent to the 4th grade."

## 4. Research and Development (Causal Inference)
**Prompt:**
"Based on your knowledge of the Geriatric Care LLaMA model, formulate a causal prompt to investigate the relationship between 'benzodiazepine use' and 'fall risk' in geriatric patients with mild dementia. The prompt should request the identification of confounding variables (e.g., comorbidities, dose, duration of use) and suggest an intervention protocol to mitigate the risk, structured in steps."

## 5. Personalized Care Plan
**Prompt:**
"Draft a 7-day home care plan for an 88-year-old patient who was discharged after an episode of pneumonia. The plan should focus on: 1) Fall prevention (with 3 concrete actions), 2) Nutrition (with 2 protein-rich meal suggestions), and 3) Medication management (with a simplified schedule). The tone should be encouraging and directed at the primary caregiver."

## 6. Educational Content Generation for Caregivers
**Prompt:**
"Create a quick guide (maximum 500 words) for caregivers on how to identify the early signs of dehydration in the elderly. The guide should include a checklist of 5 warning signs and 3 practical tips to encourage fluid intake, formatted to be printed in a large, legible font."

## 7. Administrative Assistance (Progress Note)
**Prompt:**
"Generate a SOAP progress note (Subjective, Objective, Assessment, Plan) for a 95-year-old patient in a nursing home.
- **Subjective:** Complaint of 4/10 pain in the left leg, reports 'poor sleep' last night.
- **Objective:** Stable vital signs. 1+/4+ edema in the left leg. Ambulating with the aid of a walker, but with an unstable gait.
- **Assessment:** Worsening of chronic pain in the left leg, elevated fall risk.
- **Plan:** Request the complete progress note, focusing on the need to reassess pain medication and immediate physical therapy for gait stabilization."
```

## Best Practices
1. **Geriatric Specificity:** Explicitly include the elderly patient's specific conditions in the prompt (e.g., "85-year-old patient, frail, with a history of HF and polypharmacy").
2. **Focus on Support, Not Replacement:** Structure prompts so that the AI acts as an "assistant" that generates drafts, analyses, or suggestions, keeping human clinical oversight and judgment as the final step.
3. **Emphasis on Usability:** When generating content for the elderly, instruct the AI to use simple language, short sentences, and clear formatting (e.g., "Use a 4th-grade reading-level language and avoid medical jargon").
4. **Ethical and Privacy Considerations:** Include safety and privacy instructions in the prompt, reminding the AI about the confidentiality of patient data (although entering real data should be avoided).
5. **Source Validation:** Whenever possible, ask the AI to cite the clinical guidelines or evidence sources that support its recommendations, especially in treatment contexts.

## Use Cases
1. **Clinical Decision Support:** Generation of differential diagnoses, personalized care plans, and treatment recommendations that account for polypharmacy and drug interactions in elderly patients.
2. **Education and Training:** Creation of high-fidelity simulation scenarios (e.g., geriatric anesthesia) and adaptive learning modules for healthcare professionals, focused on geriatric syndromes (delirium, falls, frailty).
3. **Adapted Communication:** Simplification of complex medical information into accessible language for elderly patients and their caregivers, including the generation of educational materials and health reminders.
4. **Research and Development:** Use of causal prompts to explore cause-and-effect relationships in large clinical datasets, identifying risk factors and optimizing chronic disease management.
5. **Administrative Assistance:** Generation of discharge reports, progress notes, and clinical documentation that meet regulatory requirements, freeing up professionals' time for direct care.

## Pitfalls
1. **Replacement of Clinical Judgment:** Blindly trusting AI outputs without proper validation by a healthcare professional, ignoring clinical nuances and the complexity of the elderly patient.
2. **Ignoring Usability:** Generating content for the elderly that is too complex, technical, or poorly formatted, leading to low acceptance and confusion.
3. **Privacy Violation:** Entering protected health information (PHI) into public or non-secure LLMs.
4. **Over-Generalization:** Using generic prompts that do not account for the frailty, comorbidities, and heterogeneity of the elderly population, resulting in inappropriate recommendations.
5. **Clinical "Hallucinations":** The AI generating false or outdated medical information, which is particularly dangerous in a sensitive field such as geriatrics.

## URL
[https://www.mdpi.com/2227-7390/13/15/2460](https://www.mdpi.com/2227-7390/13/15/2460)
