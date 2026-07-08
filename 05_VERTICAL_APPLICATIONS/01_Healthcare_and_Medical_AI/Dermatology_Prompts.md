# Dermatology Prompts

## Description
**Dermatology Prompts** refer to prompt engineering commands specifically crafted to interact with Large Language Models (LLMs) and Multimodal Models (LMMs) in the context of dermatology. The goal is to turn AI into a clinical, educational, or research assistant, capable of helping with tasks such as differential diagnosis, treatment planning, exam interpretation, medical education, and optimization of administrative routines. The effectiveness of these prompts lies in their ability to provide detailed clinical context, define the AI's role as a specialist, and require a structured, referenced output format. The field gained prominence starting in 2023, with the advancement of models such as GPT-4, which demonstrated high accuracy in generating clinical vignettes and decision support, provided the prompt is built with rigor and attention to ethics and the need for human validation. The use of prompts in dermatology requires caution, with AI being a support tool and not a substitute for the physician's clinical judgment.

## Examples
```
**1. Structured Differential Diagnosis (Role + Task + Format)**
```
Act as an experienced dermatologist. Analyze the following case: Male patient, 45 years old, presents with erythematous, pruritic papules on the trunk and limbs, with symmetric distribution, for 3 weeks. History of recent stress. No fever or systemic symptoms.
Task: List the 5 most likely differential diagnoses, from most common to rarest. For each one, create a table with the following fields: 'Diagnosis', 'Key Differentiating Signs', 'Suggested Complementary Exam'.
```

**2. Treatment Plan Optimization (Refinement)**
```
Act as a pharmacologist specialized in dermatology. The patient was diagnosed with Plaque Psoriasis (PASI 15). The initial plan is Methotrexate 15mg/week.
Task: Assess the safety and efficacy of this plan. Suggest 3 second-line treatment alternatives (biologics or orals) and create a follow-up prompt for the patient, explaining the side effects of Methotrexate in layperson's terms.
```

**3. Image Analysis (Multimodal Prompt)**
```
Act as a dermatoscopist. The attached image shows a pigmented lesion on the back.
Task: Describe the lesion using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution). Based on the description, what is the most likely diagnostic hypothesis (Melanoma, Dysplastic Nevus, or Seborrheic Keratosis)? Justify your answer based on the dermatoscopic criteria and suggest the next step (excisional biopsy or follow-up).
```

**4. Creation of a Clinical Vignette for Medical Education**
```
Act as a USMLE Step 2 CK examiner.
Task: Create a 150-word clinical vignette about a case of Atopic Dermatitis in a pediatric patient, focusing on triggering factors and initial management. The vignette should be followed by a multiple-choice question about the first-line treatment.
```

**5. Research and Synthesis of Scientific Articles**
```
Act as a senior researcher.
Task: Search PubMed for the 3 most recent articles (2024-2025) on the use of Artificial Intelligence for early detection of Non-Melanoma Skin Cancer. Summarize the main findings, the methodology used (type of AI), and the accuracy rate (AUC). Present the result in table format.
```

**6. Patient Communication Protocol**
```
Act as a medical communication specialist.
Task: Draft a concise, empathetic text to be given to a patient newly diagnosed with Vitiligo, explaining the condition, the treatment options (phototherapy, topicals), and the importance of psychological support. The tone should be informative and encouraging.
```
```

## Best Practices
**1. Define the Role and Context:** Always begin the prompt by defining the AI's role (e.g., "Act as a dermatologist with 20 years of experience and a focus on cutaneous oncology"). This guides the tone and knowledge base. **2. Standard Structure (RTF):** Use the **Role, Task, Format** framework. Require the output format (table, list, summary) to structure the response. **3. Anonymization and Ethics:** **NEVER** enter patient data that could identify them. Use only anonymized clinical data or hypothetical scenarios. **4. Require References:** Ask the AI to cite sources (e.g., "Base your answer on the guidelines of the Brazilian Society of Dermatology and cite the PubMed articles"). **5. Depth and Detail:** For complex diagnoses, request a "deep research" to obtain longer, better-grounded reports. **6. Use Multimodal Models:** For lesion analysis, use models that accept images (GPT-4V, Gemini Pro Vision), describing the image with as much clinical detail as possible in the prompt.

## Use Cases
nan

## Pitfalls
**1. Hallucinations and False References:** The AI may "hallucinate" (generate false information) or cite nonexistent articles and guidelines. **Countermeasure:** Always require source citations and verify them manually, especially in critical decisions. **2. Confidentiality Breach:** Entering patient data that could identify them (name, national ID, address) into public AIs. **Countermeasure:** Use only 100% anonymized data or hypothetical scenarios. **3. Lack of Clinical Context:** Prompts that are too short or generic (e.g., "What is acne?") result in superficial responses that are useless for clinical practice. **Countermeasure:** Provide as much clinical, demographic, and patient-history detail as possible. **4. Overconfidence (Automation Bias):** Accepting the AI's suggestion without proper clinical reasoning and human validation. **Countermeasure:** The AI is an assistant; final responsibility for diagnosis and treatment always rests with the physician. **5. Multimodal Limitation:** AI models may struggle to interpret nuances in low-quality or poorly lit images. **Countermeasure:** Supplement the image with a detailed clinical description in the prompt.

## URL
[https://www.nature.com/articles/s41746-025-01650-x](https://www.nature.com/articles/s41746-025-01650-x)
