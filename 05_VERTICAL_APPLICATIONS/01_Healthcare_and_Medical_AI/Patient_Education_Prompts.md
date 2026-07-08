# Patient Education Prompts

## Description
**Patient Education Prompts** are structured and detailed instructions provided to Large Language Models (LLMs) to generate informative, personalized, and accessible materials about health conditions, treatments, procedures, or preventive care. In the clinical context, these prompts are an essential tool for improving communication between healthcare professionals and patients, ensuring that the information is clinically accurate, ethically responsible, and adapted to the individual's level of understanding [1]. The technique requires the inclusion of crucial elements such as the **clinical objective** (e.g., explaining the use of a new medication), the **target audience** (e.g., lay patient, caregiver), the **output format** (e.g., leaflet, analogy, FAQ), and the **evidence base** (e.g., 2023 AHA guidelines) [1] [2]. The effective use of these prompts turns AI into a powerful assistant for creating personalized health content at scale.

## Examples
```
**1. Creating an Informational Leaflet (Lay Level):**
```
Act as a health educator. Create a one-page informational leaflet on "Arterial Hypertension" for an adult patient with low health literacy. The leaflet should include:
1. A simple analogy to explain what high blood pressure is.
2. Three easy-to-implement lifestyle changes.
3. What to do if they forget to take the medication.
4. Clear, encouraging language with no medical jargon.
```

**2. Procedure Explanation (Personalized):**
```
Create a step-by-step explanation for a 65-year-old patient with type 2 diabetes who will undergo a colonoscopy. The text should be written in a calm and reassuring tone. Include detailed instructions on the preparation diet (the day before) and what to expect during and after the procedure. The patient has moderate anxiety about medical procedures.
```

**3. Comparison of Treatment Options:**
```
For a 40-year-old patient diagnosed with mild to moderate Crohn's Disease, generate a comparative summary between treatment options with oral Mesalazine and Biologics (anti-TNF). The summary should be presented in table format, highlighting efficacy, route of administration (oral vs. injection), and the most common potential side effects. Use easy-to-understand language.
```

**4. Conversation Script for the Doctor:**
```
Generate a script of 5 key questions that a patient newly diagnosed with Systemic Lupus Erythematosus (SLE) should ask their rheumatologist at the next appointment. The questions should focus on prognosis, disease monitoring, and flare management.
```

**5. Hospital Discharge Instructions (Post-Surgery):**
```
Prepare discharge instructions for a patient who has just undergone total knee replacement surgery. The document should be an easy-to-follow checklist covering:
1. Incision care.
2. Warning signs of infection (what to look for and when to call).
3. Pain medication schedule (with generic and brand names).
4. Activity restrictions and initial exercises.
```

**6. Answering Common Questions (FAQ Format):**
```
Create a Frequently Asked Questions (FAQ) section for parents of children who will receive the HPV vaccine. Address at least 5 common myths about the vaccine (e.g., it causes infertility, it is unnecessary) based on evidence from the CDC (Centers for Disease Control and Prevention).
```

**7. Explanation by Analogy:**
```
Explain the mechanism of action of Insulin to a teenager newly diagnosed with Type 1 Diabetes. Use the analogy of a "key" that opens the "door" of the cell so that "energy" (glucose) can enter.
```
```

## Best Practices
**1. Clinical and Contextual Specificity:** Include patient details (age, comorbidities, disease stage) and references to up-to-date clinical guidelines (e.g., ADA, NCCN) to reduce ambiguity and increase the clinical validity of the response [1].
**2. Clear Definition of the Target Audience:** Specify the patient's health literacy level (e.g., "lay", "teenager", "elderly with low vision") so that the AI adjusts the language, tone, and format (e.g., leaflet, checklist, analogy) [2] [3].
**3. Iteration and Refinement:** The AI's first result may be generic. Use a structured feedback loop to refine the prompt, requesting adjustments in tone, the inclusion of specific information, or the simplification of complex terms [1].
**4. Emphasis on Ethics and Privacy:** When formulating prompts, prioritize the de-identification of sensitive data. Use AI to generate educational content based on conditions, not on insecure electronic health records [1].
**5. Evidence Verification:** Always request that the AI base the material on current evidence and guidelines. **Cross-check the generated information with authoritative sources** (e.g., UpToDate, PubMed) to mitigate the risk of "hallucinations" or outdated information [1].

## Use Cases
**1. Creating Personalized Health Materials:** Rapid generation of leaflets, infographics, FAQs, and explanatory videos adapted to the condition and demographic profile of a specific patient [2].
**2. Optimizing Doctor-Patient Communication:** Creating conversation scripts for doctors or patients, ensuring that all critical information is addressed during appointments [3].
**3. Simplifying Complex Documents:** Transforming complex medical terms, test results, or treatment plans into accessible and easy-to-understand language for the lay patient [2].
**4. Supporting Treatment Adherence:** Developing medication reminders, diet plans, or post-operative exercise routines in engaging and motivational formats.
**5. Training Healthcare Professionals:** Creating simulation scenarios or virtual clinical cases to train new professionals on how to effectively explain complex conditions to patients [4].
**6. Generating Multilingual Content:** Translating and culturally adapting educational materials for patients who speak different languages, ensuring clinical accuracy [2].

## Pitfalls
**1. Generation of Clinical "Hallucinations":** AI can generate false or outdated information (hallucinations) that, in a health context, is dangerous. **Mitigation:** Always require source citations and verify the content against official guidelines [1].
**2. Failure to Personalize:** Generic prompts result in generic materials. The lack of patient details (age, comorbidities) can lead to inappropriate or contraindicated advice [1].
**3. Inappropriate Language:** The generated material may use complex medical jargon, making it inaccessible to the target audience, especially those with low health literacy [2]. **Mitigation:** Specify the desired reading level (e.g., "6th-grade level").
**4. Bias and Inequity:** AI can perpetuate data biases, resulting in materials that are not culturally sensitive or that overlook the access barriers of certain population groups [1]. **Mitigation:** Include ethical and equity constraints in the prompt.
**5. Privacy Violation (HIPAA/LGPD):** Using non-de-identified patient data in prompts for insecure LLMs violates privacy laws. **Mitigation:** Use only generic or de-identified clinical information [1].

## URL
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12439060/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12439060/)
