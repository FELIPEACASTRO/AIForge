# Obstetrics & Gynecology Prompts

## Description
Prompt Engineering for Obstetrics and Gynecology (OB/GYN) refers to the art and science of structuring natural language inputs (prompts) for large language models (LLMs) and Artificial Intelligence (AI) chatbots to obtain accurate, clinically relevant, and ethically responsible results in the field of women's health. This technique is crucial for leveraging the potential of AI across various areas, including **medical education**, **clinical decision support**, **patient communication**, and **research**. The effectiveness of OB/GYN prompts depends directly on **specificity**, the **inclusion of clinical context**, and the **requirement of sources or guidelines** to mitigate risks such as "hallucinations" and biases inherent to the models [1] [2].

## Examples
```
1.  **Clinical Decision Support (Differential Diagnosis):**
    `"Act as a senior OB/GYN resident. A 32-year-old patient, G2P1, presents with vaginal bleeding in the first trimester. The beta-hCG is 1500 mIU/mL. List the 5 most likely differential diagnoses, the relative probability of each, and the crucial next diagnostic step for each scenario. Format the response in a table."`

2.  **Medical Education (Simulation Scenario):**
    `"Create a 10-minute simulation scenario for a 3rd-year medical student on the management of severe pre-eclampsia. The scenario should include: 1) A brief patient history, 2) Three critical questions the student should ask, and 3) The first-line initial treatment based on the ACOG (2023) guidelines."`

3.  **Patient Communication (Message Response):**
    `"Rewrite the following clinical message in simple, empathetic language for a patient with low health literacy. The patient is 45 years old and has been diagnosed with uterine fibroids. The original message is: 'Your fibroids are intramural and submucosal, measuring 5 cm and 3 cm, respectively. We will discuss treatment options, including embolization and myomectomy, at your next appointment.'"`

4.  **Literature and Guideline Review:**
    `"Compare the cervical cancer screening recommendations for immunocompromised patients (HIV positive) versus the general population, according to the most recent guidelines (2023-2025) from the USPSTF and ACOG. Cite the primary source for each recommendation."`

5.  **Workflow Optimization (Documentation):**
    `"Generate a SOAP note template (Subjective, Objective, Assessment, Plan) for a routine family planning consultation, focusing on the discussion of long-acting reversible contraceptive (LARC) methods. Include sections for informed consent and follow-up."`

6.  **Research and Data Analysis (Statistics):**
    `"Explain the concept of 'Number Needed to Treat' (NNT) and how it applies to the decision to prescribe progesterone for the prevention of preterm birth in patients with a short cervix. Use a didactic tone and provide a hypothetical numerical example."`
```

## Best Practices
*   **Define the Persona and Target Audience:** Start the prompt by instructing the AI to assume a specific role (e.g., "Act as a fetal ultrasound specialist", "Act as a patient educator") and define the target audience (e.g., "for a resident", "for a lay patient") [1].
*   **Clinical Specificity:** Include crucial clinical details (age, parity, test results, comorbidities) and context (e.g., "first trimester", "postmenopausal") to avoid generic responses [1].
*   **Require References and Guidelines:** Explicitly ask the AI to cite the source of its information (e.g., "based on the ACOG 2023 guidelines", "cite the original study") to facilitate validation [3].
*   **Clear Output Structure:** Specify the desired format (e.g., "in a table", "in SOAP note format", "list of 5 points") to ensure the clinical usability of the result [1].
*   **Structured Feedback Loop:** The initial prompt should be followed by refinement prompts (e.g., "Refine the response to include the magnesium sulfate dosage", "Correct the dosage error to 4g IV") to iterate and validate clinical accuracy [3].

## Use Cases
*   **Clinical Decision Support:** Generation of differential diagnoses, initial management plans, and comparison of treatment protocols (e.g., management of postpartum hemorrhage, gynecologic cancer screening) [1].
*   **Education and Training:** Creation of simulation scenarios, multiple-choice questions, summaries of complex articles, and lesson plans for residents and students [2].
*   **Patient Communication:** Drafting responses to patient messages, creation of educational materials in accessible language, and optimization of the readability of consent documents [1].
*   **Research and Literature Review:** Rapid synthesis of large volumes of literature, identification of research gaps, and comparison of clinical trial results [1].

## Pitfalls
*   **Clinical Hallucinations:** AI may generate factually incorrect or clinically implausible information. Human validation by a healthcare professional is **mandatory** [1].
*   **Biases and Inequity:** Models may perpetuate biases present in the training data, leading to recommendations that may be inadequate for minority or underrepresented population groups [1].
*   **Lack of Real-Time Awareness:** AI does not have access to real-time clinical data (e.g., electronic health records) or the latest guideline updates that emerged after its training [1].
*   **Data Privacy Violation (HIPAA/LGPD):** Entering protected health information (PHI) into prompts violates privacy regulations. Prompts must be **anonymized** and **de-identified** [1].

## URL
[https://www.sciencedirect.com/science/article/pii/S0002937825002285](https://www.sciencedirect.com/science/article/pii/S0002937825002285)
