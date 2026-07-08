# Prompt Engineering for Biomedical Applications

## Description
Prompt Engineering for Biomedical Applications refers to the art and science of crafting precise, contextual instructions (prompts) for **Large Language Models (LLMs)**, such as GPT-4 and Claude, in order to automate or semi-automate complex tasks in the field of Biomedical Engineering and clinical research. The main focus is on **evidence synthesis** and **literature screening** for systematic reviews and meta-analyses, where LLMs are used to classify thousands of scientific article abstracts.

This technique is based on converting structured clinical eligibility criteria, such as the **PICO (Population, Intervention, Comparison, Outcome)** framework, into actionable instructions for the model. The goal is to reduce manual workload, accelerate evidence synthesis, and maintain methodological rigor, allowing researchers to focus on the critical appraisal of full texts. Effectiveness depends on clarity, specificity, and the adopted prompting strategy (e.g., *zero-shot*, *few-shot*, *soft*, or *strict*).

## Examples
```
## Example 1: Systematic Review Screening (Soft Prompt - High Sensitivity)

**Scenario:** Initial abstract screening for a systematic review on the use of **Gold Nanoparticles (AuNPs)** for drug delivery in oncology.

**Prompt:**
```
You are a literature screening assistant. Your task is to classify the provided abstract as 'ACCEPT' or 'REJECT' for a systematic review.

Inclusion Criteria (PICO):
1. Population (P): In vivo (animal or human) or in vitro studies addressing cancer cells.
2. Intervention (I): Use of Gold Nanoparticles (AuNPs) as a drug delivery system.
3. Comparison (C): Any control (e.g., free drug, other nanocarrier, placebo).
4. Outcome (O): Assessment of antitumor efficacy or toxicity.
5. Study Design: Original article (not review, editorial, or letter).

Decision Rule (Soft): If the abstract does not explicitly contradict any of the criteria above, respond 'ACCEPT'. Respond 'REJECT' only if there is a clear violation (e.g., study focused on cardiovascular diseases, use of silver nanoparticles).

Abstract to be evaluated: [INSERT ABSTRACT HERE]

Decision:
```

## Example 2: Systematic Review Screening (Strict Prompt - High Specificity)

**Scenario:** Final abstract screening for a systematic review focused on **Randomized Controlled Trials (RCTs)** on the use of **Bioresorbable Vascular Scaffolds (BVS)** in diabetic patients.

**Prompt:**
```
You are an expert reviewer. Classify the abstract as 'ACCEPT' or 'REJECT'.

Inclusion Criteria (PICO/S):
1. Population (P): Human, adult patients (≥18 years), with Type 2 Diabetes Mellitus.
2. Intervention (I): Implantation of Bioresorbable Vascular Scaffolds (BVS).
3. Comparison (C): Drug-Eluting Metallic Stents (DES).
4. Outcome (O): Restenosis or Stent Thrombosis rates at 12 months.
5. Study Design (S): **MUST** be an explicitly stated Randomized Controlled Trial (RCT).

Decision Rule (Strict): Respond 'ACCEPT' only if the abstract **explicitly mentions** all 5 criteria. If any criterion is absent, ambiguous, or not explicitly an RCT, respond 'REJECT'.

Abstract to be evaluated: [INSERT ABSTRACT HERE]

Decision:
```

## Example 3: Medical Device Design Optimization

**Scenario:** Optimize the design of a new wearable sensor for continuous glucose monitoring (CGM).

**Prompt:**
```
Act as a Biomedical Engineer specializing in usability and wearable device design.

Task: Analyze the following design requirements and suggest 3 design improvements for the CGM sensor, focusing on comfort, durability, and signal accuracy.

Current Requirements:
- Material: Rigid silicone polymer.
- Size: 30mm x 15mm x 5mm.
- Application Site: Upper arm.
- Lifespan: 7 days.
- Connectivity: Bluetooth Low Energy (BLE).

Improvement Suggestions (Format: 1. [Improvement] - [Engineering Rationale]):
```

## Example 4: Data Analysis and Bioinformatics

**Scenario:** Interpret a gene expression dataset (RNA-seq) in a neurodegenerative disease model.

**Prompt:**
```
You are a Bioinformatician. You have received a list of 10 differentially expressed genes (DEGs) in neurons of Alzheimer's patients compared to controls.

DEG List (Up-regulated): APP, PSEN1, APOE, BACE1, MAPT, TREM2, CD33, PICALM, CLU, SORL1.

Task:
1. Describe the biological function of 3 critical DEGs (choose the most relevant to Alzheimer's pathogenesis).
2. Suggest a signaling pathway that connects at least 2 of these genes.
3. Propose a therapeutic target (protein or gene) based on this analysis.

Structured response:
```

## Example 5: Tissue Engineering and Biomaterials

**Scenario:** Develop a new scaffold for cartilage regeneration.

**Prompt:**
```
Act as a Biomedical Materials Engineer.

Task: Propose an extracellular matrix (ECM) scaffold for articular cartilage regeneration, considering mechanical properties and biocompatibility.

Scaffold Proposal:
1. Polymeric Material (e.g., PCL, PLA, or natural): [Choice and Rationale]
2. Structure (e.g., porous, nanofibers): [Choice and Rationale]
3. Growth Factor (e.g., TGF-β, BMP-2): [Choice and Rationale]
4. Fabrication Method (e.g., Electrospinning, 3D Printing): [Choice and Rationale]
```

## Example 6: Clinical Decision Support (Image Interpretation)

**Scenario:** Assist a radiologist in structuring a cardiac magnetic resonance imaging (MRI) report.

**Prompt:**
```
You are a clinical decision support system specializing in Medical Imaging Engineering.

Task: Based on the findings of a cardiac MRI, structure a concise report, focusing on the quantification and clinical relevance of the findings.

MRI Findings:
- Left Ventricular Ejection Fraction (LVEF): 35% (severely reduced).
- Late Gadolinium Enhancement (LGE): Subendocardial pattern in the anterior wall and septum (suggesting prior infarction).
- LV End-Diastolic Volume (LV-EDV): 220 ml (significant dilation).

Report Structure:
1. Quantification (Key Values):
2. Diagnostic Impression (Engineering): [Analysis of ventricular function and geometry]
3. Clinical Relevance: [Implications for prognosis and treatment]
```
```

## Best Practices
1. **Define Clear PICO Criteria:** Translate the PICO elements (Population, Intervention, Comparison, Outcome) into unambiguous, specific language for the LLM. Avoid vague terms and use exact keywords (e.g., "adult patients (≥18 years)" instead of "adults").
2. **Choose the Appropriate Prompting Approach:**
    *   **Zero-Shot:** For simple, straightforward criteria, without examples.
    *   **Few-Shot:** Include 2-3 labeled examples (ACCEPT/REJECT) to guide the model in ambiguous or complex cases, improving accuracy.
3. **Soft vs. Strict Strategy:**
    *   **Soft Prompt (High Sensitivity/Recall):** Instruct the model to **ACCEPT** unless a criterion is **explicitly violated**. Ideal for the first screening pass, minimizing false negatives.
    *   **Strict Prompt (High Specificity/Precision):** Instruct the model to **REJECT** unless **all** criteria are **explicitly mentioned**. Reduces false positives but may increase false negatives.
4. **Consistent Output Format:** Require a short, standardized response (e.g., only "ACCEPT" or "REJECT") to facilitate automated processing.
5. **Iterative Testing and Refinement:** Test the prompt on a small validation set and refine the instructions based on performance metrics (Precision, Recall, F1-score).
6. **Indispensable Human Oversight:** Maintain human oversight for borderline or ambiguous cases and to verify the accuracy of the LLM's decisions, mitigating hallucinations and biases.

## Use Cases
*   **Systematic Reviews and Meta-analyses:** Initial screening of thousands of abstracts to identify relevant articles based on PICO criteria.
*   **Rapid Evidence Synthesis:** Generation of summaries and extraction of key data (e.g., outcomes, sample size) from biomedical articles.
*   **Literature Classification:** Categorization of scientific articles by study type (RCT, cohort, case-control) or by specific topic (e.g., biomarkers, medical devices).
*   **Regulatory Documentation Generation:** Assistance in drafting study protocols, safety reports, and regulatory submission documents, ensuring compliance with specific guidelines.
*   **Clinical Decision Support:** Creating prompts to extract information from electronic health records (EHRs) to assist in diagnosis or treatment selection (with attention to data privacy).

## Pitfalls
*   **Overly Long or Ambiguous Prompts:** Complex or contradictory instructions that dilute the model's focus, leading to inconsistent responses.
*   **Hallucination:** The LLM infers or "invents" information (e.g., labeling a study as "randomized" when the abstract does not explicitly mention it).
*   **Data Privacy Violation:** Using cloud-based LLMs (APIs) to screen sensitive data (e.g., medical records) may violate privacy regulations (e.g., HIPAA, LGPD).
*   **Training Bias:** Biases in the LLM's training data can lead to underrepresentation or misclassification of studies from certain populations or regions.
*   **Lack of Transparency:** The "black box" nature of LLMs makes it difficult to trace the decision process, which is problematic for scientific rigor.

## URL
[https://www.mdpi.com/2673-7426/5/1/15](https://www.mdpi.com/2673-7426/5/1/15)
