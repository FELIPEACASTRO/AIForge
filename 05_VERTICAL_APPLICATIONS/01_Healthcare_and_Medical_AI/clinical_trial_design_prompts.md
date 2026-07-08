# Clinical Trial Design Prompts

## Description
**Clinical Trial Design Prompts** are specialized and structured instructions, frequently using Large Language Models (LLMs), to assist in the creation, optimization, and analysis of clinical research protocols. This Prompt Engineering technique is applied in the healthcare and life sciences domain to automate complex tasks such as defining eligibility criteria, calculating sample size, selecting endpoints, and identifying ethical and operational risks [1] [3].

The main objective is to increase efficiency, reduce protocol development time, and improve the scientific quality and regulatory compliance of clinical trials. By providing detailed clinical and statistical context, the prompts guide the AI to generate protocol sections that are coherent, evidence-based, and aligned with international best practices [2]. Recent research (2023-2025) demonstrates a growing focus on the use of LLMs for extracting PICO elements, optimizing eligibility criteria, and generating simplified versions of informed consent [1].

## Examples
```
**1. Optimizing Eligibility Criteria (Few-Shot Prompting)**

```
**Role:** You are a patient recruitment specialist for Phase II Oncology trials.
**Context:** We are designing a trial for a new PD-1 inhibitor (Intervention) in patients with Metastatic Melanoma (Population).
**Instruction:** Analyze the following inclusion/exclusion criteria (Current Criteria) and suggest modifications to optimize recruitment by 20% while maintaining scientific validity. Justify each change based on recruitment data from PD-1 trials published in the last 3 years.
**Current Criteria:** [List of current criteria]
**Output Format:** Table with Columns: Original Criterion, Suggested Modification, Justification/Reference.
```

**2. Generating the Primary and Secondary Endpoints Section**

```
**Role:** Senior biostatistician.
**Context:** Phase III, randomized, double-blind, placebo-controlled trial for a drug aimed at reducing the progression of mild to moderate Alzheimer's Disease.
**Instruction:** Propose the most appropriate primary and secondary endpoints for this trial. For the primary endpoint, specify the assessment metric (e.g., CDR-SB, ADAS-Cog), the time point (e.g., 52 weeks), and the statistical analysis method (e.g., ANCOVA).
**Suggested Primary Endpoints:**
**Suggested Secondary Endpoints:**
**Requirement:** The response must comply with FDA guidelines for Alzheimer's trials.
```

**3. Sample Size Calculation (Chain-of-Thought)**

```
**Role:** Clinical trial statistician.
**Context:** Non-inferiority trial comparing a new oral antibiotic with the gold standard for urinary tract infections.
**Instruction:** Calculate the required sample size. Use the following process (Chain-of-Thought):
1. Define the clinically acceptable non-inferiority margin (delta) (justify).
2. Estimate the success rate of the standard treatment (gold standard) based on literature (cite the source).
3. Define the statistical power (e.g., 80%) and the significance level (alpha = 0.05).
4. Present the formula used and the final sample size (N) calculation per group.
5. Include a 15% dropout rate and recalculate the final N.
```

**4. Risk Analysis and Operational Mitigation**

```
**Role:** Clinical Project Manager (CPM).
**Context:** Clinical trial protocol for advanced cell therapy requiring the collection and processing of complex samples across 10 centers in Europe.
**Instruction:** Identify the 5 main operational risks (e.g., sample logistics, staff training, local regulatory compliance) and propose a detailed mitigation strategy for each.
**Output Format:** Risk Table (Risk, Probability (High/Medium/Low), Impact (High/Medium/Low), Mitigation Strategy).
```

**5. Generating a Simplified Version of Informed Consent**

```
**Role:** Patient communication specialist.
**Context:** The following text is the "Risks and Side Effects" section of an Informed Consent Form (ICF) for a vaccine trial.
**Instruction:** Rewrite this section in 8th-grade language (simplified reading level), maintaining medical and legal accuracy. Use simple analogies to explain complex terms such as "serious adverse events" and "randomization".
**Original Text:** [Insert complex ICF text]
**Requirement:** The simplified version must have a Flesch-Kincaid index of 60 or higher.
```

**6. Extracting PICO Elements from a Scientific Article**

```
**Role:** Evidence Analyst.
**Context:** [Insert abstract or Methods section of a clinical trial article]
**Instruction:** Extract and structure the PICO elements (Population, Intervention, Comparison, Outcome) from this text.
**Output Format:**
- Population (P): [Details]
- Intervention (I): [Details]
- Comparison (C): [Details]
- Outcome (O): [Details]
```
```

## Best Practices
**1. Clinical and Role Contextualization (Role-Playing):** Begin the prompt by defining the AI's role (e.g., "You are a senior biostatistician specialized in Phase III trials") and the clinical context (e.g., "Designing a trial for a new SGLT2 inhibitor for heart failure").
**2. PICO/PICOTS Structure:** Use research frameworks (Population, Intervention, Comparison, Outcome, Time, Setting) to ensure that all critical elements of the protocol are addressed.
**3. Specificity and Clarity:** Use precise medical and statistical language. Avoid vague terms. For example, instead of "improve recruitment", use "Optimize inclusion/exclusion criteria to increase the recruitment rate by 15% without compromising internal validity".
**4. Iteration and Refinement (Few-Shot/Chain-of-Thought):** Use multi-step prompts. Ask the AI to first outline the structure (Chain-of-Thought) and then fill in the details. Use examples of successful protocols (Few-Shot) to guide the response.
**5. Validation and Reference:** Ask the AI to cite regulatory guidelines (e.g., ICH-GCP, FDA) or reference articles for the proposed design decisions, enabling human validation.
**6. Structured Output Generation:** Request the output in a structured format (e.g., Markdown, JSON, table) to facilitate integration into protocol documents [1] [2].

## Use Cases
**1. Protocol Optimization:** Generating drafts of protocol sections (e.g., Endpoints, Eligibility Criteria, Statistical Analysis Plan) to accelerate the study *design* phase.
**2. Feasibility Analysis:** Rapid assessment of the complexity and recruitment potential of a protocol by analyzing the literature and data from previous trials.
**3. Regulatory Compliance:** Generating compliance checklists for ICH-GCP, FDA, or EMA guidelines, ensuring the protocol addresses all legal and ethical requirements.
**4. Patient Communication:** Simplifying complex documents, such as the Informed Consent Form (ICF), to improve participant understanding and engagement [1].
**5. Evidence Synthesis:** Structured data extraction (PICO) from scientific articles to support the scientific rationale of the new trial [1].
**6. Training and Education:** Creating clinical trial scenarios for training new researchers and study coordinators.

## Pitfalls
**1. Hallucinations and Factual Inconsistency:** AI can generate information that seems plausible but is clinically or regulatorily incorrect (hallucinations). This is critical in healthcare, where accuracy is vital. **Mitigation:** Always validate the output against official guidelines (ICH-GCP, FDA, EMA) and review by human experts [3].
**2. Bias and Lack of Diversity:** AI can perpetuate biases present in training data, resulting in eligibility criteria that unduly exclude minority or underrepresented populations. **Mitigation:** Include an explicit instruction in the prompt to consider diversity and equity (e.g., "Ensure that eligibility criteria do not unnecessarily exclude ethnic or gender minorities").
**3. Failure to Capture Clinical Nuances:** LLMs may struggle to integrate complex nuances of rare diseases or specific drug interactions. **Mitigation:** Provide extremely detailed clinical context and use the *Retrieval-Augmented Generation (RAG)* technique, feeding the AI with trial-specific reference documents [2].
**4. Overdependence:** Blind reliance on the AI's output, without proper human review and *due diligence*, can lead to serious protocol errors that compromise patient safety and study validity. **Mitigation:** Treat the AI's output as an advanced draft or a suggestion, not as a final document [3].
**5. Format Inconsistency:** AI may fail to adhere to strict regulatory formats (e.g., *Clinical Study Protocol* - CSP). **Mitigation:** Use rigorous formatting prompts (e.g., "Structure the output in the format of section 3.1 of ICH E6(R2)") [1].

## URL
[https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-025-04348-9](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-025-04348-9)
