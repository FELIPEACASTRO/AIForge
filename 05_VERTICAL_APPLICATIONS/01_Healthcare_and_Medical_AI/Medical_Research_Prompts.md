# Medical Research Prompts

## Description
**Medical Research Prompts** refer to the art and science of crafting instructions and text inputs optimized for **Large Language Models (LLMs)**, such as GPT-4 or Claude, with the goal of obtaining accurate, clinically relevant, and ethically sound outputs in healthcare and research contexts.

This *Prompt Engineering* technique is crucial because the effectiveness of LLMs in clinical and research settings depends directly on the quality of the input *prompt*. In a field where precision is vital (for example, diagnosis, medication dosing, evidence synthesis), poorly formulated *prompts* can lead to generic, imprecise, or even dangerous information.

The main focus of *Medical Research Prompts* is **contextual specificity and adherence to evidence-based practices** [1]. They go beyond a simple question, incorporating elements such as: **Role Definition** (Role-Playing), **Inclusion of Detailed Clinical Context**, **Evidence Source Restriction**, and **Structured Output Format** (PICO, SOAP, tables). In essence, prompt engineering in the medical field transforms the LLM from a general-purpose text generator into a highly specialized decision-support and research tool, provided that the user supplies the necessary structure and context [2].

**References:**
[1] Liu J, Liu F, Wang C, Liu S. Prompt Engineering in Clinical Practice: Tutorial for Clinicians. J Med Internet Res 2025;27(1):e72644.
[2] Meskó B. Prompt Engineering for Medical Professionals: Tutorial. J Med Internet Res 2023;25(1):e50638.

## Examples
```
**1. Clinical Decision Support (Evidence-Based)**
"**Act as an internal medicine consultant** focused on evidence-based guidelines. **Clinical Scenario:** A 68-year-old male patient with a recent diagnosis of Heart Failure with Reduced Ejection Fraction (HFrEF). The patient is on Sacubitril/Valsartan and a Beta-blocker. **Question:** What is the current recommendation (Class I) for the fourth class of medications (SGLT2i or MRA) for HFrEF, according to the **2022 AHA/ACC guidelines**? **Output Instructions:** 1. List the recommended medication. 2. Describe the mechanism of action relevant to HFrEF. 3. Cite the main clinical trial that supports this recommendation (Name and Year). 4. Present the answer in table format."

**2. Research Hypothesis Formulation (PICO Format)**
"**Act as a senior epidemiology researcher.** **Objective:** Formulate a well-defined PICO research question (Population, Intervention, Comparison, Outcome). **Topic:** The use of home telemonitoring compared to standard clinical follow-up to reduce the readmission rate in elderly patients with Chronic Obstructive Pulmonary Disease (COPD). **Output Instructions:** 1. Identify and define each PICO component. 2. Formulate the complete research question."

**3. Literature Synthesis and Systematic Review**
"**Act as a medical literature reviewer.** **Task:** Analyze the following clinical trial abstracts on the use of [Medication Name] for [Condition]. **Abstracts:** [INSERT ABSTRACT 1], [INSERT ABSTRACT 2], [INSERT ABSTRACT 3]. **Output Instructions:** 1. Identify the main efficacy and safety findings in a comparative table. 2. Determine whether the results are consistent across the studies. 3. Write a concluding paragraph on the strength of the evidence, highlighting any biases or limitations."

**4. Creation of Patient Educational Material**
"**Act as a health educator.** **Target Audience:** A 75-year-old patient with low health literacy, newly diagnosed with Atrial Fibrillation. **Objective:** Explain what Atrial Fibrillation is and why the use of anticoagulants is crucial. **Output Instructions:** 1. Use simple language, analogies, and avoid medical jargon. 2. The tone should be reassuring and encouraging. 3. The explanation should be no more than 200 words. 4. Include a 'What to do' section with 3 clear action points."

**5. Workflow Optimization (Clinical Note Generation)**
"**Act as an internal medicine resident.** **Task:** Generate a clinical note in SOAP format (Subjective, Objective, Assessment, Plan) based on the following information. **Information:** *Subjective:* The patient reports intermittent atypical chest pain for 2 days, without radiation, relieved by rest. Denies dyspnea or palpitations. *Objective:* Physical examination unremarkable. ECG: Sinus rhythm, no ST-T changes. Troponin I: Negative. *Assessment:* Atypical chest pain, likely musculoskeletal cause. Low probability of Acute Coronary Syndrome (ACS). *Plan:* Hospital discharge. Advise immediate return in case of typical pain. Prescribe an anti-inflammatory. Schedule a reassessment in 7 days. **Output Instructions:** 1. Format the note strictly in SOAP format. 2. Ensure that the 'Assessment' section justifies the low probability of ACS."

**6. Data Analysis (Interpretation of Results)**
"**Act as a biostatistician.** **Task:** Interpret the results of the following phase III study. **Key Results:** *Primary Outcome (All-cause mortality):* Intervention Group (n=500): 10% mortality. Control Group (n=500): 15% mortality. *Hazard Ratio (HR):* 0.65 (95% CI: 0.45-0.94). p-value: 0.02. **Output Instructions:** 1. Explain the meaning of the Hazard Ratio (HR) of 0.65. 2. Comment on the statistical significance (p-value) and precision (95% CI). 3. Write a concise clinical conclusion on the efficacy of the intervention."
```

## Best Practices
**1. Explicitness and Specificity:** Prompts should be clear, precise, and concise. Incorporate patient-specific clinical variables (age, disease stage, comorbidities) and applicable guidelines. **2. Contextual Relevance:** Provide the complete clinical context, including the user's role (e.g., "You are a physician") and crucial patient information (history, medications). **3. Iterative Refinement:** Start with a simple prompt and progressively refine it based on the LLM's outputs, adding constraints and specifications. **4. Evidence-Based Practices:** Direct the LLM to use reliable and up-to-date sources of information, specifying the source (e.g., "Based on the 2023 AHA/ACC guidelines"). **5. Ethical Considerations:** Avoid including personally identifiable information (PII) and request neutral, evidence-based answers, acknowledging limitations.

## Use Cases
**Generation of High-Quality Summaries:** Summarize medical records, research articles, or complex guidelines. **Clinical Decision Support:** Assist with differential diagnosis, treatment selection, and management of rare diseases. **Patient Education:** Generate personalized and easy-to-understand educational materials. **Workflow Optimization:** Create checklists, clinical note templates (SOAP), and referral letters. **Research and Literature Review:** Identify relevant articles, synthesize clinical trial findings, and formulate research hypotheses. **Research Hypothesis Formulation:** Generate PICO research questions (Population, Intervention, Comparison, Outcome) from a clinical scenario. **Data Analysis (Simulated):** Ask the LLM to interpret test results or clinical trial data (with fictitious or anonymized data).

## Pitfalls
**Vagueness and Generality:** Prompts that do not specify the clinical objective, patient profile, or context. **Ignoring Context:** Failure to provide crucial patient information, leading to generic or inadequate recommendations. **Over-reliance:** Accepting the LLM's output without clinical verification or critique. **Privacy Violation:** Including personally identifiable information (PII) in prompts. **Lack of Iteration:** Not refining the prompt after an unsatisfactory output. **Absence of Evidence References:** Not asking the LLM to base its answer on specific guidelines or literature.

## URL
[https://www.jmir.org/2025/1/e72644](https://www.jmir.org/2025/1/e72644)
