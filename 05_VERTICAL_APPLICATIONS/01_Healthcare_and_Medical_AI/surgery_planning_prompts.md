# Surgery Planning Prompts

## Description
Surgery Planning Prompts are highly structured, guideline-driven instructions designed for use with Large Language Models (LLMs) in clinical settings. The main objective is to provide the LLM with detailed clinical data of a patient (demographics, comorbidities, imaging findings, tumor biology, staging) and request a structured, auditable output that assists in surgical decision-making. This technique aims to standardize the planning process, improve patient safety, and optimize surgical outcomes, often incorporating the RAG (Retrieval-Augmented Generation) technique to ensure that the LLM's suggestions are based on the most recent and field-specific clinical guidelines.

## Examples
```
1. **Prompt for Neoadjuvant Decision in Breast Cancer**
```
**Role:** You are an AI assistant for the Breast Cancer Multidisciplinary Team, specialized in surgical oncology.
**Context:** Female patient, 55 years old. Invasive ductal carcinoma, grade 3. Clinical staging T2N1M0. Estrogen receptor (ER) positive (90%), Progesterone receptor (PR) positive (80%), HER2 negative (FISH not amplified). Ki-67: 45%. No significant comorbidities. Ultrasound shows a 3.5 cm tumor and a suspicious axillary lymph node.
**Instruction:** Based on the most recent NCCN/ESMO guidelines, assess the appropriateness of neoadjuvant therapy. Provide the response in JSON format with the fields: `Neoadjuvant_Recommendation` (Yes/No), `Main_Justification`, `Suggested_Regimen`.
```

2. **Prompt for Post-Mastectomy Reconstruction Type**
```
**Role:** You are an AI-assisted breast reconstruction specialist.
**Context:** 48-year-old patient, underwent total mastectomy. History of smoking (quit 6 months ago). BMI: 32. Prior irradiation: No. Patient preference: Immediate reconstruction. Abdominal tissue availability: Adequate. Vascular condition: Good.
**Instruction:** Suggest the type of breast reconstruction (Direct Implant, Expander + Implant, TRAM, DIEP, Latissimus Dorsi) and assess the feasibility of a Direct Implant. Provide the output in a Markdown table with the columns: `Recommended_Option`, `Direct_Implant_Feasibility` (High/Medium/Low), `Risk_Factors`.
```

3. **Prompt for Surgical Action Plan in Colorectal Surgery**
```
**Role:** You are a surgical planning assistant for colorectal surgery.
**Context:** Male patient, 68 years old. Mid rectal cancer (10 cm from the anal verge). Staging: cT3N1M0. Neoadjuvant treatment: Complete chemoradiotherapy (CRT). Response: Complete clinical response (cCR) after CRT. Comorbidities: Controlled hypertension.
**Instruction:** Develop an initial surgical plan. Include: `Suggested_Procedure` (e.g., Low Anterior Resection, Abdominoperineal), `Approach` (Laparoscopic/Robotic/Open), `Stoma_Need` (Temporary/Permanent/No), and `Critical_Intraoperative_Points`.
```

4. **Prompt for Preoperative Risk Assessment**
```
**Role:** You are a preoperative risk assessment specialist (ASA/POSSUM).
**Context:** 75-year-old patient. Proposed surgery: Elective abdominal aortic aneurysm repair. History: Myocardial infarction 3 years ago, Congestive Heart Failure (NYHA Class II), Type 2 Diabetes Mellitus (HbA1c 8.5). Tests: Creatinine 1.8 mg/dL, Electrocardiogram with Atrial Fibrillation.
**Instruction:** Calculate the ASA and POSSUM risk scores (physiological and operative). List the 3 main perioperative risks and suggest a specific preoperative optimization for each risk.
```

5. **Prompt for Structured Surgical Documentation**
```
**Role:** You are a structured surgical report generator.
**Context:** [Paste the full text of the surgical dictation or intraoperative notes].
**Instruction:** Convert the raw text into a structured surgical report in XML or JSON format, with the following mandatory fields: `Surgery_Date`, `Lead_Surgeon`, `Assistants`, `Pre_Diagnosis`, `Procedure_Performed`, `Intraoperative_Findings`, `Complications` (Yes/No), `Blood_Loss` (mL), `Drains` (Type and Location).
```
```

## Best Practices
**1. Rigid Structure (Role, Context, Instruction, Format):** Always define the **Role** (e.g., surgical oncologist, interventional radiologist), provide the complete clinical **Context** (patient data), define a clear **Instruction** (the surgical question), and require a structured output **Format** (JSON, Markdown Table).
**2. Incorporating Guidelines (RAG):** For clinical accuracy, the LLM should be augmented with the most recent guidelines (e.g., NCCN, NICE, ESMO). The prompt should implicitly or explicitly reference the need for adherence to these sources.
**3. Specificity and Quantification:** Use precise clinical terms and quantifiable data (e.g., tumor size in cm, Ki-67 in %, ASA score). Avoid ambiguity.
**4. Validation and Auditing:** Design the prompt so that the output is easily auditable and comparable with the gold standard (the multidisciplinary team's decision), facilitating model validation.
**5. Focus on Safety:** Always include a 'Risk Factors' or 'Critical Points' section in the requested output to ensure the LLM considers patient safety.

## Use Cases
1. **Preoperative Decision Support:** Helping less experienced surgeons or complex cases align the surgical plan with current guidelines.
2. **Planning Standardization:** Ensuring all cases are evaluated with the same criteria, reducing variability in clinical practice.
3. **Education and Training:** Using the LLM to generate treatment plans for simulated cases, serving as a learning tool for residents.
4. **Documentation Generation:** Converting voice notes or free text into structured surgical reports ready for the electronic medical record.
5. **Risk Assessment:** Calculating risk scores (e.g., ASA, POSSUM) and identifying the main comorbidities that require preoperative optimization.

## Pitfalls
1. **Clinical Hallucinations:** The LLM can generate recommendations that are plausible but clinically incorrect or unsupported by evidence. **Mitigation:** Mandatory use of RAG and human validation.
2. **Data Bias:** If the model is trained on data from a single institution or demographic, it may suggest suboptimal plans for different populations. **Mitigation:** Equity and diversity auditing of the training data.
3. **Lack of Visual Context:** Surgical planning is inherently visual (images, anatomy). The LLM, on its own, cannot interpret images. **Mitigation:** The prompt should include structured data extracted from images by a radiologist or another AI model.
4. **Overdependence:** Blindly trusting the LLM's output without human clinical judgment. **Mitigation:** The LLM should be positioned as an 'assistant', not a 'decision-maker'.
5. **Format Inconsistency:** If the prompt is not rigid enough, the LLM may return the plan in free-text format, hindering integration with electronic medical record systems.

## URL
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12588214/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12588214/)
