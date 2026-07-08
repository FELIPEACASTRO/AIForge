# Pathology Analysis Prompts

## Description
**Pathology Analysis Prompts** refer to the application of **Prompt Engineering** techniques to interact with Large Language Models (LLMs) and multimodal models (VLM/LLM) with the goal of analyzing, structuring, and interpreting complex pathology data. The main focus is the **extraction of structured information** from free-text pathology reports (histopathology, cytopathology, surgical reports) and the **analysis of images** of whole slide images (Whole Slide Images - WSIs) in digital pathology [1] [2].

This technique is crucial because most pathology reports are generated in free-text format, which makes large-scale analysis, research, and integration into AI systems difficult. Prompts are designed to guide the LLM to accurately identify and extract critical clinical data, such as tumor type, histological grade, receptor status (ER, PR, HER2), surgical margins, and TNM staging, converting the unstructured text into a consumable data format (usually JSON or table) [3] [4].

Recent research (2023-2025) demonstrates that open-source LLMs, when well prompted, can match the accuracy of proprietary models such as GPT-4 in extracting pathology data, which has significant implications for privacy and cost in healthcare settings [1]. Adaptive prompt engineering and output format specification are key elements for success [3].

## Examples
```
**1. Structured Extraction from a Breast Cancer Report (Zero-Shot)**

```
You are a pathology data analyst. Your task is to extract the following information from the provided pathology report and format it strictly as a JSON object.

Fields to extract:
1.  `tumor_type`: (e.g., Invasive Ductal Carcinoma, Invasive Lobular Carcinoma)
2.  `histological_grade`: (e.g., Grade 1, Grade 2, Grade 3)
3.  `er_status`: (e.g., Positive, Negative, Indeterminate)
4.  `pr_status`: (e.g., Positive, Negative, Indeterminate)
5.  `her2_status`: (e.g., Positive, Negative, Equivocal)
6.  `surgical_margins`: (e.g., Free, Involved, Not evaluated)

Pathology Report:
[INSERT FULL REPORT TEXT HERE]

JSON Output:
```

**2. TNM Staging Analysis (Chain-of-Thought)**

```
Act as an oncologic pathologist. Analyze the following pathology report and determine the TNM staging (Tumor, Node, Metastasis) based on the most recent AJCC guidelines.

Reasoning Steps (CoT):
1.  Identify the primary tumor size (T).
2.  Identify the number of positive lymph nodes and the node status (N).
3.  Identify the presence or absence of distant metastasis (M).
4.  Combine the findings to determine the final TNM staging.

Report: [INSERT REPORT]

TNM Staging: [FINAL ANSWER]
Reasoning: [STEP-BY-STEP JUSTIFICATION]
```

**3. Classification of Digital Pathology Findings (Multimodal)**

```
[SYSTEM PROMPT: You are receiving a WSI (Whole Slide Image) of a renal biopsy and the text of the report.]

Instruction: Compare the WSI image with the report text. Identify whether there is a discrepancy between the morphological finding in the image (e.g., segmental glomerulosclerosis) and the textual diagnosis.

Primary Finding in the Image: [DESCRIBE THE VISUALLY DOMINANT FINDING]
Textual Diagnosis: [INSERT REPORT TEXT]

Question: Is the textual diagnosis complete and consistent with the primary finding in the image? If not, what is the discrepancy?

Answer:
```

**4. Extraction of Dimensions and Focality (Specific Prompt)**

```
Extract the tumor dimensions and focality from the report. Respond strictly with the fields 'max_dimension_mm' and 'focality'. If the information is not present, use 'not specified'.

Report: "The specimen measures 5.0 x 3.0 x 2.0 cm. There is an invasive focus measuring 12 mm on the greatest axis. No other foci are identified."

Output JSON:
{
  "max_dimension_mm": [VALUE],
  "focality": [Single focus/Multiple foci/not specified]
}
```

**5. Generation of a Clinical Summary for the Electronic Health Record**

```
You are a clinical documentation assistant. Create a concise summary (maximum 3 sentences) of the following pathology report, highlighting only the final diagnosis, the grade, and the receptor status.

Report: [INSERT REPORT]

Clinical Summary:
```
```

## Best Practices
**1. Output Format Specification (JSON/Table):** Always instruct the LLM to return the information in a structured format (JSON, table, YAML) to facilitate processing and integration into laboratory information systems (LIS) or research databases [1] [3].

**2. Zero-Shot and Few-Shot Prompting:** For data extraction tasks, start with **Zero-Shot Prompting** (instructions only) and, if accuracy is insufficient, use **Few-Shot Prompting** (instructions + 1 to 5 examples of reports and their correct extractions) to refine performance [2] [4].

**3. Chain-of-Thought (CoT):** For complex diagnoses or the analysis of multiple findings, ask the LLM to "think out loud" or justify its conclusion before providing the final answer. This improves traceability and accuracy [5].

**4. Clear Definition of Extraction Fields:** Explicitly define the data fields to be extracted (e.g., Tumor Type, Histological Grade, Receptor Status, Surgical Margins) and the allowed values (e.g., "Positive/Negative", "Grade 1/2/3") [3].

**5. Iteration and Adaptive Refinement:** The prompt engineering process should be iterative. Adjust the prompt dynamically based on performance feedback, especially to handle the idiosyncratic terminology of different pathologists and institutions [3].

**6. Contextualization and Role:** Begin the prompt by defining the LLM's role (e.g., "You are an experienced oncologic pathology analyst") and the task context (e.g., "Your task is to extract critical data from a breast cancer pathology report") [3].

## Use Cases
**1. Automated Data Extraction for Research:** Convert thousands of free-text pathology reports into a structured database for epidemiological studies, clinical trials, and AI model development [1].

**2. Populating Electronic Health Records (EHR):** Automate the entry of critical data (e.g., staging, grade, biomarker status) from pathology reports directly into the patient's electronic health record, reducing manual error and accelerating the clinical workflow [3].

**3. Development of AI Models in Digital Pathology:** Use prompts to generate high-quality labels from textual reports, which are then used as "ground truth" to train Computer Vision models that analyze whole slide images (WSIs) [2].

**4. Generation of Clinical Summaries:** Create standardized and concise summaries of complex reports to facilitate communication between pathologists, oncologists, and other clinical specialists [5].

**5. Quality Control and Auditing:** Use prompts to verify the consistency and completeness of reports, ensuring that all mandatory fields (e.g., according to ICCR protocols) have been addressed [3].

## Pitfalls
**1. Hallucinations and Clinical Inaccuracy:** The LLM may generate clinically incorrect information or "hallucinate" data that is not in the report, especially in ambiguous texts or those with typos. Human validation is indispensable [1].

**2. Failure to Extract Missing Fields:** If a data field (e.g., the status of a specific receptor) is not explicitly mentioned in the report, the LLM may fail to return the value "Not specified" and instead try to infer it or leave the field empty, breaking the structured format [3].

**3. Sensitivity to Language Variation:** Pathology terminology varies significantly between different pathologists and institutions (e.g., abbreviations, jargon). Non-adaptive prompts may perform poorly on reports with an idiosyncratic writing style [3].

**4. Output Format Breakage:** The instruction to return a structured format (JSON, table) may be ignored by the LLM if the prompt is too long or if the extraction complexity is high, resulting in free text that invalidates automation [4].

**5. Data Bias:** If the base model was trained predominantly on reports from a single ethnicity, cancer type, or language (e.g., English), it may perform worse when analyzing reports from different populations or languages (e.g., Portuguese) [1].

## URL
[https://www.nature.com/articles/s43856-025-00808-8](https://www.nature.com/articles/s43856-025-00808-8)
