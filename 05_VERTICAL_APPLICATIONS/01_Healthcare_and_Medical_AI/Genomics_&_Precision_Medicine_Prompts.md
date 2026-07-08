# Genomics & Precision Medicine Prompts

## Description
**Genomics and Precision Medicine Prompts** refer to specialized, structured instructions designed to interact with Large Language Models (LLMs) and multimodal AI models in the context of analyzing genomic, transcriptomic, proteomic, and multi-omics data, with the goal of advancing personalized medicine. This prompt engineering technique is crucial for tasks that require a high degree of precision and domain knowledge, such as interpreting genetic variants, designing gene-editing experiments (e.g., CRISPR), predicting disease risk, recommending personalized therapies, and communicating complex genetic results to patients. The effectiveness of these prompts depends on the ability to integrate complex biological data, scientific literature, and clinical guidelines, often using advanced techniques such as Retrieval-Augmented Generation (RAG) and specialized role-playing assigned to the model. The focus is on transforming raw sequencing data into actionable clinical insights and automated research protocols.

## Examples
```
**1. Genetic Variant Interpretation (ACMG/AMP):**
```
You are a bioinformatician specializing in genetic variant classification. Analyze the following variant in the BRCA1 gene: c.5266dupC (p.Gln1756Profs*10).
1. Classify the variant according to the ACMG/AMP guidelines (e.g., Pathogenic, Likely Pathogenic, VUS, etc.).
2. Justify the classification by citing the relevant PVS1, PS1-4, PM1-6, PP1-5, BA1, BS1-4, BP1-6 evidence.
3. Describe the associated clinical phenotype and the implications for cancer screening.
```

**2. Automated CRISPR Experiment Design:**
```
Assume the role of a genome engineer using the CRISPR-Cas12a system.
Objective: Knockout of the TGFβR1 gene in A549 lung cancer cells.
Task: Generate a complete protocol.
1. Suggest 3 high-efficacy gRNA sequences with low off-target activity, citing the prediction tool used.
2. Recommend the most suitable delivery method (transfection/transduction) for the A549 cell line.
3. Describe the validation protocol (e.g., Western Blot, sequencing) and the required PCR primers.
```

**3. Drug Response Prediction (Pharmacogenomics):**
```
Patient: 65 years old, diagnosed with colon cancer.
Genotype: *2/*3 variant in the CYP2D6 gene.
Medication: Irinotecan (metabolized by UGT1A1).
Prompt: Based on the CYP2D6 genotype and the drug Irinotecan, assess the risk of toxicity and efficacy.
1. What is the patient's metabolizer status for CYP2D6?
2. Is Irinotecan affected by this variant? If not, which gene is the main determinant of toxicity (UGT1A1)?
3. Provide a dose adjustment recommendation or an alternative drug, citing the CPIC guidelines.
```

**4. Clinical Report Generation for the Patient:**
```
You are a genetic counselor.
Result: Pathogenic mutation in the MLH1 gene, confirming Lynch Syndrome.
Audience: Lay patient, high anxiety level.
Instruction: Create a 3-paragraph text that explains the result in a clear, empathetic, and non-alarming way.
1. What Lynch Syndrome and the MLH1 gene are.
2. What the associated cancer risks are.
3. What the next steps and screening options are (e.g., annual colonoscopy).
```

**5. Multi-Omics Data Analysis:**
```
Input data: List of 50 differentially expressed genes (RNA-Seq) and 10 somatic variants (WES) in a tumor.
Prompt: Integrate the expression and variant data to identify the most impacted signaling pathways.
1. List the top 3 signaling pathways (e.g., KEGG, Reactome) that contain both differentially expressed genes and genes with variants.
2. Suggest a therapeutic target (gene/protein) located at the intersection of these pathways.
3. Explain the biological rationale for your suggestion.
```
```

## Best Practices
**1. Provide Detailed Genomic Context:** Include as much raw genomic data (sequences, VCF variants, expression data) or structured summaries (genes, mutations, phenotypes) directly in the prompt as possible. The more specific the biological context, the more accurate the response will be. **2. Define the Role (Role-Playing) and the Task:** Begin the prompt by assigning a specialized role to the model (e.g., "You are a bioinformatician specializing in oncogenomics" or "You are a genetic counselor"). Then clearly define the task (e.g., "Analyze the pathogenicity of this variant" or "Generate a gene-editing protocol"). **3. Use RAG (Retrieval-Augmented Generation):** For critical tasks, such as genetic counseling or experimental design, integrate the RAG technique. This involves providing the LLM with reference documents (clinical guidelines, scientific articles, laboratory protocols) so that it bases its responses on verified information. **4. Specify the Output Format:** Request the output in a structured format, such as JSON, a Markdown table, or a specific clinical report format, to facilitate analysis and integration with other systems. **5. Iteration and Refinement:** Genomics is complex. Use the output of the first prompt as input for a second prompt, refining the question or requesting cross-validation. For example, request the analysis of a variant and then request a review of that analysis based on a new article.

## Use Cases
**1. Diagnosis and Interpretation of Rare Diseases:** Rapid analysis of sequencing panels to identify pathogenic variants and suggest differential diagnoses. **2. Automated Genetic Counseling:** Generation of result summaries and answers to frequently asked questions for patients, freeing up genetic counselors' time for more complex cases. **3. Laboratory Protocol Optimization:** Automated design of primers, probes, and CRISPR guides, reducing experimental planning time (as demonstrated by CRISPR-GPT [1]). **4. Pharmacogenomics and Dose Selection:** Prediction of individual drug response based on the patient's genetic profile, minimizing adverse effects and optimizing efficacy. **5. Therapeutic Target Discovery:** Integration of multi-omics data (genomics, transcriptomics, proteomics) to identify new genes or signaling pathways for drug development. **6. Polygenic Risk Score (PRS) Analysis:** Interpretation of polygenic risk scores for common diseases (e.g., diabetes, heart disease) and translation into lifestyle or screening recommendations.

## Pitfalls
**1. Hallucinations and Clinical Inaccuracy:** The greatest risk is the LLM generating clinically incorrect information or hallucinating references and genetic data. **Mitigation:** Always use RAG with verified sources (e.g., ClinVar, HGMD, CPIC guidelines) and require citations of articles or databases. **2. Lack of Biological Context:** Prompts that are too short or generic (e.g., "What does this gene do?") fail to provide the necessary context (e.g., cell type, tissue, disease condition). **Mitigation:** Be hyper-specific about the biological system and the type of input data. **3. Bias and Inequality:** AI models are predominantly trained on data from populations of European ancestry. This can lead to inaccurate or biased interpretations for variants in underrepresented populations. **Mitigation:** Include the patient's ethnicity or ancestry in the prompt, if relevant, and ask the model to consider the limitations of the reference data. **4. Exposure of Sensitive Data (PHI):** Entering patient data (PHI - Protected Health Information) directly into the prompt violates privacy regulations (e.g., HIPAA, LGPD). **Mitigation:** Always fully anonymize the input data, using variant IDs, sequences, or phenotype summaries, and never names, dates of birth, or medical record numbers. **5. Confusion Between Causation and Correlation:** The LLM may confuse a statistical association (correlation) with a biological causal relationship. **Mitigation:** Ask the model to clearly distinguish between association evidence and proven biological mechanisms.

## URL
[https://www.nature.com/articles/s41551-025-01463-z](https://www.nature.com/articles/s41551-025-01463-z)
