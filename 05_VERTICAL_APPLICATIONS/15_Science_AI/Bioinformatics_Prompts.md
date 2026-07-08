# Bioinformatics Prompts

## Description
Bioinformatics Prompts refers to the practice of prompt engineering applied to the field of bioinformatics and life sciences. It involves creating, refining, and implementing detailed instructions to guide Large Language Models (LLMs) in performing complex tasks of biological data analysis and interpretation. Given the high-dimensional and heterogeneous nature of the data (genomics, transcriptomics, proteomics), prompt precision is crucial to mitigate 'hallucinations' and ensure scientifically valid results. An effective bioinformatics prompt generally follows a structure that includes: **Context** (defining the LLM's role, e.g., 'You are a biostatistician'), **Data** (providing sequences, structures, or datasets for processing), **Task** (the specific instruction, e.g., 'Predict the secondary structure of the RNA'), and **Format** (specifying the desired output, e.g., 'Respond in JSON format with the confidence score').

## Examples
```
1. **DNA Sequence Analysis:**\n```\nContext: You are a genomics expert. The provided DNA sequence is from a gene of interest for antibiotic resistance.\nData: ATGGCCATAGCTTGACTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGL.
2. **Protein Structure Prediction:**
```
Context: You are a protein modeler with access to an LLM trained on protein sequences and structures (e.g., AlphaFold).
Data: Amino acid sequence of Protein X (e.g., 'MGKVKV...').
Task: Predict the secondary and tertiary structure of Protein X. List the identified functional domains.
Format: Return the secondary structure in modified FASTA format and the list of domains in JSON format.
```
3. **Gene Expression Analysis:**
```
Context: You are a biostatistician analyzing RNA-seq data from a cancer experiment.
Data: Gene counts table (CSV) for 50 samples (25 control, 25 treatment).
Task: Identify the 10 most differentially expressed genes (p-value < 0.01 and |log2FC| > 2) between the control and treatment groups.
Format: Return a Markdown table with GeneID, log2FC, p-value, and a brief description of the gene's function (citing UniProt).
```
4. **Code Generation for Analysis:**
```
Context: You are a junior bioinformatics programmer.
Task: Write a Python script using the Biopython library to perform multiple sequence alignment (MSA) using the ClustalW algorithm for the provided sequences.
Data: FASTA sequences (e.g., >Seq1\\nATGC..., >Seq2\\nATGC...).
Format: Return only the complete, functional Python code.
```
5. **Scientific Literature Review:**
```
Context: You are a literature reviewer for an article on new therapeutic targets for Alzheimer's Disease.
Task: Summarize the most recent findings (last 2 years) on the role of hyperphosphorylated Tau protein as a drug target, focusing on Phase II or III clinical trials.
Format: Return a concise summary (maximum 300 words) and provide 3 URLs of primary research articles (PubMed or arXiv).
```
6. **Molecular Docking Simulation:**
```
Context: You are a medicinal chemist.
Task: Describe the molecular docking procedure for ligand 'XYZ' in the active site of protein 'PDB ID: 1A2C'. Mention the software tools (e.g., AutoDock Vina) and the key parameters that would be used.
Format: Return a step-by-step list of the protocol.
```
7. **Genetic Variant Interpretation:**
```
Context: You are a clinical geneticist.
Data: Variant: c.1521_1523delCTT in the CFTR gene.
Task: Describe the pathogenicity classification (ACMG/AMP) and the disease associated with this variant.
Format: Return a concise paragraph and the classification in bold.
```
8. **PCR Primer Design:**
```
Context: You are a molecular biologist.
Data: Target sequence: [500bp sequence].
Task: Design a pair of PCR primers (Forward and Reverse) with Tm between 58-62°C and a GC content of 40-60%.
Format: Return the primers in list format (Primer F: [Sequence], Primer R: [Sequence]).
```
```

## Best Practices
**1. Role and Context Definition (Role-Playing):** Always start by defining the LLM's role (e.g., 'You are a biostatistician', 'You are a genomics expert') to guide the tone, knowledge, and response style. **2. Clear Structure (Context, Data, Task, Format):** Use the C-D-T-F structure (Context, Data, Task, Format) to ensure the LLM has all the necessary information. The **Format** is crucial for bioinformatics, requiring structured outputs (JSON, CSV, Python code). **3. Providing Explicit Data:** Include the sequences (DNA, RNA, Protein) or input data directly in the prompt, or instruct the LLM to simulate the use of a specific data file. **4. Grounding Instructions:** Ask the LLM to 'ground' its responses in known biological databases (e.g., UniProt, PDB, BLAST) or to cite relevant scientific articles, even if the LLM cannot access them in real time. This helps mitigate 'hallucinations'. **5. Chain-of-Thought (CoT) Prompting for Complex Reasoning:** For tasks that require multiple steps (e.g., metabolic pathway analysis), use CoT (Chain-of-Thought) by asking the LLM to 'think step by step' before providing the final answer. **6. Bioinformatics Specificity:** Use the correct technical terminology (e.g., 'sequence alignment', 'molecular docking', 'secondary structure prediction') to avoid ambiguities.

## Use Cases
**1. Sequence Analysis:** Gene prediction, motif identification, multiple sequence alignment (MSA), and analysis of genetic variations (SNPs). **2. Structure Prediction:** Prediction of secondary and tertiary structures of proteins and RNA, and identification of functional domains. **3. Code Generation:** Creation of Python scripts (using Biopython, Pandas, etc.) to automate bioinformatics tasks, such as FASTA file processing, gene expression data analysis (RNA-seq), and visualization. **4. Drug Discovery and Medicinal Chemistry:** Molecular docking simulation, ligand property prediction, and identification of new therapeutic targets. **5. Literature Review and Synthesis:** Summarizing scientific articles, extracting clinical trial data, and generating research hypotheses based on vast biological knowledge. **6. Education and Training:** Creation of interactive tutorials and complex problem solving for bioinformatics and biology students.

## Pitfalls
**1. Over-Reliance on 'Hallucinations':** The LLM may generate factually incorrect sequences, structures, or interpretations. **Pitfall:** Accepting the output without cross-verification against biological databases or established bioinformatics tools. **2. Terminology Ambiguity:** The use of imprecise biological or bioinformatics terms can lead to irrelevant or incorrect results. **Pitfall:** Failing to specify the data type (e.g., genomic DNA vs. cDNA) or the desired algorithm (e.g., local vs. global alignment). **3. Context Limitations:** LLMs have token limits, which prevent the analysis of very long genomic or protein sequences. **Pitfall:** Attempting to process sequences with more than 10,000 bases/residues in a single prompt without instructing the LLM to use a chunked processing approach. **4. Lack of Output Structure:** Failing to specify a structured output format (JSON, table) results in free text that is difficult to parse or integrate into bioinformatics pipelines. **Pitfall:** Asking only for 'the answer' instead of 'the answer in JSON format with fields X, Y, and Z'. **5. Ignoring the Need for External Tools:** LLMs are language models, not bioinformatics tools. **Pitfall:** Relying on the LLM to perform complex calculations or high-precision alignments that require specialized software (e.g., BLAST, Clustal Omega). The LLM should be used to *interpret* or *generate code* for these tools.

## URL
[https://arxiv.org/html/2503.04490v1](https://arxiv.org/html/2503.04490v1)
