# Drug Discovery Prompts

## Description
Prompt Engineering for Drug Discovery (Drug Discovery Prompts) is the art and science of creating optimized instructions for Large Language Models (LLMs) and Multimodal Models (MLLMs) with the goal of accelerating and improving the stages of the drug development cycle. This technique allows researchers and medicinal chemists to use the natural language processing capabilities of LLMs for complex tasks, such as **virtual screening**, **molecule optimization**, **ADMET property prediction** (Absorption, Distribution, Metabolism, Excretion, and Toxicity), and the **analysis of vast amounts of scientific literature** [1] [2].

Instead of relying solely on specifically trained machine learning (ML) models, Prompt Engineering adapts general-purpose models to the scientific domain, acting as an "evaluator, collaborator, and scientist" in the R&D process [3].

## References
[1] Certara. Best Practices for AI Prompt Engineering in Life Sciences in 2025. Available at: [https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/](https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/)
[2] Othman, Z. K. et al. Advancing drug discovery and development through GPT models: a review on challenges, innovations and future prospects. *Intelligence-Based Medicine*, 2025.
[3] Zhang, H. et al. The evolving role of large language models in scientific innovation: Evaluator, collaborator, and scientist. *arXiv preprint arXiv:2507.11810*, 2025.

## Examples
```
Below are 5 concrete and actionable prompt examples, focused on different stages of drug discovery:

1.  **ADMET Property Prediction (ADMET Prediction)**
    \`\`\`
    Act as a senior medicinal chemist. Your task is to predict the ADMET properties for the molecule with the SMILES string: C1=CC=C(C=C1)C(C(=O)O)N.
    
    Provide the response in JSON format with the following keys:
    - 'Molecule_SMILES'
    - 'Toxicity_Prediction' (e.g., 'Low', 'Medium', 'High')
    - 'Solubility_Prediction' (e.g., 'High', 'Low')
    - 'LogP_Value' (Numeric value)
    - 'Rationale' (Brief explanation of the reasoning behind the predictions)
    \`\`\`

2.  **De Novo Molecule Generation**
    \`\`\`
    Generate 5 SMILES strings for molecules that act as selective inhibitors of the [Target Protein, e.g., EGFR] receptor.
    
    **Constraints:**
    - Molecular weight between 350 and 450 Da.
    - LogP below 3.5.
    - Must contain a pyridine ring.
    
    List the 5 molecules in a numbered list format, with no additional text.
    \`\`\`

3.  **Scientific Literature Analysis (Data Extraction)**
    \`\`\`
    You are a research assistant. Analyze the following scientific article abstract: [ABSTRACT TEXT HERE].
    
    **Task:** Extract all drug-drug interactions mentioned and list them in a Markdown table format with the columns 'Drug 1', 'Drug 2', and 'Observed Effect' (e.g., 'Potentiation', 'Inhibition'). If there are no interactions, respond "No drug-drug interactions found".
    \`\`\`

4.  **Lead Optimization (Bioavailability Improvement)**
    \`\`\`
    The candidate molecule with the SMILES string: [SMILES STRING HERE] demonstrated low oral bioavailability (F% < 10) in preclinical studies.
    
    **Task:** Suggest three distinct structural modifications to increase bioavailability, focusing on improving solubility and permeability.
    
    For each suggestion, provide:
    1. The chemical rationale for the modification.
    2. The new SMILES string of the modified molecule.
    \`\`\`

5.  **Target Validation and Mechanism of Action**
    \`\`\`
    Explain the role of the protein [Protein Name, e.g., JAK2] in the pathogenesis of the disease [Disease Name, e.g., Myelofibrosis].
    
    **Instructions:**
    - Use accessible but scientifically precise language.
    - Mention the mechanism of action and the signaling pathways involved.
    - Respond in 3 concise paragraphs.
    \`\`\`
```

## Best Practices
The best practices for "Drug Discovery Prompts" are based on clarity, specificity, and the incorporation of domain knowledge [1]:

*   **Define the Persona and Context:** Begin the prompt by instructing the LLM to assume a specific role (e.g., "Act as a senior medicinal chemist" or "You are a bioinformatics expert").
*   **Prompt Structure (CDTF):** Use the four essential components: **Context** (the role and objective), **Data** (the SMILES string, the article text), **Task** (the action to be performed), and **Format** (JSON, Markdown table, numbered list) [1].
*   **One-Shot/Few-Shot Prompting:** Providing one or more examples of the desired input and output is crucial for technical tasks, such as SMILES generation or toxicity classification.
*   **Hallucination Constraint:** Include explicit instructions to mitigate "hallucination" (generation of false information). E.g.: "Use only information from peer-reviewed articles published after 2023. If the information is not available, respond 'I don't know'."
*   **Use of Structured Formats:** Whenever possible, request the output in structured formats (JSON, CSV, Markdown Table) to facilitate subsequent analysis and processing.

## Use Cases
*   **Property Prediction (ADMET):** Quickly predict the toxicity, solubility, permeability, and other pharmacokinetic properties of thousands of candidate molecules.
*   **De Novo Molecule Generation:** Create new molecular structures based on a target property profile (Target Product Profile - TPP) and specific chemical constraints.
*   **Literature Review and Synthesis:** Extract specific information (e.g., doses, clinical trial results, drug interactions) from scientific articles and patents in an automated way.
*   **Lead Optimization:** Suggest chemical modifications to improve an undesirable property (e.g., increase potency, reduce toxicity) of a lead compound.
*   **Retrosynthetic Analysis:** Propose viable chemical synthesis routes for a target molecule.

## Pitfalls
*   **Hallucination of Scientific Data:** The most serious risk. The LLM may generate invalid SMILES strings, incorrect toxicity data, or cite nonexistent articles. **Solution:** Rigorous constraints and cross-verification against reliable databases.
*   **Lack of Chemical Specificity:** Vague prompts result in chemically irrelevant or unfeasible suggestions. **Solution:** Use precise technical terms (e.g., "selective inhibitor", "hydroxyl functional group", "LogP").
*   **Over-Reliance on the LLM:** The LLM is an assistance tool, not a chemist. Blindly trusting the suggestions without experimental or computational validation can lead to dead ends.
*   **Context Limitation:** The ability to process long SMILES strings or large datasets may be limited by the model's context window size. **Solution:** Break complex tasks into smaller subtasks (Chain-of-Thought or the use of agents).

## URL
[https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/](https://www.certara.com/blog/best-practices-for-ai-prompt-engineering-in-life-sciences/)
