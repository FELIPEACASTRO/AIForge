# Chemistry Experiment Design Prompts

## Description
A Prompt Engineering technique focused on leveraging Large Language Models (LLMs) and AI agents to plan, optimize, and, in some cases, execute chemical experiments. The most advanced approach involves the use of an AI agent (such as ChemCrow, described in Nature Machine Intelligence) that employs an iterative workflow (Thought, Action, Observation - ReAct/MRKL) to interact with specialized chemistry tools and databases. This transforms a simple text request into a detailed experimental action plan, covering everything from retrosynthesis to safety assessment and yield optimization.

## Examples
```
1. **Synthesis of a Specific Compound:** "As an autonomous chemistry agent with access to reaction databases (e.g., Reaxys, SciFinder) and safety tools (e.g., PubChem), plan the synthesis of 1-phenylethanol from benzene. The plan should include the retrosynthesis route, the reaction conditions for each step (temperature, solvent, catalyst), and a safety risk assessment for the final procedure."
2. **Yield Optimization:** "The yield of the Suzuki-Miyaura reaction between iodobenzene and phenylboronic acid is 75%. Use your optimization tools (e.g., Bayesian optimization algorithms) to suggest three modifications to the reaction conditions (catalyst, ligand, solvent, or temperature) that maximize the yield to more than 90%. Present the three best suggestions with the chemical justification and the expected theoretical yield."
3. **Discovery of a New Material:** "Design a new conductive polymer (material) for use in organic solar cells (OPVs). The material must have a *band gap* below 1.8 eV and be soluble in chloroform. Use your molecular modeling tools (e.g., DFT) to suggest the molecular structure of the monomer and the polymerization synthesis procedure."
4. **Safety and Risk Analysis:** "Analyze the nitration reaction of toluene for the production of TNT. Use your safety and thermodynamics tools to identify the main risks (explosion, toxicity, byproducts) and suggest a detailed risk mitigation protocol, including the necessary personal protective equipment (PPE) and the waste disposal procedure."
5. **Retrosynthesis Route Planning:** "Determine the most efficient and cost-effective retrosynthesis route for the antiviral drug Remdesivir. Use your route search tools (e.g., retrosynthesis LLMs) to compare at least two published routes, evaluating the number of steps, the estimated cost of the starting reagents, and the toxicity of the intermediates."
6. **Simulation and Property Prediction:** "Predict the pKa of acetic acid in methanol at 25°C. Use your simulation tools (e.g., COSMO-RS) to calculate the value and compare it with the experimental value in water. Explain the observed difference based on solvent effects."
```

## Best Practices
- **Clear Definition of the Objective:** The prompt should be specific about the chemical target (molecule, reaction, property) and the desired outcome (plan, optimization, prediction).
- **Specification of Constraints:** Include practical constraints such as cost, safety, available reagents, or laboratory conditions (e.g., "without using heavy metals", "maximum temperature of 80°C").
- **Explicit Tool Integration:** Mention the tools or databases the LLM should use (e.g., "Use PubChem for safety data", "Consult ChemSpider for structures").
- **Use of the ReAct/Chain-of-Thought Format:** For complex tasks, instruct the LLM to follow a logical reasoning process (Thought, Action, Observation) before presenting the final answer.

## Use Cases
- **Organic and Inorganic Synthesis:** Planning synthesis routes for complex molecules.
- **Drug Discovery and Optimization:** Suggesting new drug candidates and optimizing their properties (ADMET).
- **Materials Science:** Designing new materials with specific properties (e.g., polymers, catalysts, semiconductors).
- **Risk and Safety Analysis:** Assessing hazards in chemical reactions and developing safety protocols.
- **Process Optimization:** Fine-tuning reaction conditions to maximize yield, selectivity, or sustainability.

## Pitfalls
- **Chemical Hallucinations:** The LLM may generate reaction routes or molecules that are thermodynamically or kinetically unfeasible.
- **Dependence on Training Data:** The model may repeat errors or biases present in the training data, especially in niche chemistry.
- **Unmitigated Safety Risk:** The lack of robust integration with safety tools can lead to the suggestion of dangerous procedures.
- **Overestimation of Capability:** The user may overestimate the LLM's ability to replace the intuition and knowledge of an experienced chemist.

## URL
[https://www.nature.com/articles/s42256-024-00832-8](https://www.nature.com/articles/s42256-024-00832-8)
