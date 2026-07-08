# Chemical Engineering Prompts

## Description
Prompt Engineering in Chemical Engineering is the strategic application of large language models (LLMs) to accelerate the research, development, and optimization of chemical processes. It involves creating precise and contextual instructions to guide the AI in performing complex tasks, such as process simulation, reactor optimization, materials synthesis, safety analysis, and regulatory compliance [1] [2]. The main focus is to overcome the limitations of LLMs, such as "hallucinations" and lack of domain-specific knowledge, by providing detailed chemical and engineering context, few-shot prompting examples, and structured output formats [3]. This technique is crucial for transforming LLMs from general-purpose tools into specialized domain assistants, capable of handling the complexity and the need for factual accuracy inherent to chemical engineering [4].

## Examples
```
**1. Reactor Optimization:** "Act as a Process Optimization Engineer. For the hydrogenation reaction of benzene to cyclohexane (C6H6 + 3H2 -> C6H12) in a PFR reactor, with an inlet temperature of 400 K and a pressure of 30 bar, suggest 3 sets of flow rate and molar ratio conditions that maximize conversion (above 95%) and minimize energy cost. Present the output in a Markdown table with the columns: 'Scenario', 'Total Molar Flow Rate (mol/s)', 'H2:Benzene Ratio', 'Estimated Conversion (%)', 'Relative Energy Cost'."

**2. Safety Analysis (HAZOP):** "As a Process Safety Specialist, perform a HAZOP (Hazard and Operability Study) analysis for a distillation column operating under vacuum. Consider the parameter 'Pressure' and the deviation 'More Pressure'. Identify the possible causes, the consequences for the process, and the recommended mitigation actions. Format the response as a concise safety report."

**3. Materials Synthesis:** "I am a materials chemist. I want to synthesize a Metal-Organic Framework (MOF) based on zinc and terephthalate ligands. Using the solvothermal synthesis method, provide a step-by-step laboratory protocol, including the exact mass of the reagents (for 1g of final product), the ideal solvent, the temperature, and the reaction time. Cite the data source (e.g., scientific article) for the protocol."

**4. Process Troubleshooting:** "The CSTR reactor at the polyethylene production plant is showing an unexpected drop in the polymerization rate. The input variables (temperature, monomer concentration, catalyst concentration) are within specifications. List 5 failure hypotheses, starting with the most likely, and suggest a diagnostic test for each. Respond in a numbered list format."

**5. Regulatory Compliance:** "For the discharge of effluents from a fertilizer plant in Brazil, what are the main water quality parameters regulated by CONAMA (National Council for the Environment)? Create a table with the 'Parameter', the 'Maximum Permitted Limit (MPL)', and the corresponding 'CONAMA Resolution'. Focus on Total Nitrogen and Total Phosphorus."

**6. Heat Exchanger Design:** "Calculate the heat transfer area required for a Shell and Tube heat exchanger to cool 10 kg/s of hot oil from 150°C to 80°C, using cooling water that enters at 25°C and exits at 40°C. Provide the typical overall heat transfer coefficient (U) for this application (oil/water) and the calculation of the Log Mean Temperature Difference (LMTD). Present the final result of the area in m²."
```

## Best Practices
**1. Deep Contextualization:** Always provide complete chemical and engineering context. Include the specific reaction, the operating conditions (temperature, pressure, flow rate), the reactor type, and the safety/cost constraints. **2. Few-Shot Prompting (Examples):** For complex tasks such as synthesis data extraction or parameter optimization, include 2-4 input-output examples to demonstrate the desired format and reasoning. **3. Role-Playing:** Begin the prompt by defining the LLM as a "Senior Chemical Engineer" or "Process Safety Specialist" to align the response with domain knowledge. **4. Structured Output:** Request the output in structured formats (JSON, Markdown tables, numbered lists) to facilitate analysis and integration with other engineering tools. **5. Cross-Validation:** Always use the LLM output as a starting point or suggestion, and not as absolute truth. Validation with simulations, experimental data, or engineering standards is mandatory.

## Use Cases
nan

## Pitfalls
**1. Factual Hallucinations:** The LLM may generate incorrect thermodynamic, kinetic, or safety data. **Countermeasure:** Always validate critical numbers (boiling points, enthalpies, explosivity limits) against reliable databases (e.g., NIST, DIPPR). **2. Lack of Domain-Specific Knowledge:** General-purpose LLMs may fail on engineering nuances, such as the difference between an ideal and a real CSTR reactor. **Countermeasure:** Use "Few-Shot" prompts or provide simulation input data (e.g., mass/energy balance equations) to contextualize the model. **3. Training Data Bias:** The model may favor common or old solutions, ignoring recent innovations or proprietary solutions. **Countermeasure:** Explicitly ask for "innovative solutions" or "unconventional alternatives" and restrict the search to a time period (e.g., "published after 2023"). **4. Ignoring Engineering Constraints:** The LLM may suggest solutions that are thermodynamically possible but economically unfeasible or mechanically impractical. **Countermeasure:** Include cost, material, and operability constraints in the prompt (e.g., "The solution must use 316 stainless steel and have a capital cost 10% lower than the current design").

## URL
[https://pubs.acs.org/doi/10.1021/acscentsci.4c01935](https://pubs.acs.org/doi/10.1021/acscentsci.4c01935)
