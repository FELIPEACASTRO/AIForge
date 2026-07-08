# Environmental Engineering Prompts

## Description
Prompt Engineering for the Environmental sector is the practice of creating highly specific and contextual instructions (prompts) for language models (LLMs) with the goal of solving complex Environmental Engineering problems, such as water resource management, waste treatment, environmental impact assessment (EIA), and pollution modeling. This technique is crucial for adapting generic foundation models (such as GPT-4 or Gemini) to the technical vocabulary, specific regulations, and scientific data of the environmental domain, transforming them into specialized assistants, or "WaterGPTs" [1]. The focus is on providing maximum technical context, input data, and regulatory constraints to ensure that outputs are accurate, reliable, and applicable in the real world. It is a rapidly evolving field, with an emphasis on reducing the environmental impact of AI itself (Green Prompt Engineering) and on creating domain-adapted language models [3].

## Examples
```
**1. Environmental Impact Assessment (EIA)**
`Act as a senior environmental consultant. Analyze the construction project of a 50 km highway in the Atlantic Forest (coordinates: [insert data]). Based on CONAMA Resolution 001/86, generate a detailed list of the 10 main expected negative environmental impacts (direct and indirect) and suggest 3 specific mitigation measures for each. Format the output as a Markdown table.`

**2. Water/Effluent Treatment Optimization**
`You are a process engineer at a Water Treatment Plant (WTP). The incoming effluent has the following characteristics: BOD = 250 mg/L, COD = 500 mg/L, pH = 6.5, Flow = 1000 m³/day. The discharge standard requires BOD < 30 mg/L. Describe the most efficient biological treatment process (conventional activated sludge) to reach this target. Calculate the minimum volume of the biological reactor and the sludge age (SRT) required, explaining the logic behind the calculations.`

**3. Atmospheric Pollutant Dispersion Modeling**
`Act as an air quality modeler. A new SO2 emitter will be installed at [Location]. The emission rate is 50 g/s. Use the Gaussian Plume model (or cite a more appropriate model) to estimate the maximum SO2 concentration at 1 km distance, under atmospheric stability condition D (moderately unstable) and a wind speed of 5 m/s. List the assumptions made and the step-by-step calculation.`

**4. Risk Analysis and Regulatory Compliance**
`You are an EHS (Environment, Health, and Safety) specialist. Analyze the annex of NBR 10004 (Classification of Solid Waste) and classify the waste "oily sludge from machine maintenance" (source code [insert code]). Justify the classification (Class I or II) and suggest the safest and most legally accepted final disposal method in Brazil.`

**5. Scientific Literature Synthesis**
`Review the 5 most recent articles (2023-2025) on the use of nanotechnology for the removal of microplastics from wastewater. Synthesize the main findings, the limitations of the technology, and the cost-benefit in a 500-word analysis. Include the references in ABNT format.`

**6. Green Infrastructure Design**
`Act as an urban drainage engineer. Design a rain garden for a 500 m² parking lot area in a city with an average annual precipitation of 1500 mm. Describe the soil layers, the selection of native plant species (cite 3 examples), and the sizing of the infiltration area to retain 80% of the runoff volume from a 25 mm design storm.`
```

## Best Practices
**1. Specificity and Context:** Always define the **role** of the LLM (e.g., "Act as a senior environmental engineer") and provide detailed **context** (e.g., type of effluent, local regulation, input data).
**2. Grounding:** Use the Retrieval-Augmented Generation (RAG) technique or include reference data and documents (standards, reports, sensor data) directly in the prompt to avoid hallucinations and ensure adherence to technical facts and regulations [1].
**3. Chain of Thought (CoT):** For complex problems (e.g., pollutant dispersion modeling), instruct the LLM to detail the process step by step before providing the final answer. E.g.: "First, list the input variables. Second, describe the mathematical model. Third, apply the data and provide the result."
**4. Structured Output Format:** Request the output in easy-to-process formats, such as Markdown tables, JSON, or Python code, to facilitate integration with other engineering tools.
**5. Human Validation:** Never blindly trust the outputs for critical decisions. Use the LLM as an assistant for drafts, preliminary analyses, or document synthesis, but final validation must be performed by a human expert [2].

## Use Cases
**1. Water Resource Management:** Optimization of water distribution networks, prediction of water quality in rivers and reservoirs, and design of efficient irrigation systems.
**2. Water and Effluent Treatment:** Simulation of treatment processes (e.g., aeration, sedimentation), calculation of chemical dosing, and diagnosis of operational failures at Treatment Plants (WTPs and WWTPs) [1].
**3. Environmental Impact Assessment (EIA) and Licensing:** Generation of draft impact reports, regulatory compliance analysis, and identification of environmental risks in new projects.
**4. Pollution Modeling and Forecasting:** Simulation of the dispersion of atmospheric or aquatic pollutants, forecasting of pollution events, and toxicological risk analysis.
**5. Solid Waste Engineering:** Optimization of collection routes, classification of hazardous waste, and conceptual design of landfills or recycling plants.
**6. Sustainability and EHS (Environment, Health, and Safety):** Development of corporate sustainability policies, creation of safety procedures, and training of employees on environmental standards [2].

## Pitfalls
**1. Technical Hallucinations:** The LLM may invent standards, reference values, or calculation procedures that appear correct but are factually incorrect or outdated. **Countermeasure:** Always use the Grounding (RAG) technique and validate the output against official sources.
**2. Lack of Geographic/Regulatory Context:** The model may suggest solutions that are not applicable due to local regulations (e.g., CONAMA in Brazil, EPA in the USA) or specific climatic/geological conditions. **Countermeasure:** Explicitly include the country, state, municipality, and the applicable regulatory standard in the prompt.
**3. Oversimplification:** Engineering problems (e.g., hydrological modeling, reactor kinetics) are complex, and the LLM may oversimplify, ignoring critical variables. **Countermeasure:** Use Chain of Thought (CoT) and require the listing of all assumptions and variables considered.
**4. Data Bias:** If the model was trained predominantly on data from developed countries, it may suggest technologies or practices that are economically unfeasible or inappropriate for the context of developing countries. **Countermeasure:** Request solutions adapted to specific budget or infrastructure constraints.
**5. Unit Confusion:** The model may mix units of measurement (e.g., mg/L with ppm, m³/s with L/s). **Countermeasure:** Specify the desired unit system (e.g., SI) and ask the model to confirm the units in the output.

## URL
[https://www.nature.com/articles/s41545-025-00509-8](https://www.nature.com/articles/s41545-025-00509-8)
