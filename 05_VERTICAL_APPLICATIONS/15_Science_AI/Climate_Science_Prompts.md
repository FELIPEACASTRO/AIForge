# Climate Science Prompts

## Description
**Climate Science Prompts** are prompt engineering instructions designed to leverage the power of Large Language Models (LLMs) in analyzing, interpreting, and communicating complex data and concepts related to climate and climate change. This technique is fundamental for translating the vast and growing body of scientific literature and raw datasets into actionable knowledge for researchers, policymakers, businesses, and the general public [1] [2]. LLMs act as a bridge, processing millions of research articles and environmental data to identify trends, simulate scenarios, and generate accessible reports in a fraction of the time human analysis would take [3]. Effectiveness lies in the ability to structure the prompt to handle the complexity inherent in climate science, requiring specificity in terms of geographic, temporal, and methodological context.

## Examples
```
**1. Data Analysis and Forecasting (Role + Context + Format):**
"**Role:** Act as a senior climatologist specializing in regional modeling. **Task:** Analyze the historical precipitation data (attached in CSV) for the Paraná River basin between 1980 and 2020. **Requirement:** Based on this data and the SSP5-8.5 emissions scenario, generate an executive summary (maximum 300 words) on the probability of extreme drought events (defined as 3 standard deviations below the mean) in the region by 2050. Include a 'Key Uncertainties' section."

**2. Communication and Public Policy (Role + Target Audience):**
"**Role:** Act as a public policy consultant for a municipal government. **Task:** Summarize the latest IPCC report (AR6, Working Group II) on 'Impacts, Adaptation and Vulnerability' for an audience of mayors and city council members. **Requirement:** The summary should focus on urban risks (floods, heat waves) and present 5 low-cost adaptation recommendations, in clear, non-technical language."

**3. Carbon Footprint Analysis (Role + Simulation):**
"**Role:** Act as a corporate sustainability analyst. **Task:** Calculate the carbon footprint (Scope 1, 2, and 3) of a logistics company that uses 60% diesel and 40% electric vehicles. **Requirement:** Propose a 5-year transition plan to achieve carbon neutrality, detailing the annual actions and the estimated investment in charging infrastructure and the purchase of carbon credits. Present the result as a comparative table (Year vs. Emissions vs. Cost)."

**4. Disaster Management (Role + Immediate Action):**
"**Role:** Act as an emergency management officer. **Task:** A Category 4 hurricane is forecast to hit the coast of Santa Catarina in 48 hours. **Requirement:** Draw up a crisis communication checklist for the population, including: 3 immediate actions (0-12h), 3 preventive actions (12-48h), and 3 post-event actions. The tone should be urgent but reassuring."

**5. Renewable Energy Optimization (Role + Code):**
"**Role:** Act as a solar energy engineer. **Task:** Write a Python script (using the Pandas library) to clean and analyze a solar irradiation dataset (attached in CSV). **Requirement:** The script should identify the daily average irradiation (in kWh/m²) and detect anomalies (days with zero or very high irradiation), preparing the data for an energy production forecasting model."

**6. Scientific Review and Critique (Role + Source):**
"**Role:** Act as a peer reviewer for the journal Nature Climate Change. **Task:** Evaluate the methodology of the article 'The Role of Aerosols in Mitigating Global Warming' (abstract and methodology attached). **Requirement:** Identify 3 strengths and 3 weaknesses in the methodological approach, and suggest an alternative line of research the author could explore to strengthen the conclusions."

**7. Education and Awareness (Role + Analogy):**
"**Role:** Act as an environmental educator for elementary school students (10-12 years old). **Task:** Explain the concept of 'Ocean Acidification'. **Requirement:** Use a simple, everyday analogy (e.g., soda or vinegar) to illustrate the chemical process and its impact on marine life. The text should be at most 150 words."

**8. Knowledge Translation (Role + Translation):**
"**Role:** Act as a technical translator and climate expert. **Task:** Translate the following excerpt from a scientific article (in English) into Portuguese, ensuring the accuracy of the technical terms: 'The projected shift in the Intertropical Convergence Zone (ITCZ) under a high-emissions scenario is likely to exacerbate drought conditions across the Sahel region, necessitating a re-evaluation of current agricultural practices.' **Requirement:** Provide the translation and a brief note (1 sentence) explaining what the ITCZ is."
```

## Best Practices
**1. Definition of Role and Perspective (Role Prompting):** Assign a specific role to the model (e.g., "As a senior climatologist", "As an environmental policy analyst") to ensure that the response is framed with the correct tone, vocabulary, and focus for the climate science domain. **2. Extreme Clarity and Specificity:** Be as precise as possible. Instead of vague questions, use prompts that specify the region, the time period, the emissions scenario (e.g., SSP2-4.5), the climate variable (e.g., precipitation, daily maximum temperature), and the desired data source. **3. Output Format Specification:** Define the output format (table, executive summary, Python code, bar chart) to structure the response and make it immediately useful for analysis or communication. **4. Use of Context (In-Context Learning):** Include excerpts of data, articles, or reports in the prompt so that the LLM uses them as a basis for the analysis, reducing the chance of hallucinations and increasing relevance. **5. Minimization of the Carbon Footprint (ROCKS/ROCAS Method):** Create concise and effective prompts to reduce unnecessary iterations and AI energy consumption, aligning prompt engineering practice with sustainability.

## Use Cases
**1. Climate Trend Forecasting:** Analysis of historical and real-time data to predict weather patterns and environmental changes in specific regions. **2. Carbon Footprint Analysis:** Evaluation of industry emissions data (Scope 1, 2, and 3) to optimize sustainability strategies and propose energy transition plans. **3. Stakeholder Communication:** Generation of accessible and visually appealing reports, translating scientific complexity into the language of policymakers, NGOs, and the public. **4. AI-Driven Disaster Management:** Use of LLMs to plan and execute timely responses to climate emergencies (hurricanes, floods), creating checklists and crisis communication plans. **5. Renewable Energy Optimization:** Forecasting solar and wind energy outputs and optimizing resource use in industrial processes to support climate action and sustainability. **6. Literature Review and Synthesis:** Processing large volumes of scientific articles and reports (such as those from the IPCC) to quickly identify trends, research gaps, and scientific consensus.

## Pitfalls
**1. Vagueness and Ambiguity:** Generic prompts (e.g., "Talk about climate change") lead to superficial or imprecise responses that fail to provide actionable insights in a field that demands scientific precision. **2. Bias and Inaccuracy (Hallucinations):** LLMs may reflect biases from their training data or "hallucinate" scientific data, especially statistics and projections. It is crucial to request the citation of verifiable sources and to cross-check information. **3. Ignoring Scientific Complexity:** Treating complex climate models as simple questions and answers. The prompt should acknowledge the uncertainty and probabilistic nature of climate forecasts, requesting confidence ranges or alternative scenarios. **4. Lack of Geographic/Temporal Context:** Not specifying the region, time period, or emissions scenario (e.g., RCP 4.5, SSP2-4.5) will result in irrelevant or overly generalist responses. **5. Confusing Text Analysis with Modeling:** LLMs are excellent at processing and summarizing texts about climate, but they do not replace complex numerical climate models. The prompt should focus on interpreting or summarizing model results, not on performing the modeling itself.

## URL
[https://symufolk.com/llm-for-climate-data-analytics/](https://symufolk.com/llm-for-climate-data-analytics/)
