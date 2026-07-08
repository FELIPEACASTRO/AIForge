# Precision Agriculture Prompts

## Description
**Prompt Engineering for Precision Agriculture** is the discipline of creating detailed instructions and contexts for Large Language Models (LLMs) and other generative AIs, enabling them to process complex agricultural data (such as soil analyses, satellite imagery, climate data, and machinery data) to generate actionable *insights*, recommendations, and simulations. It is based on the principle of **Precision Agriculture**, which aims to manage the spatial and temporal variability of the field to optimize input usage and increase productivity. Generative AI in Agriculture acts as a **virtual technical assistant**, democratizing access to technical knowledge and supporting strategic decisions, such as fertilization optimization, pest forecasting, and market analysis. The effectiveness of the prompt depends on the inclusion of specific data and on clearly defining the role the AI should assume (Role Prompting).

## Examples
```
| Practical objective | Command prompt |
| --- | --- |
| **1. Optimize fertilizer use** | `Act as an agronomist specialized in soils. Based on this soil analysis [paste the analysis data], for a crop of [corn], what is the recommended NPK formulation and what is the ideal amount per hectare to maximize productivity and avoid waste?` |
| **2. Predict the best planting window** | `Analyze the historical rainfall and temperature data from the last 10 years for the region of [your city/region] and the climate forecast for the next 3 months. What is the ideal window of days for planting [soybean] to minimize the risk of a dry spell during the flowering phase?` |
| **3. Calculate the break-even point** | `Create a spreadsheet to calculate the break-even point of my [coffee] harvest. My annual fixed costs are R$[value] and my variable costs per bag are R$[value]. The estimated selling price per bag is R$[value]. How many bags do I need to sell to cover the costs?` |
| **4. Pest/Disease Detection** | `Act as a plant pathologist. Analyze the drone image [image URL] of plot 3, which shows yellowish spots on the [wheat] crop. Identify the most likely disease, the infestation level (in %), and suggest the most effective chemical pesticide, including the recommended dosage.` |
| **5. Predictive Maintenance** | `Based on the telemetry data of tractor [Model/ID] over the last 100 hours of use (oil temperature: [value], engine vibration: [value], hydraulic pressure: [value]), identify the probability of failure over the next 30 days. If the probability is high, which component should be inspected and what is the recommended preventive maintenance?` |
| **6. Market Analysis** | `Act as a commodities market analyst. Considering the forecast of a record harvest in Brazil, the rise in interest rates in the US, and the current dollar exchange rate ([value]), what is the best selling strategy for 5,000 bags of [soybean] over the next 60 days? Suggest a minimum and maximum target price.` |
| **7. Herd Management** | `Act as a veterinarian specialized in beef cattle. Analyze the monitoring data from the electronic ear tag of animal [animal ID] (body temperature: [value], rumination time: [value], activity level: [value]). The animal is in the postpartum period. Identify any anomaly and provide a nutritional management plan to optimize recovery and milk production.` |
```

## Best Practices
**Provide Context and Specific Data:** Always include input data (soil analysis, GPS coordinates, pest history, climate data, telemetry) in the prompt. The quality of the output is directly proportional to the quality of the input data.
**Define the Persona (Role Prompting):** Ask the AI to act as an "agronomist specialist", "market consultant", or "irrigation technician" to obtain more focused and specialized responses.
**Use Advanced Techniques (CoT/RAG):** For complex problems, use Chain-of-Thought (CoT), asking the AI to detail the reasoning step by step. Using Retrieval-Augmented Generation (RAG) with the farm's internal data (productivity history, soil maps) is the best practice to ensure local relevance.
**Specify the Output Format:** Request the output in structured formats (table, JSON, spreadsheet) to facilitate application and integration with agricultural management systems.
**Human Validation:** Always validate the AI's recommendations with a professional or with field experience before implementing them, as the AI is a decision-support tool, not a substitute for agronomic knowledge.

## Use Cases
**Optimization of Input Use:** Precise recommendation of fertilizers, pesticides, and water, based on georeferenced data and soil analyses.
**Pest/Disease Forecasting and Detection:** Analysis of images (drones, satellites) to identify infestation hotspots or diseases at early stages.
**Strategic Crop Planning:** Analysis of historical data and climate forecasts to determine the ideal planting window and the productivity estimate (*yield*).
**Predictive Maintenance of Machinery:** Analysis of sensor data on tractors and harvesters to predict mechanical failures.
**Market Analysis and Commercialization:** Suggestion of the best moment to sell production, based on commodities market analysis and futures prices.
**Herd Management (Precision Livestock Farming):** Monitoring of health, behavior, and optimization of animal feeding.

## Pitfalls
**Hallucinations:** The AI may generate agronomic recommendations that are plausible but factually incorrect or not suited to the local reality.
**Sensitivity to Phrasing (Prompt Brittleness):** Small changes in the prompt can lead to drastically different results.
**Dependence on Input Data:** The quality of the output depends directly on the quality and completeness of the data provided in the prompt. Incomplete or outdated data leads to poor results.
**Lack of Understanding of Complex Systems:** LLMs may have difficulty modeling the full complexity of an agricultural ecosystem (soil, climate, biology), leading to excessive simplifications.
**Data Bias:** If the AI was trained on data from a specific region or crop, its recommendations may not be applicable to others.

## URL
[https://treinamentosaf.com.br/10-usos-praticos-da-ia-no-agronegocio-que-aumentam-o-lucro-em-35/](https://treinamentosaf.com.br/10-usos-praticos-da-ia-no-agronegocio-que-aumentam-o-lucro-em-35/)
