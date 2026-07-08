# Forestry Management Prompts

## Description
**Forestry Management Prompts** are prompt-engineering instructions specifically crafted to interact with Large Language Models (LLMs) and other Artificial Intelligence (AI) tools in the context of silviculture, ecology, and natural resource management. They are designed to translate the complexity of environmental data and forest management objectives into clear, structured commands that the AI can process. The main goal is to leverage AI for tasks such as forest inventory analysis, interpretation of remote sensing data (such as NDVI), silvicultural treatment planning, fire risk modeling, and forest health assessment. The effectiveness of these prompts lies in their ability to provide **detailed technical context**, **define the structure of the input data**, and **request a specific, actionable output format**, turning AI into a powerful decision-support tool for forestry professionals.

## Examples
```
**1. Forest Inventory Analysis (CoT):**
"I have forest inventory data for a 50-hectare stand of Pinus elliottii. The data includes the columns: 'parcela_id', 'DAP_cm', 'Altura_m', 'Volume_m3'. Please follow these steps: 1. Calculate the mean, median, and standard deviation of the DBH and Height. 2. Estimate the total timber volume in the stand. 3. Suggest a thinning plan (removal percentage and target trees) to reach a density of 400 trees/ha, justifying the decision based on the DBH distribution data."

**2. Remote Sensing Interpretation:**
"Analyze the following time series of NDVI (Normalized Difference Vegetation Index) values for a Eucalyptus reforestation area (coordinates: [LAT, LON]) from 2023 to 2025: [LIST OF DATES AND NDVI VALUES]. Identify seasonal patterns, detect any abrupt drops that may indicate disturbance (e.g., fire or pest), and correlate the vigor peaks with the periods of highest precipitation in the region. Present the analysis in a concise report format."

**3. Silvicultural Treatment Planning:**
"I am a forester planning the management of a 15-year-old stand of Australian Cedar (Cedrela odorata). The objective is high-value timber production. The site characteristics are: Elevation: 850m, Soil: Clayey, Current density: 650 trees/ha. What is the best pruning and thinning strategy to maximize diameter growth and stem quality? Present a schedule of interventions for the next 10 years, including the technical justification for each action."

**4. Fire Risk Assessment:**
"Act as an expert in forest fire risk modeling. Based on the following conditions: Dominant species: Pinus taeda, Topography: Steep slope (35% grade), Fuel moisture: 8%, Wind speed: 25 km/h. Describe the most likely fire spread scenario (rate of spread, flame height) and suggest three immediate preventive measures for the field crew. Use the Rothermel model as the basis for the analysis."

**5. Harvest Optimization:**
"Generate Python code (using the Pandas library) to process a CSV file called 'colheita.csv'. The file has columns 'Coordenada_X', 'Coordenada_Y', 'Volume_m3', and 'Custo_Colheita_R$'. The goal is to identify the 10 parcels with the best cost-benefit ratio (Volume/Cost) to prioritize harvesting. The code should load the file, calculate the metric, and print the 10 best parcels in descending order."

**6. Interpretation of Soil Test Results (Few-Shot):**
"Interpret soil test results for reforestation purposes. Example analysis: Sample A: pH 5.2, P 5 mg/dm3, K 40 mg/dm3. Analysis: Moderately acidic soil, deficient in Phosphorus and Potassium. Requires liming and NPK 04-14-08 fertilization. Now, analyze this new sample: Sample B: pH 6.5, P 15 mg/dm3, K 120 mg/dm3. What is the correction and fertilization recommendation for planting Teak (Tectona grandis)?"
```

## Best Practices
**1. Detailed Contextualization:** Always begin the prompt by clearly defining the forestry context (species, location, type of management, input data). The accuracy of the result depends on the accuracy of the context. **2. Definition of Technical Terms:** Avoid ambiguities. If you use specific silvicultural or ecological terminology (e.g., "windthrow", "DBH", "NDVI"), define it briefly, especially if the AI model is not specialized. **3. Clear Data Structure:** When providing data (tables, CSVs), describe the structure (columns, units of measurement) so the AI can process it correctly. **4. Specific Output:** Request the desired output format (table, summary, Python code, treatment plan) to ensure the response is useful. **5. Use of Chain-of-Thought (CoT):** For complex analyses (e.g., carbon calculation, risk modeling), guide the AI with a step-by-step process to improve the accuracy and traceability of the reasoning.

## Use Cases
nan

## Pitfalls
**1. Vague Requests:** Asking the AI to "talk about forest health" without specifying the region, species, pathogen, or time period. Lack of specificity leads to generic, useless responses. **2. Lack of Geographic/Ecological Context:** Omitting crucial information such as the biogeoclimatic zone, soil type, or management history. Forestry is highly site-dependent. **3. Assuming the AI's Technical Knowledge:** Using regional species codes (e.g., 'DF' for Douglas-fir) or non-standard units of measurement without defining them. **4. Overload of Unstructured Data:** Trying to paste large blocks of raw data without describing the structure or what is expected from the analysis. **5. Ignoring the Need for Sources:** Using the AI to obtain factual data (e.g., allometric equations) without requesting the source or scientific reference, which can lead to critical calculation errors in management.

## URL
[https://aiforester.com/learning/prompt-engineering-for-forestry.html](https://aiforester.com/learning/prompt-engineering-for-forestry.html)
