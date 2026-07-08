# Soil Analysis Prompts

## Description
Prompt Engineering for Soil Analysis refers to the creation of structured and detailed instructions for Large Language Models (LLMs) and other Artificial Intelligence (AI) systems with the goal of processing, interpreting, and generating recommendations based on soil analysis data. Rather than being a prompt technique in itself, it is a **specialized application** of Prompt Engineering in the domain of Agriculture and Soil Science. Prompts are crucial for translating raw data (such as laboratory results, IoT sensor readings, or remote sensing data) into actionable information for farmers, agronomists, and researchers. Effectiveness depends on the accuracy of the input data and on the prompt's ability to define the agronomic context, the variables of interest (pH, NPK, organic matter, texture), and the desired output format (e.g., fertilizer recommendation, deficiency diagnosis).

## Examples
```
1. **Nutrient Deficiency Diagnosis:**
```
Act as an expert agronomist. Analyze the following soil analysis results for a corn crop at the V6 stage: [pH: 5.8, Organic Matter: 2.5%, Phosphorus (P): 12 ppm (Low), Potassium (K): 150 ppm (Medium), total Nitrogen (N): 0.1%].
1. Identify the most critical nutrient deficiency.
2. Explain the probable cause.
3. Suggest an immediate intervention.
Output format: Table with Deficiency, Cause, and Recommendation.
```

2. **Liming and Gypsum Recommendation:**
```
Based on the soil data (Crop: Soybean, Soil Type: Clayey, current pH: 4.9, Base Saturation (V%): 35%, CEC at pH 7.0: 15 cmolc/dm³), calculate the liming requirement to raise V% to 60%.
1. What is the amount of limestone (ECCE 80%) needed per hectare?
2. What is the amount of gypsum (if needed) to neutralize the Toxic Aluminum (Al³⁺: 1.5 cmolc/dm³)?
3. Provide the step-by-step calculation.
```

3. **IoT Sensor Data Interpretation:**
```
Interpret the following soil sensor data series for a coffee plantation (Catuaí variety):
[Day 1: Moisture 45%, Temperature 22°C, Electrical Conductivity 0.8 dS/m]
[Day 2: Moisture 38%, Temperature 25°C, Electrical Conductivity 0.9 dS/m]
[Day 3: Moisture 30%, Temperature 28°C, Electrical Conductivity 1.1 dS/m]
What does the downward trend in moisture and the increase in EC indicate? What is the irrigation recommendation for Day 4, considering the wilting point at 25%?
```

4. **Soil Texture Classification (From Raw Data):**
```
Classify the soil texture based on the following particle-size composition:
[Sand: 65%, Silt: 20%, Clay: 15%]
1. Use the textural triangle to determine the class.
2. Describe two agronomic implications of this texture (e.g., drainage, water retention).
```

5. **Fertilization Optimization with Constraints:**
```
Create a fertilization plan for the wheat crop (productivity target: 5 ton/ha) in soil with the following characteristics: [P: 18 ppm (Medium), K: 200 ppm (Good), pH: 6.2].
Constraint: The budget allows a maximum of 100 kg/ha of NPK fertilizer (formula 10-20-20).
1. Calculate the ideal dose of N, P₂O₅, and K₂O.
2. Adjust the recommendation to the budget constraint, prioritizing the most limiting nutrient.
3. Justify the prioritization.
```

6. **Salinity Risk Analysis:**
```
Assess the salinity risk for an irrigated soil in a semi-arid region.
Data: [Electrical Conductivity (EC): 4.5 dS/m, pH: 7.8, Exchangeable Sodium Percentage (ESP): 10%].
1. Classify the soil (Saline, Sodic, Saline-Sodic, or Normal).
2. Describe the impact of this condition on the cotton crop.
3. Suggest a management measure to mitigate the problem.
```
```

## Best Practices
**Provide Structured and Complete Data:** Always include as much soil analysis data as possible (pH, NPK, OM, CEC, Aluminum, etc.), along with the **agronomic context** (crop, development stage, climate, productivity target). **Define the Role (Role Prompting):** Begin the prompt by instructing the LLM to act as an expert (e.g., "Act as an agronomist specialized in tropical soils") to activate the model's specialized knowledge. **Specify the Output Format:** Request the output in an easy-to-use format (table, list, JSON) to ensure the information is actionable and not just a block of prose. **Include Constraints and Variables:** If the recommendation has constraints (budget, type of fertilizer available, local legislation), include them explicitly so the LLM considers them in the optimization. **Request Justification:** Ask the LLM to justify its recommendations. This helps validate the response and identify possible hallucinations or calculation errors.

## Use Cases
**Fertility Recommendation:** Generation of optimized fertilization and liming plans based on soil analyses and productivity targets. **Rapid Problem Diagnosis:** Identification of nutrient deficiencies, toxicity, or pH problems from laboratory data or visual symptoms. **IoT Sensor Interpretation:** Translation of real-time data (moisture, temperature, EC) into management decisions (irrigation, leaching). **Education and Training:** Creation of case-study scenarios for agronomy students or training of agricultural technicians. **Variability Mapping:** Interpretation of multiple soil samples from different management zones to identify variability patterns and optimize input application.

## Pitfalls
**Overconfidence (Hallucination):** The LLM may "hallucinate" data or recommendations, especially if the prompt is vague or if the input data is incomplete. The accuracy rate in soil science is moderate (maximum of 65% in tests), requiring **human validation**. **Ignoring the Local Context:** The LLM may provide generic recommendations that do not apply to local legislation, soil type, or agricultural practices. The prompt should always include the geographic context. **Unit and Conversion Errors:** Mixing units of measurement (e.g., ppm vs. mg/dm³, kg/ha vs. lb/acre) can lead to catastrophic errors in recommendations. The prompt should be explicit about the units. **Insufficient Input Data:** The lack of crucial data (such as CEC or Aluminum content) will result in incomplete or incorrect recommendations. **Image/Chart Interpretation:** Text-only LLMs cannot directly interpret soil analysis charts or microscopy images (such as those used in IAEM), requiring the data to first be converted into structured text.

## URL
[https://www.sciencedirect.com/science/article/pii/S2950289625000028](https://www.sciencedirect.com/science/article/pii/S2950289625000028)
