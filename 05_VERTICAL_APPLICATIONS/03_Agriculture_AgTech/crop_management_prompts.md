# Crop Management Prompts

## Description
Crop Management Prompts are specialized instructions provided to Large Language Models (LLMs) or multimodal Artificial Intelligence (AI) models to support agricultural decision-making. They are designed to simulate the expertise of an agronomist or specialist, providing analyses, diagnoses, and actionable recommendations to optimize production, manage pests and diseases, and improve sustainability. The effectiveness of these prompts lies in including specific context (crop type, growth stage, weather conditions, soil type) to generate accurate and relevant responses, turning AI into a decision-support tool in the field.

## Examples
```
**1. Multimodal Disease Diagnosis:** 'Act as a plant pathologist. Analyze the attached image of a soybean leaf (variety M8210, R2 stage). Describe the symptoms, identify the most likely disease, and provide a 7-day integrated management plan, including the recommended fungicide active ingredient and the justification for the choice.'
**2. Fertilization Optimization:** 'Based on the soil analysis (pH 5.8, P 12 ppm, K 150 ppm, OM 2.5%) and the yield target of 80 bags/hectare of corn (V6 stage), calculate the dose of nitrogen (N) and potassium (K) needed for topdressing. Recommend the best N source (urea or ammonium nitrate) and justify the choice in terms of efficiency and cost-effectiveness (considering the current urea price of R$ 3,000/ton).'
**3. Irrigation Management:** 'I am a cowpea grower (flowering stage) in a semi-arid region. The reference evapotranspiration (ETo) over the last 3 days was 6.5 mm/day. The soil is sandy loam and the field capacity is 15%. The current irrigation depth is 10 mm. Calculate the new depth and the irrigation interval to keep water stress at 20% of the available water, explaining the simplified water balance calculation.'
**4. Crop Planning:** 'Act as an agricultural planning consultant. For the next summer season, in an area with a history of root-knot nematodes (Meloidogyne javanica), suggest 3 crop rotation options that are economically viable and help reduce the nematode population. For each option, list the crop, the ideal planting period, and the main agronomic benefit.'
**5. Weed Control:** 'Identify the weed in the image (if multimodal) or described as "broad-leaf, prostrate, with milky sap" (if textual). The crop is ratoon sugarcane. Recommend an effective and selective post-emergence herbicide mixture, detailing the dose per hectare and the ideal application timing (weed height).'
**6. Sensor Data Analysis (NDVI):** 'Interpret the following NDVI data from a wheat area (grain-filling stage): "Zone 1 (10 ha): NDVI 0.85; Zone 2 (5 ha): NDVI 0.60". Describe the likely cause of the difference (assuming Zone 2 has a nutritional deficiency) and suggest a precision management action for Zone 2, such as variable-rate nitrogen application, specifying the additional amount needed.'
**7. Regulatory Compliance:** 'What is the restriction on using the active ingredient "Glyphosate" (Roundup) in permanent preservation areas (APP) in the state of São Paulo, Brazil? Summarize the applicable federal and state legislation and indicate the minimum application distance from water bodies, acting as an environmental lawyer.'
```

## Best Practices
**Context Specificity:** Always include as much detail as possible, such as crop, variety, phenological stage (e.g., V4, flowering), soil type, management history, and recent weather data.
**Persona Definition:** Assign the AI an expert persona (e.g., 'Act as an agronomist with 20 years of experience in soybeans in the Cerrado') to raise the quality and relevance of the recommendations.
**Requesting Reasoning:** Ask the AI to justify its recommendations, citing agronomic principles or data (if the model has access to them), which helps mitigate 'hallucinations'.
**Multimodal Use (if applicable):** For pest or disease diagnosis, include images and request a visual analysis, followed by a management plan.
**Focus on Action:** Structure the prompt so that the output is a clear, sequential action plan, not just a description of the problem.

## Use Cases
**Pest/Disease Diagnosis and Management:** Identification of problems from descriptions or images and suggestion of chemical or biological treatments.
**Plant Nutrition Optimization:** Recommendation of fertilizer doses and types based on soil analyses and crop stage.
**Crop Planning:** Assistance in choosing varieties, planting dates, and spacing, considering historical data and weather forecasts.
**Irrigation Management:** Calculation of water requirements and suggestion of irrigation schedules.
**Regulatory Compliance:** Interpretation of local regulations on pesticide use and management practices.
**Sensor Data Analysis:** Processing and interpretation of data from drones, satellites, and field sensors for mapping productivity and crop health.

## Pitfalls
**Hallucinations (Incorrect Recommendations):** The AI may suggest nonexistent, outdated, or unsuitable products or practices for the region. Always verify recommendations with a local agronomist.
**Lack of Local Specificity:** Generic responses that do not account for micro-climatic conditions, soil types, or farm-specific regulations.
**Over-Reliance:** Using AI as the sole source of decision-making, ignoring practical experience and field observation.
**Inclusion of Sensitive Data:** Avoid sharing confidential farm information (such as exact location or financial data) in prompts to public models.
**Ambiguous Prompts:** Open-ended or vague questions that result in equally vague and unhelpful responses.

## URL
[https://cropsandsoils.extension.wisc.edu/articles/ai-in-agriculture/](https://cropsandsoils.extension.wisc.edu/articles/ai-in-agriculture/)
