# Climate-Smart Agriculture Prompts

## Description
**Climate-Smart Agriculture (CSA) Prompts** are prompt engineering instructions designed to leverage Large Language Models (LLMs) and other generative AIs to solve the complex challenges of modern agriculture, which is under increasing pressure from climate change. The main objective is to optimize agricultural practices to achieve **food security** and **sustainable development** in a constantly changing climate.

These prompts are characterized by their need for **highly specific context** (soil, climate, crop, and location data) and by their request for **predictive and prescriptive analyses** that address the three pillars of CSA:

1.  **Sustainable Increase of Productivity and Income:** Optimization of inputs, irrigation, and crop management.
2.  **Adaptation and Resilience Building:** Selection of drought/heat-resistant crops, water and soil management.
3.  **Reduction/Removal of Greenhouse Gases (Mitigation):** Soil carbon sequestration strategies (Carbon Farming) and emission reduction.

The effectiveness of these prompts lies in the ability to transform complex remote sensing, meteorological, and soil analysis data into **actionable recommendations** for the farmer, promoting evidence-based decision-making for a more resilient and sustainable agricultural future.

## Examples
```
**1. Irrigation Optimization and Water Resilience**
```
Act as a water resilience expert. My corn farm (100 hectares, clay soil, coordinates [LAT, LONG]) is forecast to face a 30-day drought. Based on the soil moisture data (average of 45% in the root zone) and historical evapotranspiration (5mm/day), provide an emergency irrigation schedule (drip) to maximize crop survival while minimizing water use. Present the answer in a table with 'Day', 'Water Volume (m³/ha)' and 'Justification'.
```

**2. Carbon Sequestration Plan (Carbon Farming)**
```
Assume the role of a Carbon Farming consultant. Develop a 5-year plan to maximize carbon sequestration on a 500-hectare beef cattle farm (degraded pasture, semi-arid climate). The plan should include the implementation of high-density rotational grazing, planting of cover crops (recommended species), and no-till techniques. Estimate the potential CO2e capture per hectare/year and suggest monitoring metrics.
```

**3. Crop Adaptation to Extreme Weather Events**
```
I am a farmer in the [Region/State] region and the climate model predicts a 2°C increase in average temperature and a 15% reduction in precipitation over the next 10 years. My current crop is [Current Crop]. Recommend 3 alternative crops or genetically adapted varieties that demonstrate greater tolerance to water and thermal stress. For each recommendation, list the soil requirements and market potential.
```

**4. Integrated Pest and Disease Management (IPM) with Climate Focus**
```
Act as a plant pathologist. The increase in humidity and temperature due to an El Niño event is raising the risk of [Disease/Pest Name] in my [Crop] plantation. Describe a preventive, low-environmental-impact Integrated Pest Management (IPM) protocol. The protocol should prioritize biological control and the use of natural pesticides, with an action plan for the first week.
```

**5. Fertilizer Use Optimization for N2O Mitigation**
```
Based on the soil analysis (pH 5.5, total N 0.1%, Organic Matter 2.5%) for a soybean field, and seeking to reduce Nitrous Oxide (N2O) emissions, a potent GHG, suggest a nitrogen fertilizer application strategy. Include the most efficient form of nitrogen, the ideal application timing, and the application rate (kg/ha), justifying how the practice contributes to climate mitigation.
```

**6. Farm Renewable Energy Feasibility Analysis**
```
Evaluate the feasibility of installing a photovoltaic solar energy system on my 200-hectare farm ([LAT, LONG]). My average monthly consumption is 5,000 kWh. Provide an estimate of the required system size (kWp), the approximate investment cost, and the expected payback period, considering current tax incentives for Climate-Smart Agriculture.
```

**7. Development of an Early Warning System**
```
Create a prompt for an AI model that monitors satellite data (NDVI, EVI) and meteorological data (temperature, precipitation) to generate an early warning of water or nutritional stress in a 50-hectare sugarcane area. The alert should be triggered when the NDVI drops 10% below the historical average and the accumulated precipitation is 20% below the expected for the month. Define the input parameters and the alert output format.
```
```

## Best Practices
**1. Specificity and Geographic Context:** Always include specific data such as soil type, crop, local climate (temperature, historical precipitation), and geographic coordinates. Climate-Smart Agriculture is inherently local. **2. Role-Playing Definition:** Start the prompt by defining the AI's role (e.g., "Act as an agronomist specializing in carbon sequestration" or "Simulate being a water resilience expert"). **3. Inclusion of Input Data:** Provide raw or summarized data (e.g., soil analysis results, remote sensing data, pest history) so that the AI can perform predictive and prescriptive analyses. **4. Focus on Triple Solutions (CSA):** Direct the prompt to address the three pillars of CSA: productivity, adaptation/resilience, and mitigation (emission reduction/carbon sequestration). **5. Structured Output Format:** Request the answer in a specific format (e.g., table, numbered list, step-by-step action plan) to facilitate practical application in the field.

## Use Cases
**1. Input Optimization:** Determine the ideal amount of fertilizers and pesticides based on soil data and climate forecast, reducing costs and minimizing environmental pollution. **2. Crop Rotation and Agroforestry Planning:** Develop long-term plans that increase soil organic matter (carbon sequestration) and resilience to pests and diseases. **3. Climate Risk Management:** Create adaptation strategies for extreme weather events (droughts, floods, heat waves), including the selection of more resistant crop varieties. **4. Certification and Carbon Markets:** Generate reports and monitoring plans so that farmers can participate in carbon credit programs, quantifying the sequestered CO2. **5. Rural Training and Extension:** Create educational materials and best-practice guides (in accessible language) to disseminate CSA techniques among farming communities. **6. Soil Health Monitoring:** Analyze sensor data and satellite images to diagnose nutritional deficiencies or water stress in real time, enabling precise interventions.

## Pitfalls
**1. Lack of Geographic Specificity:** Using generic prompts without including exact soil, climate, and location data. CSA requires hyper-localized recommendations. **2. Overconfidence in Input Data:** Assuming that the AI can compensate for a lack of quality data (Garbage In, Garbage Out). The accuracy of the output depends on the accuracy of the soil analysis, remote sensing, and meteorological data provided. **3. Ignoring the Socioeconomic Context:** Focusing only on technical optimization without considering the farmer's financial capacity, labor availability, or local supply chains. **4. Requesting Non-Actionable Outputs:** Asking for theoretical analyses instead of concrete and implementable action plans (e.g., "Talk about CSA" vs. "Create a crop rotation plan for the next cycle"). **5. Model Bias:** The AI may favor high-tech solutions (e.g., drones, IoT) that may not be accessible or appropriate for small-scale farmers, unless the prompt specifies the resource constraint.

## URL
[https://weam.ai/blog/prompts/chatgpt-prompts-for-farming/](https://weam.ai/blog/prompts/chatgpt-prompts-for-farming/)
