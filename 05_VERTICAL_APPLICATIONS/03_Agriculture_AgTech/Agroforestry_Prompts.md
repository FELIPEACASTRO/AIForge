# Agroforestry Prompts

## Description
"Agroforestry Prompts" are prompt engineering instructions specifically formulated to interact with Large Language Models (LLMs) and Artificial Intelligence (AI) systems with the goal of obtaining information, analyses, recommendations, and action plans related to Agroforestry Systems (AFS). AFS are land-use systems that integrate trees with agricultural crops and/or livestock in a spatial and temporal manner, seeking ecological and economic benefits. The prompts are designed to handle the structural complexity and species diversity inherent to AFS, requesting multimodal data analysis (remote sensing, soil data, climate) and the generation of practical knowledge.

## Examples
```
1. **AFS Planning:** "Develop an implementation plan for a successional Agroforestry System for a 5-hectare area in the [Biome Name, e.g., Atlantic Forest] biome. The main focus is the production of [Main Crop, e.g., cacao] and [Tree Species, e.g., inga]. Include the list of pioneer, secondary, and climax species, the recommended spacing, and a management schedule for the first 5 years."
2. **Data Analysis:** "Analyze the following remote sensing data (NDVI and surface temperature) from my agroforestry area (attach data or describe the period) and identify areas with water or nutritional stress. Suggest specific management interventions for coordinates [X, Y]."
3. **Species Optimization:** "For an AFS with crops of [Crops, e.g., coffee and banana] and shade trees of [Shade Species, e.g., erythrina], what is the ideal spatial arrangement to maximize light interception by the main crop without compromising tree growth? Present the answer in a comparative table format."
4. **Pest and Disease Management:** "Based on the symptoms of [Symptom, e.g., yellowing leaves and presence of mealybugs] in the [Crop, e.g., cassava] crop within my AFS, provide an Integrated Pest Management (IPM) protocol that uses only biological and natural methods, without the use of agrochemicals."
5. **Harvest Recommendation:** "Considering the climate forecast for the next 30 days in the [Region Name] region and the maturation stage of the [Crop, e.g., açaí] crop, what is the ideal harvest window to maximize product yield and quality? Justify the answer based on humidity and temperature data."
6. **Carbon Sequestration Calculation:** "Calculate the above-ground carbon sequestration potential (in tonnes of CO2 equivalent per hectare/year) for a mature AFS with a density of [Number] trees/hectare, composed of the species [Species, e.g., mahogany, ipe, and rubber tree]. Cite the calculation methodology used."
7. **Integration with Livestock (Silvopastoral):** "Design a silvopastoral system for a beef cattle farm in [Region]. Recommend forage and tree species that provide shade and supplementary forage, and detail the sustainable animal stocking rate for the system."
```

## Best Practices
* **Contextual Specificity:** Include as much detail as possible about the context (biome, soil type, climate, existing crops and tree species, farmer's objectives) so that the LLM can provide hyper-localized and relevant recommendations.
* **Requesting Structured Data:** Request output in structured formats (tables, lists, JSON) to facilitate analysis and integration with other agricultural management tools.
* **Multimodal Focus:** In advanced prompts, reference the need for multimodal data analysis (satellite images, soil sensor data, climate models) to simulate the capability of more complex AI systems.
* **Emphasis on Sustainability:** Direct the LLM toward solutions that prioritize agroecological principles, such as biodiversity, nutrient cycling, and biological pest control.
* **Validation and Iteration:** Treat the LLM's output as an initial recommendation and use follow-up prompts to refine the analysis, question assumptions, and validate the feasibility of the suggestions.

## Use Cases
* **AFS Planning and Design:** Generation of layouts, species selection, and optimized planting schedules for different soil-climate conditions and production objectives.
* **Diagnosis and Monitoring:** Early identification of plant stress, diseases, or nutritional deficiencies through the analysis of remote sensing and field sensor data.
* **Management Decision-Making:** Obtaining recommendations on pruning, irrigation, fertilization, and pest control in real time.
* **Education and Rural Extension:** Creation of educational materials, best-practice guides, and answers to frequently asked questions for farmers and technicians.
* **Research and Modeling:** Generation of hypotheses, simulation of climate change scenarios, and calculation of ecosystem services (e.g., carbon sequestration).

## Pitfalls
* **Oversimplification:** The LLM may simplify the complexity inherent to AFS (interactions between species, microclimates), leading to generic or inadequate recommendations.
* **Data Bias:** The quality and bias of the LLM's training data may result in recommendations that favor conventional agriculture practices over agroecological approaches.
* **Lack of Local Context:** Without accurate input data about the site (soil, microclimate), the suggestions may be impractical or ineffective.
* **Technical Hallucinations:** The LLM may generate species names, management methods, or scientific data that seem plausible but are factually incorrect or nonexistent.
* **Dependence on Multimodal Data:** The effectiveness of more advanced prompts depends on the LLM's ability to process and interpret non-textual data (images, maps), which is not always guaranteed in publicly accessible models.

## URL
[https://www.researchgate.net/publication/393576880_Artificial_Intelligence_for_Agroforestry_A_Review](https://www.researchgate.net/publication/393576880_Artificial_Intelligence_for_Agroforestry_A_Review)
