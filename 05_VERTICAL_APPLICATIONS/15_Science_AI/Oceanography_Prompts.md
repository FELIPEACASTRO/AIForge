# Oceanography Prompts

## Description
**Oceanography Prompts** refers to natural-language instructions (prompts) designed to interact with Large Language Models (LLMs) that are specialized or adapted for the domain of **Ocean Science** (Oceanography). Unlike generic prompts, these are formulated for complex tasks such as oceanographic data analysis, predictive modeling, simulation of marine phenomena, and retrieval of highly specific information from vast corpora of scientific literature and datasets.

The development of models like **OceanGPT** (the first LLM for the ocean domain) and platforms like **OceanAI** demonstrate the need for prompts that incorporate specialized knowledge, technical terminology (e.g., "CORA reanalysis", "pelagic deposits"), and output-format requirements for scientific analysis (e.g., code, data visualizations).

A complementary prompt framework, the **OCEAN** process (Objective, Context, Examples, Assess, Negotiate), is also relevant as a **best practice** for refining interaction with any LLM, ensuring that results are focused, contextually accurate, and scientifically rigorous.

## Examples
```
1.  **Specific Data Query:**
    `"Show the water level in Boston from the CORA reanalysis in June 1993. Generate the result as a time-series line chart and provide the Python code used for the visualization."`

2.  **Phenomenon Analysis:**
    `"Explain the mechanism of formation and dissipation of a moderate-intensity El Niño. Include the main oceanographic indicators (SST, thermocline) and suggest a short-term forecasting model."`

3.  **Underwater Robotics Simulation:**
    `"Simulate trajectory planning for an Autonomous Underwater Vehicle (AUV) to map a 500m x 500m coral reef. The AUV must maintain an altitude of 5 meters above the seafloor. Generate the simulation code in Python using the 'auv_toolkit' library."`

4.  **Scientific Literature Review:**
    `"Conduct a systematic review of the impact of ocean acidification on mollusk calcification over the past 5 years (2020-2025). List the 5 most cited articles and summarize their main conclusions."`

5.  **Educational Content Generation:**
    `"Create a prompt for a generic LLM that instructs it to act as an oceanographer. The goal is to generate a 45-minute lesson plan for high school students on thermohaline circulation, including a hands-on activity and assessment questions."`

6.  **Predictive Modeling:**
    `"Based on sea surface temperature (SST) data from the South Tropical Atlantic over the past 10 years, predict the probability of tropical cyclone formation in the next season. Justify the forecast with the relevant climate indices."`

7.  **Species Identification:**
    `"Describe the morphological characteristics and habitat of the species *Vampyroteuthis infernalis* (Vampire Squid). Create an image prompt to generate a photorealistic representation of the creature in its natural environment."`
```

## Best Practices
*   **Be Specific and Technical:** Use correct oceanographic terminology (e.g., "bathymetry", "western boundary current", "diatom phytoplankton") to refine the model's search of its specialized knowledge.
*   **Define the Output Format:** For scientific tasks, specify the desired format (e.g., "Generate the result as Python code", "Markdown Table", "Summary in IMRaD format").
*   **Cite the Data Source (if applicable):** When possible, include the data source or dataset to be queried (e.g., "Argo float data", "CORA reanalysis", "MODIS satellite imagery").
*   **Use the OCEAN Framework:** Apply the **O**bjective, **C**ontext, **E**xamples, **A**ssess, **N**egotiate framework to refine the interaction, especially for complex or high-precision tasks.
*   **Human Validation:** Always treat the LLM output as a starting point. Human validation and interpretation of results are crucial in scientific research.

## Use Cases
*   **Data Analysis and Visualization:** Query large oceanographic datasets (temperature, salinity, currents) using natural language and generate visualizations or analysis scripts.
*   **Literature Review and Synthesis:** Accelerate scientific research by summarizing articles, identifying trends, and comparing methodologies in oceanographic publications.
*   **Modeling and Forecasting:** Assist in creating predictive models for tides, waves, ocean circulation, and extreme weather events.
*   **Education and Training:** Generate teaching materials, quizzes, and interactive simulations for oceanography students.
*   **Marine Robotics:** Simulate and plan missions for AUVs and ROVs (Remotely Operated Vehicles), including route optimization and anomaly detection.

## Pitfalls
*   **Overreliance on Generic LLMs:** Non-specialized models (such as generic GPT-4 or Gemini) may "hallucinate" scientific data or technical terminology, leading to incorrect results.
*   **Lack of Scientific Context:** Prompts that are too vague or lack sufficient technical context will result in superficial or irrelevant responses for oceanographic research.
*   **Ignoring Data Validation:** Accepting the LLM output without verifying the data source or the scientific validity of the generated model/forecast.
*   **Real-Time Data Limitations:** Most LLMs are trained on historical data. Queries about real-time or very recent ocean conditions may fail or be inaccurate.
*   **Not Specifying Units and Scales:** Failing to define units of measurement (e.g., Celsius vs. Kelvin, meters vs. feet) or spatial/temporal scales can lead to interpretation errors.

## URL
[https://oceangpt.blue/oceangpt-en/](https://oceangpt.blue/oceangpt-en/)
