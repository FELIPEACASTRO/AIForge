# Geophysics Prompts

## Description
Geophysics Prompts refer to the application of **Prompt Engineering** techniques to interact with Large Language Models (LLMs) and Foundation Models (FMs) in the geosciences domain, with a particular focus on geophysics (seismology, gravimetry, magnetometry, etc.). This emerging area, with significant publications from 2024-2025 onward, aims to automate complex tasks, generate synthetic data, interpret geophysical data, and assist in research. The core of the technique lies in providing context, specialized language, and structured instructions so that the LLM acts as a "geophysical agent" capable of performing data analysis, generating code, interpreting results, and even assisting in decision-making within intelligent seismic workflows. The effectiveness of prompts in geophysics depends on the inclusion of technical terminology (e.g., "seismic attribute analysis", "well data inversion"), format specifications (e.g., JSON, Python code), and a clear definition of the model's role (e.g., "Act as an exploration geophysicist").

## Examples
```
1.  **Seismic Interpretation (Agent Prompt):**
    "Act as an interpretation geophysicist. Analyze the following seismic dataset (provide link or description of attributes) and identify the main faults and horizons. Generate a summary in Markdown and then write the Python code (using the 'segyio' library) needed to visualize the seismic section, highlighting the amplitude anomalies that may indicate hydrocarbons."

2.  **Synthetic Data Generation (Structured Prompt):**
    "Generate 100 synthetic seismic data records in JSON format. Each record should include the following fields: 'Location' (geographic coordinates), 'Depth' (in meters), 'P-Wave Velocity' (in m/s), 'Acoustic Impedance' (in kg/m²s), and 'Lithology' (e.g., Sandstone, Shale, Limestone). Ensure that the correlation between 'P-Wave Velocity' and 'Lithology' is geologically plausible."

3.  **Seismic Attribute Analysis (Analysis Prompt):**
    "Based on the principles of exploration geophysics, explain the relationship between the 'Coherence' seismic attribute and the presence of geological faults. Then, provide an optimized prompt for an LLM that requests the analysis of a coherence map to identify structural discontinuities, specifying the output as a list of coordinates and the fault classification (normal, reverse, thrust)."

4.  **Literature Review (Research Prompt):**
    "Conduct a concise literature review on the application of Convolutional Neural Networks (CNNs) in salt dome detection in 3D seismic data. Identify the three most relevant articles published between 2023 and 2025 and summarize the main contribution of each one in a Markdown table with columns for 'Title', 'Lead Author', and 'Main Methodology'."

5.  **Parameter Optimization (Workflow Prompt):**
    "You are an expert in seismic processing. Describe the ideal workflow for pre-stack depth migration (PSDM) of a marine dataset. For each step (e.g., velocity picking, velocity modeling), provide a prompt that an LLM could use to optimize a specific parameter (e.g., 'Suggest an initial RMS velocity value for the water layer, given the average depth of 500m')."

6.  **Well Data Interpretation (Multimodal/Integration Prompt):**
    "Act as a well geophysicist. Integrate the logging data (Gamma Ray, Resistivity, Density) with the seismic interpretation. What is the most effective prompt for a multimodal LLM that, upon receiving the logging chart (image) and the lithology log (text), suggests the best well-tie with the seismic horizon 'Top Reservoir'?"
```

## Best Practices
*   **Role-Playing:** Start the prompt by defining the LLM's role (e.g., "Act as an exploration geophysicist", "You are an expert in seismology") to evoke specialized knowledge.
*   **Technical Specificity:** Use correct geophysical terminology (e.g., "travel time", "Kirchhoff migration", "Bouguer anomaly") to refine the search and the accuracy of the response.
*   **Output Structure:** Require structured output formats (JSON, CSV, Python code, Markdown table) to facilitate integration with geophysical analysis tools.
*   **Data Context:** Whenever possible, provide the data context (survey type, study area, key parameters) or ask the LLM to simulate a realistic context.
*   **Chain-of-Thought (CoT):** For complex tasks (e.g., interpretation), instruct the LLM to use *Chain-of-Thought* (CoT) or to decompose the problem into logical geophysical steps before providing the final answer.

## Use Cases
*   **Synthetic Data Generation:** Creating large training datasets for Machine Learning models in geophysics (e.g., fault detection, facies classification) when real data is scarce.
*   **Workflow Automation:** Automating repetitive steps in seismic workflows (e.g., quality control, horizon picking, processing parameter optimization).
*   **Interpretation and Analysis:** Assisting in the interpretation of seismic, gravimetric, and magnetic data, identifying patterns and anomalies that may be difficult to detect manually.
*   **Research and Education:** Rapid synthesis of scientific literature, explanation of complex geophysical concepts, and generation of specific instructional material.
*   **Disaster Monitoring:** Using multimodal LLMs to integrate satellite data (images) and text reports for rapid damage assessment after geological events (e.g., earthquakes, landslides).

## Pitfalls
*   **Geologically Implausible Hallucinations:** The LLM may generate synthetic data or interpretations that violate fundamental physical or geological principles (e.g., unrealistic seismic velocities). **Mitigation:** Include physical and geological constraints in the prompt.
*   **Excessive Dependence on Training Data:** The LLM may reflect biases or limitations of the training data, especially if it does not include high-quality geophysical data. **Mitigation:** Always validate the LLM's output with human expert knowledge.
*   **Vague Prompts:** Generic prompts (e.g., "Talk about seismology") result in superficial responses that are unusable for geophysical applications. **Mitigation:** Always be specific about the method, the data, and the expected result.
*   **Ignoring the Multimodal Nature:** Geophysics is inherently multimodal (seismic data, well logs, maps). Failing to integrate or reference the need to analyze different data types in the prompt limits the LLM's usefulness.

## URL
[https://pubs.geoscienceworld.org/seg/tle/article/44/2/142/651624/Intelligent-seismic-workflows-The-power-of](https://pubs.geoscienceworld.org/seg/tle/article/44/2/142/651624/Intelligent-seismic-workflows-The-power-of)
