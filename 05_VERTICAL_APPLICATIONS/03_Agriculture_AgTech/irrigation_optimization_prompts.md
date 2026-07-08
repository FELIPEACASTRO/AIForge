# Irrigation Optimization Prompts

## Description
Irrigation Optimization Prompts are structured, contextual instructions provided to Large Language Models (LLMs) to analyze real-time and historical agricultural data (such as soil moisture, weather forecasts, crop type, growth stage, and IoT sensor data) and generate precise, actionable recommendations on when, where, and how much to irrigate. This prompting technique falls under **Precision Agriculture**, where the LLM acts as an intelligent decision-support assistant, translating complex data into natural language and specific commands for automated irrigation systems. The effectiveness of these prompts depends on their ability to integrate with APIs and **Function Calling** capabilities to access and process external data in real time.

## Examples
```
1.  **Data Analysis and Recommendation Prompt:**
    ```
    Act as a precision irrigation specialist. Analyze the following data:
    - Crop: Corn (V6 Stage)
    - Location: Fazenda Esperança, Coordinates: [34.0522, -118.2437]
    - Soil Moisture (Root Zone): 45% (Critical Threshold: 40%, Field Capacity: 65%)
    - Weather Forecast (Next 48h): 0% chance of rain, ETc (Crop Evapotranspiration): 6 mm/day
    - Irrigation System: Center Pivot (Efficiency: 85%)
    
    Based on the need to replenish the consumed water and keep the moisture above the critical threshold, calculate the required water depth (in mm) and provide the irrigation recommendation in JSON command format for the system.
    ```

2.  **System Diagnosis and Adjustment Prompt:**
    ```
    The Section 3 irrigation pump failed unexpectedly. The monitoring system indicates that the soil moisture in the root zone of the Soybean crop (R2 Stage) has dropped to 38%. The ambient temperature is 35°C.
    
    What is the immediate risk to the crop? Propose a 72-hour contingency plan, including redistributing irrigation from Sections 1 and 2 to compensate for the failure, and the extra water depth (in mm) that should be applied to those sections to mitigate the water stress in Section 3.
    ```

3.  **Weekly Schedule Optimization Prompt:**
    ```
    Generate an optimized irrigation schedule for the next week (7 days) for the Wheat crop (Tillering Stage).
    - Soil Type: Clayey (High retention)
    - Current Moisture: 60%
    - Rain Forecast: 10mm on Day 4
    - Average Daily ETc: 5.5 mm
    
    The goal is to keep the moisture between 55% and 70%. Present the day-by-day schedule, indicating the water depth (mm) to be applied or 'None'.
    ```

4.  **Scenario Simulation Prompt:**
    ```
    Simulate the impact of a heat wave (average temperature of 40°C, ETc of 8 mm/day) over 5 consecutive days on the Lettuce crop (Heading Stage).
    
    If irrigation is maintained on the standard schedule of 4 mm every 2 days, what will the water stress level (in percentage of depletion) be at the end of the period? What would be the ideal water depth to avoid the stress?
    ```

5.  **Sensor Data Interpretation Prompt:**
    ```
    Interpret the following soil moisture sensor (TDR) dataset and generate a recommendation.
    - Sensor 1 (15cm): 52%
    - Sensor 2 (30cm): 48%
    - Sensor 3 (45cm): 40%
    - Crop: Grape (Ripening Stage)
    - Requirement: Keep the 30-45cm zone under mild stress (40-45%) to optimize fruit quality.
    
    Is the current irrigation adequate? If not, what adjustment (percentage increase/decrease in the depth) do you suggest?
    ```
```

## Best Practices
*   **Clear Prompt Structure:** Define the **role** of the LLM (e.g., "Agricultural Hydrology Specialist"), provide the **context** (crop, soil, system), and include **input data** (moisture, weather, ETc) in an organized way (tables, lists, or JSON).
*   **Real-Time Data Integration:** The prompt should be designed to trigger functions (Function Calling) that fetch data from external APIs (IoT sensors, weather services) before generating the response.
*   **Actionable and Formatted Output:** Specify the desired output format (e.g., JSON, XML, or a system command) so that the response can be directly consumed by an automated irrigation system.
*   **Definition of Critical Thresholds:** Include in the prompt the water stress limits and field capacity specific to the crop and soil, allowing the LLM to make precise calculations.
*   **Iteration and Refinement:** Use follow-up prompts to refine the recommendations (e.g., "How does this recommendation change if the pivot efficiency is 90%?").

## Use Cases
*   **Dynamic Irrigation Scheduling:** Creation of irrigation schedules that automatically adjust to weather changes and crop growth stages.
*   **Water Stress Diagnosis:** Analysis of sensor data and satellite imagery (NDVI) to identify areas of the crop under stress and recommend localized interventions.
*   **Water Resource Optimization:** Calculation of the minimum water depth needed to maximize productivity, resulting in water and energy savings.
*   **Scenario Simulation:** Forecasting the impact of extreme weather events (droughts, heat waves) and planning mitigation strategies.
*   **Grower Assistance:** Providing educational, data-driven explanations to rural producers about irrigation decisions.

## Pitfalls
*   **Dependence on Inaccurate Data:** The LLM is only as good as the data it receives. Uncalibrated sensor data or incorrect weather forecasts will lead to flawed recommendations.
*   **Lack of Agronomic Context:** Not providing enough detail about the soil type, crop, root depth, and phenological stage can result in generic, ineffective recommendations.
*   **Non-Actionable Output:** If the prompt does not require a structured output format (such as JSON or a specific command), the LLM may generate descriptive text that cannot be used to automate the system.
*   **Ignoring Latency:** In real-time systems, latency in fetching data via API and in generating the LLM's response can delay the decision, impacting the crop.
*   **Overestimating the LLM's Capability:** The LLM is a reasoning and translation tool, not a hydrological model. It must be fed processed data and should not be asked to perform complex water balance calculations without the help of external tools (Function Calling).

## URL
[https://dr-arsanjani.medium.com/enhancing-agricultural-decision-making-with-function-calling-in-llms-a-vision-for-the-future-cc11960bf5d1](https://dr-arsanjani.medium.com/enhancing-agricultural-decision-making-with-function-calling-in-llms-a-vision-for-the-future-cc11960bf5d1)
