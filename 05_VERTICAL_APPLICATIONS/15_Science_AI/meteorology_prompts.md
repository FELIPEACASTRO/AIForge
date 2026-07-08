# Meteorology Prompts

## Description
Meteorology Prompts are advanced Prompt Engineering techniques that use Large Language Models (LLMs) to process large volumes of raw meteorological data (such as numerical model data in CSV or JSON format) and generate structured, concise weather forecast summaries tailored to a specific target audience. The technique involves defining a specialized role for the LLM (e.g., hydrometeorologist), imposing strict format and content constraints (e.g., word limit, risk tone, use of markdown), and integrating external data to ensure that the output is factually accurate and operationally useful. The main goal is to transform complex and technical data into clear and actionable communication, such as flood alerts or flight reports [1].

## Examples
```
**Example 1: Flood Forecast Summary (Based on [1])**

```
**Role:** You are a UK hydrometeorologist preparing a flood-focused forecast for **{region_name}**.
**Tone:** Formal, concise, risk-aware (flood guidance).
**Constraint:** ≤ 300 words (weather + coastal/tidal).
**Input Data:**
## 2 Input Data (for model use only)
- **Weather model:** ECMWF IFS.
{csv_data_with_rain_and_wind}
- **Tidal series (proxy):** {tidal_txt_with_tide_cycle}
**Instructions:**
1. Start with 3 sentences: Coverage period, base model, and a concise title (##### Title: ...).
2. Describe the features by broad regions, not by specific locations.
3. Main focus on moderate/heavy rainfall. Omit general temperature/winds unless they are critical.
4. Include ###### Coastal/Tidal Information. Highlight strong coastal winds coinciding with high tides.
5. **Close with (choose 1):** "Nothing currently indicates that it could exacerbate or create new flood risk." OR "Possibility of increased flood risk based on the forecast conditions."
```

**Example 2: Flight Report for Pilots**

```
**Role:** You are an Aeronautical Meteorology Officer.
**Audience:** General aviation pilots (VFR).
**Location:** Congonhas Airport (SBMT) and 50nm area.
**Input Data:** METAR/TAF for SBMT, SBGR, SBKP. Satellite image (link/description).
**Instructions:**
1. **Format:** 150-word summary. Use standard aeronautical abbreviations (e.g., OVC, BKN, VMC, IFR).
2. **Focus:** Ceiling, Visibility, Wind (direction and gusts), and Thunderstorm Risk (TS).
3. **Output:** Start with "METAR/TAF SUMMARY SBMT/50NM". Highlight IFR or marginal VFR conditions.
4. **Hazard:** If TS is forecast, include an alert in uppercase: "ALERT: RISK OF ISOLATED THUNDERSTORMS BETWEEN 1500Z AND 1800Z."
```

**Example 3: Heat Wave Alert for Public Health**

```
**Role:** You are a climate risk analyst for the Health Department.
**Location:** City of São Paulo.
**Input Data:** Maximum temperature and relative humidity forecast for the next 5 days. Alert threshold: 32°C and RH < 30%.
**Instructions:**
1. **Tone:** Informative and preventive.
2. **Output:** Create a 100-word bulletin.
3. **Content:** Indicate the days on which the alert threshold will be exceeded. Provide 3 public health recommendations (e.g., hydration, peak hours).
4. **Format:** Use a simple table for the alert days.
```

**Example 4: Analysis of Conditions for Agriculture**

```
**Role:** Agronomic Consultant.
**Crop:** Soybean (grain filling stage).
**Location:** Western Region of Bahia.
**Input Data:** Accumulated precipitation forecast (mm) and evapotranspiration (ET) for 7 days.
**Instructions:**
1. **Analysis:** Evaluate the water balance.
2. **Output:** One paragraph (max. 80 words) on the suitability of the conditions for the crop stage.
3. **Recommendation:** A specific recommendation (e.g., need for supplementary irrigation or risk of fungal diseases due to humidity).
```

**Example 5: Surf Conditions Summary**

```
**Role:** Wave Forecasting Specialist.
**Location:** Maresias Beach, SP.
**Input Data:** Wave height (m), period (s), swell direction, and wind (direction/speed) forecast for the next 24h (6h intervals).
**Instructions:**
1. **Tone:** Enthusiastic and technical.
2. **Output:** A summary per period (Morning, Afternoon, Evening).
3. **Focus:** Wave quality for surfing (e.g., "Classic Conditions", "Choppy sea").
4. **Detail:** Include the best tide window for surfing.
```
```

## Best Practices
**Role and Tone Definition:** Assign the LLM a specific role (e.g., "UK hydrometeorologist") and a tone (e.g., "formal, concise, risk-aware") to ensure appropriate terminology and style [1]. **Strict Output Constraints:** Use word limits (e.g., "≤ 300 words") and formatting instructions (e.g., "Use markdown ###### for titles") to force conciseness and structure [1]. **Temporal Contextualization:** Provide explicit start and end dates for the forecast period (e.g., "Today is {today_day} {today_date}. Final day: {final_day_name}") to anchor the LLM in real time [1]. **Focus on Relevance:** Prioritize critical information for the use case (e.g., moderate/heavy rain for flood risk) and omit non-essential details (e.g., temperature or general winds unless they are critical) [1]. **Clear Data Structure:** Present the input data (e.g., model data CSV, list of sampled locations, tidal data) in clearly labeled sections for "model use only" [1]. **Avoid Hyperbole and Extrapolation:** Instruct the LLM to avoid overly confident spatial forecasts or extrapolations (e.g., "Avoid overly confident spatial forecasts") when the input data is point-based [1].

## Use Cases
**Flood Risk Forecasting:** Generation of weather forecast summaries focused on parameters relevant to flooding (e.g., precipitation rates, tides) for emergency response teams [1]. **Aeronautical Reports:** Creation of concise and standardized weather bulletins for pilots, focusing on visibility, ceiling, and thunderstorm risk. **Public Health Bulletins:** Generation of heat wave, extreme cold, or air quality alerts, with public health recommendations. **Agriculture Support:** Analysis of climatic conditions (rain, evapotranspiration, temperature) to provide irrigation, planting, or harvesting recommendations. **Media Communication:** Creation of scripts or texts for weather news, focusing on significant events (e.g., named storms). **Analysis of Maritime Conditions:** Generation of forecasts for nautical activities, fishing, or surfing, detailing wave height, swell direction, and winds.

## Pitfalls
**Data Hallucination:** The LLM may "hallucinate" meteorological data or extrapolate forecasts in an overly confident way if the input data is ambiguous or insufficient. This is especially dangerous in spatial forecasts based on point data [1]. **Loss of Hydrological Context:** The LLM, without river or soil data, may make unfounded claims about the real impact of floods. It is crucial to restrict the LLM to meteorological information only (e.g., "You have no information about current or forecast hydrology, so do not mention it") [1]. **Constraint Violation:** LLMs may ignore format constraints (e.g., word limit, use of markdown) or include generic calls to action, unless the instructions are extremely explicit and repeated [1]. **Inconsistent Data Interpretation:** Unless the prompt includes reference tables (e.g., Beaufort Scale for winds) or defines clear thresholds (e.g., "Big tide if >7.0m"), the LLM may interpret numerical data inconsistently [1]. **Lack of Uncertainty Management:** Most meteorology prompts focus on deterministic data, which can lead to summaries that do not communicate forecast uncertainty, especially over longer time horizons [1].

## URL
[https://medium.com/@rob_cowling/%EF%B8%8Fhow-to-prompt-llms-to-craft-weather-summaries-for-flood-forecasting-cd9936828daf](https://medium.com/@rob_cowling/%EF%B8%8Fhow-to-prompt-llms-to-craft-weather-summaries-for-flood-forecasting-cd9936828daf)
