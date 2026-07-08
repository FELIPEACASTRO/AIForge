# Yield Prediction Prompts

## Description
Yield Prediction Prompts are structured instructions provided to Large Language Models (LLMs) or other AI models to request the prediction of a future outcome (the "yield") based on historical data, contextual data, and specific rules. Unlike simple text-generation prompts, these prompts require the model to perform data analysis, identify patterns, and apply *forecasting* logic to estimate metrics such as profit, agricultural production, market performance, or industrial process outcomes. The effectiveness of these prompts depends critically on the clarity of the request, the inclusion of relevant data (whether directly in the prompt or through tools/context), and, in some cases, the request for step-by-step reasoning (Chain-of-Thought) to justify the prediction. Recent research (2025) suggests that, for complex tasks such as forecasting, basic prompt refinement yields limited gains, but including references to *base rates* can bring benefits. The main focus is to convert the LLM's reasoning capability into a predictive analysis tool.

## Examples
```
**1. Agricultural Harvest Prediction (Agriculture):**
"Act as an agronomist. Predict the yield (tons per hectare) of corn for the next harvest, considering the following data:
- 3-year history: [2.5, 2.8, 2.6] t/ha
- Accumulated rainfall (last 3 months): 350mm (Historical average: 400mm)
- Average temperature: 25°C (Ideal: 24°C)
- Soil type: Clay (High nutrient content)
- Diseases/Pests: Mild presence of leafhopper.
Provide the prediction and detailed reasoning."

**2. Sales Forecast (Business/Finance):**
"You are a sales analyst. Forecast the quarterly revenue (Q4) for the product 'Software X'.
- Revenue Q1, Q2, Q3: [$1.2M, $1.5M, $1.8M]
- Competitor launch: Q3 (estimated impact of -10% on sales)
- Marketing campaign: Launch in Q4 (estimated increase of +20% in sales).
Calculate the revenue forecast and justify the impact weightings."

**3. Process Performance Prediction (Industry/Technology):**
"As a production engineer, predict the defect rate (Yield Rate) of the semiconductor assembly line for the next week.
- Average defect rate (last 4 weeks): [3.2%, 3.0%, 3.5%, 3.1%]
- Scheduled maintenance: Yes, at the beginning of the week (expected reduction of 0.5% in the defect rate).
- New batch of raw material: Lower quality (expected increase of 1.0% in the defect rate).
What is the predicted defect rate? Show the step-by-step calculation."

**4. Return on Investment Prediction (Finance):**
"Analyze the following investment in digital marketing:
- Initial investment: $50,000
- Historical Cost per Acquisition (CPA): $50
- Expected Conversion Rate (CVR): 5%
- Average Order Value (AOV): $200
Predict the Return on Investment (ROI) after 6 months, assuming 1,000 clicks per month. Present the ROI as a percentage and the expected number of customers."

**5. Web Traffic Prediction (Technology):**
"Predict the number of daily active users (DAU) for the next month, based on the following data:
- Average DAU (last 3 months): [10,000, 12,000, 15,000]
- Launch event: A major feature will be released in the middle of the month (expected increase of 30% in DAU).
- Seasonality: School holidays at the end of the month (expected reduction of 15% in DAU).
What is the DAU forecast for the next month? Use the extrapolation and adjustment method."
```

## Best Practices
**1. Structure and Context:** Always define the model's role (e.g., "You are an experienced financial analyst") and provide as much context and historical data as possible. **2. Structured Data:** Include the input data in a structured format (tables, JSON, CSV) to facilitate the model's analysis. **3. Explicit Reasoning (CoT):** Ask the model to detail the reasoning process (Chain-of-Thought) before presenting the final prediction. This increases transparency and accuracy. **4. Limits and Variables:** Specify the prediction's time horizon, the desired output metrics, and any variables or assumptions the model should consider. **5. Scenarios:** Request predictions across multiple scenarios (optimistic, pessimistic, baseline) for a more complete risk analysis.

## Use Cases
nan

## Pitfalls
**1. Lack of Context:** Providing only raw data without specifying the model's role, the prediction's objective, or the time horizon. **2. Insufficient or Irrelevant Data:** Expecting an accurate prediction with very few data points or including variables that have no proven impact on the outcome. **3. Confirmation Bias:** Framing the prompt in a way that guides the model toward a desired answer, rather than allowing an objective analysis. **4. Ignoring *Base Rates*:** Failing to provide or request that the model consider base rates or the historical frequency of the event, which is crucial for more robust predictions. **5. Metric Ambiguity:** Not clearly defining the "yield" metric (e.g., gross profit vs. net profit, total production vs. production per area).

## URL
[https://arxiv.org/abs/2506.01578](https://arxiv.org/abs/2506.01578)
