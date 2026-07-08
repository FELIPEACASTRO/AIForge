# Time Series Forecasting Prompts

## Description
The *Time Series Forecasting Prompts* technique refers to the engineering of structured instructions for Large Language Models (LLMs) with the goal of performing complex time series analysis and forecasting tasks. Instead of relying exclusively on traditional statistical or machine learning models (such as ARIMA, Prophet, or specialized neural networks), this approach leverages the reasoning and context-comprehension capabilities of LLMs to identify patterns, trends, seasonality, and anomalies in time series data. Success lies in careful formatting of the input data (often as text sequences or "patches") and in a clear definition of the task, the constraints, and the desired output format.

## Examples
```
### 1. Zero-Shot Forecasting with Context

**Objective:** Obtain a quick baseline forecast for quarterly sales.

```
## System
You are an expert in time series analysis focused on the retail sector.
Your task is to identify patterns, trends, and seasonality to forecast accurately.

## User
Analyze this quarterly sales time series: [120, 150, 130, 180, 140, 170, 150, 200, 160, 190, 170, 220]
- **Dataset:** Quarterly sales of an electronics store.
- **Frequency:** Quarterly
- **Features:** Sales value only.
- **Horizon:** 4 quarters ahead.

## Task
1. Forecast the next 4 quarters.
2. Point out the main seasonal or trend patterns observed.

## Constraints
- **Output:** Markdown list with the forecasts (0 decimal places).
- Add an explanation of up to 40 words about the driving factors.

## Evaluation Hook
End with: "Confidence: X/10. Assumptions: [...]".
```

### 2. Patch-Based Prompting (PatchInstruct)

**Objective:** Forecast the hourly temperature using recent data segments.

```
## System
You are a time-series forecasting expert in meteorology and sequential modeling.
Input: overlapping patches of size 3, reverse chronological (most recent first).

## User
Patches:
- Patch 1: [25.5, 25.4, 25.6]
- Patch 2: [25.8, 25.5, 25.4]
- Patch 3: [26.1, 25.8, 25.5]
...
- Patch N: [22.3, 22.1, 22.0]

## Task
1. Forecast next 3 values.
2. In ≤40 words, explain the recent trend.

## Constraints
- Output: Markdown list, 2 decimals.
- Ensure predictions align with the observed trend.

## Example
- Input: [20.0, 20.1, 20.2] → Output: [20.3, 20.4, 20.5].

## Evaluation Hook
Add: "Confidence: X/10. Assumptions: [...]".
```

### 3. Stationarity Testing and Transformation

**Objective:** Verify whether a stock price series is stationary and obtain code to transform it if it is not.

```
## System
You are a quantitative time series analyst.

## User
- **Dataset:** Daily closing price of stock XYZ over 5 years.
- **Frequency:** Daily
- **Suspected trend:** Non-linear with varying volatility.
- **Business context:** Financial risk analysis.

## Task
1. Explain how to test for stationarity using:
   - Augmented Dickey-Fuller (ADF)
   - KPSS
   - Visual inspection of plots
2. If it is not stationary, suggest the appropriate transformations (differencing, log, etc.).
3. Provide Python code (using `statsmodels` and `pandas`) to perform the tests and transformations.

## Constraints
- Keep the explanation to a maximum of 120 words.
- The code must be ready to copy and paste.

## Evaluation Hook
End with: "Confidence: X/10. Assumptions: [...]".
```

### 4. Autocorrelation Analysis (ACF/PACF)

**Objective:** Identify significant lags in network traffic data for feature engineering.

```
## System
You are a time series expert for network monitoring.

## User
- **Dataset:** Network traffic volume (in GB) measured every 5 minutes.
- **Size:** 2016 observations (1 week).
- **Frequency:** 5 minutes.
- **Raw sample:** [1.2, 1.3, 1.2, 1.5, ...]

## Task
1. Provide Python code to generate ACF and PACF plots.
2. Explain how to interpret the plots to identify:
   - Autoregression (AR) lags
   - Moving Average (MA) components
   - Seasonal patterns
3. Recommend lag features based on the significant lags.
4. Show the Python code to create these features, handling missing values.

## Constraints
- Output: Explanation of up to 150 words + Python snippets.
- Use `statsmodels` and `pandas`.

## Evaluation Hook
End with: "Confidence: X/10. Main lags flagged: [list lags]".
```

### 5. Anomaly Detection

**Objective:** Identify anomalous readings in the sensors of an industrial machine.

```
## System
You are an expert in anomaly detection in IoT sensor time series data.

## User
- **Time series:** [20.1, 20.2, 20.0, 19.9, 20.1, 35.5, 20.3, ...]
- **Context:** Temperature reading of a motor operating continuously.
- **Expected limits:** 15°C to 30°C.

## Task
1. Identify any data points that appear to be anomalies.
2. For each anomaly, provide:
   - The anomalous value.
   - The index (or timestamp) of the anomaly.
   - A brief explanation of why it is considered an anomaly (e.g., "sudden spike", "outside expected limits").
3. Suggest a strategy to handle/remove the anomalies before modeling.

## Constraints
- Format the output as a Markdown table.

## Evaluation Hook
End with: "Detection confidence: X/10. Assumptions: [e.g., 'Anomalies are single points and not regime changes']".
```
```

## Best Practices
1. **Patch-Based Prompting (PatchInstruct):** Divide the time series into overlapping "patches" (segments) and feed them to the LLM. This reduces token usage, preserves interpretability, and allows the model to detect short-term temporal patterns efficiently.
2. **Zero-Shot with Context:** Provide the LLM with a clear description of the dataset (domain, frequency, forecast horizon) so that it can establish a forecasting baseline without additional training.
3. **Neighbor-Augmented Prompting:** Include "neighboring" or correlated time series in the prompt to help the LLM identify shared structures and patterns, refining the forecast for the target series.
4. **Detailed Prompt Structure:** Use clear sections such as `## System`, `## User`, `## Task`, `## Constraints`, and `## Evaluation Hook` to guide the model precisely.
5. **Code Inclusion:** Ask the LLM to generate Python code (for example, with `statsmodels` and `pandas`) for tasks such as stationarity testing (ADF/KPSS), autocorrelation analysis (ACF/PACF), and seasonal decomposition (STL).

## Use Cases
1. **Quick Baseline Forecasting:** Generation of initial (zero-shot) forecasts to establish a benchmark.
2. **Exploratory Data Analysis (EDA):** Identification of stationarity, trends, seasonality, and anomalies.
3. **Feature Engineering:** Generation of code to create *lag* features, moving windows, and cyclical components.
4. **Series Decomposition:** Separation of the series into trend, seasonality, and residual components for business *insights*.
5. **Domain-Specific Forecasting:** Application in meteorology, traffic, sales, finance, and other domains with sequential data.

## Pitfalls
1. **Context Limitation (Token Limit):** Attempting to feed very long time series directly into the prompt, exceeding the token limit and resulting in truncation or loss of information.
2. **Hallucinations and Lack of Statistical Rigor:** LLMs may generate plausible but statistically incorrect forecasts, or ones without adherence to formal hypothesis tests (such as stationarity).
3. **Inadequate Data Format:** Failing to format the data in a structured way (e.g., as *patches* or clear lists), forcing the LLM to process raw data inefficiently.
4. **Ignoring the Temporal Structure:** Treating the time series as an ordinary dataset, without instructing the LLM about the temporal dependency, frequency, and forecast horizon.
5. **Overconfidence:** Blindly trusting the LLM's forecast without rigorous validation using error metrics (MAE, RMSE) and *backtesting* tests.

## URL
[https://towardsdatascience.com/prompt-engineering-for-time-series-analysis-with-large-language-models/](https://towardsdatascience.com/prompt-engineering-for-time-series-analysis-with-large-language-models/)
