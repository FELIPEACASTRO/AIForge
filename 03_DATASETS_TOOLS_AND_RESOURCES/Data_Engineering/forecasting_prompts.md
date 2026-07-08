# Forecasting Prompts

## Description
**Forecasting Prompts** is a prompt engineering technique that uses Large Language Models (LLMs) to perform **Time Series Forecasting (TSF)** tasks. Instead of relying exclusively on statistical models or traditional machine learning architectures, this approach translates the time series data and the forecasting problem into a format the LLM can understand, usually structured text or token sequences.

The most notable innovation is the concept of **Patch-Based Prompting**, as exemplified by the **PatchInstruct** framework [1]. This technique divides the time series into smaller segments, called "patches," that encapsulate local temporal patterns. These patches are then tokenized and presented to the LLM along with natural-language instructions, allowing the model to use its pattern-recognition and sequence-modeling capabilities to generate future forecasts [1]. The main benefit is the ability to produce accurate and efficient forecasts without the need for extensive *fine-tuning* or complex architectural modifications, drastically reducing inference *overhead* [1].

## Examples
```
1.  **Sales Forecasting (Zero-Shot)**
    ```
    **Instruction:** "You are a data analyst. Forecast the next 7 days of sales based on the historical data provided. The result should be a list of 7 integers.
    **Historical Data (Daily Sales):** [120, 135, 140, 155, 160, 145, 170, 185, 190, 205, 210, 225, 230, 245]"
    ```

2.  **Traffic Forecasting (Patch-Based - Conceptual)**
    ```
    **Instruction:** "Analyze the following 'patches' of traffic data (average vehicles per hour) from the past week. Patch 1 is the Monday trend, patch 2 is the Tuesday trend, and so on. Forecast the average traffic value for next Monday's peak (Patch 8).
    **Patch 1 (Mon):** [450, 510, 620]
    **Patch 2 (Tue):** [460, 525, 635]
    **Patch 3 (Wed):** [445, 505, 615]
    **Patch 4 (Thu):** [470, 530, 640]
    **Patch 5 (Fri):** [500, 600, 750]
    **Patch 6 (Sat):** [300, 350, 400]
    **Patch 7 (Sun):** [250, 300, 320]"
    ```

3.  **Stock Price Forecasting (With Context)**
    ```
    **Instruction:** "Based on the closing prices of the last 30 days and considering that the company announced a successful new product yesterday, forecast the closing price of the stock for the next 5 days. Provide the forecast as a list of floats.
    **Closing Prices:** [50.2, 51.1, 50.8, 52.5, 53.0, 54.1, 55.5, 56.0, 55.8, 57.2, ... (20 more values)]"
    ```

4.  **Energy Demand Forecasting (With Seasonality)**
    ```
    **Instruction:** "The time series represents energy consumption (in MWh) per hour over the last 48 hours. Identify the daily seasonal pattern and forecast consumption for the next 6 hours.
    **Time Series:** [45, 42, 40, 41, 48, 55, 65, 72, 70, 68, 65, 60, 55, 50, 48, 45, 42, 40, 41, 48, 55, 65, 72, 70, 68, 65, 60, 55, 50, 48, 45, 42, 40, 41, 48, 55, 65, 72, 70, 68, 65, 60, 55, 50, 48, 45, 42, 40]"
    ```

5.  **Weather Forecasting (Simplified Multivariate)**
    ```
    **Instruction:** "Forecast the maximum temperature (Tmax) and precipitation (P) for the next 3 days. Use the daily data from the last 10 observations.
    **Data:**
    Day 1: Tmax=25, P=0
    Day 2: Tmax=26, P=0
    Day 3: Tmax=24, P=5
    Day 4: Tmax=22, P=15
    Day 5: Tmax=23, P=10
    Day 6: Tmax=25, P=0
    Day 7: Tmax=27, P=0
    Day 8: Tmax=28, P=0
    Day 9: Tmax=26, P=2
    Day 10: Tmax=25, P=5
    **Output Format:** [Tmax Day 11, P Day 11], [Tmax Day 12, P Day 12], [Tmax Day 13, P Day 13]"
    ```

6.  **Anomaly Forecasting (Detection Instruction)**
    ```
    **Instruction:** "The time series represents the latency of a server (in ms). Forecast the next value and, more importantly, determine whether the forecast value is an anomaly (Yes/No) based on the standard deviation of the input data.
    **Latency:** [50, 52, 51, 53, 50, 54, 55, 52, 51, 50, 53, 52, 51, 50, 52, 51, 50, 53, 52, 51]"
    ```

7.  **Forecasting with Decomposition (Conceptual)**
    ```
    **Instruction:** "The time series has been decomposed into Trend, Seasonality, and Residual. Forecast the next value of the Trend and of the Seasonality, and combine them for the final forecast.
    **Trend:** [100, 105, 110, 115, 120]
    **Seasonality:** [5, -2, -3, 5, -2]
    **Residual:** [0.1, -0.5, 0.2, 0.0, -0.1]"
    ```
```

## Best Practices
*   **Efficient Tokenization (Patching):** Instead of dumping the entire time series, use techniques such as *Patch-Based Prompting* to tokenize the series into meaningful segments. This reduces token usage and lets the LLM capture short-term temporal patterns more effectively [1] [2].
*   **Structured Instruction:** Clearly define the LLM's role ("You are a data analyst..."), the forecast horizon (how many steps ahead), and the desired output format (list, JSON, single number) [2].
*   **Normalization and Scaling:** Although LLMs are robust, normalizing the input data (for example, to a range of 0 to 1 or Z-scores) can improve forecast stability and accuracy.
*   **Provide Context:** Include relevant metadata (holidays, marketing events, regulatory changes) that may influence the time series, since LLMs are excellent at incorporating textual information [2].
*   **Explicit Decomposition:** For complex time series, decompose them into components (trend, seasonality, residual) and provide these components separately to the LLM, instructing it to forecast each part and then recombine them [1].

## Use Cases
*   **Finance and Commerce:** Forecasting stock prices, trading volumes, product demand in e-commerce, and inventory management [2].
*   **Energy and Utilities:** Forecasting electricity demand, gas consumption, and renewable energy production.
*   **Logistics and Supply Chain:** Forecasting shipments, delivery delays, and warehousing capacity needs [2].
*   **Healthcare:** Forecasting disease outbreaks, hospital demand, and medication consumption.
*   **Systems Monitoring:** Forecasting server latency, CPU usage, and anomaly detection in system logs.

## Pitfalls
*   **Numerical Hallucination:** LLMs can "hallucinate" numbers or sequences that appear plausible but are not mathematically valid or consistent with the input data.
*   **Context Limitation (Token Limit):** Long time series can exceed the LLM's token limit, requiring summarization or *patching* techniques that may lead to the loss of crucial information [2].
*   **Ignoring Complex Dependencies:** LLMs may struggle to model inter-series dependencies (in multivariate forecasts) or long-term relationships without explicit instructions or auxiliary architectures [1].
*   **Format Sensitivity:** Forecast accuracy is highly sensitive to how the data is formatted and presented in the prompt. An inconsistent or ambiguous data format can lead to incorrect results.
*   **Training Bias:** The LLM may introduce biases from its general training (for example, knowledge of world events) that may not be relevant or may distort a purely data-based forecast.

## URL
[https://arxiv.org/html/2506.12953v1](https://arxiv.org/html/2506.12953v1)
