# Technical Analysis Prompts

## Description
Technical Analysis Prompts are structured and detailed instructions provided to Large Language Models (LLMs) with the goal of simulating the analysis of market data, such as prices and volumes, to identify patterns, trends, and trading signals. This prompt engineering technique is crucial in the financial sector, especially for traders and analysts seeking to automate or accelerate the interpretation of technical indicators (such as RSI, MACD, Moving Averages) and chart patterns (such as Head and Shoulders, Triangles).

The effectiveness of these prompts lies in the ability to provide the LLM with the necessary **context** (historical data, specific indicators, time horizon) and the **role** (financial analyst, quantitative trader) so that it can apply the analytical and statistical reasoning inherent to its training. The prompt should be clear, concise, and, ideally, request a structured output to facilitate integration with trading systems or reports.

## Examples
```
1.  **Multiple Indicator Analysis:**
    ```
    Act as a quantitative analyst. Analyze the asset [ASSET NAME/TICKER] based on the following data: [INSERT PRICE AND VOLUME DATA]. Calculate and interpret the RSI (14 periods) and the MACD (12, 26, 9). The RSI is at [VALUE] and the MACD is [ABOVE/BELOW] the signal line. Provide a conclusion about the momentum and trend, and suggest a trading signal (Buy, Sell, Neutral).
    ```

2.  **Chart Pattern Identification:**
    ```
    You are an expert in Japanese candlestick patterns. Given the sequence of closing prices for the last 30 days for [ASSET NAME], identify whether there is a reversal or continuation pattern (e.g., Hammer, Morning Star, Bearish Engulfing). If a pattern is found, describe it and explain its expected price implication.
    ```

3.  **Moving Average Crossover Strategy:**
    ```
    Create a trading strategy based on the crossover of the 9- and 21-period Exponential Moving Averages (EMA) for the asset [ASSET NAME]. Define the entry and exit rules: Buy when EMA(9) crosses above EMA(21). Sell when EMA(9) crosses below EMA(21). Evaluate the effectiveness of this strategy over the last 6 months (based on the data provided) and suggest a percentage Stop Loss.
    ```

4.  **Volatility Analysis (Bollinger Bands):**
    ```
    Act as a risk manager. For the asset [ASSET NAME], the Bollinger Bands (20 periods, 2 standard deviations) are [WIDE/NARROW]. The current price is [ABOVE/BELOW/WITHIN] the upper/lower band. Interpret this scenario in terms of volatility and the probability of a significant price movement. What would be a high-probability trade entry in this context?
    ```

5.  **Combining Technical Analysis and Sentiment:**
    ```
    You are a market analyst. The asset [ASSET NAME] is showing an "Ascending Triangle" pattern (bullish continuation pattern). However, sentiment analysis on social media (data provided) indicates growing pessimism. Synthesize these two conflicting pieces of information and provide a balanced trading recommendation, justifying which factor (Technical or Sentiment) should carry greater weight at the moment.
    ```

6.  **Calculation and Interpretation of Fibonacci Levels:**
    ```
    The asset [ASSET NAME] rose from [MINIMUM PRICE] to [MAXIMUM PRICE]. Calculate the Fibonacci retracement levels of 38.2%, 50%, and 61.8%. If the price is currently at the 50% level, describe the importance of this level as support/resistance and what the next price zone to watch is.
    ```

7.  **Refinement Prompt (Chain-of-Thought):**
    ```
    Step 1: Analyze the chart of [ASSET NAME] and identify the primary trend (short, medium, and long term).
    Step 2: Calculate the ADX (Average Directional Index) to measure the strength of that trend.
    Step 3: Based on the trend and the strength of the ADX, generate a 100-word report on the health of the asset.
    ```
```

## Best Practices
*   **Define the Role (Role-Playing):** Begin the prompt by defining the LLM as a "Financial Analyst", "Quantitative Trader", or "Risk Specialist". This directs the tone and focus of the response.
*   **Provide Structured Data:** Instead of just describing, provide price and volume data in a structured format (table, CSV, or list of key points) so that the LLM can process them accurately.
*   **Specify Indicators and Parameters:** Explicitly mention which technical indicators should be used (e.g., 200 EMA, 14 RSI, MACD with default settings) and the time horizon (daily, weekly, 4 hours).
*   **Request Justification (Chain-of-Thought):** Ask the LLM to detail the reasoning behind the conclusion (e.g., "Explain why an RSI of 75 indicates overbought conditions before giving the sell signal"). This increases transparency and reliability.
*   **Define the Output Format:** Request the output in a specific format (e.g., "Provide the answer in a table with columns: Indicator, Value, Interpretation, Suggested Action").

## Use Cases
*   **Trading Signal Generation:** Creation of automated Buy/Sell alerts based on the interpretation of multiple indicators.
*   **Strategy Backtesting:** Rapid simulation of the performance of a trading strategy on historical data without the need for complex coding.
*   **Education and Training:** Explanation of complex technical analysis concepts (e.g., Elliott Waves, Dow Theory) in a simplified way with practical examples.
*   **Market Reports:** Generation of daily or weekly summaries about the technical situation of an asset portfolio.
*   **Risk Analysis:** Identification of critical support and resistance levels (using Pivots or Fibonacci) to set Stop Loss and Take Profit orders.

## Pitfalls
*   **Data Hallucination:** The LLM may "invent" price data or indicator values if they are not explicitly provided. **Always provide the input data.**
*   **Overgeneralization:** Requesting a technical analysis without specifying the asset or the time horizon will result in a generic and useless response.
*   **Confusing Technical Analysis with Fundamental Analysis:** Mixing the two types of analysis in the same prompt without a clear role can lead to confusing conclusions. Keep the focus strictly on price/volume data, unless the combination is intentional and well defined.
*   **Blind Dependence:** Treating the LLM output as a final trading signal. AI analysis should be a support tool, not a substitute for human decision-making and risk management.
*   **Absence of Volatility Context:** Failing to consider the market context (high or low volatility) when interpreting indicators can lead to false signals.

## URL
[https://blog.galaxy.ai/chatgpt-prompts-for-trading](https://blog.galaxy.ai/chatgpt-prompts-for-trading)
