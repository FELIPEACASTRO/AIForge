# Trading Strategy Prompts

## Description
**Trading Strategy Prompts** are structured, detailed instructions provided to Large Language Models (LLMs) to assist in the creation, optimization, backtesting, and analysis of trading strategies in financial markets (stocks, forex, cryptocurrencies, commodities). The technique is based on advanced prompt engineering, such as **Role Prompting** and **Chain-of-Thought**, to force the AI to operate as a quantitative analyst or algorithm developer. The main goal is not to predict the market, but rather to automate the generation of trading logic, backtesting code (in Python, Pine Script, MQL5, etc.), and risk and performance analysis. The effectiveness lies in the ability to provide specific financial context, market data (or the structure for it), and clear risk constraints, turning the AI into a powerful tool for research and development of algorithmic strategies.

## Examples
```
**1. Code Generation for a Moving Average Strategy**

```
**Role:** Act as a senior trading algorithm developer.
**Task:** Generate the complete Python code (using the `pandas` and `backtesting.py` libraries) for a Moving Average crossover strategy.
**Logic:**
1. Fast Moving Average (SMA) of 20 periods.
2. Slow Moving Average (SMA) of 50 periods.
3. **Buy Signal:** When the Fast SMA crosses above the Slow SMA.
4. **Sell Signal:** When the Fast SMA crosses below the Slow SMA.
**Expected Output:** The complete Python code, including the strategy class and the backtesting function.
```

**2. Parameter Optimization and Risk Analysis**

```
**Role:** Act as a Quantitative Risk Strategist.
**Context:** The strategy is a "Volatility Breakout" on Bitcoin (BTC/USD) on a 4-hour chart.
**Current Parameters:** ATR (Average True Range) window = 14; Stop Loss multiplier = 2.5.
**Task:** Suggest 3 sets of optimized parameters (ATR and Multiplier) that aim to maximize the Profit Factor while keeping the Max Drawdown below 15%.
**Expected Output:** A comparative table with the 3 parameter sets, the estimated Profit Factor, and the Max Drawdown.
```

**3. Creating a Risk Management System**

```
**Role:** Act as a Portfolio Manager.
**Task:** Develop a risk management system (Money Management) for a $50,000 trading account.
**Rules:**
1. Maximum risk per trade: 1% of total capital.
2. Maximum daily risk: 3% of total capital.
3. Calculate the position size (in units) for a trade where the Stop Loss is 50 pips away.
**Expected Output:** The position size calculation and a summary of the risk rules in list format.
```

**4. Indicator-Based Technical Analysis**

```
**Role:** Act as a Market Technical Analyst.
**Asset:** Tesla (TSLA) stock.
**Interval:** Daily.
**Indicators:** Bollinger Bands (20, 2) and MACD (12, 26, 9).
**Task:** Analyze TSLA's current situation based on the given indicators.
**Rationale:** Describe what each indicator is signaling (e.g., price touching the lower band, MACD crossing the signal line).
**Conclusion:** Provide a 3-point summary of the trading bias (bullish, bearish, or neutral) and the nearest support/resistance level.
```

**5. Backtesting a Mean Reversion Strategy**

```
**Role:** Act as a Backtesting Engineer.
**Strategy:** Mean Reversion on the S&P 500 (SPY).
**Logic:** Buy when the closing price is 2 standard deviations below the 20-day Moving Average. Sell when the price returns to the Moving Average.
**Task:** Describe the 5 main backtesting challenges for this strategy (e.g., transaction cost, slippage, volatility).
**Expected Output:** A numbered list of the challenges and a suggestion on how to mitigate each one.
```

**6. Prompt for News Sentiment Analysis**

```
**Role:** Act as a Market Sentiment Analyst.
**Input:** [Insert here the text of a recent news item about the European Central Bank (ECB)].
**Task:** Analyze the text and classify the overall sentiment (Bullish, Bearish, Neutral) for the EUR/USD pair.
**Expected Output:**
1. **Sentiment:** [Classification]
2. **Justification:** [Concise explanation based on the text]
3. **Trading Implication:** [Suggested short-term trading action]
```

**7. Prompt for Creating a Trading Plan**

```
**Role:** Act as a Personal Trading Coach.
**Task:** Create a structured trading plan for a beginner trader focused on swing trading stocks.
**Required Sections:**
1. Goal Setting (Realistic)
2. Entry and Exit Rules (Generic)
3. Risk Management Rules (Stop Loss and Position Size)
4. Daily Analysis Routine
5. Trading Psychology Rules (e.g., Do not trade after 3 consecutive losses)
**Expected Output:** The complete trading plan in list or bullet-point format.
```
```

## Best Practices
**1. Define the Role (Role Prompting):** Start the prompt by instructing the AI to act as a "Senior Quantitative Trading Strategist", "Market Analyst with 10 Years of Experience", or "High-Frequency Algorithm Developer". This forces the model to use more specialized vocabulary and reasoning.
**2. Structure and Format:** Use XML tags (`<INPUT>`, `<OUTPUT>`, `<RATIONALE>`) or clear formatting to delimit the sections of your prompt and the expected response. This improves the AI's accuracy and processing capability.
**3. Be Specific and Contextual:** Instead of asking for a "trading strategy", ask for a "mean reversion strategy for the EUR/USD pair on a 1-hour chart, using the RSI (14) indicator with overbought/oversold levels at 70/30".
**4. Chain-of-Thought:** Ask the AI to first detail the **rationale** behind the strategy, then the **code**, and finally the **risk parameters**. This ensures the model "thinks" before coding, reducing hallucinations.
**5. Provide Example Data:** If possible, include a small snippet of historical data (in CSV or JSON format) so the AI can base its analysis on real context, even if only for the purpose of demonstrating the logic.
**6. Focus on Process, Not Prediction:** Use prompts to optimize the process (backtesting, risk management, indicator analysis) instead of trying to obtain direct predictions of the future price, which are inherently unreliable.

## Use Cases
**1. Development of Algorithmic Strategies:** Generating functional code (Python, Pine Script) for new trading strategies, such as arbitrage, momentum, or mean reversion.
**2. Parameter Optimization:** Identifying optimal parameters (e.g., Moving Average periods, RSI levels) to maximize the performance of an existing strategy under different market conditions.
**3. Backtesting and Simulation:** Creating backtesting frameworks and analyzing performance metrics (Sharpe Ratio, Drawdown, Profit Factor) to validate the robustness of the strategy.
**4. Risk Management:** Developing capital management rules and calculating position size to limit risk exposure.
**5. Sentiment Analysis:** Processing large volumes of financial news or social media data to extract a sentiment bias (bullish/bearish) that can be integrated into the strategy.
**6. Education and Learning:** Generating detailed explanations of complex trading concepts, such as Elliott Wave Theory or options pricing models.
**7. Creating Trading Plans:** Structuring personal trading plans and market analysis routines.

## Pitfalls
**1. Data Hallucination:** The AI may invent historical data, backtesting results, or technical indicators. **Mitigation:** Use the AI only to generate the logic and the code; run the backtesting on real trading platforms with verified data.
**2. Overfitting:** Creating prompts that result in strategies excessively optimized for a specific dataset. **Mitigation:** Ask the AI to include a "Strategy Robustness" section and to suggest Out-of-Sample Testing.
**3. Lack of Financial Context:** Using generic prompts without defining the asset, the timeframe, and the market conditions. **Mitigation:** Always use **Role Prompting** and provide as much technical and fundamental context as possible.
**4. Confusing Prediction with Analysis:** Asking the AI to "predict tomorrow's price". **Mitigation:** Focus on prompts that analyze the **probability** of a move based on predefined conditions (e.g., "What is the probability of an upward move if the RSI is below 30?").
**5. Ignoring Risk:** Focusing only on profit and neglecting risk management in the prompt. **Mitigation:** Make risk management (Stop Loss, Position Size, Max Drawdown) a mandatory component of the prompt's output.

## URL
[https://roguequant.substack.com/p/prompt-engineering-for-traders-how](https://roguequant.substack.com/p/prompt-engineering-for-traders-how)
