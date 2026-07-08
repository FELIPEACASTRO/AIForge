# Cryptocurrency Analysis Prompts

## Description
The **Cryptocurrency Analysis Prompts** technique consists of structuring input commands for Large Language Models (LLMs) with the goal of obtaining detailed analyses, market forecasts, asset comparisons, and insights into trading strategies in the volatile crypto-asset market. These prompts are designed to turn AI into a research and analysis assistant, capable of processing large volumes of data (news, *on-chain* data, technical and fundamental analysis) and synthesizing complex information into actionable reports. The effectiveness lies in the technical precision of the language used and in requesting structured output formats, which facilitate decision-making for traders and investors.

## Examples
```
1. **Detailed Technical Analysis:** "Act as a senior technical analyst. Analyze the BTC/USDT pair on the 4-hour chart. Based on the RSI, MACD, and Bollinger Bands indicators, what is the probability of a bullish or bearish breakout in the next 24 hours? Present the analysis in a table format with the key support and resistance levels."

2. **Tokenomics Comparison:** "Compare the *tokenomics* of [Token A] and [Token B]. Include metrics such as total supply, inflation/deflation rate, *staking* mechanism, and initial distribution. Conclude which asset presents a more sustainable economic model in the long term."

3. **Market Sentiment Analysis:** "Monitor the last 24 hours of news and social media about [Asset Name]. Classify the overall sentiment as 'Extremely Optimistic', 'Optimistic', 'Neutral', 'Pessimistic', or 'Extremely Pessimistic'. Justify the classification with three main headlines or trends."

4. **Whitepaper Summary:** "Summarize the *whitepaper* of [Project Name] in 5 main points, focusing on the solution the project offers, the consensus technology, and the roadmap for the next 12 months. Use accessible language for a beginner investor."

5. **Trading Strategy Generation:** "Create a *swing trade* strategy for Ethereum (ETH) based on a 50-period exponential moving average (EMA). Define the entry point, the *stop-loss*, and the *take-profit* in percentage terms. Present the strategy in a clear step-by-step format."

6. **On-Chain Data Interpretation:** "Explain what the recent increase in the number of active addresses and the decrease in the balance on exchanges of [Asset Name] suggests about investor behavior. What is the implication for the price in the short term?"
```

## Best Practices
**1. Technical Specificity:** Use technical market terms (e.g., *on-chain data*, *tokenomics*, *moving average convergence divergence - MACD*) to guide the AI toward deeper and less generic analyses. **2. Function/Role Structure:** Begin the prompt by defining the AI's role (e.g., "You are a senior crypto-asset *research* analyst...") to establish the context and tone of the response. **3. Structured Output Format:** Request the output in structured formats (e.g., Markdown table, JSON, list of pros and cons) to facilitate reading and integration with other tools. **4. Providing Data:** Whenever possible, provide the raw data or the specific context (e.g., "Analyze the whitepaper of [Token Name]...") to prevent the AI from hallucinating or using outdated data. **5. Human Verification (Cross-Check):** Never use the AI output as the sole basis for financial decisions. The information should be treated as a *copilot* and always verified against real-time data sources and human analysis.

## Use Cases
**1. Strategy Development:** Create and refine trading strategies (e.g., *scalping*, *swing trade*, long-term investing) based on technical and fundamental indicators. **2. Rapid Due Diligence:** Summarize and analyze *whitepapers*, team profiles, and *tokenomics* models of new projects in minutes. **3. Sentiment Analysis:** Monitor and interpret community and media sentiment about a specific asset, helping to identify peaks of euphoria or panic. **4. Education and Simulation:** Use AI to explain complex concepts (e.g., *liquidity pools*, *impermanent loss*, *sharding*) or simulate market scenarios (e.g., "What happens if Ethereum migrates to PoS?"). **5. Content Generation:** Create research reports, blog articles, or *newsletters* about market trends and asset analyses.

## Pitfalls
**1. Blind Trust (Over-Reliance):** Treating the AI output as an investment oracle. AI does not have access to real-time data (unless explicitly connected) and cannot predict unexpected events (black swans). **2. Vague Prompts:** Using prompts like "What will happen to Bitcoin?" results in generic and useless answers. The lack of technical specificity is the most common error. **3. Ignoring the Data Source:** Assuming that the AI is using the most recent or correct data. It is crucial to verify whether the LLM has access to up-to-date market data (via plugins or APIs) before trusting price analyses. **4. Data Hallucination:** AI can invent statistics, dates, or market events. Always request the citation of the source for critical data. **5. Failure to Define the Role:** Not defining the AI's role (e.g., analyst, trader, developer) can lead to responses that mix different perspectives or that are not focused on the financial objective.

## URL
[https://www.geeksforgeeks.org/websites-apps/chatgpt-prompts-for-crypto-traders/](https://www.geeksforgeeks.org/websites-apps/chatgpt-prompts-for-crypto-traders/)
