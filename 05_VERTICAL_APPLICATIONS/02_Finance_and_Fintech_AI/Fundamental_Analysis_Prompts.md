# Fundamental Analysis Prompts

## Description
**Fundamental Analysis Prompts** are structured, detailed instructions provided to Large Language Models (LLMs) with the goal of performing an in-depth assessment of the financial health, operational performance, and intrinsic value of a company or asset. Unlike generic prompts, they require the AI to adopt a **specialized persona** (e.g., equity analyst, accountant, portfolio manager) and to process specific financial data (such as 10-K reports, income statements, balance sheets, and market news) to calculate metrics, identify trends, perform sector comparisons, and formulate investment conclusions. The effectiveness of these prompts lies in their ability to mitigate the AI's tendency to "hallucinate" data by requiring verifiable sources and transparent, logical reasoning, transforming the LLM from a text generator into a quantitative and qualitative analysis assistant [1] [2]. The most appropriate subcategory is **Finance**.

## Examples
```
**Example 1: Liquidity and Solvency Analysis**
```
**Role:** Act as a senior credit analyst.
**Task:** Analyze the liquidity and solvency of "Company X" based on the Balance Sheet data for 2023 and 2024.
1. Calculate the Current Ratio and the Total Debt Ratio for both years.
2. Identify the YoY percentage change in each ratio.
3. Compare the results with the sector average (Current Ratio: 1.5x; Total Debt: 0.6x).
**Output:** Present the calculations in a table and provide a 4-sentence opinion on the company's financial health.
```

**Example 2: Profitability and Efficiency Assessment**
```
**Role:** You are a value-focused portfolio manager.
**Task:** Assess the profitability and operational efficiency of "Company Y" in the last quarter (Q3 2024).
1. Calculate the Gross Margin, Operating Margin, and Return on Equity (ROE).
2. Explain the main variations (above 5%) in operating costs compared to the previous quarter.
3. Determine whether the company is generating value for shareholders.
**Output:** Respond in the form of an executive report, highlighting the key metrics in bold and citing the data sources.
```

**Example 3: Cash Flow and Investment Analysis**
```
**Role:** Mergers and Acquisitions (M&A) Analyst.
**Task:** Analyze the Free Cash Flow (FCF) of "Company Z" over the last 5 years.
1. Calculate the FCF and FCF per Share.
2. Assess the sustainability of capital expenditures (CAPEX) relative to depreciation.
3. Project the FCF for the coming year, assuming 8% revenue growth and stable margins.
**Output:** Present the historical data in a list and the projection in a paragraph, with a note on the quality of the FCF.
```

**Example 4: Qualitative Competitive Advantage Analysis**
```
**Role:** Market strategist.
**Task:** Perform a qualitative analysis of the competitive advantage (Moat) of "Company Alpha" in the SaaS sector.
1. Apply Porter's Five Forces framework to assess the sector's attractiveness.
2. Identify and describe the type of Moat (e.g., Network Effects, Economies of Scale, Intangible Assets).
3. Conclude whether the Moat is durable and defensible.
**Output:** Structure the response with clear headings and subheadings for each section of the analysis.
```

**Example 5: Earnings Call Report Analysis**
```
**Role:** Investor Relations Analyst.
**Task:** Review the transcript of "Company Beta"'s latest Earnings Call.
1. Extract and list all mentions of "margin growth" and "regulatory challenges".
2. Summarize the CEO's overall tone (optimistic, cautious, neutral).
3. Identify 3 key questions asked by analysts and management's answers.
**Output:** Use bullet points for the lists and a paragraph for the tone summary.
```

**Example 6: Scenario and Sensitivity Analysis**
```
**Role:** Financial risk consultant.
**Task:** Perform a sensitivity analysis for the Earnings Per Share (EPS) of "Company Gamma".
1. Calculate the current EPS.
2. Model the EPS under three scenarios: (A) 10% increase in raw material cost, (B) 5% drop in sales volume, (C) Combination of (A) and (B).
3. Present the percentage impact on EPS for each scenario.
**Output:** Comparative table of the scenarios and a conclusion on the resilience of the EPS.
```
```

## Best Practices
**1. Define the Role and Context:** Begin the prompt by instructing the AI to act as a financial, credit, or equity analyst, specifying the sector and market (e.g., "Act as an equity analyst specialized in semiconductor technology").
**2. Detailed Prompt Structure:** Use the **Role, Task, Output** structure. The task should be broken into clear sub-tasks (e.g., "1. Analyze YoY revenue growth. 2. Calculate the Debt/EBITDA ratio. 3. Compare with the sector average").
**3. Provide Input Data:** Whenever possible, include the raw data or the link to the source (e.g., "Based on the following data from the 2023 10-K: [data/link]"). The AI should not "guess" the numbers.
**4. Require Chain-of-Thought Reasoning:** Ask the AI to show the calculation steps and the reasoning behind the conclusion (e.g., "Explain the process of calculating Free Cash Flow before presenting the final result").
**5. Validation and Auditability:** Request references and footnotes for each data point or claim (e.g., "For each financial metric, cite the section and page of the report from which the data was taken").
**6. Output Specificity:** Define the output format (table, bullet points, paragraph), the level of detail, and the tone (e.g., "Present the results in a Markdown table, with a 3-paragraph executive summary at the end").

## Use Cases
**1. Due Diligence and Company Valuation:** Automate the extraction of key metrics (P/E, EV/EBITDA, Margins) from financial reports to speed up the *due diligence* process in mergers and acquisitions (M&A) or venture capital investments.
**2. Earnings Report Analysis:** Process *earnings call* transcripts to summarize management sentiment, identify risks and opportunities, and extract analysts' key questions.
**3. Sector Comparison:** Perform comparative analyses of multiples and financial ratios among competing companies in a specific sector, identifying *outliers* and market leaders.
**4. Scenario Modeling and Stress Testing:** Create prompts to simulate the impact of macroeconomic variables (e.g., interest rate increase, inflation) or microeconomic variables (e.g., loss of a key customer) on a company's financial statements.
**5. Report and Memo Generation:** Generate drafts of equity research reports, investment memos, or sections of annual reports, saving analyst time on the initial drafting.
**6. ESG Analysis (Environmental, Social, and Governance):** Extract and analyze non-financial data from sustainability reports to assess the impact of ESG factors on the company's long-term risk and value.

## Pitfalls
**1. Financial Data Hallucination:** The biggest risk is the AI inventing numbers, ratios, or report dates that look authentic but are false. This is especially dangerous in finance, where accuracy is critical [3].
**2. Failure in Complex Math:** LLMs are language models, not calculators. They can make mistakes in complex calculations, such as Discounted Cash Flow (DCF) or the aggregation of data from multiple sources [4].
**3. Over-reliance on Training Data:** The AI may base its analysis on outdated data or general knowledge, ignoring critical, recent information that was not provided in the prompt (e.g., a recent regulatory event or a new quarterly report) [5].
**4. Bias and Generalization:** The AI may apply implicit biases from its training data, or overgeneralize from a single data point, failing to consider the specific nuances of the sector or company.
**5. Ambiguity in Financial Language:** Terms like "revenue" or "profit" can have different accounting definitions (e.g., IFRS vs. GAAP). Lack of specification in the prompt can lead to incorrect calculations or invalid comparisons.
**6. Ignoring the Data Source:** Failing to specify the data source (e.g., "Use only audited data from the 10-K") can lead the AI to mix data from unreliable sources (e.g., news articles, forums) with official data.

## URL
[https://www.ai-street.co/p/effective-prompts-for-investment-research](https://www.ai-street.co/p/effective-prompts-for-investment-research)
