# Financial Analysis Prompts

## Description
The **Financial Analysis Prompts** technique refers to the art and science of crafting optimized instructions and questions for large language models (LLMs) in order to perform complex tasks in finance, accounting, investment, and risk. It involves providing specific financial context, structured data (such as balance sheets, income statements, or market data) and defining the desired output format (such as reports, summaries, or forecasts). Its effectiveness lies in the ability to turn raw data into actionable *insights*, automating trend analysis, risk assessment, scenario planning, and regulatory compliance. It is a critical application of Prompt Engineering, since accuracy and compliance are essential in the financial sector.

## Examples
```
1. **Financial Statement Analysis:** "Based on the attached Balance Sheet and Income Statement for Company X (2023-2024), act as a Senior Credit Analyst. Calculate and interpret the following ratios: Current Ratio, Total Debt Ratio, and Net Margin. Present the results in a Markdown table and provide a 2-paragraph summary of the company's financial health."

2. **Scenario Planning (Stress Test):** "Given the attached investment portfolio (Stocks: 60%, Bonds: 30%, Gold: 10%), simulate the financial impact of an 'inflationary crisis' scenario (15% inflation, 20% drop in the stock market). What would the portfolio's percentage loss be? Suggest 3 defensive actions to mitigate this risk."

3. **Budget Optimization:** "Analyze the Marketing Department's budget for the last quarter. Identify the 3 largest expense areas and suggest 2 areas where a 10% cut would be most feasible, justifying the operational impact of each cut. The result should be a concise report."

4. **Compliance Analysis:** "Review last month's high-value transaction statement. Identify any transaction that might raise a 'red flag' under Anti-Money Laundering (AML) regulations, specifying the reason for suspicion and the recommended next regulatory step."

5. **Cash Flow Forecast:** "Using the historical cash flow data from the last 6 months (average inflow: R$ 500k/month, average outflow: R$ 450k/month, with a December outflow peak of R$ 600k), project the cash balance for the next 3 months. Include a sensitivity analysis for a 10% drop in revenue."

6. **Asset Valuation (M&A):** "Act as an M&A Advisor. Assess the financial viability of acquiring 'Startup Y' (Annual Revenue: R$ 5M, Net Income: R$ 500k, Total Debt: R$ 1M). Use the Revenue Multiple method (5x) and suggest a fair purchase price, listing 3 key financial risks of the transaction."

7. **Interpreting Market Indicators:** "Explain what Company Z's P/E (Price/Earnings) ratio of 25x means for a long-term investor, comparing it with the sector average (15x). The explanation should be clear and accessible, suitable for a beginner investor."
```

## Best Practices
**1. Provide Structured Data:** Always include real financial data (tables, CSVs, or lists) directly in the prompt or reference a document/data source. The accuracy of the analysis depends on the quality and format of the input data. **2. Define the Role (Persona):** Start the prompt by defining the LLM as a "Senior Financial Analyst", "Certified Accountant", or "Risk Specialist" to steer the tone and depth of the response. **3. Specify the Output Format:** Request the output in a clear, usable format, such as a "Markdown Table", "3-Paragraph Executive Summary", or "JSON with the KPIs". **4. Include Constraints and Assumptions:** For scenario or risk analyses, clearly define the assumptions (e.g., "Assume a 15% increase in the Selic rate") to ensure the analysis is relevant. **5. Request Sources and Justifications:** Ask the LLM to cite data sources or justify the calculation methodology to ensure traceability and confidence in the results.

## Use Cases
**1. Strategic and Budget Planning:** Creating annual budgets, revenue and expense projections, and sensitivity analysis for different economic scenarios. **2. Risk Management and Compliance:** Credit risk assessment, transaction fraud detection, monitoring financial activities for adherence to regulations (e.g., IFRS, BACEN, CVM). **3. Investment and Valuation:** Analyzing stock portfolios, valuing assets for mergers and acquisitions (M&A), and financial *due diligence*. **4. Accounting and Reporting:** Automated generation of executive summaries of financial statements, expense categorization, and creation of performance reports for stakeholders. **5. Process Optimization:** Payroll analysis for cost optimization and automation of repetitive data-analysis tasks.

## Pitfalls
**1. Data Hallucinations:** The LLM may fabricate financial data, ratios, or regulations. **Countermeasure:** Always request the source citation or the calculation formula. **2. Lack of Context:** Overly generic prompts that fail to specify the role, target audience, or time period lead to superficial answers. **Countermeasure:** Use the *Persona* and *Constraint* techniques. **3. Bias and Oversimplification:** LLMs may oversimplify complex risk analyses or ignore regulatory nuances. **Countermeasure:** Request a "multiple-scenario" analysis or the inclusion of "non-quantifiable risks". **4. Leakage of Confidential Information:** Entering sensitive financial data into public LLM platforms. **Countermeasure:** Use *on-premise* LLMs or secure APIs, and anonymize confidential data. **5. Format Errors:** Failing to specify the output format can result in disorganized, hard-to-process data. **Countermeasure:** Require structured formats such as JSON or Markdown tables.

## URL
[https://www.glean.com/blog/30-ai-prompts-for-finance-professionals](https://www.glean.com/blog/30-ai-prompts-for-finance-professionals)
