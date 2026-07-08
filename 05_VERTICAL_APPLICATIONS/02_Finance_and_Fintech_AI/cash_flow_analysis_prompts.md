# Cash Flow Analysis Prompts

## Description
Cash Flow Analysis Prompts are structured instructions provided to Large Language Models (LLMs) to process financial data on cash inflows and outflows. They turn AI into a financial assistant capable of analyzing trends, identifying risks, creating budgets, and forecasting future cash flow. Their effectiveness lies in the prompt's ability to provide context (the AI's role, data, objective) and response guidelines (format, metrics, recommendations) to obtain actionable and accurate financial *insights*. This technique is fundamental to financial management, allowing professionals and business owners to make proactive decisions to optimize an organization's liquidity and solvency.

## Examples
```
**1. Detailed Cash Flow Analysis:**
`#CONTEXT: You are a senior financial analyst. #DATA: [Insert cash inflow and outflow data for the last 6 months, categorized by source/destination]. #OBJECTIVE: Provide a concise analysis highlighting patterns, anomalies, and the 3 main drivers of cash flow variation. #GUIDELINES: 1. Calculate the Cash Flow Margin. 2. Identify the month with the highest and lowest liquidity. 3. Suggest an immediate action to improve the ending balance.`

**2. 90-Day Cash Flow Forecast:**
`#CONTEXT: You are a financial modeling expert. #DATA: [Insert 12 months of historical data and assumptions for sales growth (5%) and cost increase (2%)]. #OBJECTIVE: Create a cash flow projection for the next 3 months. #GUIDELINES: 1. Present the results in table format (Month 1, Month 2, Month 3). 2. Highlight any month with a projected negative cash balance. 3. Explain the forecasting methodology used.`

**3. Identifying Liquidity Risks:**
`#CONTEXT: You are a financial risk auditor. #DATA: [Insert the current cash conversion cycle (e.g., 45 days) and the average collection period (e.g., 60 days)]. #OBJECTIVE: Identify liquidity vulnerabilities and propose mitigation strategies. #GUIDELINES: 1. List 3 critical risks (e.g., dependence on a single customer, seasonality). 2. For each risk, provide an actionable mitigation strategy (e.g., negotiate payment terms with suppliers).`

**4. Working Capital Optimization:**
`#CONTEXT: You are an operational efficiency consultant. #DATA: [Insert the current Working Capital value and the desired minimum cash balance]. #OBJECTIVE: Suggest 5 ways to optimize the use of working capital to free up cash. #GUIDELINES: The suggestions should focus on (a) inventory management, (b) accounts receivable, and (c) accounts payable. Present in a numbered list format.`

**5. Scenario Analysis (Optimistic vs. Pessimistic):**
`#CONTEXT: You are a business strategist. #DATA: [Insert the base cash flow from the last quarter]. #OBJECTIVE: Simulate the impact on the ending cash balance under two scenarios: Optimistic (15% increase in sales) and Pessimistic (10% drop in sales and a 30-day delay in collections). #GUIDELINES: Compare the ending cash balances of the three scenarios (Base, Optimistic, Pessimistic) and provide a conclusion on the company's financial resilience.`

**6. Creating a Cash Budget:**
`#CONTEXT: You are a budget planner. #DATA: [Insert the categories of fixed and variable expenses and the revenue sources]. #OBJECTIVE: Prepare a detailed monthly cash budget. #GUIDELINES: 1. Separate revenues and expenses. 2. Include a column for budget variance (Budgeted vs. Actual). 3. The budget should be easily exportable to a spreadsheet.`
```

## Best Practices
**1. Structure and Context:** Always define the AI's role (e.g., "senior financial analyst"), provide the context (raw or summarized data), and set a clear objective (e.g., "identify the top 3 risks"). **2. Response Guidelines:** Use the `#RESPONSE GUIDELINES` section to specify the output format (table, report, list), the metrics to be calculated (cash flow margin, cash conversion cycle), and the type of recommendation expected (mitigation strategies, budget adjustments). **3. Iterate and Refine:** Start with broad prompts and use follow-up prompts to deepen the analysis (e.g., "Based on the forecast, what are the short-term financing options?"). **4. Confidentiality:** Never enter confidential or personally identifiable data. Use anonymized or simulated data.

## Use Cases
**1. Budget Planning:** Creating detailed cash budgets and short- and long-term financial projections. **2. Risk Management:** Proactively identifying periods of low liquidity or insolvency risks. **3. Working Capital Optimization:** Suggestions to accelerate collections (accounts receivable) and manage payments (accounts payable) to free up cash. **4. Performance Analysis:** Comparing actual cash flow with the budget, identifying deviations and their causes. **5. Strategic Decision-Making:** Assessing the financial impact of investments, acquisitions, or cost cuts before their implementation. **6. Reporting and Communication:** Generating executive summaries and cash flow reports for *stakeholders* and boards of directors.

## Pitfalls
**1. Inserting Unstructured Raw Data:** Providing financial data in a long, disorganized text format (instead of tables or lists with clear labels) leads to interpretation errors and inaccurate results. **2. Lack of Context:** Failing to define the AI's role or the objective of the analysis results in generic and unhelpful responses. The AI needs to know whether it should act as an accountant, a risk analyst, or a strategist. **3. Ignoring Assumptions:** Failing to include business assumptions (e.g., seasonality, new product launches, inflation) in the cash flow forecast results in unrealistic projections. **4. Over-reliance:** Treating the AI's output as an established fact without human validation. AI is an analysis tool, not a substitute for financial *due diligence*. **5. Single, Long Prompts:** Trying to solve the entire analysis in a single complex prompt. It is more effective to use a series of follow-up prompts to refine the analysis and deepen the *insights*.

## URL
[https://www.godofprompt.ai/blog/prompts-to-improve-your-cash-flow](https://www.godofprompt.ai/blog/prompts-to-improve-your-cash-flow)
