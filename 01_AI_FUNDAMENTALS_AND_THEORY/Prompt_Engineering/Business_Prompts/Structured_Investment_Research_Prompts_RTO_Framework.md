# Structured Investment Research Prompts (RTO Framework)

## Description

Prompt Engineering for Investment Research refers to applying structured techniques to maximize the accuracy and relevance of Large Language Model (LLM) responses in financial and market analysis. The core of this technique is the three-pillar structure: **Role**, **Task**, and **Output**. By defining the LLM as an expert (e.g., an equity analyst), detailing the task with specific metrics and time periods, and formatting the desired output (e.g., in bullet points with direct quotations), analysts significantly mitigate the risk of hallucinations and obtain actionable *insights*. The practice is crucial for scaling market analysis, allowing investors to examine a larger volume of companies and test more hypotheses with quality and auditability [1].

## Statistics

**Academic Citations:** Recent research (2023-2025) highlights the impact of *prompt engineering* on financial decision-making [3] [5]. The article "Review of Prompt Engineering Techniques in Finance" (2025) was cited by more than 40 sources, indicating the growing relevance of the topic [2]. **Performance Metrics:** Research articles compare the performance of LLMs (including GPT-5, Gemini 2.5 Pro, Claude Opus) on financial reasoning tasks, with analyses involving more than 60 runs to measure the effectiveness of different *prompts* [6]. **Scalability:** The use of structured *prompts* enables an analysis of "80% quality across a thousand companies, instead of 99% across 10 companies," dramatically increasing the scalability of research [1].

## Features

**Three-Pillar Structure (RTO):** Defines the **Role** of the LLM (e.g., credit analyst), the **Task** to be performed (e.g., YoY revenue analysis), and the desired **Output** format (e.g., table, executive summary). **Advanced Techniques:** Integration of complex reasoning methods such as *Chain-of-Thought* (CoT), *Tree-of-Thought* (ToT), and *Graph-of-Thought* (GoT) to improve the models' financial reasoning capabilities [2] [3]. **Auditability:** A requirement for auditable sources and direct citations to mitigate hallucinations, a critical point in financial analysis. **Library Frameworks:** Use of centralized repositories to store, manage, and share *templates* of structured and tested *prompts* [4].

## Use Cases

**Financial Statement Analysis:** Summarizing data, analyzing trends, drafting disclosure notes, and identifying anomalies in financial reports (Balance Sheet, Income Statement, Cash Flow). **Portfolio Analysis:** Assessing risk and return, and suggesting asset rebalancing. **Due Diligence and Market Research:** Automating deal screening, market research, and drafting investment memoranda. **Hypothesis Testing:** Using LLMs to determine the direction of future earnings and the impact of macroeconomic events on specific sectors. **Prospecting:** Generating *prompts* for planning, investment, and client prospecting for financial advisors [4].

## Integration

Effective integration requires adopting the RTO (Role, Task, Output) structure and using *prompts* specific to each type of analysis.

**Best Practices:**
1.  **Define the Role:** Always begin the *prompt* by defining the LLM's role (e.g., "Act as an *equity* analyst specialized in investment banking...").
2.  **Specify the Task:** Detail the period (Q4 2024), the metrics (*investment banking* revenue, QoQ), and the context (YoY comparison) [1].
3.  **Require a Format (Output):** Ask for the output in a consumable format (e.g., "Start with Q4 performance, followed by YoY/sequential trends, and then management's outlook. Quote management directly when discussing the outlook.").

**Prompt Examples:**
*   **Financial Statement Analysis:** "As a financial analyst, analyze the Balance Sheet and Income Statement of [Company Name] for the last quarter. Identify the top three changes relative to the previous quarter and explain the potential impact on future cash flow."
*   **Portfolio Analysis:** "Assess the risk and return of my investment portfolio based on current market conditions, suggesting a more defensive asset allocation and justifying the change based on [Macroeconomic Event]."
*   **Trend Analysis:** "Analyze the impact of [Government Regulation] on the [Sector] sector and identify three companies that are best positioned to benefit, providing a brief justification for each one."

## URL

https://www.ai-street.co/p/effective-prompts-for-investment-research