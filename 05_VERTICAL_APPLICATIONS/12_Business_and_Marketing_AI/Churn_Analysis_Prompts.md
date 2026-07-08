# Churn Analysis Prompts

## Description
**Churn Analysis** is a prompt engineering technique focused on leveraging Large Language Models (LLMs) to process customer data, identify behavioral patterns that lead to cancellation (churn), and generate actionable *insights* for retention. Instead of merely predicting churn, these prompts are designed to act as a **virtual data analyst**, performing tasks such as customer segmentation, root cause analysis, conceptual predictive model creation, and intervention strategy development.

The effectiveness of *Churn Analysis Prompts* lies in their ability to structure data input (often via copy-pasting data samples or describing the *features* of a dataset) and demand an analytical and strategic output, transforming raw data into business intelligence. They are crucial for SaaS, telecommunications, and subscription service companies, where customer retention is directly linked to financial health.

## Examples
```
**Prompt Examples**:

1.  **Cohort Analysis and Risk Factors**
    ```
    **Role:** Senior Data Scientist.
    **Task:** Analyze churn over the last 6 months by acquisition cohort (monthly).
    **Data:** [Insert here a snippet of churn data in CSV or Markdown format, or describe the columns: 'Customer_ID', 'Acquisition_Date', 'Months_Active', 'Monthly_Usage_Hours', 'Support_Tickets', 'Churn_Status'].
    **Output:** Markdown table with the churn rate per cohort and a 150-word executive summary highlighting the top 3 risk factors and the most problematic cohort.
    ```

2.  **Creation of Risk Segments**
    ```
    **Context:** We are a B2B SaaS platform. Churn is defined as non-renewal after 12 months.
    **Task:** Create 3 high churn risk customer segments (e.g., 'Extreme Risk', 'Moderate Risk', 'Latent Risk') based on the following metrics: [Low Login Frequency, 50% Drop in Key Feature Usage, Opening 3+ Support Tickets in the Last 30 Days].
    **Output:** For each segment, provide a description, the risk scoring criteria, and 2 specific retention actions.
    ```

3.  **Churn Feedback Sentiment Analysis**
    ```
    **Task:** Analyze the following 10 comments from customers who canceled and categorize the sentiment (Negative, Neutral, Positive) and the main reason (Price, Feature, Support, Competition).
    **Data:** [List of 10 cancellation feedback comments].
    **Output:** Table with 'Comment', 'Sentiment', and 'Main Reason'. Then, suggest a product or process change to mitigate the most frequent reason.
    ```

4.  **Predictive Model Simulation**
    ```
    **Role:** Machine Learning Engineer.
    **Task:** Simulate a classification model (e.g., Random Forest) to predict churn.
    **Features:** [Customer Age, Contract Length, Contract Value (MRR), Number of Logins/Week, Feature X Usage, Feature Y Usage].
    **Output:** Explain which would be the 3 most important features for the prediction and why, based on market knowledge. Create a follow-up prompt to refine the analysis.
    ```

5.  **Retention Playbook Creation**
    ```
    **Context:** Customer 'ID 456' is in the 'Extreme Risk' segment (70% drop in usage and 0 interactions with Support).
    **Task:** Create a 3-step retention *playbook* for the Customer Success Manager (CSM) to use.
    **Steps:** 1. Initial Contact (Channel and Message), 2. Value Offer (Incentive), 3. Follow-up (Next Action).
    **Output:** A detailed script for Step 1 (email or chat message) and the logic behind the Value Offer.
    ```

6.  **Competitive Churn Analysis**
    ```
    **Task:** Analyze the following feedback from customers who migrated to competitor 'X'.
    **Feedback:** [List of 5 reasons why customers went to competitor X].
    **Output:** Identify the main differentiator of competitor X as perceived by customers and suggest 3 improvement points in our product/service to neutralize this advantage.
    ```
```

## Best Practices
**Best Practices**:
1.  **Provide Context and Structured Data**: Always include as much relevant data as possible (e.g., CSV, JSON, or a detailed description of the dataset) and the business context (e.g., product type, analysis period, churn definition).
2.  **Define the Role (Role-Playing)**: Start the prompt by defining the LLM as a "Senior Data Scientist," "Customer Success Analyst," or "Retention Specialist" to ensure a response with the correct perspective.
3.  **Specify the Output (Output Structuring)**: Ask for the output in a specific format (e.g., Markdown table, JSON, 150-word executive summary) to facilitate analysis and integration into reports.
4.  **Root Cause Analysis**: Don't just ask for the prediction; ask for the analysis of the **predictive factors** and the **justification** for the churn risk score.
5.  **Iteration and Refinement**: Use the initial output for follow-up prompts, such as "Based on the top 3 risk factors, create 5 personalized retention actions for the 'Low Activity Users' segment."

## Use Cases
**Use Cases**:
1.  **Segmentation of At-Risk Customers**: Identify groups of customers with a high probability of cancellation for targeted retention campaigns.
2.  **Support Resource Optimization**: Analyze support tickets and interactions to identify patterns of dissatisfaction that precede churn, enabling proactive intervention.
3.  **Product Hypothesis Validation**: Use the LLM to analyze cancellation feedback and validate whether the lack of a specific feature or a usability issue is driving churn.
4.  **Retention Content Creation**: Generate drafts of emails, chat messages, or personalized offers for at-risk customers, tailored to their usage profile and reason for dissatisfaction.
5.  **Fast Executive Reports**: Transform raw data or Machine Learning model summaries into a clear and concise executive summary for leadership, saving the analyst's time.
6.  **Churn Analysis by Category**: Analyze churn across different product lines or services to understand where the problem is most acute and why.

## Pitfalls
**Common Pitfalls**:
1.  **Excessive Raw Data Injection**: Trying to insert an entire CSV file (thousands of rows) directly into the prompt. LLMs have context limits and may fail or generate inaccurate results. **Solution**: Provide representative samples or just the statistical description of the data.
2.  **Lack of Churn Definition**: Not clearly defining what constitutes "churn" for the business (e.g., immediate cancellation, non-renewal, inactivity for 90 days). This leads to vague analyses.
3.  **Confirmation Bias**: Asking the LLM to confirm a pre-existing hypothesis (e.g., "Price is the main reason for churn, right?"). The LLM may just regurgitate the hypothesis instead of performing an objective analysis.
4.  **Ignoring the LLM's Role**: Treating the LLM as statistical software that executes code. The LLM is a **reasoning and language engine**. It should be used to **interpret** data and **generate strategies**, not for complex statistical calculations that require tools like Python/Pandas.
5.  **Unstructured Output**: Not specifying the output format. This results in long blocks of text that are difficult to digest and use in business reports.

## URL
[https://pt.linkedin.com/pulse/guia-pr%C3%A1tico-de-engenharia-prompt-do-b%C3%A1sico-ao-adrianno-esnarriaga-polqf](https://pt.linkedin.com/pulse/guia-pr%C3%A1tico-de-engenharia-prompt-do-b%C3%A1sico-ao-adrianno-esnarriaga-polqf)
