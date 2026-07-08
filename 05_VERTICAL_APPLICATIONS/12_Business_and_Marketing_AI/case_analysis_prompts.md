# Case Analysis Prompts

## Description
Case Analysis Prompts are a Prompt Engineering technique that instructs a Large Language Model (LLM) to act as an analyst, consultant, or expert to examine a complex scenario, problem, or situation (the "case"). The goal is to obtain a structured analysis, diagnoses, recommendations, and detailed action plans, rather than just factual answers.

This technique is fundamentally structural, requiring the prompt to clearly define:
1.  **The Role (Persona):** The LLM should assume a specific role (e.g., "Strategy Consultant," "Senior Lawyer," "Data Analyst").
2.  **The Case Context:** All data, facts, constraints, and objectives relevant to the analysis.
3.  **The Analysis Structure:** The desired output format (e.g., "SWOT Analysis," "Porter's Five Forces," "Legal Opinion").
4.  **The Expected Outcome:** The specific question to be answered or the decision to be made.

By providing a clear structure and rich context, Case Analysis Prompts transform the LLM from a mere text generator into a powerful tool for reasoning and solving complex problems, applicable across various domains such as business, law, technology, and healthcare.

## Examples
```
1.  **Business Analysis (SWOT):**
    ```
    Act as a Senior Strategy Consultant. Analyze the case of the company 'TechNova', a SaaS startup that saw its growth stall at 5% last quarter, despite a 20% increase in marketing spend. The product is a project management software.
    Perform a complete SWOT Analysis, focusing on: Strengths (unique product features), Weaknesses (pricing structure and customer support), Opportunities (expansion into the European market), and Threats (a new competitor with a price 50% lower).
    Conclude with three priority action recommendations to resume growth to 15% next quarter.
    ```

2.  **Legal Analysis (Opinion):**
    ```
    Act as a Senior Lawyer specialized in Consumer Law in Brazil.
    Case: A customer bought a product online that arrived damaged. The store refuses to accept the return, claiming the damage occurred during transport, the carrier's responsibility. The customer has a 7-day cooling-off period.
    Draft a concise Legal Opinion, citing the relevant articles of the Consumer Defense Code (CDC) and applicable case law.
    What is the probability of success in a lawsuit and what is the best strategy for the customer?
    ```

3.  **Technical Analysis (Root Cause Analysis - RCA):**
    ```
    Act as a DevOps Engineer.
    Case: A critical failure occurred in the e-commerce system during Black Friday, resulting in 4 hours of downtime. The log indicates a spike in database (DB) requests followed by a deadlock. The DB was running on a medium-sized instance.
    Perform a Root Cause Analysis (RCA) using the 5 Whys method.
    Identify the root cause and propose a three-step mitigation plan (short, medium, and long term) to prevent recurrence.
    ```

4.  **Marketing Analysis (Segmentation):**
    ```
    Act as a Digital Marketing Analyst.
    Case: A fitness company launched a new home workout app, but the retention rate after the first month is only 15%. The initial target audience was "young adults (18-30 years old)."
    Analyze the case and suggest a new, more promising target audience segmentation.
    Create a detailed persona profile for the new segment and suggest a new value proposition focused on that persona's needs.
    ```

5.  **Scenario Analysis (Strategic Decision):**
    ```
    Act as a CEO.
    Case: Your company needs to decide between two investment options: Option A (Invest R$ 10 million in R&D for a high-risk/high-reward product) or Option B (Invest R$ 5 million in process optimization for a 15% efficiency gain).
    Analyze the pros and cons of each option, considering the current economic scenario (high inflation, uncertain interest rates).
    Recommend the best option and justify your decision based on a risk/reward matrix.
    ```

6.  **Data Analysis (Report Interpretation):**
    ```
    Act as a Data Scientist.
    Interpret the following dataset of a product's sales over the last 6 months: [January: 100k, February: 120k, March: 90k, April: 150k, May: 110k, June: 180k].
    Identify trends, anomalies, and possible correlations with external events (e.g., March had an extended holiday, June had a discount campaign).
    Forecast sales for the next quarter (July, August, September) and justify the forecast.
    ```

7.  **Product Analysis (Feature Prioritization):**
    ```
    Act as a Product Manager.
    Case: You have three new features to prioritize: A (critical bug fixes), B (new feature requested by the most important customer), C (usability improvement affecting 80% of users).
    Use the RICE framework (Reach, Impact, Confidence, Effort) to analyze and prioritize the features.
    Present the RICE score for each and the recommended priority order.
    ```

8.  **Healthcare Analysis (Differential Diagnosis):**
    ```
    Act as a General Practitioner.
    Case: A 45-year-old patient presents with chronic fatigue, unexplained weight gain, and cold sensitivity. Blood tests show elevated TSH and low free T4.
    Perform a Differential Diagnosis, listing the possible conditions and the most likely one.
    Suggest the next investigation steps and the recommended initial treatment.
    ```

9.  **Sustainability Analysis (ESG):**
    ```
    Act as an ESG Consultant (Environmental, Social, and Governance).
    Case: A mining company is under public pressure due to a small tailings leak. The company has a history of good governance, but the environmental pillar is at risk.
    Analyze the impact of the leak on the company's reputation and ESG metrics.
    Propose a crisis communication strategy and three concrete actions to strengthen the environmental pillar over the next 12 months.
    ```

10. **Career Analysis (Development Plan):**
    ```
    Act as a Career Coach.
    Case: An IT professional with 5 years of experience in Front-end development wants to migrate to the Machine Learning field. He has a basic knowledge of Python and statistics.
    Analyze the skills gap and create a 6-month Individual Development Plan (IDP).
    The IDP should include courses, hands-on projects, and success metrics for the career transition.
    ```
```

## Best Practices
*   **Define the Role (Persona) Clearly:** Always start with "Act as a [Specialist/Professional]" to direct the tone, knowledge, and perspective of the response.
*   **Provide Rich Context:** Include all data, constraints, history, and objectives of the case. The quality of the analysis depends directly on the richness of the context provided.
*   **Structure the Output:** Use phrases such as "Use the [X] structure," "Draft a [Y]," or "Respond in [Z] format" to ensure the LLM delivers an organized and usable analysis (e.g., SWOT, 5 Whys, IDP).
*   **Be Specific About the Objective:** The prompt should end with a clear question or a request for a decision (e.g., "What is the best strategy?", "Recommend option A or B and justify").
*   **Iteration and Refinement:** If the first analysis is superficial, use follow-up prompts to dig deeper into specific points (e.g., "Now, deepen the analysis of Threat X and propose detailed countermeasures").

## Use Cases
*   **Business Consulting:** Simulation of market scenarios, feasibility analysis of new products, strategic planning (SWOT Analysis, PESTEL).
*   **Legal Area:** Drafting preliminary opinions, litigation risk analysis, interpretation of complex contractual clauses, and identification of precedents.
*   **Product Development:** Feature prioritization (RICE, MoSCoW), user feedback analysis, and diagnosis of usability problems.
*   **IT and Engineering:** Root Cause Analysis (RCA) for system failures, architecture planning, and security risk assessment.
*   **Academia and Education:** Creation of case studies for teaching, solving complex problems in disciplines such as finance and administration.

## Pitfalls
*   **Insufficient Context:** Providing only a vague summary of the case. The LLM cannot analyze what it does not know.
*   **Vague Objective:** Asking only to "analyze this case." The LLM may deliver a generic analysis without focus on the necessary decision or outcome.
*   **Over-Reliance on Fictitious Data:** The LLM may "hallucinate" data, statistics, or legal precedents. The analysis should always be validated by a human expert, especially in critical areas such as law and healthcare.
*   **Ignoring Structure:** Not defining the output format. This results in a continuous text that makes it difficult to extract actionable information.
*   **Confirmation Bias:** Structuring the case in a way that induces the LLM to confirm a pre-existing hypothesis, rather than performing an objective analysis.

## URL
[https://www.sybill.ai/blogs/chatgpt-create-case-studies](https://www.sybill.ai/blogs/chatgpt-create-case-studies)
