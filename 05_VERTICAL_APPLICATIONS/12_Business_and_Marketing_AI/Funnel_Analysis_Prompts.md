# Funnel Analysis Prompts

## Description
**Funnel Analysis Prompts** are an advanced *Prompt Engineering* technique focused on instructing Large Language Models (LLMs) to process data from marketing, sales, product (AARRR), or user experience (UX) funnels in order to identify bottlenecks (*leakages*), calculate conversion rates, and suggest actionable optimizations. The technique requires the user to provide the AI with the **context** (what is being analyzed), the **raw data** (metrics per stage), and the **instruction** (what the AI should do with that data, usually acting as an analyst or consultant). The goal is to transform quantitative data into qualitative and strategic *insights*, enabling the AI to simulate an analytical consulting process. This technique is fundamental for *growth* and *performance* optimization in digital environments.

## Examples
```
**Example 1: E-commerce (Purchase Funnel)**
**Role:** Act as a Conversion Rate Optimization (CRO) Analyst for a fashion e-commerce store.
**Context:** The purchase funnel is: **Homepage Visit -> Product View -> Add to Cart -> Checkout -> Purchase Completed**.
**Data:** Over the past 4 weeks, we had 100,000 visits, 40,000 product views, 8,000 add-to-cart actions, 2,000 checkout starts, and 500 completed purchases. The AOV (Average Order Value) is R$ 300.
**Instruction:** Identify the stage with the highest abandonment rate, calculate the potential incremental revenue if that stage's conversion rate were increased by 10%, and suggest 3 specific CRO tactics for that leakage point.

**Example 2: SaaS (Activation Funnel)**
**Role:** Act as an Activation-focused Product Manager (Activation Manager) for a B2B SaaS project management software.
**Context:** The activation funnel is: **Sign-up -> App Installation -> First Project Creation -> Invite Team Member -> Weekly Usage**.
**Data:** Of the 5,000 new sign-ups last month, 3,500 installed the app, 1,500 created their first project, 500 invited a member, and only 100 reached weekly usage.
**Instruction:** Analyze the transition from "App Installation" to "First Project Creation." What is the main perceived barrier? Create a 3-step *onboarding* email *prompt* to reduce this *drop-off*, focusing on immediate value (*Aha! Moment*).

**Example 3: Lead Generation (Marketing Funnel)**
**Role:** Act as a Marketing Automation Specialist for a B2B consulting company.
**Context:** The lead funnel is: **Blog Visit -> Ebook Download (MQL) -> Demo Request (SQL) -> Meeting Scheduled**.
**Data:** Last quarter, 50,000 blog visits generated 5,000 downloads, 200 demo requests, and 50 scheduled meetings.
**Instruction:** Focus on the conversion from MQL to SQL. Analyze the conversion rate and suggest 5 *lead scoring* criteria that, if implemented, could improve the quality of the leads reaching the sales team.

**Example 4: Content and Engagement (Media Funnel)**
**Role:** Act as a Content Strategist for an online news channel.
**Context:** The engagement funnel is: **Article View -> 50% Scroll -> Click on Related Article -> Newsletter Subscription**.
**Data:** Where is the largest *drop-off*? Propose a change to the newsletter's *call-to-action* (CTA) and a new content format (e.g., quiz, infographic) for the "Click on Related Article" stage to increase conversion to the newsletter.

**Example 5: Retention Analysis (Churn Funnel)**
**Role:** Act as a Customer Data Scientist for a subscription streaming service.
**Context:** The *churn* (attrition) funnel is: **Active Subscription -> Weekly Usage -> Reduced Usage -> Cancellation -> Reactivation**.
**Data:** 10,000 active users. 500 reduced their usage last month. Of those, 100 cancelled. 10 reactivated.
**Instruction:** Describe the profile of the 100 users who cancelled (based on fictional engagement data: watched less than 2 hours/week, did not use the favorites list feature). Based on this profile, create a *prompt* for an AI model to generate 3 personalized retention offers and the ideal timing to send them.

**Example 6: Product Funnel (UX/UI)**
**Role:** Act as a UX/UI Designer.
**Context:** The feature usage funnel is: **Open Feature -> Interact with Filter -> Apply Filter -> Save Configuration**.
**Data:** 5,000 feature opens, 4,000 filter interactions, 1,500 filter applications, 500 configuration saves.
**Instruction:** The *drop-off* between "Interact with Filter" and "Apply Filter" is high. List 3 usability (UX) hypotheses for this leakage and suggest an interface (UI) A/B test to validate the most likely hypothesis.

**Example 7: Complex Sales Funnel (B2B)**
**Role:** Act as a Sales Strategy Consultant.
**Context:** The sales funnel is: **Prospecting -> Qualification -> Proposal -> Negotiation -> Closing**.
**Data:** 100 prospects, 50 qualifications, 20 proposals sent, 10 negotiations, 5 closings.
**Instruction:** Analyze the conversion from "Proposal" to "Negotiation." What is the conversion rate? Create a *prompt* *template* for the sales team to use in the CRM, asking the AI for a predictive analysis of the loss risk for each proposal, based on 3 input variables (e.g., customer response time, number of *stakeholders* involved, proposal value).
```

## Best Practices
**1. Define the Funnel Clearly:** First and foremost, map the funnel stages in a logical and sequential way (e.g., AARRR - Acquisition, Activation, Retention, Revenue, Referral). Clarity of the funnel is the foundation for the AI's analysis. **2. Provide Structured and Contextualized Data:** Present the data in an organized way (table, list) and include the business context (industry, target audience, revenue model). **3. Assign a "Role" (Persona):** Ask the AI to act as a specific expert (e.g., "Act as a CRO Analyst," "Act as a Data Scientist"). This improves the quality and focus of the responses. **4. Request Actions and Hypotheses:** Do not just ask for the identification of the problem, but also for suggested actions, A/B tests, or root cause hypotheses. **5. Use the Analysis for Scenarios:** Ask the AI to calculate the potential impact of improvements (e.g., "Calculate the revenue increase if the conversion of stage X goes from 5% to 7%").

## Use Cases
**1. Conversion Optimization (CRO):** Identify the exact point of greatest abandonment in an e-commerce or SaaS funnel to focus optimization efforts. **2. Product Activation Analysis:** Understand why new users are not completing *onboarding* or reaching the *Aha! Moment* in applications. **3. Retention and *Churn* Strategy:** Analyze the abandonment funnel to predict and prevent customer *churn* (cancellation), generating personalized retention offers. **4. *Lead Scoring* and Qualification:** Help marketing and sales teams refine lead scoring criteria (MQL to SQL) based on historical conversion data. **5. Scenario Simulation (*What-If*):** Calculate the potential financial impact of hypothetical improvements in the conversion rate of a specific stage. **6. UX/UI Diagnosis:** Apply funnel logic to software feature usage flows to identify usability flaws.

## Pitfalls
**1. Incomplete or Biased Data:** Providing the AI with partial, outdated, or attribution-biased data (e.g., attributing all sales to the last click). The AI can only analyze what is provided. **2. Poorly Defined Funnel:** Not clearly mapping the funnel stages in a logical and sequential way, resulting in confusing or irrelevant analyses. **3. Lack of Business Context:** Not informing the AI of the business model (SaaS, e-commerce, B2B), the target audience, or the specific objectives, leading to generic recommendations. **4. Ignoring Data Quality:** Not including data integrity validation in the analysis (e.g., event duplication, *bots*), which can distort conversion rates. **5. Excessive Focus on Vanity Metrics:** Asking the AI to optimize top-of-funnel metrics (e.g., views) without connecting the impact on bottom-of-funnel metrics (e.g., revenue, LTV). **6. Non-Actionable Recommendations:** Requesting analyses without explicitly asking for concrete *tactics* or *test hypotheses*, resulting in theoretical *insights*. **7. Not Segmenting the Analysis:** Analyzing the funnel as a whole, without segmenting by channel (organic vs. paid), device (mobile vs. desktop), or user cohort, missing the opportunity to identify specific leakages. **8. Confusing Cause with Correlation:** Accepting the AI's conclusions without applying your own critical judgment, especially in root cause analyses, where correlation may be confused with causation.

## URL
[https://founderpal.ai/prompts-examples/funnel-analysis](https://founderpal.ai/prompts-examples/funnel-analysis)
