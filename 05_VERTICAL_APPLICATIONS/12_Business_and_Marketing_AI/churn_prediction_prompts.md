# Churn Prediction Prompts

## Description
Churn Prediction Prompts are prompt engineering instructions designed to use Large Language Models (LLMs) to analyze customer data and identify individuals at high risk of cancellation (churn). They act as an interpretation and analysis layer over structured or unstructured data (such as usage logs, support history, feedback, and sentiment), allowing the LLM to classify customers, suggest personalized retention actions, and justify its predictions. This technique is crucial for Customer Success, Marketing, and RevOps teams, transforming churn prediction from a purely statistical exercise into an actionable, contextualized intelligence tool [1] [2]. The most advanced approach involves creating *embeddings* of customer data (such as the concatenation of categorical attributes) and using these vectors as input for classification models, with LLMs being used to generate the *embeddings* or to interpret the results and generate retention strategies [3].

## Examples
```
1. **Risk Classification and Retention Action:**
   `"Analyze the following data for customer {CUSTOMER_ID}: Product Usage: {USAGE_LAST_MONTH}, Feedback Score (NPS): {NPS}, Support Ticket History: {TICKETS_LAST_6_MONTHS}. If usage dropped by more than 50% OR the NPS is negative, classify the customer as 'High Risk' and suggest the 'Next Best Action' (e.g., 'Discount Offer', 'Proactive CS Call', 'Sending Educational Content'). Return the result in JSON format with the fields: 'Risk', 'Justification', 'Suggested_Action'."`

2. **Sentiment Analysis for Churn:**
   `"Based on the last 10 support interactions and social media comments from customer {CUSTOMER_ID}, perform a sentiment analysis. If the average sentiment is 'Negative' and the customer has not used the main feature {FEATURE_NAME} in the last week, predict the churn probability (0-100%) and the main perceived reason."`

3. **Segmentation of At-Risk Customers:**
   `"Use usage and demographic data to identify 5 distinct groups of customers who have shown signs of churn in the last 90 days. For each group, describe the 'Risk Behavior Pattern' and suggest a specific 'Retention Campaign'."`

4. **Interpretation of Categorical Data (For LLM as Embedder/Interpreter):**
   `"The customer has the following categorical attributes: {PLAN_TYPE}, {LOGIN_FREQUENCY}, {PAYMENT_STATUS}, {PREVIOUS_CANCELLATION_REASON}. Concatenate these attributes into a coherent sentence and then generate a 1536-dimension embedding vector for this sentence. (Instruction for use in an ML architecture, as in [3].)"`

5. **Creating a Proactive Alert:**
   `"Act as a Customer Success specialist. Review the profile of customer {CUSTOMER_ID} and the following data: {COMPLETE_DATA}. Create an internal alert for the CS team, including: 'Risk Status', 'Warning Signs (Bullet Points)', and 'Immediate Recommendation'."`

6. **Root Cause Analysis (Post-Churn):**
   `"Customer {CUSTOMER_ID} cancelled the service. Analyze the complete history of interactions, usage, and feedback. Determine the 'Main Root Cause' of the churn and provide 3 'Lessons Learned' to avoid similar cases in the future."`

7. **Scenario Simulation:**
   `"If customer {CUSTOMER_ID} receives an offer of a 20% discount and one month of free consulting, what is the estimated probability that they will stay? Justify your answer based on the success history of similar offers for customers with the {CUSTOMER_PROFILE} profile."`
```

## Best Practices
- **Detailed Contextualization:** Provide the LLM with as much context as possible about the customer, including usage data, support history, and feedback.
- **Clear Definition of Churn:** Explicitly define what constitutes "churn" for your business (e.g., 30 days of inactivity, subscription cancellation).
- **Output Structure:** Ask the LLM to format the output in a structured way (JSON, table) to facilitate integration with CRM or automation systems.
- **Risk Metrics:** Use quantifiable metrics (e.g., a 50% drop in usage, a sentiment score below 3/5) to classify risk objectively.
- **Actionable Suggestions:** Require the LLM not only to predict churn but also to suggest the Next Best Action for retention.

## Use Cases
- **Customer Success:** Proactive identification of at-risk customers and suggestion of personalized interventions (email, call, offer).
- **Marketing and Sales:** Customer segmentation for targeted retention campaigns and personalization of value messages.
- **Data Analysis:** Interpretation of unstructured data (feedback text, call transcripts) to extract churn warning signals.
- **Product Development:** Root cause analysis of churn to inform the product *roadmap* and prioritize fixing failures or improving features.
- **Hybrid Predictive Modeling:** Generation of high-quality *embeddings* from customer data (structured and unstructured) to feed traditional Machine Learning models, improving prediction accuracy [3].

## Pitfalls
- **Over-Reliance on Raw Data:** Feeding the LLM large volumes of raw data without preprocessing or summarization can lead to inaccurate results or high computational cost.
- **Data Bias:** If the training or input data reflects a historical bias (e.g., only low-value customers received retention offers), the LLM may perpetuate that bias in its suggestions.
- **Lack of Context:** Generic prompts that do not clearly define the input variables or the prediction objective (e.g., "Predict churn") result in vague, non-actionable outputs.
- **Hallucinations:** The LLM may "hallucinate" justifications or suggest retention actions that are not viable or not based on the data provided.
- **Ignoring Hybrid Architecture:** Believing that the LLM can replace traditional statistical models. Research suggests that combining LLM *embeddings* with traditional classifiers (such as Logistic Regression) may be the most robust approach [3].

## URL
[https://www.getcensus.com/blog/top-10-llm-prompts-for-revops-and-marketing-teams](https://www.getcensus.com/blog/top-10-llm-prompts-for-revops-and-marketing-teams)
