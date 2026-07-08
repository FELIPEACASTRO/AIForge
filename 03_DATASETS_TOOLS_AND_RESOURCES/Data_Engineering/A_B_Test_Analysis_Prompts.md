# A/B Test Analysis Prompts

## Description
A/B Test Analysis Prompts are structured instructions provided to Large Language Models (LLMs) to interpret, narrate, and generate actionable *insights* from raw A/B experiment data. Rather than replacing traditional statistical analysis, the LLM acts as a narrative and governance layer, translating complex metrics (such as *lift*, confidence intervals, and *p*-values) into clear guidance for executive, product, and growth decision-making. The main goal is to accelerate the *insight* cycle, improve cross-functional alignment, and ensure that narrative conclusions reflect the underlying statistical rigor. It is crucial that the prompt requires the LLM to stick to the provided statistical results, avoiding hallucinations or over-interpretation. This technique is fundamental for Experimentation and Conversion Rate Optimization (CRO) teams.

## Examples
```
**1. Statistical and Narrative Analysis (Complete)**
"Act as a Senior Data Scientist. Analyze the following A/B Test results. Hypothesis: Version B will increase the Conversion Rate (CVR) by 5%. Data: Version A (Control): 50,000 users, 1,500 conversions (CVR 3.0%). Version B (Variant): 50,000 users, 1,650 conversions (CVR 3.3%). Statistical Significance: 95% confidence, *p*-value of 0.02.
1. Determine whether Version B is the winner based on significance.
2. Calculate the exact percentage *lift*.
3. Provide a 3-paragraph narrative for the C-Level, explaining the result, the business impact, and the 3 recommended next actions."

**2. Segment Investigation (Drill-Down)**
"The overall A/B Test failed to reach significance (p=0.15). However, we suspect an effect in the 'New Users' segment. Provide the data for both segments (New Users: A=100 conversions/5k visits, B=150 conversions/5k visits; Returning Users: A=1,400 conversions/45k visits, B=1,500 conversions/45k visits).
1. Calculate the *lift* and significance (assume a Z-test) for the 'New Users' segment.
2. Explain the discrepancy between the overall result and the segment result.
3. Suggest a segmentation strategy for the rollout of Version B."

**3. Risk Report and Metric Conflict**
"The A/B Test of the new *checkout* page (Version B) showed a 10% increase in the primary metric (Purchase Completion Rate) with 99% confidence. However, the secondary metric (Subscription Cancellation Rate) also increased by 2%.
1. Draft a risk alert for the Product team.
2. Suggest 3 hypotheses for the increase in the Cancellation Rate.
3. Propose a follow-up test to mitigate the secondary risk."

**4. Prompt Optimization (Meta-Analysis)**
"Analyze the 5 versions of prompts we used to generate product descriptions. The success metric was the Click-Through Rate (CTR) on the 'Buy Now' link.
- Prompt 1 (Benefit Focus): CTR 4.5%
- Prompt 2 (Urgency Focus): CTR 5.1%
- Prompt 3 (Feature Focus): CTR 3.9%
Based on these results (assume significance), what is the most effective *copywriting* principle? Create a new 'Master Prompt' that combines the best elements of Prompt 2 and Prompt 1."

**5. Interpretation of Bayesian Results**
"Act as a statistician. I received the results of a Bayesian A/B test. The 'Probability of B Being Better' is 98.5%. The 'Expected Uplift' is 6.2%.
1. Explain what these two numbers mean to a non-technical marketing manager.
2. What is the risk of implementing Version B?
3. What would be the sample size needed to reach 99.9% Probability of B Being Better, keeping the current *uplift*?"
```

## Best Practices
**1. Provide the Full Context:** Include the initial hypothesis, the experimental unit, the primary and secondary metrics, the statistical method used (e.g., frequentist or Bayesian), and the pre-specified significance threshold. **2. Clear Data Structure:** Present the test data (impressions, clicks, conversions, revenue, etc.) in a structured format (Markdown table, CSV, or JSON) to avoid interpretation errors. **3. Separate Fact from Inference:** Ask the LLM to clearly distinguish between conclusions based on statistical significance and qualitative inferences (the "why" behind the result). **4. Ask for Next Steps:** Do not settle for the conclusion alone. Request actionable recommendations for the next round of tests or for implementing the winning variant. **5. Use the Expert Persona:** Start the prompt by instructing the LLM to act as a "Senior Data Scientist" or "Conversion Rate Optimization (CRO) Specialist".

## Use Cases
nan

## Pitfalls
**1. Confusing Narrative with Statistical Rigor:** Blindly trusting the LLM's interpretation without verifying the underlying statistical data. The LLM is a narrator, not a statistical engine. **2. Lack of Context:** Failing to provide the hypothesis, metrics, and statistical method. This leads to generic and potentially incorrect analyses. **3. Over-interpretation of Non-Significant Results:** Asking the LLM to find deep *insights* in a test that failed to reach significance, which can lead to false conclusions (*false positives*). **4. Ignoring Secondary Metrics:** Focusing only on the primary metric and not asking the LLM to analyze the impact on secondary or guardrail metrics (e.g., revenue per user, bounce rate). **5. Data Hallucination:** The LLM may "invent" data or statistics if the prompt is too vague or if the context is insufficient. Always provide the raw or summarized data explicitly.

## URL
[https://www.gurustartups.com/reports/using-chatgpt-for-a-b-test-result-analysis](https://www.gurustartups.com/reports/using-chatgpt-for-a-b-test-result-analysis)
