# Pricing Strategy Prompts

## Description
**Pricing Strategy Prompts** are a set of structured, detailed instructions provided to language models (LLMs) to assist in the analysis, creation, and optimization of pricing strategies for products or services. They turn AI into a pricing consultant, capable of processing complex data (such as competitive analysis, price elasticity, costs, and perceived value) and generating actionable recommendations. This technique is widely used in business contexts, especially in SaaS (Software as a Service) and e-commerce, where dynamic and value-based pricing is crucial for maximizing revenue and profit margin [1] [2]. The effectiveness of these prompts lies in their ability to integrate principles of economics, behavioral psychology, and data analysis into a single query, requiring the AI to act in a specific role (persona) and use recognized business frameworks (such as Jobs-to-Be-Done or ReAct Prompting) [1] [2].

## Examples
```
**1. Competitive Analysis and Price Positioning:**
"Act as a senior pricing analyst. Compile a comparative pricing table for the top 5 competitors in the 'Project Management Software for SMBs' niche. Include: plan name, monthly/annual price, main value features, and discount tactics (if any). Based on this analysis, suggest an initial price point for our new 'Pro' plan that maximizes acquisition while maintaining a 40% margin."

**2. Value-Based Pricing Calculation:**
"Our marketing automation software saves the customer an average of 10 work hours per month. If the average hourly cost of the target employee is R$ 50.00, calculate an ideal monthly subscription price using the value-based pricing model. Assume we capture 20% of the value saved. Present the calculation step by step and justify the value capture rate."

**3. Price Elasticity Simulation:**
"Create a price elasticity simulation scenario for our product 'Personal Finance Ebook.' The current price is R$ 99.00 and we sell 500 units per month. Simulate the impact on revenue and profit if the price is adjusted by ±10% and ±20%. What is the price that maximizes total revenue, assuming a demand elasticity of -1.5?"

**4. Tiered Pricing Architecture Design:**
"Design a three-tier pricing structure (Basic, Premium, Enterprise) for a B2B video streaming service. Define the main differentiator of each tier (for example, number of users, video quality, support), the recommended psychological price for each, and an 'upgrade nudge' to move customers from Basic to Premium."

**5. Psychological Price Optimization:**
"Audit our SaaS product's current pricing page (URL: [Page URL]). Apply the 'Charm Pricing' technique (prices ending in 9) and 'Price Anchoring' for the 'Premium' plan of R$ 199.00. Suggest the new price and the anchoring copy that should be used to increase the perception of value and conversion rate."

**6. Discount and Promotion Strategy:**
"Develop a discount policy for our consulting service. Specify the types of discounts allowed (volume, loyalty, seasonal), the maximum discount ceiling for new customers (in %), and the guidelines to avoid brand devaluation. Present the result as a discount governance guide."

**7. Willingness-to-Pay Mapping:**
"Using the Jobs-to-Be-Done framework, define two distinct personas for our meditation app. For each persona, estimate a 'willingness-to-pay' (WTP) range justified by their functional and emotional 'jobs.' Also identify the 'perceived price fairness' triggers for each one."
```

## Best Practices
**1. Detailed Contextualization:** Always provide the AI with as much context as possible, including the business problem, market niche, target audience (ICP), and geographic context. Prompts such as "Act as a senior pricing analyst" define the AI's role and improve the quality of the response [2].
**2. Providing Data:** Integrate real or hypothetical but realistic data, such as margin reports, churn data, cost per work hour, or demand elasticity. The AI works best when it has "input data" to analyze and process [1].
**3. Specifying the Output Format:** Request the result in a structured format (comparative table, step-by-step calculation, checklist, decision tree diagram) to ensure the output is directly actionable and easy to apply [2].
**4. Focus on Value and Psychology:** Direct the AI toward advanced pricing frameworks, such as **Value-Based Pricing** and **Behavioral Pricing Theory**, to go beyond cost-based pricing [2].
**5. Iteration and Refinement:** Use the prompts as a starting point. The AI's result should be a draft or recommendation that needs to be validated and refined by a human pricing expert [1].

## Use Cases
**1. Revenue Optimization in SaaS:** Help Software as a Service companies define tiered pricing models, usage metrics, and price points that maximize MRR (Monthly Recurring Revenue) and minimize churn [2].
**2. Competitiveness Analysis in E-commerce:** Use sales and competition data (Amazon, Etsy, etc.) to identify mispriced products and recommend adjustments to increase competitiveness and profit margin [1].
**3. New Product Launches:** Determine the ideal entry price for a new product or service, using value-based pricing (VBP) calculation and customer willingness-to-pay (WTP) mapping [2].
**4. Structuring Discounts and Promotions:** Create discount governance policies to avoid brand devaluation, defining discount ceilings and allowed promotion types (volume, loyalty, seasonal) [2].
**5. International Expansion:** Develop price localization matrices for different global markets, adjusting values for Purchasing Power Parity (PPP) and considering taxes and cultural perceptions [2].

## Pitfalls
**1. Relying Solely on AI:** The biggest mistake is accepting the AI's price recommendation without human validation and A/B testing. AI may not capture cultural or regulatory nuances or real-time market dynamics [1].
**2. Lack of Context and Data:** Using generic prompts without providing specific data (costs, margins, customer data, competition) leads to superficial and ineffective responses. The quality of the output is directly proportional to the quality of the *input* [2].
**3. Ignoring Price Psychology:** Focusing only on numbers and costs, ignoring the impact of psychological pricing techniques (such as anchoring, the decoy effect, or prices ending in 9), which are crucial for the perception of value [2].
**4. Not Specifying the Format:** Not requesting a structured output format (table, list, calculation) results in long paragraphs of text that are difficult to extract and apply directly to the business [2].
**5. Disregarding Localization:** For global products, not including price localization (adjustments for Purchasing Power Parity - PPP, local taxes) can lead to uncompetitive prices or market arbitrage [2].

## URL
[https://medium.com/@slakhyani20/10-chatgpt-prompts-for-pricing-strategy-creation-8ecb05e47d68](https://medium.com/@slakhyani20/10-chatgpt-prompts-for-pricing-strategy-creation-8ecb05e47d68)
