# User Journey Mapping Prompts

## Description
**User Journey Mapping Prompts** are a category of *Prompt Engineering* techniques focused on leveraging Large Language Models (LLMs) to simulate, analyze, and optimize a customer's or user's experience with a product or service.

Rather than simply generating content, these prompts instruct the AI to act as a specialist (e.g., Omnichannel Consultant, Service Designer, Data Scientist) to break the journey down into its essential components: **stages**, **touchpoints**, **actions**, **thoughts**, **emotions**, and **pain points**.

The main objective is to transform raw data or scenario descriptions into a structured, actionable map, allowing UX, Product, and Marketing teams to identify gaps, moments of truth, and opportunities for innovation. The technique is particularly powerful for simulating complex scenarios (e.g., accessibility, comparison with competitors) and for quickly generating the first version of a map, which can then be refined by humans.

## Examples
```
**1. End-to-End Journey Blueprint:**
`"Build a detailed customer journey map for [PRODUCT/SERVICE] focused on the persona [PERSONA DESCRIPTION]. Divide the journey into: Awareness, Consideration, Purchase, Onboarding, Usage, and Retention. For each stage, list: (a) customer goals, (b) touchpoints, (c) emotions, (d) pain points, and (e) measurable KPIs. Conclude with the three biggest 'moments of truth' and suggested actions to optimize them."`

**2. Persona-Specific Scenario (Deep-Dive):**
`"Create a granular scenario for the persona 'John, 35, Marketing Manager' purchasing [PRODUCT]. Describe step by step what he thinks, feels, and does at each touchpoint, from the first moment of awareness to one month after purchase. Highlight the emotional highs and lows. Finish with a table summarizing the friction points and quick-win fixes."`

**3. Omnichannel Touchpoint Audit:**
`"Assume the role of an omnichannel consultant. List all the online and offline touchpoints a customer encounters when interacting with [BRAND] (website, app, social media, email, phone support). For each touchpoint, specify its primary purpose, success metric, typical problems, and one recommended Customer Experience (CX) improvement, ranked by impact vs. effort."`

**4. Emotion Curve Visualization:**
`"Imagine you are a service designer tracing an emotion curve. Describe, in sequence, the customer's emotional intensity (-5 to +5) at each step of subscribing to and using [SUBSCRIPTION SERVICE]. Provide a narrative explanation for each data point and recommend design interventions to flatten negative valleys and amplify positive peaks."`

**5. Future-State Journey Ideation:**
`"You are facilitating a design sprint. Imagine a 12-month 'future-state' journey for [BRAND] that eliminates today's three main pain points: [PAIN 1], [PAIN 2], and [PAIN 3]. Describe innovative touchpoints and technologies (e.g., AI chat, predictive support) introduced at each stage and explain how they transform the customer experience. Provide an implementation roadmap prioritized by ROI."`

**6. Data-Driven Optimization (SaaS Funnel):**
`"Act as a data scientist optimizing our SaaS onboarding journey. Funnel data: Sign-up [70%], First Moment of Value [40%], Activation [25%]. Identify the two stages with the largest drop-off. Hypothesize the root causes, design three A/B test ideas to address them, and define the success metrics for each test."`

**7. Accessibility and Inclusion Review:**
`"Evaluate the journey of a customer with [ACCESSIBILITY NEED, e.g., visual impairment] using our e-commerce site. Detail the barriers encountered during product discovery and checkout. Recommend WCAG-compliant fixes and inclusive design improvements, indicating quick wins versus long-term improvements."`
```

## Best Practices
**1. Provide Detailed Context:** Always begin the prompt by defining the problem, the product/service, and the target persona. The more context (such as funnel data, current journey stages, or business pain), the richer the AI's output.

**2. Specify the Output Format:** Explicitly ask for the desired format (e.g., "Deliver the result in a Markdown table with 5 columns", "Use an emotion curve from -5 to +5", "Conclude with an action playbook").

**3. Use Known Frameworks:** Integrate design and business frameworks (such as **Jobs-to-Be-Done**, **Omnichannel**, or **WCAG** for accessibility) to guide the AI toward a more structured and professional result.

**4. Focus on Specific Pain Points:** Instead of mapping the entire journey, use prompts to focus on critical stages (e.g., "Post-Purchase", "SaaS Onboarding") or specific problems (e.g., "Checkout abandonment rate").

**5. Request Optimized Actions:** Do not ask only for the map; ask for optimization suggestions, such as "the three biggest 'moments of truth'", "improvements ranked by impact vs. effort", or "A/B test ideas".

## Use Cases
**1. Product Design and UX:**
*   **Gap Identification:** Reveal friction points and moments of frustration that lead to abandonment or *churn*.
*   **Feature Prioritization:** Use the AI's "impact vs. effort" analysis to decide which UX improvements to develop first.

**2. Marketing and Sales:**
*   **Content Creation:** Map the customer's thoughts and emotions at each stage to create more resonant and targeted marketing messages.
*   **Funnel Optimization:** Use data-driven prompts to identify the highest-drop-off stages in the sales funnel and suggest A/B tests.

**3. Business Strategy and Innovation:**
*   **Competitive Benchmarking:** Compare the customer journey with that of competitors to identify strategic advantages and opportunities to "leapfrog" in customer experience (CX).
*   **Future-State Ideation:** Create long-term visions (12-18 months) for the customer experience, incorporating new technologies (AI, AR/VR) and eliminating current pain points.

**4. Customer Support and Retention:**
*   **Post-Purchase Design:** Create the ideal playbook for the first 90 days after purchase, focusing on activation, proactive support, and reducing *churn* risk.
*   **Accessibility Review:** Evaluate the journey from the perspective of users with specific needs (e.g., visual impairment) to ensure compliance and inclusion.

## Pitfalls
**1. Lack of Specific Context:** The most common mistake is using generic prompts. The AI cannot map a useful journey without details about the **product**, the **persona**, and the **business problem** being solved.

**2. Confusing the Map with Reality:** The map generated by the AI is a **structured hypothesis**, not the absolute truth. It is a mistake to use it without validation through real user research (interviews, analytics data).

**3. Ignoring the Voice of the Customer (VoC):** Not including qualitative data (interview quotes, support complaints) in the prompt results in a sterile map based on generic assumptions.

**4. Excessive Focus on Positive Stages:** The AI may tend to over-optimize the emotional "highs". The real value of mapping lies in identifying and resolving the "lows" (pain points and friction).

**5. Not Specifying the Format:** Asking only "Create a journey map" without defining the structure (table, list, output format) can lead to a disorganized and hard-to-use response.

**6. Not Asking for Actions:** A map without optimization actions is just a descriptive document. The mistake is not explicitly requesting that the AI suggest interventions and priorities.

## URL
[https://medium.com/@slakhyani20/10-chatgpt-prompts-for-customer-journey-mapping-14c667b1b451](https://medium.com/@slakhyani20/10-chatgpt-prompts-for-customer-journey-mapping-14c667b1b451)
