# Discovery Prompts

## Description
Discovery Prompts are a Prompt Engineering technique focused on accelerating and deepening the **Product Discovery** and **User Research** phase. The core of the technique is instructing the Large Language Model (LLM) to take on the role of an expert in Lean Startup, Product Discovery, or UX Research. The main goal is to use AI to help **define the problem, generate solution hypotheses, create research plans (interviews, surveys, tests), and suggest low-cost, low-tech validation methods** ("processualize before productizing"). Rather than asking the AI to build the final solution, the Discovery Prompt uses it to structure the initial learning and validation process, minimizing the waste of resources on unvalidated solutions. It is an approach that prioritizes learning and risk reduction before full software development.

## Examples
```
**Example 1: Hypothesis Generation and Validation Plan (Role-Playing)**

**Instruction:** Act as a **senior Product Manager** specialized in Lean Startup. Our problem is: "Small business users spend a lot of time manually entering data into our management software, which causes a high abandonment rate after the first month."

**Prompt:** "Based on the problem, generate a clear **Value Hypothesis** (format: 'We believe that [action] for [audience] will result in [measurable outcome]'). Then, propose **three low-cost experiments** (e.g., Concierge MVP, Smoke Test Landing Page, Interviews) to validate this hypothesis before any software development. For each experiment, define the **success criterion**."

**Example 2: Interview Script Creation (User Research)**

**Instruction:** Act as a **UX Researcher** focused on in-depth interviews. Our hypothesis is: "The introduction of an automatic spreadsheet import feature will reduce initial setup time by 50% for new customers."

**Prompt:** "Create a **semi-structured interview script** with 8 open-ended questions to validate this hypothesis. The questions should focus on the **current pain** (how they do it today), the **need** (what they expect from a solution), and the **willingness to pay/use** (perceived value). Include a 'stress test' question to refute the hypothesis."

**Example 3: Competitor and Gap Analysis (Benchmark)**

**Instruction:** Act as a **Competitive Intelligence Analyst**. Our product is a meditation app. We want to discover what is missing in our offering for advanced users.

**Prompt:** "Identify **three direct and two indirect competitors** (e.g., puzzle games) in the digital wellness market. Analyze the **'discovery' features** (how the user finds new content) and the **retention strategies** for users who have used the app for more than 6 months. Present the results in a table, highlighting the **gaps** we can explore in our Discovery."

**Example 4: Defining Metrics for a Manual MVP**

**Instruction:** Act as a **Growth and Metrics Specialist**. We are manually testing a news curation service for executives (the 'Manual MVP').

**Prompt:** "What are the **three most important success metrics** for this Manual MVP, focused on learning and validation rather than scale? Define the **quantitative target** (e.g., 'X% of Y') for each metric that would indicate the value hypothesis has been validated and that we should 'productize' the service."

**Example 5: Hypothesis Refutation (Critical Thinking)**

**Instruction:** Act as a **Product Skeptic**. Our team is very excited about the idea of a '24/7 support bot'.

**Prompt:** "List **five critical reasons** why a 24/7 support bot might **fail** in our context (Small Service Businesses). For each reason, suggest a **research question** we should ask users to try to **refute** the bot idea before building it."

**Example 6: Structuring Quantitative Research (Survey)**

**Instruction:** Act as a **Quantitative Research Specialist**. We want to measure the frequency and intensity of the pain of 'managing supplier invoices' across our base of 500 customers.

**Prompt:** "Create a **mini-survey** with 5 questions (including Likert scale and multiple-choice questions) to measure the **frequency** and **severity** of this problem. Include a key demographic question for segmentation. The goal is to obtain data to prioritize this pain point on the roadmap."

**Example 7: Analysis of Existing Data (Support Feedback)**

**Instruction:** Act as a **Product Data Scientist**. We have 500 support tickets opened in the last month.

**Prompt:** "If I provide you with the text of these 500 tickets, which **five problem categories** would you suggest to group them? Which **three keywords** would you use to quickly identify whether the problem is related to 'usability', 'value', or 'technical viability'? The goal is to use AI to structure the analysis of qualitative data."
```

## Best Practices
1. **Persona/Role Definition (Role-Playing):** Start the prompt by instructing the AI to take on the role of an expert (e.g., "Act as a senior Product Manager...").
2. **Detailed Problem Context:** Provide as much context as possible about the problem, the target audience, and the business objective.
3. **Focus on Validation:** Explicitly ask the AI to suggest **validation methods** (e.g., "What are 3 low-cost experiments to validate this hypothesis?").
4. **Investigative Questions:** Use the prompt to generate questions that deepen understanding of the problem (e.g., "Generate 10 interview questions for users about this pain point.").
5. **Iteration and Refinement:** Use the AI's output as a starting point for follow-up prompts, refining the problem or the hypothesis.

## Use Cases
1. **Hypothesis Generation:** Create value, usability, and feasibility hypotheses for a new feature or product.
2. **Research Design:** Develop interview scripts, survey questionnaires, and usability test plans.
3. **Competitor Analysis (Benchmark):** Ask the AI to identify indirect competitors and their solutions to a specific problem.
4. **Metrics Definition:** Generate suggestions for success metrics (KPIs) for the validation phase (e.g., "Success Metrics for a Manual MVP").
5. **Problem Prioritization:** Use AI to analyze feedback data and suggest prioritization of user pain points.

## Pitfalls
1. **Confusing Discovery with Solution:** Asking the AI to "build the product" instead of "structuring the research on the problem".
2. **Lack of Context:** Providing a vague problem, resulting in generic and useless research suggestions.
3. **Accepting the Output as Absolute Truth:** The AI may suggest research methods that are not suited to the real context of the company or the product.
4. **Ignoring Manual Execution:** The value of the Discovery Prompt lies in structuring manual validation. The mistake is skipping validation and going straight to development.
5. **Confirmation Bias:** Using AI only to confirm a preconceived idea, rather than actively seeking to refute the hypothesis.

## URL
[https://calirenato82.substack.com/p/prompt-ia-discovery-operacionalizar-produtizar](https://calirenato82.substack.com/p/prompt-ia-discovery-operacionalizar-produtizar)
