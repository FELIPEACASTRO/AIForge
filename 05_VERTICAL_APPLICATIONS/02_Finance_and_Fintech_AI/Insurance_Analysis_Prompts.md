# Insurance Analysis Prompts

## Description
Prompt engineering for insurance analysis refers to the creation and optimization of instructions (prompts) for Large Language Models (LLMs) with the goal of automating and enhancing tasks specific to the insurance sector. This includes analyzing policy documents, processing claims, risk assessment (underwriting), fraud detection, and customer interaction. The key is to provide the LLM with sufficient context and structure so that it acts as an insurance expert, ensuring accuracy and regulatory compliance. Accuracy and regulatory compliance are critical aspects that require the use of advanced prompting techniques, such as Chain-of-Thought (CoT), to ensure the traceability and justification of decisions.

## Examples
```
### Example 1: Policy Summarization
**Role:** Senior Policy Analyst
**Instruction:** "You are a Senior Policy Analyst. Your task is to summarize the insurance policy provided below. The summary must be no more than 200 words and must mandatorily highlight: 1) The maximum coverage limit for 'Third-Party Damages', 2) The deductible amount for 'Collision', and 3) The two main coverage exclusions. Use the exact section of the policy to justify each point."
**Input:** `[Full policy text]`

### Example 2: Claim Evaluation
**Role:** Claims Adjuster
**Instruction:** "Act as a Claims Adjuster. Analyze the 'Claim Details' and the 'Policy Excerpt' provided. Determine whether the claim is covered. **Reasoning Steps (Chain-of-Thought):** 1. Identify the relevant coverage clause. 2. Check whether any exclusion applies. 3. Conclude eligibility. Your final answer must be 'COVERED' or 'NOT COVERED', followed by your step-by-step justification."
**Input:** `Claim Details: [Event description]`, `Policy Excerpt: [Relevant clauses]`

### Example 3: Proposal Comparison
**Role:** Insurance Advisor
**Instruction:** "You are an impartial Insurance Advisor. Compare the two auto insurance proposals below. Create a comparison table that includes: Annual Premium, Theft Coverage, Deductible, and Risk Rating (Low, Medium, High) based on the exclusions. Recommend the best option for a client who prioritizes the lowest total cost in the event of a claim."
**Input:** `Proposal A: [Details]`, `Proposal B: [Details]`

### Example 4: Underwriting Risk Analysis
**Role:** Risk Underwriter
**Instruction:** "Analyze the applicant's profile and claims history. Based on the data, assign a risk score from 1 (Low) to 10 (High) and justify the score. If the score is above 7, suggest an exclusion clause or a 15% premium increase."
**Input:** `Applicant Profile: [Age, Location, Property Type]`, `Claims History: [Date, Type, Amount]`

### Example 5: Denial Letter Generation
**Role:** Legal Communications Agent
**Instruction:** "Generate a formal claim denial letter for the client [Client Name]. The reason for denial is the 'Natural Disasters Exclusion Clause' (Section 4.B of the policy). The letter must be empathetic, cite the exact section of the policy, and inform the client about the appeals process."
**Input:** `Client Name: João Silva`, `Policy Number: 12345`, `Claim Date: 01/01/2025`
```

## Best Practices
1. **Role Assignment:** Begin the prompt by defining the LLM's role (e.g., "You are a senior property and casualty insurance underwriter").
2. **Structure and Variables:** Use structured prompt templates with clear variables (e.g., `{policy_text}`, `{claim_details}`) to ensure the model receives all necessary information.
3. **Reasoning Strategies:** Implement advanced prompting strategies such as **Chain-of-Thought (CoT)** for complex tasks (e.g., claim evaluation), asking the model to justify its reasoning step by step.
4. **Focus on Compliance and Accuracy:** Include explicit instructions for the model to cite the exact section of the policy or regulation that supports its conclusion.
5. **Rigorous Evaluation:** Use evaluation frameworks with domain-specific metrics (e.g., accuracy in clause extraction, regulatory compliance) and human review.

## Use Cases
1. **Underwriting and Risk Assessment:** Analyze insurance applicant data (claims history, demographic information) to determine eligibility and premium.
2. **Claims Processing:** Assess the validity of a claim, calculate the payout amount, and generate the decision letter by comparing the claim details with the policy terms.
3. **Policy Document Analysis:** Summarize complex policies, extract specific clauses (exclusions, deductibles), and compare different insurance proposals.
4. **Fraud Detection:** Analyze patterns in claim reports and supporting documents to identify anomalies or fraud indicators.
5. **Customer Service (Specialized Chatbots):** Answer complex customer questions about coverage and claim processes based on internal documents.

## Pitfalls
1. **Hallucinations and Inaccuracy:** The LLM may generate factually incorrect information or cite nonexistent clauses, which is critical in a regulated sector.
2. **Bias and Unfairness:** If the training data contains bias, the model may lead to unfair or discriminatory underwriting or claims decisions.
3. **Lack of Specific Context:** Overly generic prompts fail to capture the complexity and technical terminology of the insurance domain.
4. **Overreliance on Zero-Shot:** For high-risk tasks, relying solely on simple (zero-shot) prompts without CoT or external validation is dangerous.
5. **Data Insecurity (Data Leakage):** The use of sensitive customer data in prompts without proper anonymization or on non-secure models.

## URL
[https://github.com/ozturkoktay/insurance-llm-framework](https://github.com/ozturkoktay/insurance-llm-framework)
