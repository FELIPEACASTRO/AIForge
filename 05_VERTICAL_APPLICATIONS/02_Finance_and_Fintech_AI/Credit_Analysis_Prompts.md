# Credit Analysis Prompts

## Description
**Credit Analysis Prompts** are structured and detailed instructions, often incorporating advanced Prompt Engineering techniques such as **Labeled Guide Prompting (LGP)** [3], used to guide Large Language Models (LLMs) in performing complex credit risk assessment tasks. The central objective is to convert structured financial data (such as balance sheets, income statements, and credit scores) into a natural-language description that the LLM can process, allowing it to perform everything from binary risk classification (High/Low) [1] to the generation of complete, justified, and interpretable risk reports [3]. This approach focuses on leveraging the reasoning and text generation capabilities of LLMs to increase the transparency and reliability of the credit decision process, aligning it with regulatory and audit requirements [2].

## Examples
```
### Example 1: Risk Classification with Explicit Reasoning (CoT)
**Role:** You are a senior credit risk analyst.
**Instruction:** Analyze the applicant's data and determine whether the default risk is **HIGH** or **LOW**.
**Applicant Data:**
- Credit Score (FICO): 680
- Annual Income: R$ 120,000
- Debt-to-Income (DTI): 35%
- History of Late Payments (last 2 years): 2
**Reasoning Process (Step by Step):**
1. Evaluate the Credit Score (680).
2. Evaluate the DTI (35%).
3. Evaluate the Payment History.
4. Conclude the final risk and justify it.
**Required Output:** Only the word **HIGH** or **LOW**.

### Example 2: Structured Risk Report Generation (LGP)
**Role:** You are a corporate credit risk analysis expert.
**Instruction:** Generate a detailed risk report for the company "TechCorp S.A.", using the Labeled Guide Prompting technique to ensure completeness.
**Company Data:**
- Net Revenue (2024): R$ 50M
- Net Profit (2024): R$ 5M
- Current Ratio: 1.2
- Sector: Technology (High Growth, High Volatility)
**Items to Be Addressed (LGP):**
- **[QUANTITATIVE_ANALYSIS]:** Assessment of financial indicators and credit score.
- **[QUALITATIVE_ANALYSIS]:** Assessment of the sector, management, and macroeconomic context.
- **[FINAL_RISK]:** Risk classification (A, B, C, D) and suggested credit limit.
- **[REGULATORY_JUSTIFICATION]:** Explanation of how the decision aligns with Central Bank Circular X.

### Example 3: Interpretability (XAI) and Influencing Factors
**Role:** You are a credit risk AI model focused on interpretability (XAI).
**Instruction:** Based on the **HIGH** risk decision for the applicant, identify and describe the 3 factors that contributed most to this classification, as if they were "SHAP values" in natural language.
**Risk Decision:** HIGH
**Applicant Data:** [Insert complete data]
**Required Output:**
1. **Main Factor:** [Description of the factor and its impact]
2. **Second Factor:** [Description of the factor and its impact]
3. **Third Factor:** [Description of the factor and its impact]

### Example 4: Analysis of Unstructured Documents
**Role:** You are a credit risk data extractor.
**Instruction:** Read the excerpt from the articles of association and extract the clauses that represent a potential risk to granting credit.
**Excerpt from the Articles of Association:** [Insert excerpt]
**Required Output (JSON):**
```json
{
  "clausulas_de_risco": [
    {"clausula": "Cláusula 4.1", "risco_associado": "Restrição de Venda de Ativos"},
    {"clausula": "Cláusula 7.3", "risco_associado": "Subordinação de Dívida"}
  ]
}
```

### Example 5: Scenario Simulation (Stress Test)
**Role:** You are a credit risk modeler.
**Instruction:** Simulate the impact of a 5 percentage point increase in the interest rate (Stress Scenario) on the applicant's DTI and probability of default.
**Applicant Data:** [Insert complete data, including loan amount and current rate]
**Required Output:**
- **DTI in the Base Scenario:** [Value]%
- **DTI in the Stress Scenario:** [Value]%
- **Probability of Default (Stress):** [Value]%
- **Recommendation:** [Maintain/Review/Deny]
```

## Best Practices
| Practice | Description | Source |
| :--- | :--- | :--- |
| **Labeled Guide Prompting (LGP)** | Decompose the task into labeled sub-tasks (e.g., `[QUANTITATIVE_ANALYSIS]`) to ensure the LLM addresses all dimensions of the problem (what, why, how), promoting abductive reasoning and completeness [3]. | [3] |
| **Chain-of-Thought Reasoning (CoT)** | Require step-by-step analysis and human-like justifications for risk decisions. This increases transparency, facilitates auditing, and reduces the error rate on complex queries by about 20% [2]. | [1], [2] |
| **Strict Output Control** | Specify a strict output format (JSON, XML, or labeled text) to facilitate automated integration with credit systems and ensure quality control. | [3] |
| **Regulatory Context** | Include in the prompt the need to adhere to regulatory frameworks (e.g., Basel III) to ensure the compliance of the analysis and the accuracy of the model [2]. | [2] |
| **Few-Shot Learning** | Provide annotated examples of successful and unsuccessful credit analyses to refine the model's behavior and increase accuracy. | [3] |

## References
[1] Chen, Q. (2025). Explore the Use of Prompt-Based LLM for Credit Risk Classification. *Journal of Computer and Communications*, 13, 33-46.
[2] Joshi, S. (2025). Leveraging Prompt Engineering to Enhance Financial Market Integrity and Risk Management. *World Journal of Advanced Research and Reviews*, 25(01), 1775-1785.
[3] Teixeira, A. C., et al. (2023). Enhancing Credit Risk Reports Generation using LLMs: An Integration of Bayesian Networks and Labeled Guide Prompting. *4th ACM International Conference on AI in Finance*.

## Use Cases
1. **Automated Risk Classification:** Quickly determine the probability of default of a borrower (individual or corporate) for initial screening [1].
2. **Risk Report Generation:** Create detailed and justified reports for analysts and credit committees, with greater reliability and acceptance by human analysts [3].
3. **Interpretability (XAI):** Generate clear explanations of the factors that most influence the risk decision, converting complex metrics (such as SHAP values) into natural language.
4. **Unstructured Data Analysis:** Process documents such as financial statements, contracts, news, and emails to extract relevant risk features.
5. **Stress Testing:** Assess the impact of adverse macroeconomic scenarios (interest rate increases, recession) on the credit portfolio.

## Pitfalls
1. **Financial Hallucinations:** The risk of the LLM generating factually incorrect data or analyses is high and critical in the financial sector, requiring rigorous validation of the output.
2. **Bias and Unfairness:** The model may perpetuate or amplify biases present in the training data, leading to discriminatory or unfair credit decisions.
3. **Lack of Interpretability (Black Box):** Without the requirement of explicit reasoning (CoT), the LLM's decision can be opaque, hindering auditing and regulatory acceptance.
4. **Prompt Complexity:** Excessively long or complex prompts can confuse the model or exceed the token limit, resulting in truncated or irrelevant outputs.
5. **Regulatory Misalignment:** Failing to incorporate regulatory context into the prompt can lead to non-compliant analyses, exposing the institution to legal risks.

## URL
[https://dl.acm.org/doi/fullHtml/10.1145/3604237.3626902](https://dl.acm.org/doi/fullHtml/10.1145/3604237.3626902)
