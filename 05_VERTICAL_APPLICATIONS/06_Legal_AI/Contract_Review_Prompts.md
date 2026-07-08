# Contract Review Prompts

## Description
The **Contract Review Prompts** technique is a set of structured, detailed instructions provided to a Large Language Model (LLM) so that it performs analysis, summarization, risk identification, legal compliance verification, and optimization suggestions on contractual documents. This technique turns the LLM into a virtual legal assistant, allowing legal and business professionals to dramatically accelerate the *due diligence* and contract negotiation process, focusing on critical points such as abusive clauses, information gaps, and alignment with applicable legislation (e.g., LGPD, CDC) [1]. Its effectiveness lies in the ability to assign a **persona** (e.g., corporate law attorney), define the **objective** (e.g., identify risks for the contracting party), and request a specific **output format** (e.g., comparative table), increasing the accuracy and usefulness of the analysis [2].

## Examples
```
**1. Risk and Abusiveness Identification (Specific Persona)**
`Act as a consumer law attorney. Analyze the following adhesion contract and identify 3 clauses that may be considered abusive or that represent a high risk for the consumer. Justify each point and suggest more balanced wording. [Paste contract]`

**2. Summarization and Plain Language**
`Act as a plain-language specialist. Summarize the following service agreement in 10 bullet points, focusing exclusively on the rights and duties of my party (Contractor), payment terms, and termination conditions. Use simple, direct language. [Paste contract]`

**3. Legal Compliance Verification (LGPD)**
`Act as a legal consultant specialized in LGPD. Analyze the following partnership agreement and identify whether it is in full compliance with the General Data Protection Law (LGPD) regarding the processing of personal data. Point out 2 points of non-compliance and propose the ideal corrective clause. [Paste contract]`

**4. Clause Optimization for Negotiation**
`Act as a legal negotiator. In the 'Intellectual Property' clause of the software development agreement, which favors the Contracting party, suggest 3 alternative wording options that make it more balanced or advantageous for my party (Developer). Explain the legal impact of each option.`

**5. Identifying Gaps and Omissions**
`Act as a contract auditor. Analyze the following real estate purchase and sale agreement. Which 3 pieces of information or clauses crucial to the buyer's security are missing, considering Brazilian legislation? Suggest where they should be inserted in the document.`

**6. Master Prompt for Complete Analysis**
`**Persona:** Attorney specialized in corporate law and AI. **Contract Type:** Service Agreement (Digital Marketing). **My Role:** Contractor. **Mission:** Analyze the following contract and: a) Identify 3 high-risk or ambiguous points for the Contractor; b) For each point, propose 2 alternative wording options that protect me more; c) Suggest 3 crucial questions to ask a human attorney. **Format:** Structured table. [Paste contract]`
```

## Best Practices
**1. Define the Legal Persona:** Start the prompt by instructing the AI to act as a specific specialist (e.g., "Act as a consumer law attorney" or "jurist experienced in M&A"). This aligns the focus of the analysis and the language of the response. **2. Specify the Objective and Format:** Be clear about what you want (e.g., "Identify 3 high-risk clauses" or "Summarize in 10 bullet points"). Request the result in a structured format, such as a table, to facilitate reading and comparison. **3. Provide Context and Role:** Indicate what your role is in the contract (Contracting party, Contractor, Buyer) and the type of contract. This allows the AI to assess risk from your perspective. **4. Iterate and Refine:** Use the result of the first analysis for follow-up prompts. For example, after identifying a risky clause, ask a new prompt to "suggest 3 alternative wording options" for that specific clause. **5. Use the Master Prompt:** For complex reviews, use a "Master Prompt" that includes all the elements of context, objective, and format in a single structure, such as the provided example, to ensure a complete and multifaceted analysis [1].

## Use Cases
**1. Rapid *Due Diligence*:** Initial analysis of large volumes of contracts in mergers and acquisitions (M&A) or audits, to quickly identify liabilities and risks. **2. Clause Optimization:** Generating suggestions for wording more favorable to the user's party, strengthening the negotiating position. **3. Compliance Verification:** Ensuring that contracts are aligned with new regulations (e.g., LGPD, sector-specific laws) before signing. **4. Training and Education:** Using prompts to simulate negotiation scenarios and train new attorneys or sales teams on the critical points of a contract. **5. Standardization:** Creating an automated review *checklist* to ensure that all contracts of the same type (e.g., NDA, Terms of Service) contain the essential and protective clauses [1] [2].

## Pitfalls
**1. Blind Trust (Hallucinations):** The biggest mistake is blindly trusting the AI's output. LLMs can **"hallucinate"** (invent) legal precedents, statutes, or interpretations. Final review by a human professional is indispensable. **2. Lack of Context:** Failing to specify the type of contract, the user's role (Contracting party/Contractor), or the legal jurisdiction (e.g., Brazil, USA) leads to generic and irrelevant analyses. **3. *Token* Limitation:** Very long contracts may exceed the model's *token* limit, resulting in incomplete or truncated analyses. It is necessary to divide the document into sections for analysis. **4. Absence of Attached Documents:** The AI cannot analyze attached documents or external references that were not included in the prompt, which can lead to an incomplete assessment of contractual risk. **5. Vague Prompts:** Requests such as "Review this contract" are ineffective. The lack of a clear objective (e.g., "focus on penalties", "focus on termination") results in superficial output [3].

## URL
[https://treinamentosaf.com.br/contratos-a-prova-de-falhas-7-prompts-ia-revisam-em-minutos-2025/](https://treinamentosaf.com.br/contratos-a-prova-de-falhas-7-prompts-ia-revisam-em-minutos-2025/)
