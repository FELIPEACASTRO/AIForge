# Legal Research Prompts

## Description
Legal Prompt Engineering is the process of creating precise and detailed instructions for Generative Artificial Intelligence (GenAI) systems in order to obtain accurate, relevant, and useful responses for legal tasks such as case law research, document analysis, drafting pleadings, and summarizing regulations. It is a crucial skill for lawyers seeking to maximize the efficiency and accuracy of AI tools while mitigating risks such as "hallucinations" (factually incorrect responses) [1] [2]. Precision in formulating the prompt is what distinguishes a generic result from an actionable and verifiable legal work product [3]. The main focus is clarity, the definition of a role (role-playing), and the requirement of a logical reasoning structure (such as the IRAC format) to ensure the mandatory human verification of the final product [1].

## Examples
```
**1. Case Law Research and Summarization (IRAC)**

> **Prompt:** "Act as a senior Consumer Law attorney. Analyze the case [Case Name/Case Number] and apply the IRAC format (Issue, Rule, Application, Conclusion). The focus should be on the **rule of law** used to determine liability for moral damages in cases of improper credit blacklisting. Cite the specific article of the Consumer Protection Code (CDC) and summarize the decision in up to 300 words."

**2. Contract Clause Analysis**

> **Prompt:** "You are the legal counsel for a SaaS startup. Review Clause 7 ('Indemnification') of the [Attach the Contract/Paste the Clause] and answer: Is the clause favorable to my client (the Licensor)? What are the three greatest liability risks it imposes? Rewrite the clause in plain, non-legal language while preserving the same legal effect, so that the sales team can understand it."

**3. Drafting a Court Pleading**

> **Prompt:** "Act as a civil attorney with 10 years of experience. Draft an initial petition for an eviction action for non-payment. Include the sections 'Facts', 'Legal Grounds' (citing the Tenancy Law and the Code of Civil Procedure), and 'Requests'. The amount in controversy is R$ [Amount]. The defendant has been delinquent for 4 months. Maintain a formal and persuasive tone."

**4. Summarizing a Complex Regulation**

> **Prompt:** "You are a regulatory compliance specialist. Analyze the [Specific Regulation, e.g., General Data Protection Law - LGPD]. Summarize the **five main obligations** of a Data Controller. For each obligation, cite the exact article of the LGPD and provide a practical example of how a medium-sized company should comply with that obligation. The output should be a concise table."

**5. Generating Interview Questions**

> **Prompt:** "Act as a labor attorney. I am about to interview a former employee who alleges workplace harassment. Generate a list of 10 strategic, non-leading questions for the interview, focused on gathering objective facts and avoiding admissions of guilt. Include a brief legal justification for each question."

**6. Comparing Case Law**

> **Prompt:** "Compare and contrast recent decisions of the Superior Court of Justice (STJ) on the application of the loss-of-a-chance doctrine in medical malpractice cases. Identify at least two divergent rulings (if any) and highlight the criteria the STJ has used to quantify the compensation. The result should be a comparative analysis, not just a summary."
```

## Best Practices
**1. Define the AI's Role (Role-Playing):** Begin the prompt by instructing the AI to act as a specific professional (e.g., "Act as a senior civil attorney", "You are the legal counsel for a high-growth startup") to ensure the response adopts the correct tone and perspective [1] [4].

**2. Use the IRAC Format (Issue, Rule, Application, Conclusion):** Explicitly ask the AI to structure its legal analysis using the IRAC format. This allows the user to easily diagnose the AI's chain of reasoning and verify the accuracy of each step [1].

**3. Provide Broad Context, but Do Not Steer the Answer:** Include as much relevant context and facts as possible before asking the question. However, frame open-ended questions to avoid influencing the result (what is known as "leading the witness") [1].

**4. Require Verifiable Citations:** Always request that the AI cite the specific section of the law, regulation, or case law. Even if the AI's citations may be incorrect (hallucinations), they provide a starting point for human verification, which is mandatory [1] [5].

**5. Iterate and Refine the Prompt:** If the initial response is not satisfactory, use the AI itself to refine the prompt. Ask: "Based on your previous response, which key concepts can I adjust or refine to change the result?" [1].

**6. Use Plain Language for the Output:** Ask the AI to summarize or explain complex concepts in "plain, non-legal language" to facilitate communication with clients or non-legal parties [1].

## Use Cases
nan

## Pitfalls
**1. Hallucinations and False Citations:** The most serious error. The AI may invent cases, laws, or citations that appear real but are factually incorrect. **Solution:** Never rely on a citation without verifying it against the primary source [2] [5].

**2. Overly Generic Prompts:** Asking to "draft a petition" or "summarize the law" without providing context, role, jurisdiction, or a specific objective. This results in vague and useless responses [4].

**3. Leakage of Confidential Information:** Entering sensitive client or case data into general-purpose AI models (such as the public ChatGPT) that do not guarantee confidentiality. **Solution:** Use only secure legal AI tools or private/local models for confidential data [1].

**4. Failure to Define the Role:** Not specifying the AI's role (e.g., lawyer, judge, consultant). The AI may assume a "personal assistant" role and provide answers that prioritize user satisfaction over legal accuracy [1].

**5. Blind Trust in the Response:** Treating the AI's output as a final product. The AI should be seen as a "junior lawyer" or assistant whose work must always be reviewed, verified, and validated by a human professional [1] [2].

## URL
[https://www.lsuite.co/blog/mastering-ai-legal-prompts](https://www.lsuite.co/blog/mastering-ai-legal-prompts)
