# Prompt Engineering for Contract Review (Legal Tech)

## Description

Prompt Engineering techniques applied to the review and analysis of legal contracts. The focus is on using Large Language Models (LLMs) for tasks such as identifying key clauses, assessing risks, verifying legal compliance, and suggesting negotiation strategies. The prompt methodology is based on assigning a **role** (e.g., "experienced lawyer"), defining the **task** (e.g., "meticulously review"), specifying the **focus** (e.g., "harmful clauses"), and requesting a structured **output format** (e.g., "detailed analysis and amendment suggestions"). The adoption of AI for contract review is growing rapidly, with case studies indicating significant gains in speed and accuracy.

## Statistics

- **Adoption:** The adoption of AI for contract review has grown significantly, with some research indicating that AI adoption in legal practice nearly tripled from 11% in 2023 to 30% in 2024 [1]. Other data point to a 75% year-over-year growth in adoption for contract review [2].
- **Accuracy and Speed:** Specialized AI tools achieve 90-95% accuracy on standard clauses [3]. A case study demonstrated 98% accuracy in contract analysis, reducing review time by more than 60% [4] [5]. Comparative benchmarks (such as ContractEval) are used to evaluate LLM performance on clause-level contractual risk tasks [6].
- **Trade-off:** There is a trade-off between inference speed and accuracy in LLMs for legal tasks, with precision and *recall* being key metrics in contract review [7].

**References:**
[1] ABA Tech Survey Finds Growing Adoption of AI in Legal Practice [8]
[2] AI Adoption in Legal Contract Review Grows 75% Year-over-Year [9]
[3] The Best AI Contract Redlining Tools of 2025 [10]
[4] AI Contract Analysis Reaches Critical Accuracy Milestone [11]
[5] AI Adoption Case Study: Luminance's legal team reduced time spent on contract review by over 60% [12]
[6] ContractEval: Benchmarking LLMs for Clause-Level Legal Risk Identification [6]
[7] Benchmark Of OpenAI, Anthropic, And Google LLMs And … [7]
[8] https://www.lawnext.com/2025/03/aba-tech-survey-finds-growing-adoption-of-ai-in-legal-practice-with-efficiency-gains-as-primary-driver.html
[9] https://www.legaltech-talk.com/ai-adoption-in-legal-contract-review-grows-75-year-over-year-marking-early-industry-transformation/
[10] https://www.dioptra.ai/resources/best-ai-contract-redlining-tools-2025-speed-precision
[11] https://www.concord.app/blog/ai-contract-analysis-reaches-critical-accuracy-milestone
[12] https://www.techuk.org/resource/ai-adoption-case-study-learn-how-luminance-s-legal-team-reduced-time-spent-on-contract-review-with-ai.html
[6] https://arxiv.org/html/2508.03080v1
[7] https://www.spotdraft.com/blog/benchmark-of-llms-oct-2024

## Features

- **Role Assignment:** Defines the LLM's context and persona (e.g., "experienced lawyer") to ensure the correct perspective and tone.
- **Structured Instruction:** Uses multi-step prompts (e.g., identify, analyze, suggest) to guide the LLM through complex analyses.
- **Focus on Risk and Compliance:** Specific prompts to identify harmful clauses, ambiguities, financial/legal risks, and ensure adherence to applicable laws and regulations.
- **Negotiation Strategy Generation:** Ability to generate strategic notes and suggestions for contract negotiations.
- **Comparison with Standards:** Prompts that request the comparison of contract clauses with industry standards or "best practices".

## Use Cases

- **Contractual Risk Analysis:** Identification of ambiguities, unfavorable or high-risk clauses (e.g., indemnification, limitation of liability).
- **Compliance Verification:** Ensuring that the contract is in full compliance with the laws and regulations applicable to the jurisdiction.
- **Data Extraction:** Rapid summarization and extraction of key terms (e.g., effective dates, values, parties).
- **Due Diligence:** Accelerating the review of large volumes of contracts in mergers and acquisitions (M&A).
- **Negotiation Preparation:** Generation of *talking points* and counterproposal strategies based on the contract analysis.
- **Comparison with Standards:** Assessment of deviations from industry standards or internal templates.

## Integration

**Prompt Examples and Best Practices:**

1.  **Prompt for Highlighting Key Terms:**
    *Prompt:* "Act as an experienced contract lawyer. Your task is to meticulously review the provided contract and highlight its key terms and clauses, such as scope of work, payment terms, confidentiality obligations, termination conditions, and liability clauses. Provide a written summary detailing these points and their implications, along with suggested modifications that benefit the client."

2.  **Prompt for Identifying Harmful Clauses:**
    *Prompt:* "As a lawyer specializing in contract law, examine the contract to identify any harmful clauses that may disadvantage the client. Focus on hidden fees, automatic renewals, limitations of liability, and terms that restrict the client's rights. Present a detailed analysis explaining why they are harmful and suggest amendments or exclusions to protect the client."

3.  **Best Practices:**
    * **Specificity and Clarity:** Be clear and specific about the objective of the review (e.g., "only indemnification clauses").
    * **Relevant Context:** Include details about the type of contract, the parties, and the applicable legal jurisdiction.
    * **Iterative Review:** Use multiple prompts in sequence to deepen the analysis (e.g., first identify the risk, then request a mitigation suggestion).
    * **Request a Structured Format:** Ask for the output in table or list format to facilitate human review.

## URL

https://promptadvance.club/blog/chatgpt-prompts-for-contract-review