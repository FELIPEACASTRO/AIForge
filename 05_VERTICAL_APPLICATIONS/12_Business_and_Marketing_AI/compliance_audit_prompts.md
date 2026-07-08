# Compliance Audit Prompts

## Description
The practice of prompt engineering focused on creating detailed and contextual instructions for language models (LLMs) with the aim of assisting in Internal Audit, Risk, and Compliance (GRC) tasks. This includes analyzing regulatory documents, identifying control gaps, summarizing risk reports, and generating audit trails. The goal is to increase the efficiency, accuracy, and comprehensiveness of audit processes, transforming unstructured data into actionable *insights* aligned with *frameworks* such as COSO, NIST, and ISO 27001. These prompts act as a bridge between the abundance of data and the need for strategic and documentable *insights* for audit committees and regulators.

## Examples
```
**1. Control Gap Analysis**
```
Act as an information security auditor. Analyze the "2024 Penetration Test Report" and the "ISO 27001 Encryption Standard." Identify all control gaps related to data encryption in transit and at rest. List the gaps in a table with three columns: "Control Gap," "Associated Risk," and "ISO 27001 Reference."
```

**2. Risk Summary for Executives**
```
Based on the "2025 Enterprise Risk Matrix" and the "Last Quarter's Incident Reports," generate a one-page risk summary memo for executive leadership. The memo should highlight the top 3 operational risks by probability and impact, quantify the potential impact in financial terms (if possible), and suggest two short-term mitigation strategies for each risk.
```

**3. Audit Trail Creation**
```
You are a GRC documentation specialist. List the steps taken in this compliance review (including referenced source documents, applied filters, and interaction dates) in a format suitable for audit evidence documentation. The result should be a structured log file with timestamps and links to the source documents.
```

**4. Regulatory Compliance Assessment (GDPR/LGPD)**
```
Analyze the "Customer Privacy Policy" and the "User Data Processing Flow." Create a checklist to assess compliance with the "Data Subject Rights" requirements (e.g., right to be forgotten, portability) of the GDPR and the LGPD. For each item, provide a score from 1 to 5 (1=Non-Compliant, 5=Fully Compliant) and a brief justification.
```

**5. Transaction Anomaly Identification**
```
Act as a fraud analyst. Examine the "Last Month's Financial Transactions" dataset (attached). Identify and list all transactions that exceed 3 standard deviations from the average value or that involve a non-approved vendor. Format the output as an exception report, including Transaction ID, Amount, Vendor, and Flagging Justification.
```

**6. Third-Party Contract Review**
```
Analyze the "Service Contract with Vendor X." Extract and list all clauses related to data security requirements, right to audit, and liability for data breach. Compare these clauses with our "Minimum Security Standard for Third Parties" and point out any discrepancies or high-risk areas.
```

## Best Practices
**I-I-O Structure (Instruction, Information, Objective):** Use a clear structure: 1. **Instruction** (What to do - e.g., "Analyze"), 2. **Information** (Where to apply - e.g., "Document X"), 3. **Objective** (The expected result - e.g., "In table format, highlighting high risks"). **Role Definition (Role-Playing):** Begin the prompt with "Act as a senior internal auditor specialized in [compliance area]." This improves the quality and tone of the response. **Contextualization and Constraints:** Provide as much context as possible (e.g., "Considering the LGPD and the GDPR") and define constraints on format, length, and *frameworks* (e.g., "Use the COSO framework for the assessment"). **Human Validation (Human-in-the-Loop):** Never blindly trust the LLM's output. The output should be a draft or preliminary analysis that requires final review and validation by a human auditor. **Clarity on Input Data:** Specify which documents or data the LLM should analyze (e.g., "Attach the SOC 2 report and the internal risk matrix").

## Use Cases
**Internal Audit:** Accelerate control testing, identify exceptions in large volumes of data, and automate audit trail creation. **Risk Management:** Draft risk summaries aligned with *frameworks* (COSO, NIST), analyze risk registers, and quantify the potential impact of vulnerabilities. **Regulatory Compliance:** Perform gap analyses between internal policies and external regulations (e.g., GDPR, LGPD, SOX). **Document Review:** Summarize long reports (e.g., SOC 2, third-party reports), extract specific contractual clauses, and compare regulatory documents. **AI Governance:** Audit the use of other AI tools within the organization to ensure they comply with ethical and legal principles (e.g., transparency, fairness, explainability).

## Pitfalls
**Over-Reliance:** Treating the LLM's output as absolute truth, ignoring the need for human judgment and validation. **Hallucinations and Inaccuracy:** The LLM may "hallucinate" regulatory references or create non-existent control gaps. Cross-verification is essential. **Sensitive Data Leakage:** Entering confidential data or PII (Personally Identifiable Information) into the prompt. It is crucial to anonymize or use simulated/masked data. **Vague Prompts:** Using prompts like "Help me with the audit." The lack of context, role, and output format leads to useless results. **Algorithmic Bias:** If the LLM is used to assess bias (e.g., in recruitment), it may perpetuate or amplify existing bias if not instructed to use specific fairness metrics.

## URL
[https://empoweredsystems.com/blog/prompt-engineering-for-internal-auditors-a-new-skill-for-a-new-era/](https://empoweredsystems.com/blog/prompt-engineering-for-internal-auditors-a-new-skill-for-a-new-era/)
