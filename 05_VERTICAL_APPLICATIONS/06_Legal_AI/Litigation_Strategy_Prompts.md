# Litigation Strategy Prompts

## Description
Prompt Engineering for Litigation Strategy (Litigation Strategy Prompts) is the application of prompt engineering in the legal domain, focused on optimizing interaction with Large Language Models (LLMs) to assist with litigation tasks. The primary goal is to turn the AI into a strategic support tool, capable of performing case analysis, legal research, document review (eDiscovery), and drafting procedural documents. The technique emphasizes **assigning a specific role** to the AI (e.g., senior IP attorney), requiring **structured reasoning** (such as the IRAC format: Issue, Rule, Application, Conclusion) to mitigate hallucinations, and the need to provide **broad context** without leading the answer. The focus is on obtaining verifiable results and maintaining attorney-client privilege and data confidentiality.

## Examples
```
**Example 1: Risk and Strategy Analysis (IRAC)**
```
**Role:** You are a senior litigation attorney specializing in intellectual property law.
**Context:** [INSERT CASE FACTS, DOCUMENTS, RELEVANT CASE LAW].
**Task:** Analyze the case of [CASE NAME] and determine the likelihood of success on a preliminary injunction motion.
**Instructions:**
1. Present your analysis in the IRAC format (Issue, Rule, Application, Conclusion).
2. Identify the three strongest arguments for the defense and the three strongest for the plaintiff.
3. Conclude with a strategic recommendation (e.g., settlement, litigation, specific motion).
4. Cite the relevant sections of IP law and precedents.
```

**Example 2: Summary of a Complex Regulation**
```
**Role:** You are the general counsel of a high-growth technology company.
**Task:** Provide a comprehensive and accurate analysis of the Cybersecurity Regulation [REGULATION NAME, e.g., LGPD, GDPR, NY DFS 500].
**Instructions:**
1. Summarize the main provisions and obligations.
2. Identify the compliance deadlines.
3. Explain the scope and applicability.
4. Discuss the implications of non-compliance.
5. Maintain a professional and objective tone.
6. Cite the specific section of the regulation for each point.
```

**Example 3: Contract Review for AI Training (Risk Mitigation)**
```
**Document:** [ATTACH DPA/VENDOR CONTRACT]
**Task:** Review the attached DPA and extract all provisions related to **AI model training**.
**Output Instructions:**
1. Answer clearly: Does the vendor reserve the right to use customer data for AI training?
2. Are there any limits (e.g., anonymization, deletion, restrictions)?
3. Cite precisely the section number and title (e.g., Section 4.2 – Use of Data).
4. Write in plain, non-legal language, in a single concise paragraph.
```

**Example 4: Prompt Refinement (Metaprompting)**
```
**Context:** [INSERT YOUR PREVIOUS PROMPT]
**Task:** Based on my previous prompt, identify the key areas or concepts I can adjust or refine to improve the quality and accuracy of your next response.
**Instructions:**
1. Highlight words, phrases, or ideas that, if clarified or altered, would significantly impact the direction or depth of your analysis.
2. Suggest "knobs" or "levers" I can use, such as: increasing the level of detail, changing the tone (more technical/formal), or focusing on a different aspect of the topic.
```

**Example 5: Drafting a Counterargument**
```
**Role:** You are a skeptical litigation attorney.
**Context:** [INSERT THE OPPOSING PARTY'S MAIN ARGUMENT].
**Task:** Develop a list of 5 to 7 solid and well-founded legal counterarguments against the thesis presented.
**Instructions:**
1. For each counterargument, provide a brief justification and cite a precedent or legal principle that supports it.
2. Maintain a persuasive and aggressive tone.
```

**Example 6: Super-Prompt Structure (Template)**
```
**[TOPIC/DOCUMENT]:** [Description of the document or topic and the type of request (e.g., legal analysis, contract review)].
**[RESEARCH AND REASONING REQUIREMENTS]:**
1. Include relevant laws, regulations, and precedents, with paragraph/section citations.
2. Analyze each section separately and flag problematic language with specific recommendations.
3. Provide a critical analysis of strengths, weaknesses, risks, and opportunities.
4. Share a list of assumptions and limitations in your analysis, as well as counterarguments and business implications.
**[RESPONSE FORMAT INSTRUCTIONS]:**
1. Begin with an executive summary of 3 to 5 bullet points.
2. Use clear headings and subheadings.
3. Bold the main findings and recommendations.
4. Conclude with a "next steps" section with actionable recommendations.
```
```

## Best Practices
1. **Assign a Role to the AI (Role-Playing):** Ask the AI to act as an expert in the relevant practice area (e.g., "You are a criminal defense attorney with 20 years of experience").
2. **Require Structured Reasoning (IRAC):** Request that the AI structure its analysis in the IRAC format (Issue, Rule, Application, Conclusion) to make it easier to verify the reasoning.
3. **Provide Broad Context, but Do Not Lead the Answer:** Give as much context and background documents as possible, but frame open-ended questions to avoid biases (do not "lead the witness").
4. **Use the AI to Refine the Prompt Itself (Metaprompting):** Ask the AI to identify key areas or concepts in your prompt that, if adjusted, would improve the quality of the response.
5. **Ask for Verifiable Questions:** Request outputs that can be easily verified, such as precise citations to sections of laws or documents, even though the AI's citations need to be checked.
6. **Trust but Verify:** Treat the AI's output as the work of a junior attorney and always perform the final human review.

## Use Cases
1. **Litigation Risk Analysis:** Assess the strengths and weaknesses of a case, identifying risks and opportunities.
2. **Document Review (eDiscovery):** Extract specific clauses, summarize DPAs (Data Processing Agreements), or identify sensitive information in large volumes of text.
3. **Legal Research:** Summarize complex regulations (e.g., NY DFS 500), find relevant precedents, and cite specific sections.
4. **Document Drafting:** Generate drafts of memos, pleadings, or sections of briefs, maintaining a professional and objective tone.
5. **Strategy Development:** Create detailed project plans for litigation and develop counterarguments.

## Pitfalls
1. **Confidentiality and Privacy:** Using public tools (such as ChatGPT or Claude) with confidential data. **Mitigation:** Use enterprise accounts, tools with encryption, and redaction of sensitive information (e.g., renaming companies to "Company 1").
2. **Hallucinations (Inaccuracy):** The AI may provide incorrect information or false citations. **Mitigation:** Always require human review ("Trust but verify") and request citations for verification.
3. **Biases and Leading:** Asking closed or biased questions that can skew the AI's analysis. **Mitigation:** Use open-ended questions and provide neutral context.
4. **Loss of Attorney-Client Privilege:** Improper use of AI tools can compromise privilege. **Mitigation:** Prioritize tools that guarantee the preservation of privilege.
5. **Overreliance:** Blindly trusting the AI's output without proper professional judgment. **Mitigation:** Treat the output as a draft or suggestion, not as a final product.

## URL
[https://www.lsuite.co/blog/mastering-ai-legal-prompts](https://www.lsuite.co/blog/mastering-ai-legal-prompts)
