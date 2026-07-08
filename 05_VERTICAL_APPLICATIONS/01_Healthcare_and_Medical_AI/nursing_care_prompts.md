# Nursing Care Prompts

## Description
Nursing Care Prompts refer to the strategic application of Prompt Engineering to guide Large Language Models (LLMs) in generating structured, accurate, and contextually relevant outputs for nursing practice and education. This technique is crucial for turning AI into a clinical and educational support tool, allowing nurses and students to quickly create detailed care plans (following formats such as NANDA), pharmacology sheets, clinical communication summaries (SBAR), and personalized study materials. The main focus is to provide the AI with a role, a specific clinical scenario, and strict formatting constraints to ensure that the output is clinically acceptable and academically rigorous. The effective use of these prompts aims to reduce the time spent on administrative and documentation tasks, allowing professionals to focus more on direct patient care.

## Examples
```
**1. Care Plan Prompt (NANDA):**
> "You are a BSN-level nursing instructor. Help me draft an RN-level care plan for an adult client with **[primary problem]** related to **[etiology]** as evidenced by **[signs/symptoms]**. Constraints: 1) Follow the NANDA format, 2) Include two SMART goals (with a timeframe), 3) Provide four nursing interventions with **rationales** and **sources of evidence**, 4) Add evaluation criteria that I can check in 24-48h. Output in a two-column table (Plan / Documentation Notes)."

**2. Pharmacology Sheet Prompt:**
> "Create a concise pharmacology sheet for **[drug/class]** covering: mechanism of action, indications, high-yield adverse effects, contraindications, interactions, nursing considerations, and patient teaching. Include two NCLEX-style questions with rationales. Format as headings and bullet points."

**3. SBAR Communication Prompt:**
> "Convert this situation into a concise SBAR for a handoff to the on-call physician. Include a one-line request/recommendation at the end. Situation: **[Briefly describe the patient scenario, e.g.: 68-year-old patient, post-cholecystectomy, with increasing abdominal pain and hypotension]**."

**4. Priority Rationalization Prompt (NGN):**
> "Explain why the correct actions are the priority for this case in three steps: 1) pathophysiology of the problem, 2) how the action changes the physiology, 3) which parameter proves success in 10-30 minutes. Case: **[Describe the clinical case]**."

**5. Therapeutic Communication Prompt:**
> "Provide five therapeutic and two non-therapeutic responses to this client statement: 'I feel like I'm failing as a parent.' Label each response and explain why."

**6. Simulation Scenario Generation Prompt:**
> "Generate a high-fidelity simulation scenario for ADN-level nursing students about **[topic]**. Include: patient history, initial vital signs, critical lab results, and three expected nursing actions (priority)."
```

## Best Practices
**1. Clarity and Context (Golden Rule):** Always define your role (e.g.: "You are a nursing instructor", "You are an ICU nurse"), the expected knowledge level (e.g.: "BSN level", "first term"), the patient scenario, and the desired output format (e.g.: "two-column table", "bulleted list").
**2. Structure and Format:** Explicitly ask for structured formats such as tables, bulleted lists, or checklists. This facilitates review and information extraction.
**3. Specific Constraints:** Include academic or clinical constraints, such as "Follow the NANDA format", "Include two SMART goals", or "Use the SBAR structure".
**4. Human Verification:** Never use AI output directly in patient care or academic work without cross-checking against reliable sources (textbooks, drug guides, hospital protocols).
**5. Action and Parameter:** When asking for interventions, ask the AI to list the **Action** and the **Parameter** that proves the action's success (e.g.: Action: Administer oxygen. Parameter: SpO₂ > 92% in 10 minutes).

## Use Cases
**1. Nursing Education:**
*   Creation of detailed, rubric-ready care plans (CP) for academic assignments.
*   Generation of NCLEX-style (Next Generation NCLEX) practice questions with rationales for study.
*   Development of concise drug sheets and mnemonics for pharmacology.
*   Transformation of lecture notes into structured study guides and flashcards for active recall.

**2. Clinical Practice and Documentation:**
*   Drafting handoff summaries using the SBAR structure (Situation, Background, Assessment, Recommendation).
*   Generation of therapeutic communication suggestions for difficult interactions with patients or family members.
*   Creation of safety protocols and checklists for specific procedures.
*   Assistance with documentation, such as formulating accurate nursing diagnoses and care goals.

**3. Research and Professional Development:**
*   Explanation of complex pathophysiologies in layman's terms for patient education.
*   Comparison and contrast of treatment guidelines or drug classes.
*   Assistance in writing articles, research proposals, or health policy documents.

## Pitfalls
**1. Over-Reliance (Hallucinations):** The greatest risk is blindly trusting AI output. LLMs may "hallucinate" facts, medication doses, or interventions that seem plausible but are clinically incorrect or dangerous. **Always verify.**
**2. Privacy Violation (PHI):** Never enter real protected health information (PHI) into prompts. The prompt should be de-identified and generalized (e.g.: "68-year-old patient with heart failure" instead of "John Smith, 68 years old, bed 302").
**3. Lack of Context:** Vague prompts (e.g.: "What is heart failure?") will result in generic and useless responses. The output should be specific to your role and level of study/practice.
**4. Bias and Cultural Inadequacy:** The AI may perpetuate biases or provide culturally insensitive advice. Nurses should review the output to ensure it is appropriate for the patient and the setting.
**5. Not Specifying the Format:** Not asking for a structured format (table, list) leads to long blocks of text that are difficult to analyze and use in a clinical or academic setting.

## URL
[https://goodnurse.com/article/199/ai-prompt-library-for-nursing-students-2025-care-plans-pharmacology-ngn-study-notes](https://goodnurse.com/article/199/ai-prompt-library-for-nursing-students-2025-care-plans-pharmacology-ngn-study-notes)
