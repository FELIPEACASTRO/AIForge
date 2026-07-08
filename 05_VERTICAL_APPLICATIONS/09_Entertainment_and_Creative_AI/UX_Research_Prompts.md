# UX Research Prompts

## Description
**UX Research Prompts** are structured, detailed instructions provided to Large Language Models (LLMs), such as ChatGPT or Gemini, to automate, accelerate, and enhance various stages of the User Experience (UX) Research process. Instead of using AI only for generic tasks, these prompts are designed to simulate the reasoning of a UX researcher, assisting in generating research plans, interview scripts, qualitative data analysis (such as transcripts), persona creation, hypothesis identification, and synthesis of findings. Adopting specific prompting frameworks (such as REFINE, CARE, and RACEF) is an essential **best practice** to ensure the AI's output is relevant, actionable, and aligned with research objectives. The trend, observed between 2023 and 2025, is for AI to become an indispensable tool for increasing the speed and scale of UX research, allowing researchers to focus on higher-value tasks, such as interpretation and strategy.

## Examples
```
**1. Research Plan Generation (REFINE Framework)**
```
**Role:** Senior UX Researcher.
**Expectation:** Generate a detailed usability research plan.
**Format:** Markdown table with columns: Phase, Objective, Method, Estimated Duration.
**Iterate:** Focus on the Remote Usability Testing phase.
**Nuance:** The product is a mobile finance app for Generation Z.
**Example:** Include metrics such as Task Success Rate and Time on Task.

**Prompt:** "Assume the role of a Senior UX Researcher. Generate a detailed usability research plan for a mobile finance app focused on Generation Z. The plan should be presented in a Markdown table with the columns: Phase, Objective, Method, and Estimated Duration. Give special focus to the Remote Usability Testing phase and include metrics such as Task Success Rate and Time on Task."
```

**2. Qualitative Transcript Analysis**
```
**Role:** Qualitative Data Analyst.
**Context:** [INSERT INTERVIEW TRANSCRIPT HERE].
**Ask:** Identify the top 5 pain points and the top 3 unmet needs of the user.
**Rules:** The output should be a numbered list, citing excerpts from the transcript to support each point.
**Example:** The format should be: "Pain Point 1: [Description] - Quote: '...'"

**Prompt:** "Assume the role of a Qualitative Data Analyst. Analyze the interview transcript provided below. Identify the top 5 pain points and the top 3 unmet needs. The output should be a numbered list, citing exact excerpts from the transcript to support each point. [INSERT INTERVIEW TRANSCRIPT HERE]"
```

**3. Interview Script Creation (CARE Framework)**
```
**Context:** We are developing a new "Collaborative Wishlist" feature for a home decor e-commerce site.
**Ask:** Create a semi-structured interview script for 60 minutes.
**Rules:** The script should have 5 sections (Introduction, Current Usage, Feature Needs, Concept Testing, Closing). The questions should be open-ended and non-leading.
**Examples:** Avoid questions like 'Did you like the feature?'. Prefer 'What would you do with this feature?'

**Prompt:** "Create a 60-minute semi-structured interview script for a study on a new 'Collaborative Wishlist' feature in a home decor e-commerce site. The script should be divided into 5 sections: Introduction, Current Usage, Feature Needs, Concept Testing, and Closing. The questions should be open-ended and non-leading. Avoid 'yes/no' questions."
```

**4. Usability Hypothesis Generation**
```
**Role:** Expert in Nielsen's Heuristics.
**Context:** The heatmap shows that 70% of users abandon the checkout page at the 'Shipping Information' step.
**Ask:** Generate 5 testable usability hypotheses to explain the abandonment.
**Rules:** Each hypothesis should follow the format: 'We believe that [USER ACTION] because [USABILITY PROBLEM], which will result in [SUCCESS METRIC]'.

**Prompt:** "Assume the role of an Expert in Nielsen's Heuristics. The heatmap indicates that 70% of users abandon the checkout page at the 'Shipping Information' step. Generate 5 testable usability hypotheses to explain this abandonment. Each hypothesis should follow the format: 'We believe that [USER ACTION] because [USABILITY PROBLEM], which will result in [SUCCESS METRIC]'."
```

**5. Synthesis of Findings for Stakeholders**
```
**Role:** Research Communicator.
**Context:** [INSERT SUMMARY OF FINDINGS HERE - e.g., 10 interviews, 3 usability tests].
**Ask:** Create a 5-point executive summary for leadership.
**Rules:** The summary should focus on business implications and high-level design recommendations. Use clear and direct language.

**Prompt:** "Based on the research findings provided below, create a 5-point executive summary for leadership. The summary should focus on business implications and high-level design recommendations. Use clear and direct language. [INSERT SUMMARY OF FINDINGS HERE]"
```
```

## Best Practices
**1. Adopt a Prompting Framework:** Use structures such as **REFINE** (Role, Expectation, Format, Iterate, Nuance, Example), **CARE** (Context, Ask, Rules, Examples), or **RACEF** (Rephrase, Append, Clarify, Examples, Focus) to ensure the LLM receives all the information needed for a high-quality response.
**2. Define the Role Clearly:** Start the prompt by instructing the AI to assume the persona of a "Senior UX Researcher," "UX Data Analyst," or "Usability Expert." This aligns the tone and focus of the response.
**3. Provide Context and Input Data:** The AI is "blind" without context. Include interview transcripts, research data, personas, or the specific design problem being addressed.
**4. Specify the Output Format:** Request the result in a structured format (e.g., "Markdown Table," "Numbered List," "5-point Summary") to facilitate analysis and integration into your workflow.
**5. Iterate and Refine:** The first result is rarely the final one. Use follow-up prompts (Iterate/Clarify) to refine, add nuance, remove irrelevant sections, or dive deeper into a specific point.
**6. Use Examples and Rules:** Including examples of what you want (or don't want) and clear rules (e.g., "Questions cannot be leading," "Focus only on usability problems") dramatically improves accuracy.

## Use Cases
nan

## Pitfalls
**1. Blindly Trusting the AI's Output:** The AI can generate plausible but factually incorrect or biased content (so-called "hallucination"). The UX researcher **must** always review, validate, and apply their professional judgment to the AI's output.
**2. Generic or Vague Prompts:** Prompts like "Help me with UX research" result in superficial responses. The lack of context, rules, and a specific format is the most common mistake.
**3. Ignoring Ethics and Privacy:** Using sensitive user data (PII - Personally Identifiable Information) in AI prompts without proper anonymization and consent is an ethical and legal violation.
**4. Failing to Iterate:** Treating the AI as a single-answer oracle. The strength of UX prompting lies in **iteration** and continuous refinement of the response.
**5. Confirmation Bias:** Asking the AI to confirm a pre-existing hypothesis instead of asking for a neutral analysis. This can reinforce biases rather than challenge them.
**6. Not Defining the Role:** Without a clear role (e.g., "Act as a usability critic"), the AI may respond with a tone or focus inappropriate for the research task.

## URL
[https://maze.co/collections/ai/user-research-prompts/](https://maze.co/collections/ai/user-research-prompts/)
