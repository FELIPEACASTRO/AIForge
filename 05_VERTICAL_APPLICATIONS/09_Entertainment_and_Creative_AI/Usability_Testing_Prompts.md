# Usability Testing Prompts

## Description
**Usability Testing Prompts** are prompt engineering instructions designed to leverage Large Language Models (LLMs), such as ChatGPT or Gemini, to automate, accelerate, and enhance various stages of the User Experience (UX) Research and Usability Testing process [1]. Rather than replacing the human researcher, these prompts act as a **copilot** or **assistant**, helping to quickly generate research artifacts such as interview scripts, survey questions, screener surveys, and even to analyze qualitative and quantitative data [1].

The effectiveness of these prompts lies in their ability to transform time-consuming, repetitive tasks—such as writing unbiased questions or categorizing large volumes of open-ended feedback—into structured, actionable outputs. Using prompting frameworks such as **REFINE**, **CARE**, and **RACEF** is essential to provide the context, rules, and output format needed for the AI to produce high-quality results relevant to the UX domain [1].

In essence, the technique allows UX researchers to focus on interpretation and strategy, while the AI handles the initial generation and processing of data, significantly reducing the time-to-insight [1].

## Examples
```
**1. Semi-Structured Interview Script Generation (REFINE)**
```
**Role:** You are a senior UX researcher.
**Task:** Create a 45-minute semi-structured interview guide for power users who canceled their subscription to our project management SaaS product in the last 3 months.
**Format:** The guide should be divided into 5 timed sections (Introduction, Context, Tasks, Reflection, Closing).
**Rules:** Include 7 core questions, with 2 follow-up questions for each. The core questions should focus on **feature gaps** and **emotional triggers** that led to cancellation (churn).
```

**2. Bias Check in Survey Questions (CARE)**
```
**Context:** I am a UX researcher preparing a survey to evaluate satisfaction with the new 'Dark Mode' feature.
**Ask:** Check the following survey questions for any bias, leading questions, or irrelevance.
**Rules:** If you find any issues, provide a new list of objective, neutral questions.
**Example Input:** "How easy was it for you to use the new and intuitive Dark Mode?" and "Do you agree that the design improved with Dark Mode?"
```

**3. Screener Survey Creation**
```
**Task:** Create a 6-question screener survey to recruit participants for a usability test.
**Target Audience:** Product Managers with at least 3 years of experience in the FinTech sector who actively use the 'Mixpanel' analytics tool at least 3 times per week.
**Format:** Provide the question, the response format (multiple choice, open-ended), and the qualification logic for each question.
```

**4. Thematic Analysis of Open-Ended Comments (RACEF)**
```
**Focus:** Analyze the 300 open-ended NPS (Net Promoter Score) comments I am providing below.
**Task:** Group the comments into main themes.
**Format:** Return a table with the following columns: 'Theme', 'Percentage of Comments', and 'Two Representative Quotes'.
**Rules:** Label each theme concisely and list the themes in descending order of frequency.
**[Insert 300 comments here]**
```

**5. Synthesis of Interview Insights**
```
**Context:** Below are the transcripts of 5 semi-structured interviews about our app's 'Document Upload' functionality.
**Task:** Synthesize the insights.
**Rules:** Code each quote by **theme** and **sentiment** (positive, negative, neutral). Deliver a ranked list of the top 5 usability problems found, including illustrative quotes for each problem and a 100-word summary of the design opportunity.
**[Insert Transcripts here]**
```

**6. Generation of Testable Hypotheses**
```
**Context:** The attached document (or text) highlights the main friction points in our checkout funnel.
**Task:** Generate 5 testable design hypotheses.
**Format:** Each hypothesis should follow the format: 'We believe that [CHANGE] will improve [METRIC] for [USER]'.
```

**7. Pulse Survey Design**
```
**Task:** Design an 8-question pulse survey to measure user satisfaction with the newly launched 'Voice Search' feature.
**Rules:** Mix 6 five-point Likert scale items covering [Accuracy, Speed, Ease of Use] with 1 NPS question. Keep the completion time under 2 minutes.
```
```

## Best Practices
**1. Adopt a Prompting Framework (REFINE, CARE, RACEF):** Structure your prompts using frameworks such as REFINE (Role, Expectation, Format, Iterate, Nuance, Example), CARE (Context, Ask, Rules, Examples), or RACEF (Rephrase, Append, Clarify, Examples, Focus) to ensure the LLM has the desired context, rules, and output format [1].
**2. Define the Role and Objective:** Start the prompt by clearly defining the AI's role (e.g., "You are a senior UX researcher") and the specific objective of the task (e.g., "Generate a discussion guide for usability testing") [1].
**3. Be Specific and Contextual:** Provide as much detail and context as possible. Include the product, the target audience, the duration, the output format (table, list, summary), and any constraints [1].
**4. Ask for Non-Bias:** Include an explicit instruction for the AI to check for and remove any **cognitive bias** or **leading questions** in the generated results, especially in interview or survey questions [1].
**5. Iterate and Refine:** The AI's first result is rarely the final one. Use follow-up prompts to iterate, refine, add nuance, or remove unnecessary sections, following the "Iterate" principle of the REFINE framework [1].
**6. Use AI for Data Analysis:** Use prompts for qualitative and quantitative data analysis tasks, such as grouping open-ended comments by theme, identifying drop-off points in funnels, or synthesizing interview transcripts [1].
**7. Ask for Citations and References:** Whenever possible, ask the AI to cite sources or provide references for its suggestions, and **always verify** those sources to ensure accuracy and avoid hallucinations [1].

## Use Cases
**1. Research Artifact Generation:**
*   **Interview Scripts:** Rapid creation of semi-structured discussion guides, including core questions, follow-ups, and time structure [1].
*   **Surveys:** Designing surveys with different question types (Likert, open-ended, closed-ended), ensuring neutrality and focus on specific hypotheses [1].
*   **Screeners:** Crafting questions to recruit very specific participant profiles (e.g., Product Managers with X years of experience in Y sector) [1].
*   **Moderation Checklists:** Generating comprehensive checklists for interview moderators, covering technical aspects, *rapport* building, and note-taking [1].

**2. Data Analysis and Synthesis:**
*   **Qualitative Thematic Analysis:** Grouping and labeling large volumes of open-ended data (e.g., NPS comments, user feedback) into actionable themes, with frequency calculation and representative quotes [1].
*   **Friction Point Identification:** Analyzing event data (logs, CSVs) to identify the main drop-off points in conversion funnels (e.g., *sign-up* or *checkout* funnels) [1].
*   **Interview Synthesis:** Coding transcripts by theme and sentiment, resulting in ranked lists of usability problems and design opportunities [1].
*   **Basic Quantitative Analysis:** Processing closed-ended survey data (Excel, CSV) to calculate averages, percentages, and statistically significant differences between user groups [1].

**3. UX Strategy and Planning:**
*   **Persona Generation:** Creating concise personas based on provided demographic and behavioral data [1].
*   **Executive Summaries:** Transforming long research reports into one-page summaries for high-level (C-level) stakeholders, focusing on *insights* and recommended actions [1].
*   **Hypothesis Generation:** Creating testable design hypotheses from identified friction points, following structured formats (e.g., "We believe that...") [1].

## Pitfalls
**1. Bias and Leading Questions:** The AI can generate biased questions if not explicitly instructed to be neutral. The AI's training data can introduce cognitive biases, resulting in distorted research data [1].
**2. Lack of Nuance and Diversity:** The AI tends to converge toward the statistical "norm," which can filter out or disregard unique and diverse participant perspectives, especially in thematic analyses. Excessive iteration with the AI can "strip away" the singularity of qualitative data [1].
**3. Hallucinations and Inaccuracy:** The AI can "hallucinate" (invent) facts, sources, or data. It is crucial that the human researcher review and validate all outputs, especially references and data synthesis [1].
**4. Overreliance:** Blindly trusting the AI to generate research scripts or analyze data without human oversight can lead to superficial or incorrect results. The AI is a powerful processing tool, but it requires expert supervision [1].
**5. Privacy and Compliance Concerns:** Processing user data (transcripts, comments) through AI tools raises privacy issues (GDPR, LGPD). It is essential to ensure user consent and proper anonymization of data before feeding it to the AI [1].
**6. Insufficient Context:** Vague or generic prompts will result in equally vague outputs. Failing to provide context (audience, product, objective) and rules (format, constraints) is a common mistake that nullifies the benefit of the tool [1].

## URL
[https://maze.co/collections/ai/user-research-prompts/](https://maze.co/collections/ai/user-research-prompts/)
