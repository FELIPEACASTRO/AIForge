# Peer Review Prompts

## Description
The **Peer Review Prompts** technique consists of instructing a Large Language Model (LLM) to take on the role of a specialized reviewer or critic to analyze an artifact (text, code, design, etc.) and provide structured, constructive, and actionable feedback. Instead of simply asking for a "review", the prompt explicitly defines the **reviewer's persona**, the **evaluation criteria** (checklist), the desired **output format** (e.g., strengths, weaknesses, improvement suggestions), and the **focus of the analysis** (e.g., cohesion, security, usability). This transforms the LLM from a content generator into a critical evaluation assistant, allowing the user to develop their own rhetorical judgment by comparing the AI's feedback with that of human reviewers. It is a fundamental technique for improving the quality of texts, code, and designs, using the AI as an experienced and methodical extra pair of eyes.

## Examples
```
**Example 1: Academic Essay Review**
```
**Persona:** You are a Professor of Rhetoric and Composition with 20 years of experience.
**Task:** Review the student essay below and provide constructive feedback.
**Checklist:**
1.  **Thesis:** Is the thesis clear, specific, and defensible?
2.  **Evidence:** Is each main point supported by solid evidence that is correctly cited?
3.  **Coherence:** Are the transitions between paragraphs smooth and is the logical structure clear?
**Output Format:**
1.  **Evaluation Summary:** (Overall grade and main point for improvement)
2.  **Specific Comments:** (3-5 numbered observations, citing the line or section of the text)
3.  **Revision Suggestion:** (One actionable step for the next stage)

[PASTE THE ESSAY HERE]
```

**Example 2: Code Review (Python)**
```
**Persona:** You are a Senior Software Engineer specialized in Python and security.
**Task:** Analyze the Python code snippet for security, performance, and PEP 8 compliance.
**Checklist:**
1.  **Security:** Are there vulnerabilities such as SQL injection, exposure of secrets, or insecure use of libraries?
2.  **Performance:** Are there inefficient loops, unnecessary queries, or opportunities for O(n) complexity optimization?
3.  **Style (PEP 8):** Does the code follow PEP 8 naming and formatting conventions?
**Output Format:**
1.  **Critical Issues (Blockers):** (Security errors or bugs)
2.  **Refactoring Suggestions:** (Performance and clarity improvements)
3.  **Style Comments:** (Formatting adjustments based on PEP 8)

[PASTE THE PYTHON CODE HERE]
```

**Example 3: UX/UI Design Critique**
```
**Persona:** You are a UX/UI Specialist focused on accessibility and mobile usability.
**Task:** Review the wireframe (described below) for the checkout screen of an e-commerce application.
**Checklist:**
1.  **Usability:** Is the checkout flow intuitive and does it minimize cognitive load?
2.  **Accessibility (WCAG):** Is the color contrast adequate and are the touch elements large enough?
3.  **Consistency:** Does the design follow known interface patterns that are consistent with the rest of the application?
**Output Format:**
1.  **Usability Issues (High Priority):** (Describe the problem and the impact on the user)
2.  **Accessibility Improvements:** (Specific suggestions based on WCAG)
3.  **Praise:** (What works well in the design)

[WIREFRAME DESCRIPTION HERE]
```

**Example 4: Technical Documentation Review**
```
**Persona:** You are a Technical Editor focused on clarity and accuracy for an audience of junior developers.
**Task:** Review the following API documentation excerpt.
**Checklist:**
1.  **Technical Accuracy:** Are the code examples and parameter descriptions 100% correct and up to date?
2.  **Clarity and Simplicity:** Is the language direct and does it avoid unnecessary jargon? Will the junior audience understand it?
3.  **Structure:** Are headings, lists, and code blocks formatted correctly to make reading easier?
**Output Format:**
1.  **Strengths:** (Where the documentation excels)
2.  **Required Revisions:** (Rewrite suggestions for clarity, with the original sentence and the suggestion)
3.  **Fact Check:** (Any technical claims that need double-checking)

[PASTE THE DOCUMENTATION EXCERPT HERE]
```

**Example 5: Prompt Review (Metaprompting)**
```
**Persona:** You are a Senior Prompt Engineer.
**Task:** Analyze the prompt below for clarity, specificity, and potential risk of "prompt injection" or ambiguity.
**Checklist:**
1.  **Clarity of Intent:** Is the prompt's objective unambiguous?
2.  **Format Specificity:** Is the output format clearly defined and restrictive?
3.  **Injection Risk:** Is there any part of the prompt that could be exploited by a malicious user to divert the AI?
**Output Format:**
1.  **Diagnosis:** (Overall assessment: Good, Needs Adjustments, Poor)
2.  **Suggested Improvements:** (How to make the prompt more robust and specific)
3.  **Risk Alert:** (If there is an injection risk, describe how to mitigate it)

[PASTE THE PROMPT TO BE REVIEWED HERE]
```
```

## Best Practices
1. **Define the Persona and Role:** Start the prompt by instructing the LLM to take on a specific role (e.g., "You are a Senior Software Engineer", "You are a Blind Reviewer for a High-Impact Academic Journal", "You are a UX/UI Designer focused on accessibility"). 2. **Provide Clear Criteria (Checklist):** Include a numbered or bulleted list of the exact points the LLM should check. This ensures the review is focused and comprehensive. 3. **Output Structure:** Specify the exact format of the response (e.g., "Use the following headings: [1. General Summary], [2. Strengths], [3. Weaknesses/Areas for Improvement], [4. Actionable Suggestions]"). 4. **Analysis in Parts:** For long documents (articles, large blocks of code), ask the LLM to review section by section or paragraph by paragraph to maintain accuracy and avoid losing context. 5. **Encourage Justification:** Ask the LLM to justify each critique or suggestion, citing the part of the text or code that led to the observation.

## Use Cases
1. **Academic and Writing Review:** Evaluating essays, theses, and research papers for thesis clarity, logical structure, use of evidence, tone, and citation style. 2. **Code Review:** Analyzing code snippets to identify bugs, security vulnerabilities, adherence to coding standards (PEP 8, etc.), complexity, and refactoring opportunities. 3. **Design Critique:** Evaluating wireframes, mockups, or UX/UI design prototypes for usability, accessibility, visual consistency, and alignment with user goals. 4. **Technical Documentation Review:** Checking manuals, FAQs, or API documentation for technical accuracy, clarity, and suitability for the target audience. 5. **Prompt Development (Metaprompting):** Using the LLM to review and refine other prompts, evaluating their clarity, specificity, and potential for prompt injection.

## Pitfalls
1. **Superficial Review:** Without a detailed checklist, the LLM may provide generic and useless feedback. 2. **Excessive Focus on Form:** The LLM may focus too much on grammatical or stylistic aspects, ignoring deep conceptual or logical flaws. 3. **AI Biases:** The LLM may introduce biases (e.g., favoring Standard White English, or popular coding styles) or weaken human critical analysis. 4. **Hidden Prompt Injection:** In contexts involving the review of third-party documents (such as in academic journals), there may be hidden adversarial text in the document to manipulate the LLM's feedback. 5. **Overconfidence:** The user may accept the AI's feedback without applying their own critical judgment, missing the opportunity for learning and in-depth revision.

## URL
[https://wac.colostate.edu/repository/collections/textgened/rhetorical-engagements/using-llms-as-peer-reviewers-for-revising-essays/](https://wac.colostate.edu/repository/collections/textgened/rhetorical-engagements/using-llms-as-peer-reviewers-for-revising-essays/)
