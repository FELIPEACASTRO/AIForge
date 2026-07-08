# Code Review Prompts

## Description
**Code Review Prompts** are structured and detailed instructions provided to a Large Language Model (LLM) so that it performs a critical and constructive analysis of a code snippet. The goal is to automate or assist the process of reviewing Pull Requests (PRs) or commits, identifying quality problems, bugs, security vulnerabilities, performance inefficiencies, and style inconsistencies [1] [2]. This technique transforms the LLM from a simple text generator into a software engineering assistant, capable of applying specific business rules and coding standards, as long as the prompt provides the necessary context and constraints [2]. The effectiveness lies in the ability to specify the LLM's **role** (e.g., "Senior Security Engineer"), the **focus** of the review (e.g., "performance only"), and the desired output **format** [1].

## Examples
```
**1. Security Review (OWASP Top 10):**

```
You are a Senior Security Engineer. Analyze the following code (language: Python, framework: Django) strictly from the perspective of the OWASP Top 10 vulnerabilities. For each vulnerability found, provide: the exact line, an explanation of the risk, and the suggested fix code. If there are no vulnerabilities, respond only with 'No critical security vulnerabilities found.'

[CODE HERE]
```

**2. Performance and Algorithm Review:**

```
Act as a performance optimization specialist. Review the function [FUNCTION NAME] in [LANGUAGE] to identify performance bottlenecks and algorithmic inefficiencies (O(n) complexity). Suggest refactorings that improve efficiency, justifying the new complexity. The focus is to reduce CPU and memory usage.

[CODE HERE]
```

**3. Style and Code Standards Review (Clean Code):**

```
You are the code reviewer responsible for maintaining the consistency of our codebase. Evaluate the code against Clean Code principles and the CamelCase naming standard. Check: 1. Are variable and function names clear? 2. Does the function do only one thing (Single Responsibility Principle)? 3. Are there unnecessary comments or dead code? Return the suggestions in a numbered list format.

[CODE HERE]
```

**4. Unit Test Review (Test Coverage):**

```
The following code is a new module in [LANGUAGE]. Review the provided unit tests. Identify any edge cases that were not covered and write the additional unit tests needed to reach 100% line coverage. Use the [TEST FRAMEWORK NAME] framework.

[CODE HERE]
```

**5. Architecture and Maintainability Review:**

```
Analyze the code as a Software Architect. The goal of this review is to ensure maintainability and modularity. Does the code adhere to the [PATTERN NAME, e.g., MVC] design pattern? Is there high coupling or low cohesion? Suggest refactorings to decouple components, if necessary.

[CODE HERE]
```
```

## Best Practices
**Be Specific and Contextual:** The prompt should include as much context as possible, such as the goal of the change, the framework/language, and the team's style rules. Use the **HITL (Human-in-the-Loop)** architecture, where the AI generates a review draft and the human reviewer approves, edits, or rejects the suggestions, providing a continuous feedback loop [2]. **Force the Output Format:** Ask the LLM to return the review in a structured format (such as JSON or Markdown with clear sections) to facilitate analysis and integration into CI/CD pipelines [2]. **Scope and Risk Constraint:** For large code bases, include only the diff and the immediate context (file header, imports). Use risk tokens (`HIGH_RISK`) for sensitive files (e.g., cryptography, SQL) so that the LLM applies stricter heuristics [2]. **Compliance Verification:** Incorporate mandatory checklists (e.g., "No hard-coded secrets", "Use of `try-catch-finally`") directly into the prompt to ensure regulatory and security compliance [2].

## Use Cases
**Review Cycle Acceleration:** Reduction of waiting time for reviews, allowing the LLM to triage and pre-approve low-risk changes (e.g., documentation) and flag only high-risk PRs for human review [2]. **Consistency Assurance:** Uniform application of style rules, naming standards, and security heuristics across the entire code base, overcoming inconsistency between human reviewers [2]. **Instant Mentorship and Onboarding:** Providing instant educational feedback for junior developers, explaining the "why" behind refactoring suggestions, accelerating knowledge transfer [2]. **Vulnerability Detection:** Combining static analysis with LLM reasoning to identify security vulnerabilities (SQLi, XSS) and compliance violations (e.g., GDPR) that traditional linters may miss [2]. **Codifying Tribal Knowledge:** Incorporating internal standards and the team's tacit knowledge (e.g., "always use `await` in `fetch` in this repository") into the LLM's prompt, preserving expertise even as senior engineers leave [2].

## Pitfalls
**False Positives/Negatives:** The LLM may flag harmless code or, worse, ignore a critical bug (hallucination), leading to **alert fatigue** and distrust from the human reviewer [2]. **Context Blindness:** The model may suggest changes that break domain contracts or business rules because the prompt contained only the diff, and not the complete call graph or runtime configuration [2]. **Security and Privacy Leakage:** Sending sensitive code (e.g., API keys) to third-party hosted LLMs may violate compliance (GDPR, HIPAA) if there is no adequate Data Processing Agreement (DPA) [2]. **Skill Erosion:** Excessive dependence on AI can lead to the **erosion of the skills** of developers in reviewing and debugging, especially juniors, who stop learning to read compiler errors and to think critically about code [2]. **Bias and Obsolete Style:** The model may impose a coding style inherited from its training or from an outdated prompt, causing friction with the team's evolving conventions [2].

## URL
[https://dev.to/dixitgurv/ai-assisted-code-review-opportunities-and-pitfalls-llp](https://dev.to/dixitgurv/ai-assisted-code-review-opportunities-and-pitfalls-llp)
