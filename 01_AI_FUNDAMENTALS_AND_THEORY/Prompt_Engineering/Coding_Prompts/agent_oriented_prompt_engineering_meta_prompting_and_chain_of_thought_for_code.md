# Agent-Oriented Prompt Engineering (Meta-Prompting and Chain-of-Thought) for Code

## Description

Advanced prompt engineering techniques that use the LLM to act as a **Senior Refactoring Agent**, following a structured and incremental workflow. **Meta-Prompting** defines the role, the objective, the scope, the guardrails, and the detailed workflow. **Stepwise Chain-of-Thought (CoT)** ensures that the process is auditable and controllable, requiring user confirmation at each step.

## Statistics

- **Migration Success Rate:** A case study (Medium, ELCA IT) showed that LLMs successfully refactored **65–70% of the methods** without manual intervention in a Java Spring migration, resulting in significant time savings.
- **Error Reduction:** The use of Stepwise CoT (Medium, Reynald) is a best practice to **prevent compounding errors** in complex tasks such as refactoring, keeping the developer in control.
- **Benchmarking:** Recent research (arXiv 2025) focuses on **benchmarking LLMs for Code Review**, seeking more robust metrics that consider the complete project context, instead of only isolated units of code.

## Features

- **Holistic Analysis:** Ability to analyze the complete *workspace* (mono/multi-repo) and respect existing configurations (package.json, tsconfig, eslintrc).
- **Structured Refactoring:** Divides the refactoring into phases (Safe/Mechanical, Moderate, High Risk) for risk management.
- **Security Guardrails:** Enforces rules such as "Keep public APIs stable", "Incremental and reviewable changes", and "Never reduce test coverage".
- **Detailed Diagnosis:** Generates a repository inventory and a table of "Findings" with Severity, Problem, Evidence, Impact, and Effort.
- **Controlled Workflow:** Uses Stepwise CoT to ensure that the developer can review and approve each step before execution.

## Use Cases

- **Legacy Code Modernization:** Refactor large code bases to improve structure, modularity, and compliance with modern best practices.
- **Framework/Language Migration:** Assist in converting code between framework or language versions, as seen in the Java Spring case study.
- **Automated Code Review:** Act as a senior reviewer, identifying *code smells*, security flaws, and style inconsistencies before human review.
- **Design Pattern Application:** Ensure the consistent application of design patterns (e.g., Dependency Inversion, Clean Architecture) across the entire repository.

## Integration

### Best Practice 1: Stepwise Chain-of-Thought (CoT) for Refactoring
**Objective:** Maintain control and avoid compounding errors in complex refactorings.
**Example Prompt:**
```
"Help me refactor the code in the file 'service.js'. Proceed one step at a time. Do not move on to the next step until I provide the keyword 'next'."
```
**Expected LLM Response (Example):**
```
"Step 1: Analyze the current structure of the code. I identified the following problems: 1. Unrelated functions in the same file. 2. Code duplication in error handling. 3. Functions with high cyclomatic complexity. Type 'next' for me to proceed with the first refactoring step."
```

### Best Practice 2: Meta-Prompting (Refactoring Agent Template)
**Objective:** Provide comprehensive context and constraints for a repository refactoring.
**Prompt Template (Adapted):**
```
"You are a senior software engineer acting as a repository refactoring agent.

OBJECTIVE: Analyze the entire workspace and propose/refactor changes that make the code base more structured, consistent, and easy to maintain, following recognized best practices and the project's existing conventions. Keep the public behavior stable.

SCOPE:
- Treat this as a mono- or multi-repo (automatically detected).
- Respect existing tools/configurations (package.json, tsconfig, eslintrc, etc.).
- Exclude generated/third-party paths (node_modules, .git, dist, etc.).

GUARDRAILS:
- Keep public APIs and external contracts stable.
- Keep changes incremental and reviewable: prefer small, focused *diffs*.
- Add or adjust tests when the refactorings are non-trivial; never reduce test coverage intentionally.

WORKFLOW:
1. Inventory: Build a map of the repository and summarize the conventions.
2. Findings: Produce a list of problems with ID, Severity, Problem, Evidence, and Impact.
3. Refactoring Plan: Propose phases (Safe, Moderate, High Risk).
4. Execution: Present concrete *diffs* for the top 3-5 high-impact, low-risk changes.

Start by scanning the workspace and producing the Repository Map and the Findings Table."
```

## URL

https://imrecsige.dev/snippets/llm-prompt-for-refactoring-your-codebase-using-best-practices/
