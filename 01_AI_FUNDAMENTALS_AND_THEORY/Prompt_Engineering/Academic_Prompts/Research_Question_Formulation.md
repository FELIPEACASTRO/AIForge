# Research Question Formulation

## Description
Research Question Formulation, in the context of Prompt Engineering, is the **fundamental technique of structuring and refining queries (prompts) for Large Language Models (LLMs) in order to obtain responses that are not only correct but also advance knowledge or solve a specific problem, simulating the rigor of a scientific or business investigation.** It is the art of asking the AI the "right questions", transforming a vague need into a specific, well-targeted, context-rich instruction. The focus is on the **quality of the input**, recognizing that the precision and usefulness of the LLM's output are directly proportional to the clarity and depth of the question posed. This technique aligns with the philosophy that the main focus is not "prompt engineering" itself, but rather **problem formulation** or the research question.

## Examples
```
1. **Academic (Thesis Refinement):**
```
**Context:** I am a doctoral researcher in Computer Science. My initial topic is "The Impact of Generative AI on Software Developer Productivity".
**Instruction:** Refine this topic into 3 distinct and testable research questions, each focused on a different aspect (code efficiency, developer satisfaction, and cost-benefit). For each question, suggest a research methodology (e.g., case study, controlled experiment, survey).
**Format:** Numbered list with the Research Question in bold, followed by the Suggested Methodology.
```

2. **Product Development (User Story):**
```
**Context:** We are a development team for a financial management application. We identified that many users abandon registration at the "Bank Connection" step.
**Instruction:** Formulate the main Research Question we must answer to solve this problem. Then, create a complete User Story following the format "As a [User Type], I want [Goal], so that [Benefit]".
**Constraint:** The User Story must focus on reducing friction and increasing trust.
```

3. **Business Strategy (Market Analysis):**
```
**Context:** Our B2B SaaS company is considering expanding into the European market. The product is a marketing automation tool for small and medium-sized enterprises (SMEs).
**Instruction:** Formulate the most critical Strategic Research Question we need to answer before allocating significant resources. Then, list 5 tactical sub-questions that the AI should answer to support answering the main question.
```

4. **Diagnosis and Troubleshooting (Meta-Prompting):**
```
**Instruction:** I am facing a persistent latency problem in my PostgreSQL database after the last software update. Instead of giving me a direct solution, act as a Senior Systems Engineer. Ask me 5 crucial diagnostic questions about my environment and configuration (OS version, hardware type, error logs, etc.) that you would need in order to start formulating a root cause hypothesis.
```

5. **Scientific Hypothesis Generation:**
```
**Context:** We observed that the click-through rate (CTR) on our social media ads is 30% higher on images containing the color blue compared to other colors.
**Instruction:** Formulate a testable null hypothesis (H0) and alternative hypothesis (H1) for an A/B experiment aimed at confirming or refuting this observation.
**Format:** H0: [Null Hypothesis] and H1: [Alternative Hypothesis].
```

6. **Reflection and Self-Correction (Refinement Prompt):**
```
**Instruction:** Analyze the prompt I just used: "Write a marketing email". Identify the 3 main deficiencies of this prompt in terms of context, instruction, and format. Then, rewrite the prompt to make it an advanced "research prompt" for creating a high-conversion marketing email.
```
```

## Best Practices
**Be Specific and Detailed:** Provide as much context, constraints, and intent as possible. Clarity in the input is the most critical factor for output quality. **Break Down the Question:** Divide complex tasks into a series of smaller, sequential questions. **Ask for the Reasoning (Chain-of-Thought):** Request that the AI explain the logic behind its suggestions. This enables verification and progressive refinement. **Define the Format:** Specify the desired output format (table, list, code, etc.) and the tone (formal, didactic, technical). **Iteration is Key:** Use the AI's response to refine and deepen the next question, in a continuous cycle of investigation.

## Use Cases
**Academic Research:** Generating hypotheses, refining research questions for theses and articles, and planning systematic literature reviews. **Product/Software Development:** Defining user requirements (User Stories), prioritizing *features* in *roadmaps*, and analyzing UI/UX. **Consulting and Business Strategy:** Analyzing market scenarios, formulating strategies for entering new markets, and identifying risks and opportunities. **Complex Problem Solving (Troubleshooting):** Diagnosing persistent problems in systems by asking the AI to pose diagnostic questions to understand the context. **Educational Content Creation:** Developing lesson plans and creating multiple-choice or open-ended questions based on a text.

## Pitfalls
**Ambiguity and Over-Generalization:** Using prompts that are too broad ("Help me improve my processes") or ambiguous ones that allow multiple interpretations. **Lack of Context:** Failing to provide the scenario, the AI's role, or the input data needed for the AI to understand the depth of the question. **Multiple/Compound Questions:** Trying to solve several problems in a single query, resulting in superficial and incomplete answers. **Unconscious Biases:** Framing the question in a way that leads the AI toward a predetermined answer, limiting creativity and critical analysis. **Ignoring Iteration:** Treating the prompt as a single interaction rather than a progressive refinement process.

## URL
[https://medium.com/@petrusje/engenharia-de-prompts-a-arte-de-fazer-perguntas-certas-para-ia-14f9e5c57045](https://medium.com/@petrusje/engenharia-de-prompts-a-arte-de-fazer-perguntas-certas-para-ia-14f9e5c57045)
