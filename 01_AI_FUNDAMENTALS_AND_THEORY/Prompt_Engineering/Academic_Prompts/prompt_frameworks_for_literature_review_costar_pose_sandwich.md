# Prompt Frameworks for Literature Review (CO-STAR, POSE, Sandwich)

## Description

This resource details the application of specific **Prompt Engineering Frameworks** to optimize **Academic Research and Literature Review** using Large Language Models (LLMs) such as ChatGPT. The **CO-STAR**, **POSE**, and **Sandwich** frameworks provide systematic structures for creating prompts, aiming to improve the quality, relevance, and depth of AI-generated literature reviews. The focus is on transforming interaction with the AI from a simple query into a structured, didactic process that is essential for the academic environment.

## Statistics

Studies (Islam et al., 2025) demonstrate that exposure to these frameworks significantly improves students' prompting behavior, resulting in more effective prompts and higher-quality literature reviews. The improvement is observed in the structure and organization of the work. The **Markdown Table + CoT** framework proved to be highly effective (up to 94.35% accuracy in one study) for tasks involving the extraction of structured data from academic abstracts (Lee et al., 2025). The use of frameworks helps mitigate the risk of hallucinations and confabulations, a growing problem in LLMs.

## Features

**CO-STAR (Context, Objective, Style, Tone, Audience, Response):** A comprehensive structure for defining all parameters of the AI's output. **POSE (Persona, Output Format, Style, Example):** Focused on assigning a role to the AI and providing a clear response format and examples. **Sandwich (Iterative/Defensive):** Emphasizes iteration (Draft -> Feedback -> Refinement) and prompt robustness (Initial Instruction -> Content -> Final Instruction). Ability to reduce hallucinations and improve accuracy in extracting academic information.

## Use Cases

1. **Structuring Academic Work:** Creating outlines, summaries, and research plans for articles, theses, and dissertations. 2. **Structured Data Extraction:** Collecting and organizing specific information (metrics, methodologies, results) from multiple abstracts or articles. 3. **Critical Analysis and Gap Identification:** Generating sections that critically evaluate the existing literature and point out areas for future research. 4. **Writing Refinement:** Adjusting the style, tone, and formality of the text to meet academic publication standards.

## Integration

**Example Prompt (CO-STAR):** "Context: I am a master's student writing a literature review on 'The impact of AI on higher education'. Objective: Generate a detailed outline of the review. Style: Academic and formal. Tone: Neutral and analytical. Audience: Professors and peers. Response: Present the outline in Markdown format with 5 main sections. Include subsections for research gaps and future directions." **Best Practices:** 1. **Define the Role (POSE):** Begin the prompt with "Act as a senior researcher in [your field]". 2. **Provide Examples (POSE):** Include a short text excerpt and the desired output format. 3. **Iterate (Sandwich):** Use the initial output as a draft and provide specific feedback to refine the critical analysis and originality. 4. **Use CoT (Chain-of-Thought):** Ask the AI to "Think step by step" before providing the final answer.

## URL

https://arxiv.org/abs/2509.01128
