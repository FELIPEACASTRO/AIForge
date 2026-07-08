# Prompt Engineering for Scientific Writing: Rules and Applications of LLMs

## Description

Prompt Engineering for Scientific Writing (Scientific Writing Prompts) is a set of techniques and best practices for using Large Language Models (LLMs) in the research and academic writing process. The main focus is to maximize the benefits of LLMs, such as accelerating writing and assisting with coding, while minimizing the inherent risks, such as hallucination and plagiarism. The guidelines focus on establishing ethical and methodological safeguards, as well as suggesting practical uses to optimize the scientific workflow. The approach emphasizes transparency, factual verification, and adherence to journals' editorial policies.

## Statistics

The use of LLMs could put up to 300 million jobs at risk globally [1]. The popularity of ChatGPT set a record for the fastest-growing user base in history [3]. The main article that establishes the "Ten Simple Rules" was published in *PLoS Computational Biology* in January 2024 [2], serving as a foundational resource for the recent academic consensus. The study highlights the risk of LLM "hallucination," citing examples where the model provided factually incorrect information but with false or mistaken references [2]. The citation rate of the main article (14 citations in 2024) indicates its rapid adoption as a reference for ethical guidelines on the use of LLMs in science [2].

## Features

**Safeguards (Rules 1-5):**
1.  **Adherence to Journal Rules:** Consult and follow the target journal's guidelines on the use of LLMs.
2.  **Risk Assessment:** Outline relevant risks (e.g., bias, inequality of access) before using the LLM.
3.  **Plagiarism Prevention:** Avoid using LLM-generated content without attribution and ensure that its use does not constitute plagiarism.
4.  **Confidentiality:** Do not share confidential data or unpublished preliminary results with the LLM.
5.  **Factual Verification:** Always verify the veracity of all LLM-generated content through a subject matter expert.

**Usage Suggestions (Rules 6-10):**
6.  **Inclusive Data Search:** Use LLMs to assist in collecting "gray literature" (NGO/government reports) for meta-analyses.
7.  **Content Summarization:** Generate concise summaries of long articles or meeting minutes to optimize reading time.
8.  **Writing Refinement:** Use LLMs to refine written English (grammar, tone, idiom), especially for non-native speakers.
9.  **Code Improvement:** Generate code *snippets*, debug errors, and translate code between programming languages.
10. **Kick-starting Writing:** Overcome writer's block and "blank page" anxiety by generating article outlines and structures.

## Use Cases

nan

## Integration

**1. Article Outline Generation (Rule 10):**
*   **Prompt:** "Act as a senior scientific reviewer. Generate a 4-section structure (Introduction, Literature Review, Methodology, Results and Discussion) for a research article. The topic is: [Effects of Climate Change on Biodiversity in Tropical Ecosystems]. The context is [Ecology] and the tone should be [Formal and Academic]. Include 3 to 4 subsections for each main section."

**2. Article Summarization (Rule 7):**
*   **Prompt:** "I want you to act as a scientific article summarizer. I will provide the text of an article. Respond with a bold title for each section, including: General Information, Background, Question/Hypothesis, Main Findings, and Contributions. The summary of each section should be concise, clear, and informative. [Insert the article text here]."

**3. Code Debugging (Rule 9):**
*   **Prompt:** "I am using Google Earth Engine and received the error 'Too many concurrent aggregations' with the following code: [Insert the problematic code]. Identify the cause of the error and suggest a solution using the `ee.List.slice()` function to split the list of IDs into smaller chunks."

**Best Practices:**
*   **Contextualization:** Always define the LLM's role (e.g., "Act as a senior reviewer", "You are a research assistant").
*   **Constraints:** Specify the output format (e.g., "4-section structure", "List in Markdown format"), the tone, and the target audience.
*   **Iteration:** Use the result of the initial prompt as the basis for follow-up prompts (e.g., "Expand subsection B of the Literature Review").

## URL

https://pmc.ncbi.nlm.nih.gov/articles/PMC10829980/
