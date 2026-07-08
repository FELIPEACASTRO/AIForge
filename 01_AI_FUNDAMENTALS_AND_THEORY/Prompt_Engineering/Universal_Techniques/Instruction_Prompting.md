# Instruction Prompting

## Description
**Instruction Prompting** is the fundamental and most basic technique of Prompt Engineering, centered on the ability of Large Language Models (LLMs) to follow directives expressed in natural language. It consists of providing clear and concise commands to the model so that it performs a specific task. Unlike more advanced techniques that require examples (Few-Shot) or chain-of-thought reasoning (Chain-of-Thought), Instruction Prompting relies purely on the instruction to guide the model to perform new or previously unseen tasks, without the need for specific training or large labeled datasets. It is the foundation for effective communication with AI, allowing users to transform complex tasks (such as text formatting, data extraction, or evaluation) into simple and scalable commands.

## Examples
```
**1. Data Extraction and Formatting**

**Prompt:**
```
Read the following text and extract the full name, job title, and company. Format the output as a JSON object.

Text: "Dear Mr. João Silva, as Project Manager at TechSolutions, I would like to schedule a meeting."
```

**2. Conditional Summary**

**Prompt:**
```
Summarize the following article in English. The summary should be no more than 100 words and be written in a formal, objective tone.

Article: [Insert the article text here]
```

**3. Classification and Categorization**

**Prompt:**
```
Classify the following sentence into one of the categories: [Sales], [Technical Support], [Billing], or [General].

Sentence: "My bill this month seems incorrect, I need help to verify the amounts."
```

**4. Code Generation with Constraints**

**Prompt:**
```
Write a Python function named 'calcular_imc' that takes two arguments (weight in kg and height in meters) and returns the Body Mass Index (BMI). Do not include comments in the code.
```

**5. Translation and Tone Adaptation**

**Prompt:**
```
Translate the following paragraph from English into Portuguese. The tone of the translation should be informal and friendly.

Paragraph: "The prompt engineering technique is essential for maximizing the utility of large language models."
```

**6. Structured Evaluation and Feedback**

**Prompt:**
```
Evaluate the following text passage based on two criteria: 'Grammar' and 'Clarity'. Assign a score from 1 to 10 for each criterion and provide a brief rationale for the score.

Text: "Despite popular belief, there is no solid evidence to support the idea that video games lead to violent behavior."
```
```

## Best Practices
**Clarity and Specificity:** Use direct action verbs (e.g., "Write", "Analyze", "Compare") and avoid ambiguities. The more specific the command, the better the result. **Separation of Instruction and Context:** Use delimiters (such as `###`, `"""`, or XML tags) to clearly separate the main instruction from the context or input data. **Instructions First:** Place the most important instruction at the beginning of the prompt to ensure the model prioritizes it. **Iteration and Refinement:** Start with simple instructions and refine them progressively based on the model's outputs. **Avoid Negations:** Focus on what the model should do, not on what it should not do. For example, use "Summarize in 50 words" instead of "Do not write more than 50 words".

## Use Cases
**Data Processing:** Formatting and extracting structured data (e.g., names, addresses, dates) from unstructured text. **Writing and Editing:** Generating content with specific constraints of format, tone, length, or style (e.g., formal emails, short social media posts). **Classification and Labeling:** Assigning categories or labels to texts (e.g., classifying support tickets, sentiment analysis). **Automated Review and Feedback:** Evaluating texts (e.g., essays, summaries) based on defined criteria, providing scores and justifications. **Removal of Sensitive Information (PII):** Automatically identifying and replacing personal data (e.g., names, phone numbers, emails) in documents for privacy purposes.

## Pitfalls
**Ambiguity and Imprecision:** Using vague language or terms with multiple meanings. The model may interpret the instruction differently from the user's intent. **Conflicting Instructions:** Including commands that contradict each other in the same prompt (e.g., "Be concise, but detail all points"). **Information Overload:** Trying to include too many complex tasks in a single instruction, which can cause the model to ignore parts of the command. **Lack of Delimiters:** Not separating the instruction from the input data, causing the model to confuse what is the command and what is the context. **Overconfidence:** Assuming that the model will understand the context or intent without it being explicitly stated. **Use of Negations:** Telling the model what **not** to do (e.g., "Do not use jargon") is less effective than telling it what **to do** (e.g., "Use simple language").

## URL
[https://learnprompting.org/docs/basics/instructions](https://learnprompting.org/docs/basics/instructions)
