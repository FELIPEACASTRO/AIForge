# Assessment Creation Prompts

## Description
**Assessment Creation Prompts** are structured and detailed instructions provided to Large Language Models (LLMs) to generate, modify, or refine assessment materials such as tests, quizzes, assignments, and rubrics. The main goal is to save time for educators and content creators, enabling AI to act as an assistant in crafting assessment items that are valid, reliable, and aligned with specific learning objectives. The effectiveness of these prompts lies in their ability to specify the question format, difficulty level, content context, and, ideally, the inclusion of answers and justifications.

## Examples
```
1.  **Creating a Multiple-Choice Question Bank:**
    ```
    Act as a university professor of [Field: Computer Science].
    Create 10 multiple-choice questions on the topic "[Topic: Data Structures - Linked Lists]".
    The questions should be at a [Level: Intermediate/Advanced] difficulty level and test [Skill: Application and Analysis].
    For each question, provide 4 options, one of which is correct. Include the correct answer and a brief justification for the answer.
    Output format: Question, Options (A, B, C, D), Correct Answer, Justification.
    ```
2.  **Generating a Case Study for Essay Assessment:**
    ```
    Act as a senior HR consultant.
    Create a detailed case study for an essay assessment on "[Topic: Leadership in Crises]".
    The case study should include: 1) A business crisis scenario (maximum 300 words); 2) The student's role (e.g., CEO, Communications Manager); 3) An assessment question that requires a 500-word response, focused on [Focus: Communication Strategy and Ethical Decision-Making].
    ```
3.  **Difficulty Level Variation (Bloom's Taxonomy):**
    ```
    Take the following multiple-choice question: "[Original Question]".
    Rewrite this question so that it assesses [Bloom's Taxonomy Level: Evaluation/Creation], instead of merely [Original Level: Knowledge/Comprehension].
    Keep the central topic and the multiple-choice format, but create a new scenario or require deeper analysis.
    ```
4.  **Creating Distractors for an Existing Question:**
    ```
    The multiple-choice question is: "What is the main function of the HTTP protocol in web communication?" (Correct Answer: C).
    The current options are: A, B, C (Correct), D.
    My option D is weak. Generate a new option D that is a plausible but incorrect distractor, related to [Related Concept: Network Security], to confuse the student who does not master the topic.
    ```
5.  **Developing an Assessment Rubric:**
    ```
    Act as an educational assessment specialist.
    Create a 4-level analytic rubric (Exemplary, Proficient, Developing, Insufficient) to evaluate a [Project Type: Design Thinking] project.
    The rubric should have 4 main criteria: [Criteria: 1. Problem Definition, 2. Idea Generation, 3. Prototyping, 4. Presentation].
    Describe in detail what constitutes each level for each criterion.
    ```
6.  **Simulating a Specific Exam:**
    ```
    Create a mock exam of 15 multiple-choice questions on "[Topic: History of Brazil - Regency Period]".
    The style of the questions should faithfully mimic the formulation standard and complexity of the [Examining Board: ENEM].
    Include the source of each question (e.g., Text 1, Image 1) and provide the correct answer with a commented solution.
    ```
```

## Best Practices
*   **Complete Contextualization:** Always provide the learning context (course objectives, reference material, target audience) before requesting the creation of the assessment.
*   **Role Prompting:** Assign the AI a specific role (e.g., "Assessment Specialist", "Certified Exam Author") to refine the tone and quality of the questions.
*   **Format and Output Specification:** Require a clear output format (e.g., JSON, Markdown, CSV) and the question type (multiple choice, essay, etc.).
*   **Difficulty Control:** Use Bloom's Taxonomy or terms such as "application level", "analysis level", or "creation level" to control the cognitive depth of the assessment.
*   **Critical Review:** Never use generated content without thorough human review to verify factual accuracy, the validity of distractors, and pedagogical alignment.

## Use Cases
*   **Formal Education:** Rapid creation of tests, quizzes, and exams for schools, universities, and technical courses.
*   **Corporate Training:** Development of proficiency assessments and feedback questionnaires for employee training and development modules.
*   **Certifications:** Generation of question banks for professional certification exams in various fields (IT, Finance, Healthcare).
*   **Self-Assessment:** Creation of practice tests for students who want to simulate exams and test their knowledge.
*   **Market/Opinion Research:** Development of structured questionnaires and surveys for data collection.

## Pitfalls
*   **Vagueness in Instructions:** Requesting only "Create a test about X" results in superficial questions that only test memorization.
*   **Factual Hallucination:** The AI may generate incorrect questions or answers, requiring mandatory human verification.
*   **Low-Order Questions:** The AI tends to generate questions that focus on the lowest levels of Bloom's Taxonomy (remember, understand), unless explicitly instructed to go beyond.
*   **Unintended Bias:** The generated content may reflect biases present in the AI's training data, which can affect the fairness and validity of the assessment.
*   **Lack of Specific Context:** Without the source material (text, lesson), the AI may create questions that are generic or irrelevant to the exact content taught.

## URL
[https://cetli.upenn.edu/resources/generative-ai/using-ai-to-create-assessments/](https://cetli.upenn.edu/resources/generative-ai/using-ai-to-create-assessments/)
