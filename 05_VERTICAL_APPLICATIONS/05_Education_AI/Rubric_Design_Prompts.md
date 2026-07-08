# Rubric Design Prompts

## Description
The **Rubric Design Prompts** technique refers to engineering structured and detailed prompts so that Large Language Models (LLMs) generate high-quality, clear, and customized assessment rubrics. The goal is to automate the most challenging part of creating rubrics: formulating specific **quality descriptors** for each criterion and performance level. An effective prompt should include: the AI's **role** (e.g., assessment specialist), the **task** or **project** to be evaluated, the course's **learning objectives**, the desired **scoring scale** (e.g., 4 levels), the specific **assessment criteria**, and a clear instruction for the descriptors to focus on the **quality** rather than merely the quantity of the work. This technique is widely used in Education to save time and ensure assessment consistency.

## Examples
```
**Example 1: Rubric for an Argumentative Essay (Education)**

> **Role:** You are a specialist in educational assessment and rubric design.
> **Task:** Create an analytic rubric for a 1500-word argumentative essay on "The Impact of AI on Journalistic Ethics".
> **Learning Objectives:** The student must demonstrate: 1) The ability to formulate a clear and defensible thesis; 2) Use of evidence from reliable sources; 3) A logical and cohesive structure; 4) Awareness of ethical implications.
> **Scale:** 4 levels: Excellent (4), Good (3), Satisfactory (2), Insufficient (1).
> **Criteria:** Thesis and Central Argument, Use of Evidence and Sources, Structure and Cohesion, Ethical Analysis.
> **Instruction:** Generate the rubric in Markdown table format. For each criterion and level, create a descriptor that focuses on the **quality** of execution and the depth of analysis, using clear, student-oriented language.

**Example 2: Rubric for Code Review (Technology)**

> **Role:** Act as a Senior Software Engineer specializing in Python code quality.
> **Task:** Create a rubric to assess the quality of a Pull Request (PR) from a junior developer. The PR implements a new REST API endpoint.
> **Criteria:** Readability and Style (PEP 8), Efficiency and Performance, Unit Test Coverage, Documentation (Docstrings and Comments), Error Handling.
> **Scale:** 3 levels: Senior Standard (3), Acceptable (2), Requires Refactoring (1).
> **Instruction:** Generate the rubric in nested list format. The descriptors must be technical and practical, detailing what constitutes "Senior Standard" code in each criterion.

**Example 3: Rubric for User Experience Design (Design)**

> **Role:** You are a UX/UI Designer focused on usability and accessibility.
> **Task:** Develop a rubric to evaluate a low-fidelity prototype of a mobile financial management app.
> **Objectives:** Assess intuitive navigation, compliance with accessibility guidelines (WCAG 2.1), and effectiveness in solving the user's problem.
> **Criteria:** Usability (Task Flow), Accessibility (Contrast and Font Size), Visual Consistency, Problem Resolution.
> **Scale:** 5 levels: Exceeds Expectations (5), Fully Meets (4), Partially Meets (3), Below Expectations (2), Does Not Meet (1).
> **Instruction:** The rubric must be delivered in table format. For the "Accessibility" criterion, the descriptors must reference specific WCAG principles.

**Example 4: Rubric for Performance Review (Business/HR)**

> **Role:** Human Resources consultant specializing in 360-degree performance assessment.
> **Task:** Create a rubric to evaluate the quarterly performance of a Project Manager.
> **Criteria:** Team Leadership and Mentoring, Risk and Budget Management, Stakeholder Communication, Delivery of Results (Deadline and Quality).
> **Scale:** 4 levels: Exceptional, Exceeds Expectations, Meets Expectations, Needs Improvement.
> **Instruction:** Generate a concise rubric. The descriptors must be behavioral and measurable, describing observable actions at each performance level.

**Example 5: Rubric for a Social Media Post (Marketing)**

> **Role:** Digital Marketing Strategist and Copywriter.
> **Task:** Create a rubric to evaluate the effectiveness of a single Instagram post for the launch of a new product.
> **Criteria:** Engagement (Click/Comment Rate), Message Clarity (Value Proposition), Visual Quality (Brand Alignment), Call to Action (CTA).
> **Scale:** 3 levels: High Impact, Medium Impact, Low Impact.
> **Instruction:** Generate the rubric in table format. Include a column for "Weight" (in %) for each criterion, with Engagement being the heaviest (40%).
```

## Best Practices
**1. Modular Structure:** Divide the prompt into clear sections (Role, Task, Objectives, Criteria, Scale, Instructions for Descriptors).
**2. Focus on Quality:** Explicitly instruct the AI to generate descriptors that focus on the **quality** of the work (depth of understanding, clarity, accuracy) and not just the quantity.
**3. Specificity is Key:** Provide as much detail as possible about the task, learning objectives, and criteria. The more specific, the more aligned the rubric will be.
**4. Audience Language:** Ask the AI to use language appropriate for the audience (e.g., "student-friendly language" or "technical language for peers").
**5. Output Format:** Specify the desired output format (e.g., "Generate the rubric in Markdown table format").
**6. Human Review:** Always review and adjust the AI-generated rubric. It is a drafting tool, not a final product.

## Use Cases
**1. Education and Assessment:** The primary use case. Teachers and instructors use it to quickly create rubrics for essays, projects, presentations, oral exams, and lab work, ensuring transparency and consistency in grading.
**2. Software Development:** Engineering teams use it to create Code Review Rubrics, assessing criteria such as readability, performance, security, and test coverage.
**3. Product Design (UX/UI):** Designers use it to evaluate prototypes, usability tests, and design artifacts, focusing on criteria such as usability, accessibility, and alignment with user needs.
**4. Performance Management (HR):** Human Resources departments use it to develop employee performance assessment rubrics, defining clear expectations for different seniority levels and roles.
**5. Content Creation and Marketing:** Marketing professionals use it to create content quality rubrics (blog posts, videos, social media posts), assessing engagement, SEO, message clarity, and brand alignment.

## Pitfalls
**1. Generic Descriptors:** The most common mistake is failing to be specific enough, resulting in vague descriptors such as "Good work" or "Did everything". The AI needs to be instructed to focus on **quality and specificity**.
**2. Focus on Quantity:** The prompt fails to instruct the AI to describe the **quality** of performance, resulting in descriptors that merely count items (e.g., "Included 5 sources" instead of "Integrated 5 sources critically and relevantly").
**3. Misaligned Criteria:** Failing to include the **Learning Objectives** or the task requirements in the prompt. This causes the generated rubric to assess skills that are not the focus of the work.
**4. Inadequate Scale:** Using a scoring scale (e.g., 1 to 10) without clearly defining what each point means. The AI needs quality labels (Exemplary, Proficient) to create useful descriptors.
**5. Single Long Prompt:** Trying to include all information in a single block of text without formatting. The AI processes structured and modular prompts better.

## URL
[https://blog.ctl.gatech.edu/2024/05/01/unlocking-academic-excellence-using-generative-ai-to-create-custom-rubrics/](https://blog.ctl.gatech.edu/2024/05/01/unlocking-academic-excellence-using-generative-ai-to-create-custom-rubrics/)
