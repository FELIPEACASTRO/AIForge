# Role Prompting

## Description
**Role Prompting** is a prompt engineering technique that consists of instructing a large language model (LLM) to assume a **specific persona, role, or character** when generating a response. By assigning a role (such as "culinary critic", "lawyer", "MBA professor", or "marketing strategist"), the user guides the model to adopt a **style, tone, vocabulary, and focus** of response that is consistent with that function.

This technique is powerful because:
1. **Improves Clarity and Precision:** The model aligns its output with the expectations and knowledge inherent to the role, resulting in more contextual, higher-quality responses.
2. **Increases Reasoning Performance:** Recent studies suggest that Role Prompting can, surprisingly, improve model performance on reasoning and explanation tasks, beyond merely styling the text.
3. **Facilitates Style Imitation:** It is an obvious use case for styling text and imitating the way a specific professional or character would communicate.

## Examples
```
**1. Marketing Strategist (Business Focus)**
```
**Role:** You are a senior Digital Marketing Strategist with 15 years of experience in B2B SaaS.
**Task:** Analyze the following product pitch and suggest 3 customer acquisition channels with the highest ROI, justifying your choice for each one.
**Pitch:** [Insert the product pitch here]
```

**2. History Teacher (Education Focus)**
```
**Role:** You are a High School History Teacher, known for making complex subjects engaging and easy to understand.
**Task:** Explain the impact of the Industrial Revolution on the social structure of the 19th century to a student who has never heard about the subject. Use modern analogies.
```

**3. Senior Code Reviewer (Technology Focus)**
```
**Role:** You are a Senior Software Engineer, an expert in Python and clean design patterns. Your focus is performance and security.
**Task:** Review the code below. Identify security vulnerabilities and suggest refactorings to improve efficiency and readability.
**Code:** [Insert the Python code snippet here]
```

**4. Art Critic (Creativity Focus)**
```
**Role:** You are a renowned Art Critic, with a writing style that evokes the elegance and skepticism of the early 20th century.
**Task:** Write a 200-word critique of the digital artwork "The Garden of Bits", focusing on its composition, use of color, and cultural relevance.
```

**5. Financial Advisor (Finance Focus)**
```
**Role:** You are a CFP® (Certified Financial Planner) Financial Advisor focused on retirement planning for young professionals.
**Task:** A 28-year-old client with a stable income of R$ 8,000/month and R$ 50,000 in student debt (6% p.a. interest) asks about the best investment strategy. What is your priority advice and why?
```

**6. Sports Nutritionist (Health Focus)**
```
**Role:** You are a Sports Nutritionist with experience in ketogenic diets for endurance athletes.
**Task:** Create a one-day meal plan (breakfast, lunch, dinner, 2 snacks) for a marathon runner in an intense training phase who follows a ketogenic diet. Include the approximate macronutrient count.
```

**7. Patent Attorney (Legal Focus)**
```
**Role:** You are a Patent Attorney specializing in software intellectual property.
**Task:** Explain, in layman's terms, the difference between patent, copyright, and trade secret for a first-time entrepreneur who has developed a new algorithm.
```
```

## Best Practices
1. **Be Specific and Clear:** Define the role unambiguously. The more details about the function, the target audience, and the objective, the better.
2. **Use Non-Intimate Interpersonal Roles:** Research indicates that non-intimate interpersonal roles (such as "coworker" or "mentor") tend to produce better results than generic occupational roles.
3. **Prefer Gender-Neutral Terms:** The use of gender-neutral terms generally leads to better performance and avoids perpetuating gender biases present in the training data.
4. **Focus on the Role or the Audience:**
    * **Do:** Role Prompt – "You are a [role]."
    * **Do:** Audience Prompt – "You are speaking with a [role]."
    * **Don't:** Interpersonal Prompt – "You are speaking with your [role]."
5. **Avoid Imaginative Constructions:** It is more effective to specify the role directly than to ask the model to "Imagine that you are..."
6. **Two-Step Approach (for Reasoning):**
    * **Step 1:** Assign the role and add details. Ask the LLM to generate an initial output.
    * **Step 2:** Present the main question or task to the LLM.

## Use Cases
- **Education:** Acting as a Private Tutor or Career Mentor to explain complex concepts in an accessible way.
- **Business/Marketing:** Assuming the role of Digital Marketing Strategist or Copywriter to create content focused on conversion and ROI analysis.
- **Health:** Simulating a General Practitioner or Nutritionist to analyze health plans or diets (with the caveat that the AI's output does not replace a professional).
- **Creativity:** Playing a Film Critic or 19th-Century Poet to generate content with a specific style and tone.
- **Software Development:** Acting as a Senior Software Engineer or Code Reviewer to suggest performance and security improvements in code snippets.
- **Finance:** Acting as a CFP® Financial Advisor to provide advice on investment and retirement planning.
- **Legal:** Assuming the role of Patent Attorney to explain intellectual property concepts in layman's terms.

## Pitfalls
1. **Reinforcement of Stereotypes and Biases:** Role Prompting can inadvertently reinforce stereotypes or biased behaviors if the role is poorly represented or biased in the LLM's training data.
2. **Incorrect Representation of the Role:** If the role is not well represented in the training data, the model may respond inaccurately or inappropriately, compromising output quality.
3. **Research Limitation:** Current best practices are limited by the number of specific roles and models tested in research.
4. **Use of Intimate Roles:** Intimate interpersonal roles (such as "friend" or "mother") tend to be less effective than professional or non-intimate roles.

## URL
[https://learnprompting.org/docs/advanced/zero_shot/role_prompting](https://learnprompting.org/docs/advanced/zero_shot/role_prompting)
