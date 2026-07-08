# Differentiated Instruction Prompts

## Description
**Differentiated Instruction Prompts** are a Prompt Engineering technique focused on adapting the output of a Large Language Model (LLM) to meet the varied learning needs of a target audience, typically in an educational context. The goal is to personalize the content, process, product, or learning environment, aligning with the principles of Carol Ann Tomlinson's Differentiated Instruction.

The technique involves including explicit parameters in the prompt that define the **level of complexity** (e.g., reading level, vocabulary), the **delivery format** (e.g., informative article, poem, quiz, graphic organizer), the **student's interest** (e.g., *Fortnite*, *Minecraft*, sports theme), and the **cognitive skill** to be worked on (e.g., summarize, analyze, infer, apply).

By specifying these elements, the user (usually an educator) transforms the AI into a powerful tool for generating tailored teaching and assessment materials, saving time and ensuring that each student receives an appropriate and relevant challenge for their learning profile [1] [2].

## Examples
```
1. **Differentiation by Reading Level and Interest:**
   `Act as a history tutor. Create a 500-word informative article about the French Revolution. The article should be written at a 5th-grade reading level, but with an engaging tone for a 9th-grade student who loves strategy games. Include a section that compares the logistics of the Revolution with that of a strategy game.`

2. **Differentiation by Cognitive Skill and Format:**
   `You are a science teacher's assistant. Generate a set of 3 short-answer questions about the water cycle. The questions should focus on the skill of **analysis** (not just memorization) and should be formatted as a challenge for a student who already masters the basic concept.`

3. **Differentiation by Product (Output) and Level:**
   `For a student with ADHD who needs visual structure, create a blank graphic organizer (just the structure text) to help them plan an argumentative essay about the importance of recycling. The organizer should have only 3 main sections and use simple, direct language.`

4. **Differentiation by Process (Instruction):**
   `Create a 45-minute lesson plan to teach the skill of **inference** in a narrative text. The plan should include a modeling activity (I do), a guided activity (we do), and an independent activity (you do). The narrative text should be about a sporting event.`

5. **Differentiation for Multilingual Students (ELL/ESL):**
   `Translate the following paragraph about photosynthesis into Portuguese, and then create a list of 5 keywords in English with their simplified definitions in Portuguese. [PARAGRAPH HERE]`

6. **Differentiation for Enrichment (Advanced):**
   `Act as a university professor. Create a research prompt for an advanced student who already masters the concept of gravity. The prompt should require the student to explore the relationship between Einstein's theory of relativity and quantum mechanics, and should be formatted as a 1500-word essay proposal.`

7. **Material Differentiation (Multiple Levels):**
   `Generate three versions of the same informative text about the structure of a cell:
   a) Version A: 3rd-grade reading level, with simple analogies (e.g., a cell as a house).
   b) Version B: 7th-grade reading level, with standard scientific vocabulary.
   c) Version C: 11th-grade reading level, including details about organelles and their biochemical functions.`
```

## Best Practices
**Specificity and Context:** Always include the user's role (e.g., "9th-grade History Teacher"), the target audience (e.g., "Students with a 6th-grade reading level"), the output format (e.g., "A 5-question multiple-choice quiz"), and the instructional focus (e.g., "Focus on the skill of inference"). **Modularity:** Create prompts that can be easily adapted for different levels, interests, or skills by changing only one or two parameters (e.g., changing the reading level from "4th grade" to "8th grade"). **Human Verification:** Never use AI-generated content without a careful review to ensure factual accuracy, appropriate reading level, and cultural sensitivity. **Pedagogical Integration:** Use AI as an assistant for material creation, but keep the teacher's central role in providing *scaffolding* (support), *feedback*, and human connection.

## Use Cases
**Creating Personalized Teaching Materials:** Rapid generation of texts, exercises, *quizzes*, and lesson plans adapted for different reading levels, learning styles (visual, auditory, kinesthetic), and students' thematic interests. **Support for Specific Needs:** Creating materials with accommodations for students with learning difficulties, ADHD, or English language learners (ELL/ESL), adjusting the complexity of the language and the presentation format [1]. **Adaptive Formative Assessment:** Generating assessment questions that adjust in real time to the student's progress, focusing on specific skills that need reinforcement (e.g., generating more questions about "cause and effect" for a student who showed difficulty in that area). **Enrichment and Acceleration:** Creating research projects and in-depth materials for advanced students, allowing them to explore content at a higher level of complexity.

## Pitfalls
**Lack of Verification:** Blindly trusting the reading level or factual accuracy generated by the AI. The specified reading level may be inaccurate, requiring readability verification tools and human review [2]. **Replacing the Teacher:** Using AI to replace pedagogical planning and human *scaffolding* (support). AI is a material-creation tool, not a substitute for the teacher's expertise and connection [2]. **Generic Prompts:** Using vague prompts that do not specify the audience, level, or skill. This results in undifferentiated, low-quality content. **Bias and Inappropriateness:** AI may generate content that is biased or inappropriate for the student's age/culture. Sensitivity review is crucial [2].

## URL
[https://schoolai.com/blog/strategies-using-ai-tutors-improve-differentiated-instruction/](https://schoolai.com/blog/strategies-using-ai-tutors-improve-differentiated-instruction/)
