# Flashcard Generation Prompts

## Description
**Flashcard Generation Prompts** are prompt engineering instructions designed to have Large Language Models (LLMs) transform raw texts, lecture notes, articles, or complex concepts into structured flashcards (usually in Question/Answer or Cloze Deletion format). The technique aims to automate the tedious part of creating flashcards, allowing the user to focus on review and active learning. The effectiveness of these prompts is maximized when combined with science-based learning principles, such as **Active Recall** and **Spaced Repetition**.

## Examples
```
1. **Basic Prompt (Strict Question/Answer)**
```
Act as an active learning expert. Analyze the following text and extract 10 crucial facts. For each fact, create a flashcard in the strict format:
Question: [The question should require recalling a single fact.]
Answer: [The answer should be concise and direct.]
Use the following text: [PASTE THE TEXT HERE]
```

2. **High-Retention Prompt (Exam/Test)**
```
Create 15 flashcards on the topic "[TOPIC]", in question and answer format. Mix key concepts, definitions, and include 3 "exam traps" (questions that test exceptions or easily confused details).
Output Format:
### Flashcard [NUMBER]
Question:
Answer:
```

3. **Prompt with Bloom's Taxonomy (Application Level)**
```
Based on the concept of "[CONCEPT]", generate 5 flashcards.
- 2 should be at the 'Remember' level (definition).
- 2 should be at the 'Understand' level (explain in your own words).
- 1 should be at the 'Apply' level (present a scenario and ask how the concept would be used).
Use the Question/Answer format.
```

4. **Prompt for Cloze Deletion**
```
Transform the following paragraph into 5 flashcards in Cloze Deletion format, ideal for Anki. The blank should be placed on a key word or phrase.
Output Format:
[TEXT WITH BLANK]
Answer: [REMOVED WORD/PHRASE]
Paragraph: [PASTE THE PARAGRAPH HERE]
```

5. **Prompt with Chain-of-Thought (CoT) for Quality**
```
You are a high-quality flashcard generator. Before generating the flashcard, follow the Chain-of-Thought (CoT) process:
1. Identify the main fact in the text.
2. Formulate a concise question that requires recalling that fact.
3. Provide the direct answer.
4. Create the final flashcard.
Generate 8 flashcards from the text: [PASTE THE TEXT HERE]
Output Format:
CoT Process: [Your reasoning]
Flashcard: Question: [X] | Answer: [Y]
```

6. **Prompt for Language Flashcards (Vocabulary)**
```
Create 10 English vocabulary flashcards for the B2 level, using the words from the following text. Each flashcard should include:
1. The word in English.
2. The definition in Portuguese.
3. An example sentence in English.
Text: [PASTE THE ENGLISH TEXT HERE]
```

7. **Prompt for Programming Flashcards (Concept/Syntax)**
```
Generate 7 flashcards on the concept of "Object-Oriented Programming (OOP)" in Python. Include:
- 3 definition cards (Classes, Objects, Inheritance).
- 2 syntax cards (Question: How do you define a class in Python? Answer: [EXAMPLE CODE]).
- 2 use-case cards (Question: Give an example of polymorphism).
```

8. **Comparison Prompt (Analysis)**
```
Create a comparison flashcard between "[CONCEPT A]" and "[CONCEPT B]".
Question: What are the 3 main differences between [CONCEPT A] and [CONCEPT B]?
Answer: [List of 3 concise differences].
Use the reference text: [PASTE THE TEXT HERE]
```
```

## Best Practices
1. **One Fact Per Card Principle:** The prompt should instruct the LLM to create flashcards that address only a single fact, concept, or skill per card. This reduces cognitive load and improves retention.
2. **Active and Contextual Phrasing:** The prompt should require questions to be phrased actively (requiring generation, not recognition) and to provide enough context to be unambiguous.
3. **Clear Output Structure:** Define a strict output format (e.g., `Question: [X] | Answer: [Y]`) or a table/CSV format to make it easier to import into spaced repetition systems (such as Anki or Quizlet).
4. **Mapping to Bloom's Taxonomy:** Ask the LLM to create cards that cover different cognitive levels (Remember, Understand, Apply, Analyze). Start with "Remember" cards and progress to "Apply" or "Analyze" cards to deepen mastery.
5. **Use of Advanced Techniques (CoT and Few-Shot):** For higher-quality results, incorporate **Chain-of-Thought (CoT)** and **Few-Shot Prompting** (providing 1-2 examples of ideal flashcards) to guide the LLM in the precise and concise extraction of facts.
6. **Human Quality Control:** Always manually review a sample of the generated cards (10-20%) to check for accuracy, conciseness, and ambiguity. Human personalization and editing activate the **Generation Effect**, improving memorization.

## Use Cases
*   **Academic Studies:** Transform lecture notes, textbook summaries, or scientific articles into sets of flashcards for exams.
*   **Language Learning:** Create cards for vocabulary, verb conjugation, and contextual sentences.
*   **Preparation for Competitive Exams/Certifications:** Generate cards focused on key concepts, exam "traps", and case law.
*   **Programming/Technical Topics Learning:** Create cards for language syntax, algorithms, terminal commands, or software architecture concepts.
*   **Medicine and Sciences:** Generate cards for anatomy (with image occlusion), pharmacology (mechanism vs. clinical use), and biochemical pathways.

## Pitfalls
1. **Long or Ambiguous Cards:** The LLM may generate cards with questions or answers that are too long, violating the "one fact per card" principle, or questions that may have multiple correct answers.
2. **Over-Reliance:** Blindly trusting AI-generated content without manual review can lead to memorizing incorrect or poorly formulated information, losing the benefit of the **Generation Effect**.
3. **Focus Only on Facts:** Generating only "Remember" level cards (definitions, dates) and neglecting higher-level cards (Application, Analysis), resulting in superficial memorization.
4. **Incompatible Format:** The LLM may not adhere to the requested output format, making automated import into spaced repetition software difficult.
5. **Card Overload:** Generating an excessive number of cards (more than 20-30 new ones per day) can lead to burnout and the inability to maintain a spaced review routine.

## URL
[https://blog.educate-ai.com/en/flashcards-creation-modern-methods-tips-tools-for-effective-learning](https://blog.educate-ai.com/en/flashcards-creation-modern-methods-tips-tools-for-effective-learning)
