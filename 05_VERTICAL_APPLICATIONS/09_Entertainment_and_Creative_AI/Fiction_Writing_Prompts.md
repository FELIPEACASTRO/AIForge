# Fiction Writing Prompts

## Description
**Fiction Writing Prompts** is a category of prompt engineering focused on using Large Language Models (LLMs) to assist with creative writing tasks, such as idea generation, character development, worldbuilding, dialogue creation, and drafts of scenes or chapters. The technique goes beyond a simple request, requiring the user to define complex narrative parameters (genre, tone, point of view, character arc, conflict) to guide the AI toward producing cohesive, stylistically consistent, and creative text. Its effectiveness lies in the ability to provide the AI with as much context and as many constraints as possible to avoid clichés and "generic AI writing" (AI-slop). The most advanced approaches, such as **Few-Shot Prompting** (providing examples of the desired writing) and **Chain-of-Thought** (asking the AI to reason about the story structure before writing), are crucial for achieving high-quality results and narrative complexity.

## Examples
```
**1. Scene Generation with a Defined Style (Few-Shot)**
*   **Prompt:** "Take on the role of a noir crime fiction author, like Raymond Chandler. The tone should be cynical, the prose concise, and the setting should be a rainy Los Angeles street at night. Write the opening scene where the private detective, Jack Rourke, meets a femme fatale in his office. The text should be no more than 200 words.
    *   **Style Example (Few-Shot):** 'The rain tapped on the window like a million hurried fingers. The smell of old coffee and desperation was the perfume of my room. She walked in, and the world stopped spinning. She had legs that went all the way to next year and eyes that promised trouble.'
    *   **Task:** Write the opening scene."

**2. Character Development with Internal Conflict (CoT)**
*   **Prompt:** "You are a psychologist and writer. The protagonist is a former soldier named Kael, who suffers from post-traumatic stress and now works as a gardener.
    *   **Step 1 (CoT):** Describe Kael's internal conflict in 3 key points (e.g., guilt over the war, fear of enclosed spaces, desire for redemption).
    *   **Step 2:** Write an internal monologue of Kael as he prunes a rosebush, where the act of pruning triggers a combat memory. The monologue should use gardening metaphors to describe the violence."

**3. Worldbuilding and Magic Rules (Constraint-Based)**
*   **Prompt:** "Genre: Dark Fantasy. Create a magic system called 'Shadow Weaving'.
    *   **Constraints:** The magic must be based on negative emotions (fear, envy, grief). Each use must have a physical cost (e.g., memory loss, chronic pain).
    *   **Task:** Describe the first time the protagonist, a young orphan, uses Shadow Weaving to defend herself from a guard. Describe the immediate physical cost."

**4. Dialogue with Subtext (Subtext Prompting)**
*   **Prompt:** "Write a 5-line dialogue between a father (Mr. Alistair) and his daughter (Lia) in a kitchen.
    *   **Context:** They are discussing Lia's future in college, but the real subtext is that the father fears she will abandon him, and Lia fears disappointing him.
    *   **Instruction:** None of the lines may directly mention 'fear' or 'abandonment'. The subtext must be conveyed through questions about logistics and future plans."

**5. Plot Twist Brainstorming (Iterative Prompting)**
*   **Prompt:** "Genre: Psychological Thriller. The protagonist has just discovered that his neighbor is a serial killer.
    *   **Task 1:** List 5 possible plot twists for the ending of the book.
    *   **Task 2:** Choose the most shocking twist (e.g., the protagonist is the killer, but suffers from dissociative amnesia).
    *   **Task 3:** Write the final paragraph of the book revealing this twist, using a tone of terrifying epiphany."
```

## Best Practices
**1. Set the Role and Tone (Role and Tone Setting):** Start the prompt by instructing the AI to take on a specific role (e.g., "You are a dark science fiction novelist") and a tone (e.g., "The tone should be melancholic and descriptive"). This constrains the generation space and improves stylistic coherence.
**2. Use the Prompt Structure (Context-Task-Constraint-Example):** Provide context (genre, setting), the task (what to write), constraints (word limit, point of view), and, ideally, a writing example (Few-Shot Prompting) to refine the style.
**3. Sensory and Emotional Details:** Include specific details about what the characters see, hear, smell, touch, and feel. The AI tends to focus on actions; the writer must force it to focus on the experience.
**4. Chain-of-Thought (CoT) Prompting:** For complex story arcs, ask the AI to first outline the logic of the scene or of the character development before writing the final text. E.g.: "First, list 3 ways this event affects the protagonist's motivation. Then, write the scene."
**5. Iteration and Refinement:** Do not expect perfection in the first draft. Use follow-up prompts to refine (e.g., "Rewrite the previous paragraph, increasing the suspense and shifting the point of view to that of the antagonist").

## Use Cases
**1. Overcoming Writer's Block:** Generating initial ideas, first paragraphs, or plot synopses when the author is stuck.
**2. Character and Dialogue Development:** Creating detailed character profiles, exploring their internal motivations (using CoT), or generating dialogue drafts to test a character's voice.
**3. Worldbuilding:** Defining the rules of magic systems, cultures, history, or geography of a fictional world, ensuring internal consistency.
**4. Rapid Drafting:** Generating drafts of scenes or entire chapters so the author can focus on editing and refinement, speeding up the writing process.
**5. Genre and Style Exploration:** Experimenting with different genres (e.g., *steampunk*, *cyberpunk*, magical realism) or imitating the style of specific authors (Few-Shot Prompting) to find the ideal voice for a project.
**6. Revision and Editing:** Using the AI to identify clichés, suggest alternatives for weak phrases, or rewrite a text in a different tone (e.g., from passive to active).

## Pitfalls
**1. The Vagueness Trap:** Generic prompts like "Write a love story" result in clichés and a lack of originality. The AI fills the gaps with what is statistically most likely.
**2. Information Overload:** Trying to include too many non-essential details in a single long prompt can confuse the AI, diluting the crucial instructions. It is best to use short, iterative prompts.
**3. Blind Trust:** Accepting the AI's output without critical review. The AI may introduce plot inconsistencies, continuity errors, or dialogue that sounds "robotic" (so-called *AI-slop*).
**4. The Context Vacuum:** Not defining the AI's role (e.g., "You are an editor", "You are a horror writer") or the target audience. The lack of context leads to an inconsistent tone and style.
**5. Failure to Iterate:** Treating the prompt as a single interaction. Creative writing with AI is an iterative process. Not using follow-up prompts to refine, expand, or correct the generated text is a common mistake.

## URL
[https://www.prompthub.us/blog/the-few-shot-prompting-guide](https://www.prompthub.us/blog/the-few-shot-prompting-guide)
