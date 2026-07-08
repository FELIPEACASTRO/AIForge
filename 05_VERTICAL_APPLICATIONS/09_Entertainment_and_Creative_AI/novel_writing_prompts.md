# Novel Writing Prompts

## Description
The **Novel Writing Prompts** technique is a specialized application of Prompt Engineering focused on using Large Language Models (LLMs) to assist across all phases of creating a novel, from initial *brainstorming* to chapter editing and revision. Instead of asking the AI to write the entire book all at once, the method involves creating structured, interactive prompts that guide the AI in building complex elements such as *world-building*, detailed character profiles, plot arcs, glossaries, and appendices. The goal is to turn the AI into a writing co-pilot that maintains narrative consistency and deepens the project's reference material.

## Examples
```
### Prompt Examples (5-10 Concrete Examples)

**1. Initial Setup Prompt (Master Prompt):**
```
# Role and Context
You are an expert Grimdark Fantasy writer and editor, with meticulous attention to detail. Your goal is to assist in creating a high-quality, well-structured novel.

## Writing Style
- Maintain a cynical and pessimistic tone.
- Use dense descriptive prose and concise dialogue.
- Third person perspective, past tense.
- Avoid clichés like "dive into" or "unleash your potential".

## Process
- Review all of the project's knowledge files before responding.
- Maintain narrative consistency with the "World-Building Framework" and the "Character Profiles".
- End each interaction with a question that helps me move the novel forward.
```

**2. Reference Document Creation Prompt:**
```
* TITLE: The Last Breath of the Sun
* Genre: Post-Apocalyptic Dark Fantasy
* Premise: In a world where the sun died a century ago, a drug-addicted former paladin must escort a child who holds the key to reigniting the light across lands infested by creatures of darkness.
* Protagonist: Kael, a fallen paladin, cynical, with a broken moral code and an addiction to "Star Dust" (a local drug).

Create the following documents for this novel, detailing them extensively:
1. World-Building Framework (Cosmology, Geography, Magic System, Society).
2. Character Profile: Kael (History, Motivations, Character Arc).
3. Supporting Cast Profiles (Main Allies and Antagonists).
4. Plot Outline (Chapter-by-Chapter Outline, Character Arcs, Themes).
```

**3. Chapter Generation Prompt:**
```
Based on the "Plot Outline" and using the "World-Building Framework" and the "Character Profile: Kael" as context, develop the Draft of Chapter 3.

Before writing, create a detailed execution plan for the chapter, including:
- Chapter Objective (What should be achieved).
- Key Plot Points.
- Conflict (Internal and External).
- Vocabulary Keywords (for consistency).
- Estimated Word Count (minimum 2500 words).

The chapter should end with Kael making a risky decision that puts him in immediate danger.
```

**4. Reference Document Update Prompt:**
```
Chapter 3 has been completed. Please update the following documents to incorporate all the new terms, locations, and events introduced in Chapter 3:
1. Plot Outline (Mark Chapter 3 as completed and revise the progression of Chapter 4).
2. Glossary (Add new world-specific terms, such as "Star Dust" and "The Lightless").
3. Index (Add chapter references for new secondary characters and locations).
```

**5. Critique and Editing Prompt (Development):**
```
Analyze the text of Chapter 5 (attached) and provide a developmental critique.

Focus on the following aspects:
- **Pacing:** Where does the action slow down or speed up too much?
- **Continuity:** Are there any inconsistencies with the "World-Building Framework" or Kael's "Character Profile"?
- **Dialogue:** Does the dialogue between Kael and the Fallen Paladin sound authentic and advance the plot?

Provide a detailed report with concrete revision suggestions.
```

**6. Scene Expansion Prompt:**
```
The current scene (lines 15-30 of Chapter 7) is too short. Expand this scene to focus on the sensory description of the environment: the smell of ozone and burnt metal, the sound of the wind howling through the ruins, and the feeling of fine sand under Kael's boots. Increase the word count by at least 500.
```

**7. Dialogue Generation Prompt:**
```
Create a tense dialogue between the protagonist (Kael) and the main antagonist (The Shadow).

**Context:** Kael has been captured. The Shadow is trying to convince him to join her cause, exploiting Kael's broken moral code and his drug addiction.
**Objective:** The dialogue should reveal an unexpected weakness in the Shadow and a moment of hesitation in Kael.
```

**8. Title/Cover Brainstorming Prompt:**
```
Generate 10 alternative titles for the novel "The Last Breath of the Sun" that are darker and evoke a sense of despair and dark fantasy. In addition, describe in detail a cover image that captures the essence of the genre and the premise.
```
```

## Best Practices
1.  **Establish a "Master Prompt":** Define an initial prompt that establishes the AI's role (e.g., "Expert novel writer and editor"), the context, and the main responsibilities (e.g., maintaining consistency, producing content without length restrictions).
2.  **Maintain an Up-to-Date Knowledge Base:** Use the platform's file system or context feature (such as Claude's "Project Knowledge" or NovelCrafter's "Codex") to store and continuously update reference documents (Chapters, Character Profiles, World Structure, Plot Outline, Glossary, Appendix).
3.  **Modular and Iterative Structure:** Divide the writing process into modules (chapters, scenes, profiles) and use a new conversation thread for each main chapter or scene. This helps manage the context window and ensures the AI focuses only on the most relevant information.
4.  **Define a Writing Style:** Include specific instructions about tone, point of view (POV), verb tense, and even stylistic aversions (e.g., "Avoid excessive use of em dashes or semicolons") to ensure the generated prose aligns with the author's voice.
5.  **Constant Review and Refinement:** After generating each chapter or element, use editing and critique prompts (e.g., "Critique the pacing, structure, and continuity of this chapter") to refine the material and ensure cohesion with the rest of the work.

## Use Cases
*   **Brainstorming and Initial Conception:** Generate ideas for titles, premises, genres, and high-level structures.
*   **World-Building:** Create extensive details about cosmology, magic systems, geography, politics, and historical timelines.
*   **Character Development:** Craft psychological profiles, character arcs, motivations, and distinct voices for protagonists and supporting cast.
*   **Detailed Plot Outlining:** Create a chapter-by-chapter breakdown, including character arc progression and thematic elements.
*   **Consistency Maintenance:** Use AI to update reference documents (Glossary, Index) with each new chapter, ensuring that terms, locations, and events are used consistently.
*   **Editing and Critique:** Request a developmental critique from the AI on the pacing, structure, and continuity of an already-written chapter.

## Pitfalls
*   **Context Inconsistency:** Failure to update the AI's knowledge base with new chapters or revisions, leading to continuity errors (the biggest problem in long projects).
*   **Generic AI "Voice":** The AI may fall into clichés or use generic language if the prompt does not include a detailed style guide or examples of the author's writing.
*   **Overreliance:** Trusting the AI to write the complete draft without author intervention, resulting in a story that lacks emotional depth and authentic voice.
*   **Context Window Limitation:** On platforms without a "project" or "knowledge base" feature, the conversation context can be lost, requiring the author to input the "story so far" summary or the last chapter with each new prompt.
*   **Repetitive Names:** The AI may reuse character or location names if it is not instructed to use random name generators or if the author does not review and replace the generated names.

## URL
[https://www.reddit.com/r/WritingWithAI/comments/1kje334/aiassisted_novel_writing_guide/](https://www.reddit.com/r/WritingWithAI/comments/1kje334/aiassisted_novel_writing_guide/)
