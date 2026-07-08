# World-Building Prompts

## Description
**World-Building Prompts** are a Prompt Engineering technique focused on instructing Large Language Models (LLMs) to create detailed and coherent fictional environments, settings and universes. Instead of asking for a single response, the method involves a **hierarchical and iterative** approach, where the initial prompt establishes the world's fundamental rules and context, and subsequent prompts refine the details in layers (geography, history, culture, characters, etc.). The goal is to leverage the LLM's ability to maintain **contextual consistency** across long sessions, ensuring that details generated at lower levels (such as a character's personality or a building's construction material) are aligned with the global context established at the outset. This technique is essential for content creators, writers, game developers and anyone who needs a rich, ready-to-use fictional environment.

## Examples
```
1. **Global Context Creation (Step 1):**
   \`\`\`
   Create the fundamental context for a fantasy world.
   **Genre:** Post-Apocalyptic Dark Fantasy.
   **Magic Rule:** Magic is fueled by emotion, but each use drains the user's vitality, manifesting as a physical illness.
   **Technology:** Industrial Revolution level of technology, but with steam-powered machines enhanced by magic crystals.
   **Central Conflict:** The struggle between the last fortified city-states and the hordes of "Hollows" (emotionless beings) that roam the devastated lands.
   \`\`\`

2. **Location Refinement (Step 2):**
   \`\`\`
   Based on the global context, describe the city-state of **"Aethelgard"**.
   **Location:** Built inside an ancient extinct volcano, using basalt and recycled metal.
   **Government:** A military theocracy led by a "Council of Iron Priests".
   **Culture:** A society obsessed with order and emotional suppression to avoid the use of magic and the attraction of the Hollows.
   **Structure:** Describe 3 distinct districts and their purposes.
   \`\`\`

3. **Character Creation with Context (Step 3):**
   \`\`\`
   Create a character who lives in the **"The Pit"** district of Aethelgard (the poorest and most dangerous district).
   **Name:** Kael.
   **Role:** A "Steam Engineer" who secretly uses emotional magic to repair machines.
   **Personality:** Cynical, distrustful, but with a strong sense of justice.
   **Flaw:** His magic is slowly making him blind.
   **Prompt:** Describe Kael, his appearance, his motivation and a brief encounter with him.
   \`\`\`

4. **Item/Artifact Generation:**
   \`\`\`
   Create an important artifact for the world of Aethelgard.
   **Name:** The "Heart of Basalt".
   **Function:** A magic crystal that absorbs the emotion of an area, making it safe, but leaving people apathetic.
   **History:** Describe its origin and how it is used by the Council of Iron Priests.
   \`\`\`

5. **Social Rule/Law Creation:**
   \`\`\`
   What is the most important law and the most severe punishment in Aethelgard?
   **Law:** The "Law of Emotional Stillness".
   **Punishment:** Being exiled to the devastated lands, where the Hollows will find you.
   **Prompt:** Write a short public notice about this law, as it would appear on a city wall.
   \`\`\`

6. **Dialogue Prompt (Consistency Test):**
   \`\`\`
   Write a 5-turn dialogue between Kael (the Cynical Engineer) and an Iron Priest about a broken machine. The dialogue should reflect Kael's distrust and the Priest's obsession with rules.
   \`\`\`

7. **Geography Prompt:**
   \`\`\`
   Describe the landscape immediately outside the walls of Aethelgard. Include details about the flora, fauna and the ruins of the previous civilization.
   \`\`\`

8. **Historical Fact Prompt:**
   \`\`\`
   Create a crucial historical event that led to the formation of Aethelgard. The event must involve an outbreak of uncontrolled emotional magic.
   \`\`\`

9. **Culture/Religion Prompt:**
   \`\`\`
   Describe an annual ritual or festival in Aethelgard. The ritual should be a celebration of order and emotional suppression.
   \`\`\`

10. **Structure Prompt (JSON):**
    \`\`\`
    Generate a JSON list of 5 typical buildings in the **"The Pit"** district of Aethelgard. For each building, include: "name", "function", "main_material" and "size_m2".
    \`\`\`
```

## Best Practices
**Hierarchical Consistency:** Start with the global context (type of world, basic rules) and refine in layers (regions, cities, districts, buildings, characters). The consistency of the higher level should be reinforced at the lower levels. **Rich Natural-Language Context:** Use rich, unstructured descriptions for higher-level elements (such as the world's history or the city's character), since LLMs excel at maintaining context and tone. **Structured Output for Details:** For elements that need to be used in a game format or database (sizes, functions, inventory), explicitly request structured output (JSON, lists). **Character Injection:** Use the world and location context to influence character creation, avoiding the "relentless positivity" of LLMs. Ask for flaws, limitations and culturally appropriate personalities. **Iteration and Tuning:** Use the LLM to generate lists of ideas (types of cities, names) and then use your own choices to refine and fill in the details. The process is collaborative.

## Use Cases
**Creative Writing and Narrative:** Creation of rich, detailed settings for novels, short stories, screenplays and comics. **Game Development (RPG and Video Games):** Rapid generation of lore, history, factions, cities and characters for tabletop RPGs (such as D&D) or for the initial design of video game worlds. **Experience Design (UX/UI):** Creation of fictional scenarios and personas to test product usability in simulated contexts. **AI Model Training:** Generation of large volumes of consistent textual data to train or refine AI models on narrative and contextual coherence tasks. **Education:** Use in creative writing or history classes to simulate the creation of civilizations and cultures.

## Pitfalls
**Contextual Inconsistency:** The LLM may "forget" rules or details established in earlier prompts, especially in long sessions. It is crucial to reintroduce the main context (e.g., "Remember the Magic Rule:...") in subsequent prompts. **"Relentless Positivity":** The LLM's tendency to create overly positive or generic characters and scenarios. It is necessary to actively request flaws, conflicts, grit and realism. **Unstructured Generation:** Asking for complex details without specifying the format can result in text that is difficult to parse or use in a structured system (such as a game). **Information Overload:** Trying to define too many details at once in the initial prompt can dilute the LLM's focus. The approach should be hierarchical and gradual. **Lack of Conflict:** A world without inherent conflicts (social, political, environmental) will be boring. The initial prompt should establish a central conflict.

## URL
[https://ianbicking.org/blog/2023/02/world-building-with-gpt](https://ianbicking.org/blog/2023/02/world-building-with-gpt)
