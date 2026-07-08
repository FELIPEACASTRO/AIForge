# Recommendation System Prompts

## Description
Prompt engineering for Recommendation Systems (RS) involves creating natural-language instructions for Large Language Models (LLMs) with the goal of generating **personalized recommendations** of items (such as news, products, movies, etc.) for specific users. This approach leverages the LLMs' reasoning, context comprehension, and natural language generation capabilities to simulate the behavior of a traditional recommendation system.

Instead of relying solely on complex matrix models, LLMs can process the user's interaction history, item profiles, and task constraints (provided in the prompt) to produce recommendation lists and, crucially, **explanations** for those recommendations. The **RecPrompt** framework [1] is a notable example that uses an automated feedback loop to optimize prompts, demonstrating that LLM-generated prompts can outperform traditional deep neural models on certain recommendation metrics.

## Examples
```
## Example 1: News Recommendation (Based on RecPrompt)
**Context (System Prompt):** You are a personalized news recommendation system. Your task is to analyze the user's reading history and a list of candidate articles to suggest the 5 most relevant articles.

**Input (User Prompt):**
```
# User Profile
News Reading History:
1. "Advances in Nuclear Fusion: New Reactor Breaks Energy Record"
2. "The Impact of AI on Digital Content Creation"
3. "Water Crisis in Europe: Rationing Measures"

# Candidate Articles
1. "Google's New Privacy Policies"
2. "Discovery of an Exoplanet with Potential for Life"
3. "The Future of Electric Cars and Sustainability"
4. "Analysis of the Latest Season of a Science Fiction Series"
5. "Investment Trends in Renewable Energy"

# Task
Based on the user's history, rank the 5 Candidate Articles by relevance.
Output Format:
<START>
[Most relevant article]
[Second most relevant]
...
[Least relevant article]
<END>
```

## Example 2: E-commerce Product Recommendation
**Context (System Prompt):** You are a shopping assistant. Recommend 3 products from the candidate list that best align with the user's shopping profile and current goal.

**Input (User Prompt):**
```
# User Profile
- Recent Purchases: DSLR Camera, 50mm Lens, Professional Tripod.
- Browsing: ND Filters, Camera Equipment Backpacks.
- Current Goal: Buy accessories for landscape photography.

# Candidate Products
1. Ergonomic Camera Backpack (Large Capacity)
2. Sensor Cleaning Kit
3. Drone with 4K Camera
4. 128GB High-Speed Memory Card
5. Book: "Mastering Portrait Photography"

# Task
Recommend the 3 most suitable products for the landscape photography goal. Include a brief justification for each.
```

## Example 3: Movie/Series Recommendation with Explanation
**Context (System Prompt):** Act as a movie curator. Recommend a movie or series and provide a detailed explanation (maximum 50 words) of why it fits the user's taste.

**Input (User Prompt):**
```
# Viewing History
- Favorite Movies: "Inception", "Interstellar", "Blade Runner 2049" (Science Fiction, Complex Themes, Strong Visual Direction).
- Preferred Genres: Sci-Fi, Psychological Thriller.
- Avoided Genres: Romantic Comedy, Slasher Horror.

# Task
Recommend a single title that I will probably love.
```

## Example 4: Constraint-Based Recommendation (Travel)
**Context (System Prompt):** You are a travel agent specialized in personalized itineraries.

**Input (User Prompt):**
```
# Travel Preferences
- Destination: Europe.
- Duration: 10 days.
- Budget: Medium (excluding airfare).
- Interests: Ancient History, Local Cuisine, Fewer Crowds.

# Task
Suggest 3 European cities that meet these constraints. For each city, list one historical landmark and one typical dish.
```

## Example 5: Code/Tool Recommendation (Technology)
**Context (System Prompt):** You are a software development expert. Recommend the best Python library for the described task.

**Input (User Prompt):**
```
# Task
I need to implement a natural language processing (NLP) system to classify sentiment in large volumes of text (more than 1 million documents). The system must be scalable and fast.

# Task
Recommend the most suitable Python library (excluding NLTK and SpaCy) and justify the choice in terms of scalability and performance.
```
```

## Best Practices
1. **Define the Persona (Role-Playing):** Start the prompt with a clear role instruction (e.g., "You are a movie curator", "You are an e-commerce expert"). This aligns the style and focus of the LLM's response.
2. **Provide Detailed Context:** Include as much relevant user data as possible (interaction history, preferences, demographics, current intent) and the candidate items. Recommendation quality is directly proportional to the richness of the context.
3. **Structure the Output (Output Formatting):** Specify the desired output format (JSON, numbered list, XML tags such as `<START>` and `<END>`). This makes it easier to parse and integrate the LLM's response into a larger recommendation system.
4. **Request Explanations (Explainability):** Ask the LLM to justify its recommendations. **Explainability** is one of the biggest benefits of LLMs in RS, increasing user trust.
5. **Use Chain-of-Thought Reasoning:** For complex tasks, instruct the LLM to "think step by step" (e.g., "1. Analyze the profile. 2. Filter the candidates. 3. Rank and justify."). This improves the accuracy and traceability of the recommendation process.

## Use Cases
*   **Content Recommendation:** Suggesting news, articles, videos, music, or podcasts.
*   **E-commerce and Retail:** Recommending products, "next purchase" suggestions, or creating personalized collections.
*   **Travel and Tourism:** Suggesting destinations, hotels, activities, or itineraries based on constraints and interests.
*   **Education (EdTech):** Recommending courses, study materials, or personalized learning paths.
*   **Health:** Suggesting medical articles, wellness plans, or healthcare professionals based on history and symptoms.
*   **Cold Start Recommendation:** Using prompts to infer the preferences of new users based on demographic information or answers to initial questions.

## Pitfalls
*   **Item Hallucinations:** The LLM may invent items or products that do not exist in the provided candidate list, requiring an external validation step.
*   **Popularity Bias:** LLMs tend to favor popular or widely discussed items, even if they are not the most relevant to the user's niche profile.
*   **Cost and Latency:** Running complex prompts on large LLMs (such as GPT-4) can be slow and expensive, making them unsuitable for high-frequency, low-latency recommendation systems.
*   **Context Dependency:** Recommendation quality is highly dependent on the amount of context the LLM can process (token limit), which can be a problem for users with very long interaction histories.
*   **Lack of Interaction:** LLMs do not interact with the environment in a traditional way (like a reinforcement learning model), limiting their ability to learn from real-time feedback unless a framework such as RecPrompt is implemented.

## URL
[https://www.prompthub.us/blog/recprompt-a-prompt-engineering-framework-for-llm-recommendations](https://www.prompthub.us/blog/recprompt-a-prompt-engineering-framework-for-llm-recommendations)
