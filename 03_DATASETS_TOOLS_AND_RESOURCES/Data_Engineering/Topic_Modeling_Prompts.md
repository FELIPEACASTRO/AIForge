# Topic Modeling Prompts

## Description
**Topic Modeling Prompts** refer to the technique of using Large Language Models (LLMs) to perform the task of **Topic Modeling**, traditionally done by statistical algorithms such as LDA (Latent Dirichlet Allocation) or neural models such as BERTopic. The prompt-based approach, popularized by frameworks such as **TopicGPT** [1], reframes topic modeling as a series of natural language generation and classification tasks.

Instead of inferring word and document distributions, the LLM is instructed to:
1.  **Generate Topics:** Analyze a *corpus* of documents and create **generalizable** topic labels and concise descriptions.
2.  **Refine Topics:** Merge duplicate or irrelevant topics and organize the hierarchy.
3.  **Assign Topics:** Classify new documents into the generated topics, providing a **supporting quote** (Grounding) to justify the assignment.

This technique stands out by generating topics that are **more coherent and interpretable** by humans (semantic coherence) than traditional methods, in addition to enabling **Zero-Shot** modeling (without the need for training on a specific *corpus*) [2]. The main advantage is the ability to leverage the LLM's vast world knowledge to create high-quality topic labels.

## Examples
```
**1. High-Level Topic Generation (Based on `generation_1.txt` [1])**

\`\`\`
You will receive a list of documents. Your task is to identify high-level generalizable topics that describe the content.

[Instructions]
1. The topic labels must be as GENERALIZABLE as possible. They must not be document-specific.
2. Each new topic must have a level number (e.g., [1]), a short label, and a topic description.
3. Topics must be broad enough to accommodate future subtopics.
4. If a topic already exists in the [Existing Topics] list, return the existing topic.

[Existing Topics]
{Topics}

[Documents]
{List of input documents}

Your response must be in the format:
[Level] Topic Label: Topic Description
\`\`\`

**2. Subtopic Generation (Level 2)**

\`\`\`
The high-level topic is: [1] Technology: Discussions about digital innovations and devices.
You will receive documents that were assigned to this topic. Your task is to generate more specific subtopics.

[Instructions]
1. The new subtopics must be specific, but still generalizable within the scope of [1] Technology.
2. Each subtopic must have a level number (e.g., [1.1]), a short label, and a description.

[Documents]
{List of input documents assigned to [1] Technology}

Your response must be in the format:
[Level] Subtopic Label: Subtopic Description
\`\`\`

**3. Topic Refinement and Merging (Based on `refinement.txt` [1])**

\`\`\`
You will receive a list of topics that belong to the same level of a hierarchy. Your task is to merge topics that are paraphrases or near-duplicates.

[Rules]
1. Perform the following operations as many times as necessary:
   - Merge relevant topics into a single topic.
   - Do nothing and return "None" if no modification is necessary.
2. When merging, the output format must contain the updated level indicator, label, and description, followed by the original topics.

[Topic List]
[1] Generative AI: Models that create content.
[2] Content Creation Models: Discussions about GPTs and DALL-E.

[Your Response]
[1] Generative Artificial Intelligence: Content Creation Models and Applications. Original Topics: [1] Generative AI, [2] Content Creation Models.
\`\`\`

**4. Topic Assignment with Grounding (Based on `assignment.txt` [1])**

\`\`\`
You will receive a document and a topic hierarchy. Assign the document to the most relevant topic in the hierarchy.

[Instructions]
1. The topic labels MUST be present in the provided hierarchy. You MUST NOT create new topics.
2. The supporting quote MUST be taken from the document. You MUST NOT invent quotes.

[Topic Hierarchy]
[1] Personal Finance: Budgeting, saving, and investing.
[2] Health and Wellness: Exercise, diet, and mental health.

[Document]
"The best way to start investing is with a low-cost ETF, ensuring diversification and minimizing fees."

Your response must be in the format:
[Level] Topic Label: Assignment Reasoning (Supporting Quote)

[Your Response]
[1] Personal Finance: The document discusses investment strategies ("The best way to start investing is with a low-cost ETF...").
\`\`\`

**5. Topic Analysis Prompt (Summarization)**

\`\`\`
Act as a data analyst. The topic identified for the documents below is: **[3] Customer Feedback on Usability**.
Your task is to summarize the top 5 concerns and the top 3 improvement suggestions mentioned in the documents.

[Documents]
{List of 100 customer feedback entries}

[Output Format]
**Main Concerns:**
1. ...
2. ...
...
**Improvement Suggestions:**
1. ...
2. ...
\`\`\`
```

## Best Practices
**Clarity and Structure:** Use XML tags or clear delimiters (such as `[Document]`, `[Topics]`) to separate the input text from the instructions. This helps the LLM process the context more efficiently [1].
**Iteration and Refinement:** Do not expect the final result in a single step. Use sequential prompts (such as Generation -> Refinement -> Assignment) to build and validate the topic hierarchy, as seen in the TopicGPT framework [1].
**Generalization:** When generating topics, instruct the model to create **generalizable** labels rather than document-specific ones. This ensures the topics are useful for classifying new texts [1].
**Validation with Quotes:** Require the LLM to justify the assignment of a topic to a document with a **direct quote** from the text. This increases traceability and confidence in the result (Grounding) [1].
**Level Control:** Explicitly define the desired level of detail (e.g., "only high-level topics" or "subtopics for [Topic X]").
**Use of Hybrid Models:** To optimize costs and performance, use more powerful models (such as GPT-4 or Claude Opus) for the **Generation** and **Refinement** steps (which are less frequent) and lighter models (such as GPT-3.5 or Gemini Flash) for the **Assignment** step (which is more massive) [1].

## Use Cases
**Customer Feedback Analysis:** Automatically identify the main themes and issues in product reviews, support tickets, or social media comments.
**Legal/Regulatory Document Classification:** Categorize large volumes of legal texts into topics such as "Contract Law", "Intellectual Property", or "Environmental Regulation" with high semantic accuracy.
**Academic Research and Literature Review:** Analyze abstracts of scientific articles to identify emerging trends, research gaps, and the evolution of subfields in an area.
**News and Media Analysis:** Monitor event coverage and identify the dominant angles and narratives across different news sources.
**Market Intelligence:** Extract topics from competitor reports, patents, or earnings-call transcripts to identify market strategies and innovations.
**Content Organization:** Automatically create tags, categories, or hierarchical indexes for websites, digital libraries, or knowledge management systems.

## Pitfalls
**Dependency on LLM Quality:** The quality of the generated topics is directly proportional to the LLM's reasoning capability and context. Weaker models may generate incoherent or redundant topics.
**Cost and Latency:** Topic modeling with LLMs is significantly more expensive and slower than traditional methods (LDA, BERTopic), especially for very large *corpora*, since each document or batch requires an API call [1].
**Hallucinations in Assignment:** The LLM may "hallucinate" the supporting quote or assign a topic based on inferences that are not explicitly in the text, violating the *grounding* principle. The instruction to "NOT invent quotes" must be strict.
**Model Bias:** The generated topics may reflect the biases present in the LLM's training data, rather than reflecting only the content of the input *corpus*.
**Ambiguous Instructions:** Poorly formulated prompts or those with conflicting rules can lead to inconsistent results, such as topics that are too specific (document-specific) or too broad (vague semantics).
**Context Limit:** Topic modeling generally involves analyzing a large number of documents. An aggregation or sampling mechanism is necessary to handle the LLM's context limit. TopicGPT solves this in parts, but it is a limitation inherent to the use of LLMs [1].

## URL
[https://arxiv.org/abs/2311.01449](https://arxiv.org/abs/2311.01449)
