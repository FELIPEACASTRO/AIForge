# Sentiment Analysis Prompts

## Description
**Prompt Engineering for Sentiment Analysis** is the practice of crafting specific, structured instructions for Large Language Models (LLMs) with the goal of classifying, extracting, or quantifying the polarity (positive, negative, neutral) and the emotion (joy, anger, sadness, etc.) expressed in a text. Instead of relying on traditional Machine Learning models pre-trained on large sentiment datasets, *prompt engineering* leverages the reasoning and context-understanding capabilities of LLMs to perform the task *zero-shot* (without examples) or *few-shot* (with a few examples). This technique enables rapid adaptation to new domains and the performance of more complex analyses, such as **Aspect-Based Sentiment Analysis (ABSA)**, where sentiment is evaluated in relation to specific entities or features within the text. It is a fundamental technique in the application of LLMs to Natural Language Processing (NLP) tasks in business and research contexts, enabling the extraction of emotional *insights* from large volumes of textual data.

## Examples
```
**1. Simple Classification (Zero-Shot)**
```
Instruction: Classify the sentiment of the following text as POSITIVE, NEGATIVE, or NEUTRAL.
Text: "The service was fast, but the product arrived defective. I was frustrated."
Sentiment:
```

**2. Aspect-Based Classification (ABSA)**
```
Instruction: Analyze the text and classify the sentiment for the aspects 'Food' and 'Service' using the labels: Positive, Negative, Neutral.
Text: "The food was excellent, seasoned just right. However, the waiter took 30 minutes to bring the check."
Output Format (JSON):
```

**3. Emotion Extraction (Fine-Grained)**
```
Instruction: Which emotion (Joy, Anger, Sadness, Surprise, Fear, Disgust) is dominant in the text?
Text: "I can't believe I won the lottery! I'm jumping for joy!"
Dominant Emotion:
```

**4. Numeric Scale Classification**
```
Instruction: Classify the sentiment of the text on a scale from 1 (Very Negative) to 5 (Very Positive).
Text: "It's an okay product, does what it promises, but it didn't surprise me at all."
Rating (1-5):
```

**5. Prompt with Justification (Chain-of-Thought)**
```
Instruction: 1. Justify the sentiment of the text. 2. Classify the final sentiment as POSITIVE or NEGATIVE.
Text: "Despite arriving late, the driver was very polite and the car was clean."
Justification:
Final Sentiment:
```

**6. Sarcasm/Irony Detection**
```
Instruction: Classify the sentiment of the text. If there is irony, indicate it.
Text: "Oh, how wonderful! My flight was canceled and I'll spend the night at the airport. How lucky I am."
Sentiment:
Irony Detected: (Yes/No)
```

**7. Sentiment Review Prompt (Few-Shot)**
```
Instruction: You are a review moderator. Classify the sentiment of the text.
Example 1: Text: "I loved the movie." Output: POSITIVE
Example 2: Text: "Horrible, I don't recommend it." Output: NEGATIVE
Text: "It could be better, but it's not the worst I've seen."
Sentiment:
```
```

## Best Practices
**1. Specificity and Clarity:** Clearly define the task (classification, extraction, summarization) and the desired output format (JSON, single label, numeric scale). Use delimiters (triple quotes, XML tags) to isolate the input text from the prompt.
**2. Few-Shot Prompting:** Include 1 to 3 example pairs (input text, desired output) to guide the model, especially for more complex or domain-specific sentiment analysis tasks (e.g., financial jargon).
**3. Scale and Label Definition:** If the task is classification, provide the exact list of allowed labels (e.g., POSITIVE, NEGATIVE, NEUTRAL). For granular sentiment analysis, define a scale (e.g., 1 to 5) and what each point represents.
**4. "Think Step by Step" Instructions (Chain-of-Thought):** For ambiguous or complex texts, instruct the model to first justify its classification before providing the final label. This increases transparency and accuracy.
**5. Handling Ambiguity and Irony:** Include explicit instructions on how to deal with sarcasm, irony, or texts that contain mixed polarities, asking the model to identify the dominant intent or to classify as "Mixed/Ambiguous."

## Use Cases
**1. Social Media Monitoring (Social Listening):** Classifying the sentiment of mentions of a brand, products, or campaigns in real time to identify reputation crises or positive trends.
**2. Customer Review Analysis (Reviews):** Processing large volumes of product, service, or app reviews to extract *insights* about customer satisfaction and identify specific strengths and weaknesses (ABSA).
**3. Market and Competitive Research:** Analyzing sentiment toward competing products or market trends to inform strategic decisions.
**4. Customer Support:** Prioritizing support tickets based on the negative sentiment or frustration expressed by the customer, ensuring faster service for critical cases.
**5. Internal Feedback Analysis:** Evaluating sentiment in employee satisfaction surveys or internal communications to measure team morale and identify organizational culture issues.
**6. News Classification:** Determining the polarity of news articles about stocks or companies to support investment decisions (Financial Sentiment Analysis).

## Pitfalls
**1. Label Ambiguity:** Failing to clearly define the sentiment labels (e.g., using "Good" instead of "POSITIVE") or allowing labels outside the desired set.
**2. Lack of Domain Context:** Using generic prompts for texts from specific domains (e.g., medicine, finance) where certain words have different connotations. The LLM may fail to understand the jargon.
**3. Ignoring Irony and Sarcasm:** Failing to instruct the model to correctly detect and interpret the inverted polarity caused by figures of speech.
**4. Overly Long Prompting:** Including too many irrelevant instructions or excessive examples, which can dilute the model's focus and increase latency and cost.
**5. Model Bias:** The model may reflect biases present in its training data, leading to inconsistent or unfair classifications for certain groups or topics. It is crucial to test the robustness of the prompt.
**6. Unstructured Output:** Failing to specify an output format (e.g., JSON, XML) for the classification, resulting in free text that is difficult to process automatically.

## URL
[https://medium.com/@alexandrerays/construindo-um-classificador-de-sentimentos-com-prompt-engineering-f6673bd15a91](https://medium.com/@alexandrerays/construindo-um-classificador-de-sentimentos-com-prompt-engineering-f6673bd15a91)
