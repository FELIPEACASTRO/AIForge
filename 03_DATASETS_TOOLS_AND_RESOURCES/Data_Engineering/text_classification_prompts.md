# Text Classification Prompts

## Description
Text Classification is a fundamental Natural Language Processing (NLP) technique that involves categorizing a text into one or more predefined classes or labels. In the context of Large Language Models (LLMs), **Text Classification Prompts** are carefully crafted instructions that guide the model to perform this categorization task. Instead of training a traditional machine learning model, the LLM is instructed, via prompt, to act as a classifier. Effectiveness lies in the clarity and precision with which the categories and the output format are defined. Techniques such as *Zero-Shot* (no examples), *Few-Shot* (with a few examples), and specifying a structured output format (such as JSON or a single word) are crucial to ensuring consistent and usable results in data workflows [1] [2]. The main advantage is the ability to perform complex classifications without the need for large labeled training datasets, leveraging the LLM's pre-trained knowledge.

## Examples
```
**1. Sentiment Classification (Zero-Shot):**
```
Classify the following text into one of these categories: Positive, Negative, or Neutral. Return ONLY the category.
Text: "The service was fast, but the product arrived defective."
Category:
```

**2. Intent Classification (Few-Shot):**
```
You are a customer intent classifier. Classify the customer request into one of the following intents: 'Technical Support', 'Billing Inquiry', 'Service Cancellation'. If it does not fit, use 'Other'.

Example 1:
Request: "My internet stopped working and I need urgent help."
Intent: Technical Support

Example 2:
Request: "I would like to know the amount of my next bill."
Intent: Billing Inquiry

Request: "I want to close my account and stop being charged."
Intent:
```

**3. Topic Classification with Structured Output (JSON):**
```
Classify the following news article into a single main topic. The allowed topics are: 'Politics', 'Economy', 'Sports', 'Technology', 'Health'.
Return the result in JSON format with the keys "id_topico" (integer) and "nome_topico" (string).
Article: "The stock market hit a new record after the release of the inflation data."
JSON:
```

**4. Ticket Severity Classification (Numeric Scale):**
```
Classify the severity of the support ticket on a scale from 1 to 5, where 1 is 'Low' and 5 is 'Critical'. Return ONLY the number.
Ticket: "The email server is slow, but still functional for most users."
Severity:
```

**5. Document Relevance Classification (Boolean):**
```
Determine whether the following paragraph is relevant to the topic 'AI Regulation'. Answer ONLY with 'Yes' or 'No'.
Paragraph: "The new bill aims to establish ethical guidelines for the development and use of artificial intelligence systems in the public sector."
Relevant:
```
```

## Best Practices
**1. Clear Definition of Categories:** MUST define the categories unambiguously and mutually exclusively. Provide a brief description of each category within the prompt.
**2. Output Format Specification:** Always instruct the LLM to return the classification in a structured format (e.g., JSON, a single word, or a specific label) to facilitate automated *downstream* processing.
**3. Use of Few-Shot Prompting:** For more complex tasks or domain-specific categories, include 2 to 5 example pairs (input text, output label) to guide the model.
**4. Single-Output Instructions:** Ask the model to return *only* the category label, without explanations or additional text, unless the explanation is explicitly requested.
**5. Handling Ambiguity/Other:** Include an "Other" or "Not Applicable" category and instruct the model to use it when the text does not clearly fit the defined categories.

## Use Cases
**1. Sentiment Analysis:** Classifying product reviews, social media comments, or customer feedback as positive, negative, or neutral.
**2. Support Ticket Routing:** Automatically classifying support tickets or customer emails to route them to the correct department (e.g., Billing, Technical, Sales).
**3. Content Filtering:** Classifying user-generated content or news articles into categories such as *spam*, *inappropriate content*, *fake news*, or specific topics (e.g., Sports, Politics).
**4. Document Classification:** Categorizing legal, medical, or business documents (e.g., Contract, Invoice, Report) for organization and information retrieval.
**5. Purchase Intent Analysis:** Classifying *chatbot* interactions or call transcripts to determine customer intent (e.g., *Interest*, *Doubt*, *Ready to Buy*).

## Pitfalls
**1. Ambiguous Categories:** Defining categories that overlap or are not clear, leading to inconsistent or incorrect classifications by the LLM.
**2. Lack of Output Constraint:** Not specifying the output format, resulting in long, unstructured responses that are difficult to process automatically.
**3. Model Bias:** The LLM may introduce biases present in its training data, affecting the classification of sensitive texts or minority groups.
**4. Overfitting in Few-Shot:** Using *Few-Shot* examples that are too specific or that do not represent the diversity of the real dataset, limiting the model's ability to generalize.
**5. Ignoring Context:** Not providing enough context about the domain or the task, causing the LLM to use its general knowledge instead of the specific classification rules.
**6. Overly Long Prompt:** Including too many categories or examples, which can exceed the model's context limit or dilute the main instruction.

## URL
[https://www.promptingguide.ai/prompts/classification](https://www.promptingguide.ai/prompts/classification)
