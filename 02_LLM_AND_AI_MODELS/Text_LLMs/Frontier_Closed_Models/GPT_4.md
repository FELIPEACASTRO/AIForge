# GPT-4

## Description

GPT-4 is a large multimodal model (accepting image and text inputs, emitting text outputs) that represents the latest milestone in OpenAI's effort to scale up deep learning. Its unique value proposition lies in its reliability, creativity, and ability to handle much more nuanced instructions than GPT-3.5.

## Statistics

Exam Performance: Passed a simulated bar exam with a score around the top 10% (GPT-3.5 was in the bottom 10%). MMLU (Multilingual Massive Multitask Language Understanding): Outperforms the English-language performance of GPT-3.5 and other LLMs in 24 of the 26 languages tested. Model Size (unofficial estimate): Estimated to have 1.5 trillion parameters, compared to 175 billion for GPT-3.5.

## Features

Multimodality: Accepts text and image inputs (Vision). Reliability and Steerability: Significant improvements in factuality and the ability to steer the AI's behavior through "system" messages. Creativity: More creative and capable of handling complex, nuanced instructions. Context: Ability to process a larger context (up to 32k tokens in some versions).

## Use Cases

Solving complex problems in professional domains (e.g., law, medicine). Multimodal content creation (e.g., image description). Socratic tutoring and personalized AI agents (via Steerability). Programming and code optimization.

## Integration

Available via API and ChatGPT Plus. Supports system messages to customize the AI's style and task. Integration Example (Python - via `openai` SDK): ```python\nfrom openai import OpenAI\nclient = OpenAI()\nresponse = client.chat.completions.create(\n    model=\"gpt-4\",\n    messages=[\n        {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},\n        {\"role\": \"user\", \"content\": \"Explain the concept of instruction tuning in one sentence.\"}\n    ]\n)\nprint(response.choices[0].message.content)\n```

## URL

https://openai.com/index/gpt-4-research/
