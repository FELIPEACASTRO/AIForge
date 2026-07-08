# Few-Shot Prompting

## Description
Few-Shot Prompting is a fundamental Prompt Engineering technique that leverages the concept of **In-Context Learning** in Large Language Models (LLMs). Unlike Zero-Shot Prompting, where only the instruction is provided, Few-Shot Prompting involves including a small number of input-output pair examples (the "shots") directly in the prompt. These demonstrations serve to condition the model, showing the format, style, and type of response expected for the task. This technique is particularly effective for more complex or specific tasks, where the model's pre-trained knowledge is not sufficient, or when the output needs to follow a very particular format or style. The effectiveness of Few-Shot Prompting was first observed when models were scaled to a sufficient size, demonstrating the ability of LLMs to learn from examples provided in the immediate context of the query.

## Examples
```
**Example 1: Sentiment Classification (3-Shot)**

**Task:** Classify the customer's opinion as "Positive", "Negative", or "Neutral".

```
Review: The product arrived two days late, but the quality is excellent.
Sentiment: Neutral

Review: The interface is confusing and technical support does not respond.
Sentiment: Negative

Review: The best purchase I made this year! I recommend it to everyone.
Sentiment: Positive

Review: The price is fair and delivery was on time, but the manual is very vague.
Sentiment:
```

**Expected Output:** Neutral

---

**Example 2: Entity Extraction (3-Shot)**

**Task:** Extract the customer name and the order number from a support text, formatting the output as JSON.

```
Text: Hello, I'm Ana Silva and my order 45678 has not arrived.
JSON: {"customer": "Ana Silva", "order": "45678"}

Text: I would like to know the status of order 12345. My name is João Pereira.
JSON: {"customer": "João Pereira", "order": "12345"}

Text: The item I bought, order 98765, came wrong. I'm speaking on behalf of Maria Souza.
JSON: {"customer": "Maria Souza", "order": "98765"}

Text: Please check order 54321. My name is Carlos Eduardo.
JSON:
```

**Expected Output:** {"customer": "Carlos Eduardo", "order": "54321"}

---

**Example 3: Translation with Style (2-Shot)**

**Task:** Translate sentences from Portuguese to English, maintaining a **formal and corporate** tone.

```
Portuguese: A reunião foi adiada para a próxima semana.
English: The meeting has been postponed until next week.

Portuguese: Por favor, envie o relatório de progresso até o final do dia.
English: Kindly submit the progress report by the close of business today.

Portuguese: Precisamos de uma solução para otimizar nossos processos internos.
English:
```

**Expected Output:** We require a solution to optimize our internal processes.

---

**Example 4: Code Generation (2-Shot)**

**Task:** Generate a Python function that computes the area of geometric shapes, following the documentation and naming pattern.

```
# Example 1: Area of the Square
def calculate_square_area(side):
    """Computes the area of a square."""
    return side * side

# Example 2: Area of the Circle
def calculate_circle_area(radius):
    """Computes the area of a circle."""
    import math
    return math.pi * radius**2

# Example 3: Area of the Triangle
def calculate_triangle_area(base, height):
    """Computes the area of a triangle."""
```

**Expected Output:**
```python
    return (base * height) / 2
```

---

**Example 5: Summarization with a Specific Format (3-Shot)**

**Task:** Summarize a paragraph in a single sentence that starts with "In short,".

```
Paragraph: Generative artificial intelligence is revolutionizing content creation, allowing machines to produce texts, images, and music of ever-increasing quality. This has profound implications for creative industries and for the automation of routine tasks.
Summary: In short, generative AI is transforming content creation and automating tasks across various industries.

Paragraph: Global warming is a complex challenge that requires international cooperation, the transition to renewable energy, and the adoption of strict sustainability policies across all sectors of the economy.
Summary: In short, combating global warming requires global collaboration and an urgent shift toward sustainable practices and clean energy.

Paragraph: The Few-Shot Prompting technique allows language models to learn a new task pattern from a few examples provided in the prompt, significantly improving performance on specific tasks without the need for retraining.
Summary:
```

**Expected Output:** In short, Few-Shot Prompting enhances the performance of LLMs on specific tasks by providing a few in-context learning examples.
```

## Best Practices
1. **Consistency and Clarity:** Keep the formatting, structure, and style of the examples **uniform** and **consistent**. The model learns the pattern from the examples, and any inconsistency can lead to unpredictable outputs. 2. **Token Management:** Prioritize **concise** and **direct** examples to avoid exceeding the context window limit (*token limit*). For repetitive patterns, it is more efficient to summarize the rule than to include many long examples. 3. **Ideal Number of Examples:** The number of "shots" should be determined empirically, generally ranging between **2 to 5 examples**. More complex tasks may require more, but it is crucial to find a balance so as not to waste tokens or confuse the model. 4. **Alignment with the Task:** The examples provided should be **highly relevant** and closely aligned with the type of input and output expected for the final query. Including examples that represent challenging scenarios or "edge cases" can improve the model's robustness. 5. **Task Focus:** Keep the prompts **specific to the task**. Avoid mixing task types (such as classification and summarization) in the same prompt, unless the tasks are clearly separated and defined.

## Use Cases
1. **Text Classification:** Categorizing sentiments (positive/negative), spam detection, or classifying support tickets into specific categories. 2. **Summarization and Information Extraction:** Summarizing long texts into a specific format (e.g., a single sentence, or a JSON format), or extracting specific entities from a text. 3. **Translation and Transcreation:** Translating sentences in a particular style or tone, or translating technical terms that require a specific vocabulary. 4. **Code Generation:** Providing code examples so that the model generates new functions or scripts following the same syntax and style pattern. 5. **Style and Tone Modeling:** Generating content (marketing, creative writing) that mimics a specific tone of voice (e.g., formal, humorous, technical) or a document format (e.g., email, blog post, tweet).

## Pitfalls
1. **Formatting Inconsistency:** The most common mistake is variation in the structure of the examples, which prevents the model from identifying the input-output pattern. 2. **Excess or Shortage of Examples:** Using too many examples can lead to context window overflow and loss of information, while using too few can result in suboptimal performance. 3. **Mixing Tasks:** Trying to solve fundamentally different tasks in the same prompt without clear separation. 4. **Ignoring Token Limits:** Not considering the cost and token limit, especially in models with smaller context windows. 5. **Confusing with Complex Reasoning:** Few-Shot Prompting is not the ideal solution for tasks that require **multiple reasoning steps** (such as complex math or logic problems). In these cases, techniques such as Chain-of-Thought Prompting (CoT) are more appropriate.

## URL
[https://www.promptingguide.ai/pt/techniques/fewshot](https://www.promptingguide.ai/pt/techniques/fewshot)
