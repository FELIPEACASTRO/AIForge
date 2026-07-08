# Zero-Shot Prompting

## Description
**Zero-Shot Prompting** is the most fundamental and direct way of interacting with a Large Language Model (LLM). The technique consists of providing the model with an instruction or question to perform a specific task, without including prior examples (demonstrations) of input-output pairs. The model must rely entirely on its internal knowledge, acquired during pre-training on vast datasets, and on its ability to follow instructions (enhanced by techniques such as *Instruction Tuning* and *RLHF* - Reinforcement Learning from Human Feedback) to generate the appropriate response. It is the standard and simplest approach, ideal for well-defined tasks and for modern models that already have high generalization capability.

## Examples
```
1. **Sentiment Classification:**
   ```
   Classify the sentiment of the following text as "Positive", "Negative" or "Neutral".
   Text: "Despite the delay, the service was impeccable and the product exceeded my expectations."
   Sentiment:
   ```

2. **Entity Extraction:**
   ```
   Extract the person's name, the organization and the location from this text.
   Text: "Dr. Ana Silva, CEO of TechSolutions, will give a talk in São Paulo next week."
   Person:
   Organization:
   Location:
   ```

3. **Simple Translation:**
   ```
   Translate the following sentence into Portuguese.
   Sentence: "The quick brown fox jumps over the lazy dog."
   Translation:
   ```

4. **Text Summarization:**
   ```
   Summarize the paragraph below in a single sentence.
   Paragraph: "Photovoltaic solar energy is the leading growing source of renewable energy in the world. It converts sunlight directly into electricity, using photovoltaic cells, and has a significantly lower environmental impact than fossil fuels."
   Summary:
   ```

5. **Code Generation (Simple Function):**
   ```
   Write a function in Python that computes the factorial of a positive integer.
   ```

6. **Question Answering (Factual):**
   ```
   What is the capital of Canada and what is its official language?
   ```

7. **Style Rewriting:**
   ```
   Rewrite the following sentence in a more formal and professional tone.
   Sentence: "We gotta figure out a way to wrap this up soon, okay?"
   Rewrite:
   ```
```

## Best Practices
*   **Be Explicit and Clear:** The instruction should be as clear and detailed as possible, defining the task, the output format and any constraints.
*   **Use Delimiters:** For longer prompts or prompts with input data, use delimiters (such as `###`, `"""`, or XML tags) to separate the instruction from the context or the data.
*   **Specify the Output Format:** Explicitly request the desired format (e.g., "Respond in JSON format", "List in bullet points", "Only the keyword").
*   **Negative Instructions (Avoid):** Avoid telling the model what *not* to do. Instead, tell it what it *should* do. For example, instead of "Do not include the introduction", say "Start directly with the first point".
*   **Modern Models:** Use more recent, enhanced (*Instruction-Tuned*) models, as they are inherently more effective at Zero-Shot Prompting.

## Use Cases
*   **Fast Classification:** Classifying emails, support tickets or customer comments into predefined categories (e.g., urgency, topic, sentiment).
*   **Data Extraction:** Extracting specific information (names, dates, values) from documents or unstructured text.
*   **Translation and Summarization:** Simple sentence translation tasks or concise summaries where extreme contextual accuracy is not critical.
*   **Initial Content Generation:** Creating drafts, titles, or outlines of articles and code for simple, direct tasks.
*   **Model Capability Testing:** Quickly assessing the generalization capability of a new LLM across a variety of tasks.

## Pitfalls
*   **Ambiguity:** Vague or ambiguous instructions lead to inconsistent or incorrect results. The model has no examples to infer the intent.
*   **Complex Tasks:** Not suitable for tasks that require multi-step reasoning, planning or very specific and uncommon knowledge (where Few-Shot or Chain-of-Thought would be better).
*   **Excessive Reliance:** Relying too much on the model's generalization capability for tasks where formatting or style are crucial.
*   **Lack of Context:** Failing to provide the context necessary for the task, assuming that the model "knows" what you want.
*   **Hallucinations:** In fact- or data-generation tasks, the absence of examples or context can increase the likelihood of the model "hallucinating" information.

## URL
[https://www.promptingguide.ai/techniques/zeroshot](https://www.promptingguide.ai/techniques/zeroshot)
