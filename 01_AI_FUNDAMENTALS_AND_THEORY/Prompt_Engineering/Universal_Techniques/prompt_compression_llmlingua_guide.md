# Prompt Compression - LLMLingua

## Description

**Prompt Compression** is a prompt engineering technique that aims to reduce the length (number of tokens) of the input prompt for Large Language Models (LLMs) while preserving the essential information and the intent of the task. The main goal is to mitigate the challenges of long prompts, such as increased inference cost, reduced processing speed, and high memory consumption. The technique is crucial for optimizing applications such as Retrieval-Augmented Generation (RAG), where the retrieved context can be excessively long.

Compression can be categorized into two main methods:
1.  **Hard Prompt Methods:** Involve the direct removal of low-information tokens (Filtering) or rewriting the prompt for conciseness (Paraphrasing). Examples include SelectiveSentence and LLMLingua (which uses a smaller LLM to identify and remove non-essential tokens).
2.  **Soft Prompt Methods:** Compress the text into a smaller number of special tokens or *embeddings* (such as GIST or ICAE), which are then used by the main LLM.

The most notable and successful implementation is **LLMLingua**, which uses a smaller, faster LLM (such as GPT2-small or LLaMA-7B) to calculate the perplexity and importance of each token, removing those that contribute least to the essential information.

## Statistics

- **Compression Rate:** LLMLingua achieves up to a **20x compression rate** (a 95% reduction in tokens) with minimal performance loss on ICL (In-Context Learning) tasks [1] [2].
- **Cost Reduction:** Inference cost reduction of up to **3.5x** [3] or **80%** in RAG (Retrieval-Augmented Generation) applications [4].
- **Performance Improvement:** LongLLMLingua, a variation of the technique, demonstrated a performance improvement of **17.1%** with a **4x** compression in long contexts [1].
- **Capability Retention:** In empirical studies, the LLM retained between **62.26% and 72.89%** of its original performance even with significant compression [5].
- **Citation:** The LLMLingua technique (Compressing Prompts for Accelerated Inference of Large Language Models) was published in 2023 [6].

**References:**
[1] https://www.llmlingua.com/
[2] https://medium.com/@sahin.samia/prompt-compression-in-large-language-models-llms-making-every-token-count-078a2d1c7e03
[3] https://builder.aws.com/content/2n9wWygDkfoZd74eAsaBNtEjZON/prompt-compression-using-amazon-bedrock-reduce-rag-costs
[4] https://towardsdatascience.com/how-to-cut-rag-costs-by-80-using-prompt-compression-877a07c6bedb/
[5] https://medium.com/@sahin.samia/prompt-compression-in-large-language-models-llms-making-every-token-count-078a2d1c7e03
[6] https://arxiv.org/html/2310.05736v2

## Features

- **Cost and Latency Reduction:** Drastically reduces the number of input tokens, resulting in lower costs per API call and faster inference.
- **Performance Preservation:** Maintains the main LLM's responsiveness, ensuring that information critical to the task is retained.
- **Context Optimization:** Allows more relevant context to fit within the LLM's context window, improving performance on Retrieval-Augmented Generation (RAG) and In-Context Learning (ICL) tasks.
- **Hybrid Approach (LLMLingua):** Uses a smaller compression model (such as GPT2-small) to pre-process the prompt, making the technique agnostic to the main model (it can be used with GPT-4, Claude, etc.).
- **Multi-Component Compression:** LLMLingua allows the compression of different parts of the prompt, such as `Instructions`, `Context`, and `Examples`.

## Use Cases

- **Retrieval-Augmented Generation (RAG):** Optimization of the context retrieved from vector databases, where the information may be redundant or excessive. Compression ensures that only the most relevant excerpts are passed to the LLM, reducing costs and latency.
- **In-Context Learning (ICL):** Compression of ICL examples (few-shot examples) within the prompt, allowing more examples to fit in the context window and improving the model's learning capability.
- **Dialogue and Chatbot Applications:** Reduction of conversation history to maintain relevant context without exceeding the token limit or increasing costs prohibitively.
- **Low-Latency Inference:** Use in applications that require fast responses, such as real-time virtual assistants, where reducing prompt processing time is critical.
- **Cost Optimization:** Implementation in any LLM application where the cost per input token is a significant concern.

## Integration

**Best Practices and Integration Examples (LLMLingua):**

1.  **Prompt Splitting:** For the best compression, the prompt should be divided into components: `Instructions`, `Context`, and `Questions`. LLMLingua allows the user to define the relative importance of each section.
2.  **Use in RAG:** Compression is ideal for the retrieved context in RAG applications. The long, redundant context is compressed before being passed to the main LLM, reducing cost by up to 80%.
3.  **Code Example (Conceptual):**
    ```python
    from llmlingua import LLMLingua

    # Initialize the compressor with a small model
    llm_lingua = LLMLingua(model_name="gpt2")

    long_prompt = {
        "instruction": "Answer the question based on the provided context.",
        "context": "The context retrieved from a vector database, containing many paragraphs about the history of AI and the development of LLMs...",
        "question": "What is the main benefit of prompt compression?"
    }

    # Compress the prompt
    compressed_prompt = llm_lingua.compress_prompt(
        long_prompt,
        rate=0.5, # Desired compression rate (50% of the original size)
        force_context_compress=True
    )

    # The compressed prompt is then sent to the main LLM (e.g., GPT-4)
    # response = llm_main.generate(compressed_prompt)
    ```
4.  **Rate Configuration:** Start with lower compression rates (e.g., 2x or 4x) and increase gradually, monitoring the performance drop. LLMLingua achieves the best balance between compression and performance around 4x.

## URL

https://www.llmlingua.com/
