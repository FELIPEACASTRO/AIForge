# Prompt Compression (LLMLingua)

## Description
**LLMLingua** is a coarse-to-fine prompt compression technique developed by Microsoft Research to accelerate the inference of Large Language Models (LLMs) and reduce costs, especially in scenarios with long prompts, such as Chain-of-Thought (CoT) and In-Context Learning (ICL) [1].

The technique is based on the premise that natural language is inherently redundant and that LLMs can infer from compressed prompts, even if they are difficult for humans to understand. The method uses an auxiliary Small Language Model (SLM) (such as GPT2-small or LLaMA-7B) to calculate perplexity (PPL) and identify redundant *tokens* and sentences in the prompt.

The compression process involves three main components [2]:
1.  **Budget Controller:** Defines the target compression rate and allocates the token "budget" to different parts of the prompt, preserving semantic integrity.
2.  **Iterative Token-Level Compression:** Removes redundant *tokens* iteratively, modeling the interdependence between the compressed content.
3.  **Distribution Alignment:** Uses *instruction tuning* to align the auxiliary SLM with the language distribution of the target (black-box) LLM, ensuring that the compression is effective for the final model.

LLMLingua has demonstrated the ability to achieve up to **20x compression** with little performance loss on tasks such as reasoning and ICL, resulting in a 1.7x to 5.7x speedup in end-to-end latency [3].

## Examples
```
**Example 1: RAG Context Compression (Retrieval-Augmented)**
*   **Original Prompt (Input to LLMLingua):** "Context: [10,000 tokens of retrieved documents about the merger of Company X and Y]. Question: What was the main strategic reason for the merger, according to the document?"
*   **Compressed Prompt (Output to LLM):** "Context: [500 compressed tokens]. Question: Main strategic reason for the merger?"

**Example 2: ICL Example Compression (In-Context Learning)**
*   **Original Prompt (Input to LLMLingua):** "Example 1: [Complex math problem and detailed solution with CoT]. Example 2: [Another problem and solution]. Question: Solve the following problem: [New problem]."
*   **Compressed Prompt (Output to LLM):** "Example 1: [Compressed CoT]. Example 2: [Compressed CoT]. Question: Solve: [New problem]."

**Example 3: Long Dialogue Compression (Chatbot)**
*   **Original Prompt (Input to LLMLingua):** "Conversation History: [50 conversation turns]. User: I want to know the return policy for electronic items purchased more than 30 days ago."
*   **Compressed Prompt (Output to LLM):** "History: [Compressed essence of the 50 turns]. User: Return policy for electronics > 30 days."

**Example 4: Meeting Summary Compression**
*   **Original Prompt (Input to LLMLingua):** "Meeting Transcript: [20,000 tokens of transcript]. Instruction: Generate an executive summary with the top 3 decisions and the people responsible."
*   **Compressed Prompt (Output to LLM):** "Transcript: [1,000 tokens of compressed transcript]. Instruction: Executive summary: 3 decisions, people responsible."

**Example 5: Detailed Instruction Compression**
*   **Original Prompt (Input to LLMLingua):** "Instructions: You are a Python expert. Respond concisely, use only the Pandas library, and the output format must be JSON. The task is: [Task description]."
*   **Compressed Prompt (Output to LLM):** "Instructions: Python expert. Concise response. Use Pandas. JSON output. Task: [Task description]."
```

## Best Practices
**1. Prioritize Key Information:** Use LLMLingua to compress the part of the prompt that contains contextual information (such as ICL examples or RAG documents) and keep the main instruction and the question intact.
**2. Use an Optimized Small Model:** The effectiveness of LLMLingua depends on a small model (such as GPT2-small or LLaMA-7B) to calculate perplexity and identify redundancy. Make sure the auxiliary model is aligned with the target LLM.
**3. Monitor the Compression Rate:** Start with lower compression rates (for example, 5x) and increase gradually, monitoring the performance metric (such as EM or accuracy) to find the ideal balance point between cost/speed and quality.
**4. Leverage Recoverability:** In critical scenarios, use a powerful LLM (such as GPT-4) to decompress the compressed prompt, ensuring that no essential information has been lost.
**5. RAG Integration:** Integrate LLMLingua into RAG *frameworks* (such as LlamaIndex or LangChain) to compress the retrieved documents before passing them to the LLM, optimizing cost and latency.

## Use Cases
**1. Cost and Latency Optimization in APIs:** Drastic reduction of the number of input *tokens* sent to black-box LLMs (such as GPT-4 or Claude) via API, resulting in cost savings and lower response latency.
**2. RAG Frameworks (Retrieval-Augmented Generation):** Compression of retrieved documents and excerpts before they are inserted into the LLM prompt, allowing the inclusion of more relevant context and improving information density.
**3. In-Context Learning (ICL) Acceleration:** Compression of long, detailed ICL examples (including chains of thought - CoT) to maintain the LLM's reasoning capability with a smaller prompt.
**4. KV Cache Compression:** Use of the technique to compress the *Key-Value Cache* during inference, which improves decoding speed and allows longer contexts.
**5. Long Context Summarization:** Application in very long context scenarios, such as meeting transcripts, conversation histories, or lengthy documents, to extract the essence and facilitate summarization or question answering.

## Pitfalls
**1. Loss of Critical Information:** Excessive compression (very high rates, such as 20x without validation) can remove *tokens* or sentences that, although they appear redundant, are crucial to the accuracy of the LLM's response, especially in complex reasoning tasks (CoT).
**2. SLM Misalignment:** If the auxiliary Small Language Model (SLM) is not well aligned with the target (black-box) LLM, the compression can be ineffective or harmful, since the SLM may not correctly identify redundancy from the LLM's point of view.
**3. Compression Latency:** Although LLMLingua reduces the LLM's inference latency, the compression process itself introduces additional latency. On very short prompts, the time spent on compression can offset the speed gain in inference.
**4. Debugging Difficulty:** The compressed prompt is nearly illegible to humans. This makes debugging and optimizing the prompt much more difficult, since the researcher cannot easily inspect what the LLM is actually receiving.
**5. Additional SLM Cost:** Running the auxiliary SLM to perform the compression adds a computational cost (and potentially a financial one, if it is a paid service) that must be considered in the overall cost-benefit calculation.

## URL
[https://arxiv.org/abs/2310.05736](https://arxiv.org/abs/2310.05736)
