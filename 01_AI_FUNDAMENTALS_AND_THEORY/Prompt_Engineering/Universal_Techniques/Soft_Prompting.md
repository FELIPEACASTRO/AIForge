# Soft Prompting

## Description

**Soft Prompting** (also known as *learned prompts*, *continuous prompts* or *prompt embeddings*) is a **Parameter-Efficient Fine-Tuning (PEFT)** technique that adapts pre-trained Large Language Models (LLMs) to specific tasks without the need to train all of their parameters. Unlike **Hard Prompts** (discrete, manually created textual prompts), Soft Prompts are **learnable tensors** (vectors of *virtual tokens*) that are concatenated with the model's input *embeddings* and optimized directly on a training dataset.

This approach allows the model to remain frozen, while only a small set of prompt parameters is trained, resulting in significantly more efficient adaptation in terms of time and computational cost. The main downside is that these *virtual tokens* are not human-readable.

## Statistics

*   **Parameter Efficiency:** Prefix Tuning demonstrated performance comparable to full *fine-tuning*, but with **1000x fewer** trainable parameters.
*   **Scalability:** The performance of Prompt Tuning **scales** with increasing model size, matching traditional *fine-tuning* on larger models.
*   **Recent Research (2024):** The work "Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models" (ICLR 2024) investigated the **Low-Norm Effect** in *soft-prompts* for Vision-Language Models (VLMs), suggesting that reducing the norm of certain learned prompts can **improve the performance** of VLMs.

## Features

Soft Prompting encompasses several PEFT sub-techniques, each with variations in how the prompt *embeddings* are inserted and optimized:

1.  **Prompt Tuning:** Adds learnable prompt *tokens* only to the input *embeddings*. Good for text classification and scalable with model size.
2.  **Prefix Tuning:** Inserts optimizable prefix parameters into **all** layers of the model. Ideal for Natural Language Generation (NLG).
3.  **P-Tuning:** Uses a prompt encoder (such as LSTM) and allows the prompt *tokens* to be inserted **anywhere** in the input sequence. Designed for Natural Language Understanding (NLU).
4.  **Multitask Prompt Tuning (MPT):** Learns a single prompt for multiple types of tasks, enabling efficient *transfer learning*.
5.  **Context-Aware Prompt Tuning (CPT):** Refines only the *embeddings* of specific context *tokens* to improve *few-shot* classification.

## Use Cases

*   **Model Adaptation:** Efficient adaptation of pre-trained LLMs to a wide variety of *downstream* tasks (e.g.: classification, generation, NLU) without the need for full *fine-tuning*.
*   **Low-Data Environments:** Prefix Tuning is particularly effective in *low-data settings*.
*   **Transfer Learning:** Multitask Prompt Tuning enables *transfer learning* of a single learned prompt to multiple tasks.
*   **Multimodal Models:** Recent research applies Soft Prompting in Vision-Language Models (VLMs) such as CLIP for task adaptation.

## Integration

Since Soft Prompts are learnable tensors and not human-readable text, there are no "prompt examples" in the traditional sense of input text. Integration occurs through the implementation of one of the sub-techniques (PEFT).

**Best Practices (PEFT):**
*   **Technique Choice:** The choice of sub-technique depends on the task: **Prompt Tuning** for classification, **Prefix Tuning** for NLG and **P-Tuning** for NLU.
*   **Implementation:** Use libraries such as Hugging Face's **🤗 PEFT (Parameter-Efficient Fine-Tuning)**, which provides ready-made implementations.
*   **Optimization:** Optimization is done via *backpropagation* on the training dataset, updating only the prompt parameters while the base model remains frozen.

## URL

https://huggingface.co/docs/peft/en/conceptual_guides/prompting