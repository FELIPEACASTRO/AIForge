# Efficient Long-Context Transformer Variants (Infini-attention, ModernBERT, HMT)

## Description

Transformer variants for 2024-2025 focus on solving the efficiency and context limitations of the original models. Three notable architectures are **Infini-attention**, **ModernBERT**, and the **Hierarchical Memory Transformer (HMT)**. Infini-attention, proposed by Google, enables an "infinite" context by combining masked local attention with a long-term compressive memory, keeping computational and memory complexity bounded. ModernBERT is a modernization of BERT, focused on efficiency and extended context (up to 8k tokens), using improvements such as *Rotary Positional Embeddings* (RoPE) and GeGLU layers. HMT mimics human memory, using a hierarchical system to preserve and retrieve relevant information across long sequences, turning the model into a memory-augmented recurrent model. These innovations are crucial for applications that require deep understanding of extensive documents and maintaining coherence in long conversations.

## Statistics

**Infini-attention:** Time complexity $O(1)$ and memory complexity $O(1)$ with respect to sequence length (after the first segment). Enables context windows of 1 million tokens or more. **ModernBERT:** Trained on 2 trillion tokens. Context of up to 8k tokens. Performance 9 percentage points higher than competing long-context models on retrieval tasks with ColBERT. **HMT:** Demonstrates consistent improvement in long-context generation quality across various Transformer architectures.

## Features

**Infini-attention:** Infinite context with bounded memory and time complexity. Combines masked local attention and long-term linear attention with compressive memory. **ModernBERT:** Improved efficiency, extended context (up to 8k tokens), superior performance on *embedding* tasks and long-context retrieval (with ColBERT). Uses RoPE and GeGLU. **HMT:** Long-context processing through hierarchical memory. Mimics human memorization, using memory-augmented recurrence to preserve and retrieve relevant tokens.

## Use Cases

**Infini-attention:** Analysis of ultra-long documents (contracts, financial reports, extensive source code), chatbots with unlimited conversational memory, and processing of time-series data sequences. **ModernBERT:** High-quality text *embedding* tasks, information retrieval in large databases (*retrieval-augmented generation* - RAG), and understanding documents with extended context. **HMT:** Coherent text generation in long narratives, summarization of books or scientific articles, and Question Answering tasks over extensive documents.

## Integration

Integration is mainly facilitated through the **Hugging Face Transformers** library.
**ModernBERT:** Pre-trained models are available on the Hugging Face Hub (e.g., `answerdotai/ModernBERT-base`).
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

inputs = tokenizer("Example text for ModernBERT.", return_tensors="pt")
outputs = model(**inputs)
# The output contains the embeddings of the last hidden state
```
**Infini-attention and HMT:** PyTorch implementations are available in GitHub repositories, which can be integrated into existing model architectures.
**Infini-attention (PyTorch implementation example):**
```python
# Conceptual example of using the Infini-attention layer
from infini_attention_pytorch import InfiniAttention

# Initialize the layer
infini_attn = InfiniAttention(dim=512, heads=8)

# Apply attention to an input sequence
output = infini_attn(x) # x is the input tensor
```
Adopting these variants requires replacing the standard attention mechanism with the new architecture (Infini-attention) or using pre-trained models (ModernBERT).

## URL

Infini-attention (Paper): https://arxiv.org/abs/2404.07143; ModernBERT (Hugging Face): https://huggingface.co/blog/modernbert; HMT (Paper): https://arxiv.org/abs/2405.06067