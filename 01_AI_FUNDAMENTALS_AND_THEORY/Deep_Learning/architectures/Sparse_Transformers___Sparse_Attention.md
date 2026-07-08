# Sparse Transformers / Sparse Attention

## Description

Sparse Transformers and Sparse Attention are architectural innovations that solve the quadratic complexity ($O(n^2)$) bottleneck of standard Transformers with respect to sequence length ($n$). The core technique is the sparse factorization of the attention matrix, which restricts which tokens interact, reducing complexity to $O(n\sqrt{n})$ (original Sparse Transformer) or $O(n)$ (in variants such as BigBird and Longformer). This enables the efficient processing of sequences with tens of thousands of tokens, which was infeasible with dense attention. Sparse attention encompasses subtypes such as **local attention** (where each token attends only to nearby neighbors) and **global attention** (where special tokens attend to the entire sequence).

## Statistics

- **Longer Sequences:** Enables sequences that are **10x** longer for BERT base and **16x** longer for BERT large in pre-training, compared to dense attention.
- **Faster Computation:** Training speedup of up to **6.3x** for BERT base, **5.3x** for BERT large, and **6.1x** for GPT2 (DeepSpeed).
- **Inference Improvement:** Up to **3.13x** faster inference on BERT-Base compared to Longformer (DeepSpeed).
- **Memory Complexity:** Reduces the memory footprint to $O(wn)$, where $w$ is the size of the local attention window.

## Features

- **Complexity Reduction:** Transforms time and memory complexity from $O(n^2)$ to $O(n\sqrt{n})$ or $O(n)$.
- **Long Sequence Processing:** Enables the processing of sequences with thousands of tokens (up to 4096 in Longformer, tens of thousands in the original Sparse Transformer).
- **Flexible Attention Mechanisms:** Supports **local**, **global**, and **random** attention, or any combination of these.
- **Sparsity Structures:** Implements popular structures such as **Fixed** (OpenAI Sparse Transformer), **BigBird**, and **BSLongformer**.
- **Hardware Optimization:** Implementations such as DeepSpeed Sparse Attention use GPU-optimized sparse kernels, such as those developed with Triton.

## Use Cases

- **Generative Modeling of Long Sequences:** Pixel prediction in images or long text generation.
- **Long Document Processing:** Analysis of extensive documents, such as financial reports, scientific papers, or legal texts (e.g., Long Document Comprehension).
- **Language Model Pre-training:** More efficient training of models such as BERT and GPT2 with larger input sequences.
- **Computer Vision:** Processing of very long pixel or patch sequences.
- **Tabular Data Analysis:** Application of transfer learning to tabular data.

## Integration

The most robust integration is through training optimization libraries such as **DeepSpeed Sparse Attention** or through pre-implemented models in libraries such as **Hugging Face Transformers**.

**1. Integration with DeepSpeed (Python/PyTorch):**
DeepSpeed replaces the standard attention module with optimized sparse kernels, configured via a JSON file.

```python
# Example sparsity configuration in DeepSpeed (ds_config.json)
{
  "sparse_attention": {
    "enabled": true,
    "block_size": 64,
    "sparsity_structure": "fixed", // "fixed", "bigbird", "bslongformer", or "variable"
    "local_window_size": 256,
    "num_global_blocks": 1,
    "random_blocks": false
  }
}

# Run training with the DeepSpeed launcher
# deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
```

**2. Integration with Hugging Face (Longformer/BigBird):**
Models such as Longformer and BigBird already have sparse attention implemented internally.

```python
from transformers import LongformerModel, LongformerTokenizer
import torch

# Initialize the Longformer tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

# Example of long text
long_text = "..." * 1000 # Sequence with more than 512 tokens

# Tokenize the text and set the global attention mask (required for Longformer)
inputs = tokenizer(long_text, return_tensors='pt', max_length=4096, truncation=True)
global_attention_mask = torch.zeros_like(inputs['input_ids'])
global_attention_mask[:, 0] = 1 # Set the CLS token as global

# Pass the global attention mask to the model
outputs = model(**inputs, global_attention_mask=global_attention_mask)
```

## URL

https://openai.com/index/sparse-transformer/ | https://arxiv.org/abs/1904.10509 | https://www.deepspeed.ai/2020/09/08/sparse-attention.html
