# Efficient Transformers: Linformer, Performer, Reformer

## Description

Linformer, Performer, and Reformer are efficient Transformer architectures designed to overcome the quadratic complexity (O(n²)) limitation of the standard self-attention mechanism, enabling the processing of significantly longer data sequences with greater time and memory efficiency. **Linformer** achieves linear complexity (O(n)) by approximating self-attention with a low-rank matrix. **Performer** also achieves linear complexity (O(n)) through the FAVOR+ algorithm (Fast Attention Via Positive Orthogonal Random Features), which provides an unbiased estimate of softmax attention. **Reformer**, meanwhile, uses Locality-Sensitive Hashing (LSH) to reduce complexity to O(n log n) and Reversible Residual Layers to dramatically optimize memory usage, enabling context windows of up to 1 million words. All are crucial for long-sequence modeling applications in NLP, genomics, and computer vision.

## Statistics

Linformer: O(n) complexity in time and space. Performer: O(n) complexity in time and space. Reformer: O(n log n) complexity in attention, context window of up to 1 million words, memory usage of 16 GB on a single accelerator.

## Features

Linformer: Linear Attention (O(n)), Low-Rank Matrix Approximation, Memory and Time Efficiency. Performer: Linear Attention (O(n)), FAVOR+ Algorithm, Kernel-Based Attention, Compatibility with Pre-training. Reformer: LSH Attention (O(n log n)), Reversible Residual Layers, Chunking, Extreme Efficiency for Long Sequences.

## Use Cases

Long-Sequence Modeling (NLP, Genomics); Text Classification (Linformer); Image and Video Generation (Reformer); Applications in Vision Transformers (Linformer); Large-Scale Context Understanding.

## Integration

Linformer and Performer are integrated primarily via third-party PyTorch libraries (e.g., lucidrains). Reformer has official, native integration in the Hugging Face Transformers library, making it easy to use pre-trained models. All provide conceptual code examples in Python/PyTorch for implementing their attention mechanisms and architectures.

## URL

https://arxiv.org/abs/2006.04768 (Linformer), https://research.google/blog/rethinking-attention-with-performers/ (Performer), https://research.google/blog/reformer-the-efficient-transformer/ (Reformer)
