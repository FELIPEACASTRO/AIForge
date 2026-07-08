# The Pile (EleutherAI)

## Description
The Pile is a vast and diverse open-source language modeling dataset created by EleutherAI. With an approximate size of 825 GiB (or 886 GB), it is composed of 22 smaller, high-quality subsets, spanning a wide range of domains. Its main characteristic is diversity, which aims to improve the general knowledge and generalization ability of Large Language Models (LLMs) trained on it. The dataset was released in 2020 but remains a fundamental reference in LLM research. A more recent version focused on open licenses, called "The Common Pile v0.1" (8 TB), was announced in 2025.

## Statistics
**Size:** Approximately 825 GiB (or 886 GB).
**Composition:** 22 distinct data subsets.
**Main Version:** The original version was released in 2020.
**Recent Related Version:** "The Common Pile v0.1" (8 TB), announced in June 2025, is a version focused on public domain content and open licenses.
**Samples:** There is no total number of documents readily available, but the dataset is composed of more than 400 billion tokens.

## Features
Domain Diversity: Includes 22 data subsets, such as code, scientific articles (arXiv, PubMed Central), books, chat conversations (Ubuntu IRC), legal documents (FreeLaw), and web pages (Pile-CC). Curated Quality: The subsets were carefully selected to ensure high quality and relevance for LLM training. Open Source: Publicly available to the research community. Format: The data is provided in `jsonlines` format compressed with `zstandard`.

## Use Cases
Training general-purpose Large Language Models (LLMs), such as GPT-J and GPT-NeoX. Evaluating the generalization ability and world knowledge of language models (using Pile BPB - Bits Per Byte). Research on data diversity and its impact on language model performance. Fine-tuning models for specific tasks in domains such as science, medicine, and programming.

## Integration
The Pile dataset can be accessed and downloaded from several sources. The primary download source was The Eye, but Hugging Face is the most recommended and up-to-date integration method for use in Machine Learning projects.

**Via Hugging Face Datasets (Recommended):**
```python
from datasets import load_dataset

# O dataset completo é muito grande, o Hugging Face geralmente requer
# a especificação de um subconjunto ou o uso de streaming.
# Para carregar o dataset completo (pode ser inviável devido ao tamanho):
# dataset = load_dataset("EleutherAI/pile")

# Para carregar um subconjunto específico (ex: Pile-CC):
# dataset = load_dataset("EleutherAI/pile", "pile_cc")

# Para carregar o dataset em modo streaming (recomendado para datasets grandes):
# dataset = load_dataset("EleutherAI/pile", streaming=True)
```

**Direct Download:**
Direct download of the `.jsonl.zst` files can be done through mirrors such as Academic Torrents or community repositories, since the original The Eye link may be inactive. It is recommended to check the official GitHub repository for updated download links.

## URL
[https://pile.eleuther.ai/](https://pile.eleuther.ai/)
