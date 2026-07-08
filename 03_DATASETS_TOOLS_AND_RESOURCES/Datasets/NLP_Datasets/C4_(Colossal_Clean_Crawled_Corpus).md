# C4 (Colossal Clean Crawled Corpus)

## Description
**C4 (Colossal Clean Crawled Corpus)** is a massive English text dataset, cleaned and crawled from the web, created by Google for the pre-training of the T5 model (Text-to-Text Transfer Transformer). It is derived from an April 2019 *snapshot* of Common Crawl, applying a series of rigorous filters to remove low-quality content, duplicates, source code, incomplete sentences, and non-English content. The main goal is to provide a high-quality corpus for training large-scale language models (LLMs). Although the original version is text-only, recent research (2023-2025) highlights the importance of its multimodal extension, **Multimodal C4 (mmc4)**, which interleaves millions of images with the text, expanding its use to multimodal models.

## Statistics
**C4 Version (English - Standard):**
- **Dataset Size:** 806.87 GiB
- **Samples (Documents):** 364,613,570 (training split) + 364,724 (validation split)
- **TFDS Version:** 3.1.0 (most recent)

**Multilingual Version (mC4):**
- **Dataset Size:** 38.49 TiB
- **Languages:** 101

**Multimodal Version (mmc4 - 2023):**
- **mmc4 (Full):** 571 million images, 101.2 million documents, 43 billion tokens.
- **mmc4-ff (Fewer Faces):** 375 million images, 77.7 million documents, 33 billion tokens.

## Features
- **Rigorous Cleaning:** Application of filters to remove low-quality content, code, short sentences, and duplicates, resulting in a high-fidelity text corpus.
- **Based on Common Crawl:** Derived from a 2019 *snapshot* of Common Crawl, a massive source of web data.
- **Focus on English Text:** The original version is focused on English content.
- **Multilingual Version (mC4):** A configuration covering 101 languages and generated from 86 Common Crawl *dumps*.
- **Multimodal Extension (mmc4):** A more recent version (2023) that adds 571 million images aligned with the text, expanding the dataset's capabilities.

## Use Cases
- **Pre-training of Large-Scale Language Models (LLMs):** It was the foundational dataset for training the T5 model and is widely used as a *baseline* for other LLMs.
- **Text Generation:** Training models for high-quality text generation tasks.
- **Multimodal Models (with mmc4):** Training models that integrate text and image, such as *vision-language* models and *multimodal large language models* (MLLMs).
- **Natural Language Processing (NLP) Research:** Serves as a clean and massive corpus for various NLP research tasks.

## Integration
The C4 dataset is not provided directly for download by Google due to its size and bandwidth costs. Instead, the recommended integration method is **reproduction** from the raw Common Crawl data, using the open-source tools provided by the T5 project and TensorFlow Datasets (TFDS).

**Integration Method (TFDS):**
1.  **Installation:** Install `tfds-nightly` with the `c4` dependency: `pip install tfds-nightly[c4]`
2.  **Distributed Generation:** Due to the size (~7 TB of raw data) and processing time (~335 CPU-days), it is highly recommended to use a distributed processing service such as **Google Cloud Dataflow** with Apache Beam, following the detailed instructions in the T5 repository.
3.  **Access via Hugging Face:** Pre-processed and smaller versions (such as `allenai/c4` or `brando/small-c4-dataset`) are available on Hugging Face Datasets for lighter experimentation.

**Multimodal Integration (mmc4):**
The Multimodal C4 (mmc4-ff) version is available for direct download on Hugging Face Datasets:
- `jmhessel/mmc4-ff` (~218GB)
- `jmhessel/mmc4-core-ff` (~20GB)

## URL
[https://www.tensorflow.org/datasets/catalog/c4](https://www.tensorflow.org/datasets/catalog/c4)
