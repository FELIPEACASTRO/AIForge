# BookCorpus

## Description
**BookCorpus** is a massive text corpus, originally composed of about 11,038 self-published books extracted from the independent e-book distribution platform Smashwords. It was created for training influential language models such as **BERT** and its derivatives, and is notable for providing long-form text, which is crucial for learning long-range dependencies in Natural Language Processing (NLP) models. A 2021 retrospective analysis highlighted that the dataset has "documentation debt," contains copyright restrictions, includes thousands of duplicate books, and presents a significant bias in genre representation (for example, Romance and Fantasy are over-represented). The cleaner and most widely used version currently is **BookCorpusOpen**, which attempts to mitigate these shortcomings.

## Statistics
- **Number of Books (2021 Analyzed Version):** 11,038 (with only 7,185 unique books).
- **Number of Sentences:** 74,004,228
- **Number of Words:** 984,346,357
- **Download Size (Hugging Face Version):** 4.61 GB (dataset files) / 3.02 GB (converted Parquet files).
- **Notable Versions:**
    - **Original BookCorpus (2015):** Initial version used to train BERT.
    - **BookCorpusOpen:** A cleaner and more popular version that aims to resolve duplication and copyright issues.
    - **Repository Versions (e.g., Hugging Face):** Re-hosted and pre-processed variants.

## Features
- **Long-Form Text:** Ideal for training language models on understanding context and long-range dependencies.
- **Genre Diversity:** Although biased, it includes a variety of fiction genres, such as Romance, Fantasy, and Science Fiction.
- **Large Scale:** One of the first large-scale datasets for pre-training NLP models.
- **Foundation for Foundational Models:** Served as a foundation for the development of models such as BERT, RoBERTa, and GPT-N.

## Use Cases
- **Pre-training of Language Models:** Used to train large-scale NLP models such as BERT, RoBERTa, and the GPT series of models.
- **Learning Sentence Representations:** Ideal for unsupervised learning of sentence and paragraph encodings.
- **Language Generation Tasks:** The long-form text is useful for training models to generate coherent narratives and texts.
- **NLP Research:** Used as a benchmark and training corpus in various academic research on language understanding and generation.

## Integration
The most recommended and accessible way to use BookCorpus is through dataset repositories such as **Hugging Face**, which generally offer pre-processed and cleaner versions (such as BookCorpusOpen or refined variants).

**Usage Example with the Hugging Face `datasets` Library (`rojagtap/bookcorpus` Version):**

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("rojagtap/bookcorpus")

# Access the training split
train_data = ds["train"]

# Display the first example
print(train_data[0])
```

**Alternatives:**
- **Kaggle:** Various refined or partial versions of BookCorpus are available on Kaggle.
- **Unofficial Repositories:** Due to copyright issues and removal of the original source (Smashwords), the original version is no longer directly available, requiring you to seek re-hosted or derived versions.

## URL
[https://huggingface.co/datasets/rojagtap/bookcorpus](https://huggingface.co/datasets/rojagtap/bookcorpus)
