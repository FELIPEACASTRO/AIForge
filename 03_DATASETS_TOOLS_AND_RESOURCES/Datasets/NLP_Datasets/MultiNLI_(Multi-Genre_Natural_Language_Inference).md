# MultiNLI (Multi-Genre Natural Language Inference)

## Description
The **Multi-Genre Natural Language Inference (MultiNLI)** is a large-scale, crowdsourced corpus composed of 433 thousand sentence pairs annotated with textual inference (entailment) information. It was modeled after the SNLI corpus but differs in that it spans a variety of **ten genres** of spoken and written text (such as Fiction, Letters, Telephone Speech, 9/11 Report, etc.). The main goal of MultiNLI is to enable a distinct and more robust generalization evaluation, known as **cross-genre** evaluation, where models are tested on a genre different from the one they were trained on. The corpus is a fundamental resource for the development and evaluation of natural language understanding (NLU) models and served as the basis for the shared task of the RepEval 2017 Workshop [1] [2].

## Statistics
- **Dataset Size**: 433,000 (433k) annotated sentence pairs.
- **File Size**: 227 MB (ZIP).
- **Main Version**: 1.0 (Version 0.9 also exists, but differs only in the `pairID` and `promptID` fields).
- **Split (Approximate)**:
    - Training: ~392.7k pairs
    - Development (Matched): ~9.8k pairs
    - Development (Mismatched): ~9.8k pairs
    - Test (Matched): ~9.8k pairs (available via Kaggle/GLUE)
    - Test (Mismatched): ~9.8k pairs (available via Kaggle/GLUE)

## Features
- **Multi-Genre**: Includes 10 distinct text genres, which makes it more challenging and representative of real language than SNLI.
- **Textual Inference**: Each sentence pair (Premise and Hypothesis) is labeled with one of three relations: **entailment**, **contradiction**, or **neutral**.
- **Cross-Genre Evaluation**: The development and test sets are divided into two parts: *Matched* (the same genre as the training set) and *Mismatched* (genres not seen during training), allowing an evaluation of the model's generalization capability.
- **Format**: Distributed in ZIP files containing the data in JSON Lines (.jsonl) and tab-separated text (.txt) formats.
- **License**: The license is detailed in the data description paper [1].

## Use Cases
- **Training NLU Models**: Primarily to train and fine-tune Natural Language Inference (NLI) models, such as BERT, RoBERTa, and LLMs.
- **Generalization Evaluation**: Used to test a model's ability to generalize its understanding of textual inference to new domains (genres) of text.
- **NLU Research**: Serves as a fundamental benchmark for research in Natural Language Understanding, especially in reasoning and inference tasks.
- **Transfer Learning**: Used as a pre-training or *fine-tuning* dataset for tasks related to semantics and textual reasoning.

## Integration
The MultiNLI dataset (version 1.0) can be downloaded directly from the official NYU website.
1. **Download**: Download the ZIP file (227MB) through the link provided in the URL section.
2. **Extraction**: The file contains the data in `.jsonl` and `.txt` formats.
3. **Usage**: For most modern applications, the dataset is easily accessible through libraries such as **Hugging Face Datasets** or **TensorFlow Datasets (TFDS)**, which handle the download, splitting, and pre-processing automatically.

**Integration Example (Hugging Face Datasets):**
```python
from datasets import load_dataset

# Load the MultiNLI dataset
# The 'mismatch' is the most challenging part for generalization evaluation
dataset = load_dataset("multi_nli", split="validation_mismatched")

# Display an example
print(dataset[0])
```

## URL
[https://cims.nyu.edu/~sbowman/multinli/](https://cims.nyu.edu/~sbowman/multinli/)
