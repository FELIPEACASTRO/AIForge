# GLUE Benchmark (General Language Understanding Evaluation)

## Description
The **General Language Understanding Evaluation (GLUE) Benchmark** is a collection of resources for training, evaluating, and analyzing natural language understanding (NLU) systems. GLUE is composed of a *benchmark* of nine NLU tasks based on sentences or sentence pairs, selected to cover a diverse range of *dataset* sizes, text genres, and degrees of difficulty. In addition, it includes a diagnostic *dataset* for detailed analysis of model performance across a wide range of linguistic phenomena. The main goal of GLUE is to drive research in the development of general and robust NLU systems, favoring models that share information across tasks using *transfer learning* techniques. Although the original *benchmark* is from 2018, it remains a fundamental reference, and subsequent *benchmarks* such as SuperGLUE (2019) and variations such as Adversarial GLUE (2021) and PrivacyGLUE (2023) demonstrate its continued relevance and the evolution of research in the field.

## Statistics
GLUE is a collection of smaller *datasets*, totaling hundreds of thousands of training samples. The training statistics for the nine original tasks are:
- **CoLA (Corpus of Linguistic Acceptability):** 8.5k training samples.
- **SST-2 (Stanford Sentiment Treebank):** 67k training samples.
- **MRPC (Microsoft Research Paraphrase Corpus):** 3.7k training samples.
- **STS-B (Semantic Textual Similarity Benchmark):** 7k training samples.
- **QQP (Quora Question Pairs):** 364k training samples.
- **MNLI (Multi-Genre Natural Language Inference):** 393k training samples.
- **QNLI (Question-answering NLI):** 105k training samples.
- **RTE (Recognizing Textual Entailment):** 2.5k training samples.
- **WNLI (Winograd NLI):** 634 training samples.

**Version:** The original *benchmark* was introduced in 2018. Variations and successors include **SuperGLUE** (2019) and **PrivacyGLUE** (2023).

## Features
- **Set of 9 NLU Tasks:** Includes single-sentence classification tasks (CoLA, SST-2), similarity and paraphrase (MRPC, STS-B, QQP), and textual inference (MNLI, QNLI, RTE, WNLI).
- **Linguistic Diagnostics:** An auxiliary *dataset* designed to evaluate the understanding of specific linguistic phenomena.
- **Model-Agnostic Format:** Any system capable of processing sentences and sentence pairs is eligible.
- **Focus on Transfer Learning:** The tasks are selected to favor models that use parameter-sharing or *transfer learning* techniques across tasks.

## Use Cases
- **NLU Model Evaluation:** Primarily used to measure and compare the performance of natural language understanding models, such as BERT, RoBERTa, and T5, across a diverse set of tasks.
- **Transfer Learning and Pre-training:** Used to train models on multiple tasks simultaneously (*multi-task learning*) or for *fine-tuning* pre-trained models on specific tasks.
- **Linguistic Analysis:** The diagnostic *dataset* enables a more in-depth analysis of the capabilities and shortcomings of models with respect to specific linguistic phenomena.
- **Development of General NLU Systems:** Serves as an engine to drive research toward more robust and generalizable NLU systems.

## Integration
The GLUE *dataset* can be accessed and downloaded through *scripts* provided by the community and by Hugging Face Datasets. The most common method is to use the Hugging Face `datasets` library or third-party *scripts*, such as `download_glue_data.py` (although the original link may be outdated, the functionality is maintained in projects such as `jiant`).

**Usage example with Hugging Face Datasets (Python):**
```python
from datasets import load_dataset

# Load the GLUE dataset (example: CoLA)
dataset = load_dataset("glue", "cola")

# The dataset is loaded as a DatasetDict object
print(dataset)
```
The *benchmark* is also integrated into platforms such as TensorFlow Datasets (TFDS).

## URL
[https://gluebenchmark.com/](https://gluebenchmark.com/)
