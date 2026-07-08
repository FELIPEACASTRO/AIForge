# SQuAD (Stanford Question Answering Dataset)

## Description
The **Stanford Question Answering Dataset (SQuAD)** is one of the most influential benchmarks for **Question Answering (QA)** systems and reading comprehension in Natural Language Processing (NLP). The dataset is composed of questions posed by human crowdworkers on a set of Wikipedia articles.

**Main Versions:**

1.  **SQuAD 1.1:** Contains more than **100,000 question-answer pairs** across more than 500 articles. The defining characteristic of this version is that the answer to each question is always a **text segment (span)** extracted directly from the corresponding reading passage (Extractive QA).
2.  **SQuAD 2.0:** A more challenging version that combines the questions from SQuAD 1.1 with more than **50,000 unanswerable questions** created adversarially by crowdworkers. To perform well on SQuAD 2.0, models must not only answer questions when possible, but also determine when the paragraph does not contain the answer and abstain from answering.

SQuAD is widely used to train and evaluate the ability of NLP models to read a text and extract the correct answer to a question.

## Statistics
**Versions:** SQuAD 1.1 and SQuAD 2.0.
**SQuAD 1.1:**
*   **Samples:** More than 100,000 question-answer pairs.
*   **Articles:** More than 500 Wikipedia articles.
**SQuAD 2.0:**
*   **Samples:** More than 150,000 questions in total.
*   **Unanswerable Questions:** More than 50,000.
*   **Size (v2.0):** Training (40 MB), Development (4 MB).

## Features
*   **Extractive QA Format:** Answers are text segments (spans) from the context.
*   **Base Context:** Wikipedia articles.
*   **SQuAD 2.0:** Includes more than 50,000 unanswerable questions, requiring models to determine the absence of an answer.
*   **Evaluation Metrics:** Primarily **Exact Match (EM)** and **F1 Score**.
*   **License:** Distributed under the **CC BY-SA 4.0** license.

## Use Cases
*   **Training Extractive QA Systems:** The main use for developing models that locate the exact answer in a text.
*   **Reading Comprehension Evaluation:** Serves as a standard benchmark for measuring machines' ability to understand and process text.
*   **Development of Robust Models (SQuAD 2.0):** Essential for training models that can distinguish answerable from unanswerable questions, crucial for real-world applications.
*   **NLP Research:** Used to test new language model architectures (such as BERT, T5, etc.) and transfer learning techniques.
*   **Virtual Assistants and Chatbots:** The technology developed with SQuAD is the foundation for systems that answer questions based on knowledge documents.

## Integration
SQuAD is easily accessible and can be integrated into NLP projects in several ways:

1.  **Hugging Face Datasets:** The most recommended and modern way to access the dataset, allowing direct loading with just a few lines of Python code:
    ```python
    from datasets import load_dataset
    # Para SQuAD 1.1
    squad_v1 = load_dataset("squad")
    # Para SQuAD 2.0
    squad_v2 = load_dataset("squad_v2")
    ```
2.  **Direct Download (JSON):** The original JSON files can be downloaded from the official site for manual use or in frameworks that do not support automatic loading.
    *   **SQuAD 2.0 Training:** `train-v2.0.json` (40 MB)
    *   **SQuAD 2.0 Development:** `dev-v2.0.json` (4 MB)
3.  **Kaggle:** The dataset is also available on Kaggle, making it easy to use in your notebooks.

**Usage Instructions:** The dataset is typically used to fine-tune pre-trained language models (such as BERT, RoBERTa, ELECTRA) for the Question Answering task. The process involves feeding the model with the context paragraph and the question, and the model must predict the start and end indices of the answer span in the text. For SQuAD 2.0, the model must also predict whether the question is "unanswerable".

## URL
[https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
