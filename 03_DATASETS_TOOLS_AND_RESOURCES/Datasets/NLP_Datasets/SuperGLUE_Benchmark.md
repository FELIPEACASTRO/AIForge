# SuperGLUE Benchmark

## Description
**SuperGLUE (Super General Language Understanding Evaluation)** is a Natural Language Processing (NLP) model evaluation benchmark designed to be more challenging than its predecessor, GLUE. It was created to measure progress in general-purpose language understanding systems, focusing on tasks that require deeper reasoning, inference, and contextual understanding. The benchmark is composed of a set of more difficult language understanding tasks, improved resources, and a public leaderboard. The goal is to provide a single metric that summarizes progress across a diverse set of NLP tasks, especially after models reached human performance on the original GLUE benchmark.

## Statistics
SuperGLUE is a benchmark composed of 10 tasks (8 main and 2 diagnostic), each with its own dataset.
*   **Total Download Size (TFDS):** Approximately 733.32 KiB.
*   **Total Dataset Size (TFDS):** Approximately 2.15 MiB.
*   **Version:** The final version of the benchmark was released in 2019, but it remains a relevant evaluation standard, with models being continuously submitted and evaluated.
*   **Samples:** The number of samples varies by task. For example, BoolQ has more than 15,000 training and development examples. The complete set of tasks totals tens of thousands of examples.

## Features
SuperGLUE is composed of 8 main language understanding tasks and 2 diagnostic tasks for error analysis. The main tasks include:
*   **BoolQ:** Answering yes/no questions based on a paragraph.
*   **CB (CommitmentBank):** Determining the textual inference relationship (entailment, contradiction, neutral) between two sentences.
*   **COPA (Choice of Plausible Alternatives):** Choosing the most plausible alternative for a given scenario (cause or effect).
*   **MultiRC (Multi-Sentence Reading Comprehension):** Answering questions about a text, where the answer can be one or more sentences.
*   **ReCoRD (Reading Comprehension with Commonsense Reasoning):** Filling in blanks in a text based on commonsense reasoning.
*   **RTE (Recognizing Textual Entailment):** Determining whether one sentence logically entails another.
*   **WiC (Words in Context):** Determining whether a word appears with the same meaning in two different sentences.
*   **WSC (Winograd Schema Challenge):** Resolving pronominal reference ambiguities that require commonsense reasoning.

The diagnostic tasks are AX-b (Broadcoverage Diagnostics) and AX-g (WinoGender Schema Diagnostics). The benchmark is characterized by requiring models that demonstrate more robust inference and reasoning capabilities.

## Use Cases
*   **Language Model Evaluation:** This is the main use case, serving as a rigorous test for general-purpose language models (LLMs) and pre-trained models (such as BERT, RoBERTa, T5, etc.).
*   **NLP Research:** Used by researchers to develop and test new architectures and transfer learning techniques on more complex language understanding tasks.
*   **Error Analysis:** The diagnostic tasks (AX-b and AX-g) are used to perform qualitative and error analyses, helping to understand model shortcomings.
*   **Performance Comparison:** Serves as a public leaderboard to compare the performance of different NLP systems on a standardized set of tasks.

## Integration
The SuperGLUE dataset can be accessed and used in several ways:
1.  **Official Page:** The complete dataset can be downloaded directly from the SuperGLUE tasks page (URL: `https://super.gluebenchmark.com/tasks`).
2.  **Download Script:** The official site provides a download script (part of the `jiant` toolkit) to obtain the data.
3.  **Hugging Face Datasets:** For use in modern NLP projects, the dataset is available on the Hugging Face Hub (e.g., `Hyukkyu/superglue` or `aps/super_glue`), allowing easy loading via Python code:
    ```python
    from datasets import load_dataset
    # Para carregar uma tarefa específica, como BoolQ
    dataset = load_dataset("super_glue", "boolq")
    ```
4.  **TensorFlow Datasets:** The dataset is also available in the TensorFlow Datasets catalog.

Integration is facilitated by widely used NLP tools and libraries. Using the versions hosted on Hugging Face or TensorFlow is recommended for greater convenience.

## URL
[https://super.gluebenchmark.com/](https://super.gluebenchmark.com/)
