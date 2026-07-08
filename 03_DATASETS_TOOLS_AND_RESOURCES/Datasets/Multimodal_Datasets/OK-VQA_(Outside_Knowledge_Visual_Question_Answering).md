# OK-VQA (Outside Knowledge Visual Question Answering)

## Description
OK-VQA (Outside Knowledge Visual Question Answering) is a Visual Question Answering (VQA) dataset that requires models to use **external knowledge** (beyond the content of the image) to answer the questions. Unlike traditional VQA datasets, where the answers can be inferred from the image alone, OK-VQA contains more than 14,000 open-ended questions that were manually filtered to ensure the need for external knowledge (for example, from Wikipedia). The dataset was designed to drive research on models that can integrate visual and textual information from external knowledge sources. An improved version, **OK-VQA v2.0**, was released in 2023, correcting and removing 41.4% and 10.6% of the original dataset, respectively, to improve quality. The dataset is often used together with A-OKVQA (Augmented OK-VQA), which provides rationales for the answers.

## Statistics
- **Samples:** 14,055 open-ended questions.
- **Answers:** 5 ground truth reference answers per question.
- **Versions:**
    - **v1.0 (2019):** Original version.
    - **v1.1 (2020):** Update with improvements to the *word stemming* method for answers.
    - **v2.0 (2023):** Improved version with corrections and removals for higher quality.
- **Size:** The size of the annotation files (JSON) is small (a few MB). The full dataset, including the COCO images, is significantly larger (the COCO 2014 images are ~25 GB). The version on Hugging Face (annotations only) is about **832 MB** (lmms-lab/OK-VQA).

## Features
- **External Knowledge Requirement:** The main characteristic is the need for knowledge beyond the image to answer the questions.
- **Open-Ended Questions:** More than 14,000 questions that require free-form answers.
- **Multiple Reference Answers:** Each question has 5 ground truth reference answers collected by humans.
- **Knowledge Categories:** The questions are categorized into 10 types of external knowledge, such as People and Everyday Life, Science and Technology, History, Geography, Sports and Recreation, etc.
- **Based on COCO:** The images come from the COCO (Common Objects in Context) dataset.

## Use Cases
- **VQA Model Evaluation:** Serves as a challenging benchmark for Visual Question Answering models that require knowledge-based reasoning.
- **External Knowledge Research:** Development of methods to integrate external knowledge bases (such as Wikipedia) into Computer Vision and Natural Language Processing models.
- **Multimodal Reasoning:** Training models to perform complex reasoning that combines visual information and real-world facts.
- **Knowledge Transfer:** Study of the ability of models to transfer knowledge from large language models to VQA tasks.

## Integration
The OK-VQA dataset can be downloaded directly from the official Allen AI website, where the annotation files (questions and answers) are provided in JSON format. The corresponding images come from the COCO dataset.

**Integration Steps:**
1.  **Download Annotations:** Download the training and test annotation JSON files (v1.1 or v2.0, if available) from the official website.
2.  **Obtain Images:** The images are not included in the annotation download and must be obtained from the **COCO 2014** dataset (training and validation sets).
3.  **Processing:** Use the JSON annotations to map the questions and answers to the corresponding COCO images.
4.  **Alternative (Hugging Face):** The dataset is also available on platforms such as Hugging Face, where it can be loaded directly using Python's `datasets` library:
    ```python
    from datasets import load_dataset
    # For the Hugging Face version (may not include the COCO images)
    dataset = load_dataset("lmms-lab/OK-VQA")
    ```

## URL
[https://okvqa.allenai.org/](https://okvqa.allenai.org/)
