# VizWiz-VQA (Visual Question Answering)

## Description
VizWiz-VQA is a unique Visual Question Answering (VQA) dataset, as it is the first to be built from real visual questions asked by blind and low-vision people. The images and questions were collected through a mobile application, where users took a photo and recorded a spoken question about it. The dataset is crucial for the development of assistive technologies, as it reflects the everyday visual challenges faced by this population. The most recent version (as of January 2023) replaced answers such as "unsuitable" with "unanswerable" for greater clarity and features a larger and cleaner dataset. The dataset supports VQA tasks, including predicting the correct answer and predicting the answerability of the question.

## Statistics
- **Updated Version (Jan/2023):**
  - **Training:** 20,523 image/question pairs and 205,230 answer/confidence pairs.
  - **Validation:** 4,319 image/question pairs and 43,190 answer/confidence pairs.
  - **Test:** 8,000 image/question pairs.
- **Total Samples:** 32,842 image/question pairs.
- **Versions:** The Jan/2023 version is the most recent, replacing the previous Dec/2019 version. The main change was the replacement of "unsuitable" with "unanswerable" and the expansion of the training and validation sets.

## Features
- **Real Origin:** Images and questions collected from blind and low-vision people in everyday situations.
- **Multimodal Format:** Combines images and natural-language questions (originally spoken).
- **Detailed Annotation:** Each visual question has 10 crowdsourced answers, allowing for a robust evaluation.
- **Answerability Challenge:** Includes the task of predicting whether a visual question can be answered, addressing image quality and question clarity.
- **Updated Version:** The 2023 version features an improved annotation scheme and a larger, cleaner dataset.

## Use Cases
- **Assistive Technologies:** Development of VQA systems that can help blind and low-vision people obtain information about the world around them.
- **VQA Research:** Training and evaluation of VQA models in a real-world data scenario, with low-quality images and complex questions.
- **Image Quality Analysis:** Study of the relationship between image quality (often poor due to the user's visual impairment) and the answerability of questions.
- **Visual Privacy:** The VizWiz-Priv dataset, a related subset, is used to recognize the presence and purpose of private visual information.

## Integration
The VizWiz-VQA dataset can be downloaded directly from the official website (vizwiz.org).
1. **Downloading the Files:** Download the image sets (training, validation, and test) and the JSON annotation files (training, validation, and test) through the links provided in the "Dataset" section of the VQA page.
2. **File Structure:** The JSON files contain the details of each visual question, including the image, the question, the answer type, and the 10 crowdsourced answers with their confidence levels.
3. **Sample Code:** The website provides sample code and APIs to demonstrate how to parse the JSON files and evaluate methods against the ground truth.
4. **Submission for Challenges:** To participate in the challenges, results must be submitted to the EvalAI evaluation server, following the specific instructions for the `test-dev`, `test-challenge`, and `test-standard` partitions.

## URL
[https://vizwiz.org/tasks-and-datasets/vqa/](https://vizwiz.org/tasks-and-datasets/vqa/)
