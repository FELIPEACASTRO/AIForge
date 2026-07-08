# TextVQA

## Description
**TextVQA** is a Visual Question Answering (VQA) dataset that requires models to read and reason about the text present in images in order to answer questions about them. It was created to address the limitation of earlier VQA datasets, which had a small proportion of questions that required reading text. The dataset is essential for developing models that incorporate the scene text modality to answer questions. The goal is to stimulate progress in VQA models that can "read" the text embedded in images.

## Statistics
**Main Version:** v0.5.1 (and v0.5, which is identical except for the OCR tokens).
**Total Images:** 28,408 images (from OpenImages).
**Total Questions:** 45,336 questions.
**Total Answers:** 453,360 ground truth answers (10 per question).
**Dataset Split (v0.5.1):**
*   **Train:** 34,602 questions, 21,953 images (6.6GB).
*   **Validation:** 5,000 questions, 3,166 images.
*   **Test:** 5,734 questions, 3,289 images (926MB).
**Size:** The question/answer dataset is relatively small (about 132MB in total), but the image set is approximately 7.5GB.

## Features
**Multimodal Nature:** Combines computer vision and Natural Language Processing (NLP), with a focus on scene text. **Reading Requirement:** The questions are formulated so that the answer can only be determined by reading and understanding the text visible in the image. **Integrated OCR:** Provides OCR (Optical Character Recognition) tokens extracted by the Rosetta system, in addition to the question and answer annotations. **Image Base:** The images come from the OpenImages dataset. **Evaluation:** Evaluation is performed through the EvalAI server, using the accuracy metric.

## Use Cases
**Visual Question Answering (VQA) with Scene Text:** Training and evaluation of VQA models that need to extract and understand textual information in images. **Optical Character Recognition (OCR) in Context:** Development of more robust OCR systems that operate in real-world scenarios and integrate visual context. **Vision and Language Models (VLMs):** Benchmarking of multimodal models that integrate text, image, and scene text for complex reasoning. **Assistance for People with Visual Impairment:** The original study points out that a dominant class of questions asked by visually impaired users involves reading text in images of their surroundings.

## Integration
The TextVQA dataset is available for download on the official website and on Hugging Face. The recommended version is **v0.5.1**. The data is split into JSON files for questions/answers and ZIP files for the images and OCR tokens.
1.  **Download:** The question/answer files and the OCR tokens (Rosetta OCR tokens \[v0.2\]) are available in JSON format. The images (from OpenImages) are provided in separate ZIP files for the train and test sets.
2.  **Structure:** The JSON files contain the `question_id`, the `question`, the `image_id` (from OpenImages), and up to 10 `answers` (ground truth answers).
3.  **Usage:** Researchers are encouraged to use their own OCR systems, although the provided OCR tokens are useful for the baseline. Results are submitted for evaluation through the EvalAI server.
4.  **License:** The dataset is available under the **CC BY 4.0** license.

## URL
[https://textvqa.org/](https://textvqa.org/)
