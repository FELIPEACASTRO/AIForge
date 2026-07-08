# GQA (Scene Graph Question Answering)

## Description
GQA (Scene Graph Question Answering) is an innovative dataset for **Real-World Visual Reasoning and Compositional Question Answering (VQA)**. It was created to overcome the limitations of earlier VQA datasets, which were susceptible to language biases and a lack of semantic compositionality. GQA uses detailed **Scene Graphs** to represent objects, attributes, and relationships in images, and **Functional Programs** to structure the reasoning logic of the questions. This enables a more precise diagnosis of model performance and encourages the development of more robust and interpretable VQA systems.

## Statistics
- **Images:** 113K images.
- **Questions:** More than 22 million diverse reasoning questions.
- **Versions and Download Sizes (Main Data):**
    - Scene Graphs (ver 1.1): 42.7MB
    - Questions (ver 1.2): 1.4GB
    - Images (ver 1.1): 73.9GB (Total) or 20.3GB (Image Files)
- **Balanced Version:** 1.7M questions.

## Features
- **Compositional Reasoning:** The questions require multiple reasoning skills, spatial understanding, and multi-step inference.
- **Scene Graphs:** Each image is associated with a detailed scene graph (objects, attributes, and relationships), based on Visual Genome but refined.
- **Functional Programs:** Each question is associated with a structured representation of its semantics, a functional program that specifies the reasoning steps needed to answer it.
- **Improved Metrics:** Includes new metrics to test the consistency, validity, and plausibility of model answers, in addition to accuracy.
- **Balanced Dataset:** A balanced version of 1.7M questions was created to mitigate language biases.

## Use Cases
- Development and evaluation of **Visual Reasoning** and **Compositional VQA** models.
- Research in **Scene Understanding** and **Model Interpretability** (thanks to the functional programs and scene graphs).
- Training models to be more robust to language and conditional biases.

## Integration
The dataset can be downloaded directly from the official Stanford page (see the main URL). The main components are:
1. **Scene Graphs:** `scene_graphs.json` file (ver 1.1 / 42.7MB).
2. **Questions:** `questions.json` file (ver 1.2 / 1.4GB).
3. **Images:** `images.zip` file (ver 1.1 / 20.3GB for image files).
The download page also offers Spatial Features (32.1GB) and Object Features (21.4GB) separately. You must agree to the terms of use to perform the download.

## URL
[https://cs.stanford.edu/people/dorarad/gqa/](https://cs.stanford.edu/people/dorarad/gqa/)
