# Something-Something V2 (Temporal Reasoning)

## Description
The **Something-Something (version 2)** dataset is a collection of **220,847 labeled video clips** of humans performing basic, predefined actions with everyday objects. It was specifically designed to train machine learning models in a detailed understanding of human gestures and interactions, requiring **temporal reasoning** to distinguish between similar actions, such as "lifting something up completely" versus "lifting something up and then letting it drop down". The dataset is crucial for advancing research in **video action recognition** and **visual common sense understanding**.

## Statistics
- **Total Size:** 19.4 GB (videos).
- **Samples (Videos):** 220,847 video clips.
- **Action Categories:** 174 classes.
- **Crowdsourcing Actors:** More than 1,300 unique.
- **Version:** V2 (Version 2), released in 2018, being the most used for benchmarks.
- **Resolution:** 240px in height.

## Features
- **Focus on Temporal Reasoning:** Actions are defined by template phrases (e.g., "Putting [something] onto [something]"), where the order and temporal relationship between objects and actions are crucial.
- **Large Scale:** Contains 220,847 short, trimmed videos.
- **Action Diversity:** Covers 174 distinct action categories.
- **Detailed Annotations:** Includes object annotations (318,572 annotations, 30,408 unique objects) for the training and validation sets.
- **High Quality:** Each video was verified by five different crowdsourcing actors to ensure label accuracy.
- **Format:** Videos in WebM format (VP9 codec) with a height of 240px.

## Use Cases
- **Video Action Recognition:** It is the primary benchmark for models that seek to classify actions in videos, especially those that depend on context and temporal order.
- **Temporal Reasoning:** Training and evaluation of neural network models (such as the Temporal Relation Network - TRN) capable of learning and reasoning about temporal relationships between video frames.
- **Computer Vision and Robotics:** Development of systems that require a fine understanding of object manipulations and human gestures for imitation or interaction tasks.
- **Video Language Models (Video LLMs):** Used in recent research (2024-2025) to enhance temporal reasoning capabilities and long video understanding in multimodal models.

## Integration
The dataset can be accessed through the official Qualcomm (TwentyBN) website, where the videos are provided in a large TGZ file, split into 1 GB parts (total size of 19.4 GB). The annotations (labels and splits) are provided in separate JSON files.

**Usage Instructions (Example with Hugging Face):**
For research and development use, version V2 is available on Hugging Face Datasets, facilitating integration with machine learning libraries:

```python
from datasets import load_dataset

# Loads the dataset (metadata only, videos need to be downloaded separately)
dataset = load_dataset("HuggingFaceM4/something_something_v2")

# To prepare the dataset for action recognition models,
# it is common to follow the instructions of libraries such as MMAction2.
# Video download must be done from the primary source.
```

## URL
[https://www.qualcomm.com/developer/software/something-something-v-2-dataset](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)