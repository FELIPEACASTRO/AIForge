# Charades (Activity Recognition)

## Description
Charades is a large-scale dataset composed of 9,848 videos of daily indoor activities, collected via Amazon Mechanical Turk. Its main goal is to drive research in activity recognition in unstructured video and common-sense reasoning for daily human activities. The videos are short (average of 30 seconds) and show 267 different users acting out sentences that include objects and actions from a fixed vocabulary. The dataset is frequently used as a benchmark for action recognition and temporal localization tasks.

## Statistics
- **Videos:** 9,848 videos of daily indoor activities.
- **Temporal Annotations:** 66,500 annotations for 157 action classes.
- **Object Labels:** 41,104 labels for 46 object classes.
- **Textual Descriptions:** 27,847 textual descriptions.
- **Total Size:** The dataset in its original size is approximately 55 GB (videos), with the version scaled to 480p at 13 GB.
- **Versions:** The original version (v1) was released in 2016. Notable extensions include Charades-Ego and Charades-STA.

## Features
- **Daily Activity Videos:** 9,848 videos of everyday indoor activities.
- **Rich Annotations:** Includes 66,500 temporal annotations for 157 action classes, 41,104 labels for 46 object classes, and 27,847 textual descriptions.
- **Multi-label Nature:** Activities can occur simultaneously or sequentially, making it ideal for multi-label activity recognition.
- **Data Diversity:** Collected from 267 different users, ensuring a variety of scenarios and acting styles.
- **Variations:** Has extensions such as Charades-Ego (first- and third-person videos) and Charades-STA (for temporal localization of activities with sentences).

## Use Cases
- **Human Activity Recognition (HAR):** Main use case, focused on identifying human actions and activities in unstructured videos.
- **Temporal Action Localization:** Used to determine the exact start and end of an activity within a video.
- **Multi-label Activity Recognition:** Ideal for models that need to identify multiple actions occurring simultaneously or sequentially.
- **Common-Sense Reasoning:** Research on understanding how objects and actions relate to each other in everyday scenarios.
- **Egocentric Vision:** The Charades-Ego extension is used to train models that understand activities from a first-person perspective.
- **Video Caption Generation:** The textual descriptions and temporal annotations support the development of *video captioning* models.

## Integration
The Charades dataset can be accessed and downloaded directly from the official Allen AI page (prior.allenai.org/projects/charades) or through platforms such as Hugging Face Datasets.

**Download Options:**
1.  **Official Page (Allen AI):** Offers several download options, including:
    *   Data (scaled to 480p, 13 GB)
    *   Data (original size, 55 GB)
    *   RGB and Optical Flow Frames
    *   Annotations and Evaluation Code (3 MB)
2.  **Hugging Face Datasets:** Can be loaded directly into Python environments using the `datasets` library:
    ```python
    from datasets import load_dataset
    # For the Charades-STA version
    dataset = load_dataset("HuggingFaceM4/charades")
    ```
3.  **GitHub Repositories:** Starter code and baseline algorithms are available in repositories such as `gsig/charades-algorithms` to assist with integration into frameworks like PyTorch and Torch.

**Usage Instructions:**
After downloading, the annotations (in CSV format) and the videos must be processed. The evaluation code and baseline scripts provided by the authors are essential for setting up the environment and running activity recognition models. It is recommended to start with the scaled version (480p) for initial testing.

## URL
[https://prior.allenai.org/projects/charades](https://prior.allenai.org/projects/charades)
