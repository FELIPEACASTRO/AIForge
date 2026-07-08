# ActivityNet

## Description
ActivityNet is a large-scale video benchmark designed for understanding human activities. Its goal is to cover a wide range of complex activities of interest in daily life. The dataset is used to compare algorithms on tasks such as global video classification, trimmed activity classification, and activity detection. The main version is ActivityNet 1.3, which includes 200 activity classes. The dataset is notable for its semantic ontology structure, which organizes activities according to social relationships and locations of occurrence, providing a rich hierarchy with at least four levels of depth. The original dataset was presented in 2015, but it remains a fundamental reference and is frequently used in recent challenges and research (2023-2025) through its extensions such as ActivityNet-QA and ActivityNet-Entities.

## Statistics
- **Main Version:** ActivityNet 1.3 (released in 2016, but the basis for recent research).
- **Activity Classes:** 200 classes.
- **Total Videos:** Approximately 20,000 videos (10,024 training, 4,926 validation, 5,044 test).
- **Activity Instances:** 15,410 activity instances in the training set.
- **Total Duration:** 849 hours of video.
- **Average:** 1.54 activity instances per video.

## Features
- **Scale and Diversity:** 200 activity classes, 20,000 videos in total (training, validation, and test), and 849 hours of video.
- **Semantic Ontology:** Hierarchical structure of activities with four levels of depth, allowing the study of relationships between activities.
- **Detailed Annotations:** Provides annotations of activity instances with time segments (start and end) in untrimmed videos.
- **Extensions:** Served as a basis for more recent datasets such as ActivityNet-QA (Video Question Answering) and ActivityNet-Entities (Entity bounding box annotations).
- **Integration with FiftyOne:** Native support for loading, visualization, and evaluation through the open-source FiftyOne tool.

## Use Cases
- **Human Activity Detection and Recognition:** Mainly in untrimmed videos.
- **Video Classification:** Global video classification.
- **Temporal Action Localization:** Precise identification of the time segments where activities occur.
- **Computer Vision Research:** Benchmark for the development and comparison of new video understanding algorithms.
- **Visual Question Answering (VQA):** Used as a basis for the ActivityNet-QA dataset.
- **Entity Grounding:** Used as a basis for the ActivityNet-Entities dataset, which adds bounding box annotations for entities mentioned in the captions.

## Integration
The ActivityNet dataset (versions 100 and 200) can be easily loaded, visualized, and evaluated using Voxel51's open-source **FiftyOne** tool.

**Usage Instructions with FiftyOne:**
1.  **Installation:** Install FiftyOne via pip: `pip install fiftyone`
2.  **Loading:** Use `fiftyone.zoo` to load the dataset, specifying the desired version and split. For example, for version 200 (ActivityNet 1.3):
    ```python
    import fiftyone.zoo as foz
    dataset = foz.load_zoo_dataset("activitynet-200", split="validation")
    ```
3.  **Video Download:** To obtain the complete videos, it is necessary to fill out a request form on the official website to gain temporary access to the files hosted on Google Drive or Baidu Drive.

**Annotation Structure:**
The annotations are provided in JSON files containing the database, the hierarchical taxonomy, and the version. The "database" key contains video information (duration, URL, subset) and the "annotations" key lists the activity instances with `label` and `segment` (start and end time in seconds).

## URL
[http://activity-net.org/](http://activity-net.org/)