# DAVIS (Densely Annotated VIdeo Segmentation)

## Description
DAVIS (Densely Annotated VIdeo Segmentation) is a fundamental dataset and *benchmark* for the problem of **Video Object Segmentation (VOS)**. The most widely used version, DAVIS 2017, expanded the original version (DAVIS 2016) to include multi-object annotations per sequence. The dataset is crucial for the development and evaluation of VOS algorithms, both in semi-supervised scenarios (where the object mask is provided in the first *frame*) and unsupervised scenarios (where no human *input* is provided). The dataset is known for the high quality of its annotations and the diversity of its video scenarios.

## Statistics
- **DAVIS 2017:** 150 video sequences, totaling 10,459 annotated *frames* and 376 object instances.
- **DAVIS 2016:** 50 video sequences, totaling 3,455 annotated *frames*.
- **Dataset Size (DAVIS 2017):** Approximately 792.26 MiB (TFDS 480p version) or several GBs for the Full-Resolution version.
- **Resolution:** Official annotations in 480p, but full-resolution images (up to 4K) are available.
- **Versions:** DAVIS 2016, DAVIS 2017, and extensions such as DAVIS-17 Moving. The most recent challenge was in 2020.

## Features
- **Dense Analysis:** High-quality pixel-level segmentation annotations for each relevant *frame*.
- **Multiple Versions:** Includes DAVIS 2016 (single object) and DAVIS 2017 (multiple objects).
- **Varied Resolutions:** Available in 480p for standard evaluation and in full resolution (Full-Resolution, up to 4K) for research.
- **Evaluation Modes:** Supports evaluation for semi-supervised VOS (with a first-*frame* mask) and unsupervised VOS (without human *input*).
- **Annual Challenges:** It was the basis for annual challenges (DAVIS Challenge) from 2017 to 2020, driving the state of the art.

## Use Cases
- **Video Object Segmentation (VOS):** The primary use case for training and evaluating VOS models.
- **Multiple Object Tracking (MOT):** DAVIS 2017, with multi-object annotations, is relevant for tracking tasks.
- **Real-Time Computer Vision:** Development of efficient algorithms that operate on video sequences.
- **Computer Vision Research:** Used as a standard *benchmark* for the state of the art in video segmentation.
- **Applications:** Robotics, autonomous vehicles, surveillance, and advanced video editing.

## Integration
The dataset and the evaluation code are available on the official website. Usage generally involves:
1.  **Download:** Downloading the `TrainVal` and `Test-Dev/Test-Challenge` files (images and annotations) from the official website.
2.  **Evaluation Code:** Using the provided Python or MATLAB repositories to load the dataset and evaluate the model results.
3.  **Structure:** The dataset is organized into video sequences, with subfolders for images (*JPEGImages*) and annotations (*Annotations*).
4.  **Use in Frameworks:** The dataset is also available in ready-to-use formats for libraries such as TensorFlow Datasets (TFDS).
**Download Link:** The direct link to the DAVIS 2017 downloads section is `https://davischallenge.org/davis2017/code.html`. Acceptance of the terms of use is required.

## URL
[https://www.davischallenge.org/](https://www.davischallenge.org/)
