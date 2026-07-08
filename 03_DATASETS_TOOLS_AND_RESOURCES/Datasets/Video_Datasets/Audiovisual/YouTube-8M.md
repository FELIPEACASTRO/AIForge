# YouTube-8M

## Description
**YouTube-8M** is a large-scale, labeled, and diverse video *dataset*, developed by Google Research to accelerate research in large-scale video understanding, representation learning, noisy-data modeling, transfer learning, and domain adaptation. The *dataset* consists of millions of YouTube video IDs, with high-quality machine-generated annotations from a vocabulary of more than 3,800 visual entities. It is notable for providing pre-computed audiovisual *features*, allowing researchers to train robust baseline models in less than a day using a single GPU. The most recent version (June 2019) includes **YouTube-8M Segments**, an extension with about 237K human-verified labeled segments across 1,000 classes, focusing on the temporal localization of entities in 5-second videos. The *dataset* is made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Statistics
**June 2019 Version (Segments - Current):**
- **Samples:** 230K human-verified labeled segments.
- **Classes:** 1,000 classes.
- **Format:** Frame-level features.

**May 2018 Version (Video - Current):**
- **Videos:** 6.1 Million video IDs.
- **Duration:** 350,000 hours of video.
- **Features:** 2.6 Billion audiovisual *features*.
- **Classes:** 3,862 classes.
- **Total Size (Frame-level features):** 1.53 Terabytes.
- **Total Size (Video-level features):** 31 Gigabytes.
- **Labels:** Average of 3.0 labels per video.

## Features
- **Massive Scale:** Millions of videos and billions of pre-computed audiovisual *features*.
- **Diversity:** Videos sampled uniformly to preserve the distribution of popular YouTube content.
- **Pre-computed Features:** Provides video-level *features* (average of RGB and audio *features*) and frame-level *features* (RGB and audio every second).
- **Multi-label Labels:** Each video has multiple labels associated with Knowledge Graph entities.
- **Human-Verified Segments:** The Segments version adds human-verified annotations for temporal localization of entities.
- **Easy Access:** The data is provided as TensorFlow Record files, optimized for large-scale training.

## Use Cases
- **Large-Scale Video Classification:** Training and evaluating models to categorize videos with multiple labels.
- **Temporal Event Localization:** Using the Segments version to identify the exact moment when an entity or event occurs in the video.
- **Representation Learning:** Developing new neural network architectures to extract video and audio *features*.
- **Noisy-Data Modeling:** Research on how to handle machine-generated annotations that may contain noise.
- **Transfer Learning and Domain Adaptation:** Using the *dataset* as a foundation to transfer knowledge to more specific video tasks.

## Integration
The *dataset* is distributed as **TensorFlow Record** files and can be downloaded using a Python *script* provided by Google Research.
1.  **Installation:** Make sure you have Python and `curl` installed.
2.  **Download the Script:** The download *script* (`download.py`) is accessed via `curl` and executed with Python.
3.  **Directory Structure:** Create a directory for the data, for example: `mkdir -p ~/data/yt8m/video; cd ~/data/yt8m/video`.
4.  **Download (Example for *Video-level features*):**
    ```bash
    curl data.yt8m.org/download.py | partition=2/video/train mirror=us python
    curl data.yt8m.org/download.py | partition=2/video/validate mirror=us python
    curl data.yt8m.org/download.py | partition=2/video/test mirror=us python
    ```
    - **Mirror:** Replace `mirror=us` with `mirror=eu` or `mirror=asia` to speed up the transfer depending on your location.
    - **Subsampling:** It is possible to download a fraction of the *dataset* using the `shard=1,100` parameter for 1/100 of the data.
5.  **Starter Code:** Google Research provides a GitHub repository with starter code for training and evaluating models.
    - **Frame-level features:** Require about 1.53 TB of disk space.
    - **Video-level features:** Require about 31 GB of disk space.

## URL
[https://research.google.com/youtube8m/](https://research.google.com/youtube8m/)
