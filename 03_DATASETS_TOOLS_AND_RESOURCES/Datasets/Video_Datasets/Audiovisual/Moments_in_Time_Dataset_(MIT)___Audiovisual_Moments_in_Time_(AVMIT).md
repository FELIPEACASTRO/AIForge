# Moments in Time Dataset (MIT) / Audiovisual Moments in Time (AVMIT)

## Description
The **Moments in Time Dataset (MIT)** is a massive collection of videos, originally composed of one million 3-second clips, each labeled with one of 339 action classes, for the recognition and understanding of dynamic events. The most recent and relevant version is **Audiovisual Moments in Time (AVMIT)**, published in 2023, which focuses on audiovisual actions. AVMIT is an annotated subset of MIT, focusing on the correspondence between audio and visual streams, making it a valuable resource for training multimodal models.

## Statistics
**Moments in Time (MIT) Original:**
*   **Samples:** 1 million 3-second videos.
*   **Classes:** 339 action classes.

**Audiovisual Moments in Time (AVMIT - 2023):**
*   **Annotations:** 171,630 annotations across 57,177 audiovisual videos.
*   **Duration:** 23,160 videos (19.3 hours) of labeled audiovisual actions.
*   **Curated Test Set:** 960 videos (16 classes, 60 videos each).
*   **File Size (Embeddings):** 2.6 GB (total of the `.tar` files on Zenodo).
*   **Version:** 1 (Published on August 16, 2023).

## Features
**MIT Original:**
*   **Scale:** One million short videos (3 seconds).
*   **Classes:** 339 human-labeled action classes.
*   **Focus:** Capturing the essence of a dynamic event (action).
*   **Diversity:** Large inter-class and intra-class variation (e.g., "opening" doors, gifts, eyes).

**AVMIT (Audiovisual Moments in Time - 2023):**
*   **Nature:** Annotated audiovisual extension of MIT.
*   **Multimodal Analysis:** Focused on audiovisual correspondence (visual action and sound).
*   **Additional Resources:** Offers pre-computed feature *embeddings* (VGGish/YamNet for audio and VGG16/EfficientNetB0 for visual), facilitating input for research in audiovisual Deep Neural Networks (DNNs).
*   **Curated Test Set:** Includes a curated test set of 16 distinct action classes, with 60 videos each, suitable for controlled experiments.

## Use Cases
*   **Action Recognition in Videos:** Training models to identify dynamic events in short clips.
*   **Multimodal Learning:** Research on models that integrate visual and auditory information for a richer understanding of events (especially with AVMIT).
*   **Transfer Learning:** Using models pre-trained on MIT/AVMIT as *backbones* for computer vision and audio tasks in new domains.
*   **Audiovisual Correspondence Analysis:** Studying the relationship between what is seen and what is heard in real-world events.

## Integration
The original **Moments in Time** dataset requires filling out a request form on the official MIT-IBM Watson AI Lab website to obtain the download links.

The most recent version, **Audiovisual Moments in Time (AVMIT)**, is available on Zenodo and includes the pre-computed feature *embeddings*, along with a CSV file containing the test set metadata.

**Integration Steps (AVMIT):**
1.  **Download the Files:** Download the `.tar` files and the `.csv` from the Zenodo page.
    *   `AVMIT_VGGish_VGG16.tar` (498.7 MB)
    *   `AVMIT_YamNet_EffNetB0.tar` (2.1 GB)
    *   `test_set.csv` (test set metadata)
2.  **Access to Videos:** AVMIT uses videos from the original MIT. The `test_set.csv` file contains the `video_location` column that indicates the location of the original MIT videos that should be used.
3.  **Using the Embeddings:** The `.tar` files contain the feature *embeddings*, which can be loaded directly into machine learning models for audiovisual action recognition tasks.

**Note:** To obtain the raw MIT videos, it is necessary to follow the request process on the official website.

## URL
[http://moments.csail.mit.edu/](http://moments.csail.mit.edu/)
