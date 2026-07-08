# AudioSet (Google)

## Description
AudioSet is a large-scale dataset composed of a collection of **2,084,320** 10-second audio clips, extracted from YouTube videos and labeled by humans. Its main goal is to provide a comprehensive resource for training audio event classification models. The dataset is based on a hierarchical **ontology** of **632 classes** of sound events, covering a wide range of everyday sounds, from human and animal sounds to natural, environmental, and musical sounds. The dataset does not provide the raw audio files directly, but rather the metadata (YouTube video IDs, start and end times) and 128-dimensional audio *embeddings* extracted at 1Hz using the VGGish model.

## Statistics
- **Total Size:** 2,084,320 audio segments.
- **Segment Duration:** 10 seconds.
- **Classes:** 632 sound event classes.
- **Split:**
    - Evaluation: 20,383 segments.
    - Balanced Training: 22,176 segments.
    - Unbalanced Training: 2,042,985 segments.
- **Size of *Embeddings*:** 2.4 GB (in TensorFlow Record format).
- **Version:** The main referenced version is "v1" (initial 2017 release), with label quality updates (rerating) included.

## Features
- **Extensive Ontology:** 632 sound event classes organized hierarchically.
- **Large Scale:** More than 2 million labeled audio segments.
- **Fixed Duration:** All audio clips are 10 seconds long.
- **Metadata and *Embeddings*:** Provides YouTube video IDs, time metadata, and audio *embeddings* (features) instead of the raw audio files.
- **Subset Split:** Divided into evaluation (20,383 segments), balanced training (22,176 segments), and unbalanced training (2,042,985 segments) sets.

## Use Cases
- **Audio Event Classification (AEC):** Training models to identify and classify sounds in recordings.
- **Automatic Content Tagging:** Application on video platforms (such as YouTube) to categorize and index content based on audio.
- **Surveillance and Monitoring Systems:** Detection of specific sound events (e.g., alarms, gunshots, baby crying).
- **Audio Processing Research:** Development of new neural network architectures and audio *embedding* techniques (such as VGGish).
- **Transfer Learning:** Use of pre-trained *embeddings* (VGGish) as *features* for related audio tasks.

## Integration
The dataset is made available in two formats:
1.  **CSV Files:** Contain the metadata for each segment: YouTube video ID, start time, end time, and labels (classes). The main files are `eval_segments.csv`, `balanced_train_segments.csv`, and `unbalanced_train_segments.csv`.
2.  ***Audio Embeddings* (Features):** 128-dimensional audio *features* extracted at 1Hz, stored as TensorFlow Record files (2.4 GB in total).

**Accessing the *Embeddings*:**
The *embeddings* can be downloaded manually via `tar.gz` from Google Cloud Storage (GCS) *buckets* or by using the `gsutil` utility for synchronization:
`gsutil rsync -d -r features gs://{region}_audioset/youtube_corpus/v1/features` (where {region} is 'us', 'eu', or 'asia').

**Accessing the Raw Audio:**
The dataset does not provide the raw audio files. It is necessary to use the YouTube video IDs and the time metadata to download and extract the audio clips from the original YouTube videos, which requires third-party tools and is subject to video availability. The VGGish model and supporting code are available in the TensorFlow models GitHub repository.

## URL
[https://research.google.com/audioset/](https://research.google.com/audioset/)
