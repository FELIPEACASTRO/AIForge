# UrbanSound8K

## Description
**UrbanSound8K** is an audio dataset widely used for the classification of urban sounds. It contains **8,732 labeled sound excerpts** (with a maximum duration of 4 seconds) from 10 distinct classes of urban sounds. The classes are: air conditioner, car horn, children playing, dog bark, drilling, idling engine, gunshot, jackhammer, siren, and street music. The excerpts were extracted from field recordings uploaded to the Freesound.org site. The dataset is pre-organized into **10 *folds*** to facilitate cross-validation and ensure the comparability of results with the existing literature. A CSV file (`UrbanSound8k.csv`) accompanies the audio files, providing detailed metadata for each excerpt, including the ID of the original recording, the start and end time of the excerpt, a salience classification (1=foreground, 2=background), and the *fold* to which it belongs.

## Statistics
- **Dataset Size:** Approximately **6.0 GB** (Zenodo) to **7.0 GB** (Hugging Face/Kaggle) in compressed/raw format.
- **Samples:** **8,732** labeled sound excerpts.
- **Classes:** **10** classes of urban sounds (air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music).
- **Version:** The original and most cited version is from 2014. More recent versions (2023-2025) are generally adaptations or extensions of the original dataset, such as US8K_AV.

## Features
- **Multiclass and Labeled:** 10 distinct classes of urban sounds.
- **Audio Format:** WAV files with variable sampling rates and bit depths (retaining those of the original Freesound file).
- **Cross-Validation Structure:** Pre-divided into 10 *folds* to facilitate model evaluation and ensure reproducibility.
- **Rich Metadata:** Includes a CSV file with detailed information such as `fsID` (Freesound ID), `start`, `end`, `salience`, `fold`, `classID`, and `class`.
- **Short Duration:** Sound excerpt with a maximum duration of 4 seconds.

## Use Cases
- **Urban Sound Classification (ESC):** The main use case, serving as a *benchmark* for the automatic recognition of sound events in urban environments.
- **Noise Monitoring Systems:** Development of systems to monitor and analyze noise pollution in cities.
- **Autonomous Vehicles (AV):** Recent research (2024-2025) uses UrbanSound8K (and its extensions, such as US8K_AV) to give vehicles the ability to "hear" and interpret environmental sounds (horns, sirens, jackhammers) to improve safety and decision-making.
- **Smart Cities:** Applications in public safety (detection of gunshots or sirens) and traffic management.
- **Deep Learning Research:** Used to test and compare the performance of new neural network architectures (CNN, RNN, LSTM, hybrid models) for audio processing.

## Integration
The dataset can be obtained in two main ways:

1.  **Download via Browser:** By filling out the download form on the official page to receive a direct link.
2.  **Download via Python (Recommended):** Using the `soundata` package.
    *   Install the package: `pip install soundata`
    *   Use the Python code to download and load the dataset, following the example provided in the `soundata` documentation.

**Usage Instructions:** It is crucial to use the **10 pre-defined *folds*** for 10-fold cross-validation, as recommended by the authors. Reordering the data or using fewer *folds* can lead to inflated results that are not comparable with the literature.

**Acknowledgment:** The dataset must be cited in academic research using the reference of the original paper: Salamon, J., Jacoby, C. & Bello, J.P. A dataset and taxonomy for urban sound research. *Proceedings of the 22nd ACM International Conference on Multimedia* (2014).

## URL
[https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html)
