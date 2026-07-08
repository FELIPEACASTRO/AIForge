# ESC-50 (Environmental Sound Classification)

## Description
ESC-50 (Environmental Sound Classification) is a labeled dataset composed of 2,000 environmental audio recordings, each 5 seconds long, suitable for evaluating environmental sound classification methods. The dataset is balanced, containing 50 distinct classes (40 clips per class), grouped into 5 main categories: Animals, Natural Phenomena, Materials, Non-Vocal Human Sounds, and Domestic/Urban Sounds. The recordings were extracted from the **Freesound** database and pre-processed to ensure quality and uniformity. It is widely used as a standard benchmark in audio classification research.

## Statistics
- **Total Size:** Approximately 500 MB (WAV files).
- **Samples:** 2,000 audio clips.
- **Clip Duration:** 5 seconds each.
- **Classes:** 50 environmental sound classes.
- **Samples per Class:** 40 clips per class (balanced dataset).
- **Sampling Rate:** 44.1 kHz.
- **Version:** The original version was introduced in 2015. Derived and enhanced versions, such as **ESC50Mix** (2023), which adds sound mixtures, continue to be developed based on ESC-50.

## Features
- **50 Environmental Sound Classes:** Includes sounds such as dog barking, rain, car engine, coughing, etc.
- **Class Structure:** The 50 classes are grouped into 5 main categories to facilitate analysis and hierarchical classification.
- **5-Second Recordings:** All audio clips have a fixed duration of 5 seconds.
- **Audio in WAV Format:** The audio files are provided in WAV format, with a sampling rate of 44.1 kHz.
- **Split into 5 Folds:** The dataset is pre-divided into 5 folds for cross-validation, as suggested by the authors, ensuring fair and reproducible model evaluation.

## Use Cases
- **Environmental Sound Classification (ESC):** The primary task the dataset was designed for.
- **Sound Event Detection (SED):** Although it is a classification dataset, it is frequently used for pre-training or as a basis for detection tasks.
- **Audio Monitoring Systems:** Applications in security, wildlife monitoring, and smart home systems (e.g., glass-break detection, smoke alarm).
- **Deep Learning Research:** Benchmark for the development and testing of new convolutional neural network (CNN) and transformer architectures for audio processing.
- **Transfer Learning:** Use of the dataset to pre-train models that will be fine-tuned for more specific audio tasks.

## Integration
The dataset can be downloaded directly from the official GitHub repository or from platforms such as Kaggle and Hugging Face.

**Download and Usage:**
1. **GitHub (Main Source):** Download the ZIP file from the official repository.
   - `git clone https://github.com/karolpiczak/ESC-50.git`
2. **Kaggle:** Available for download and use in Kaggle notebooks.
3. **Hugging Face Datasets:** Can be loaded directly into Python projects using the `datasets` library.
   - `from datasets import load_dataset`
   - `dataset = load_dataset("ashraq/esc50")`

The dataset is composed of an audio folder (`audio`) and a CSV metadata file (`meta/esc50.csv`) that contains the file name, the class, the cross-validation fold ID, and the main category. It is recommended to use the provided 5-fold split for training and testing.

## URL
[https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
