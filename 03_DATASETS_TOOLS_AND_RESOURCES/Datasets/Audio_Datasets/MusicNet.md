# MusicNet

## Description
MusicNet is a collection of 330 freely-licensed classical music recordings, totaling more than 1 million annotated labels that indicate the precise timing of each note, the instrument that plays it, and its position in the metrical structure of the composition. The labels are acquired from musical scores aligned to the recordings via "dynamic time warping". The dataset is a fundamental resource for training machine learning models and a common benchmark for automatic music transcription tasks. A more recent version, MusicNet-16k + EM for YourMT3 (April 2023), provides audio resampled to 16 kHz and refined labels (MusicNet EM) for specific tasks.

## Statistics
**Recordings:** 330 classical music recordings. **Labels:** More than 1 million annotated labels. **Size:** The main file `musicnet.tar.gz` is 11.1 GB. **Versions:** Version 1.0 (November 2016). MusicNet-16k + EM for YourMT3 version (v6, April 2023), at 6.7 GB. **Duration:** The original dataset contains more than 330 hours of audio.

## Features
Contains PCM-encoded audio (.wav) and corresponding note labels in CSV format. Includes track-level metadata and reference MIDI files. The labels are verified by trained musicians, with an estimated error rate of 4%. The MusicNet-16k version offers audio resampled to 16 kHz (mono, 16-bit) and refined labels (MusicNet EM) for better performance on transcription tasks.

## Use Cases
**Automatic Music Transcription (AMT):** The main application for training models that convert audio into musical notation. **Instrument Recognition:** Identification of the instrument that plays each note. **Metrical Structure Analysis:** Study of the note's position within the rhythmic and metrical structure of the composition. **Machine Learning Research for Music:** Serves as a benchmark for comparing the performance of different methods.

## Integration
The original dataset can be downloaded in three main files: `musicnet.tar.gz` (audio and CSV labels), `musicnet_metadata.csv` (metadata), and `musicnet_midis.tar.gz` (reference MIDI files). Access and use are facilitated by a PyTorch interface available on GitHub, which allows loading and processing the data efficiently. For the MusicNet-16k version, the download is done directly through Zenodo, and the usage instructions and data splits are in the YourMT3 project repository.

## URL
[https://zenodo.org/records/5120004](https://zenodo.org/records/5120004)
