# LJSpeech Dataset

## Description
The **LJ Speech Dataset** is a public domain corpus widely used for training and evaluating **Text-to-Speech (TTS)** models. It consists of 13,100 short audio clips of a single female speaker reading passages from 7 non-fiction books in English. Each audio clip is accompanied by a corresponding transcription, both in its original form and in a normalized version where numbers and abbreviations are expanded into full words. The texts were published between 1884 and 1964, and the audio was recorded in 2016-2017 as part of the LibriVox project. The dataset is known for its high recording quality and consistency, being a reference standard in speech synthesis research.

## Statistics
*   **Most Recent Version:** 1.1
*   **Download Size:** 2.6 GB
*   **Total Clips:** 13,100
*   **Total Duration:** 23 hours, 55 minutes, and 17 seconds (23:55:17)
*   **Average Clip Duration:** 6.57 seconds
*   **Total Words:** 225,715
*   **Sampling Rate:** 22050 Hz
*   **Audio Format:** WAV, mono, 16-bit PCM

## Features
*   **Public Domain:** The dataset, including texts, audio, and annotations, is in the public domain, with no usage restrictions.
*   **Single Speaker:** Contains recordings of a single female speaker, which makes it ideal for single-speaker TTS models.
*   **Normalized Transcription:** Includes original and normalized transcriptions (with numbers and abbreviations expanded), facilitating preprocessing for TTS models.
*   **High Quality:** Audio in WAV format, mono, 16-bit PCM, with a sampling rate of 22050 Hz.
*   **Segmentation:** The audio clips are short (1 to 10 seconds), automatically segmented with manual alignment and quality assurance (QA) for accuracy.

## Use Cases
*   **Speech Synthesis (Text-to-Speech - TTS):** This is the main use case, serving as a reference dataset for training and evaluating TTS models, such as Tacotron, FastSpeech, Glow-TTS, and VITS.
*   **Voice Cloning:** Used to create single-speaker voice cloning models.
*   **Audio Processing Research:** Used for experiments in audio preprocessing, spectrogram analysis, and text normalization techniques.
*   **Training Speech Language Models:** Although it is a TTS dataset, it is frequently used together with other larger datasets (such as LibriSpeech) for pre-training components of speech language models.

## Integration
The dataset can be downloaded directly from the creator's website or accessed through platforms such as Hugging Face Datasets and Kaggle.

**Direct Download:**
1.  Access the main URL: [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)
2.  Click the download link (2.6 GB).

**Usage with Python (Hugging Face Datasets Example):**
For use in machine learning projects, the most common integration is via the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Loads version 1.1 of the dataset
ljspeech_dataset = load_dataset("keithito/lj_speech")

# The dataset is loaded as a DatasetDict object with the 'train', 'validation', and 'test' splits (if applicable)
# The data structure includes 'audio' (file path and audio array), 'text' (original transcription), and 'normalized_text' (normalized transcription).
print(ljspeech_dataset)
```

**File Structure:**
The dataset consists of:
*   A `metadata.csv` file (or `transcripts.csv`) with three fields per line: `ID` (WAV file name), `Transcription` (original transcription), and `Normalized Transcription` (normalized transcription).
*   A folder containing 13,100 audio files in `.wav` format.

## URL
[https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)
