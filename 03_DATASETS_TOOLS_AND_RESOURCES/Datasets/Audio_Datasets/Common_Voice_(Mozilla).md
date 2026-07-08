# Common Voice (Mozilla)

## Description
**Common Voice** is a project by the Mozilla Foundation that aims to build the largest and most diverse open voice corpus in the world. The dataset is composed of voice clips recorded by volunteers, who read donated sentences, and is designed to mitigate bias in Artificial Intelligence (AI) systems and democratize speech technology. As of version 23.0, the datasets are distributed exclusively through the **Mozilla Data Collective** [1]. The dataset is ideal for training Automatic Speech Recognition (ASR), Speech Synthesis (TTS), and other Natural Language Processing (NLP) applications [2].

## Statistics
- **Most Recent Version (2025)**: Common Voice 23.0 (released in September 2025) [3].
- **Total Hours Recorded**: **35,921** hours [3].
- **Validated Hours**: **24,600** hours [3].
- **Number of Languages**: **286** languages (in version 23.0) [3].
- **File Size (Example)**: The full download of version 23.0 is approximately **3.51 GB** (for the Single Word Target segment) [2].
- **Samples**: Millions of voice clips.

## Features
- **Massive Multilingualism**: Supports more than 137 languages, with version 23.0 expanding to 286 languages [1] [3].
- **Speech and Text Data**: Each entry consists of an audio clip (MP3) and the corresponding text.
- **Demographic Metadata**: Includes optional demographic metadata (age, sex, accent) to aid in training more accurate and less biased models.
- **Open License**: Distributed under the **CC0** license (Creative Commons Zero), allowing unrestricted and free use for any purpose.
- **Validated Data**: The data is validated by other volunteers, ensuring the quality of the corpus.
- **Speech Types**: Includes scripted speech and, more recently, spontaneous speech [2].

## Use Cases
- **Automatic Speech Recognition (ASR)**: Training ASR models for voice-to-text transcription.
- **Speech Synthesis (TTS)**: Creation of synthetic voices (although the dataset is primarily for ASR, the text and audio data are useful).
- **Natural Language Processing (NLP)**: Research and development in areas such as accent identification, emotion detection, and linguistic diversity analysis.
- **AI Democratization**: Development of voice technologies for low-resource languages, combating linguistic bias in commercial systems [1].

## Integration
The Common Voice dataset is distributed as a `.tar.gz` file per language. The download is done through the **Mozilla Data Collective** [1].

**Download Steps:**
1.  Access the **Mozilla Data Collective** (main URL).
2.  Search for "Common Voice" and select the desired dataset (e.g., "Common Voice Scripted Speech 23.0").
3.  The download is usually initiated after providing an email address and accepting the terms of use, which include the commitment not to attempt to identify the speakers.
4.  For large file downloads, it is recommended to use command-line tools such as `curl` with the `-C` option to resume interrupted downloads [2].

**File Structure:**
Each `.tar.gz` file contains:
-   `clips/`: `.mp3` files of the audio clips.
-   `.tsv` (tab-separated values) files for different partitions: `train.tsv`, `dev.tsv`, `test.tsv`, `validated.tsv`, `invalidated.tsv`, `other.tsv`.
-   Each line of the `.tsv` contains the `client_id` (anonymized), the file path, the transcription (`text`), and demographic metadata [2].

**Usage with Libraries:**
The dataset is widely supported by NLP libraries, such as **Hugging Face Datasets**, where it can be loaded directly:
```python
from datasets import load_dataset

# Example for version 13.0 (newer versions may require manual download)
common_voice = load_dataset("mozilla-foundation/common_voice_13_0", "pt")
```

## URL
[https://datacollective.mozillafoundation.org/](https://datacollective.mozillafoundation.org/)
