# CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)

## Description
The **CSTR VCTK Corpus** (Centre for Speech Technology Voice Cloning Toolkit) is an English multi-speaker speech dataset, designed primarily for research in **speech synthesis (Text-to-Speech - TTS)**, especially for speaker-adaptive synthesis systems. The most common version (v0.92) includes speech data uttered by **110 native English speakers** with a variety of regional accents. Each speaker reads about 400 sentences, selected from a newspaper (The Herald, Glasgow), plus the "Rainbow Passage" and an accent-elicitation paragraph. The corpus is widely used to train HMM-based (Hidden Markov Model) TTS models and, more recently, multi-speaker speech synthesis systems based on Deep Neural Networks (DNNs) and neural waveform modeling, such as WaveNet. The corpus is notable for its high recording quality, carried out in a hemi-anechoic chamber at the University of Edinburgh, using high-fidelity microphones (DPA 4035 and Sennheiser MKH 800), with a sampling rate of 96kHz and 24 bits, subsequently converted to 48kHz and 16 bits.

## Statistics
- **Total Size:** 10.94 GB (main file).
- **Speakers:** 110 native English speakers (109 with complete transcriptions, `p315` lost the text file).
- **Samples:** Each speaker reads approximately 400 sentences. The total number of audio clips is about 44,000.
- **Total Duration:** Approximately 44 hours of speech.
- **Sampling Rate:** 48 kHz (originally recorded at 96 kHz).
- **Main Version:** 0.92 (available since 2019-11-13).

## Features
- **Multi-speaker and Multi-accent:** Contains 110 native English speakers with diverse regional accents, making it ideal for adaptive and multi-speaker TTS models.
- **High Recording Quality:** Recorded in a hemi-anechoic chamber with professional microphones (DPA 4035 and Sennheiser MKH 800) at 96kHz/24 bits, subsequently downsampled to 48kHz/16 bits.
- **Varied Content:** The sentences include newspaper texts, the "Rainbow Passage" (for phonetic analysis), and an accent-elicitation paragraph.
- **Focus on Speech Synthesis:** Originally intended for HMM-based TTS systems and, currently, crucial for the development of neural TTS models (such as VITS and WaveNet).

## Use Cases
- **Speech Synthesis (Text-to-Speech - TTS):** Training high-quality TTS models, including neural systems such as WaveNet, Tacotron, and VITS.
- **Voice Cloning:** Development of voice cloning systems and speaker-adaptive synthesis.
- **Speech Recognition:** Although not the main focus, it can be used for training and evaluating multi-speaker speech recognition models.
- **Accent Analysis:** Research in phonetic variation and regional English accents.
- **Speech Enhancement:** Used as a basis for creating derived datasets for speech enhancement (e.g., VCTK-RVA for voice attributes).

## Integration
The VCTK dataset (version 0.92) is available for download in the Edinburgh DataShare repository. The main download is a **10.94 GB** file that contains the audio and text files.

**Integration Steps:**
1. **Access:** Navigate to the resource page on Edinburgh DataShare (main URL provided).
2. **Download:** Click the download link for the "Main file including audio and text files (10.94Gb)".
3. **Structure:** The corpus is generally organized into folders for each speaker (`p225`, `p226`, etc.), containing the audio files (`.wav`) and the corresponding transcription files (`.txt`).
4. **Usage:** For use in machine learning projects, it is common to use libraries such as `torchaudio` or `tensorflow_datasets` which may offer wrappers for VCTK, or to manually process the audio and text files to create training pairs. For example, `tensorflow_datasets` offers a ready-to-use version of VCTK.
5. **Citation:** It is mandatory to cite the original work when using the corpus.

## URL
[https://datashare.ed.ac.uk/handle/10283/3443](https://datashare.ed.ac.uk/handle/10283/3443)
