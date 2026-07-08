# LibriSpeech ASR Corpus

## Description
LibriSpeech is a large-scale corpus (approximately 1000 hours) of read English speech, designed specifically for training and evaluating Automatic Speech Recognition (ASR) systems. The dataset is derived from read audiobooks from the LibriVox project, which are in the public domain. The corpus was carefully segmented and aligned to provide accurate transcriptions for each audio segment. It is one of the most widely used benchmarks in ASR research due to its size and its division into "clean" and "other" subsets, which represent different levels of noise difficulty and recording quality. A notable extension is Multilingual LibriSpeech (MLS), which expands the concept to 8 languages, including Portuguese.

## Statistics
- **Total Speech Size:** Approximately 1000 hours.
- **Total File Size:** About 60 GB (audio and transcriptions only).
- **Main Splits:**
    - `train-clean-100`: 100 hours (training, clean)
    - `train-clean-360`: 360 hours (training, clean)
    - `train-other-500`: 500 hours (training, other)
    - `dev-clean`: ~5 hours (development, clean)
    - `dev-other`: ~5 hours (development, other)
    - `test-clean`: ~5 hours (test, clean)
    - `test-other`: ~5 hours (test, other)
- **Original Version:** Released in 2015. The most recent version is the original, but the dataset remains a standard benchmark, with extensions such as Multilingual LibriSpeech (MLS) and LibriSpeech-Long (Google DeepMind, 2024).

## Features
- **Large Scale:** Approximately 1000 hours of speech.
- **Public Domain:** Derived from LibriVox audiobooks, under the CC BY 4.0 license.
- **Difficulty Split:** Training and test subsets divided into "clean" (easier) and "other" (more challenging, with more noise or recording variations).
- **Sampling Rate:** Audio at 16kHz.
- **Precise Alignment:** Speech segments carefully aligned with their transcriptions.
- **Extensions:** Variations exist such as LibriSpeech-PC (with restored punctuation and capitalization) and Multilingual LibriSpeech (MLS) with support for multiple languages.

## Use Cases
- **Automatic Speech Recognition (ASR):** Training and evaluation of state-of-the-art ASR models.
- **Speech Processing:** Research in speech segmentation, alignment, and synthesis.
- **Transfer Learning:** Use as a pre-training dataset for ASR models in other languages or domains.
- **Model Evaluation:** Serves as a standard benchmark for comparing the performance of different speech model architectures (e.g., RNNs, Transformers, Conformer).

## Integration
The LibriSpeech dataset can be downloaded directly from the official OpenSLR site (SLR12) in several `.tar.gz` files, each corresponding to a specific split (training, development, test).

**Direct Download (OpenSLR):**
The user should download the files of interest (e.g., `train-clean-100.tar.gz`, `test-clean.tar.gz`) and extract them.

**Integration with Libraries:**
Many machine learning and audio processing libraries, such as **Torchaudio** (PyTorch), offer APIs for simplified downloading and loading of LibriSpeech.

*Example with Torchaudio (Python):*
```python
import torchaudio

# O Torchaudio gerencia o download e o carregamento
dataset = torchaudio.datasets.LIBRISPEECH(
    root="/caminho/para/dados",
    url="train-clean-100", # ou "test-clean", "dev-other", etc.
    download=True
)
# O dataset pode ser iterado para obter (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
```
The dataset is also available on **Hugging Face Datasets** (`openslr/librispeech_asr`), making it easy to use in language model pipelines.

## URL
[https://www.openslr.org/12](https://www.openslr.org/12)
