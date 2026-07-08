# Kinetics (Action Recognition)

## Description
Kinetics is a collection of large-scale, high-quality datasets designed for the recognition of human actions in videos. The dataset consists of URL links to YouTube video clips, covering human-object interactions (such as playing instruments) and human-human interactions (such as shaking hands). Each clip is approximately 10 seconds long and is annotated with a single action class. The most recent and comprehensive version is Kinetics-700-2020.

## Statistics
The Kinetics dataset has multiple versions: Kinetics-400 (400 classes), Kinetics-600 (600 classes), and Kinetics-700-2020 (700 classes). The Kinetics-700-2020 version is the most up-to-date and contains approximately 635,000 video clips in total. The total size of the dataset (videos) is about 710 GB. The data splits (CVDF split) are: Train (534,073 videos), Test (64,260 videos), and Validation (33,914 videos).

## Features
Large scale with 700 human action classes. High annotation quality, with each clip annotated by humans. Focus on dynamic human actions and interactions. The videos are short (about 10s) and extracted from YouTube. The dataset is frequently used as a benchmark for action recognition models.

## Use Cases
Training and evaluation of video action recognition models (Action Recognition). Research in computer vision and deep learning for video analysis. Transfer learning (pre-training) for more specific video tasks. Development of surveillance and behavior analysis systems.

## Integration
The dataset consists of YouTube URLs. Due to the volatility of the links, the CVDF (Computer Vision Foundation) hosts the videos on AWS S3. Integration is typically done through download scripts (available on the official GitHub) that download the videos from the URLs or from the hosted tar.gz files. Tools such as FiftyOne also offer simplified methods to load and manage the dataset (e.g., `foz.load_zoo_dataset(\"kinetics-700-2020\")`). It is necessary to have `ffmpeg` installed to work with the video files.

## URL
[https://deepmind.google/research/open-source/kinetics](https://deepmind.google/research/open-source/kinetics)
