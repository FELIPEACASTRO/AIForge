# VoxCeleb (Speaker Recognition)

## Description
**VoxCeleb** is a large-scale audio-visual dataset designed for speaker recognition and speaker verification tasks in "in the wild" (uncontrolled) scenarios. The dataset is composed of speech clips extracted from celebrity interview videos uploaded to YouTube. The most recent and comprehensive version is **VoxCeleb2**, which contains more than 1 million utterances from 6,112 celebrities. The dataset is notable for its diversity, spanning a wide range of ethnicities, accents, professions, and ages, and for capturing speech in real conditions, including background noise, laughter, and overlapping speech. Although the direct download links from the official site have been removed for privacy reasons, the dataset remains the gold standard for research in the area, with annual challenges (VoxSRC) and various third-party implementations for download and use.

## Statistics
**VoxCeleb1:**
- **Speakers:** 1,251 celebrities
- **Utterances:** > 150,000
- **Duration:** Not specified (but shorter than VoxCeleb2)

**VoxCeleb2 (Most used version):**
- **Speakers:** 6,112 celebrities
- **Utterances:** > 1,092,009 (Development) + 36,237 (Test) = **> 1,128,246**
- **Total Duration:** **> 2,000 hours**
- **Videos:** 145,569 (Development) + 4,911 (Test) = **150,480**
- **Versions:** VoxCeleb1, VoxCeleb2, and annual challenges (VoxSRC) that use and expand the dataset.

## Features
- **Audio-Visual:** Contains audio and video data, enabling the development of multimodal models.
- **"In the Wild":** Collected from YouTube videos, which ensures variability in pose, lighting, background noise, and audio quality.
- **Large Scale:** VoxCeleb2 has more than 1 million utterances and 6,112 identities.
- **Diversity:** Spans a wide range of ethnicities, accents, professions, and ages.
- **Celebrity Focus:** The identity of each speaker is a public celebrity, facilitating data collection.

## Use Cases
- **Speaker Identification:** Identifying who is speaking from a voice sample.
- **Speaker Verification:** Confirming whether the identity claimed by a speaker matches their voice.
- **Emotion Recognition from Voice:** Although not the primary focus, the dataset has been used for emotion studies (e.g., EmoVoxCeleb).
- **Multimodal Speech Processing:** Research on fusing audio and video information (face-tracks) to improve the robustness of systems.
- **Research Challenges (VoxSRC):** Used as the basis for the VoxCeleb Speaker Recognition Challenge, one of the main benchmarks in the area.

## Integration
Due to the removal of direct links from the official site for privacy reasons, integration of VoxCeleb is typically performed through third-party scripts or mirror repositories.

1.  **Download Scripts:** The most common method is to use Python/Shell scripts available in repositories such as GitHub (e.g., `clovaai/voxceleb_trainer` or `walkoncross/voxceleb2-download`) that automate the process of downloading the YouTube videos (based on URL and timestamp metadata) and extracting the audio/video segments.
2.  **Third-Party Repositories:** Some platforms such as Academic Torrents or Hugging Face (e.g., `ProgramComputer/voxceleb`) offer preprocessed versions or links to the complete dataset, although the user should always verify the license and the integrity of the data.
3.  **Evaluation Protocols:** For speaker verification tasks, the evaluation protocols (test pair lists) are provided on the official site and are essential to ensure the reproducibility of results.

*Note: It is necessary to accept the Terms and Conditions and be aware of the privacy issues and the Creative Commons Attribution-ShareAlike 4.0 International licensing.*

## URL
[https://www.robots.ox.ac.uk/~vgg/data/voxceleb/](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
