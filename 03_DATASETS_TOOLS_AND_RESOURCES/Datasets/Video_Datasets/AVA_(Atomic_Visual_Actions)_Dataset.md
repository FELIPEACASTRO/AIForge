# AVA (Atomic Visual Actions) Dataset

## Description
The **AVA (Atomic Visual Actions)** is a video dataset for audiovisual annotations that aims to improve the understanding of human activity. The main dataset, **AVA Actions v2.2**, densely annotates 80 atomic visual actions in 430 15-minute movie clips, where actions are localized in space and time. The project also includes **AVA-Kinetics** (a combination with Kinetics-700 for action localization across a wider variety of scenes), **AVA ActiveSpeaker** (association of speech activity with a visible face), and **AVA Speech** (audio-based speech activity annotation), making it a robust multimodal resource.

## Statistics
- **AVA Actions v2.2:** 430 videos (235 training, 64 validation, 131 test), each with 15 minutes annotated at 1-second intervals. Total of 1.62M action labels.
- **AVA-Kinetics v1.0:** 430 videos from AVA v2.2 + 238k videos from Kinetics-700.
- **AVA ActiveSpeaker v1.0:** 3.65 million labeled frames across approximately 39 thousand facial tracks.
- **AVA Speech v1.0:** Approximately 46 thousand labeled segments, spanning 45 hours of data.

## Features
- **Spatio-Temporal Action Localization:** Actions are localized in space (bounding boxes) and time (1-second intervals), allowing a detailed analysis of the dynamics of human activity.
- **Atomic Actions:** Annotation of 80 atomic visual actions (for example, "stand", "shake hands", "talk").
- **Multiple Labels per Person:** Allows a person to have multiple action labels simultaneously.
- **Multimodal:** Includes sub-datasets for speech and active speaker analysis, integrating vision and audio.

## Use Cases
- **Action Recognition and Localization in Videos:** Training models to identify and localize human actions in real time.
- **Active Speaker Detection and Speech Activity Analysis:** Applications in conferencing systems, security, and human-computer interaction.
- **Computer Vision Research:** Development of new algorithms for understanding human activities and modeling social interactions.
- **Transfer Learning:** Use of AVA-Kinetics to expand the generalization of action localization models.

## Integration
The dataset is provided as CSV files containing the annotations. The original videos are identified by YouTube IDs and must be downloaded separately (which may require third-party tools, such as `youtube-dl` or specific scripts). The CSV annotation format includes: `video_id`, `middle_frame_timestamp`, `person_box` (x1, y1, x2, y2 normalized), `action_id`, and `person_id`. The annotation files (CSV) are available for direct download on the official page. The evaluation code (Frame-mAP) is available on the ActivityNet GitHub.

## URL
[https://research.google.com/ava/](https://research.google.com/ava/)