# Medical Image Segmentation

This directory covers segmentation of medical images across radiology, pathology, microscopy, ophthalmology, cardiology, radiation oncology, surgery, and multimodal clinical imaging.

## Content Map

| Subdirectory | Scope |
|---|---|
| `foundation_models/` | Medical segmentation foundation models, promptable segmentation, SAM-style adaptation, and generalist biomedical image models. |

## Core Topics

- Organ, lesion, vessel, tumor, cell, tissue, and anatomy segmentation.
- 2D, 3D, 4D, multimodal, and longitudinal segmentation.
- U-Net families, nnU-Net, MONAI, transformers, diffusion-assisted segmentation, and promptable segmentation.
- Dice, IoU, Hausdorff distance, surface distance, calibration, clinical review, and failure-case analysis.

## Source Families

- MONAI, nnU-Net, TotalSegmentator, Medical Segmentation Decathlon, Grand Challenge, TCIA, BraTS, KiTS, LiTS, and MICCAI challenge resources.
- FDA, clinical validation, DICOM, PACS, de-identification, and medical-device documentation where deployment is involved.

## Reference Links

- MONAI documentation: https://monai-dev.readthedocs.io/
- MONAI tutorials: https://github.com/Project-MONAI/tutorials
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet
- Medical Segmentation Decathlon: http://medicaldecathlon.com/
- Grand Challenge: https://grand-challenge.org/
- The Cancer Imaging Archive: https://www.cancerimagingarchive.net/

## Routing Rules

- Put general medical imaging resources in `../`.
- Put genomics and drug-discovery resources in sibling healthcare folders.
- Put edge deployment and compression in `../../Edge_AI/Model_Compression/`.
- Put general segmentation theory in the computer-vision or deep-learning foundations folders.
