# Computer Vision Data Augmentation

This directory covers image and video augmentation for computer vision models.

## Scope

- Cropping, resizing, color jitter, blur, noise, CutMix, MixUp, mosaic, geometric transforms, test-time augmentation, and synthetic generation.
- Track task, label preservation, bounding-box/mask handling, parameter ranges, and validation impact.

## Reference Links

- Albumentations: https://albumentations.ai/docs/
- TorchVision transforms: https://pytorch.org/vision/stable/transforms.html
- Kornia augmentation: https://kornia.readthedocs.io/en/latest/augmentation.html
- NVIDIA DALI: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/

## Routing Rules

- Put general augmentation in the parent data augmentation directory.
- Put medical imaging augmentation in healthcare imaging folders.
