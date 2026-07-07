# Data Augmentation

This directory covers methods and tools for increasing training diversity through label-preserving or label-aware transformations.

## Scope

- Image, video, audio, text, tabular, time-series, geospatial, graph, and multimodal augmentation.
- Synthetic data generation, perturbation, mixup/cutmix, back-translation, paraphrasing, masking, cropping, noise injection, and simulation.
- Augmentation policies, AutoAugment-style search, domain randomization, and robustness testing.

## Source Families

- Albumentations, torchvision transforms, TensorFlow/Keras augmentation layers, Kornia, NVIDIA DALI, imgaug, torchaudio, nlpaug, TextAttack, tsaug, and synthetic-data tools.
- Domain-specific augmentation references for medical imaging, remote sensing, speech, OCR, code, and safety evaluation.

## Reference Links

- Albumentations documentation: https://albumentations.ai/docs/
- TensorFlow data augmentation tutorial: https://www.tensorflow.org/tutorials/images/data_augmentation
- TorchVision transforms: https://pytorch.org/vision/stable/transforms.html
- Kornia augmentation: https://kornia.readthedocs.io/en/latest/augmentation.html
- NVIDIA DALI: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/

## Validation Standard

Document the target label, invariance assumption, transformation parameters, train/validation split safety, class-balance effect, artifact risk, and measured impact on validation and out-of-domain performance.

## Routing Rules

- Put feature creation in `../Feature_Engineering/`.
- Put general training strategies in `../../../01_AI_FUNDAMENTALS_AND_THEORY/Machine_Learning/training_strategies/`.
- Put synthetic datasets under the relevant dataset modality folder.
