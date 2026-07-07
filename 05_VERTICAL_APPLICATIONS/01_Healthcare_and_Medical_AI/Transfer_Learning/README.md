# Healthcare Transfer Learning

This directory covers transfer learning for healthcare AI, including pretrained model adaptation, domain adaptation, fine-tuning, self-supervised learning, multimodal transfer, and clinical validation.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Medical_Imaging/` | Transfer learning for radiology, pathology, microscopy, segmentation, classification, detection, and report-aligned imaging models. |

## Core Patterns

- ImageNet, medical-image, and foundation-model initialization.
- Self-supervised and weakly supervised pretraining.
- Few-shot adaptation, linear probing, parameter-efficient fine-tuning, and adapter methods.
- Domain shift handling across scanners, hospitals, populations, protocols, and labels.
- Clinical validation with calibration, subgroup analysis, and human review.

## Source Families

- MONAI, TorchVision, Hugging Face, timm, PyTorch Lightning, and medical challenge baselines.
- TCIA, PhysioNet, MIMIC, UK Biobank, Grand Challenge, BraTS, KiTS, LiTS, and other licensed medical datasets.

## Reference Links

- MONAI documentation: https://monai-dev.readthedocs.io/
- PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- TorchVision models: https://pytorch.org/vision/stable/models.html
- Hugging Face models: https://huggingface.co/models
- timm: https://huggingface.co/docs/timm/index
- PhysioNet: https://physionet.org/

## Routing Rules

- Put segmentation-specific transfer notes in `../Medical_Imaging/Segmentation/`.
- Put edge optimization in `../Edge_AI/Model_Compression/`.
- Put generic transfer-learning theory in the foundations/deep-learning area.
- Put regulated deployment notes under MLOps and governance folders.
