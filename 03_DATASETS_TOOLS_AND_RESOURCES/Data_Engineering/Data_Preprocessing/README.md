# Data Preprocessing

This directory covers preprocessing steps that prepare raw data for AI/ML use.

## Scope

- Cleaning, normalization, encoding, imputation, parsing, deduplication, augmentation, tokenization, and modality-specific transforms.
- Track input format, output schema, fit/transform split safety, leakage risks, and reproducibility.

## Reference Links

- scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- pandas documentation: https://pandas.pydata.org/docs/
- TensorFlow data augmentation: https://www.tensorflow.org/tutorials/images/data_augmentation
- TorchVision transforms: https://pytorch.org/vision/stable/transforms.html

## Routing Rules

- Put augmentation-specific material in `Augmentation/`.
- Put production pipelines in data-pipeline directories.
