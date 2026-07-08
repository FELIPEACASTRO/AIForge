# TransformEHR

## Description

A Transformer-based generative encoder-decoder model, pretrained with a novel objective of predicting all of a patient's diseases and outcomes at a future visit from prior visits. Designed to improve the prediction of disease outcomes using longitudinal Electronic Health Records (EHR). It has been shown to be capable of capturing complex relationships in longitudinal EHR data.

## Statistics

Outperformed prior models (such as BERT, LSTM) in prediction tasks.
*   **Pancreatic Cancer:** AUROC of 81.95%.
*   **Intentional Self-Harm (in patients with PTSD):** AUPRC improved by 24% relative to BERT (AUPRC of 16.67%).
*   **Citations:** 106 (in 2023).

## Features

Transformer encoder-decoder architecture; Novel pretraining objective for predicting future diseases; Ability to fine-tune with limited data; High performance in clinical prediction tasks.

## Use Cases

Prediction of rare diseases (e.g., pancreatic cancer); Prediction of critical clinical outcomes (e.g., intentional self-harm in patients with PTSD); Transfer learning to new EHR datasets.

## Integration

The TransformEHR fine-tuning code is publicly available on GitHub. The model can be adapted for various clinical prediction tasks.
**Example Repository:**
```
https://github.com/whaleloops/TransformEHR/
```

## URL

https://www.nature.com/articles/s41467-023-43715-z