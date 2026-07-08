# Snap-MAE (Snapshot Ensemble Learning into Masked Autoencoders)

## Description

A self-supervised learning (SSL) model that integrates *snapshot* ensemble learning into the pre-training process of the Masked Autoencoder (MAE). It uses a cyclic cosine scheduler to capture diverse model representations within a single training cycle, optimizing computational efficiency and improving performance.

## Statistics

Consistently outperforms vanilla MAE, ViT-S, and ResNet-34 across all performance metrics in the tested use cases. Reduces the computational burden associated with ensemble learning, producing multiple pre-trained models from a single pre-training phase.

## Features

Integration of *snapshot ensemble learning* into MAE; Use of a cyclic cosine scheduler; Optimization of computational efficiency; Generation of multiple diverse pre-trained models; Robustness in pre-training with unlabeled data.

## Use Cases

Multi-label classification of pediatric thoracic diseases; Diagnosis of cardiovascular diseases.

## Integration

A direct and effective implementation to improve SSL-based pre-training in medical imaging. (The paper provides the theoretical basis for implementation, but the official repository was not found in the search.)

## URL

https://www.nature.com/articles/s41598-025-15704-3