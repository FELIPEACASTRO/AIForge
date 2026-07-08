# Ensemble Model with Transfer Learning for Early Detection of Alzheimer's Disease (AD)

## Description

A robust deep learning *framework* that employs *transfer learning* and hyperparameter tuning of the InceptionResnetV2, InceptionV3, and Xception architectures. It uses an *ensemble voting* mechanism to combine predictions and optimize accuracy and robustness in classifying four stages of Alzheimer's Disease (AD): Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. The ensemble model outperformed all individual baseline models.

## Statistics

- **Overall Accuracy:** 98.96%
- **Precision (Mildly Demented):** 100%
- **Precision (Moderately Demented):** 100%
- **Recall (Mildly Demented):** 100%
- **Recall (Moderately Demented):** 100%
- **Performance Metrics:** The ensemble model achieved superior performance across all metrics (Accuracy, Precision, Recall, F1-Score) compared to the individual base models (InceptionResNetV2, InceptionV3, Xception).
- **Misclassifications:** Only 9 misclassifications of "Very Mild Demented" samples as "Non-Demented," the lowest error among all evaluated architectures.
- **Publication:** Scientific Reports, Volume 15, Article number: 34634 (2025).

## Features

- **Weighted Ensemble Architecture:** Combines the predictions of three pre-trained CNN models (InceptionResNetV2, InceptionV3, Xception) using a *weighted voting* mechanism.
- **Transfer Learning:** Uses models pre-trained on ImageNet for feature extraction, with *fine-tuning* of the final layers for AD-specific classification.
- **Multi-Class Classification:** Classifies magnetic resonance imaging (MRI) scans into four AD progression categories.
- **Pre-processing and Data Augmentation:** Includes resizing, grayscale conversion, and data augmentation techniques (horizontal flip, zoom, shear) to mitigate data imbalance and improve generalization.

## Use Cases

- **Early Diagnosis of Alzheimer's Disease (AD):** Classification of AD stages (Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented) from magnetic resonance imaging (MRI) scans.
- **Clinical Decision Support:** Providing an automated, robust second opinion for radiologists and neurologists, assisting in early intervention and disease management.
- **Neuroimaging Research:** Serving as a high-performance baseline architecture for future research seeking to incorporate multimodal data (clinical, genetic) for even more accurate diagnosis.

## Integration

The study's source code is available in a public GitHub repository, which facilitates integration and reproduction of the model. The *framework* is implemented in a Google Colab environment, suggesting the use of popular deep learning libraries such as TensorFlow/Keras.

**Code Resources:**
- **GitHub Repository:** `https://github.com/muhammadmo/Alzheimer_Classification_MRI`
- **Dataset:** The study used the ADNI dataset, available on Kaggle: `https://www.kaggle.com/datasets/praneshkumarm/multidiseasedataset`

## URL

https://www.nature.com/articles/s41598-025-22025-y
