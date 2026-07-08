# RSNA 2024 Lumbar Spine Degenerative Classification - 1st Place Solution

## Description

1st place solution for the RSNA 2024 Lumbar Spine Degenerative Classification competition, focused on the degenerative classification of the lumbar spine from medical images. The **2-stage** approach is the unique value proposition: the first stage focuses on the precise localization of coordinates (`test_label_coordinates.csv`) using 3D and 2D ConvNeXt and Efficientnet-v2-l models, and the second stage performs the severity prediction using **Multiple Instance Learning (MIL)** with bi-LSTM and attention. The solution demonstrated that smaller models (ConvNeXt-small) and convolutional architectures outperformed larger models and Vision Transformers.

## Statistics

**Approach:** 2 stages (Coordinate Localization + Severity Prediction). **Models:** 3 types of models (`instance_number` prediction, coordinate prediction, and severity prediction). **Architectures:** 3D ConvNeXt, 2D ConvNeXt-base, Efficientnet-v2-l, ConvNeXt-small, and Efficientnet-v2-s with MIL. **Losses:** L1 Loss and Cross Entropy Loss. **Data Augmentation:** Random shifting of coordinates and `instance_number`, ShiftScaleRotate, RandomBrightnessContrast.

## Features

1. **Precise Localization:** Use of 3D and 2D models to predict the `instance_number` and the (x, y, z) coordinates of the vertebrae, essential for cropping and focusing on the area of interest. 2. **Severity Classification with MIL:** The second stage uses a Multiple Instance Learning (MIL) approach with bi-LSTM and attention for the final severity classification. 3. **Robustness:** Use of an ensemble of coordinate predictions and data augmentation (such as `instance_number` shifting) to increase model robustness. 4. **Architecture Optimization:** Discovery that smaller models (ConvNeXt-small) and convolutional architectures were more effective.

## Use Cases

1. **AI-Assisted Medical Diagnosis:** Automated classification of the severity of lumbar spine degenerative disease from MRI scans. 2. **Medical Image Analysis:** Application of localization and classification techniques on 3D (DICOM) images to identify and assess pathologies. 3. **Computer Vision Model Development:** Demonstration of an effective 2-stage pipeline for complex computer vision tasks on medical data.

## Integration

The solution uses PyTorch and custom modules. The key MIL component is implemented with the following structure in Python/PyTorch:
```python
class LSTMMIL(nn.Module):
    def __init__(self, input_dim):
        super(LSTMMIL, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    def forward(self, bags):
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = self.lstm(bags)
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_lstm).squeeze(1)
        return weighted_instances, aux_attn_scores
```
The training code is on Google Colab, and the inference code is in a separate Kaggle notebook.

## URL

https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution
