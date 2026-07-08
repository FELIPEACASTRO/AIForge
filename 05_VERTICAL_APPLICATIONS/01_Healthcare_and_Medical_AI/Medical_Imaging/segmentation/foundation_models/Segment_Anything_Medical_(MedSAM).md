# Segment Anything Medical (MedSAM)

## Description

**Segment Anything Medical (MedSAM)** is a pioneering foundation model designed for **universal medical image segmentation**. It was developed to bridge the generalization gap of traditional segmentation models, which are typically specific to a single modality or disease. MedSAM is trained on a massive dataset of **1,570,263 image-mask pairs**, covering 10 imaging modalities and more than 30 cancer types. Its unique value proposition lies in its ability to deliver accurate and efficient segmentation across a broad spectrum of tasks, demonstrating **better accuracy and robustness** than specialist models and outperforming the original SAM in medical scenarios, especially on targets with weak boundaries or low contrast [1] [2].

## Statistics

**Training Dataset:** 1,570,263 image-mask pairs. **Data Diversity:** 10 imaging modalities (CT, MRI, Endoscopy, etc.) and more than 30 cancer types. **Evaluation:** 86 internal validation tasks and 60 external validation tasks, demonstrating superior robustness and generalization [1]. **Performance:** Consistently outperforms the original SAM and achieves performance equal to or better than specialist models [1].

## Features

**Universal Segmentation:** Ability to segment anatomical structures, lesions, and pathological regions across diverse imaging modalities (CT, MRI, Endoscopy, etc.). **Promptable Segmentation:** Uses *prompts* (points, bounding boxes) for interactive segmentation, offering a balance between automation and customization. **Refined SAM Architecture:** Based on the SAM architecture (image encoder, prompt encoder, and mask decoder), but fine-tuned for the medical domain. **MedSAM2 (Extension):** An enhanced version for **segmentation in 3D medical images and videos**, allowing structures to be delineated in volumetric scans with a single click [3].

## Use Cases

**Diagnosis and Treatment Planning:** Precise segmentation of organs, tumors, and lesions for radiotherapy and surgery planning. **Disease Monitoring:** Consistent tracking of disease progression, such as tumor growth, across sequential exams. **Medical Research:** Analysis of large medical imaging datasets to accelerate the discovery and validation of new biomarkers. **Diverse Clinical Applications:** Segmentation of structures in CT, MRI, and visual inspection of internal organs via Endoscopy [1].

## Integration

Integration is typically performed via PyTorch, following the structure of the official repository. The model requires downloading a pre-trained *checkpoint*.

**Installation (MedSAM):**
```bash
git clone https://github.com/bowang-lab/MedSAM
cd MedSAM
pip install -e .
# Download the model checkpoint (medsam_vit_b.pth)
```

**Usage Example (Conceptual - Python/PyTorch):**
```python
import torch
from medsam import SamPredictor, build_medsam
import numpy as np

# 1. Load the model and the predictor
medsam_checkpoint = "path/to/medsam_vit_b.pth"
model = build_medsam(checkpoint=medsam_checkpoint)
predictor = SamPredictor(model)

# 2. Load and process the image (image_data as a numpy array)
# predictor.set_image(image_data)

# 3. Define the prompt (example: a central point)
# input_point = np.array([[500, 375]])
# input_label = np.array([1]) # 1 for foreground

# 4. Predict the mask
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
# The mask (masks[0]) is the segmentation result
```

## URL

https://github.com/bowang-lab/MedSAM