# CheXpert Plus: Radiology Multimodal Dataset

## Description

CheXpert Plus is a comprehensive, multimodal collection that extends the original CheXpert dataset, combining chest X-ray images with aligned radiology reports, patient demographic data, and additional image formats. This dataset was released in 2024 and represents a significant advance for training multimodal AI models, such as Large Language Models (LLMs) applied to radiology. It is notable for being the largest publicly released text dataset in radiology, facilitating the development of systems that can interpret and generate medical reports with greater accuracy and clinical context.

## Statistics

**Total Unique Image-Report Pairs:** 223,462. **Studies:** 187,711. **Patients:** 64,725. **Text Size:** 36 million text tokens, including 13 million impression tokens. **Image Format:** DICOM. **Annotated Pathologies:** 14. **Release:** 2024.

## Features

Multimodal (chest X-ray images and radiology reports). Radiology reports meticulously divided into 11 subsections. Includes 47 DICOM metadata elements. Annotations for 14 thoracic pathologies. 8 metadata elements about patient information. Focus on text-image alignment for training vision-language models.

## Use Cases

**Multimodal Model Training:** Development of Large Language Models (LLMs) for radiology, such as CXR-LLaVA. **Radiology Report Generation:** Creation of systems that generate reports from images. **Multilabel Classification:** Detection and classification of 14 thoracic pathologies. **Bias and Fairness Research:** Analysis of biases in imaging AI due to the inclusion of demographic data. **Feature Engineering:** Extraction of text features (NLP) and image features (visual) for diagnostic tasks.

## Integration

The dataset is made available by the Stanford Center for Artificial Intelligence in Medicine & Imaging (AIMI). Access requires acceptance of the Terms and Conditions and can be obtained through the AIMI portal. The official GitHub repository (Stanford-AIMI/chexpert-plus) provides information and potentially scripts for downloading and processing. Use in machine learning projects generally involves aligning the image-text pairs for tasks such as report generation and multimodal classification. Access example (Python pseudocode):
```python
# Access requires registration and approval on the AIMI portal
# Example of use after downloading and unpacking:

import pandas as pd
import os

# Load the metadata file (example)
metadata_path = 'path/to/chexpert_plus/metadata.csv'
df_metadata = pd.read_csv(metadata_path)

# Access an image-report pair
for index, row in df_metadata.head().iterrows():
    image_path = os.path.join('path/to/chexpert_plus/images', row['dicom_id'] + '.dcm')
    report_text = row['full_report_text']
    
    print(f"Image: {image_path}")
    print(f"Report: {report_text[:200]}...")
    # Implement DICOM image loading and text processing logic
```

## URL

https://aimi.stanford.edu/datasets/chexpert-plus