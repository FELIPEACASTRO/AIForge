# BraTS-Africa Dataset (BraTS 2023 Extension)

## Description

The BraTS-Africa Dataset is a crucial extension of the BraTS (Brain Tumor Segmentation) challenge, launched in 2023, with the goal of expanding the diversity of brain tumor magnetic resonance imaging (MRI) data to include Sub-Saharan African (SSA) populations. It is the first publicly available, annotated brain imaging dataset from Africa, addressing the lack of geographic representation in medical AI datasets. The dataset was curated from six diagnostic centers in Nigeria and is fundamental to developing more generalizable and equitable brain tumor segmentation models, especially in resource-limited settings.

## Statistics

**Size:** 3.7 GB. **Patients:** 146 patients. **Origin:** Six diagnostic centers in Nigeria. **Acquisition Period:** January 2010 to December 2022. **Modalities:** T1, T1 CE, T2, T2 FLAIR. **Format:** NIfTI. **Annotations:** Tumor sub-region segmentations (Core, Edema, Enhancing).

## Features

**Multiparametric MRI Images (mpMRI):** Includes T1, contrast-enhanced T1 (T1 CE), T2, and T2 FLAIR. **Segmentation Annotations:** Tumor sub-region segmentations (Necrotic/Non-Enhancing Tumor Core, Peritumoral Edema, and Enhancing Tumor) annotated by experts. **Radiomic Features:** Recent research (2024-2025) demonstrates the use of **Radiomics** techniques to extract hundreds of quantitative features (such as shape, intensity, and texture) from tumoral and peritumoral regions, aiming at glioma subtype prediction and treatment response assessment. Tools such as `pyradiomics` are commonly used for this extraction.

## Use Cases

**Brain Tumor Segmentation:** Training and evaluation of Deep Learning models (such as U-Net and MedNeXt) for automatic segmentation of gliomas in African populations. **Model Generalization:** Use as a test dataset to evaluate the robustness and equity of models trained on predominantly Western data. **Radiomics Research:** Extraction of radiomic features for prognosis prediction, molecular glioma subtypes, and therapy response assessment. **Low-Resource Tool Development:** Building lightweight, efficient models for use in clinical settings with limited computational resources.

## Integration

The dataset is available through **The Cancer Imaging Archive (TCIA)**. Access to the imaging and segmentation data requires the use of the IBM-Aspera-Connect plugin for download. Alternatively, the dataset can be accessed via **Kaggle** (unofficial or smaller versions) or through repositories of research projects that use it.

**Access Example (TCIA - Download Required):**
1.  Go to the dataset page on TCIA.
2.  Use the IBM-Aspera-Connect plugin to download the `Radiology Images and Segmentations - BraTS 2023 Challenge` file (which includes the BraTS-Africa data).

**Integration Example (Python - Conceptual for Radiomics):**
```python
# Conceptual example of radiomic feature extraction
# Requires prior download of the NIfTI files from TCIA
import SimpleITK as sitk
from radiomics import featureextractor

# 1. Load the image and the segmentation mask
image_path = 'path/to/image_T1CE.nii.gz'
mask_path = 'path/to/segmentation_mask.nii.gz'
image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

# 2. Configure the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()
# Settings can be adjusted to extract specific features (e.g., GLCM, GLRLM)

# 3. Run the extraction
result = extractor.execute(image, mask)

# 4. Display the extracted features
# print(result)
```

## URL

https://www.cancerimagingarchive.net/collection/brats-africa/
