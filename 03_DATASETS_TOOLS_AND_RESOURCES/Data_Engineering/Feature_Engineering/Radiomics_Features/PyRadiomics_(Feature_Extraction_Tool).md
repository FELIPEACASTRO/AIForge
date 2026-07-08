# PyRadiomics (Feature Extraction Tool)

## Description

Open-source, widely adopted Python package for the extraction of radiomic features from medical images (2D and 3D) and binary masks. PyRadiomics implements the feature definitions standardized by the Image Biomarker Standardization Initiative (**IBSI**), ensuring reproducibility. It is the fundamental tool for engineering **Texture**, **Shape**, and **Intensity** (First Order) features in Radiomics projects. The package is actively maintained and is compatible with common medical image formats such as DICOM, NIfTI, and NRRD.

## Statistics

PyRadiomics is a software tool, not a dataset.
*   **Extracted Features:** Approximately 1500 features per image (depending on the configurations and filters applied).
*   **Feature Classes:** 8 main classes (First Order, 3D/2D Shape, GLCM, GLRLM, GLSZM, NGTDM, GLDM).
*   **Standard:** Compliant with the IBSI standard (Image Biomarker Standardization Initiative).
*   **Compatibility:** Supports 2D and 3D images in formats such as DICOM, NIfTI, and NRRD.
*   **Reference Datasets:** Used together with standardized datasets such as **Open-radiomics** (which includes BraTS 2020/2023 and TCIA NSCLC) to ensure reproducibility.

## Features

PyRadiomics extracts a large number of features (approximately 1500 per image, depending on the configurations), categorized into:
*   **First Order Features (Intensity):** 19 features, including Mean, Median, Standard Deviation, Entropy, Skewness, and Kurtosis.
*   **Shape Features:** 16 3D features and 10 2D features, such as Volume, Surface Area, Sphericity, and Compactness.
*   **Texture Features (Higher Order):** Include gray-level co-occurrence matrix features (GLCM - 24 features), gray-level run length matrix features (GLRLM - 16 features), gray-level size zone matrix features (GLSZM - 16 features), neighboring gray tone difference matrix features (NGTDM - 5 features), and gray-level dependence matrix features (GLDM - 14 features).

## Use Cases

*   **Prognosis and Survival Prediction:** Use of radiomic features to predict patient response to treatment and survival across various cancer types (e.g., Non-Small Cell Lung Cancer - NSCLC, Gliomas).
*   **Tumor Classification:** Distinguishing between high- and low-grade tumors (e.g., HGG vs. LGG in the BraTS 2023 dataset) based on texture and intensity characteristics.
*   **Assessment of Tumor Heterogeneity:** Quantification of spatial and intensity variation within a Region of Interest (ROI) to better characterize the tumor phenotype.
*   **Development of Imaging Biomarkers:** Creation of predictive and descriptive models that correlate image features with genomic and clinical data (Radiogenomics).

## Integration

Feature extraction is typically performed via Python, using the `RadiomicsFeatureExtractor` class.

```python
from radiomics import featureextractor
import SimpleITK as sitk

# 1. Initialize the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Optional: Configure the extractor (e.g., disable features, change binWidth)
# extractor.disableAllFeatures()
# extractor.enableFeatureClassByName('FirstOrder')
# extractor.settings['binWidth'] = 25

# 2. Load the image and the mask (ROI)
imageName = 'path/to/image.nrrd'
maskName = 'path/to/mask.nrrd'
image = sitk.ReadImage(imageName)
mask = sitk.ReadImage(maskName)

# 3. Extract the features
result = extractor.execute(image, mask)

# 4. Print the result
print('Extraction complete. Extracted features:')
for featureName in result.keys():
    print(f'  {featureName}: {result[featureName]}')
```

The package also offers a command-line interface (`pyradiomics`) for batch extraction. The example code and data are available in the official GitHub repository.

## URL

https://pyradiomics.readthedocs.io/