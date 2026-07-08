# PI-CAI Public Training and Development Dataset (ADC)

## Description

PI-CAI (Prostate Cancer Artificial Intelligence) is a large-scale challenge that succeeded PROSTATEx, focused on the detection and diagnosis of clinically significant prostate cancer (csPCa) using multiparametric magnetic resonance imaging (mpMRI). The public training and development dataset contains 1500 mpMRI exams, including T2-weighted (T2W) data, DWI (Diffusion-Weighted Imaging), and Apparent Diffusion Coefficient (ADC) maps. Although the public dataset does not include Ktrans (Volume Transfer Constant) or SUV (Standardized Uptake Value), it is fundamental for imaging biomarker research, since ADC is one of the three requested biomarkers and the dataset is the most recent and robust in its domain. The PI-CAI challenge is an essential resource for the development of AI algorithms in radiology.

## Statistics

**Public Dataset Size:** 1500 mpMRI exams. **Total Cohort Size:** More than 10,000 exams. **Origin:** Multi-center (4 centers in the Netherlands and Norway) and multi-vendor (Siemens and Philips). **Sequences:** T2W, DWI (high b-value), ADC. **License:** CC BY-NC 4.0. **Status:** Active and in use for cutting-edge research (2023-2025).

## Features

**Included Imaging Biomarkers:** Apparent Diffusion Coefficient (ADC). **Imaging Sequences:** T2-weighted (T2W), DWI (high b-value). **Labels:** csPCa annotations (clinically significant prostate cancer) and basic clinical information (patient age, prostate volume, PSA, PSA density). **Characteristics:** Multi-center and multi-vendor dataset, with 1500 public cases for training and development. The total dataset includes more than 10,000 exams. Does not include Ktrans or SUV in the public training set for AI algorithms.

## Use Cases

**AI Algorithm Development:** Training and validation of *Deep Learning* models for the detection and diagnosis of clinically significant prostate cancer (csPCa). **Radiomics:** Extraction of radiomic features from ADC maps for Gleason score prediction and treatment response. **Human-Machine Comparison:** Benchmarking the performance of AI algorithms against experienced radiologists. **Biomarker Research:** Study of the usefulness of ADC as a quantitative biomarker in mpMRI.

## Integration

The public training and development dataset (1500 cases) is available via Zenodo and the annotations via GitHub.

**Dataset Access:**
`zenodo.org/record/6624726` (DOI: 10.5281/zenodo.6624726)

**Annotation (Labels) Access:**
`github.com/DIAGNijmegen/picai_labels`

**Feature Extraction Example (Conceptual, using PyRadiomics for ADC):**
The extraction of radiomic features from ADC images (and Ktrans/SUV, if available in other datasets) can be performed with the PyRadiomics library.

```python
from radiomics import featureextractor
import SimpleITK as sitk

# 1. Load the ADC image and the segmentation mask
# image_path should be the path to the ADC .mha file
# mask_path should be the path to the segmentation mask .mha file
image = sitk.ReadImage("path/to/adc_image.mha")
mask = sitk.ReadImage("path/to/segmentation_mask.mha")

# 2. Configure the feature extractor
# A YAML parameter file can be used to specify which features to extract
extractor = featureextractor.RadiomicsFeatureExtractor()

# 3. Run the extraction
result = extractor.execute(image, mask)

# 4. Print the extracted features
print("ADC Radiomic Feature Extraction:")
for key, val in result.items():
    print(f"\t{key}: {val}")
```

## URL

https://pi-cai.grand-challenge.org/