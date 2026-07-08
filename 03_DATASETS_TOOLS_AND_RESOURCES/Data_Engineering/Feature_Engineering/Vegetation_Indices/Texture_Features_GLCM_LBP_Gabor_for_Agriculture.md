# Texture Features (GLCM, LBP, Gabor) for Agriculture and Livestock

## Description

The core resource is the set of texture features (GLCM, LBP, Gabor) and their application to problems in Agriculture, Livestock, and Biomass, as demonstrated in recent studies (2024-2025). The Gray-Level Co-occurrence Matrix (GLCM) is the most prominent, providing second-order statistics such as Homogeneity, Contrast, Entropy, Energy, Correlation, Mean, Variance, Dissimilarity, and Second Moment. The Local Binary Pattern (LBP) and Gabor Filters are used to capture texture information at different scales and orientations, and are frequently combined with Deep Learning techniques to improve interpretability and performance in tasks such as soil texture classification and biomass estimation.

## Statistics

**GLCM:** Typically generates a feature vector of 8 to 14 dimensions per analysis window (e.g., 3x3, 5x5, 7x7, up to 21x21). In a case study (2025), the combination of 8 GLCM features across 4 bands and 10 window sizes resulted in 320 texture features.
**LBP:** The number of features varies with the parameters (radius and number of points), but is typically a histogram vector of 59 or 256 dimensions.
**Example Dataset (2025):** Soil Texture Image Dataset with 4,000 labeled images across 5 texture classes.

## Features

**GLCM (Gray-Level Co-occurrence Matrix):** Extracts 8 to 14 second-order statistics (e.g., Homogeneity, Contrast, Entropy) that describe the spatial relationship between pixels.
**LBP (Local Binary Patterns):** Robust and efficient texture descriptor that captures texture information at macro and micro levels.
**Gabor Filters:** Band-pass filters that use kernels with variable parameters (gamma, theta, lambda, phi) to enhance specific texture and orientation patterns.
**Integration:** Used as hand-crafted features in Machine Learning and Deep Learning frameworks (e.g., ATFEM, Random Forest) to complement learned features.

## Use Cases

**Biomass Estimation:** Used to estimate Aboveground Biomass (AGB) in plantations (e.g., rubber) from remote sensing images (multispectral UAV).
**Soil Texture Classification:** Classification of different soil texture types (e.g., Loamy Sand, Sandy Clay) from soil images.
**Mapping of Soil Physical-Chemical Properties:** Estimation of characteristics such as moisture content, organic carbon content, and other soil properties.
**Detection of Diseases and Stress in Crops:** Texture analysis can identify subtle changes in foliage or soil that indicate stress or disease.

## Integration

GLCM and LBP feature extraction is commonly performed using image processing libraries in Python, such as **`scikit-image`** (`feature.graycomatrix` and `feature.local_binary_pattern` modules) and **`OpenCV`**.
**GLCM Example (Python - scikit-image):**
```python
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np

# Input image (converted to grayscale)
image = rgb2gray(input_image) * 255
image = image.astype(np.uint8)

# Compute GLCM
glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

# Extract properties (features)
contrast = graycoprops(glcm, 'contrast')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')
# Other features such as Dissimilarity, ASM (Second Moment), etc.
```
Commercial tools such as **ENVI 5.3** (Co-occurrence Measures) are also used for GLCM extraction in remote sensing.

## URL

https://www.nature.com/articles/s41598-025-17384-5