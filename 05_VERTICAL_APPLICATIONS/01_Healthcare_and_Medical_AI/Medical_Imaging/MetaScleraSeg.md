# MetaScleraSeg

## Description

MetaScleraSeg is a meta-learning framework designed for generalized segmentation of the sclera (the white part of the eye). It addresses the challenge of the lack of generalization in traditional deep learning models to new data domains (such as different ethnicities, image qualities, or datasets). The framework uses a meta-sampling strategy to simulate domain variations and a style-invariant UNet 3+ base model, optimized through a bilevel optimization strategy to learn transferable knowledge across domains. It was published in 2023.

## Statistics

- **Publication:** Neural Computing and Applications (2023).
- **Citations:** 11 (as of August 2023, according to arXiv).
- **Performance:** Demonstrated superiority compared to baseline models in domain generalization protocols (cross-dataset, cross-ethnicity, cross-quality).
- **Typical Metric:** The F1-score (or Dice Score) for sclera segmentation on datasets such as SBVPI generally reaches values above 96% for state-of-the-art methods. MetaScleraSeg outperforms these baselines in unseen-domain scenarios.

## Features

- **Meta-Sampling:** Strategy to simulate domain variations (domain shifts) in real-world scenarios.
- **Style-Invariant Base Model:** Uses a modified UNet 3+ architecture to ensure the model focuses on essential features, ignoring style variations.
- **Bilevel Optimization:** Employs a meta-optimization strategy to update the base model, enabling it to generalize well to unseen target domains.
- **Robust Generalization:** Designed to work across cross-validation protocols (cross-dataset, cross-ethnicity, and cross-quality).

## Use Cases

- **Ocular Biometrics:** Precise sclera segmentation for identity recognition systems.
- **Generalized Ocular Diagnosis:** Creation of AI models that can be deployed across different clinics or geographic regions, handling variations in equipment and population (ethnicity).
- **Few-Shot Learning:** Application in scenarios where new medical data domains have few labeled samples.

## Integration

The official project code is available on GitHub, implemented in PyTorch.
**Execution Example (Test):**
```bash
# Clone the repository
git clone https://github.com/lhqqq/MetaScleraSeg.git
cd MetaScleraSeg

# Install dependencies (assuming Python and PyTorch environment configured)
# pip install -r requirements.txt (if available)

# Run the test script (requires the dataset and pre-trained models)
python test.py
```
**Note:** The dataset (CDSS) and pre-trained models are provided via Baidu Drive, as described in the repository's README.

## URL

https://github.com/lhqqq/MetaScleraSeg
