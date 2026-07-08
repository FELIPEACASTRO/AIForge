# Agriculture-Vision Dataset

## Description

Agriculture-Vision is an extensive dataset of high-resolution, multi-band aerial imagery, designed for the analysis of agricultural patterns and the detection of anomalies in crop fields. The original dataset (2020) and its subsequent extensions (2021, 2022, 2023) were created for the annual CVPR Agriculture-Vision challenge. It stands out for being multi-band (RGB and Near-Infrared - NIR) and for including detailed annotations of nine types of field anomalies by agronomy experts. The anomalies include: cloud shadow, double plant, drought, endrow, nutrient deficiency, planter skip, storm damage, water, waterway, and weed cluster. The dataset is fundamental for the development of Computer Vision models in Precision Agriculture.

## Statistics

**Original Dataset (2020):**
*   **Total Images:** 94,986 images of 512x512 pixels.
*   **Crop Fields:** Sampled from 3,432 farms in the USA.
*   **Split:** 56,944 (Train) / 18,334 (Validation) / 19,708 (Test).
*   **Channels:** 4 channels (RGB and NIR).
*   **File Size:** The challenge subset (2020) is approximately 4.4 GB.

**Extensions (2021 onward):**
*   **Resolution:** Images of up to 10 cm/pixel.
*   **Additional Data:** Full-field imagery sequences from 52 fields, totaling 261 high-resolution images, for weakly supervised methods.
*   **File Size (2021):** The new 2021 dataset is approximately 20 GB.

## Features

High-resolution aerial imagery (up to 10 cm/pixel); Multi-band (RGB and Near-Infrared - NIR); Semantic segmentation annotations for 9 types of field anomalies; 512x512 pixel images; Includes full-field imagery for weakly supervised methods (from 2021 onward).

## Use Cases

**Agricultural Anomaly Detection:** Precise identification and localization of problems such as planter skips, double plants, nutrient deficiency, and water damage.
**Semantic Segmentation:** Training models to segment different patterns and anomalies in aerial imagery.
**Precision Agriculture:** Support for decision-making to optimize resource use (water, fertilizers) and increase crop productivity.
**Multi-band Model Development:** Research and development of computer vision architectures that use channel information beyond RGB (NIR).

## Integration

The dataset can be accessed and downloaded directly from the Amazon S3 Bucket, without the need for an AWS account, using the AWS CLI with the `--no-sign-request` flag.
Example of accessing the original dataset (2020) via AWS CLI:
```bash
aws s3 ls --no-sign-request s3://intelinair-data-releases/agriculture-vision/cvpr_paper_2020/
```
The dataset is also available on the Hugging Face Hub, allowing access via Python's `datasets` library, although the direct download of the `tar.gz` file is the primary method.
Example of accessing the 2021 dataset via AWS CLI:
```bash
aws s3 ls --no-sign-request s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/
```

## URL

https://www.agriculture-vision.com/
