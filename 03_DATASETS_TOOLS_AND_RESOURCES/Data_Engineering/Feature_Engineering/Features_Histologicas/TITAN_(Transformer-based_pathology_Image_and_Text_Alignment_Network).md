# TITAN (Transformer-based pathology Image and Text Alignment Network)

## Description

A multimodal whole-slide foundation model for pathology, pre-trained using 335,645 whole-slide images (WSIs) through visual self-supervised learning and vision-language alignment with corresponding pathology reports and 423,122 synthetic captions. It extracts general-purpose slide representations and generates pathology reports without the need for fine-tuning or clinical labels.

## Statistics

Pre-trained on 335,645 WSIs across 20 organ types. Uses 423,122 fine-grained synthetic ROI (Region of Interest) captions and 183 thousand pathology reports for vision-language fine-tuning. Encodes millions of high-resolution ROIs (8,192 × 8,192 pixels at 20× magnification).

## Features

Multimodal Alignment (Image and Text). Whole-slide representation learning. Zero-shot classification. Cross-modal retrieval (histological slides and clinical reports). Pathology report generation. Outperforms ROI and slide foundation models across various tasks.

## Use Cases

General-purpose slide representation learning, cancer subtyping, biomarker prediction, outcome prognosis, slide retrieval, rare cancer retrieval, language-guided zero-shot classification.

## Integration

The model is a foundation model, suggesting that its features can be extracted and used in downstream tasks. The paper mentions that it can be applied 'off-the-shelf' for clinical outcome prediction. Additional details about code availability should be consulted in the 'Code availability' section of the paper.

## URL

https://www.nature.com/articles/s41591-025-03982-3