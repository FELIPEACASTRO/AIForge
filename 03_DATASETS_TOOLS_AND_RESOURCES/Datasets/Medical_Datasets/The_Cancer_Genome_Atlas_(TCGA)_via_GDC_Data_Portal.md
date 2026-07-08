# The Cancer Genome Atlas (TCGA) via GDC Data Portal

## Description

The Cancer Genome Atlas (TCGA) is a comprehensive, coordinated program to map the genomic alterations of cancer. The data is accessed and harmonized through the Genomic Data Commons (GDC) Data Portal, which provides a computational platform for cancer researchers. The GDC harmonizes clinical and genomic data, including whole genome sequencing (WGS), whole exome sequencing (WES), RNA-Seq, miRNA-Seq, methylation, and clinical data.

## Statistics

Data from Data Release 44.0 (October 2025): 88 Projects, 69 Primary Sites, 48,763 Cases, 1,223,539 Files, 22,580 Genes, 3,082,397 Mutations. The data is continuously updated and expanded with new projects such as APOLLO-OV and CCDI-MCI.

## Features

Harmonized multi-omics data (genomics, transcriptomics, epigenomics, proteomics) and standardized clinical data. Includes somatic variants, gene expression, RNA-Seq counts, methylation data, and slide images (WSIs).

## Use Cases

Identification of new biomarkers and therapeutic targets, classification of cancer subtypes, survival and prognosis studies, and development of machine learning models for predicting drug response and disease progression.

## Integration

Access via the GDC Data Portal (web interface) or the GDC API. Tools such as TCGADownloadHelper (Python) simplify extraction and preprocessing. The GDC Data Transfer Tool is recommended for downloading large volumes of data. Example use of TCGADownloadHelper:

```python
# Installation (example)
# pip install TCGADownloadHelper

from TCGADownloadHelper import TCGADownloadHelper

# Initialize the helper
tcga_helper = TCGADownloadHelper(project_name='TCGA-BRCA', data_type='RNA-Seq', file_type='counts')

# Download the data
tcga_helper.download_data()

# Prepare the data for analysis (example)
df = tcga_helper.prepare_data()
print(df.head())
```

## URL

https://portal.gdc.cancer.gov/