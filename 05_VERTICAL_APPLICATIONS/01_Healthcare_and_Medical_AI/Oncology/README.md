# Oncology AI

This directory covers AI for cancer research and care: imaging, pathology, genomics, treatment response, trial matching, risk prediction, molecular profiling, and clinical decision support.

## Scope

- Cancer imaging, radiomics, pathology, omics, EHR, treatment, survival, recurrence, toxicity, and biomarker models.
- Dataset provenance, cohort definitions, censoring, treatment leakage, multi-institution validation, and clinical utility.
- Safety and governance for high-stakes decision support.

## Reference Links

- NCI Genomic Data Commons: https://gdc.cancer.gov/
- GDC and AI: https://gdc.cancer.gov/content/gdc-and-artificial-intelligence-ai
- The Cancer Imaging Archive: https://www.cancerimagingarchive.net/
- TCGA program: https://www.cancer.gov/ccg/research/genome-sequencing/tcga
- cBioPortal: https://www.cbioportal.org/
- ClinicalTrials.gov: https://clinicaltrials.gov/

## Routing Rules

- Put radiology-specific cancer imaging in `../Radiology/` or `../Medical_Imaging/`.
- Put radiomics feature extraction in the data-engineering radiomics folder.
- Put drug-discovery oncology work in `../Drug_Discovery/` if mechanism or molecule design is central.
