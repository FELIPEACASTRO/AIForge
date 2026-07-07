# Radiology AI

This directory covers AI for radiology workflows: acquisition, triage, detection, segmentation, reporting, follow-up, PACS/RIS integration, quality assurance, and regulated clinical deployment.

## Scope

- X-ray, CT, MRI, ultrasound, mammography, fluoroscopy, nuclear medicine, and multimodal imaging.
- DICOM/DICOMweb workflows, de-identification, routing, annotations, and imaging metadata.
- Detection, classification, localization, segmentation, report generation, worklist prioritization, and quality control.
- Clinical validation, bias analysis, monitoring, device regulation, and radiologist-in-the-loop review.

## Reference Links

- ACR Data Science Institute: https://www.acr.org/Data-Science-and-Informatics/ACR-Data-Science-Institute
- ACR DSI / AI-LAB: https://www.acrdsi.org/index.html
- DICOM standard: https://www.dicomstandard.org/
- DICOMweb: https://www.dicomstandard.org/using/dicomweb
- RSNA AI challenges: https://www.rsna.org/artificial-intelligence/ai-image-challenge
- RSNA medical imaging data: https://www.rsna.org/artificial-intelligence/data
- FDA AI-enabled medical devices: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices

## Routing Rules

- Put segmentation-specific resources in `../Medical_Imaging/Segmentation/`.
- Put regulatory medical-device material in `../AI_Medical_Devices/`.
- Put DICOM deployment and model-serving pipelines in the MLOps section when they are platform-focused.
