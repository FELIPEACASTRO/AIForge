# MIMIC-IV v3.1

## Description

The Medical Information Mart for Intensive Care IV (MIMIC-IV) is a de-identified, freely accessible database derived from electronic health records (EHR) of the Beth Israel Deaconess Medical Center. Version 3.1 is the most recent, published in 2024, and is widely used for research in medical informatics and AI, providing rich and complex data from ICU and emergency patients. It is the primary data source for the development of AI models in critical care.

## Statistics

Unique Patients: 364,627; Hospitalizations: 546,028; ICU Stays: 94,458 (for more than 65,000 patients); Publication: October 2024 (v3.1). Structure organized into `hosp` (general hospital data) and `icu` (ICU data) modules.

## Features

Includes demographic data, comorbidities, diagnoses (ICD), procedures, prescriptions, medication administration, microbiology results, and, crucially for this topic, **vital signs** (`chartevents`) and **laboratory results** (`labevents`). The vital signs include heart rate, blood pressure, oxygen saturation, respiratory rate, and temperature.

## Use Cases

Hospital mortality prediction, early sepsis detection, modeling of chronic disease progression, development of personalized risk models, and research on feature engineering of clinical time series.

## Integration

Access via PhysioNet (after credentialing and signing a data use agreement). The data are provided in tabular format (CSV) and can be loaded into databases such as PostgreSQL or BigQuery (MIMIC-IV v3.1 is available on BigQuery).

## URL

https://physionet.org/content/mimiciv/