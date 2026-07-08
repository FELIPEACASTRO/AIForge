# MIMIC-III Clinical Database v1.4

## Description

The **MIMIC-III (Medical Information Mart for Intensive Care III)** is an extensive, freely accessible, de-identified database that contains health information related to more than 40,000 patients who stayed in intensive care units (ICU) at the Beth Israel Deaconess Medical Center between 2001 and 2012 [1]. The dataset is a foundational resource for research in medical informatics and artificial intelligence in healthcare. It supports a wide range of analytical studies, including epidemiology, improvement of clinical decision rules, and development of electronic tools [1]. MIMIC-III is notable for its free availability to researchers worldwide, for covering a large and diverse population of ICU patients, and for containing highly granular data such as vital signs, laboratory results, and medications [1]. Although the most recent version dates from 2016 (v1.4), it continues to be widely used in recent research (2023-2025), often in conjunction with its successor, MIMIC-IV [2] [3].

## Statistics

*   **Patients:** 46,520 unique patients.
*   **Hospital Admissions:** 58,976 unique admissions.
*   **ICU Stays:** 61,532 unique ICU stays.
*   **Collection Period:** 2001 to 2012.
*   **Structure:** Relational database with 26 tables.
*   **Size:** Approximately 40 GB (raw data) [1].
*   **Granularity:** Contains high-frequency data (hourly vital signs) and sparse data (laboratory results, notes) [1].

## Features

MIMIC-III is a relational database composed of 26 tables, interconnected by identifiers such as `SUBJECT_ID` (unique patient), `HADM_ID` (unique hospital admission), and `ICUSTAY_ID` (unique ICU admission) [1]. The data are categorized into:
*   **Demographic and Administrative Data:** Patient information, admissions, transfers, and ICU stays.
*   **Monitoring Events:** Vital signs, continuous monitoring events, and input/output events (fluids, medications).
*   **Laboratory Results:** Results of hematology, chemistry, and microbiology tests.
*   **Billing and Coding Information:** ICD-9 codes (diagnoses and procedures), DRG, and CPT.
*   **Free-Text Notes:** Caregiver notes, discharge summaries, and imaging reports (de-identified) [1].

**Recent Feature Engineering Techniques (2023-2025):**
Recent research focuses on advanced techniques to extract value from the raw MIMIC-III data:
1.  **Two-tier Feature Selection:** Used to predict hospital mortality, combining selection methods to optimize *stacking* models [4].
2.  **Time-Series Feature Extraction:** Use of *Deep Learning* (such as LSTMs) to capture temporal and spatial features from ECG signals and vital sign data for the prediction of clinical outcomes [5] [6].
3.  **Free-Text Feature Extraction:** Application of *Deep Learning* models and natural language processing (NLP) to extract features from clinical notes for automated ICD code classification [7].
4.  **Deep Reinforcement Learning (DRL):** Used to identify optimal treatment strategies, such as in sepsis management, where DRL extracts the most relevant state features for decision-making [8].

## Use Cases

MIMIC-III is the foundation for a vast range of applications in AI research and development in healthcare:
*   **Mortality Prediction:** *Machine Learning* (ML) and *Deep Learning* models to predict hospital and 30-day mortality in ICU patients, including those with specific conditions such as sepsis and heart failure [4] [9].
*   **ICU Readmission Prediction:** Development of models to identify patients at high risk of readmission [2].
*   **Early Detection of Critical Conditions:** Use of *Ensemble* models to improve the early detection of conditions such as acute kidney injury (AKI) in septic patients [10].
*   **Treatment Optimization:** Application of *Deep Reinforcement Learning* to derive optimal treatment strategies, such as in sepsis management [8].
*   **Automatic ICD Code Classification:** Use of NLP on clinical notes to automate the coding of diagnoses and procedures [7].
*   **Physiological Signal Analysis:** Extraction of features from ECG signals and vital signs for classifying diseases such as heart failure [5].

## Integration

Access to MIMIC-III requires the researcher to complete a training course in human data protection (for example, the CITI course) and sign a Data Use Agreement (DUA) on PhysioNet [1].

**Access and Integration:**
1.  **PhysioNet:** After credentialing, the data files can be downloaded directly from PhysioNet.
2.  **Cloud Platforms:** MIMIC-III is available on the **Google Cloud Platform (GCP)** via BigQuery and on **Amazon Web Services (AWS)** via Athena, enabling scalable queries and analyses directly in the cloud [1].

**Access Example (Conceptual - BigQuery/SQL):**
Integration typically involves complex SQL queries to join the 26 tables and extract the desired cohorts and features.

```sql
-- Example query to obtain age and ICU length of stay
SELECT
    p.subject_id,
    EXTRACT(YEAR FROM p.dob) - EXTRACT(YEAR FROM a.admittime) AS age_at_admission,
    ROUND(CAST(ie.los AS NUMERIC), 2) AS icu_los_days
FROM
    physionet-data.mimiciii_clinical.patients p
INNER JOIN
    physionet-data.mimiciii_clinical.admissions a ON p.subject_id = a.subject_id
INNER JOIN
    physionet-data.mimiciii_clinical.icustays ie ON a.hadm_id = ie.hadm_id
WHERE
    a.admittime IS NOT NULL
LIMIT 10;
```
*Note: This is a conceptual SQL example for BigQuery. Actual access requires credentials and cloud configuration [1].*

## URL

https://physionet.org/content/mimiciii/