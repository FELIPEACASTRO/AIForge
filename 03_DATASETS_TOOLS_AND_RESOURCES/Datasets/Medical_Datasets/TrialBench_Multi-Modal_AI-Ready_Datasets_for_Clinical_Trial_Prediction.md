# TrialBench: Multi-Modal AI-Ready Datasets for Clinical Trial Prediction

## Description

**TrialBench** is a comprehensive set of 23 AI-ready *datasets*, curated from more than 480,000 clinical trial records from ClinicalTrials.gov (data through February 2024), enriched with information from DrugBank and TrialTrove. The goal is to facilitate the development of Artificial Intelligence models to predict critical events and optimize clinical trial design. The *datasets* are multimodal, including tabular data, free text (eligibility criteria), and graph data (molecular structure of drugs), and are organized around 8 crucial prediction tasks. It was published in 2025, ensuring its relevance and timeliness.

## Statistics

TrialBench is composed of 23 *datasets* derived from more than **480,000** clinical trial records (through Feb 2024). The following table summarizes the number of trials (in thousands) per task:
| Tasks | # Trials (Total) | # Trials (Phase I/II/III/IV) |
| :--- | :--- | :--- |
| Duration Prediction | 143.8K | 13.5K/13.4K/9.2K/7.1K |
| Dropout Prediction | 62.1K | 4.2K/15.8K/11.5K/6.9K |
| Serious Adverse Event Prediction | 31.3K | 2.0K/8.1K/4.8K/2.9K |
| Mortality Prediction | 31.3K | 2.0K/8.1K/4.8K/2.9K |
| Approval Prediction | 43.2K | 4.5K/12.5K/9.2K/4.5K |
| Failure Reason Identification | 41.4K | 4.3K/8.8K/4.2K/3.5K |
| Eligibility Criteria Design | 136.4K | 19.4K/14.2K/10.8K/10.6K |
| Dosage Determination | 12.8K | 0/12.8K/0/0 |
The full set includes 40.8K drug trials, 21.1K medical device trials, and 83.6K other-intervention trials.

## Features

A set of 23 AI-ready *datasets*, covering 8 prediction tasks: trial duration, patient dropout rate, serious adverse event, mortality, approval outcome, failure reason, drug dosage, and eligibility criteria design. The *datasets* are **multimodal**, incorporating: 1) **Categorical and Numerical Features** (e.g., study type, age); 2) **Free Text** (e.g., eligibility criteria, trial summary); 3) **Graph Data** (e.g., molecular structure of drugs via SMILES); 4) **MeSH Terms** and **ICD-10 Codes** for diseases.

## Use Cases

The main use case is the **optimization of clinical trial design** through the application of AI. This includes:
1.  **Risk Prediction:** Estimating trial duration, the probability of patient dropout, and the occurrence of serious adverse events or mortality.
2.  **Outcome Optimization:** Predicting the trial's approval outcome and identifying the likely reasons for failure.
3.  **Decision Support:** Assisting in determining the optimal drug dosage and designing more effective eligibility criteria.
The resource is ideal for researchers and data scientists working with *Machine Learning* and *Deep Learning* in **Medical Informatics** and **AI in Healthcare**.

## Integration

Integration is facilitated by dedicated packages in **Python** (`pip install trialbench`) and **R**. The Python package enables downloading and loading the *datasets* in formats optimized for *Deep Learning* (DL) or as Pandas *DataFrames*.
**Access Example (Python):**
```python
import trialbench
# Download all datasets (optional)
trialbench.function.download_all_data('data/')
# Load data for a specific task (e.g., dosage)
task = 'dose'
phase = 'All'
# Dataloader format (for DL)
train_loader, valid_loader, test_loader, num_classes, tabular_input_dim = trialbench.function.load_data(task, phase, data_format='dl')
# Or as a Pandas DataFrame
train_df, valid_df, test_df, num_classes, tabular_input_dim = trialbench.function.load_data(task, phase, data_format='df')
```

## URL

https://www.nature.com/articles/s41597-025-05680-8