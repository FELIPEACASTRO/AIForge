# MIMIC-IV-Ext-22MCTS: A 22 Millions-Event Temporal Clinical Time-Series Dataset with Relative Timestamp

## Description

MIMIC-IV-Ext-22MCTS is a temporal clinical event time-series dataset with concrete temporal information. It is derived from **MIMIC-IV-Note**, which contains de-identified clinical notes. This dataset was created to facilitate the modeling of temporal clinical events, extracting clinical events as short text spans and their respective *timestamps* relative to discharge summaries, using contextual retrieval techniques and the Llama-3.1-8B model. The *timestamp* is given in hours, being negative for historical events (before admission) and positive for events during or after admission. The dataset also includes a `Time_bin` column that discretizes time into 9 predefined categories. It is a valuable resource for pre-training and semi-supervised or weakly-supervised learning tasks.

## Statistics

**Total Discharge Summaries:** 267,284
**Total Records (Event-Timestamp Pairs):** 22,588,586
**Events per Summary (Min/Max):** 1 / 244
**Events per Summary (Average):** 84
**Temporal Distribution of Events:**
- Before admission (Historical): 36.99%
- During admission: 51.19%
- After discharge (Future): 11.80%
**Tokens per Event (Average):** 3
**Tokens per Event (Max):** 299
**Publication:** September 2025 (Version 1.0.0)

## Features

**Temporal Clinical Event Data:** Consists of clinical events extracted from discharge summaries, each associated with a relative *timestamp*.
**Relative Timestamp:** Time is measured in hours, relative to the moment of admission.
**Temporal Discretization (`Time_bin`):** Continuous time is mapped into 9 discrete categories (Bins), facilitating temporal modeling.
**Source:** Derived from the MIMIC-IV-Note dataset, ensuring grounding in real and de-identified clinical data.
***Fine-tuning* Applications:** Used for *fine-tuning* models such as BERT and GPT-2 for Q&A tasks and clinical trial matching.

## Use Cases

**Pre-training of Language Models (LLMs):** Ideal for pre-training models such as GPT-2 to generate more clinically oriented outputs.
**Temporal Clinical Event Modeling:** Used to develop and test models that predict the sequence and timing of clinical events.
**Clinical Trial Matching:** *Fine-tuning* models (e.g., BERT) to improve the matching of patients with clinical trial criteria.
**Clinical Risk Prediction:** Although with limitations for high-risk assessment due to the lack of *Ground Truth* labels, it is useful for predictive modeling and semi-supervised learning tasks.

## Integration

The dataset is available on PhysioNet and requires **Credentialed Access** for download, following PhysioNet's usage policies.
Access generally involves completing a training course on human data protection and signing a Data Use Agreement (DUA).
**Table Structure:**
- `Hadm_id`: Unique identifier for each discharge summary.
- `Event`: The clinical event in text format.
- `Time`: The event *timestamp* in hours (continuous).
- `Time_bin`: The discrete category of the *timestamp* (0 to 8).
The related code for *fine-tuning* models (such as BERT) to explore the causal relationship between clinical events is **publicly available** (although the specific URL was not provided on the summary page).

## URL

https://physionet.org/content/mimic-iv-ext-22mcts/