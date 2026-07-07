# Feature Engineering

This directory covers the theory of turning raw variables into model-ready signals. It is the conceptual counterpart to `03_DATASETS_TOOLS_AND_RESOURCES/Data_Engineering/Feature_Engineering/`, which should hold implementation-oriented resources and production data workflows.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Feature_Engineering_Techniques/` | Interactions, aggregations, transforms, binning, target encoding, time-window features, and domain-derived variables. |
| `Feature_Scaling_and_Encoding/` | Standardization, normalization, categorical encodings, missing-value indicators, text/vector encodings, and leakage-safe preprocessing. |
| `Feature_Selection/` | Filter, wrapper, embedded, model-based, permutation, stability, and pipeline-aware selection methods. |

## Core Questions

- What raw signal exists before modeling?
- Which transformation improves learnability without leaking target information?
- Which features must be computed only inside cross-validation folds?
- Which transformations must be identical between training and serving?
- Which features improve robustness, calibration, fairness, or interpretability?

## Primary Source Families

- scikit-learn preprocessing, feature extraction, feature selection, and pipelines.
- pandas, NumPy, SciPy, and statsmodels for tabular transformations and statistical checks.
- domain feature libraries such as tsfresh, PyRadiomics, GeoPandas, rasterio, and text vectorization tools.
- MLOps feature-store documentation when a feature must be reused in production.

## Reference Links

- scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html
- scikit-learn feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
- scikit-learn pipelines: https://scikit-learn.org/stable/modules/compose.html

## Routing Rules

- Put production feature serving, registry, and online/offline consistency in `../../04_MLOPS_AND_PRODUCTION_AI/MLOps_Platforms/feature_store/`.
- Put data-pipeline implementation details in `../../03_DATASETS_TOOLS_AND_RESOURCES/Data_Engineering/Data_Pipelines/`.
- Put evaluation methodology in `../Model_Evaluation/`.
- Put privacy, leakage, or re-identification risk in `../Privacy_and_Security/`.
