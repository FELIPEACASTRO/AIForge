# Data Analysis

This directory covers practical analysis of datasets before, during, and after machine-learning work: exploration, profiling, validation, visualization, statistical testing, and evaluation dashboards.

## Content Map

| Subdirectory | Scope |
|---|---|
| `Evaluation_Frameworks/` | Analysis frameworks for model outputs, datasets, human review, error slicing, and regression tracking. |
| `Public_Datasets/` | Example datasets used for exploratory analysis, benchmarks, reproducible demos, and tutorials. |
| `Visualization_and_BI_Tools/` | BI, notebooks, dashboards, plots, reporting, semantic layers, and stakeholder-facing analysis. |

## What Belongs Here

- Exploratory data analysis templates.
- Data profiling and data-quality reports.
- Statistical summaries, drift summaries, and cohort analysis.
- Visualization patterns for ML stakeholders.
- Analysis notebooks tied to public or reusable datasets.

## Primary Source Families

- pandas, Polars, DuckDB, SciPy, statsmodels, and scikit-learn examples.
- Great Expectations, Evidently, Deepchecks, whylogs, and TFDV for validation and drift.
- Plotly, Matplotlib, Seaborn, Altair, Superset, Metabase, and BI platform references.

## Reference Links

- pandas: https://pandas.pydata.org/
- SciPy statistics: https://docs.scipy.org/doc/scipy/reference/stats.html
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- Evidently: https://docs.evidentlyai.com/
- Great Expectations: https://docs.greatexpectations.io/

## Routing Rules

- Put pipeline orchestration in `../Data_Pipelines/`.
- Put feature construction in `../Feature_Engineering/`.
- Put raw dataset catalogs in `../../Datasets/`.
- Put production monitoring in `../../../04_MLOPS_AND_PRODUCTION_AI/`.
