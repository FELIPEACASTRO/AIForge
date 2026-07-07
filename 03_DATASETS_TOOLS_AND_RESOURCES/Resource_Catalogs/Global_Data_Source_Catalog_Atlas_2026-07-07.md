# Global Data Source Catalog Atlas - 2026-07-07

This atlas expands AIForge's data coverage beyond single datasets. It maps high-authority catalog sources for machine learning, public-sector data, science data, geospatial data, health data, economic data, and cloud-hosted public datasets into the repository's existing directories.

## Machine Learning And AI Dataset Hubs

| Source | What to capture | Local routing |
|---|---|---|
| [Hugging Face Datasets](https://huggingface.co/datasets) | Dataset cards, licenses, splits, tasks, languages, viewer availability, and files. | `Datasets/`, `HuggingFace_Hub/` |
| [Kaggle Datasets](https://www.kaggle.com/datasets) | Dataset pages, notebooks, competitions, licenses, and usage evidence. | `Datasets/`, challenge-specific folders. |
| [OpenML](https://www.openml.org/) | Datasets, tasks, runs, flows, benchmark suites, and reproducibility metadata. | `Datasets/Machine_Learning/`, `Model_Evaluation/` |
| [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | Classic and maintained ML datasets with task metadata. | `Datasets/Famous_Benchmarks/`, `Datasets/Machine_Learning/` |
| [Papers with Code datasets](https://paperswithcode.com/datasets) | Dataset-paper-code links and task associations. | Benchmark and task-specific dataset folders. |

## Public-Sector And International Data Portals

| Source | What to capture | Local routing |
|---|---|---|
| [Data.gov](https://data.gov/) | U.S. government datasets, tools, organizations, and geospatial resources. | `Datasets/Government_Open_Data/` when present; otherwise `Datasets/` by domain. |
| [data.europa.eu](https://data.europa.eu/) | European Union open-data portal records and cross-country data. | Global, policy, climate, transport, economic, and public-sector AI. |
| [World Bank Open Data](https://data.worldbank.org/) | Development indicators, country data, APIs, and Data360 migration notes. | Finance, development, education, health, climate. |
| [UNdata](https://data.un.org/) | United Nations statistical databases and country indicators. | Global AI ecosystem and policy context. |
| [OECD Data Explorer](https://data-explorer.oecd.org/) | OECD indicators, economies, education, productivity, innovation, and digital economy data. | Country matrix, economic AI, policy analysis. |
| [IMF Data](https://www.imf.org/en/Data) | Macroeconomic and financial data series. | Finance, risk, macro modeling. |
| [WHO Global Health Observatory](https://www.who.int/data/gho) | Health indicators, mortality, disease, and country health profiles. | Healthcare and public-health AI. |
| [FAOSTAT](https://www.fao.org/faostat/) | Food, agriculture, land, emissions, and production statistics. | Agriculture, climate, sustainability, food security. |

## Scientific, Geospatial, And Cloud Public Data

| Source | What to capture | Local routing |
|---|---|---|
| [NASA Earthdata](https://www.earthdata.nasa.gov/) | Earth-observation data, missions, DAACs, tutorials, and APIs. | `Climate_and_Geospatial/`, AgTech, climate AI. |
| [NOAA Data](https://www.noaa.gov/data) | Weather, ocean, climate, fisheries, satellites, and environmental data. | Climate, weather, energy, agriculture. |
| [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) | Sentinel data, APIs, browser access, and European Earth-observation resources. | Climate, geospatial, agriculture, disaster response. |
| [AWS Open Data Registry](https://registry.opendata.aws/) | Cloud-hosted public datasets with S3 locations and usage examples. | Cloud public data and domain datasets. |
| [Google Cloud BigQuery public datasets](https://docs.cloud.google.com/bigquery/public-data) | Queryable public datasets hosted in BigQuery. | Data engineering, analytics, finance, public datasets. |
| [Google Dataset Search](https://datasetsearch.research.google.com/) | Dataset discovery across publishers and repositories. | Lead discovery; verify against original publisher. |

## Catalog Intake Requirements

| Field | Requirement |
|---|---|
| Publisher | Organization, maintainer, country/region, and official contact if available. |
| Dataset identity | Name, stable URL, version/date, license, citation, and access method. |
| ML readiness | Task, labels, split, leakage risks, missingness, format, scale, and update cadence. |
| Governance | Privacy, personal data, sensitive attributes, terms of use, and redistribution limits. |
| Local owner | Exact AIForge dataset/domain directory and downstream model/evaluation directories. |

## Routing Notes

- Put broad portals here first, then promote specific datasets into domain folders only after license and schema checks.
- Treat search engines and catalogs as discovery aids; final evidence should point back to the original publisher.
- For country-level AI work, connect public data sources to `Global_AI_Ecosystem/` only when they support policy, readiness, adoption, skills, compute, or economic analysis.
