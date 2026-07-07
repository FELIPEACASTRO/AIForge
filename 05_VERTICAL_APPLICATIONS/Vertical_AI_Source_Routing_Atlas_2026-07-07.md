# Vertical AI Source Routing Atlas - 2026-07-07

This atlas enriches the vertical applications layer with source families that should be used before adding domain-specific AI files. It keeps healthcare, finance, agriculture, climate, education, legal, robotics, cybersecurity, government, and science sources routed by evidence type.

## Domain Source Families

| Vertical | Primary sources to use | Local routing |
|---|---|---|
| Healthcare and medical AI | [FDA AI/ML-enabled medical devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [PhysioNet](https://physionet.org/), [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) | `01_Healthcare_and_Medical_AI/` |
| Drug discovery and biology | [PubChem](https://pubchem.ncbi.nlm.nih.gov/), [RCSB PDB](https://www.rcsb.org/), [AlphaFold DB](https://alphafold.ebi.ac.uk/), [UniProt](https://www.uniprot.org/) | Healthcare, scientific models, protein structure. |
| Finance and fintech AI | [SEC EDGAR](https://www.sec.gov/edgar), [FRED](https://fred.stlouisfed.org/), [FINRA data](https://www.finra.org/finra-data), [IMF Data](https://www.imf.org/en/Data), [World Bank Data](https://data.worldbank.org/) | `02_Finance_and_Fintech_AI/` |
| Agriculture and AgTech | [FAOSTAT](https://www.fao.org/faostat/), [USDA NASS](https://www.nass.usda.gov/), [CGIAR](https://www.cgiar.org/), [NASA Earthdata](https://www.earthdata.nasa.gov/), [Copernicus Data Space](https://dataspace.copernicus.eu/) | `03_Agriculture_AgTech/` |
| Climate and sustainability | [IPCC reports](https://www.ipcc.ch/reports/), [NOAA Data](https://www.noaa.gov/data), [NASA Earthdata](https://www.earthdata.nasa.gov/), [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) | `04_Climate_and_Sustainability/` |
| Education AI | [UNESCO AI and education](https://www.unesco.org/en/artificial-intelligence/education), [OECD education data](https://www.oecd.org/en/data.html), [World Bank education data](https://data.worldbank.org/topic/education) | `05_Education_AI/` |
| Legal and governance AI | [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework), [OECD.AI](https://oecd.ai/), [UNESCO AI ethics](https://www.unesco.org/en/artificial-intelligence/recommendation-ethics), [EU AI Act](https://artificialintelligenceact.eu/) | `06_Legal_AI/`, government and public sector. |
| Robotics and embodied AI | [Gymnasium](https://gymnasium.farama.org/), [MuJoCo](https://mujoco.readthedocs.io/en/stable/), [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/), [RoboSuite](https://robosuite.ai/) | `10_Robotics_and_Embodied_AI/` |
| Autonomous vehicles | [nuScenes](https://www.nuscenes.org/), [Waymo Open Dataset](https://waymo.com/open/), [Argoverse](https://www.argoverse.org/), [KITTI](https://www.cvlibs.net/datasets/kitti/) | `11_Autonomous_Vehicles_AI/` |
| Cybersecurity AI | [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework), [CISA Known Exploited Vulnerabilities](https://www.cisa.gov/known-exploited-vulnerabilities-catalog), [MITRE ATT&CK](https://attack.mitre.org/), [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | `14_Cybersecurity_AI/` |
| Science AI | [arXiv](https://arxiv.org/), [PMLR](https://proceedings.mlr.press/), [NASA Earthdata](https://www.earthdata.nasa.gov/), [RCSB PDB](https://www.rcsb.org/), [AlphaFold DB](https://alphafold.ebi.ac.uk/) | `15_Science_AI/` |
| Government and public sector AI | [OECD AI Policy Observatory](https://oecd.ai/), [UN e-Government Knowledgebase](https://publicadministration.un.org/egovkb/en-us/), [Data.gov](https://data.gov/), [data.europa.eu](https://data.europa.eu/) | `28_Government_and_Public_Sector_AI/` |

## Evidence Requirements By Vertical

| Evidence class | Required fields |
|---|---|
| Dataset | Publisher, license, population/sample, update date, schema, privacy risk, and intended use. |
| Model | Model card, task, modality, validation cohort, benchmark, failure modes, and deployment constraints. |
| Regulation | Jurisdiction, authority, date, scope, compliance obligation, and relation to AI system lifecycle. |
| Paper | Venue, peer-review status, code/data availability, metrics, baseline, and reproducibility caveats. |
| Tool/vendor | Official docs, license/terms, deployment model, security controls, pricing/cost notes, and lock-in risk. |

## Routing Rules

- Keep official regulators and public bodies as the first source for legal, medical, financial, and government claims.
- Put domain datasets in the matching `03_DATASETS_TOOLS_AND_RESOURCES/Datasets/` folder and cross-link to the vertical application folder.
- Put model cards under `02_LLM_AND_AI_MODELS/` or scientific model folders, then cross-link to the vertical use case.
- Put operational guidance such as serving, monitoring, audits, and guardrails under `04_MLOPS_AND_PRODUCTION_AI/`.
