# Directory Enrichment Run - 2026-07-07

This run starts the repo-wide enrichment pass requested after the broad AI/ML source atlas. The objective is to enrich every directory and subdirectory with topic-appropriate information while preserving the existing AIForge organization.

## Baseline Inventory

Measured from `C:\Users\davis\Workspace\AIForge` on 2026-07-07.

| Metric | Count |
|---|---:|
| Directories excluding `.git` | 580 |
| Directories with `README.md` before Batch 01 | 214 |
| Directories without `README.md` before Batch 01 | 366 |
| Top-level README files | 6 major pillar READMEs plus root documentation |

## Top-Level Distribution

| Top-level path | Subdirectories | Markdown files | README files | Enrichment note |
|---|---:|---:|---:|---|
| `.github` | 2 | 3 | 0 | Operational metadata only; enrich if workflows, issue templates, or policy docs expand. |
| `.well-known` | 0 | 0 | 0 | Machine-readable web metadata; no narrative README needed unless policy grows. |
| `00_FRONTIER_AI_2026` | 0 | 8 | 1 | Already has a compact frontier-AI index. |
| `01_AI_FUNDAMENTALS_AND_THEORY` | 139 | 283 | 70 | Needs stronger hub README plus missing guides for prompt engineering, classical ML, deep learning, math, and evaluation. |
| `02_LLM_AND_AI_MODELS` | 33 | 71 | 18 | Needs routing guides for text LLMs and scientific models. |
| `03_DATASETS_TOOLS_AND_RESOURCES` | 116 | 266 | 18 | Needs dataset, data-engineering, storage, prompt-source, and source-atlas routing. |
| `04_MLOPS_AND_PRODUCTION_AI` | 40 | 72 | 15 | Needs production AI lifecycle map and missing deployment/evaluation guides. |
| `05_VERTICAL_APPLICATIONS` | 237 | 1133 | 92 | Largest enrichment target; healthcare, agriculture, and project showcases are first batch. |
| `docs` | 3 | 4 | 0 | Added documentation hub in this pass. |
| `tools` | 0 | 0 | 0 | Script directory; enrich via docstrings or README only if tool count grows. |

## Batch 01 Targets

| Path | Reason |
|---|---|
| `01_AI_FUNDAMENTALS_AND_THEORY/README.md` | Existing README was too thin for a 139-directory theory pillar. |
| `01_AI_FUNDAMENTALS_AND_THEORY/Prompt_Engineering/README.md` | Large prompt tree with official guides, prompt libraries, prompt ops, and domain prompts. |
| `01_AI_FUNDAMENTALS_AND_THEORY/Deep_Learning/README.md` | Core ML theory area with many subtopics but no local routing guide. |
| `01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/README.md` | Foundation algorithms need clean split across supervised, unsupervised, trees, kernels, and ensembles. |
| `02_LLM_AND_AI_MODELS/Text_LLMs/README.md` | Routes closed, open, and efficient text-model families. |
| `02_LLM_AND_AI_MODELS/Scientific_Models/README.md` | Routes protein, weather, quantum ML, and reservoir-computing models. |
| `03_DATASETS_TOOLS_AND_RESOURCES/Datasets/README.md` | Large dataset tree lacked a modality and provenance guide. |
| `03_DATASETS_TOOLS_AND_RESOURCES/Data_Engineering/README.md` | Needs data lifecycle routing plus prompt files for data workflows. |
| `03_DATASETS_TOOLS_AND_RESOURCES/Storage_and_Databases/README.md` | Needs database and vector-storage routing for AI systems. |
| `04_MLOPS_AND_PRODUCTION_AI/README.md` | Existing README was compact; expanded with lifecycle map and source atlas links. |
| `05_VERTICAL_APPLICATIONS/README.md` | Existing README was too thin for 237 subdirectories and 29 verticals. |
| `05_VERTICAL_APPLICATIONS/01_Healthcare_and_Medical_AI/README.md` | Large medical AI tree with prompts, models, datasets, devices, genomics, imaging, and clinical AI. |
| `05_VERTICAL_APPLICATIONS/03_Agriculture_AgTech/README.md` | Large AgTech/biomass tree with datasets, remote sensing, crop prediction, prompts, and edge AI. |
| `05_VERTICAL_APPLICATIONS/20_AI_Project_Showcases/README.md` | Project showcase tree needs routing by application pattern and evaluation readiness. |

## Enrichment Standard

Each directory guide should include:

1. Purpose and scope.
2. What belongs in the directory.
3. Subdirectory or topic routing.
4. High-authority source families or known local files.
5. Next enrichment tasks.
6. Clear boundaries so content is not duplicated across unrelated directories.

## Batch 01 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 01 | 226 |
| Directories without `README.md` after Batch 01 | 354 |
| New directory guides added | 12 |
| Existing pillar guides expanded | 3 |

Batch 01 established guides for the documentation hub, prompt engineering, deep learning, classical ML algorithms, text LLMs, scientific models, datasets, data engineering, storage/databases, healthcare AI, agriculture AI, and project showcases.

## Batch 02 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 02 | 241 |
| Directories without `README.md` after Batch 02 | 339 |
| New directory guides added in Batch 02 | 15 |

Batch 02 added guides for statistical learning, mathematics for ML, probabilistic ML, generative models, model evaluation, AutoML, feature engineering, multimodal datasets, medical datasets, climate/geospatial datasets, video datasets, open data portals, deployment, AI agents, and predictive AI.

## Batch 03 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 03 | 257 |
| Directories without `README.md` after Batch 03 | 323 |
| New directory guides added in Batch 03 | 16 |

Batch 03 added guides for medical imaging, genomics, drug discovery, agriculture transfer learning, AgTech biomass, business/marketing AI, computer-vision datasets, finance datasets, privacy/security, graph neural networks, machine learning, reinforcement learning, cloud platforms, general databases, vector databases, and A/B testing/canary releases.

## Batch 04 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 04 | 279 |
| Directories without `README.md` after Batch 04 | 301 |
| New directory guides added in Batch 04 | 22 |

Batch 04 added source-linked guides for feature engineering theory, data analysis, social-science datasets, business prompts, creative prompts, model optimization, NLP, data pipelines, feature stores, other AI clouds, medical image segmentation, healthcare transfer learning, healthcare edge model compression, video models, data augmentation, and AI project-showcase subdirectories for MLOps, NLP, ML frameworks, multimodal systems, deep learning, generative AI, and applied ML.

Key source families inserted directly into the new guides include scikit-learn, pandas, SciPy, ICPSR, Harvard Dataverse, World Bank Microdata, OpenAI prompting and video documentation, Anthropic prompting documentation, Google prompting and Gemini documentation, ONNX Runtime, PyTorch, TensorFlow Lite, Hugging Face, spaCy, NLTK, Stanford CoreNLP, Airflow, Kubeflow, Feast, MLflow, MONAI, nnU-Net, TCIA, Albumentations, TorchVision, and Papers with Code.

## Next Batches

| Batch | Focus |
|---|---|
| 05 | Add README guides to remaining high-priority subdirectories under project showcases, healthcare AI, prompt engineering, data engineering, and model catalogs. |
| 06 | Add source-backed enrichment files for prompt libraries, model catalogs, benchmarks, agent frameworks, and evaluation suites. |
| 07 | Continue dataset modality and vertical-domain subdirectories until every meaningful content directory has a local guide. |
