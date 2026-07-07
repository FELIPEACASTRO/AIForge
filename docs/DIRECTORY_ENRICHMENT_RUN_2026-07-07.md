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

## Batch 05 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 05 | 304 |
| Directories without `README.md` after Batch 05 | 276 |
| New directory guides added in Batch 05 | 25 |

Batch 05 added source-linked guides for high-volume healthcare, finance, prompt, agent, dataset, model, legal, science, entertainment, and predictive-AI directories. The largest remaining unguided healthcare area, radiology, now has local routing and official references; finance now has guides for fraud detection, credit scoring, general fintech AI, algorithmic trading, and risk management; prompt engineering now has guides for universal techniques, medical prompts, and coding prompts.

Key source families inserted directly into the new guides include ACR, DICOM, RSNA, FDA, WHO, HL7 FHIR, PhysioNet, CFPB, Federal Reserve, OCC, FINRA, SEC, FRED, Kaggle, IEEE-CIS, Amazon Science, OpenAI, Anthropic, Google Gemini, Hugging Face, LangGraph, CrewAI, AutoGen, Semantic Kernel, Microsoft Copilot Studio, Google Agentspace, Salesforce Agentforce, ServiceNow AI Agent Studio, AlphaFold, Materials Project, NASA Open Data, arXiv, Papers with Code, Radiant MLHub on AWS Open Data, USDA NASS, NASA Earthdata, Unity ML-Agents, Gymnasium, Farama, Open X-Embodiment, LeRobot, Common Voice, LibriSpeech, AudioSet, ABA, CourtListener, and Caselaw Access Project.

## Batch 06 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 06 | 329 |
| Directories without `README.md` after Batch 06 | 251 |
| New directory guides added in Batch 06 | 25 |

Batch 06 added source-linked guides for climate and weather AI, AI papers, model frameworks, launch documentation, academic/data-science/education/evaluation prompts, crop-yield prediction, payment and transaction AI, inference optimization, MLOps platforms, famous benchmarks, biomass papers and datasets, protein structure prediction, radiomics features, predictive-AI framework and dataset folders, video generation, open-source/open-weight LLMs, audiovisual datasets, oncology, and neurology.

Key source families inserted directly into the new guides include Copernicus, Climate Data Store, ECMWF, NOAA, NASA Earthdata, Google Earth Engine, WeatherBench, arXiv, Semantic Scholar, OpenReview, ACL Anthology, TCIA, NCI GDC, TCGA, cBioPortal, PyTorch, TensorFlow, JAX, ONNX Runtime, vLLM, Triton, KServe, TensorRT-LLM, MLflow, Kubeflow, Weights & Biases, MLCommons, SWE-bench, AlphaFold, RCSB PDB, UniProt, ESM, OpenFold, PyRadiomics, OpenNeuro, ADNI, OpenML, UCI, OpenAI video generation, Meta Llama, Mistral, Qwen, DeepSeek, Gemma, AudioSet, AVA, and VGGSound.

## Batch 07 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 07 | 360 |
| Directories without `README.md` after Batch 07 | 220 |
| New directory guides added in Batch 07 | 31 |

Batch 07 added source-linked guides for repository operations, GitHub workflows, issue templates, tooling, CSIRO biomass code examples, source modules, training/inference/loss/optimizer modules, medical informatics, healthcare edge AI, GNN models, AutoML projects, computer vision applications, deep-learning architectures, vegetation-index features and datasets, cardiology, medical LLMs, vision models, research labs, multimodal models, the general model catalog, long-context models, transfer learning, workflow orchestration, model serving, federated learning, vision-language models, and explainable AI.

Key source families inserted directly into the new guides include GitHub Actions and issue-template documentation, PyTorch, PyTorch Lightning, scikit-learn, ONNX Runtime, TensorRT, timm, Google Earth Engine, NASA Earthdata, HL7 FHIR, OHDSI OMOP, MIMIC-IV, MONAI Deploy, LiteRT, PyTorch Geometric, DGL, NVIDIA GNN resources, AutoGluon, H2O AutoML, FLAML, OpenCV, TorchVision, Hugging Face computer-vision and VLM resources, PhysioNet, PTB-XL, Med-PaLM, BioMistral, PubMed, OpenAI, Google Gemini, Anthropic, RULER, PEFT, Airflow, Dagster, Prefect, Kubeflow Pipelines, Flyte, KServe, BentoML, Triton, vLLM, Seldon, Flower, TensorFlow Federated, FedML, OpenFL, SHAP, InterpretML, Captum, and LIME.

## Batch 08 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 08 | 405 |
| Directories without `README.md` after Batch 08 | 175 |
| New directory guides added in Batch 08 | 45 |

Batch 08 added source-linked guides for high-density remaining healthcare, MLOps, storage, deployment, experiment, and project-showcase subdirectories. Healthcare additions covered synthetic data, telemedicine, drug-discovery AI models, variant calling, epidemiology surveillance, medical imaging edge compression, healthcare papers, molecular docking, mental health, patient monitoring, rare disease diagnosis, public health, proteomics, protein-folding variant interpretation, ICU monitoring, segmentation foundation models, medical image synthesis, transfer learning, continual learning, self-supervised learning, and clinical trials.

MLOps and infrastructure additions covered model serving deployment, Kubernetes, Azure/Microsoft AI cloud, Azure DeepSpeed, AutoML platforms, Kubernetes blue-green, ONNX deployment, API integration tools, object storage, key-value stores, canary deployment, A/B testing, model registry solutions, and experiment tracking. Project coverage added Kaggle winning solutions and fintech AI projects.

Key source families inserted directly into the new guides include Kaggle, FDA, Synthea, HL7 FHIR, WHO, ONC, DeepChem, TDC, ChEMBL, PubChem, Genome in a Bottle, GATK, ClinVar, gnomAD, DeepVariant, WHO data, CDC data, Our World in Data, Johns Hopkins CSSE, ONNX Runtime, TensorRT, LiteRT, OpenVINO, PubMed, Europe PMC, medRxiv, AutoDock Vina, Orphanet, HPO, OMIM, PRIDE, Human Protein Atlas, ProteomeXchange, Ensembl VEP, eICU, Segment Anything, MONAI Generative Models, FINRA, CFPB, KServe, BentoML, MLflow, Kubernetes, NVIDIA GPU Operator, DeepSpeed, Azure Machine Learning, OpenAPI, MCP, FastAPI, Amazon S3, Google Cloud Storage, Azure Blob Storage, MinIO, Redis, DynamoDB, Argo Rollouts, MLflow Model Registry, Vertex AI Model Registry, SageMaker Model Registry, Weights & Biases, Neptune, Comet ML, DoWhy, CausalML, SHAP, and InterpretML.

## Batch 09 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 09 | 449 |
| Directories without `README.md` after Batch 09 | 131 |
| New directory guides added in Batch 09 | 44 |

Batch 09 added source-linked guides for storage infrastructure, clinical dialogue, clinical decision support, biomedical NLP, medical QA, personalized medicine, MLOps observability and infrastructure, pruning tools, retail/ecommerce AI, crop health, crop growth, integrated agricultural AI, few-shot rare crop disease learning, Earth observation foundation models, finance tutorials and repositories, biomass estimation, data-science feature engineering, preprocessing augmentation, streaming platforms, histology features, ML ETL, web scraping, feature stores, public datasets, AI/ML data management, computer-vision augmentation, document databases, distributed file systems, in-memory databases, graph databases, time-series databases, phenotypic feature extraction, medical concept embeddings, topographic features, data warehouses, data lakes, and agriculture LLMs.

Key source families inserted directly into the new guides include scikit-learn, pandas, Kafka, Flink, Beam, Redpanda, OpenSlide, HistomicsTK, dbt, Airflow, Dagster, Great Expectations, Scrapy, Beautiful Soup, Playwright, Common Crawl, Feast, Tecton, OpenMetadata, DataHub, Albumentations, Kornia, NVIDIA DALI, MongoDB, Couchbase, CouchDB, Cosmos DB, HDFS, CephFS, Lustre, Redis, Dragonfly, Neo4j, TinkerPop, RDF, Neptune, InfluxDB, Timescale, Prometheus, VictoriaMetrics, HPO, OHDSI, PlantCV, UMLS, SNOMED CT, LOINC, RxNorm, SRTM, USGS 3DEP, Copernicus DEM, BigQuery, Snowflake, Redshift, Delta Lake, Iceberg, Hudi, FAO, CGIAR, PubAg, FDA CDS guidance, AHRQ, HL7 FHIR, OpenTelemetry, Evidently, Langfuse, PyTorch pruning, NNI pruning, RecBole, TensorFlow Recommenders, NASA Earthdata, USDA NASS, Prithvi, FinRL, Qlib, GEDI, PlantVillage, and CABI.

## Batch 10 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 10 | 481 |
| Directories without `README.md` after Batch 10 | 99 |
| New directory guides added in Batch 10 | 32 |

Batch 10 added source-linked guides for data preprocessing, data management, ETL, data science, data streaming, data versioning, data transformation, data sources, data cataloging, data labeling, data quality, data annotation, multimodal VQA, cloud-hosted and government open-data portals, vision-language datasets, radiomics datasets, audio multimodal datasets, autonomous-driving datasets, video action recognition, social-science AI governance and cross-domain datasets, plant disease detection, agriculture transfer learning, synthetic AgTech data, seasonal monitoring, weed segmentation, TinyML edge plant disease AI, CSIRO biomass docs, and remote-sensing vegetation/crop-type prediction.

Key source families inserted directly into the new guides include scikit-learn, pandas, TensorFlow, TorchVision, DataHub, OpenMetadata, DVC, Great Expectations, dbt, Airflow, Dagster, Spark, Jupyter, Kafka, Flink, Beam, Redpanda, PlantVillage, FAO, lakeFS, Delta Lake, Iceberg, NASA Earthdata, Radiant MLHub, Polars, Google Earth Engine, GEDI, PEFT, BlenderProc, Omniverse Replicator, Edge Impulse, OpenAPI, Common Crawl, VQA, GQA, DocVQA, AWS Open Data, Google Cloud public datasets, Azure Open Datasets, Data.gov, World Bank, COCO Captions, LAION, TCIA, PyRadiomics, AudioSet, AVA, nuScenes, Waymo Open Dataset, Argoverse, KITTI, Label Studio, CVAT, Snorkel, FiftyOne, OECD AI, Stanford AI Index, AI Incident Database, NIST AI RMF, World Bank Data, ICPSR, Kinetics, UCF101, HMDB51, and Papers with Code.

## Batch 11 Result

| Metric | Count |
|---|---:|
| Directories with `README.md` after Batch 11 | 580 |
| Directories without `README.md` after Batch 11 | 0 |
| New directory guides added in Batch 11 | 99 |

Batch 11 completed README coverage across every directory and subdirectory currently present in the repository. The new guides cover the remaining core AI/ML theory areas, prompt directories, model subdirectories, cloud and benchmark folders, medical and scientific datasets, reinforcement-learning simulators, predictive-AI templates, and AI project-showcase categories.

Key source families inserted directly into the new guides include OpenAI, Anthropic/Claude, Google Gemini, Microsoft Foundry, Promptfoo, scikit-learn, PyTorch, TensorFlow, Keras, Hugging Face Transformers, Hugging Face datasets and models, Papers with Code, OpenML, UCI, arXiv, OpenReview, TorchVision, OpenCV, Diffusers, PyTorch Geometric, DGL, Open Graph Benchmark, NVIDIA GNN resources, NIST AI RMF, ART, TensorFlow Privacy, MLCommons, HELM, lm-evaluation-harness, AWS, SageMaker, AWS Open Data, Google Cloud AI, Azure AI, GDC, cBioPortal, TCIA, PhysioNet, PubMed, RCSB PDB, AlphaFold DB, UniProt, NASA Earthdata, Copernicus, NOAA, Google Earth Engine, SEC EDGAR, FRED, FINRA, Gymnasium, MuJoCo, Isaac Lab, Unity ML-Agents, PlantVillage, FAOSTAT, USDA NASS, Radiant MLHub, TensorFlow Lite, ONNX Runtime Mobile, Core ML, ExecuTorch, Rasa, Botpress, Monash forecasting data, UCR time-series archive, GluonTS, sktime, SHAP, InterpretML, Captum, RFC 8615, security.txt, W3C DID, OpenSearch, and Web App Manifest.

## Batch 12 Result

| Metric | Count |
|---|---:|
| New source-index files added in Batch 12 | 4 |
| Repository Markdown files after Batch 12 | 2,220 |
| Unique external URLs after Batch 12 | 12,640 |

Batch 12 moved beyond directory coverage into source-depth. It added organized, topic-specific source indexes for prompt engineering, model hubs/providers, benchmark/evaluation sources, and agent frameworks/evaluation. These indexes are intended to convert broad searches into maintainable local routing files with provenance, trust ranking, metadata requirements, and caveats.

New files:

- `01_AI_FUNDAMENTALS_AND_THEORY/Prompt_Engineering/Prompt_Source_Index_2026-07-07.md`
- `02_LLM_AND_AI_MODELS/Models/Model_Hub_Provider_Index_2026-07-07.md`
- `03_DATASETS_TOOLS_AND_RESOURCES/Datasets/Benchmarks/Benchmark_And_Evaluation_Source_Index_2026-07-07.md`
- `04_MLOPS_AND_PRODUCTION_AI/AI_Agents/Agent_Frameworks_And_Evaluation_Source_Index_2026-07-07.md`

Key source families inserted directly into the new indexes include OpenAI prompt/model/Agents SDK/Cookbook/Evals docs, Anthropic prompt/model/transparency docs, Claude Code prompt library, Google Gemini model/prompt/gallery/changelog docs, Microsoft Foundry prompt/system-message/Agent Framework/Semantic Kernel docs, Hugging Face model and inference-provider docs, OpenRouter, Replicate, Together AI, Groq, LiteLLM, LangSmith, LangChain, LangGraph, AutoGen, CrewAI, LlamaIndex, Pydantic AI, Google ADK, Model Context Protocol, MLPerf, HELM, lm-evaluation-harness, SWE-bench, GPQA, Chatbot Arena, BIG-bench, HumanEval, MATH, OpenML, UCI, Promptfoo, Ragas, DeepEval, Langfuse, GAIA, AgentBench, tau-bench, OSWorld, and WebArena.

## Batch 13 Result

| Metric | Count |
|---|---:|
| New country/region matrix files added in Batch 13 | 1 |
| Repository Markdown files after Batch 13 | 2,221 |
| Unique external URLs after Batch 13 | 12,675 |

Batch 13 started the country and region source-coverage layer requested for global AIForge expansion. It added `03_DATASETS_TOOLS_AND_RESOURCES/Global_AI_Ecosystem/Country_And_Region_AI_Source_Coverage_Matrix_2026-07-07.md`, with global aggregator routing plus a priority seed matrix for the United States, European Union, Brazil, India, Canada, Singapore, Japan, United Kingdom, China, South Korea, United Arab Emirates, Saudi Arabia, Australia, France, Germany, Spain, Italy, Netherlands, South Africa, Nigeria, Chile, Mexico, Colombia, and Israel.

Key source families inserted directly into the country matrix include OECD.AI, OECD national AI repository, UNESCO AI ethics/governance sources, Stanford AI Index, Stanford Global AI Vibrancy Tool, European Commission AI Watch, AI.gov, NIST AI RMF, EU AI Act, European AI Office, Brazilian PBIA/EBIA, INDIAai, India AI Mission, Canada AI for All, Singapore NAIS, Japan AI Basic Plan, UK digital standards and parliamentary AI regulation briefings, China OECD policy pages, Korea OECD policy pages, UAE AI government policy sources, SDAIA, and OECD country pages for additional national coverage.

## Batch 14 Result

| Metric | Count |
|---|---:|
| New global country/area backlog files added in Batch 14 | 1 |
| UN M49 country/area rows added | 248 |
| Repository Markdown files after Batch 14 | 2,222 |
| Unique external URLs after Batch 14 | 12,678 |

Batch 14 added `03_DATASETS_TOOLS_AND_RESOURCES/Global_AI_Ecosystem/UN_M49_Country_Area_AI_Coverage_Backlog_2026-07-07.md`, generated from the official United Nations Statistics Division M49 country/area table. This file is a coverage-control backlog for the user-requested "all countries" expansion. It deliberately distinguishes between countries/areas already seeded in the Batch 13 country matrix and countries/areas that still need native official AI-policy or AI-ecosystem source discovery.

Key source families inserted directly into the Batch 14 backlog include United Nations Member States, UN Statistics Division M49 country or area codes, the UN M49 overview table, and the OECD.AI national AI repository. The backlog also preserves UN M49 LDC, LLDC, and SIDS flags to support inclusion-focused search prioritization.

## Batch 15 Result

| Metric | Count |
|---|---:|
| New broad source-atlas files added in Batch 15 | 8 |
| Repository Markdown files after Batch 15 | 2,230 |
| External URL mentions after Batch 15 | 23,149 |
| Unique external URLs after Batch 15 | 12,717 |
| Changed-file unique external URLs checked in Batch 15 | 146 |

Batch 15 responded to the instruction to search beyond datasets and inserted source-atlas files across machine learning, research discovery, global data catalogs, model hubs/providers, production ML tooling, agent frameworks/benchmarks, prompting/evals/PromptOps, and vertical AI applications. These files are source-first routers: each one maps official docs, primary repositories, benchmark pages, public portals, and domain authorities into the existing AIForge directory taxonomy so later additions land in the right subject folders.

New files:

- `01_AI_FUNDAMENTALS_AND_THEORY/Machine_Learning/Machine_Learning_Frameworks_And_Research_Source_Atlas_2026-07-07.md`
- `01_AI_FUNDAMENTALS_AND_THEORY/Prompt_Engineering/Prompting_Evals_And_PromptOps_Source_Atlas_2026-07-07.md`
- `02_LLM_AND_AI_MODELS/Models/Open_And_Closed_Model_Source_Atlas_2026-07-07.md`
- `03_DATASETS_TOOLS_AND_RESOURCES/Research_Platforms_and_Preprints/AI_Research_Discovery_Source_Atlas_2026-07-07.md`
- `03_DATASETS_TOOLS_AND_RESOURCES/Resource_Catalogs/Global_Data_Source_Catalog_Atlas_2026-07-07.md`
- `04_MLOPS_AND_PRODUCTION_AI/AI_Agents/Agent_Frameworks_Benchmarks_And_Runtime_Source_Atlas_2026-07-07.md`
- `04_MLOPS_AND_PRODUCTION_AI/MLOps_Platforms/Production_ML_Tooling_Source_Atlas_2026-07-07.md`
- `05_VERTICAL_APPLICATIONS/Vertical_AI_Source_Routing_Atlas_2026-07-07.md`

Key source families inserted directly into Batch 15 include scikit-learn, PyTorch, TensorFlow, Keras, JAX, XGBoost, LightGBM, CatBoost, Ray Tune, Optuna, arXiv, OpenReview, Semantic Scholar, Papers with Code, JMLR, NeurIPS, PMLR, ACL Anthology, CVF Open Access, PubMed, Hugging Face Datasets and Models, Kaggle Datasets and Models, OpenML, UCI, Data.gov, data.europa.eu, World Bank Data, UNdata, OECD Data Explorer, IMF Data, WHO GHO, FAOSTAT, NASA Earthdata, NOAA Data, Copernicus, AWS Open Data, BigQuery public datasets, OpenAI, Anthropic, Google Gemini, Microsoft Foundry, PyTorch Hub, ONNX Model Zoo, NVIDIA NGC, Replicate, OpenRouter, MLflow, Kubeflow, KServe, BentoML, Seldon, Ray Serve, Feast, DVC, lakeFS, Evidently, WhyLabs, Great Expectations, Weights and Biases, Neptune, Langfuse, Promptfoo, Airflow, Dagster, Prefect, Metaflow, Flyte, Argo Workflows, Model Context Protocol, LangGraph, AutoGen, CrewAI, LlamaIndex, Pydantic AI, Semantic Kernel, Google ADK, smolagents, SWE-bench, WebArena, OSWorld, tau-bench, AgentBench, ToolBench, Terminal-Bench, GAIA, FDA, PhysioNet, TCIA, PubChem, RCSB PDB, AlphaFold DB, UniProt, SEC EDGAR, FRED, FINRA, USDA NASS, CGIAR, IPCC, UNESCO, NIST, CISA, MITRE ATT&CK, OWASP, Gymnasium, MuJoCo, Isaac Lab, RoboSuite, nuScenes, Waymo Open Dataset, Argoverse, KITTI, and the UN e-Government Knowledgebase.

## Batch 16 Result

| Metric | Count |
|---|---:|
| New country-source seed files added in Batch 16 | 1 |
| Countries promoted from `S0 seeded` to `S2 official national source` | 2 |
| `S2 official national source` countries after Batch 16 | 141 |
| `S0 seeded` countries after Batch 16 | 53 |
| Repository Markdown files after Batch 16 | 2,231 |
| Unique external URLs after Batch 16 | 12,720 |

Batch 16 continued the country-by-country source work without forcing weak evidence. Bahamas was promoted based on an official Government of The Bahamas source establishing a National AI Committee to draft the country's first AI legislation. Maldives was promoted based on an official Ministry of Foreign Affairs AI governance statement, paired with UNESCO's AI readiness assessment report for the country. Kuwait, Timor-Leste, Georgia, Madagascar, Malawi, Seychelles, Vanuatu, Kiribati, and the Marshall Islands were preserved as deferred leads because the available evidence was unstable, not country-owned, or not AI-specific enough for `S2`.

New file:

- `03_DATASETS_TOOLS_AND_RESOURCES/Global_AI_Ecosystem/Official_National_AI_Source_Seeds_Batch_10_2026-07-07.md`

## Batch 17 Result

| Metric | Count |
|---|---:|
| New country-source seed files added in Batch 17 | 1 |
| Countries promoted from `S0 seeded` to `S2 official national source` | 2 |
| `S2 official national source` countries after Batch 17 | 143 |
| `S0 seeded` countries after Batch 17 | 51 |
| Repository Markdown files after Batch 17 | 2,232 |
| Unique external URLs after Batch 17 | 12,723 |

Batch 17 continued the remaining-country pass. Saint Kitts and Nevis was promoted based on an official St. Kitts and Nevis Information Service source for a public-sector AI assistant answering legal and regulatory questions. Iraq was promoted based on a `training.ai.gov.iq` national AI and big-data strategy page, with corroborating Iraqi News Agency evidence for national AI strategy development. Eswatini, Angola, Malawi, Seychelles, and Myanmar were preserved as deferred leads because the available evidence was unstable, secondary, social-only, or not country-owned enough for `S2`.

New file:

- `03_DATASETS_TOOLS_AND_RESOURCES/Global_AI_Ecosystem/Official_National_AI_Source_Seeds_Batch_11_2026-07-07.md`

## Next Batches

| Batch | Focus |
|---|---|
| 18 | Continue the remaining 51 `S0 seeded` countries with conservative official-source promotion and deferred-lead evidence. |
| 19 | Deepen high-value directories with ranked resources, local examples, and reproducible source-validation manifests beyond README coverage. |
| 20 | Add deeper domain-specific source indexes for healthcare, agriculture, finance, climate, education, robotics, cybersecurity, and science AI. |
