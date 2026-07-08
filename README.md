# AIForge — The Definitive Repository for AI, ML, and Data Science

**AIForge** is a comprehensive, hand-organized atlas of the **Artificial Intelligence**, **Machine Learning**, **Deep Learning**, **LLM**, and **Data Science** ecosystem — from foundational theory to production deployment and industry verticals.

> **2,200+ documents · 100% English · every directory carries an index README with keywords · a complete [`sitemap.xml`](./sitemap.xml) (2,700+ URLs) covers every folder and file** — built for maximum search-engine and LLM discoverability.

<p align="center">
  <a href="https://github.com/FELIPEACASTRO/AIForge/stargazers"><img src="https://img.shields.io/github/stars/FELIPEACASTRO/AIForge?style=social" alt="Stars"></a>
  <a href="https://github.com/FELIPEACASTRO/AIForge/network/members"><img src="https://img.shields.io/github/forks/FELIPEACASTRO/AIForge?style=social" alt="Forks"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
  <a href="https://discord.gg/RTRdCVcS3"><img src="https://img.shields.io/badge/Discord-MACHINE%20LEARNING%20KNBIS-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pillars-5-blue?style=for-the-badge" alt="5 Pillars">
  <img src="https://img.shields.io/badge/Last%20Update-2026--07-orange?style=for-the-badge" alt="Updated">
  <img src="https://img.shields.io/badge/Language-English%20only-blue?style=for-the-badge" alt="English only">
  <img src="https://img.shields.io/badge/Coverage-AI%20%2F%20ML%20%2F%20DL%20%2F%20LLM%20%2F%20MLOps-green?style=for-the-badge" alt="Coverage">
</p>

---

## 🚀 Start Here: Frontier AI 2026

**[00_FRONTIER_AI_2026](./00_FRONTIER_AI_2026/)** — a living *Innovation Radar* of the newest releases (GPT-5.5, Claude Opus 4.7 / Fable 5, DeepSeek V4, Qwen 3.7, Llama 4, Sora 2, Veo 3.1, GR00T N1.7, SGLang, AI-for-math) captured **June 2026**. This is the "what just shipped" layer; the 5 pillars below are the stable, organized core.

## Five Pillars

| # | Pillar | What lives here |
|---|---|---|
| **01** | **[AI Fundamentals & Theory](./01_AI_FUNDAMENTALS_AND_THEORY/)** | Foundational ML/DL theory, algorithms, paradigms (RL, GenAI, Transformers, SSMs), training methods, safety, evaluation. |
| **02** | **[LLM & AI Models](./02_LLM_AND_AI_MODELS/)** | Frontier LLMs, open-source models, vision/audio/video/multimodal models, MoE, diffusion, world models, frameworks. |
| **03** | **[Datasets, Tools & Resources](./03_DATASETS_TOOLS_AND_RESOURCES/)** | Curated datasets by modality, data engineering, storage & databases, cloud, **[global AI ecosystem coverage](./03_DATASETS_TOOLS_AND_RESOURCES/Global_AI_Ecosystem/)**, **[HuggingFace Hub](./03_DATASETS_TOOLS_AND_RESOURCES/HuggingFace_Hub/)**, **[research/preprint platforms](./03_DATASETS_TOOLS_AND_RESOURCES/Research_Platforms_and_Preprints/)**. |
| **04** | **[MLOps & Production AI](./04_MLOPS_AND_PRODUCTION_AI/)** | Model serving, LLM inference (vLLM/SGLang/TensorRT-LLM/llama.cpp), MLOps platforms, observability, deployment. |
| **05** | **[Vertical Applications](./05_VERTICAL_APPLICATIONS/)** | Industry-specific AI + **[Kaggle competitions & winning solutions](./05_VERTICAL_APPLICATIONS/20_AI_Project_Showcases/Kaggle/)**: Healthcare, **[Banking](./05_VERTICAL_APPLICATIONS/02_Finance_and_Fintech_AI/Banking/)** (KYC/onboarding, AML, fraud), **[Financial Markets](./05_VERTICAL_APPLICATIONS/02_Finance_and_Fintech_AI/Financial_Markets/)** (stocks, options, futures, bonds, FX, crypto, quant trading), Agriculture, Robotics, AV, more. |

---

## How to Navigate

**Browse by pillar.** Each pillar has a sub-tree of canonical, English, snake_case topics — no duplicates, no language drift.

**Use [`INDEX.md`](./INDEX.md).** The complete sitemap with every category and a one-line description.

**Use [`NAVIGATION_GUIDE.md`](./NAVIGATION_GUIDE.md).** A topic-first guide ("I want to learn about RAG" → exact path).

**Track enrichment.** The repo-wide directory enrichment pass is recorded in [`docs/DIRECTORY_ENRICHMENT_RUN_2026-07-07.md`](./docs/DIRECTORY_ENRICHMENT_RUN_2026-07-07.md), and broad AI/ML source routing lives in [`AI_ML_Data_Model_Prompt_Source_Atlas_Batch_01_2026-07-07.md`](./03_DATASETS_TOOLS_AND_RESOURCES/Global_AI_Ecosystem/AI_ML_Data_Model_Prompt_Source_Atlas_Batch_01_2026-07-07.md).

**GitHub search.** Scoped path search: `vLLM path:04_MLOPS_AND_PRODUCTION_AI/LLM_Inference`.

---

## Pillar 01 — Highlights

```
01_AI_FUNDAMENTALS_AND_THEORY/
├── Mathematics_for_ML/                  NEW — linear algebra, calculus, probability, stats, convex opt
├── Statistical_Learning/                 NEW — regression, GLMs, time series, survival, A/B testing
├── Probabilistic_ML/                     NEW — PGMs, Gaussian processes, VI, MCMC, HMMs, PPLs
├── Classical_ML_Algorithms/              NEW — SVM, trees, RF, boosting, kNN, clustering, PCA
├── AutoML/                               NEW — HPO, AutoML frameworks, auto feature engineering
├── Feature_Engineering/                  NEW — selection, scaling/encoding, techniques
├── Model_Evaluation/                     NEW — metrics, model selection & validation
├── Machine_Learning/                    Classical ML, supervised/unsupervised
├── Deep_Learning/                       Architectures, regularization, optim
├── Reinforcement_Learning/
├── Generative_Models/                   GANs, VAEs, Diffusion, Flow, EBM
├── Computer_Vision/                     Self-supervised, ViT
├── Natural_Language_Processing/
├── Multimodal/
├── Graph_Neural_Networks/
├── Vision_Transformers/
├── Vision_Language_Models/
├── Video_Understanding/
├── LLM_Architectures/
├── Long_Context_Models/
├── State_Space_Models/                  NEW — Mamba, S4, RWKV, Hyena
├── Transfer_Learning, Federated_Learning, Few_Shot_Learning,
│   Meta_Learning, Contrastive_Learning, Domain_Adaptation,
│   Online_Learning, Active_Learning
├── Optimization_Algorithms/, Model_Optimization/
├── Explainable_AI/
├── Privacy_and_Security/
├── Quantum_Machine_Learning/
├── Prompt_Engineering/                  Use-case prompt libraries
├── Modern_Fine_Tuning/                  NEW — SFT, DPO, GRPO, LoRA, QLoRA
├── AI_Safety_and_Alignment/             NEW — RLHF, Constitutional AI, interp
├── Agentic_AI/                          NEW — ReAct, Reflexion, MCP, frameworks
├── RAG_and_Retrieval/                   NEW — RAG patterns, retrievers, rerank
├── AI_Evaluation/                       NEW — benchmarks, eval frameworks
├── Causal_Inference/                    NEW — DoWhy, EconML, frameworks
├── Courses/ Universities/ Communities/ Collections/
```

## Pillar 02 — Highlights

```
02_LLM_AND_AI_MODELS/
├── Text_LLMs/
│   ├── Frontier_Closed_Models/          GPT-5, Claude 4.5, Gemini 2.5
│   ├── Open_Source_LLMs/                Llama, Qwen, Mistral, DeepSeek, GLM
│   ├── Reasoning_Models/                R1, Open-R1, Sky-T1
│   ├── Small_LLMs/                      Phi, Gemma, SmolLM
│   ├── Code_LLMs/, Specialized_LLMs/
│   ├── Efficient_Transformers/
├── Vision_Models/                       Detection, segmentation, image gen
├── Audio_Models/                        ASR, TTS, music
├── Video_Models/                        Text-to-video, image-to-video
├── Multimodal_Models/                   VLMs, GLM-V, Qwen-VL
├── Scientific_Models/                   Protein, Quantum ML, Reservoir
├── Time_Series_Models/
├── MoE_Models/                          NEW — Mixtral, DeepSeek, OLMoE
├── Diffusion_Models/                    NEW — SD3.5, FLUX, Veo, Wan, Sora
├── World_Models/                        NEW — Genie, Cosmos, V-JEPA, GR00T
├── Foundation_Models/, Frameworks/
├── Papers/, Research_Labs/, Communities/
└── Guides_and_Tutorials/, Collections/
```

## Pillar 03 — Highlights

```
03_DATASETS_TOOLS_AND_RESOURCES/
├── Datasets/
│   ├── Computer_Vision_Datasets/
│   ├── NLP_Datasets/
│   ├── Audio_Datasets/, Video_Datasets/
│   ├── Multimodal_Datasets/             VQA, autonomous driving
│   ├── Time_Series_Datasets/, Tabular_Datasets/
│   ├── Climate_and_Geospatial/          Earth observation, oceanography
│   ├── Bioinformatics_and_Genomics/
│   ├── Finance_Datasets/, Gaming_and_RL/
│   ├── Robotics_Datasets/, Social_Science_Datasets/
│   ├── Open_Data_Portals/
│   ├── Synthetic_Data/, Web_Datasets/
│   └── Famous_Benchmarks/
├── Data_Engineering/                    Pipelines, versioning, quality,
│                                        annotation, feature engineering,
│                                        feature stores, ETL, web scraping
├── Storage_and_Databases/               Vector, time-series, document,
│                                        in-memory, lakes, warehouses
└── Cloud_Platforms/                     AWS / others
```

## Pillar 04 — Highlights

```
04_MLOPS_AND_PRODUCTION_AI/
├── MLOps_Platforms/                     MLflow, Kubeflow, ZenML, Metaflow
├── Model_Serving/                       Triton, BentoML, KServe, Ray Serve
├── LLM_Inference/                       NEW — vLLM, SGLang, TensorRT-LLM,
│                                        TGI, llama.cpp, Ollama, MLX, MLC
├── Inference_Optimization/              ONNX, TensorRT, quantization
├── Model_Optimization/, Model_Registry_Solutions/
├── Workflow_Orchestration/              Airflow, Prefect, Dagster
├── Deployment/                          Kubernetes, Docker, Serverless
├── AB_Testing_and_Canary/
├── AI_Observability/                    NEW — Langfuse, LangSmith, Weave,
│                                        Phoenix, Helicone, Arize, OpenLLMetry
├── AI_Agents/                           Production agent stacks
├── Cloud_Platforms/                     Azure/Microsoft, others
└── API_Integration_Tools/
```

## Pillar 05 — Highlights

```
05_VERTICAL_APPLICATIONS/
├── 01_Healthcare_and_Medical_AI/        Imaging, clinical NLP, drug discovery,
│                                        radiology, cardiology, neurology,
│                                        genomics, telemedicine, mental health
├── 02_Finance_and_Fintech_AI/           Banking (KYC/AML/fraud) + Financial Markets (stocks, options, FX, crypto, quant), credit, risk
├── 03_Agriculture_AgTech/               Precision farming, biomass, vegetation
├── 04_Climate_and_Sustainability/       Weather, Earth obs
├── 05_Education_AI/
├── 06_Legal_AI/
├── 07_Retail_and_Ecommerce/
├── 08_Manufacturing_and_Industry_AI/    NEW — PdM, QC, digital twins
├── 09_Entertainment_and_Creative_AI/
├── 10_Robotics_and_Embodied_AI/         VLAs, manipulation, humanoids
├── 11_Autonomous_Vehicles_AI/           NEW — UniAD, GAIA, datasets, sims
├── 12_Business_and_Marketing_AI/
├── 13_Energy_AI/                        NEW — Grid, renewables, fusion
├── 14_Cybersecurity_AI/                 NEW — Red/Blue team, model security
├── 15_Science_AI/
├── 16_Edge_and_IoT_AI/
├── 17_Conversational_AI/
├── 18_Predictive_AI/
├── 19_Computer_Vision_Applications/
└── 20_AI_Project_Showcases/             AutoML, Kaggle, RL, NLP, multimodal
```

---

## 💬 Community (Reddit & Discord)

Share an AIForge link on **Reddit** or **Discord** and it renders a rich preview (title, description, image, accent color) automatically — the repo ships complete Open Graph + `theme-color` tags tuned for both.

Curated community lists:
- **Reddit:** [AI/ML subreddits](./01_AI_FUNDAMENTALS_AND_THEORY/Communities/Reddit_AI_ML_Communities.md) — r/MachineLearning, r/LocalLLaMA, r/StableDiffusion, r/PromptEngineering, and more.
- **Discord:** [AI/ML servers](./01_AI_FUNDAMENTALS_AND_THEORY/Communities/Discord_AI_ML_Communities.md) — Hugging Face (224k+), EleutherAI, LAION, MLOps Community, and more.

<p align="center">
  <a href="https://www.reddit.com/r/MachineLearning/"><img src="https://img.shields.io/badge/Reddit-r%2FMachineLearning-FF4500?logo=reddit&logoColor=white" alt="Reddit"></a>
  <a href="https://www.reddit.com/r/LocalLLaMA/"><img src="https://img.shields.io/badge/Reddit-r%2FLocalLLaMA-FF4500?logo=reddit&logoColor=white" alt="r/LocalLLaMA"></a>
  <a href="https://discord.com/invite/hugging-face-879548962464493619"><img src="https://img.shields.io/badge/Discord-Hugging%20Face-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

> 💬 **Official community — [MACHINE LEARNING KNBIS on Discord](https://discord.gg/RTRdCVcS3)** — join for AI/ML discussion, LLMs, MLOps, papers, and AIForge updates.

**Share AIForge:** [Tweet](https://twitter.com/intent/tweet?text=AIForge%20%E2%80%94%20the%20definitive%20AI%2FML%2FDeep%20Learning%20resources%20index&url=https://github.com/FELIPEACASTRO/AIForge) · [LinkedIn](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/FELIPEACASTRO/AIForge) · [Reddit](https://www.reddit.com/submit?url=https://github.com/FELIPEACASTRO/AIForge&title=AIForge%20%E2%80%94%20curated%20AI%2FML%2FDL%20resources%20index) · [Hacker News](https://news.ycombinator.com/submitlink?u=https://github.com/FELIPEACASTRO/AIForge&t=AIForge)

See the full growth plan in [DISCOVERABILITY.md](./DISCOVERABILITY.md).


---

## ❓ Frequently Asked Questions (FAQ)

**What is AIForge?**
AIForge is a free, open-source, curated index of the entire Artificial Intelligence, Machine Learning, Deep Learning, LLM, and Data Science ecosystem — 2,200+ curated documents linking to 7,500+ external resources, organized into a clean, duplicate-free, **English-only** taxonomy of 5 pillars plus a Frontier AI 2026 radar. Every directory has an index README, and a complete sitemap (2,700+ URLs) covers every folder and file.

**Who is AIForge for?**
Data scientists, ML/AI engineers, researchers, students, and anyone who wants a single, well-organized map of AI/ML resources — from theory to production.

**How is AIForge organized?**
Five pillars: (01) Fundamentals & Theory, (02) LLM & AI Models, (03) Datasets/Tools/Resources, (04) MLOps & Production AI, (05) Vertical Applications — plus (00) Frontier AI 2026 for the newest releases. Browse `INDEX.md` (full sitemap) or `NAVIGATION_GUIDE.md` (topic-first lookup).

**Where do I find the latest AI models (GPT-5.5, Claude, DeepSeek, Llama 4)?**
See [`00_FRONTIER_AI_2026`](./00_FRONTIER_AI_2026/).

**Where are Kaggle winning solutions?**
See [`05_VERTICAL_APPLICATIONS/20_AI_Project_Showcases/Kaggle`](./05_VERTICAL_APPLICATIONS/20_AI_Project_Showcases/Kaggle/) — 216-competition index, top public notebooks, and curated winning write-ups.

**Where do I find datasets, vector databases, or the HuggingFace ecosystem?**
See [`03_DATASETS_TOOLS_AND_RESOURCES`](./03_DATASETS_TOOLS_AND_RESOURCES/), including [HuggingFace_Hub](./03_DATASETS_TOOLS_AND_RESOURCES/HuggingFace_Hub/) and [Research_Platforms_and_Preprints](./03_DATASETS_TOOLS_AND_RESOURCES/Research_Platforms_and_Preprints/).

**How do I deploy/serve LLMs (vLLM, SGLang, llama.cpp)?**
See [`04_MLOPS_AND_PRODUCTION_AI/LLM_Inference`](./04_MLOPS_AND_PRODUCTION_AI/LLM_Inference/).

**Is AIForge readable by AI assistants?**
Yes — it ships an [`llms.txt`](./llms.txt) following the [llmstxt.org](https://llmstxt.org/) convention so LLMs, AI agents, and AI search engines can discover, navigate, and cite the right section.

**What language is AIForge in?**
100% English — every page, heading, filename, and directory path. This keeps the taxonomy consistent and maximizes discoverability for global search engines and AI assistants.

**How can I contribute?**
See [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## 🏷️ Topics & Keywords

`artificial-intelligence` · `machine-learning` · `deep-learning` · `large-language-models` · `llm` · `generative-ai` · `data-science` · `mlops` · `datasets` · `awesome-list` · `awesome` · `computer-vision` · `nlp` · `natural-language-processing` · `reinforcement-learning` · `transformers` · `diffusion-models` · `rag` · `ai-agents` · `huggingface` · `kaggle` · `pytorch` · `tensorflow` · `model-deployment` · `vector-database` · `ai-resources` · `curated-list`

> **Searching for** "awesome machine learning list", "AI resources collection", "deep learning index", "LLM resources", "MLOps tools list", "Kaggle winning solutions", "AI datasets directory", or "machine learning roadmap"? You're in the right place.

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). New resources should go to the canonical English snake_case directory; if no fit exists, propose a new one in your PR.

## Code of Conduct & Security

- [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- [SECURITY.md](./SECURITY.md)

## License

MIT — see [LICENSE](./LICENSE).
