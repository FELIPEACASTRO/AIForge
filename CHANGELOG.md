# Changelog

All notable changes to **AIForge**.

## [2.0.0] — 2026-06-27 — Devastating Restructure

### Reorganized
- **5 canonical pillars** with `Snake_Case_English_Only` directories. Zero duplicate categories.
- **01_AI_FUNDAMENTALS_AND_THEORY**: consolidated `Deep Learning` / `Deep_Learning` / `deep_learning` / `Deep_Learning_Architectures` into a single canonical `Deep_Learning/`. Same pattern applied to `Generative_Models`, `Graph_Neural_Networks`, `Multimodal`, `Machine_Learning`, `Model_Optimization`, `Privacy_and_Security`, `Few_Shot_Learning`, `Contrastive_Learning`, `Domain_Adaptation`, `Quantum_Machine_Learning`, `Online_Learning`, `Long_Context_Models`, `Vision_Language_Models`, `Video_Understanding`, `LLM_Architectures`, `Active_Learning`, `Optimization_Algorithms`, `Natural_Language_Processing`.
- **02_LLM_AND_AI_MODELS**: collapsed 336+ fragment stub files. Created thematic subtree: `Text_LLMs/`, `Vision_Models/`, `Audio_Models/`, `Video_Models/`, `Multimodal_Models/`, `Scientific_Models/`, `Time_Series_Models/`, `MoE_Models/`, `Diffusion_Models/`, `World_Models/`, `Foundation_Models/`, `Frameworks/`, `Papers/`, `Research_Labs/`, `Communities/`, `Guides_and_Tutorials/`, `Collections/`.
- **03_DATASETS_TOOLS_AND_RESOURCES**: collapsed 84 chaotic subdirs (many comma-separated names) into 4 canonical umbrellas: `Datasets/`, `Data_Engineering/`, `Storage_and_Databases/`, `Cloud_Platforms/`.
- **04_MLOPS_AND_PRODUCTION_AI**: consolidated `MLOps_Platforms/`, `Deployment/`, `Model_Serving/`, `Inference_Optimization/`, `Model_Optimization/`, `Workflow_Orchestration/`, `AB_Testing_and_Canary/`, `AI_Agents/`, `Cloud_Platforms/`.
- **05_VERTICAL_APPLICATIONS**: 20 sequential industry verticals — `01_Healthcare_*` through `20_AI_Project_Showcases/`.

### Removed
- 10,400+ empty/fragment stub `.md` files (e.g. `101-200._Music_Analysis_Processing:.md`, four-line `1._GLM_4.6.md` stubs).
- `06_HISTORICAL_ANALYSIS_FILES/` (40 MB of legacy audit artifacts).
- `ENRIQUECIMENTO_4_ARQUIVOS/` and `ENRIQUECIMENTO_ANEXOS/` across every pillar.
- Root-level junk: raw resource lists (`all_resources_*.txt`, `resource_list*.txt`), stray Colab notebooks, `competent-shamir/`, SEO/strategy docs (`competitor_analysis.md`, `advanced_seo_optimization_report.md`, `SUBMISSION_CHECKLIST.md`, `FINAL_DELIVERY_REPORT_V3.md`, `EXTERNAL_DISCOVERY_STRATEGY.md`, `LOGO_USAGE_GUIDE.md`, `ULTIMATE_AI_COLLECTION.md`, `USER_PROVIDED_*.md`).
- `nan/` directory artifact.

### Moved
- Verticals out of `01_AI_FUNDAMENTALS_AND_THEORY/` to `05_VERTICAL_APPLICATIONS/`: `Healthcare`, `Saúde`, `Healthcare_Science`, `AI_in_Radiology`, `Finance`, `Agriculture`, `Legal`, `Business`, `Creative`, `Design`, `Education`, `Universities`, `Science`.
- Verticals out of `02_LLM_AND_AI_MODELS/`: `AI in Cardiology`, `AI in Neurology`, `Medical_Imaging`, `drug_discovery`, `genomics`, `ai_in_proteomics` → `05_VERTICAL_APPLICATIONS/01_Healthcare_and_Medical_AI/`.
- Datasets/data tooling out of `02` to `03`.
- `Technology`, `Technology_DevOps` → `04_MLOPS_AND_PRODUCTION_AI/Infrastructure/`.

### Added (gap-fill — net-new directories with curated content)

- `01/AI_Safety_and_Alignment/` — RLHF, DPO, GRPO, Constitutional AI, mechanistic interp, red-teaming.
- `01/Agentic_AI/` — ReAct, Reflexion, Tool Use, MCP, frameworks (LangGraph, AutoGen, CrewAI, Pydantic AI, DSPy).
- `01/RAG_and_Retrieval/` — RAG, HyDE, Self-RAG, CRAG, GraphRAG, ColBERT, embedding models, re-rankers, frameworks.
- `01/AI_Evaluation/` — MMLU-Pro, GPQA, HLE, SWE-bench, ARC-AGI, LiveCodeBench, eval frameworks.
- `01/Causal_Inference/` — DoWhy, EconML, CausalML, NOTEARS, books and surveys.
- `01/Modern_Fine_Tuning/` — SFT, DPO/KTO/ORPO/SimPO/GRPO, LoRA/QLoRA/DoRA, Unsloth, Axolotl.
- `01/State_Space_Models/` — Mamba, S4, RWKV, Hyena, hybrid Jamba/Zamba.
- `02/MoE_Models/` — Mixtral, DeepSeek-V3, OLMoE, Snowflake Arctic, Jamba, DBRX.
- `02/Diffusion_Models/` — SD3.5, FLUX, Veo, Sora, Wan, HunyuanVideo, CogVideoX, plus 3D/molecule diffusion.
- `02/World_Models/` — Genie, Cosmos, V-JEPA, GAIA, OASIS, robotics VLAs.
- `04/LLM_Inference/` — vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, Ollama, MLX, MLC, ExecuTorch.
- `04/AI_Observability/` — Langfuse, LangSmith, Weave, Phoenix, Helicone, OpenLLMetry.
- `05/08_Manufacturing_and_Industry_AI/` — PdM, QC, digital twins, supply chain.
- `05/10_Robotics_and_Embodied_AI/` — VLAs, datasets, simulators, software stacks, benchmarks.
- `05/11_Autonomous_Vehicles_AI/` — UniAD, GAIA, datasets, simulators.
- `05/13_Energy_AI/` — Grid, renewables, fusion, batteries, smart buildings.
- `05/14_Cybersecurity_AI/` — Red/Blue team, model security, OWASP LLM Top 10, benchmarks.

### Documentation
- Rewrote `README.md` around the 5-pillar canonical taxonomy with a per-pillar tree view.
- Rewrote `NAVIGATION_GUIDE.md` as a topic-first lookup table.
- Regenerated `INDEX.md` as a complete machine-generated sitemap.

## [1.x] — Prior history
Snapshots in git history. See commits before this changelog entry for the legacy state.
