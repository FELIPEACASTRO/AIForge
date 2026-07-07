# AI Research Discovery Source Atlas - 2026-07-07

This file enriches `Research_Platforms_and_Preprints/` with high-authority discovery sources for AI, machine learning, data science, agents, evaluation, and applied AI. The purpose is not only to find papers, but to preserve enough evidence to know whether a paper is reproducible, current, peer reviewed, benchmark-backed, or only exploratory.

## Literature Discovery Sources

| Source | What to collect | Best local destination |
|---|---|---|
| [arXiv computer science](https://arxiv.org/archive/cs) | Category, arXiv id, version history, authors, abstract, and links to code/data when available. | Theory, models, and preprint folders. |
| [arXiv cs.LG recent](https://arxiv.org/list/cs.LG/recent) | New machine-learning submissions and replacements. | `Machine_Learning/`, `Deep_Learning/`, `Model_Evaluation/` |
| [arXiv cs.AI recent](https://arxiv.org/list/cs.AI/recent) | AI, agents, reasoning, planning, knowledge, and general intelligence papers. | `Agentic_AI/`, `AI_Evaluation/`, `Neuro_Symbolic_AI/` |
| [OpenReview](https://openreview.net/) | Reviews, decisions, author responses, conference groups, and paper revisions. | Conference-backed research notes. |
| [Semantic Scholar](https://www.semanticscholar.org/) | Citation graph, related papers, author pages, TLDRs, and paper metadata. | Literature review and source clustering. |
| [Papers with Code](https://paperswithcode.com/) | Tasks, datasets, code links, methods, and leaderboards. | Benchmark and implementation evidence. |
| [JMLR](https://www.jmlr.org/) | Peer-reviewed journal articles and surveys. | Stable theory and methodology references. |
| [Proceedings of Machine Learning Research](https://proceedings.mlr.press/) | ICML, AISTATS, COLT, UAI, and workshop proceedings. | Conference paper intake. |
| [NeurIPS proceedings](https://proceedings.neurips.cc/) | Papers, datasets/benchmarks track entries, and supplementary material. | Frontier research and benchmark files. |
| [ACL Anthology](https://aclanthology.org/) | NLP papers, shared tasks, datasets, and proceedings. | `Natural_Language_Processing/` and LLM evaluation. |
| [CVF Open Access](https://openaccess.thecvf.com/) | Computer-vision papers from CVPR, ICCV, ECCV, and WACV. | `Computer_Vision/`, `Vision_Models/`, `Vision_Language_Models/` |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Biomedical and clinical literature metadata. | Healthcare AI, medical NLP, genomics, drug discovery. |

## Intake Checklist

| Field | Required detail |
|---|---|
| Source identity | Title, authors, venue or repository, source URL, and access date. |
| Evidence class | Peer reviewed, preprint, benchmark paper, dataset paper, system report, model card, or technical blog. |
| Reproducibility | Code, data, license, environment, checkpoint, evaluation script, and hardware assumptions. |
| Evaluation | Dataset, split, metric, baseline, score date, and whether the benchmark may be contaminated. |
| Local routing | Exact AIForge directory that owns the method, model, dataset, benchmark, or application. |
| Risk notes | Unsupported claims, missing code, unavailable data, leakage risk, safety risk, or unclear license. |

## Source Ranking

| Rank | Source type | Handling |
|---|---|---|
| 1 | Official proceedings, journals, or standards bodies | Treat as primary evidence for the paper or policy. |
| 2 | Paper plus official code/data repository | Treat as reproducible candidate after license and run checks. |
| 3 | Discovery platforms such as Semantic Scholar or Papers with Code | Use for routing and cross-linking; verify claims against primary sources. |
| 4 | Blogs, newsletters, or social posts | Use only as leads unless backed by primary artifacts. |

## Expansion Queue

| Topic | Next source families |
|---|---|
| Agent papers | arXiv cs.AI, OpenReview, SWE-bench, WebArena, OSWorld, tau-bench, AgentBench. |
| ML systems | PMLR, NeurIPS, MLflow/Kubeflow/KServe docs, Ray docs. |
| Model releases | Provider model cards, Hugging Face model cards, technical reports, safety reports. |
| Applied AI | PubMed, FAO, NASA Earthdata, SEC/FINRA, UNESCO, OECD, NIST. |
