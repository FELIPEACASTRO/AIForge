# AI Papers, Models, Datasets, And Benchmarks Discovery Map

This map defines the source families AIForge should repeatedly mine to become a comprehensive AI/ML evidence repository. It emphasizes primary or high-authority sources and separates papers, code, datasets, model hubs, benchmarks, prompt engineering, agent frameworks, and evaluation infrastructure.

Access date: 2026-07-07.

Broader routing atlas: [AI_ML_Data_Model_Prompt_Source_Atlas_Batch_01_2026-07-07.md](./AI_ML_Data_Model_Prompt_Source_Atlas_Batch_01_2026-07-07.md).

## Scholarly Graphs And Paper Sources

| Source | Scope | Extraction target |
|---|---|---|
| [OpenAlex](https://openalex.org/) | Open scholarly graph across works, authors, institutions, sources, topics, funders, and countries | Country/institution AI publication coverage; paper metadata; venue and citation graph. |
| [Semantic Scholar](https://www.semanticscholar.org/) | Scientific literature graph and AI-assisted discovery | Paper metadata, citation graph, recommendations, influential papers. |
| [arXiv cs.AI](https://arxiv.org/list/cs.AI/recent), [cs.LG](https://arxiv.org/list/cs.LG/recent), [stat.ML](https://arxiv.org/list/stat.ML/recent) | AI, machine learning, and statistical ML preprints | Daily frontier-paper ingestion by category. |
| [OpenReview](https://openreview.net/) | Peer-review and submission platform for ICLR, TMLR, workshops, and other ML venues | Accepted/rejected paper metadata, reviews, venue history. |
| [ACL Anthology](https://aclanthology.org/) | NLP and computational linguistics proceedings | LLM/NLP papers, tasks, datasets, multilingual resources. |
| [NeurIPS Proceedings](https://papers.nips.cc/) | NeurIPS proceedings | Core ML papers, datasets and benchmarks track, invited talks. |
| [PMLR](https://proceedings.mlr.press/) | Proceedings of Machine Learning Research | ICML, AISTATS, COLT, UAI, and related proceedings. |
| [DBLP](https://dblp.org/) | Computer-science bibliography | Cross-venue bibliography, author disambiguation, conference history. |

## Code, Benchmark, And Leaderboard Sources

| Source | Scope | Extraction target |
|---|---|---|
| [Papers with Code](https://paperswithcode.com/) | Papers, code, datasets, methods, and evaluation leaderboards | Task/dataset/metric triples, SOTA history, code links. Verify current availability before treating as live. |
| [Hugging Face Papers](https://huggingface.co/papers) | Daily AI paper discovery linked to models, datasets, and community discussion | Trending papers, implementation links, community signal. |
| [OpenML](https://www.openml.org/) | Open ecosystem of datasets, tasks, flows, runs, and benchmarks | Machine-learning tasks, reproducible runs, dataset metadata. |
| [MLCommons](https://mlcommons.org/) | MLPerf benchmarks and AI safety/data initiatives | Training/inference benchmark families and standardized performance data. |
| [HELM](https://crfm.stanford.edu/helm/) | Holistic language-model evaluation | LLM evaluation scenarios, metrics, model comparisons. |
| [LMSYS Chatbot Arena](https://lmarena.ai/) | Human preference arena for chat models | Model battle results, arena rankings, open evaluation notes. |

## Dataset Hubs

| Source | Scope | Extraction target |
|---|---|---|
| [Hugging Face Datasets](https://huggingface.co/datasets) | Community datasets across audio, document, geospatial, image, tabular, text, time-series, video, benchmark, and trace categories | Dataset cards, modality, language, license, task, benchmark tags. |
| [Hugging Face Hub documentation](https://huggingface.co/docs/hub/en/index) | Hub documentation for models, datasets, and Spaces | Platform-scale metadata and API entry points. |
| [Kaggle Datasets](https://www.kaggle.com/datasets) | Open datasets and ML projects | Dataset categories, usability score, update date, file format, competitions linkage. |
| [OpenML datasets](https://www.openml.org/search?type=data) | Uniformly formatted datasets with metadata for automated processing | Dataset IDs, tasks, qualities, benchmark suites. |
| [Google Dataset Search](https://datasetsearch.research.google.com/) | Cross-web dataset discovery | Long-tail dataset discovery beyond ML-specific hubs. |
| [Zenodo](https://zenodo.org/) | Research outputs, datasets, software, and DOI-backed artifacts | Academic datasets, software releases, citation metadata. |

## Model And Artifact Hubs

| Source | Scope | Extraction target |
|---|---|---|
| [Hugging Face Models](https://huggingface.co/models) | Open model hub across tasks, modalities, libraries, licenses, and downloads | Model cards, task taxonomy, downloads, likes, papers, datasets. |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Interactive demos and apps | Model demos, Gradio/Streamlit apps, runnable examples. |
| [ModelScope](https://www.modelscope.cn/) | Model and dataset ecosystem with strong China/Asia coverage | Regional model/dataset coverage and multilingual resources. |
| [GitHub Topics - machine-learning](https://github.com/topics/machine-learning) | Open-source ML code repositories | Repository discovery, stars, activity, topic taxonomy. |
| [GitHub Topics - artificial-intelligence](https://github.com/topics/artificial-intelligence) | Open-source AI repositories | Broad AI code/resource discovery. |

## Prompt, Context, And Agent Sources

| Source | Scope | Extraction target |
|---|---|---|
| [OpenAI prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering) | Official prompt engineering guide for OpenAI models | Prompt patterns, model-specific guidance, tool prompts, agent prompts. |
| [Anthropic prompt engineering](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) | Official Claude prompt engineering documentation | Prompt structure, examples, XML organization, evaluation-first prompting. |
| [Google Gemini prompt design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies) | Gemini prompting guide | Prompt strategies, prompt examples, multimodal prompting notes. |
| [Microsoft Foundry prompt engineering](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/prompt-engineering) | Azure OpenAI prompt engineering documentation | Grounding, system messages, prompt testing, and safety templates. |
| [LangSmith prompt management](https://docs.langchain.com/langsmith/manage-prompts) | Prompt creation, versioning, hub, and templates | Prompt lifecycle, prompt provenance, public prompt risk labels. |
| [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents) | Agent orchestration with tools, handoffs, tracing, and guardrails | Agent framework pages and production patterns. |
| [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro) | Open protocol connecting AI apps to tools, data, prompts, and workflows | MCP servers, clients, security notes, and integration examples. |

## Extraction Priorities

1. Build repeatable queries for core topics: machine learning, AutoML, deep learning, reinforcement learning, LLMs, RAG, agents, AI safety, computer vision, NLP, time series, graph ML, diffusion, datasets, feature engineering, benchmarks, MLOps.
2. Store source metadata before summaries: title, publisher, URL, date, access date, scope, country/region, task/modality, license when available.
3. Prefer primary metadata APIs where available: OpenAlex, Semantic Scholar, arXiv, OpenReview, Hugging Face Hub, OpenML, Kaggle.
4. Keep benchmark claims separated from educational content; task/dataset/metric rows need exact definitions.
5. Re-run availability checks because some community leaderboards and paper indexes can change ownership, URL structure, or maintenance status.
