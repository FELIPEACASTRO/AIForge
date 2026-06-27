# 26 HR and Recruiting AI

> AI applied across the talent lifecycle — resume parsing and candidate-to-job matching, LLM screening and interview assist, skills extraction and taxonomies, attrition/turnover prediction, and the bias/fairness auditing and regulatory compliance (NYC LL144, EU AI Act high-risk) that increasingly govern all of it.

## Why it matters

Roughly 90% of large employers now use some form of automated tooling in resume screening, and LLMs have sharply lowered the cost of parsing unstructured applicant text. Hiring is a legally consequential, high-stakes domain: discrimination by race, gender, and age is well documented, so AI here is regulated as **high-risk** (EU AI Act Annex III) and subject to mandatory bias audits (NYC Local Law 144). This makes HR one of the few verticals where fairness auditing, explainability, and human oversight are not optional add-ons but core requirements.

## Taxonomy

| Sub-area | What it does | Representative approaches |
|---|---|---|
| Resume / CV parsing | Extract structured fields (skills, education, experience) from free-form documents | spaCy NER, regex, local LLMs (Ollama/Qwen), layout models |
| Candidate–job matching | Rank/score applicants against a job description | TF-IDF, BERT/bi-encoder embeddings, LLM evidence-aware scoring |
| Skills extraction & taxonomies | Map text to a controlled skills ontology | ESCO, O*NET, weak supervision, contrastive bi-encoders |
| LLM screening & interview assist | Summarize, score, generate questions, seniority classification | GPT/Claude/Gemini prompting, fine-tuned screeners |
| Attrition / turnover prediction | Predict which employees will leave | XGBoost, Random Forest, SHAP/LIME explainability |
| Bias / fairness auditing | Detect and mitigate allocational & representational bias | counterfactual/correspondence audits, disparate-impact metrics |
| Compliance & governance | Satisfy LL144, EU AI Act, human-in-the-loop | bias audit reports, transparency notices, oversight controls |

## Key tools and frameworks

| Tool | Focus | Link |
|---|---|---|
| SkillNER (spaCy) | Skill extraction from job text using ESCO/EMSI | https://github.com/AnasAito/SkillNER |
| pyresparser | Resume field extraction (spaCy + NLTK) | https://github.com/OmkarPathak/pyresparser |
| ResumeParser (OmkarPathak) | Agentic/local-LLM resume parsing (Qwen2.5) | https://github.com/OmkarPathak/ResumeParser |
| OpenResume | Open-source resume builder + ATS-readability parser | https://github.com/xitanggg/open-resume |
| ResuLLMe | LLM-based résumé enhancement | https://github.com/IvanIsCoding/ResuLLMe |
| End-to-End ATS (Gemini Pro) | ATS-style parsing, keyword match, candidate eval | https://github.com/praj2408/End-To-End-Resume-ATS-Tracking-LLM-Project-With-Google-Gemini-Pro |
| LLM_RESUME_PARSER | Local LLM (Ollama) resume → JSON | https://github.com/mahikshith/LLM_RESUME_PARSER |

## Skills taxonomies and ontologies

| Resource | Description | Link |
|---|---|---|
| ESCO | EU European Skills/Competences/Occupations taxonomy (~3k occupations, ~14k skills, 28 languages) | https://esco.ec.europa.eu/en |
| O*NET | US Dept. of Labor occupational database (skills, abilities, tasks) | https://www.onetonline.org/ |
| ESCOX (ESCOSkillExtractor) | LLM-based skill/occupation extraction over ESCO | https://www.sciencedirect.com/science/article/pii/S2665963825000326 |

## Datasets and benchmarks

| Dataset | Use | Link |
|---|---|---|
| IBM HR Analytics Employee Attrition | Standard attrition benchmark (1,470 rows, 35 features) | https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset |
| HR Analytics (Kaggle, 14,999 records) | Larger turnover-prediction dataset | https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset |
| ESCO skill-extraction corpora | Job-posting → skill linking (e.g. weak-supervision sets) | https://arxiv.org/abs/2209.08071 |
| JobSkape (synthetic job postings) | Synthetic data for skill matching | https://arxiv.org/abs/2402.03242 |

Common attrition baselines: XGBoost and Random Forest are repeatedly the strongest classifiers; class imbalance and black-box opacity are the principal recurring challenges, motivating SHAP/LIME explainability.

## Key papers

| Paper | Topic | Link |
|---|---|---|
| Raghavan et al. (2019) — Mitigating Bias in Algorithmic Hiring | Audit of vendor claims & practices in algorithmic hiring | https://arxiv.org/abs/1906.09208 |
| Mujtaba & Mahapatra (2024) — Fairness in AI-Driven Recruitment | Survey of biases, metrics, mitigation, auditing | https://arxiv.org/abs/2405.19699 |
| Gaebler et al. (2024) — Auditing LMs to Guide Hiring Decisions | Correspondence experiments on LLM hiring; race/gender disparities | https://arxiv.org/abs/2404.03086 |
| Seshadri et al. (2025) — Small Changes, Large Consequences | Allocational fairness of LLMs via counterfactual resumes | https://arxiv.org/abs/2501.04316 |
| Zhang et al. (2022) — Skill Extraction via Weak Supervision | ESCO-based skill extraction without heavy annotation | https://arxiv.org/abs/2209.08071 |
| Two Tickets are Better than One (2025) | Fair + accurate hiring under strategic LLM manipulation | https://arxiv.org/abs/2502.13221 |
| JobSkape (2024) — Synthetic Job Postings for Skill Matching | Synthetic-data framework for skill matching | https://arxiv.org/abs/2402.03242 |

## Regulation and governance

| Instrument | Scope | Link |
|---|---|---|
| NYC Local Law 144 (AEDT) | Mandatory bias audit + candidate notice for automated employment decision tools | https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page |
| EU AI Act — Annex III | Employment/recruitment AI classified high-risk; obligations from Aug 2026 | https://artificialintelligenceact.eu/annex/3/ |
| EU AI Act overview | Regulation 2024/1689; human oversight, transparency, worker notice | https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai |

## Cross-references in AIForge

- [05 Vertical Applications](../) — sibling deployed-AI verticals
- [Legal AI](../06_Legal_AI/) — adjacent high-regulation, compliance-heavy vertical
- [Healthcare and Medical AI](../01_Healthcare_and_Medical_AI/) — another high-risk, fairness-sensitive domain
- [Government and Public Sector AI](../28_Government_and_Public_Sector_AI/) — algorithmic accountability and audit overlap

## Sources

- https://github.com/AnasAito/SkillNER
- https://github.com/OmkarPathak/pyresparser
- https://github.com/xitanggg/open-resume
- https://github.com/IvanIsCoding/ResuLLMe
- https://esco.ec.europa.eu/en
- https://www.onetonline.org/
- https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- https://arxiv.org/abs/1906.09208
- https://arxiv.org/abs/2405.19699
- https://arxiv.org/abs/2404.03086
- https://arxiv.org/abs/2501.04316
- https://arxiv.org/abs/2209.08071
- https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page
- https://artificialintelligenceact.eu/annex/3/
- https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
