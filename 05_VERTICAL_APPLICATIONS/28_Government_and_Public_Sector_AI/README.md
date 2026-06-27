# 28 Government and Public Sector AI

> AI applied to public administration — benefits eligibility, tax compliance and fraud detection, citizen-service chatbots, smart-city/transport analytics — and the governance, transparency, and procurement layer (NIST AI RMF, OMB guidance, EU AI Act, UK ATRS) that constrains how the public sector may deploy it.

## Why it matters

Governments are among the largest deployers of consequential AI: systems that decide benefits, flag tax fraud, triage migration cases, and route emergency services directly affect rights and livelihoods. The same decisions carry the heaviest accountability, transparency, and due-process obligations, so the public-sector AI stack is defined as much by governance frameworks as by models. Generative AI alone is projected to deliver hundreds of billions in US public-sector productivity gains by the early 2030s, but mis-deployment (opaque scoring, biased eligibility models) produces high-profile harms. This makes governance, auditability, and transparency first-class engineering requirements, not afterthoughts.

## Taxonomy

| Sub-area | What it covers | Example deployments |
|---|---|---|
| Benefits & eligibility | Means-testing, entitlement determination, error/overpayment detection | Welfare eligibility, Medicare/Medicaid claim screening (CMS) |
| Tax compliance & fraud | Risk scoring of filings, audit selection, collection prioritization | IRS Risk-Based Collection Model, ML audit triage |
| Citizen services | 24/7 chatbots, service discovery, case routing, document Q&A | eCitizen / GovStack chatbot, agency virtual assistants |
| Smart city & transport | Traffic optimization, sensor/IoT analytics, demand forecasting | Signal control, transit planning, utility load |
| Law enforcement & justice | Risk assessment, biometric ID, evidence triage (heavily regulated) | Post-hoc biometric ID (court-authorized under EU AI Act) |
| Migration & border | Asylum case support, identity verification | Annex III high-risk category, EU AI Act |
| Governance & oversight | Risk management, transparency registers, AI use-case inventories | NIST AI RMF, OMB inventories, UK ATRS |

## Key frameworks and governance instruments

| Instrument | Jurisdiction | Role | Link |
|---|---|---|---|
| NIST AI Risk Management Framework (AI RMF 1.0, AI 100-1) | US | Voluntary Govern/Map/Measure/Manage lifecycle framework | https://www.nist.gov/itl/ai-risk-management-framework |
| NIST AI RMF Generative AI Profile (AI 600-1) | US | GenAI-specific risk controls | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf |
| OMB M-24-10 | US | Mandates Chief AI Officers, use-case inventories, minimum risk practices | https://www.gao.gov/blog/inside-irs-use-artificial-intelligence |
| EU AI Act | EU | Risk-tiered regulation; Annex III high-risk public uses | https://artificialintelligenceact.eu/ |
| EU AI Act Annex III (high-risk list) | EU | Biometrics, essential services, law enforcement, migration, justice | https://artificialintelligenceact.eu/annex/3/ |
| EU AI Act Article 5 (prohibited) | EU | Social scoring, untargeted facial scraping, real-time biometric ID | https://artificialintelligenceact.eu/article/5/ |
| UK Algorithmic Transparency Recording Standard (ATRS) | UK | Mandatory transparency register for public-sector algorithms | https://www.gov.uk/government/collections/algorithmic-transparency-recording-standard-hub |
| UK AI Playbook for Government | UK | Practical deployment guidance for departments | https://assets.publishing.service.gov.uk/media/67aca2f7e400ae62338324bd/AI_Playbook_for_the_UK_Government__12_02_.pdf |
| GSA AI guidance & use-case inventory | US | Federal AI adoption and inventory practices | https://www.gsa.gov/technology/government-it-initiatives/artificial-intelligence |

## Key platforms, tools, and datasets

| Resource | Type | Use | Link |
|---|---|---|---|
| GovStack AI Chatbot (EGOV-1) | Building block | Open citizen-service chatbot reference (eCitizen pilot) | https://govstack.gitbook.io/use-cases/use-cases/ai-chatbot-discoverability-government-services |
| awesome-govtech | Curated list | Platforms, tools, datasets, learning for GovTech | https://github.com/brandonhimpfen/awesome-govtech |
| Decidim | Platform | Open-source participatory democracy | https://decidim.org/ |
| Consul | Platform | Citizen participation for governments | https://consulproject.org/ |
| CKAN | Platform | Open data portal software for governments | https://ckan.org/ |
| MOSIP | Platform | Open modular digital identity | https://www.mosip.io/ |
| OpenCRVS | Platform | Civil registration & vital statistics | https://opencrvs.org/ |
| Digital Public Goods Alliance | Registry | Vetted open-source public digital infrastructure | https://digitalpublicgoods.net/ |
| data.gov | Dataset hub | US open government datasets | https://data.gov/ |
| GitHub `govtech` topic | Repos | Active GovTech open-source ecosystem | https://github.com/topics/govtech |

## Key papers

| Paper | Authors / year | Link |
|---|---|---|
| Algorithms and Decision-Making in the Public Sector | Levy, Chasalow, Riley, 2021 | https://arxiv.org/abs/2106.03673 |
| System Cards for AI-Based Decision-Making for Public Policy | Gursoy & Kakadiaris, 2022 | https://arxiv.org/abs/2203.04754 |
| Algorithmic Governance in the United States: Multi-Level Case Analysis (Federal/State/Municipal) | Dedyaev, 2026 | https://arxiv.org/abs/2602.08728 |
| Beyond Ads: Sequential Decision-Making Algorithms in Law and Public Policy | 2021 | https://arxiv.org/abs/2112.06833 |
| Beyond Algorithmic Fairness: Develop & Deploy Ethical AI-Enabled Decision-Support Tools | 2024 | https://arxiv.org/abs/2409.11489 |
| Analysing and Organising Human Communications for AI Fairness Decisions: Public Sector Use Cases | 2024 | https://arxiv.org/abs/2404.00022 |
| Understanding Artificial Intelligence Ethics and Safety (Turing Institute guidance) | Leslie, 2019 | https://arxiv.org/abs/1906.05684 |
| NIST AI RMF 1.0 (AI 100-1) | NIST, 2023 | https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf |

## Reports and evidence

| Source | Focus | Link |
|---|---|---|
| GAO-26-107522 — AI: IRS skills gaps & strategic management | IRS had 126 active AI use cases (June 2025) | https://www.gao.gov/products/gao-26-107522 |
| GAO — Inside the IRS's Use of AI | Risk-Based Collection Model, fraud detection | https://www.gao.gov/blog/inside-irs-use-artificial-intelligence |
| OECD — Governing with AI: AI in Tax Administration (2025) | Cross-country tax administration AI | https://www.oecd.org/en/publications/2025/06/governing-with-artificial-intelligence_398fa287/full-report/ai-in-tax-administration_30724e43.html |
| OECD.AI — Designing transparency for government AI (UK ATRS) | Transparency register lessons | https://oecd.ai/en/wonk/uk-algorithmic-transparency-recording-standard |

## Cross-references in AIForge

- [06 Legal AI — justice/administrative-law AI overlaps with public-sector deployments
- [14 Cybersecurity AI](../14_Cybersecurity_AI/README.md) — fraud detection and critical-infrastructure protection
- [01 Healthcare and Medical AI — regulated public-health systems (CMS, eligibility)
- [Vertical Applications index](../README.md) — sibling regulated-domain verticals

## Sources

- https://www.nist.gov/itl/ai-risk-management-framework
- https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf
- https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
- https://artificialintelligenceact.eu/ , https://artificialintelligenceact.eu/annex/3/ , https://artificialintelligenceact.eu/article/5/
- https://www.gov.uk/government/collections/algorithmic-transparency-recording-standard-hub
- https://assets.publishing.service.gov.uk/media/67aca2f7e400ae62338324bd/AI_Playbook_for_the_UK_Government__12_02_.pdf
- https://www.gao.gov/products/gao-26-107522 , https://www.gao.gov/blog/inside-irs-use-artificial-intelligence
- https://www.oecd.org/en/publications/2025/06/governing-with-artificial-intelligence_398fa287/full-report/ai-in-tax-administration_30724e43.html
- https://govstack.gitbook.io/use-cases/use-cases/ai-chatbot-discoverability-government-services
- https://github.com/brandonhimpfen/awesome-govtech , https://github.com/topics/govtech
- https://decidim.org/ , https://consulproject.org/ , https://ckan.org/ , https://www.mosip.io/ , https://opencrvs.org/ , https://digitalpublicgoods.net/
- https://www.gsa.gov/technology/government-it-initiatives/artificial-intelligence , https://data.gov/
- arXiv: 2106.03673, 2203.04754, 2602.08728, 2112.06833, 2409.11489, 2404.00022, 1906.05684

_Expanded from the seed via a verified high-value research sweep. Contributions welcome (see CONTRIBUTING.md)._
