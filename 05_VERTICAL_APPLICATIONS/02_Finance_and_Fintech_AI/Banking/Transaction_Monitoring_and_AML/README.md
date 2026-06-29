# Transaction Monitoring and AML

> Detecting and reporting money laundering, terrorist financing, and sanctions evasion by scoring transactions and relationships against rules, ML models, and watchlists — then routing alerts through case management to SAR/STR filing.

## Why it matters

Anti-money laundering (AML) is a non-negotiable regulatory obligation for any deposit-taking or payments institution; failure carries multi-billion-dollar fines and license risk. Legacy rules-only transaction monitoring (TM) systems generate enormous alert volumes with false-positive rates commonly cited at 90-95%+, burying analysts and inflating compliance cost. ML and graph analytics promise to cut noise, surface organized laundering networks that simple thresholds miss, and explain alerts well enough to satisfy model-risk and audit scrutiny. The hard constraint is explainability and defensibility: a regulator must be able to understand why an alert fired and why a SAR was or was not filed.

## Workflow (end-to-end)

1. **Ingest** — transactions, accounts, KYC/CDD attributes, counterparties, devices, and external data (sanctions/PEP lists, adverse media) land in a stream/lake.
2. **Entity resolution & enrichment** — link accounts to a single customer/beneficial owner; resolve counterparties; enrich with risk attributes, geography, MCC, and prior KYC risk rating.
3. **Screening (real-time)** — name/payment screening against sanctions, PEP, and watchlists at onboarding and per-payment (e.g., SWIFT/wire). Fuzzy name matching, transliteration, and alias handling drive most false positives here.
4. **Detection / scoring** — scenario rules (structuring, rapid movement, high-risk geographies) run alongside ML risk scores and network/graph features. Outputs are alerts with scores and reason codes.
5. **Alert triage & prioritization** — ML ranks/segments alerts; auto-closure or hibernation for low-risk; high-risk routed to analysts. This is the main false-positive-reduction lever.
6. **Case management & investigation** — analyst reviews linked alerts, transactions, network, and KYC; documents disposition.
7. **Reporting** — file a **SAR** (US) / **STR** (most other jurisdictions) with the FIU (e.g., FinCEN) within statutory deadlines; in the US, generally within 30 calendar days of initial detection.
8. **Feedback & model governance** — investigation outcomes feed labels back for retraining; performance, drift, and bias are monitored under model-risk controls (SR 11-7).

## Techniques / Models

| Approach | Where it's used | Notes |
|---|---|---|
| Scenario / threshold rules | Structuring, velocity, high-risk corridors | Transparent, regulator-friendly; high false positives; baseline everyone keeps |
| Gradient-boosted trees (XGBoost/LightGBM/CatBoost) | Alert scoring, FP reduction, fraud-AML overlap | Strong tabular baseline; pairs with SHAP for reason codes |
| Graph / network analytics | Mule networks, layering, hidden BO links | Community detection, shortest-path-to-illicit, motif/typology detection |
| Graph neural networks (GCN, GraphSAGE, EvolveGCN, GAT) | Illicit-node/subgraph classification | EvolveGCN handles temporal graphs; see Elliptic work |
| Anomaly / unsupervised detection | Novel typologies, peer-group outliers | Isolation Forest, autoencoders, peer-group/clustering; useful where labels are scarce |
| Sequence models (RNN/Transformer) | Behavioral drift over a customer timeline | Models transaction sequences vs. expected profile |
| Entity resolution | Counterparty & beneficial-owner linkage | Probabilistic/ML matching; foundational for network views |
| Name/fuzzy matching | Sanctions & PEP screening | Edit distance, phonetics, transliteration, alias graphs |
| Document AI / OCR + NLP | Adverse-media screening, EDD evidence | NER + classification over news; LLMs for summarization/triage |
| LLMs / GenAI | Alert narrative generation, SAR drafting, analyst copilot | Emerging; keep human-in-the-loop and auditable |

## Tools, Vendors & Open-Source

| Name | Type | URL |
|---|---|---|
| NICE Actimize | Enterprise AML/TM, watchlist filtering | https://www.niceactimize.com/ |
| SAS Anti-Money Laundering | Enterprise analytics AML | https://www.sas.com/en_us/software/anti-money-laundering.html |
| Quantexa | Entity resolution + network analytics | https://www.quantexa.com/solutions/aml/ |
| Feedzai | AI fraud + AML TM | https://www.feedzai.com/solutions/aml-transaction-monitoring/ |
| Featurespace (now Visa) | Adaptive Behavioral Analytics | https://www.featurespace.com/ |
| Hawk (Hawk AI) | AI TM + screening | https://hawk.ai/ |
| Napier AI | TM + client screening | https://www.napier.ai/ |
| ComplyAdvantage | AML data, sanctions/PEP/adverse media | https://complyadvantage.com/ |
| SymphonyAI Sensa | Financial-crime AI suite | https://www.symphonyai.com/financial-services/ |
| Ntropy | Transaction enrichment API | https://www.ntropy.com/ |
| OpenSanctions | Open sanctions/PEP data | https://www.opensanctions.org/ |
| IBM AMLSim | Synthetic AML simulator | https://github.com/IBM/AMLSim |
| Feedzai research-aml-elliptic | GNN baselines on Elliptic | https://github.com/feedzai/research-aml-elliptic |

## Datasets & Benchmarks

| Dataset | Description | Link |
|---|---|---|
| Elliptic Bitcoin | ~204k tx graph, ~46k labeled (≈2% illicit), 166 features | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set |
| Elliptic2 | 122k labeled subgraphs over 49M nodes / 196M edges | https://arxiv.org/abs/2404.19109 |
| IBM Transactions for AML | Large synthetic AML transaction graphs (AMLSim-style) | https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml |
| IBM AMLSim | Multi-agent simulator with injectable laundering typologies | https://github.com/IBM/AMLSim |
| Bank Account Fraud (BAF, NeurIPS 2022) | 6 privacy-preserving tabular fraud datasets w/ bias variants | https://github.com/feedzai/bank-account-fraud |
| PaySim | ~6.3M synthetic mobile-money tx, ~0.13% fraud | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| ULB Credit Card Fraud | 284,807 tx, 492 fraud (0.17%), PCA features | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| IEEE-CIS Fraud Detection | Vesta e-commerce transactions, device+identity features | https://www.kaggle.com/competitions/ieee-fraud-detection |

> Note: most public AML datasets are crypto or synthetic — real bank AML data is rarely shared for privacy/regulatory reasons. Treat them as method-validation, not production proxies.

## Regulations & Standards

- **FATF Recommendations** — global AML/CFT standard (risk-based approach, CDD, sanctions): https://www.fatf-gafi.org/en/topics/fatf-recommendations.html
- **BSA / US AML** — FFIEC BSA/AML Examination Manual: https://bsaaml.ffiec.gov/manual
- **SAR filing (US)** — FinCEN suspicious activity reporting: https://www.fincen.gov/resources/filing-information
- **KYC / CDD / EDD & Beneficial Ownership** — FinCEN BOI: https://www.fincen.gov/boi
- **EU AMLA / AMLR** — single rulebook + EU AML Authority: https://finance.ec.europa.eu/financial-crime/anti-money-laundering-and-countering-financing-terrorism_en
- **Sanctions** — OFAC (US): https://ofac.treasury.gov/ ; plus EU consolidated and UN lists.
- **SR 11-7** — Fed/OCC guidance on model risk management (validation, monitoring): https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- **GDPR / LGPD** — data-protection constraints on screening and profiling; **EU AI Act** — risk classification of AML/credit AI systems.

## Key papers

- Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* (2019): https://arxiv.org/abs/1908.02591
- Bellei et al. (Elliptic/MIT-IBM), *The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset* (2024): https://arxiv.org/abs/2404.19109
- Altman et al. (IBM), *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models* (NeurIPS 2023): https://arxiv.org/abs/2306.16424
- Jullum et al., *Fighting Money Laundering with Statistics and Machine Learning* (2022): https://arxiv.org/abs/2201.04207
- Jesus et al. (Feedzai), *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (NeurIPS 2022, BAF): https://arxiv.org/abs/2211.13358
- Oliveira et al., *GuiltyWalker: Distance to illicit nodes in the Bitcoin network* (2021): https://arxiv.org/abs/2102.05373

## Cross-references in AIForge

- [Fraud Detection](../Fraud_Detection/) — overlapping models; AML vs. fraud distinction
- [Customer Onboarding and KYC](../Customer_Onboarding_and_KYC/) — CDD/EDD that feeds risk rating
- [Identity Verification and Document AI](../Identity_Verification_and_Document_AI/) — screening evidence & onboarding
- [Transaction Categorization and Enrichment](../Transaction_Categorization_and_Enrichment/) — features for TM
- [Payments and Real Time Processing](../Payments_and_Real_Time_Processing/) — real-time screening context
- [Regulations and Compliance](../Regulations_and_Compliance/) — deeper regulatory mapping
- [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) — GNN foundations
- [Knowledge Graphs](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Knowledge_Graphs/) — entity resolution & networks
- [Anomaly and OOD Detection](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/) — unsupervised typology discovery
- [Explainable AI](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Explainable_AI/) — reason codes for defensible alerts

## Sources

- Feedzai — AML Transaction Monitoring: https://www.feedzai.com/solutions/aml-transaction-monitoring/
- Quantexa — AML solutions: https://www.quantexa.com/solutions/aml/
- ComplyAdvantage — top AML data vendors: https://complyadvantage.com/insights/top-aml-data-vendors/
- Elliptic dataset (Kaggle): https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- IBM AMLSim: https://github.com/IBM/AMLSim
- Feedzai BAF dataset: https://github.com/feedzai/bank-account-fraud
- PaySim (Kaggle): https://www.kaggle.com/datasets/ealaxi/paysim1
- ULB Credit Card Fraud (Kaggle): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- IEEE-CIS Fraud Detection (Kaggle): https://www.kaggle.com/competitions/ieee-fraud-detection
- FATF Recommendations: https://www.fatf-gafi.org/en/topics/fatf-recommendations.html
- FFIEC BSA/AML Manual — SAR: https://bsaaml.ffiec.gov/manual/RegulatoryRequirements/04
- Federal Reserve SR 11-7: https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- Weber et al. (2019), arXiv:1908.02591: https://arxiv.org/abs/1908.02591
- Altman et al. (2023), arXiv:2306.16424: https://arxiv.org/abs/2306.16424
