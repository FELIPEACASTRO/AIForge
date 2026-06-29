# Banking AI — Key Papers and Research

> Curated, verifiable research foundation for ML in banking and fintech: fraud/AML detection, credit risk, graph learning on financial networks, plus the datasets, vendors, and regulations practitioners actually ship against.

## Why it matters

Banking ML operates under hard constraints absent from most domains: extreme class imbalance (fraud is often <1% of transactions), concept drift driven by adversaries, sub-100ms decision latency at payment-network scale, and binding regulatory obligations (model validation, explainability, fair lending). Getting the research right is not academic — a 1% lift in fraud recall or a 10% cut in false positives translates directly into millions in losses avoided and customer friction removed. This page maps the literature, data, and tooling that anchor production banking AI.

## Core concepts — how it works end-to-end

**Onboarding / eKYC funnel.** Applicant submits identity → document capture + OCR/document AI → face match + liveness/biometrics → data-source checks (sanctions/PEP/adverse media screening, bureau, device) → identity-fraud risk scoring → CDD/EDD risk tiering → decision (approve / step-up / decline) → ongoing monitoring. Synthetic-identity and account-opening fraud are the dominant failure modes here (see the Bank Account Fraud dataset).

**Transaction monitoring funnel.** Ingest (real-time stream + batch) → feature engineering (velocity, aggregates, behavioral profiles, graph features) → scoring (per-event fraud model + AML typology/anomaly models) → alert/decision (block, challenge, allow) → case management & investigation → SAR/STR filing & feedback labels → model retraining. The graph layer (linking accounts, devices, counterparties) is increasingly where the alpha lives, exposing mule networks and fraud rings invisible to per-transaction models.

## Techniques / Models

| Approach | What it does | Where it's used | Notes |
|---|---|---|---|
| Gradient-boosted trees (XGBoost, LightGBM, CatBoost) | Tabular classification on engineered features | Card fraud, account-opening fraud, credit default | Still the production baseline; wins most tabular benchmarks |
| Graph Neural Networks (GCN, GraphSAGE, GAT, heterogeneous/relational GNNs) | Relational modeling of accounts/devices/counterparties | AML, fraud rings, mule detection | Reported AUROC gains over GBDT by capturing higher-order structure |
| Anomaly / outlier detection (Isolation Forest, autoencoders, one-class) | Unsupervised detection of novel patterns | AML typologies, new-fraud-vector discovery | Useful where labels are scarce/delayed |
| Sequence models (RNN/LSTM, Transformers) | Model temporal behavior of an account/card | Behavioral biometrics, transaction sequences | Captures drift and session dynamics |
| Entity resolution / network linking | Cluster identities across noisy records | KYC, beneficial ownership, network risk | Core of decision-intelligence platforms (e.g., Quantexa) |
| Document AI / OCR | Extract & validate ID docs, statements | eKYC onboarding | Templated + ML extraction, tamper detection |
| Biometrics & liveness | Face match, passive/active liveness | eKYC, step-up auth | Presentation-attack detection is the key risk |
| Sampling / cost-sensitive learning (SMOTE, focal loss, reweighting) | Handle extreme imbalance | All fraud/AML tasks | Evaluate with AUPRC / recall@fixed-FPR, not accuracy |
| Explainability (SHAP, reason codes) | Adverse-action & model-validation evidence | Credit, fraud governance | Required for fair-lending / SR 11-7 |

## Tools, Vendors & Open-Source

| Name | Category | URL |
|---|---|---|
| Feedzai | Fraud + AML transaction monitoring (RiskOps) | https://www.feedzai.com/ |
| Featurespace (now Visa) | Adaptive Behavioral Analytics, FrAML | https://www.featurespace.com/ |
| Hawk (Hawk AI) | AML/CFT, fraud, screening | https://hawk.ai/ |
| ComplyAdvantage | AML data, screening, transaction monitoring | https://complyadvantage.com/ |
| NICE Actimize | Enterprise financial-crime suite | https://www.niceactimize.com/ |
| Quantexa | Decision intelligence / entity resolution & network analytics | https://www.quantexa.com/ |
| Onfido (Entrust) | Document + biometric identity verification | https://onfido.com/ |
| Jumio | eKYC, ID verification, liveness | https://www.jumio.com/ |
| Persona | Configurable identity verification workflows | https://withpersona.com/ |
| Veriff | ID verification + liveness | https://www.veriff.com/ |
| Plaid | Bank account connectivity / data | https://plaid.com/ |
| Ntropy | Transaction enrichment & categorization | https://www.ntropy.com/ |
| PyTorch Geometric (PyG) | Open-source GNN library | https://pytorch-geometric.readthedocs.io/ |
| DGL | Open-source graph deep learning | https://www.dgl.ai/ |
| XGBoost / LightGBM | Open-source GBDT | https://xgboost.readthedocs.io/ , https://lightgbm.readthedocs.io/ |

## Datasets & Benchmarks

| Dataset | Description | Link |
|---|---|---|
| IEEE-CIS Fraud Detection | ~590k real card transactions (Vesta), ~3.5% fraud, anonymized features | https://www.kaggle.com/c/ieee-fraud-detection |
| ULB Credit Card Fraud | 284,807 European card transactions, 492 frauds (0.172%), PCA features | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| PaySim | ~6.3M simulated mobile-money transactions | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| Bank Account Fraud (BAF) | NeurIPS 2022 suite of 6 synthetic, biased/imbalanced tabular datasets (Feedzai) | https://github.com/feedzai/bank-account-fraud |
| Elliptic / Elliptic2 | Labeled Bitcoin transaction graphs for AML (licit/illicit) | https://www.elliptic.co/ |
| Fraud Dataset Benchmark (FDB) | Amazon Science: unified loaders across fraud tasks | https://github.com/amazon-science/fraud-dataset-benchmark |
| Home Credit Default Risk | Credit-default prediction with rich relational tables | https://www.kaggle.com/c/home-credit-default-risk |
| Lending Club | Historical P2P loan-level data for credit modeling | https://www.kaggle.com/datasets/wordsforthewise/lending-club |

## Regulations & Standards

| Framework | Relevance to banking ML |
|---|---|
| FATF Recommendations | Global AML/CFT standard; drives screening & monitoring obligations — https://www.fatf-gafi.org/ |
| BSA / AML (US) | Bank Secrecy Act; SAR filing, transaction monitoring requirements |
| KYC / CDD / EDD | Customer identification, due-diligence, enhanced due-diligence tiers in onboarding |
| PSD2 (EU) | Strong Customer Authentication (SCA) & open banking access |
| Basel framework | Capital/risk requirements; model-driven risk weights |
| GDPR (EU) / LGPD (BR) | Data protection, automated-decision rights, consent |
| SR 11-7 (Fed/OCC) | Model risk management: development, validation, governance — https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm |
| EU AI Act | Risk-tiering of AI systems; credit scoring named high-risk — https://artificialintelligenceact.eu/ |

## Key papers

- Cheng et al., *Graph Neural Networks for Financial Fraud Detection: A Review* (2024) — 100+ studies, unified GNN taxonomy. https://arxiv.org/abs/2411.05815
- Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (NeurIPS 2022) — introduces the BAF suite. https://openreview.net/forum?id=UrAYT2QwOX8
- Grover et al., *FDB: Fraud Dataset Benchmark and Applications* (2022). https://arxiv.org/abs/2208.14417
- Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with GNNs* (2019) — the Elliptic dataset paper. https://arxiv.org/abs/1908.02591
- Bellei et al., *The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset* (2024). https://arxiv.org/abs/2404.19109
- Dou et al., *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters* (CIKM 2020, CARE-GNN). https://arxiv.org/abs/2008.08692

## Cross-references in AIForge

- [../Fraud_Detection/](../Fraud_Detection/) — fraud modeling deep dive
- [../Transaction_Monitoring_and_AML/](../Transaction_Monitoring_and_AML/) — AML pipelines & typologies
- [../Customer_Onboarding_and_KYC/](../Customer_Onboarding_and_KYC/) — eKYC funnel
- [../Identity_Verification_and_Document_AI/](../Identity_Verification_and_Document_AI/) — OCR, liveness, biometrics
- [../Credit_Scoring_and_Underwriting/](../Credit_Scoring_and_Underwriting/) — credit risk ML
- [../Transaction_Categorization_and_Enrichment/](../Transaction_Categorization_and_Enrichment/) — enrichment & merchant ID
- [../Datasets_and_Benchmarks/](../Datasets_and_Benchmarks/) — banking dataset catalog
- [../Regulations_and_Compliance/](../Regulations_and_Compliance/) — regulatory detail
- [../Tools_Vendors_and_Platforms/](../Tools_Vendors_and_Platforms/) — vendor landscape
- [../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) — GNN theory
- [../../../../01_AI_FUNDAMENTALS_AND_THEORY/Knowledge_Graphs/](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Knowledge_Graphs/) — entity/graph fundamentals

## Sources

- Graph Neural Networks for Financial Fraud Detection: A Review — https://arxiv.org/abs/2411.05815
- Feedzai Bank Account Fraud (BAF) suite — https://github.com/feedzai/bank-account-fraud ; https://openreview.net/forum?id=UrAYT2QwOX8
- FDB: Fraud Dataset Benchmark — https://arxiv.org/abs/2208.14417 ; https://github.com/amazon-science/fraud-dataset-benchmark
- IEEE-CIS Fraud Detection — https://www.kaggle.com/c/ieee-fraud-detection
- ULB Credit Card Fraud — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Elliptic AML datasets — https://www.elliptic.co/ ; https://arxiv.org/abs/2404.19109
- SR 11-7 (Federal Reserve) — https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- EU AI Act — https://artificialintelligenceact.eu/
- FATF — https://www.fatf-gafi.org/
- Vendor sites: Feedzai, Featurespace, Hawk, ComplyAdvantage, NICE Actimize, Quantexa, Onfido, Jumio, Persona, Veriff, Plaid, Ntropy (URLs in tables above)
