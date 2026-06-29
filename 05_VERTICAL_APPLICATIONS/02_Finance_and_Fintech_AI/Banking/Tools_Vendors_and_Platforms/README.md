# Banking AI Tools, Vendors and Platforms

> The vendor and open-source landscape that banks and fintechs use to operationalize ML for fraud, AML, KYC/onboarding, transaction enrichment and decisioning — from real-time scoring engines to identity verification and entity resolution.

## Why it matters

Few banks build financial-crime and onboarding ML entirely in-house: the data scale, sub-100ms latency budgets, model governance, sanctions-list coverage and regulatory burden push most institutions toward specialized platforms or hybrid (vendor + open-source) stacks. Picking the right tool is itself an ML-engineering decision — it determines feature freshness, explainability for auditors, false-positive rates that drive operational cost, and how quickly new fraud typologies can be deployed. This page maps the major commercial vendors and open-source building blocks across the banking ML lifecycle.

## Core concepts

A modern banking ML stack spans two intertwined funnels:

- **Onboarding / eKYC funnel:** capture → document AI/OCR → face match + liveness/anti-spoofing → data extraction → sanctions/PEP/adverse-media screening → identity-fraud + synthetic-identity scoring → CDD/EDD risk rating → decision (approve / review / reject / step-up).
- **Transaction funnel:** ingest (cards, ACH/wire, RTP, open-banking) → feature computation (profiles, velocity, graph, behavioral) → real-time scoring (fraud + AML) → rules + ML fusion → alert/decline → case management & SAR/STR filing → analyst feedback → retrain.

Vendors differentiate on where they sit: **decisioning engines** (Feedzai, Featurespace), **AML/monitoring** (Hawk, NICE Actimize, SAS, Tookitaki), **screening data** (ComplyAdvantage), **entity resolution/network** (Quantexa), **identity** (Onfido/Entrust, Jumio, Veriff, Persona, Sumsub), and **data/enrichment infrastructure** (Plaid, Ntropy). Most banks run several in concert, with open-source for feature stores, graph compute and model serving.

## Techniques / Models

| Technique | Where used | Notes |
|---|---|---|
| Gradient-boosted trees (XGBoost/LightGBM/CatBoost) | Card & payment fraud, app fraud, credit | Tabular workhorse; strong with engineered velocity/profile features |
| Graph Neural Networks / network analytics | AML, mule detection, fraud rings, synthetic ID | Captures relational risk across counterparties/devices |
| Entity resolution | KYC, AML, sanctions, network building | Dedup/link people & companies across messy data |
| Anomaly / unsupervised detection | Novel typologies, AML behavioral peer-group | Autoencoders, isolation forest, clustering; finds unknown-unknowns |
| Sequence models (RNN/Transformer) | Transaction streams, behavioral biometrics | Models temporal spending/session patterns |
| Adaptive behavioral analytics | Scam/APP fraud, account takeover | Per-customer "good behavior" baselines (Featurespace ARIC) |
| Document AI / OCR | eKYC, ID/passport, statement parsing | Field extraction, MRZ/barcode, tamper detection |
| Face match + liveness / PAD | Onboarding biometrics | Presentation-attack & deepfake/injection detection |
| LLMs / GenAI | Alert narratives, SAR drafting, adverse-media triage, natural-language rules | Copilots and summarization layered on detection |
| Explainability (SHAP, whitebox) | Model governance / SR 11-7 | Reason codes for declines and regulator review |

## Tools, Vendors & Open-Source

| Name | Category | URL |
|---|---|---|
| Feedzai | AI-native fraud + AML decisioning (RiskOps) | https://www.feedzai.com/ |
| Featurespace (now Visa) — ARIC | Adaptive behavioral analytics, scam/fraud | https://www.featurespace.com/ |
| Hawk | AI-native AML transaction monitoring & screening | https://hawk.ai/ |
| ComplyAdvantage | Sanctions/PEP/adverse-media data + monitoring | https://complyadvantage.com/ |
| NICE Actimize | Enterprise fraud + AML (IFM, SAM, WL-X) | https://www.niceactimize.com/ |
| SAS | Anti-money laundering & fraud (Viya) | https://www.sas.com/en_us/software/anti-money-laundering.html |
| Oracle Financial Services (Mantas) | AML/compliance suite | https://www.oracle.com/financial-services/ |
| Quantexa | Entity resolution + contextual network analytics | https://www.quantexa.com/ |
| Tookitaki — FinCense / AFC Ecosystem | AML + fraud, community typologies (APAC) | https://www.tookitaki.com/ |
| Sardine | Fraud, KYC/KYB, device & behavior signals | https://www.sardine.ai/ |
| Sift | Digital fraud / trust & safety ML | https://sift.com/ |
| Onfido (Entrust) | Document + biometric identity verification | https://onfido.com/ |
| Jumio | KYC, document, biometrics, AML screening | https://www.jumio.com/ |
| Veriff | Identity verification & liveness | https://www.veriff.com/ |
| Persona | Configurable identity/KYC workflows | https://withpersona.com/ |
| Sumsub | KYC/KYB/AML verification | https://sumsub.com/ |
| Plaid | Open banking data + Enrich (txn enrichment) | https://plaid.com/products/enrich/ |
| Ntropy | Transaction enrichment/categorization API | https://www.ntropy.com/ |
| Featuretools | Open-source automated feature engineering | https://github.com/alteryx/featuretools |
| Feast | Open-source feature store | https://github.com/feast-dev/feast |
| PyTorch Geometric / DGL | Open-source GNN libraries | https://github.com/pyg-team/pytorch_geometric |
| PyOD | Open-source anomaly detection toolkit | https://github.com/yzhao062/pyod |
| dedupe / Splink | Open-source entity resolution / record linkage | https://github.com/moj-analytical-services/splink |

## Datasets & Benchmarks

| Dataset | Description | Link |
|---|---|---|
| IEEE-CIS Fraud Detection | ~590k card-not-present txns (Vesta), Kaggle competition | https://www.kaggle.com/c/ieee-fraud-detection |
| ULB Credit Card Fraud | 284,807 anonymized (PCA) EU card txns, 0.17% fraud | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| PaySim | ~6.3M simulated mobile-money txns | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| Bank Account Fraud (BAF), NeurIPS 2022 | Realistic privacy-preserving tabular suite (Feedzai) | https://github.com/feedzai/bank-account-fraud |
| Fraud Dataset Benchmark (FDB) | Unified benchmark across fraud tasks (AWS) | https://github.com/amazon-science/fraud-dataset-benchmark |
| IBM AMLSim | Synthetic AML transaction-network generator | https://github.com/IBM/AMLSim |
| Home Credit Default Risk | Credit underwriting tabular dataset | https://www.kaggle.com/c/home-credit-default-risk |
| Lending Club | Peer-to-peer loan performance data | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| Elliptic (Bitcoin) | Graph dataset for illicit-flow detection | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set |

## Regulations & Standards

- **FATF Recommendations** — global AML/CFT standard. https://www.fatf-gafi.org/en/topics/fatf-recommendations.html
- **BSA / AML & SAR filing (FinCEN)** — US obligations. https://www.fincen.gov/
- **KYC / CDD / EDD** — customer due-diligence tiers feeding risk rating.
- **PSD2 / SCA** — EU strong customer authentication & open banking. https://eur-lex.europa.eu/eli/dir/2015/2366/oj
- **Basel Framework** — capital/operational-risk context. https://www.bis.org/basel_framework/
- **GDPR (EU) / LGPD (Brazil)** — data protection for PII used in models. https://gdpr.eu/ | https://www.gov.br/anpd/
- **SR 11-7 (US Fed/OCC)** — model risk management & validation. https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- **EU AI Act** — high-risk classification incl. creditworthiness. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

## Key papers

- Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (BAF, NeurIPS 2022). https://arxiv.org/abs/2211.13358
- Grover et al., *Fraud Dataset Benchmark and Applications* (2022). https://arxiv.org/abs/2208.14417
- Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with GCNs* (Elliptic, 2019). https://arxiv.org/abs/1908.02591
- Lopez-Rojas et al., *PaySim: A financial mobile money simulator for fraud detection* (2016). https://www.researchgate.net/publication/313138956
- Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification* (ULB, 2015). https://ieeexplore.ieee.org/document/7376606
- Altman et al., *Realistic Synthetic Financial Transactions for AML* (IBM, NeurIPS 2023). https://arxiv.org/abs/2306.16424

## Cross-references in AIForge

- [Fraud Detection](../Fraud_Detection/)
- [Transaction Monitoring and AML](../Transaction_Monitoring_and_AML/)
- [Customer Onboarding and KYC](../Customer_Onboarding_and_KYC/)
- [Identity Verification and Document AI](../Identity_Verification_and_Document_AI/)
- [Transaction Categorization and Enrichment](../Transaction_Categorization_and_Enrichment/)
- [Open Banking and APIs](../Open_Banking_and_APIs/)
- [Credit Scoring and Underwriting](../Credit_Scoring_and_Underwriting/)
- [Regulations and Compliance](../Regulations_and_Compliance/)
- [Datasets and Benchmarks](../Datasets_and_Benchmarks/)
- [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/)
- [Anomaly and OOD Detection](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/)
- [Tabular Deep Learning](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Tabular_Deep_Learning/)
- [Explainable AI](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Explainable_AI/)
- [Feature Engineering](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Feature_Engineering/)

## Sources

- Feedzai — https://www.feedzai.com/ | RiskOps https://www.feedzai.com/riskops/
- Featurespace ARIC — https://www.featurespace.com/
- Hawk AML transaction monitoring — https://hawk.ai/solutions/aml/transaction-monitoring
- ComplyAdvantage transaction monitoring — https://complyadvantage.com/mesh/transaction-monitoring-software/
- NICE Actimize — https://www.niceactimize.com/
- Quantexa financial crime — https://www.quantexa.com/solutions/financial-crime/
- Tookitaki FinCense / AFC Ecosystem — https://www.tookitaki.com/products/anti-money-laundering-suite
- Jumio / Onfido / Veriff / Persona — vendor sites linked above
- Plaid Enrich — https://plaid.com/products/enrich/ | Ntropy — https://www.ntropy.com/blog/transaction-categorization
- BAF dataset & paper — https://github.com/feedzai/bank-account-fraud , https://arxiv.org/abs/2211.13358
- IEEE-CIS — https://www.kaggle.com/c/ieee-fraud-detection ; ULB — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
