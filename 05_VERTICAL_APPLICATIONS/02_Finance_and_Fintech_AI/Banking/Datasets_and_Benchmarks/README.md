# Banking Datasets and Benchmarks

> Public, reproducible datasets and benchmark suites for training and evaluating banking ML models — card fraud, transaction monitoring, account-opening fraud, and credit risk.

## Why it matters

Real banking data is private, regulated, and rarely shareable, so the field is unusually dependent on a small set of public datasets and synthetic simulators. Knowing each one's provenance, label semantics, base rate, and license is essential because most carry severe class imbalance (fraud at 0.1–4%), temporal drift, and bias that break naive train/test splits. Picking the wrong benchmark (e.g. an i.i.d.-shuffled split on temporal fraud data) produces leaked, over-optimistic numbers that collapse in production. These datasets are also the de facto reference for fairness, calibration, and model-risk validation evidence demanded by regulators.

## Core concepts

End-to-end use of a banking dataset for fraud/credit ML:

1. **Provenance & labels** — understand how labels were created (chargebacks, manual review, 90-day delinquency) and the label-maturity lag. Fraud labels arrive late; credit labels mature over months.
2. **Splitting** — split *temporally*, never randomly. Reserve the latest period as out-of-time (OOT) test to measure drift robustness. Group by entity (card/account) to avoid leakage.
3. **Resampling & cost** — handle imbalance with class weights, focal loss, or under/over-sampling on the train fold only; evaluate with PR-AUC, recall@low-FPR, and cost-weighted metrics rather than accuracy.
4. **Feature engineering** — aggregate velocity/RFM features over entity-time windows; for graph data, build card↔device↔email↔IP links.
5. **Evaluation** — report ROC-AUC *and* PR-AUC, recall at fixed alert budget (e.g. FPR ≤ 1%), calibration (Brier/ECE), and slice metrics for fairness.
6. **Validation evidence** — document data lineage, drift, and subgroup performance to satisfy SR 11-7 / EU AI Act model-risk expectations.

## Techniques / Models

| Approach | Where it fits | Notes |
|---|---|---|
| Gradient-boosted trees (XGBoost, LightGBM, CatBoost) | Tabular fraud & credit scoring | Default strong baseline; wins most of these benchmarks |
| Logistic regression / scorecards (WoE/IV) | Credit underwriting | Interpretable, regulator-friendly, ECOA-explainable |
| Graph Neural Networks (GraphSAGE, R-GCN, GAT) | Card-not-present & ring fraud | Exploits card↔device↔email links in IEEE-CIS |
| Anomaly / one-class (Isolation Forest, autoencoders) | Unlabeled / novel fraud | Useful when labels sparse or delayed |
| Sequence models (LSTM, Transformers) | Per-account transaction streams | Captures temporal velocity patterns |
| Entity resolution / record linkage | KYC, AML, dedup | Resolves identities across applications/accounts |
| Cost-sensitive & imbalanced learning (SMOTE, focal loss, class weights) | All fraud datasets | Address 0.1–4% base rates |
| Calibration (Platt, isotonic) | Credit decisioning | Required for PD estimates & pricing |
| Fairness-aware learning (reweighing, post-processing) | Credit & account fraud | Stress-tested directly by BAF variants |

## Tools, Vendors & Open-Source

| Name | Type | URL |
|---|---|---|
| Feedzai | Fraud & financial-crime platform; publisher of BAF | https://feedzai.com |
| Featurespace (now Visa) | Adaptive Behavioral Analytics / ARIC | https://www.featurespace.com |
| Hawk | AML & fraud transaction monitoring | https://hawk.ai |
| Quantexa | Entity resolution & contextual decisioning | https://www.quantexa.com |
| NICE Actimize | Fraud/AML enterprise suite | https://www.niceactimize.com |
| ComplyAdvantage | AML screening & risk data | https://complyadvantage.com |
| Onfido (Entrust) | Document + biometric identity verification | https://onfido.com |
| Jumio | Identity verification & KYC | https://www.jumio.com |
| Persona | Configurable IDV / KYC workflows | https://withpersona.com |
| Veriff | Identity verification | https://www.veriff.com |
| Plaid | Bank data connectivity, IDV & monitoring | https://plaid.com |
| Ntropy | Transaction enrichment / categorization | https://ntropy.com |
| Amazon Fraud Dataset Benchmark (FDB) | Open-source benchmark harness | https://github.com/amazon-science/fraud-dataset-benchmark |
| imbalanced-learn | Resampling library | https://imbalanced-learn.org |

## Datasets & Benchmarks

| Dataset | Domain | Size / base rate | License | Link |
|---|---|---|---|---|
| **IEEE-CIS Fraud Detection** (Vesta) | Card-not-present fraud | 590,540 txns, ~3.5% fraud, 431 features (identity+transaction join) | Competition | https://www.kaggle.com/c/ieee-fraud-detection |
| **Credit Card Fraud (ULB / MLG)** | EU card fraud | 284,807 txns, 492 fraud (0.172%), PCA features V1–V28 | DbCL | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **PaySim** | Mobile-money simulator | ~6.36M txns, ~0.13% fraud, 5 txn types | CC BY-SA 4.0 | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| **Bank Account Fraud (BAF)** suite (NeurIPS 2022) | Account-opening fraud | 6 variants, 6M rows, dynamic + biased | CC BY-NC-ND 4.0 | https://github.com/feedzai/bank-account-fraud |
| **Lending Club (all loans)** | Consumer credit | 2007–2018 accepted + rejected loans | CC0 | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| **Give Me Some Credit** | Credit delinquency | ~150K borrowers, 10 features, 90-day delinquency label | Competition | https://www.kaggle.com/c/GiveMeSomeCredit |
| **Home Credit Default Risk** | Loan default | 307,511 applicants, ~8% default, 7 relational tables | Competition | https://www.kaggle.com/competitions/home-credit-default-risk |
| **Statlog (German Credit)** | Credit risk | 1,000 rows, 20 attributes, good/bad label | CC BY 4.0 | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |

Notes:
- **Temporal split is mandatory** for IEEE-CIS, PaySim, and BAF — all have time columns and drift.
- **PaySim and BAF are synthetic**, generated from real anonymized data, so they sidestep privacy constraints but may understate real-world adversarial complexity.
- **ULB Credit Card** is PCA-anonymized — features are not interpretable, limiting fairness/explainability study.
- **German Credit** ships a documented cost matrix (false-good 5× costlier than false-bad) and is a standard algorithmic-bias testbed (age/sex attributes).

## Regulations & Standards

| Area | Standard | Relevance to datasets |
|---|---|---|
| AML/CFT | FATF Recommendations; US BSA | Define KYC/CDD/EDD obligations driving labels & features |
| Customer due diligence | KYC / CDD / EDD | Account-opening fraud (BAF) maps to onboarding controls |
| Open banking | PSD2 / PSD3 (EU), Section 1033 (US) | Data-sharing rails behind enrichment datasets |
| Credit risk capital | Basel III/IV | PD/LGD modeling context for credit datasets |
| Privacy | GDPR (EU), LGPD (Brazil), GLBA (US) | Why public banking data is synthetic/anonymized |
| Model risk | Fed/OCC SR 11-7; OCC 2011-12 | Validation, OOT testing, documentation expectations |
| Fair lending | ECOA / Reg B, FCRA (US) | Adverse-action explainability; bias testing (German, Home Credit) |
| AI governance | EU AI Act (credit scoring = high-risk) | Mandates data governance, bias, transparency for credit ML |

## Key papers

- Lopez-Rojas, Elmir & Axelsson — *PaySim: A Financial Mobile Money Simulator for Fraud Detection* (EMSS 2016). https://github.com/EdgarLopezPhD/PaySim
- Jesus et al. (Feedzai) — *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (NeurIPS 2022). https://arxiv.org/abs/2211.13358
- Grover et al. (Amazon) — *Fraud Dataset Benchmark and Applications*. https://arxiv.org/abs/2208.14417
- Hofmann, H. — *Statlog (German Credit Data)*, UCI ML Repository. https://doi.org/10.24432/C5NC77
- Dal Pozzolo et al. — *Calibrating Probability with Undersampling for Unbalanced Classification* (ULB credit-card work). https://www.researchgate.net/publication/283349138

## Cross-references in AIForge

- [Fraud Detection](../Fraud_Detection/) — models trained on these datasets
- [Transaction Monitoring & AML](../Transaction_Monitoring_and_AML/)
- [Credit Scoring & Underwriting](../Credit_Scoring_and_Underwriting/) — Lending Club, Home Credit, German
- [Customer Onboarding & KYC](../Customer_Onboarding_and_KYC/) — BAF, eKYC funnel
- [Identity Verification & Document AI](../Identity_Verification_and_Document_AI/)
- [Transaction Categorization & Enrichment](../Transaction_Categorization_and_Enrichment/)
- [Tools, Vendors & Platforms](../Tools_Vendors_and_Platforms/)
- [Regulations & Compliance](../Regulations_and_Compliance/)
- [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) — GNNs for card↔device fraud graphs
- [Anomaly & OOD Detection](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/)
- [Model Evaluation](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Model_Evaluation/) — PR-AUC, calibration, imbalanced metrics
- [Explainable AI](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Explainable_AI/) — adverse-action & SR 11-7 evidence

## Sources

- IEEE-CIS Fraud Detection — https://www.kaggle.com/c/ieee-fraud-detection
- Credit Card Fraud (MLG-ULB) — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- PaySim dataset — https://www.kaggle.com/datasets/ealaxi/paysim1 · simulator https://github.com/EdgarLopezPhD/PaySim
- Bank Account Fraud (BAF) — https://github.com/feedzai/bank-account-fraud · paper https://arxiv.org/abs/2211.13358
- Lending Club — https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Give Me Some Credit — https://www.kaggle.com/c/GiveMeSomeCredit
- Home Credit Default Risk — https://www.kaggle.com/competitions/home-credit-default-risk
- Statlog (German Credit) — https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Amazon Fraud Dataset Benchmark — https://github.com/amazon-science/fraud-dataset-benchmark · https://arxiv.org/abs/2208.14417
