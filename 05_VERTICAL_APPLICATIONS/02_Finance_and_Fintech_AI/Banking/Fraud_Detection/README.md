# Banking Fraud Detection

> Detecting and stopping unauthorized, deceptive, or criminal financial activity — card/payment fraud, account takeover (ATO), and authorized scams (APP fraud) — in real time across the customer and transaction lifecycle.

## Why it matters

Fraud is a direct P&L line for banks and a regulatory obligation, not just a risk function. Global card fraud losses run into tens of billions of dollars annually, and authorized push payment (APP) scams have shifted liability onto banks — in the UK, the PSR's mandatory reimbursement regime (effective 7 October 2024) forces sending and receiving PSPs to refund in-scope victims. Detection must run at payment latency (single- to double-digit milliseconds), under extreme class imbalance (fraud is often <0.5% of volume), and with explainability sufficient to satisfy investigators, customers, and regulators (SR 11-7, EU AI Act).

## Core concepts: end-to-end transaction flow

Real-time transaction fraud is an event-streaming, decisioning, and case-management problem:

1. **Ingest** — authorization/payment events stream in (ISO 8583 for cards, ISO 20022 / Faster Payments / SEPA / wire for transfers) plus session, device, and behavioral signals.
2. **Feature computation** — real-time aggregations (velocity, amount z-scores, time-since-last) joined with profile features from a low-latency feature store; entity resolution links cards, devices, accounts, and counterparties.
3. **Scoring** — one or more models (GBDT, anomaly, GNN, sequence) produce a risk score; a rules/policy engine and consortium/network intelligence combine into a decision.
4. **Decision & orchestration** — approve / decline / step-up (3-D Secure, OTP, biometric) / hold for review, often within an SLA budget.
5. **Case management** — alerts are queued, triaged, investigated, and dispositioned by analysts; SARs/STRs filed where AML-relevant.
6. **Feedback loop** — confirmed fraud, chargebacks, and disputes label data for retraining; concept drift and adversarial adaptation demand continuous monitoring.

**Fraud typologies** to model distinctly: card-not-present (CNP) / card-present, account takeover (credential stuffing, SIM-swap, session hijack), new-account / synthetic-identity fraud (overlaps with [onboarding/KYC](../Customer_Onboarding_and_KYC/)), first-party fraud, mule accounts, and **APP/scam fraud** where the genuine customer is socially engineered into authorizing the payment (so the device/behavior looks legitimate — detection relies on payee risk, beneficiary network, and intent signals).

## Techniques / Models

| Technique | Where it fits | Notes |
|---|---|---|
| Gradient-boosted trees (XGBoost, LightGBM, CatBoost) | Tabular transaction scoring (the workhorse) | Strong on engineered features; fast inference; native categorical handling (CatBoost/LightGBM); dominates Kaggle fraud benchmarks |
| Graph Neural Networks (GCN, GAT, GraphSAGE, RGCN, EvolveGCN) | ATO rings, mule networks, AML, scam beneficiary graphs | Model accounts/devices/counterparties as a graph to surface collusion invisible to per-transaction models; temporal variants for evolving graphs. See [GNNs](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) |
| Anomaly / unsupervised detection (Isolation Forest, autoencoders, one-class SVM) | Cold-start, novel fraud, low-label regimes | Catches patterns with no historical labels; high false-positive risk, usually layered with supervised models. See [Anomaly & OOD](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/) |
| Sequence models (LSTM/GRU, Transformers) | Behavioral sequences, session/transaction histories | Capture order and timing of events; useful for ATO and behavioral biometrics |
| Behavioral analytics / adaptive profiling | Per-entity baselines (Featurespace ARIC, NICE Actimize) | "Self-learning" deviation from each customer's normal behavior |
| Entity resolution + network analytics | Synthetic identity, mule rings, beneficiary risk | Link records to a single real-world entity; graph-based (Quantexa) |
| Document AI / OCR + biometrics / liveness | Onboarding & step-up, deepfake defense | ID document forgery detection, face-match, passive/active liveness — see [Identity Verification & Document AI](../Identity_Verification_and_Document_AI/) |
| Rules + consortium/network intelligence | Policy layer, shared signals | Hard policy controls plus cross-institution fraud signals (Visa/Mastercard, Sardine, consortium feeds) |
| LLMs / structured-data models | Investigator copilots, alert narratives, scam-text analysis | Emerging; used for triage assistance and explanation generation |

Modeling reality: extreme **class imbalance** (resampling/SMOTE, class weighting, focal loss, threshold tuning, PR-AUC/recall@low-FPR over accuracy), **cost-sensitive** evaluation (false declines hurt revenue and CX), **concept drift** monitoring, and **adversarial robustness** (fraudsters probe and adapt). See [Feature Engineering](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Feature_Engineering/) and [Tabular Deep Learning](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Tabular_Deep_Learning/).

## Tools, Vendors & Open-Source

| Name | Type | URL |
|---|---|---|
| Feedzai | Fraud + AML platform (RiskOps) | https://www.feedzai.com/ |
| Featurespace (ARIC, now Visa) | Adaptive behavioral fraud/AML | https://www.featurespace.com/ |
| Hawk AI | AI AML + fraud monitoring | https://hawk.ai/ |
| NICE Actimize | Enterprise fraud/AML analytics | https://www.niceactimize.com/ |
| Quantexa | Graph entity resolution & network analytics | https://www.quantexa.com/ |
| ComplyAdvantage | Screening, transaction monitoring | https://complyadvantage.com/ |
| Unit21 | Fraud/AML, no-code rules + ML | https://www.unit21.ai/ |
| Sardine | Real-time fraud, device & behavior intelligence | https://www.sardine.ai/ |
| SymphonyAI (Sensa) | AML/fraud analytics | https://www.symphonyai.com/ |
| Onfido (Entrust) | ID verification, liveness | https://onfido.com/ |
| Jumio | ID verification + AML | https://www.jumio.com/ |
| Persona | Identity verification workflows | https://withpersona.com/ |
| Veriff | Identity verification | https://www.veriff.com/ |
| Socure | Identity & fraud (ID+) | https://www.socure.com/ |
| Plaid | Account linking, identity & risk signals | https://plaid.com/ |
| Ntropy | Transaction enrichment / categorization | https://www.ntropy.com/ |
| PyOD | Open-source anomaly detection library | https://github.com/yzhao062/pyod |
| DGL / PyG | Open-source GNN frameworks | https://www.dgl.ai/ · https://github.com/pyg-team/pytorch_geometric |
| DGL-FraudDetection | GNN fraud detection examples | https://github.com/safe-graph/DGFraud |
| XGBoost / LightGBM / CatBoost | GBDT libraries | https://github.com/dmlc/xgboost · https://github.com/microsoft/LightGBM · https://github.com/catboost/catboost |

See also the curated [Tools, Vendors & Platforms](../Tools_Vendors_and_Platforms/) page.

## Datasets & Benchmarks

| Dataset | Description | Link |
|---|---|---|
| IEEE-CIS Fraud Detection | ~590K real e-commerce transactions (Vesta), hundreds of anonymized features; benchmark Kaggle competition | https://www.kaggle.com/c/ieee-fraud-detection |
| ULB Credit Card Fraud | 284,807 European card transactions (2013), PCA features, 0.172% fraud; classic imbalance benchmark | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| PaySim | Synthetic mobile-money transactions (~6.3M) with fraud labels | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| Bank Account Fraud (BAF), NeurIPS 2022 | 6× 1M-row realistic tabular account-opening datasets with controlled bias; fairness + imbalance stress test (Feedzai) | https://github.com/feedzai/bank-account-fraud |
| Elliptic (Bitcoin AML) | ~200K BTC transactions as a temporal graph, 166 features, illicit/licit labels | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set |
| Elliptic2 | Subgraph-level money-laundering dataset on the blockchain | https://www.elliptic.co/elliptic2 |
| Fraud Dataset Benchmark (FDB) | Amazon-curated suite of public fraud datasets with a common API | https://github.com/amazon-science/fraud-dataset-benchmark |
| Home Credit Default Risk | Credit-risk (adjacent: first-party/application fraud signals) | https://www.kaggle.com/c/home-credit-default-risk |
| Lending Club | Loan data (adjacent application-fraud/credit modeling) | https://www.kaggle.com/datasets/wordsforthewise/lending-club |

More in [Datasets & Benchmarks](../Datasets_and_Benchmarks/).

## Regulations & Standards

- **FATF Recommendations** — global AML/CFT standard underpinning monitoring and reporting. https://www.fatf-gafi.org/
- **BSA / AML (US)** — Bank Secrecy Act, SAR filing, FinCEN obligations. https://www.fincen.gov/
- **KYC / CDD / EDD** — customer due diligence at onboarding and on an ongoing basis (FinCEN CDD Rule).
- **PSD2 / SCA (EU)** — Strong Customer Authentication and transaction risk analysis exemptions. https://www.eba.europa.eu/
- **UK PSR APP fraud reimbursement** — mandatory reimbursement of in-scope APP scam victims (effective 7 Oct 2024) + Confirmation of Payee. https://www.psr.org.uk/
- **SR 11-7 (US Fed/OCC)** — model risk management guidance governing fraud models. https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- **EU AI Act** — risk-based obligations; credit/fraud-scoring of consumers can fall in higher-risk categories. https://artificialintelligenceact.eu/
- **GDPR (EU) / LGPD (Brazil)** — data protection, automated-decision and explanation rights affecting model use. https://gdpr.eu/ · https://www.gov.br/anpd/
- **Basel framework** — operational-risk capital includes fraud losses. https://www.bis.org/

See [Regulations & Compliance](../Regulations_and_Compliance/).

## Key papers

- Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* (KDD '19 workshop) — Elliptic dataset + GCN/EvolveGCN. https://arxiv.org/abs/1908.02591
- Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (NeurIPS 2022) — the BAF suite. https://papers.nips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html
- *Graph Neural Networks for Financial Fraud Detection: A Review* (2024). https://arxiv.org/abs/2411.05815
- Bellei et al., *The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset* (2024). https://arxiv.org/abs/2404.19109
- *Graph Computing for Financial Crime and Fraud Detection: Trends, Challenges and Outlook* (2021). https://arxiv.org/abs/2103.03227
- Vanini et al. / survey, *Application of Deep Reinforcement Learning to Payment Fraud* (2021). https://arxiv.org/abs/2112.04236
- Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification* (2015) — methodology behind the ULB dataset.

More in [Key Papers & Research](../Key_Papers_and_Research/).

## Cross-references in AIForge

- [Transaction Monitoring & AML](../Transaction_Monitoring_and_AML/) — overlapping financial-crime detection
- [Customer Onboarding & KYC](../Customer_Onboarding_and_KYC/) — new-account & synthetic-identity fraud
- [Identity Verification & Document AI](../Identity_Verification_and_Document_AI/) — document fraud, liveness, deepfake defense
- [Payments & Real-Time Processing](../Payments_and_Real_Time_Processing/) — low-latency scoring pipeline
- [Credit Scoring & Underwriting](../Credit_Scoring_and_Underwriting/) — first-party / application fraud
- [Transaction Categorization & Enrichment](../Transaction_Categorization_and_Enrichment/) — features for scoring
- [Open Banking & APIs](../Open_Banking_and_APIs/) · [Regulations & Compliance](../Regulations_and_Compliance/) · [Tools, Vendors & Platforms](../Tools_Vendors_and_Platforms/) · [Datasets & Benchmarks](../Datasets_and_Benchmarks/) · [Key Papers & Research](../Key_Papers_and_Research/)
- Fundamentals: [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) · [Anomaly & OOD Detection](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/) · [Tabular Deep Learning](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Tabular_Deep_Learning/) · [Feature Engineering](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Feature_Engineering/) · [Classical ML Algorithms](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/) · [Explainable AI](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Explainable_AI/) · [Privacy & Security](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Privacy_and_Security/)

## Sources

- Feedzai — https://www.feedzai.com/ ; AML transaction monitoring — https://www.feedzai.com/solutions/aml-transaction-monitoring/
- Featurespace (ARIC, Visa) — https://www.featurespace.com/
- Hawk AI — https://hawk.ai/ ; NICE Actimize — https://www.niceactimize.com/ ; Quantexa — https://www.quantexa.com/ ; ComplyAdvantage — https://complyadvantage.com/
- Onfido — https://onfido.com/ ; Jumio — https://www.jumio.com/ ; Persona — https://withpersona.com/ ; Veriff — https://www.veriff.com/
- IEEE-CIS — https://www.kaggle.com/c/ieee-fraud-detection ; ULB — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud ; PaySim — https://www.kaggle.com/datasets/ealaxi/paysim1
- BAF suite — https://github.com/feedzai/bank-account-fraud ; FDB — https://github.com/amazon-science/fraud-dataset-benchmark
- Elliptic GCN paper — https://arxiv.org/abs/1908.02591 ; Elliptic2 — https://arxiv.org/abs/2404.19109 ; GNN fraud review — https://arxiv.org/abs/2411.05815
- UK PSR APP reimbursement — https://www.psr.org.uk/ ; FATF — https://www.fatf-gafi.org/ ; FinCEN — https://www.fincen.gov/ ; SR 11-7 — https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm ; EU AI Act — https://artificialintelligenceact.eu/
