# Payments and Real-Time Processing

> Scoring and authorizing payments as they happen — across card, ACH, wire and instant rails — using streaming ML that must decide in tens of milliseconds whether to approve, challenge, or block.

## Why it matters

Payment authorization is the highest-stakes real-time decision in banking: a model has roughly 50–150 ms to score a transaction before the rail times out, and both errors are expensive (a false decline loses revenue and trust; a missed fraud on an instant rail is an irrevocable loss). The shift to **instant, irrevocable rails** — Brazil's Pix (40B+ transactions/year), the US RTP Network and FedNow (settle in under 20 seconds, 24/7/365) — removes the clawback window that batch fraud teams historically relied on, making sub-second inline scoring mandatory. Every market that launched instant payments without strong controls saw **authorized push payment (APP) / scam fraud** explode, pushing detection from the merchant/issuer edge into the bank's own real-time stack.

## Workflow (ingest → features → scoring → decision → case management)

1. **Ingest** — Payment events (card auth, Pix/RTP request, wire) land on a streaming bus (Kafka/Kinesis/Pulsar) with a schema: amount, merchant/MCC, counterparty, device fingerprint, geo, timestamp, account ID.
2. **Feature computation** — A stream processor (Flink, Spark Structured Streaming, ksqlDB) maintains **stateful aggregates**: velocity (count/sum over rolling windows), peer-group baselines, device/IP reputation, counterparty history. Precomputed features are served from a low-latency store (Redis / feature store) for single-digit-ms lookups.
3. **Scoring** — One or more models score the enriched event inline (GBDT for tabular signals, sequence/graph models for behavioral and network context). Output is a calibrated risk score plus reason codes.
4. **Decision / orchestration** — A policy/rules layer combines score, business rules, and rail constraints to **approve / step-up (3DS challenge, OTP) / hold / decline**. For card-not-present, this is the EMV 3-D Secure (3DS2) frictionless-vs-challenge decision.
5. **Case management & feedback** — Borderline events route to analysts; outcomes, chargebacks and confirmed-fraud labels feed back as training labels — with the **label-delay** problem, since ground truth often arrives days or weeks later.

## Techniques / Models

| Approach | Where it's used | Notes |
|---|---|---|
| **Gradient-boosted trees (XGBoost / LightGBM / CatBoost)** | Core inline transaction scoring | Workhorse for tabular fraud; fast inference, strong on engineered velocity/aggregate features |
| **Streaming aggregates + rules** | Velocity, limits, negative lists | First line on instant rails; deterministic, auditable, low-latency |
| **Sequence models (RNN/LSTM/Transformer)** | Behavioral / per-account transaction sequences | Capture temporal patterns and account drift; basis of "adaptive behavioral analytics" |
| **Graph Neural Networks (GNN)** | Mule networks, APP fraud, ring detection | Model accounts/devices/counterparties as a graph; surface multi-hop and camouflaged fraud |
| **Entity resolution / network analytics** | Linking accounts, mules, synthetic identities | Contextual decisioning across fragmented data (e.g. Quantexa-style) |
| **Anomaly / unsupervised detection** | Cold-start, novel attacks, low-label rails | Autoencoders, isolation forest; complements supervised models |
| **Calibration & cost-sensitive learning** | Threshold setting under extreme imbalance | Fraud is <1% of volume; precision-recall and cost curves drive thresholds |
| **Risk-based authentication scoring** | 3DS2 frictionless vs challenge | Issuer-side model decides step-up using device + transaction context |

## Tools, Vendors & Open-Source

| Name | Type | What it does | URL |
|---|---|---|---|
| Feedzai | Vendor | AI-native real-time fraud & financial-crime platform, omnichannel transaction scoring | https://www.feedzai.com/ |
| Featurespace (Visa) | Vendor | Adaptive Behavioral Analytics (ARIC) for real-time fraud & AML | https://www.featurespace.com/ |
| Hawk | Vendor | Explainable ML for transaction fraud + AML across rails (~150 ms) | https://hawk.ai/ |
| Quantexa | Vendor | Entity resolution + graph/network analytics for fraud & financial crime | https://www.quantexa.com/ |
| ComplyAdvantage | Vendor | AI fraud detection (APP, ATO, synthetic ID, scams) + AML screening | https://complyadvantage.com/ |
| NICE Actimize | Vendor | Enterprise fraud, AML and case-management suite | https://www.niceactimize.com/ |
| Stripe Radar | Vendor | ML fraud prevention for card payments / chargebacks | https://stripe.com/radar |
| Sift | Vendor | Real-time fraud scoring on a large cross-app data network | https://sift.com/ |
| Plaid | Vendor | Account data, risk & fraud signals for fintech | https://plaid.com/solutions/fraud/ |
| Apache Kafka | Open-source | Distributed event streaming / ingestion backbone | https://kafka.apache.org/ |
| Apache Flink | Open-source | Stateful stream processing for real-time features & detection | https://flink.apache.org/ |
| Feast | Open-source | Feature store for online (low-latency) + offline serving | https://feast.dev/ |
| Redis | Open-source | In-memory store for sub-ms online feature/state lookups | https://redis.io/ |

## Datasets & Benchmarks

| Dataset | Description | Link |
|---|---|---|
| IEEE-CIS Fraud Detection | ~590K Vesta e-commerce transactions, ~3.5% fraud; rich numeric/categorical features | https://www.kaggle.com/c/ieee-fraud-detection |
| Bank Account Fraud (BAF), NeurIPS 2022 | Realistic, privacy-preserving tabular suite for account-opening fraud; bias/imbalance focus (Feedzai) | https://github.com/feedzai/bank-account-fraud |
| PaySim | 6.3M simulated mobile-money transactions, ~0.13% fraud | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| ULB Credit Card Fraud | 284K European card transactions (PCA features), highly imbalanced | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Fraud Dataset Benchmark (FDB) | Standardized multi-task fraud benchmark + Python API (Amazon) | https://github.com/amazon-science/fraud-dataset-benchmark |

## Regulations & Standards

- **EMV 3-D Secure (3DS2 / 3DS 2.2/2.3)** — EMVCo authentication protocol for card-not-present; risk-based frictionless-vs-challenge flow. https://www.emvco.com/emv-technologies/3-d-secure/
- **PSD2 / Strong Customer Authentication (SCA)** — EU two-factor requirement for electronic payments; defines risk-based exemptions (e.g. Transaction Risk Analysis).
- **PCI DSS** — Payment Card Industry Data Security Standard for handling cardholder data. https://www.pcisecuritystandards.org/
- **Pix / RTP / FedNow rule books** — Scheme rules and emerging fraud controls (limits, confirmation of payee, liability) for instant rails.
- **BSA/AML & FATF** — Suspicious-activity reporting and transaction-monitoring obligations that intersect with payment scoring.
- **SR 11-7** — US model risk management guidance (validation, monitoring) applicable to payment-scoring models.
- **EU AI Act** — Risk-tiered obligations potentially relevant to credit/fraud decisioning systems.

## Key papers

- *Fraud Dataset Benchmark and Applications* — Grover et al., 2022. https://arxiv.org/abs/2208.14417
- *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation (BAF suite)* — Jesus et al., NeurIPS 2022. https://github.com/feedzai/bank-account-fraud
- *Graph Neural Networks for Financial Fraud Detection: A Review* — 2024. https://arxiv.org/abs/2411.05815
- *CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks* — 2024. https://arxiv.org/abs/2402.14708
- *Effective High-order Graph Representation Learning for Credit Card Fraud Detection (HOGRL)* — 2025. https://arxiv.org/abs/2503.01556
- *Application of Deep Reinforcement Learning to Payment Fraud* — 2021. https://arxiv.org/abs/2112.04236
- *Investigation of 3-D Secure's Model for Fraud Detection* — 2020. https://arxiv.org/abs/2009.12390

## Cross-references in AIForge

- [Fraud Detection](../Fraud_Detection/) — model families and detection patterns
- [Transaction Monitoring and AML](../Transaction_Monitoring_and_AML/) — post-event monitoring & SAR workflows
- [Transaction Categorization and Enrichment](../Transaction_Categorization_and_Enrichment/) — feature inputs for scoring
- [Open Banking and APIs](../Open_Banking_and_APIs/) — PSD2 / data access rails
- [Regulations and Compliance](../Regulations_and_Compliance/) — PSD2, SR 11-7, AI Act
- [Tools, Vendors and Platforms](../Tools_Vendors_and_Platforms/) — broader vendor landscape
- [Datasets and Benchmarks](../Datasets_and_Benchmarks/) — banking ML datasets
- [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) — GNN fundamentals
- [Anomaly and OOD Detection](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/) — unsupervised methods
- [Tabular Deep Learning](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Tabular_Deep_Learning/) — models for tabular payment data
- [Online Learning](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Online_Learning/) — streaming / drift-aware learning

## Sources

- Feedzai — https://www.feedzai.com/
- Featurespace — https://www.featurespace.com/
- Hawk — https://hawk.ai/solutions/fraud
- Quantexa real-time payments fraud — https://www.quantexa.com/blog/managing-the-challenges-of-payment-fraud-in-today-s-faster-economy/
- FICO — Real-time payment rails like FedNow need enhanced fraud protection — https://www.fico.com/blogs/real-time-payments-rails-fednow-need-enhanced-fraud-protection
- iPiD — US instant payments fraud crisis (APP fraud 2024) — https://ipid.tech/blog/us-payments-fraud-crisis-instant-payments
- Alloy — RTP vs FedNow — https://www.alloy.com/blog/real-time-payments-rtp-vs-fednow
- EMVCo — EMV 3-D Secure — https://www.emvco.com/emv-technologies/3-d-secure/
- Conduktor — Real-time fraud detection with streaming — https://www.conduktor.io/glossary/real-time-fraud-detection-with-streaming
- IEEE DataPort — IEEE-CIS Fraud Detection — https://ieee-dataport.org/documents/ieee-cis-fraud-detection
- feedzai/bank-account-fraud — https://github.com/feedzai/bank-account-fraud
