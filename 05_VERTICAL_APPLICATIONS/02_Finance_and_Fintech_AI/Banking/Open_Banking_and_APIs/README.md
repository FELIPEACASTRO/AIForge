# Open Banking and APIs

> Standardized, consent-based, API-driven access to bank account data and payment initiation — the connective tissue (PSD2, FAPI, Open Finance, Pix) that turns raw bank data into ML-ready signals for fintech.

## Why it matters

Open banking converts what used to be siloed, screen-scraped bank statements into clean, permissioned, real-time API feeds — the single richest source of behavioral signal in consumer fintech. Aggregated account, balance, and transaction data powers nearly every downstream banking ML model: cash-flow underwriting, fraud and AML monitoring, transaction categorization, income verification, and PFM. For practitioners, the API layer (and its regulatory consent rails) is the upstream dependency that determines data coverage, freshness, schema quality, and whether a model can even be deployed in a given market.

## Core concepts — how it works end-to-end

1. **Consent & authorization.** The user authorizes a third party (a TPP — AISP for data, PISP for payments) to access their bank. Auth uses OAuth 2.0 / OpenID Connect hardened by the **FAPI** security profile (PKCE, mTLS or private_key_jwt, signed request objects, sender-constrained tokens). PSD2 mandates **Strong Customer Authentication (SCA)** — two of: knowledge, possession, inherence — with dynamic linking of amount + payee.
2. **Connection / aggregation.** An aggregator (Plaid, TrueLayer, Tink, Belvo) abstracts thousands of bank connections behind one API. Where regulated APIs exist (EU, UK, Brazil), it uses them; otherwise it may fall back to credential-based connections. Output: accounts, balances, holders, and a transaction stream.
3. **Ingestion & normalization.** Raw transaction records (messy descriptors, MCCs, counterparties, timestamps, amounts) are pulled via webhooks/polling and normalized into a canonical schema.
4. **Enrichment.** ML cleans the descriptor: merchant extraction, **categorization**, recurrence/subscription detection, income/payroll identification, counterparty entity resolution. This is the bridge from "raw API data" to "features."
5. **Scoring / decisioning.** Enriched, aggregated features feed cash-flow underwriting, affordability, fraud, AML, and PFM models. **Payment initiation (PIS / Pix / Pay-by-Bank)** closes the loop by moving money A2A.
6. **Monitoring & re-consent.** Consent expires (PSD2: 90/180-day re-auth cycles), connections break, and schemas drift — production systems need connection-health monitoring and graceful degradation.

## Techniques / Models

| Task | ML approach | Notes |
|---|---|---|
| Transaction categorization | GBDT / fine-tuned transformers / LLMs over descriptor text + amount | Short, noisy text; class imbalance; hierarchical taxonomies |
| Merchant / counterparty extraction | NER + entity resolution + embedding retrieval | Map "SQ *COFFEE 0421" → canonical merchant |
| Recurrence / subscription detection | Sequence models, time-series clustering | Detect recurring debits, income cadence |
| Income & cash-flow estimation | Aggregation features + regression / GBDT | Powers cash-flow underwriting where bureau data is thin |
| Fraud / account-takeover on A2A payments | GBDT + GNN + sequence/anomaly models | Real-time scoring on payment initiation events |
| Account aggregation quality / connection health | Anomaly detection, drift detection | Detect broken/stale connections, schema changes |
| Synthetic / scam payment detection (Pix, APP fraud) | Graph + behavioral models | Mule-account and money-flow graph analysis |
| Consent fraud / synthetic identity at onboarding | Tabular GBDT + device/behavioral biometrics | Ties open-banking data to KYC/identity signals |

## Tools, Vendors & Open-Source

| Name | What it does | URL |
|---|---|---|
| Plaid | US/CA/EU data aggregation, auth, identity, income, payments | https://plaid.com |
| TrueLayer | UK/EU open banking data + Pay-by-Bank (PIS) | https://truelayer.com |
| Tink (Visa) | EU aggregation, enrichment, payments | https://tink.com |
| Belvo | Latin America (BR/MX/CO) Open Finance + Pix via Open Finance | https://belvo.com |
| Pluggy | Brazil Open Finance aggregation | https://pluggy.ai |
| MX | US aggregation + data enhancement / categorization | https://www.mx.com |
| Yapily | EU/UK API-only open banking infrastructure | https://www.yapily.com |
| GoCardless | Bank payments + open banking data | https://gocardless.com |
| Ntropy | Transaction enrichment / categorization API (ML) | https://www.ntropy.com |
| Salt Edge | Global aggregation + PSD2 compliance toolkit | https://www.saltedge.com |
| Ozone API | Open banking / FAPI-compliant API platform (bank side) | https://ozoneapi.com |
| Authlete | FAPI-conformant OAuth/OIDC authorization server | https://www.authlete.com |
| Nordigen / GoCardless Bank Account Data | Free EU bank data API | https://gocardless.com/bank-account-data/ |

## Datasets & Benchmarks

| Dataset | Relevance | Link |
|---|---|---|
| Bank Account Fraud (BAF), NeurIPS 2022 | Realistic, privacy-preserving tabular account-opening fraud suite | https://github.com/feedzai/bank-account-fraud |
| IEEE-CIS Fraud Detection | Large transaction-fraud benchmark | https://www.kaggle.com/c/ieee-fraud-detection |
| ULB Credit Card Fraud | Classic highly imbalanced anomaly benchmark | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| PaySim | Synthetic mobile-money transaction simulator | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| Home Credit Default Risk | Alt-data / cash-flow style credit features | https://www.kaggle.com/c/home-credit-default-risk |
| Lending Club Loan Data | Loan-level outcomes (categorization/underwriting) | https://www.kaggle.com/datasets/wordsforthewise/lending-club |

> Note: there is no large public *consented open-banking transaction* corpus (privacy/regulation); the above are the standard proxies for transaction-ML R&D.

## Regulations & Standards

| Regime | Region | Relevance |
|---|---|---|
| PSD2 + EBA RTS on SCA & CSC | EU/EEA | Mandates open APIs, TPP access, SCA. [EBA RTS](https://www.eba.europa.eu/legacy/regulation-and-policy/regulatory-activities/payment-services-and-electronic-money-0) |
| PSD3 / PSR (proposed) | EU | Successor framework strengthening data access & fraud rules |
| FAPI 1.0 (Baseline/Advanced) & FAPI 2.0 | Global | OAuth/OIDC security profile for financial APIs ([OpenID FAPI WG](https://openid.net/wg/fapi/)) |
| UK Open Banking Standard | UK | OBIE/JROC implementation of regulated open banking |
| CFPB §1033 Personal Financial Data Rights | US | US "open banking" rule (final 2024; staggered compliance, under litigation) — [CFPB](https://www.consumerfinance.gov/personal-financial-data-rights/) |
| Open Finance Brasil + Pix | Brazil | Phased BCB framework; FAPI-BR security profile; Pix initiation via Open Finance |
| GDPR / LGPD | EU / Brazil | Consent, data minimization, purpose limitation on shared data |
| FATF / BSA-AML, KYC/CDD | Global / US | Downstream obligations when open-banking data feeds monitoring |

## Key papers & specs

- *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (BAF), NeurIPS 2022 — https://papers.nips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html
- FAPI Security Profile 1.0 — Part 1 (Baseline) — https://openid.net/specs/openid-financial-api-part-1-1_0.html
- FAPI Security Profile 1.0 — Part 2 (Advanced) — https://openid.net/specs/openid-financial-api-part-2-1_0.html
- EBA RTS on SCA & Common and Secure Communication — https://www.eba.europa.eu/legacy/regulation-and-policy/regulatory-activities/payment-services-and-electronic-money-0
- CFPB Final Rule: Required Rulemaking on Personal Financial Data Rights — https://www.federalregister.gov/documents/2024/11/18/2024-25079/required-rulemaking-on-personal-financial-data-rights
- Pix API specification (Banco Central do Brasil) — https://github.com/bacen/pix-api
- Ntropy — transaction enrichment LLM/SLM benchmark repo — https://github.com/ntropy-network/enrichment_models

## Cross-references in AIForge

- [../Transaction_Categorization_and_Enrichment/](../Transaction_Categorization_and_Enrichment/) — enrichment models that consume open-banking feeds
- [../Payments_and_Real_Time_Processing/](../Payments_and_Real_Time_Processing/) — Pix / Pay-by-Bank / PIS scoring
- [../Fraud_Detection/](../Fraud_Detection/) — A2A and account-opening fraud
- [../Transaction_Monitoring_and_AML/](../Transaction_Monitoring_and_AML/) — downstream AML on shared data
- [../Credit_Scoring_and_Underwriting/](../Credit_Scoring_and_Underwriting/) — cash-flow underwriting from aggregated data
- [../Customer_Onboarding_and_KYC/](../Customer_Onboarding_and_KYC/) and [../Identity_Verification_and_Document_AI/](../Identity_Verification_and_Document_AI/) — consent + identity linkage
- [../Regulations_and_Compliance/](../Regulations_and_Compliance/) — PSD2/§1033/GDPR/LGPD detail
- [../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) — graph models for money-flow / mule detection
- [../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/) — connection-health & fraud anomaly methods
- [../../../../01_AI_FUNDAMENTALS_AND_THEORY/Natural_Language_Processing/](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Natural_Language_Processing/) — descriptor parsing / merchant NER
- [../../../../01_AI_FUNDAMENTALS_AND_THEORY/Privacy_and_Security/](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Privacy_and_Security/) — consent, differential privacy, data minimization

## Sources

- EBA — RTS on SCA & secure communication under PSD2: https://www.eba.europa.eu/legacy/regulation-and-policy/regulatory-activities/payment-services-and-electronic-money-0
- ECB — The revised Payment Services Directive (PSD2): https://www.ecb.europa.eu/press/intro/mip-online/2018/html/1803_revisedpsd.en.html
- OpenID Foundation — FAPI Working Group: https://openid.net/wg/fapi/
- CFPB — Personal Financial Data Rights (§1033): https://www.consumerfinance.gov/personal-financial-data-rights/
- Banco Central do Brasil — Open Finance: https://www.bcb.gov.br/en/financialstability/open_finance
- Banco Central do Brasil — Pix API repo: https://github.com/bacen/pix-api
- Open Finance Brasil — developer/spec wiki: https://openfinancebrasil.atlassian.net/wiki/spaces/OF/overview
- Plaid: https://plaid.com — TrueLayer: https://truelayer.com — Belvo: https://belvo.com — Ntropy docs: https://docs.ntropy.com/enrichment/introduction
- Feedzai — Bank Account Fraud dataset: https://github.com/feedzai/bank-account-fraud
