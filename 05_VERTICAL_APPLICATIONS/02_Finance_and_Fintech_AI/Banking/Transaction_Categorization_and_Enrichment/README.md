# Transaction Categorization and Enrichment

> Turning raw, cryptic bank/card transaction strings (e.g. `SQ *COFFEE 8675309 BROOKLYN NY`) into structured, verified records: clean merchant name, logo, normalized location, merchant ID, MCC, and a spending category.

## Why it matters

Transaction enrichment is the foundation layer of modern retail banking, PFM, SMB lending, and card products: nothing downstream (budgeting, cash-flow underwriting, fraud signals, subscription detection, dispute resolution) works well on raw descriptor strings. Banks and fintechs use it to power customer-facing money-management features, to feed credit-decisioning models with clean cash-flow data, and to reduce "what is this charge?" support tickets and friendly-fraud disputes. Because the raw input is noisy, abbreviated, multilingual, and inconsistent across networks/acquirers, this is a genuinely hard NLP + entity-resolution problem rather than a simple lookup.

## Workflow

End-to-end enrichment pipeline (ingest -> clean -> resolve -> categorize -> serve):

1. **Ingest** — pull transactions from card networks/processors, core banking, or an aggregator (Plaid, MX, Tink, Finicity/Mastercard, Yodlee). Each record has a descriptor string, amount, sign (debit/credit), date, currency, and sometimes a raw MCC.
2. **Parse & clean (merchant cleaning)** — strip acquirer prefixes (`SQ *`, `TST*`, `PAYPAL *`), store numbers, terminal IDs, city/state tokens, phone numbers, and dates. Normalize casing/encoding. This is the highest-leverage step: a clean merchant name drives everything else.
3. **Entity resolution / brand resolution** — map the cleaned string to a canonical merchant entity (brand, parent company, stable merchant ID). Combine fuzzy/string matching, learned embeddings, and co-occurrence signals (customers who shop at A also shop at B).
4. **Enrich** — attach logo, website, normalized address/geocode, merchant category, and counterparty type (recurring biller, P2P, payroll, transfer).
5. **Categorize** — assign a PFM/accounting category (Groceries, Utilities, Travel, SaaS, Payroll...) reflecting *purchase intent*, not just terminal metadata. MCC is a weak prior here, not ground truth.
6. **Detect patterns** — recurring/subscription detection, income identification, transfers between own accounts, refunds.
7. **Serve & feedback** — expose via API/feed; capture user re-categorizations and lender corrections as labels to retrain.

**Why MCC alone is not enough:** MCCs are self-/acquirer-assigned, coarse (~1000 codes), often misclassified, and vary across a merchant's own locations — so production systems treat MCC as one feature among many, not the answer.

## Techniques / Models

| Technique | Where used | Notes |
|---|---|---|
| Regex / rule-based normalization | Merchant string cleaning, prefix/terminal stripping | Cheap, high precision; brittle to new formats; first stage of most pipelines |
| Fuzzy / approximate string matching | Brand resolution against a merchant gazetteer | Levenshtein/Jaro-Winkler, token-set ratio; struggles with abbreviations |
| TF-IDF / n-gram + linear models (LogReg, SVM) | Baseline categorization | Strong, interpretable baseline on description text |
| Gradient-boosted trees (XGBoost/LightGBM) | Categorization with text + amount + MCC + time features | Common production workhorse for tabular+text features |
| Word/char & merchant embeddings | Entity resolution, semantic categorization | fastText/BERT text embeddings; merchant embeddings from co-occurrence graphs |
| Graph / co-occurrence embeddings | Merchant entity resolution by "correlated shoppers" | e.g. DeepTrax bipartite account–merchant graphs |
| Sequence / Transformer models | Contextual transaction embeddings, recurring detection | Pretrained autoregression over transaction sequences (foundation models) |
| RNN/Transformer hybrids | Categorization + cash-flow prediction (PFM/BFM) | Jointly model history for SMB business financial management |
| LLMs (fine-tuned or prompted) | Low-resource/long-tail merchants, SMB descriptors, synthetic labels | Used when no MCC/lookup hit exists; calibration needed for label balance |
| Synthetic data generation | Cold-start & imbalanced SMB categories | Generate realistic descriptors to augment scarce labels |
| Active learning / human-in-the-loop | Label collection from user/lender corrections | Closes the retraining loop, targets low-confidence cases |

## Tools, Vendors & Open-Source

| Name | Type | What it does | Link |
|---|---|---|---|
| Plaid Enrich | Aggregator + enrichment | Merchant, category, location, counterparty enrichment (US/CA) | https://plaid.com/products/enrich/ |
| Ntropy | Enrichment API (LLM-based) | Consumer + SMB categorization, merchant cleaning | https://www.ntropy.com/ |
| MX | Aggregator + enrichment | Categorization engine, cleansing, PFM insights | https://www.mx.com/products/data-enhancement/ |
| Tink (Visa) | Aggregator + enrichment | PFM categories, enriched data API | https://docs.tink.com/api-data-enrichment |
| Finicity (Mastercard) | Aggregator + enrichment | Transaction categorization, FCRA-compliant data | https://www.finicity.com/manage/transactions/ |
| Yodlee (Envestnet) | Aggregator + enrichment | AI/ML transaction data enrichment | https://www.yodlee.com/transaction-data-enrichment |
| Spade | Issuer/card-side enrichment | Real-time, authorization-time merchant enrichment | https://www.spadeapi.com/ |
| Heron Data | SMB enrichment | Categorizes business bank transactions for lenders | https://www.herondata.io/ |
| Tapix | Enrichment | Merchant clean-up, logos, categories for banks | https://www.tapix.io/ |
| Meniga | PFM platform | Categorization + money-management for banks | https://www.meniga.com/ |
| Featurespace | Research / models | Foundation purchasing model + fraud (transaction embeddings) | https://github.com/Featurespace/foundation-model-paper |
| Open Banking Tracker | Directory | Comparison of enrichment/merchant-intelligence vendors | https://www.openbankingtracker.com/embedded-finance/category/transaction-enrichment |

## Datasets & Benchmarks

Public, fully-labeled *categorization* datasets are scarce (data is proprietary/PII-sensitive); related transaction datasets used for modeling, embeddings, and fraud benchmarks:

| Dataset | Use | Link |
|---|---|---|
| PaySim (synthetic mobile money) | Synthetic transaction sequences | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| Sparkov / synthetic credit card transactions | Merchant & category fields, simulation | https://www.kaggle.com/datasets/kartik2112/fraud-detection |
| ULB Credit Card Fraud | Anonymized real card transactions (PCA features) | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| IEEE-CIS Fraud Detection | Rich card-transaction features incl. merchant/device | https://www.kaggle.com/competitions/ieee-fraud-detection |
| Bank Account Fraud (BAF / NeurIPS 2022) | Realistic tabular banking benchmark | https://github.com/feedzai/bank-account-fraud |
| Featurespace foundation-model results | Scripts/benchmarks for transaction-sequence pretraining | https://github.com/Featurespace/foundation-model-paper |

> Note: most of the above carry fraud/amount labels rather than PFM category labels — useful for embeddings, recurring detection, and feature engineering, but production categorization typically relies on proprietary labeled corpora plus user feedback.

## Regulations & Standards

- **GDPR / LGPD / CCPA** — transaction data is personal data; enrichment and profiling trigger purpose-limitation, minimization, and consent rules.
- **PSD2 / Open Banking (UK/EU)** — governs consented access to account/transaction data that feeds enrichment.
- **GLBA / FCRA (US)** — when enriched/categorized cash-flow data is used for credit decisions it can fall under FCRA; vendors advertise FCRA-compliant feeds.
- **SR 11-7 (US Fed/OCC) & EU AI Act** — model risk management / governance apply when categorization feeds underwriting or other consequential decisions.
- **MCC standard (ISO 18245)** — the merchant category code scheme itself; treat as input signal, not authoritative ground truth.

## Key papers

- DeepTrax: Embedding Graphs of Financial Transactions — https://arxiv.org/abs/1907.07225
- Towards a Foundation Purchasing Model: Pretrained Generative Autoregression on Transaction Sequences (Featurespace, ICAIF '23) — https://arxiv.org/abs/2401.01641
- Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation (2025) — https://arxiv.org/abs/2508.05425
- Deep learning enhancing banking services: hybrid transaction classification and cash-flow prediction (Journal of Big Data, 2022) — https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00651-x
- Bank transactions embeddings help to uncover current macroeconomics — https://arxiv.org/abs/2110.12000
- Pre-trained Embeddings for Entity Resolution: An Experimental Analysis — https://arxiv.org/abs/2304.12329

## Cross-references in AIForge

- [Fraud Detection](../Fraud_Detection/) — enriched merchant features feed fraud models
- [Transaction Monitoring and AML](../Transaction_Monitoring_and_AML/) — counterparty/entity resolution overlaps
- [Credit Scoring and Underwriting](../Credit_Scoring_and_Underwriting/) — cash-flow underwriting consumes categorized data
- [Open Banking and APIs](../Open_Banking_and_APIs/) — aggregators that source the raw transactions
- [Conversational and Agentic Banking](../Conversational_and_Agentic_Banking/) — PFM/assistant features built on enriched data
- [Tools, Vendors and Platforms](../Tools_Vendors_and_Platforms/) — vendor landscape
- [Datasets and Benchmarks](../Datasets_and_Benchmarks/) — banking datasets
- [Regulations and Compliance](../Regulations_and_Compliance/) — GDPR/PSD2/FCRA detail
- [Natural Language Processing](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Natural_Language_Processing/) — text cleaning/classification fundamentals
- [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) — merchant co-occurrence graphs
- [Feature Engineering](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Feature_Engineering/) — transaction feature design
- [Classical ML Algorithms](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/) — GBDT / linear baselines

## Sources

- Plaid Enrich — https://plaid.com/products/enrich/ and https://plaid.com/docs/enrich/
- Ntropy enrichment docs — https://docs.ntropy.com/enrichment/introduction
- MX — What is PFM — https://www.mx.com/blog/what-is-pfm/
- Tink data enrichment API — https://docs.tink.com/api-data-enrichment
- Finicity transactions — https://www.finicity.com/manage/transactions/
- Yodlee transaction data enrichment — https://www.yodlee.com/transaction-data-enrichment
- Open Banking Tracker — transaction enrichment providers — https://www.openbankingtracker.com/embedded-finance/category/transaction-enrichment
- Tapix — why MCC codes don't help much — https://www.tapix.io/resources/post/why-mcc-codes-do-not-help-much-with-payment-categorization
- Ramp — MCC reference guide — https://ramp.com/blog/merchant-category-code-list
- Brex — automated merchant classification — https://medium.com/brexeng/how-we-built-a-mostly-automated-system-to-solve-credit-card-merchant-classification-f9108029e59b
- DeepTrax — https://arxiv.org/abs/1907.07225
- Foundation Purchasing Model — https://arxiv.org/abs/2401.01641
- SME transaction categorization — https://arxiv.org/abs/2508.05425
- Hybrid classification + cash-flow — https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00651-x
- Feedzai Bank Account Fraud dataset — https://github.com/feedzai/bank-account-fraud
