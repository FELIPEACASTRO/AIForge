# Credit Scoring and Underwriting

> Estimating an applicant's probability of default (PD) and turning it into a lend/no-lend, price, and limit decision — under hard fair-lending and model-risk constraints.

## Why it matters

Credit decisioning is the core P&L engine of any lender: a few basis points of error in the PD estimate, multiplied across a portfolio, is the difference between profit and charge-offs. Unlike most ML, the output is a regulated, contestable, consumer-facing decision — every decline must be explained (ECOA/Reg B adverse-action notices), every model must be independently validated (SR 11-7), and disparate impact on protected classes is illegal regardless of intent. This forces a discipline rare elsewhere in ML: interpretable-by-design models, reject inference, fairness testing, and exhaustive documentation are first-class deliverables, not afterthoughts.

## Core concepts

End-to-end credit decisioning pipeline:

1. **Application intake & bureau pull** — applicant data + tri-bureau report (Experian/Equifax/TransUnion or local equivalent), plus optional cash-flow / alternative data (open banking, telco, rent).
2. **Feature engineering** — bureau attributes, trade-line aggregations, debt-to-income (DTI), utilization, derogatory/inquiry counts; cash-flow features (income stability, NSF events) from bank transactions.
3. **Binning & WOE transform** — continuous features discretized into monotone bins; Weight of Evidence (WOE) encoding makes a logistic scorecard linear in log-odds and auditable.
4. **PD model** — logistic-regression scorecard (still dominant for explainability) or monotone-constrained GBDT; output mapped to a score (e.g., points-to-double-the-odds scaling).
5. **Reject inference** — training data is biased: you only observe repayment for *approved* loans. Inference techniques (parceling, augmentation, fuzzy/EM, self-learning) re-introduce rejects to debias the through-the-door population.
6. **Decision & pricing** — score → cutoff (swap-set analysis), risk-based pricing (APR), credit-limit assignment; downstream LGD/EAD models give expected loss for Basel/IFRS 9 provisioning.
7. **Adverse-action & explanation** — for declines, derive top reason codes (max 4 under Reg B) from the model that made the decision.
8. **Validation, monitoring & governance** — independent validation (SR 11-7), population stability index (PSI), score drift, vintage curves, fair-lending (disparate-impact) testing, and ongoing recalibration.

**Key metrics:** AUC / Gini (Gini = 2·AUC − 1), KS statistic, divergence, PSI (stability), Brier score (calibration), and approval-rate vs. bad-rate trade-off curves.

## Techniques / Models

| Approach | Role in credit | Notes |
|---|---|---|
| Logistic-regression scorecard (WOE + IV) | Incumbent PD model | Monotone, fully interpretable, native reason codes; regulator-friendly |
| Gradient-boosted trees (XGBoost / LightGBM / CatBoost) | High-accuracy PD | Often top of benchmarks; needs SHAP + **monotonic constraints** for compliance |
| Monotonic GBDT / EBM (Explainable Boosting Machine) | Interpretable non-linear | Glass-box GAMs; accuracy near GBDT with additive transparency |
| Reject inference (parceling, augmentation, EM, self-learning, generative) | Debias training sample | Corrects MNAR selection bias from prior decline policy |
| SHAP / counterfactual explanations | Adverse-action reasons, audit | Reason codes should derive from the *deciding* model, not a post-hoc proxy |
| Survival / hazard models (Cox, GBM survival) | Time-to-default, prepayment | Captures *when*, not just *whether*, default occurs |
| Graph neural networks / entity resolution | Synthetic-identity & first-party fraud at origination | Links applicants across shared attributes/devices |
| Tabular deep learning (TabNet, FT-Transformer, TabM) | PD on large alt-data | Rarely beats GBDT on tabular; useful for very high-dim alt-data |
| LLM / document AI | Income & bank-statement extraction, stipulation review | Parse paystubs, tax forms, bank statements for cash-flow underwriting |

## Tools, Vendors & Open-Source

| Name | Type | URL |
|---|---|---|
| OptBinning | OSS — optimal binning & scorecards | https://github.com/guillermo-navas-palencia/optbinning |
| ING skorecard | OSS — sklearn-compatible scorecards | https://github.com/ing-bank/skorecard |
| InterpretML (EBM) | OSS — glass-box GAMs | https://github.com/interpretml/interpret |
| SHAP | OSS — model explanations | https://github.com/shap/shap |
| XGBoost | OSS — GBDT (monotone constraints) | https://github.com/dmlc/xgboost |
| LightGBM | OSS — GBDT | https://github.com/microsoft/LightGBM |
| Fairlearn / AIF360 | OSS — fairness assessment & mitigation | https://github.com/fairlearn/fairlearn |
| Zest AI | Vendor — ML underwriting + fair-lending tooling | https://www.zest.ai/ |
| Upstart | Vendor — alt-data lending platform | https://www.upstart.com/ |
| FICO Platform | Vendor — decisioning & scorecards | https://www.fico.com/ |
| Experian Ascend / PowerCurve | Vendor — bureau + decisioning | https://www.experian.com/ |
| SAS Credit Scoring | Vendor — modeling & MRM | https://www.sas.com/ |
| Plaid | Vendor — open-banking / cash-flow data | https://plaid.com/ |
| Nova Credit | Vendor — cross-border & cash-flow credit data | https://www.novacredit.com/ |

## Datasets & Benchmarks

| Dataset | Task | Link |
|---|---|---|
| Home Credit Default Risk | PD on thin-file + alt-data | https://www.kaggle.com/c/home-credit-default-risk |
| LendingClub loan data | Charged-off vs. fully-paid P2P loans | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| Bank Account Fraud (BAF), NeurIPS 2022 | Origination fraud + fairness stress-test | https://github.com/feedzai/bank-account-fraud |
| Give Me Some Credit | 2-year serious delinquency | https://www.kaggle.com/c/GiveMeSomeCredit |
| German Credit (Statlog) / South German Credit | Classic creditworthiness benchmark | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| Default of Credit Card Clients (Taiwan) | Default prediction | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |

## Regulations & Standards

- **ECOA / Regulation B (12 CFR 1002)** — prohibits credit discrimination; requires adverse-action notices with specific principal reasons (≤4). Reg B text: https://www.consumerfinance.gov/rules-policy/regulations/1002/
- **FCRA** — governs use of credit-report data and adverse-action disclosures tied to bureau scores.
- **CFPB guidance on AI/complex models (2023–24)** — confirms ECOA adverse-action specificity applies to ML/“black-box” models. https://www.consumerfinance.gov/about-us/newsroom/cfpb-issues-guidance-on-credit-denials-by-lenders-using-artificial-intelligence/
- **SR 11-7 / OCC 2011-12** — Model Risk Management: governance, independent validation, monitoring. https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- **Basel III IRB (PD/LGD/EAD)** — internal-ratings-based capital requirements. https://www.bis.org/basel_framework/
- **IFRS 9 / CECL** — expected-credit-loss provisioning driven by PD/LGD/EAD.
- **Fair lending — disparate treatment & disparate impact** — illegal regardless of intent; requires less-discriminatory-alternative search.
- **EU AI Act** — credit-scoring of natural persons classified as **high-risk** (Annex III). https://artificialintelligenceact.eu/
- **GDPR Art. 22 / LGPD** — rights around automated individual decision-making and explanation.

## Key papers

- Lessmann, Baesens, Seow, Thomas (2015), *Benchmarking state-of-the-art classification algorithms for credit scoring* — EJOR. https://www.sciencedirect.com/science/article/abs/pii/S0377221715004208
- Gunnarsson et al. (2022), *Deep Learning vs. Gradient Boosting: Benchmarking ML algorithms for credit scoring*. https://arxiv.org/abs/2205.10535
- Jesus et al. (2022), *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (BAF), NeurIPS. https://arxiv.org/abs/2211.13358
- Bücker et al. (2022), *Transparency, auditability and explainability of ML models in credit scoring*. https://arxiv.org/abs/2009.13384
- Dastile, Celik, Potsane (2020), *Statistical and ML models in credit scoring: A systematic literature survey*. https://doi.org/10.1016/j.asoc.2020.106263
- *Best Practices for Responsible Machine Learning in Credit Scoring* (2024). https://arxiv.org/abs/2409.20536
- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* (SHAP). https://arxiv.org/abs/1705.07874

## Cross-references in AIForge

- [Fraud Detection](../Fraud_Detection/) — origination & first-party fraud overlap
- [Customer Onboarding & KYC](../Customer_Onboarding_and_KYC/) — identity & application intake
- [Transaction Categorization & Enrichment](../Transaction_Categorization_and_Enrichment/) — cash-flow underwriting features
- [Regulations & Compliance](../Regulations_and_Compliance/) — fair lending, SR 11-7, EU AI Act
- [Datasets & Benchmarks](../Datasets_and_Benchmarks/) — banking ML datasets
- [Explainable AI](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Explainable_AI/) — SHAP, counterfactuals, reason codes
- [Tabular Deep Learning](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Tabular_Deep_Learning/) — GBDT vs. deep tabular
- [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) — entity resolution / synthetic identity
- [Causal Inference](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Causal_Inference/) — reject inference & selection bias
- [Uncertainty Quantification](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Uncertainty_Quantification/) — calibration of PD estimates

## Sources

- Federal Reserve SR 11-7 — https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- CFPB, ECOA/Reg B & AI credit denials — https://www.consumerfinance.gov/about-us/newsroom/cfpb-issues-guidance-on-credit-denials-by-lenders-using-artificial-intelligence/
- Reg B (12 CFR 1002) — https://www.consumerfinance.gov/rules-policy/regulations/1002/
- Lessmann et al. (2015), EJOR — https://www.sciencedirect.com/science/article/abs/pii/S0377221715004208
- Gunnarsson et al. (2022), arXiv 2205.10535 — https://arxiv.org/abs/2205.10535
- Feedzai BAF / NeurIPS 2022 — https://github.com/feedzai/bank-account-fraud
- OptBinning — https://github.com/guillermo-navas-palencia/optbinning
- ING skorecard — https://github.com/ing-bank/skorecard
- BIS Basel Framework — https://www.bis.org/basel_framework/
- EU AI Act (Annex III high-risk) — https://artificialintelligenceact.eu/
