# Banking AI Regulations and Compliance

> The regulatory and model-governance frameworks that constrain how ML/AI is built, validated, deployed, and explained across banking — from FATF/BSA-AML and KYC to model-risk SR 11-7, GDPR/LGPD, PSD2, Basel, and the EU AI Act.

## Why it matters

In banking, the model is only half the product — the other half is the evidence that it is fair, explainable, validated, monitored, and legally defensible. A credit, fraud, or AML model that cannot be explained to a regulator or auditor is a liability regardless of its AUC, and non-compliance carries multi-billion-dollar fines, consent orders, and license risk. This page maps the regulations that govern banking AI to the ML practices they require (validation, reason codes, bias testing, human oversight, audit logging), so practitioners can design models that ship and survive examination.

## Core concepts — how compliance wraps the ML lifecycle

Regulation does not sit beside the model; it constrains every stage of the lifecycle. A defensible banking-ML program threads governance through end to end:

1. **Use-case classification** — decide the risk tier (e.g., EU AI Act high-risk for credit scoring; "material model" under SR 11-7). Tier determines documentation and oversight burden.
2. **Data governance** — lawful basis (GDPR Art. 6), purpose limitation, special-category-data rules (Art. 9), retention, and lineage. Fair-lending: exclude/control for prohibited and proxy attributes (ECOA/Reg B in the US).
3. **Development & documentation** — model design, assumptions, feature provenance, and limitations recorded; "conceptual soundness" is an SR 11-7 validation pillar.
4. **Independent validation** — separate team tests conceptual soundness, outcomes analysis (back-testing), and benchmarking; ongoing monitoring for drift.
5. **Explainability & adverse-action** — reason codes for declines (US: adverse-action notices under ECOA/FCRA); meaningful information about logic under GDPR Art. 13-15/22.
6. **Human oversight & contestability** — right to human intervention for solely-automated decisions (GDPR Art. 22; EU AI Act human-oversight requirement).
7. **Deployment controls** — change management, access controls, audit logging, and post-market/ongoing monitoring (EU AI Act Art. 72; SR 11-7 monitoring).
8. **Reporting & retention** — SAR/STR filing (AML), conformity assessment & technical files (EU AI Act Annex IV), examiner-ready model inventory.

## Techniques / Models (and the compliance hook each one triggers)

| Approach | Where used in banking | Compliance hook |
|---|---|---|
| GBDT (XGBoost/LightGBM/CatBoost) | Credit scoring, fraud/AML alert scoring | Needs SHAP/reason codes for adverse-action & SR 11-7 conceptual soundness |
| Logistic regression / scorecards (WOE) | Regulated credit underwriting | Inherently interpretable; still requires validation & fair-lending testing |
| Graph / GNNs | AML network detection, mule rings | Explainability of network-derived alerts; data-lineage of linked entities |
| Anomaly / unsupervised | Novel AML typologies, fraud | Hard to explain — pair with rules/post-hoc explanations for defensibility |
| Sequence models (RNN/Transformer) | Behavioral fraud, transaction monitoring | Logging + drift monitoring; opacity raises model-risk tier |
| Document AI / OCR + NER | KYC doc capture, EDD evidence | Biometric/special-category data rules; audit trail of extraction |
| Biometrics / liveness | eKYC face match, anti-spoofing | GDPR Art. 9 biometric data; bias testing across demographics |
| Entity resolution | Beneficial-ownership, sanctions screening | Accuracy/false-match governance; data-minimization |
| LLMs / GenAI | SAR drafting, analyst copilots, summarization | EU AI Act GPAI duties; human-in-the-loop; hallucination/audit controls |
| Explainability (SHAP, LIME, counterfactuals) | Cross-cutting | The mechanism that produces reason codes and validation evidence |
| Fairness/bias tooling (AIF360, Fairlearn) | Credit, fraud thresholds | ECOA/Reg B disparate-impact testing; EU AI Act bias-monitoring |

## Tools, Vendors & Open-Source

| Name | Type | URL |
|---|---|---|
| ModelOp | Model governance / SR 11-7 lifecycle | https://www.modelop.com/ |
| Credo AI | AI governance & EU AI Act compliance | https://www.credo.ai/ |
| Holistic AI | AI risk & governance platform | https://www.holisticai.com/ |
| Fiddler AI | Model monitoring & explainability | https://www.fiddler.ai/ |
| Arthur | ML monitoring, bias & performance | https://www.arthur.ai/ |
| Monitaur | ML governance & assurance | https://www.monitaur.ai/ |
| ComplyAdvantage | AML data: sanctions/PEP/adverse media | https://complyadvantage.com/ |
| NICE Actimize | Enterprise AML/fraud + governance | https://www.niceactimize.com/ |
| Quantexa | Entity resolution & network analytics | https://www.quantexa.com/ |
| SAS Model Risk Management | Model inventory & validation | https://www.sas.com/en_us/software/model-risk-management.html |
| OpenSanctions | Open sanctions/PEP datasets | https://www.opensanctions.org/ |
| AI Fairness 360 (IBM) | Open-source bias metrics & mitigation | https://github.com/Trusted-AI/AIF360 |
| Fairlearn | Open-source fairness assessment | https://fairlearn.org/ |
| SHAP | Open-source model explanations | https://github.com/shap/shap |

## Datasets & Benchmarks

Compliance research uses the same public datasets as fraud/credit/AML — chosen here because several ship with explicit **bias variants** for fair-lending and AI-Act testing.

| Dataset | Why relevant to compliance | Link |
|---|---|---|
| Bank Account Fraud (BAF, NeurIPS 2022) | 6 tabular datasets w/ controlled bias variants for fairness eval | https://github.com/feedzai/bank-account-fraud |
| Home Credit Default Risk | Underwriting; fairness & adverse-action study | https://www.kaggle.com/competitions/home-credit-default-risk |
| Lending Club | Credit-decisioning fairness & explainability research | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| IEEE-CIS Fraud Detection | Fraud-model validation & explainability | https://www.kaggle.com/competitions/ieee-fraud-detection |
| ULB Credit Card Fraud | Imbalanced-class validation methodology | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| PaySim | Synthetic AML — safe to share for governance demos | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| Elliptic Bitcoin | AML graph; explainability of network alerts | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set |

## Regulations & Standards

**Financial-crime (AML/CFT)**
- **FATF Recommendations** — global AML/CFT standard, risk-based approach (R1), CDD, sanctions: https://www.fatf-gafi.org/en/topics/fatf-recommendations.html
- **FATF RBA for the Banking Sector** (guidance): https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Risk-based-approach-banking-sector.html
- **BSA / US AML** — FFIEC BSA/AML Examination Manual: https://bsaaml.ffiec.gov/manual
- **KYC / CDD / EDD & Beneficial Ownership** — FinCEN BOI: https://www.fincen.gov/boi
- **SAR filing (US)** — FinCEN: https://www.fincen.gov/resources/filing-information
- **EU AMLA / AMLR** — single rulebook + EU AML Authority: https://finance.ec.europa.eu/financial-crime/anti-money-laundering-and-countering-financing-terrorism_en
- **Sanctions** — OFAC (US): https://ofac.treasury.gov/

**Model risk & prudential**
- **SR 11-7** — Fed/OCC supervisory guidance on model risk management (development, validation, governance): https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- **SR 21-8** — interagency statement on model risk management for BSA/AML systems: https://www.federalreserve.gov/supervisionreg/srletters/SR2108.htm
- **Basel Framework (BCBS)** — capital/risk standards; supervisory expectations on model governance: https://www.bis.org/basel_framework/

**Data protection & decisioning**
- **GDPR Art. 22** — automated individual decision-making & profiling (right to human intervention): https://gdpr-info.eu/art-22-gdpr/
- **GDPR (full text, EUR-Lex)**: https://eur-lex.europa.eu/eli/reg/2016/679/oj
- **LGPD (Brazil)** — Lei 13.709/2018, incl. automated-decision review rights: https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm
- **ICO guidance** — automated decision-making & profiling: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/individual-rights/individual-rights/rights-related-to-automated-decision-making-including-profiling/

**Payments & open banking**
- **PSD2** — Directive (EU) 2015/2366; **SCA RTS** = Commission Delegated Regulation (EU) 2018/389: https://eur-lex.europa.eu/eli/reg_del/2018/389/oj
- **EBA — payment services / PSD2 RTS hub**: https://www.eba.europa.eu/

**AI-specific**
- **EU AI Act** — Regulation (EU) 2024/1689; credit scoring & life/health insurance pricing are Annex III high-risk; high-risk obligations apply from **2 Aug 2026** (note: an Omnibus reform reached provisional political agreement on 7 May 2026 that would defer some Annex III dates — treat as planning baseline, verify current status): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- **EU AI Act overview** (Commission): https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- **NIST AI RMF 1.0** (Govern/Map/Measure/Manage; voluntary, widely used for assurance): https://www.nist.gov/itl/ai-risk-management-framework
- **NIST AI 100-1 (PDF)**: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

**US fair-lending (credit AI)**
- **ECOA / Regulation B** — adverse-action & nondiscrimination (CFPB): https://www.consumerfinance.gov/rules-policy/regulations/1002/
- **CFPB circular on adverse-action for complex/AI models** (2022-03): https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/

## Key papers

- Goodman & Flaxman, *European Union regulations on algorithmic decision-making and a "right to explanation"* (2016): https://arxiv.org/abs/1606.08813
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP, NeurIPS 2017): https://arxiv.org/abs/1705.07874
- Wachter, Mittelstadt & Russell, *Counterfactual Explanations without Opening the Black Box* (2017): https://arxiv.org/abs/1711.00399
- Barocas, Hardt & Narayanan, *Fairness and Machine Learning* (textbook): https://fairmlbook.org/
- Bellamy et al. (IBM), *AI Fairness 360: An Extensible Toolkit* (2018): https://arxiv.org/abs/1810.01943
- Jesus et al. (Feedzai), *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (BAF, NeurIPS 2022): https://arxiv.org/abs/2211.13358

## Cross-references in AIForge

- [Transaction Monitoring and AML](../Transaction_Monitoring_and_AML/) — FATF/BSA-AML in practice; SAR/STR
- [Customer Onboarding and KYC](../Customer_Onboarding_and_KYC/) — CDD/EDD, eKYC funnel
- [Identity Verification and Document AI](../Identity_Verification_and_Document_AI/) — biometrics/liveness & data rules
- [Fraud Detection](../Fraud_Detection/) — model validation & explainability overlap
- [Credit Scoring and Underwriting](../Credit_Scoring_and_Underwriting/) — fair-lending, adverse-action, EU AI Act high-risk
- [Open Banking and APIs](../Open_Banking_and_APIs/) — PSD2/SCA context
- [Payments and Real Time Processing](../Payments_and_Real_Time_Processing/) — SCA at payment time
- [Explainable AI](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Explainable_AI/) — reason codes & defensibility
- [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/) — GNN foundations for AML

## Sources

- Federal Reserve SR 11-7: https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- Federal Reserve SR 21-8: https://www.federalreserve.gov/supervisionreg/srletters/SR2108.htm
- FATF Recommendations: https://www.fatf-gafi.org/en/topics/fatf-recommendations.html
- FATF RBA Banking Sector: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Risk-based-approach-banking-sector.html
- FFIEC BSA/AML Manual: https://bsaaml.ffiec.gov/manual
- FinCEN BOI: https://www.fincen.gov/boi
- EU AI Act (EUR-Lex, Reg. 2024/1689): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- EU AI Act overview (EC): https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- GDPR Art. 22: https://gdpr-info.eu/art-22-gdpr/
- GDPR full text (EUR-Lex): https://eur-lex.europa.eu/eli/reg/2016/679/oj
- LGPD (Planalto): https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm
- PSD2 SCA RTS (Reg. 2018/389): https://eur-lex.europa.eu/eli/reg_del/2018/389/oj
- EBA: https://www.eba.europa.eu/
- Basel Framework (BIS): https://www.bis.org/basel_framework/
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
- NIST AI 100-1 PDF: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- CFPB Reg B (1002): https://www.consumerfinance.gov/rules-policy/regulations/1002/
- CFPB Circular 2022-03: https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/
- ICO automated decision-making guidance: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/individual-rights/individual-rights/rights-related-to-automated-decision-making-including-profiling/
