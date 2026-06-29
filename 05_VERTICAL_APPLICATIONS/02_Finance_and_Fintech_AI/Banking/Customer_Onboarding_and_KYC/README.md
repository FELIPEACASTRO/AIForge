# Customer Onboarding and KYC

> Digital/eKYC onboarding: how banks and fintechs verify a new customer's identity, run due diligence and AML screening, and convert applicants into funded accounts — increasingly with document AI, biometrics, entity resolution and risk models.

## Why it matters

Onboarding is where regulatory obligation (KYC/CDD), fraud risk, and growth economics collide. Manual KYC is slow and costly, while every extra step in the funnel destroys conversion — industry reports routinely cite 50%+ abandonment in poorly designed flows. ML lets institutions verify identity and screen for financial crime in seconds (document OCR, face match, liveness, sanctions/PEP screening) while keeping false-positive friction low. Get it wrong and you face fines under BSA/AML and FATF expectations, deepfake/synthetic-identity fraud at account opening, or a leaky funnel that bleeds revenue.

## Workflow — the eKYC / onboarding funnel

End-to-end digital onboarding is an orchestrated, multi-step funnel where each stage can pass, fail, or route to manual review:

1. **Capture / pre-fill** — collect PII, device & network signals; pre-fill from bank data aggregation (e.g. Plaid) or government digital ID (eIDAS wallet).
2. **Document verification** — capture government ID; OCR + MRZ parsing extract fields; classify document type; detect tampering/forgery (security features, fonts, copy-move).
3. **Biometric verification** — selfie capture, **liveness / presentation-attack detection (PAD)**, face match between selfie and ID portrait; deepfake/injection-attack detection.
4. **Data & identity validation** — cross-check extracted identity against authoritative sources, credit-header / utility data, and prior records (**entity resolution / dedupe**).
5. **AML screening (CDD)** — fuzzy-match name + DOB against **sanctions, PEP, watchlists, adverse media**; resolve alerts.
6. **Risk scoring & routing** — compute customer risk rating; **EDD** (enhanced due diligence) for high-risk (source of funds/wealth, beneficial ownership/UBO).
7. **Decision & orchestration** — auto-approve, decline, or step-up; a decision/orchestration layer chooses vendors and rules per segment and geography.
8. **Provisioning & ongoing monitoring** — open & fund account; feed into perpetual KYC (pKYC) and transaction monitoring.

Key funnel metrics: pass-rate / **auto-approval rate**, drop-off per step, time-to-onboard, fraud capture rate, false-positive (manual-review) rate, and cost per verified user.

## Techniques / Models

| Approach | Where used in onboarding | Notes |
|---|---|---|
| **Document AI / OCR + MRZ parsing** | Field extraction from IDs, passports, proof-of-address | CNN/Transformer OCR + checksum validation of MRZ |
| **Image forensics / forgery detection** | Detect tampered or synthetic ID documents | Copy-move, font/JPEG artifacts, security-feature checks; PAD for printed/screen replays |
| **Face recognition / 1:1 face match** | Match selfie to ID portrait | Deep metric-learning embeddings; threshold tuned to FMR/FNMR |
| **Liveness / Presentation Attack Detection (PAD)** | Confirm a live person, not a photo/mask/video | Active & passive PAD; evaluated via ISO/IEC 30107-3 (APCER/BPCER) |
| **Deepfake & injection-attack detection** | Block synthetic faces & camera-feed spoofing | Growing threat at account opening |
| **Entity resolution / record linkage** | Dedupe applicants, link to existing customers, UBO graphs | Fuzzy matching (Jaro-Winkler, Levenshtein) + ML/graph linkage |
| **Fuzzy name matching / screening** | Sanctions, PEP, adverse-media name matching | Phonetic + string-distance; NLP for adverse-media relevance/sentiment |
| **GBDT (XGBoost/LightGBM/CatBoost)** | Onboarding fraud & risk scoring on tabular signals | Workhorse for account-opening fraud (e.g. BAF benchmark) |
| **Graph / GNN methods** | Synthetic-identity rings, shared-device/PII networks | Connect applicants via shared attributes |
| **NLP / LLMs** | Adverse-media triage, document summarization, KYC analyst copilots | GenAI increasingly used to draft CDD/EDD narratives |
| **Anomaly detection** | Device/behavioral/velocity outliers at signup | Complements supervised fraud models |

## Tools, Vendors & Open-Source

| Name | Category | Link |
|---|---|---|
| Onfido (Entrust) | Document + biometric IDV | https://onfido.com/ |
| Jumio | IDV, biometric auth, AML screening | https://www.jumio.com/ |
| Veriff | IDV, liveness, fraud | https://www.veriff.com/ |
| Persona | IDV + KYC orchestration | https://withpersona.com/ |
| Sumsub | IDV + AML + orchestration | https://sumsub.com/ |
| iDenfy | IDV / KYC | https://www.idenfy.com/ |
| Microblink | ID/document scanning & OCR | https://microblink.com/ |
| Regula | Document & biometric forensics | https://regulaforensics.com/ |
| ComplyAdvantage | AML / sanctions / PEP / adverse media | https://complyadvantage.com/ |
| LSEG World-Check | Risk-screening data | https://www.lseg.com/en/risk-intelligence/screening-solutions/world-check-kyc-screening |
| Moody's (Orbis/Grid) | KYC data, UBO, screening | https://www.moodys.com/web/en/us/kyc.html |
| NICE Actimize | AML / WatchList Filtering | https://www.niceactimize.com/ |
| Quantexa | Entity resolution / decision intelligence | https://www.quantexa.com/ |
| Plaid | Bank-data aggregation, Identity, IDV | https://plaid.com/ |
| Alloy | Identity decisioning / KYC orchestration | https://www.alloy.com/ |
| ComplyCube | IDV + AML | https://www.complycube.com/ |
| Tesseract OCR | Open-source OCR engine | https://github.com/tesseract-ocr/tesseract |
| PaddleOCR | Open-source OCR/Document AI | https://github.com/PaddlePaddle/PaddleOCR |

## Datasets & Benchmarks

| Dataset | Relevance | Link |
|---|---|---|
| Bank Account Fraud (BAF), NeurIPS 2022 | Account-opening fraud; bias/fairness benchmark (the closest public analog to onboarding) | https://github.com/feedzai/bank-account-fraud |
| IEEE-CIS Fraud Detection | Large tabular fraud benchmark (identity + transaction features) | https://www.kaggle.com/c/ieee-fraud-detection |
| PaySim | Synthetic mobile-money transactions | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| ULB Credit Card Fraud | Classic imbalanced fraud dataset | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Home Credit Default Risk | Applicant data, thin-file onboarding/credit | https://www.kaggle.com/c/home-credit-default-risk |
| Lending Club | Loan applications / underwriting | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| MIDV-500 / MIDV-2020 | ID-document images for OCR/IDV research | http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html |

> Note: real eKYC selfie/ID-document datasets are largely proprietary (privacy/biometric law). MIDV and BAF are the most useful public proxies.

## Regulations & Standards

| Framework | Relevance |
|---|---|
| **FATF Recommendation 10** + Guidance on Digital ID | Global CDD/KYC standard; assurance levels for digital identity |
| **BSA / AML (US, FinCEN CIP)** | Customer Identification Program & CDD rule |
| **EU AMLD / AMLR & AMLA** | EU AML obligations; single rulebook + new authority |
| **CDD / EDD / KYB / UBO** | Risk-based diligence; beneficial-ownership identification |
| **eIDAS 2.0 / EUDI Wallet** | EU digital identity wallets for reusable KYC |
| **GDPR / LGPD (Brazil)** | PII & biometric data; consent, minimization |
| **ISO/IEC 30107-3** | Presentation-attack-detection testing (APCER/BPCER) |
| **NIST SP 800-63** | Digital identity assurance levels (IAL/AAL) |
| **SR 11-7 (US Fed/OCC)** | Model risk management for KYC/fraud/risk models |
| **EU AI Act** | Remote biometric ID & risk classification of IDV/scoring models |

## Key papers

- Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation* (BAF, NeurIPS 2022) — https://papers.nips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html
- *Identity Card Presentation Attack Detection: A Systematic Review* — https://arxiv.org/abs/2511.06056
- *Privacy-Aware Detection of Fake Identity Documents (FakeIDet2)* — https://arxiv.org/abs/2508.11716
- *OCR Graph Features for Manipulation Detection in Documents* — https://arxiv.org/abs/2009.05158
- *Zero-to-One IDV: A Conceptual Model for AI-Powered Identity Verification* — https://arxiv.org/abs/2503.08734
- FATF, *Guidance on Digital Identity* (2020) — https://www.fatf-gafi.org/content/dam/fatf-gafi/guidance/Guidance-on-Digital-Identity-report.pdf
- *Tracking financial crime through code and law: a review of regtech in AML/CTF* — https://arxiv.org/abs/2511.15764

## Cross-references in AIForge

- Sibling: [Identity Verification & Document AI](../Identity_Verification_and_Document_AI/)
- Sibling: [Fraud Detection](../Fraud_Detection/)
- Sibling: [Transaction Monitoring & AML](../Transaction_Monitoring_and_AML/)
- Sibling: [Credit Scoring & Underwriting](../Credit_Scoring_and_Underwriting/)
- Sibling: [Regulations & Compliance](../Regulations_and_Compliance/)
- Sibling: [Tools, Vendors & Platforms](../Tools_Vendors_and_Platforms/)
- Sibling: [Open Banking & APIs](../Open_Banking_and_APIs/)
- Fundamentals: [Graph Neural Networks](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Graph_Neural_Networks/)
- Fundamentals: [Computer Vision](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Computer_Vision/)
- Fundamentals: [Tabular Deep Learning](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Tabular_Deep_Learning/)
- Fundamentals: [Anomaly and OOD Detection](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Anomaly_and_OOD_Detection/)
- Fundamentals: [Privacy and Security](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Privacy_and_Security/)

## Sources

- ComplyCube — What is eKYC: https://www.complycube.com/en/what-is-ekyc-electronic-know-your-customer/
- LSEG — KYC process & sanctions screening: https://www.lseg.com/en/risk-intelligence/glossary/kyc/kyc-process
- Jumio — IDV & biometric authentication: https://www.jumio.com/
- Veriff / Persona / Onfido (Entrust) vendor sites (linked above)
- ComplyAdvantage — What is AML screening: https://complyadvantage.com/insights/what-is-aml-screening/
- Quantexa — Decision Intelligence / entity resolution: https://www.quantexa.com/
- Feedzai — Bank Account Fraud dataset (NeurIPS 2022): https://github.com/feedzai/bank-account-fraud
- FATF — Public consultation & Guidance on Digital Identity: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Consultation-digital-id-guidance.html
- ISO/IEC 30107 PAD overview: https://www.duckduckgoose.ai/glossary/iso-iec-30107--biometric-presentation-attack-detection
- arXiv reviews on ID-document PAD & regtech AML (linked above)
