# Identity Verification and Document AI

> AI systems that prove a person is who they claim to be at account opening and step-up events, by reading government-issued ID documents, matching the photo to a live selfie, and detecting spoofs, deepfakes, and forged documents.

## Why it matters

Identity verification (IDV) is the front door of every regulated bank and fintech: it gates account opening, satisfies KYC/CDD obligations, and is the single largest source of onboarding drop-off and first-party fraud. Done well it raises auto-approval (straight-through) rates and conversion while keeping fraud losses and AML risk low; done poorly it either lets synthetic identities and money mules in, or blocks legitimate customers behind friction. The threat surface is shifting fast from physical document forgery and "presentation attacks" (printed photos, masks, replayed video) toward generative-AI deepfakes and camera/API "injection attacks" that bypass the device sensor entirely.

## Workflow (the eKYC funnel)

End-to-end onboarding IDV pipeline; each stage produces signals that feed a decision/orchestration layer:

1. **Capture** — user photographs an ID document and takes a selfie/short video (mobile SDK with auto-capture, glare/blur quality gates).
2. **Document classification & OCR** — identify document type and country template; extract visual zone fields and parse the **MRZ** (machine-readable zone) / PDF417 barcode; cross-check that printed text == MRZ == chip (NFC eMRTD read where available).
3. **Document authenticity** — verify security features (holograms, fonts, microprint, fixed layout), tamper/recapture detection, and template consistency to catch forged or "photo-of-a-screen" documents.
4. **Face match (1:1)** — compare the document portrait to the live selfie face embedding; produce a similarity score and threshold decision.
5. **Liveness / PAD & deepfake/injection detection** — confirm a live human is present (active challenge or passive) and reject presentation attacks, replays, deepfakes, and virtual-camera injection.
6. **Data/database verification** — validate name/DOB/address against authoritative or credit-bureau sources; **AML/sanctions/PEP screening**; device, email, phone, and behavioral risk signals.
7. **Decision & orchestration** — fuse all signals into approve / decline / **manual review** (case management); apply risk-based step-up (EDD) and ongoing re-verification.

## Techniques / Models

| Task | Typical ML approach |
|---|---|
| Document detection & type classification | CNN object detection / image classifiers; template matching |
| Text extraction (OCR) | CNN+RNN+CTC text recognizers; OCR-free seq2seq transformers (Donut) |
| Structured field extraction | Layout-aware transformers (LayoutLM/LayoutLMv2/LayoutXLM), DocParser |
| MRZ / barcode parsing | OCR + checksum validation (ICAO 9303); PDF417 decode |
| Document forgery / tamper detection | CNN classifiers, recapture/screen detection, copy-move forensics |
| Face matching (ID-to-selfie) | Deep face embeddings + metric learning; domain-specific nets (DocFace+) |
| Liveness / Presentation Attack Detection (PAD) | Texture/rPPG cues, frequency-domain CNNs, multi-frame/active challenge |
| Deepfake & injection detection | Spatio-temporal CNNs, transformers, frequency-artifact detectors |
| Risk decisioning / fusion | GBDT, logistic models, rules + ML orchestration over signals |
| Entity resolution / dedup | Embedding similarity, graph/identity clustering for synthetic-ID detection |

## Tools, Vendors & Open-Source

| Name | Type | URL |
|---|---|---|
| Onfido (Entrust IDV) | Document + biometric IDV | https://onfido.com |
| Jumio | Document + biometric IDV, AML | https://www.jumio.com |
| Persona | IDV/KYC/KYB orchestration platform | https://withpersona.com |
| Veriff | Document + biometric IDV | https://www.veriff.com |
| Trulioo | Global database identity & KYB | https://www.trulioo.com |
| Socure | US identity & synthetic-fraud (database-first) | https://www.socure.com |
| Sumsub | KYC/AML, high-risk & crypto coverage | https://sumsub.com |
| AU10TIX | High-volume document forensics | https://www.au10tix.com |
| Alloy | Identity decisioning / orchestration | https://www.alloy.com |
| Tesseract OCR | Open-source OCR engine | https://github.com/tesseract-ocr/tesseract |
| PaddleOCR | Open-source OCR toolkit | https://github.com/PaddlePaddle/PaddleOCR |
| Donut | OCR-free document understanding (HF) | https://github.com/clovaai/donut |
| LayoutLMv2 / LayoutXLM | Layout-aware doc models (HF Transformers) | https://huggingface.co/docs/transformers/model_doc/layoutlmv2 |
| InsightFace | Open-source face recognition | https://github.com/deepinsight/insightface |
| FaceForensics++ | Open deepfake detection toolkit | https://github.com/ondyari/FaceForensics |

## Datasets & Benchmarks

| Dataset | Use | Link |
|---|---|---|
| MIDV-500 | ID document analysis in mobile video | https://arxiv.org/abs/1807.05786 |
| MIDV-2020 | 1,000 mock IDs (video/scan/photo), synthetic faces & fields | https://arxiv.org/abs/2107.00396 |
| MIDV-2020 portal | Download & annotations | https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html |
| SROIE / CORD | Receipt/doc key-information extraction (transfers to ID OCR) | https://github.com/clovaai/cord |
| FaceForensics++ | Deepfake detection benchmark (1,000 videos, 4 manipulations) | https://github.com/ondyari/FaceForensics |
| FakeAVCeleb | Audio-video multimodal deepfake dataset | https://arxiv.org/abs/2108.05080 |
| NIST FRTE/FATE | Govt face recognition & PAD/liveness evaluations | https://pages.nist.gov/frvt/ |

> Note: real ID-to-selfie corpora are rarely public due to privacy/security law; researchers use mock/synthetic sets (MIDV) and PAD/deepfake proxies. Production thresholds should be validated on in-house, demographically representative data.

## Regulations & Standards

- **FATF Guidance on Digital Identity (2020)** — risk-based use of digital ID for CDD: https://www.fatf-gafi.org/content/dam/fatf-gafi/guidance/Guidance-on-Digital-Identity-report.pdf
- **NIST SP 800-63-4 / 800-63A** — identity proofing, IAL assurance levels, biometric requirements: https://pages.nist.gov/800-63-4/ ; https://csrc.nist.gov/pubs/sp/800/63/A/4/final
- **ISO/IEC 30107-3** — Presentation Attack Detection testing; APCER/BPCER error metrics: https://www.iso.org/standard/79520.html
- **ICAO Doc 9303** — machine-readable travel documents / MRZ & eMRTD chip standard: https://www.icao.int/publications/pages/publication.aspx?docnum=9303
- **eIDAS / eIDAS 2.0 (EU)** — electronic identification & EU Digital Identity Wallet: https://digital-strategy.ec.europa.eu/en/policies/eidas-regulation
- **BSA/AML, KYC/CDD/EDD** — US Customer Identification Program & due diligence obligations: https://www.fincen.gov/resources/statutes-and-regulations/bank-secrecy-act
- **GDPR / LGPD** — biometric data is "special category"; consent, retention, DPIA: https://gdpr.eu/
- **EU AI Act** — remote biometric identification classed high-risk; transparency for deepfakes: https://artificialintelligenceact.eu/
- **SR 11-7 (US)** — model risk management for the ML models used in IDV decisioning: https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm

## Key papers

- Shi & Jain, *DocFace: Matching ID Document Photos to Selfies* (2018) — https://arxiv.org/abs/1805.02283
- Shi & Jain, *DocFace+: ID Document to Selfie Matching* (2018) — https://arxiv.org/abs/1809.05620
- Kim et al., *OCR-free Document Understanding Transformer (Donut)* (2021) — https://arxiv.org/abs/2111.15664
- Xu et al., *LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding* (2020) — https://arxiv.org/abs/2012.14740
- Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images* (2019) — https://arxiv.org/abs/1901.08971
- Bulatov et al., *MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis* (2021) — https://arxiv.org/abs/2107.00396
- Yu et al., *Deep Learning for Face Anti-Spoofing: A Survey* (2021) — https://arxiv.org/abs/2106.14948

## Cross-references in AIForge

- ../Customer_Onboarding_and_KYC/ — the broader onboarding/KYC funnel this feeds
- ../Fraud_Detection/ — first-party & synthetic-identity fraud signals
- ../Transaction_Monitoring_and_AML/ — sanctions/PEP screening downstream
- ../Regulations_and_Compliance/ — FATF, BSA/AML, eIDAS, EU AI Act detail
- ../Tools_Vendors_and_Platforms/ — full vendor landscape
- ../Datasets_and_Benchmarks/ — banking ML datasets index
- ../../../../01_AI_FUNDAMENTALS_AND_THEORY/Computer_Vision/ — detection & recognition backbones
- ../../../../01_AI_FUNDAMENTALS_AND_THEORY/Vision_Transformers/ — document & face transformers
- ../../../../01_AI_FUNDAMENTALS_AND_THEORY/Generative_Models/ — deepfake generation/detection background
- ../../../../01_AI_FUNDAMENTALS_AND_THEORY/Privacy_and_Security/ — biometric data protection

## Sources

- FATF Guidance on Digital Identity (2020) — https://www.fatf-gafi.org/content/dam/fatf-gafi/guidance/Guidance-on-Digital-Identity-report.pdf
- NIST SP 800-63-4 — https://pages.nist.gov/800-63-4/
- NIST FRVT/FATE programs — https://pages.nist.gov/frvt/
- ISO/IEC 30107-3 PAD testing (iBeta overview) — https://www.ibeta.com/iso-30107-3-presentation-attack-detection-confirmation-letters/
- DocFace / DocFace+ — https://arxiv.org/abs/1805.02283 , https://arxiv.org/abs/1809.05620
- Donut (OCR-free) — https://arxiv.org/abs/2111.15664
- FaceForensics++ — https://github.com/ondyari/FaceForensics
- MIDV-2020 — https://arxiv.org/abs/2107.00396
- Trulioo — https://www.trulioo.com ; Veriff — https://www.veriff.com ; Persona — https://withpersona.com
