# Finance LLM Benchmarks & NLP Datasets

> A curated, verified catalog of the benchmarks and labeled datasets used to train and evaluate financial NLP models and large language models (LLMs) — question answering, numerical/tabular reasoning, document QA over SEC filings, summarization, sentiment, named-entity/relation extraction, central-bank tone, and agentic/robustness suites. Every entry below was confirmed to exist with a live URL (2024–2026). Free vs paid, task type, size, and Brazil 🇧🇷 equivalents are flagged.

This page complements the repo's existing pages on generic data APIs (yfinance/Polygon/FRED), arXiv q-fin, and working papers. Here the unit is the **labeled NLP/LLM artifact** — a benchmark you score a model against, or a corpus you fine-tune on — not a price feed.

**How to read this page.** Most modern financial LLM evaluation is *aggregated* into two umbrella benchmarks (FinBen/FLARE and the Open FinLLM Leaderboard), which themselves wrap the individual datasets in the rest of this page. Start there, then drill into the task-specific tables.

---

## 0. Caveats before you trust any number

- **Contamination / look-ahead.** Many of these datasets predate today's LLMs and are almost certainly in pre-training corpora (FinQA, Financial PhraseBank, StockNet). High scores can reflect memorization. Prefer **newer, held-out, or perturbed** suites (FailSafeQA, FAMMA, BizFinBench) for honest evaluation.
- **Survivorship & point-in-time.** Forecasting sets (StockNet/ACL18, BigData22, FNSPID) align text to *historical* prices for *surviving* tickers; treating them as tradable signals introduces look-ahead and survivorship bias.
- **Licensing is uneven.** Financial PhraseBank is CC BY-NC-SA 3.0, i.e. non-commercial (sentences sourced from LexisNexis news); FNSPID is CC BY-NC-4.0 (non-commercial); FinanceBench's open slice is 150 of 10,231 items, itself CC BY-NC-4.0 (full set is separately licensed). Always read the dataset card before commercial use.
- **"Numerical reasoning" ≠ arithmetic.** FinQA-family tasks score the *reasoning program*, not just the final number; report exact-match on programs vs. execution accuracy carefully.
- **Brazil gap.** Portuguese-language financial NLP resources exist but are small and scattered (see §9); there is no B3 equivalent of FinBen yet.

---

## 1. Umbrella benchmarks & leaderboards (start here)

| Suite | What it is | Coverage | Free/Paid | Access |
|-------|------------|----------|-----------|--------|
| **FinBen** (The-FinAI) | Holistic open financial benchmark; the de-facto academic standard | 36 datasets, 24 tasks, 7 aspects (IE, textual analysis, QA, text-gen, risk management, forecasting, decision-making) | Free, open | Paper <https://arxiv.org/abs/2402.12659> · code <https://github.com/The-FinAI/PIXIU> · NeurIPS 2024 D&B |
| **FLARE** (PIXIU) | Original eval harness inside PIXIU; precursor to FinBen; `flare-*` tasks | Sentiment, QA, NER, headline, movement prediction, summarization | Free, open | <https://github.com/The-FinAI/PIXIU> · datasets under <https://huggingface.co/TheFinAI> |
| **Open Financial LLM Leaderboard (OFLL)** | Live HF Space; zero-shot scoring of public models on finance tasks; Linux Foundation / FINOS-backed | IE, sentiment, credit scoring, stock-movement forecasting, risk, QA | Free | FINOS Space <https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard> · TheFinAI mirror <https://huggingface.co/spaces/TheFinAI/Open-FinLLM-Leaderboard> · paper <https://arxiv.org/abs/2501.10963> |
| **Open FinLLM Reasoning Leaderboard** | Reasoning-focused split (CoT, multi-step) | Math/financial reasoning tasks | Free | <https://huggingface.co/spaces/TheFinAI/open-finllm-reasoning-leaderboard> |
| **FINOS Open-Financial-LLMs-Leaderboard (repo)** | Source + docs for the leaderboard | — | Free | <https://github.com/finos-labs/Open-Financial-LLMs-Leaderboard> · docs <https://finllm-leaderboard.readthedocs.io/> |
| **PIXIU** (The-FinAI) | First open financial LLM family (FinMA 7B/30B) + FIT instruction data + FLARE | LLaMA-based FinMA models on HF | Free | <https://github.com/The-FinAI/PIXIU> |

> **Practical tip:** if you only run one thing, run FinBen via PIXIU — it pulls most of the §2–§7 datasets for you. The HF leaderboards are good for *comparing* models you don't host.

---

## 2. Question Answering (QA)

| Dataset | Task | Size | Free | Link |
|---------|------|------|------|------|
| **FinanceBench** (Patronus AI) | Open-book QA over SEC filings; RAG eval | 10,231 Qs (open slice: 150 annotated) | Open slice CC BY-NC-4.0; full set licensed | HF <https://huggingface.co/datasets/PatronusAI/financebench> · repo <https://github.com/patronus-ai/financebench> · paper <https://arxiv.org/abs/2311.11944> |
| **FinQA** | Numerical QA over a financial report page (table+text), with gold reasoning program | 8,281 QA pairs | Free | <https://github.com/czyssrs/FinQA> · ACL <https://aclanthology.org/2021.emnlp-main.300/> |
| **ConvFinQA** | Multi-turn conversational extension of FinQA | 3,892 conversations / 14,115 Qs | Free | <https://github.com/czyssrs/ConvFinQA> · paper <https://arxiv.org/abs/2210.03849> |
| **TAT-QA** | QA over hybrid table-and-text passages; arithmetic/compare/count | 16,552 Qs over 2,757 contexts | Free | <https://github.com/NExTplusplus/TAT-QA> · ACL 2021 |
| **MultiHiertt** | QA over *multiple* hierarchical tables + text (annual reports) | 10,440 Qs | Free | <https://github.com/psunlpgroup/MultiHiertt> · ACL 2022 |
| **DocFinQA** | Long-context FinQA: full SEC report as context (~123k words avg vs <700 for FinQA); Python answer programs | 7,437 QA over full filings | Free | <https://arxiv.org/abs/2401.06915> · ACL 2024 short |
| **FinTextQA** | Long-form financial QA (textbooks/regulation) | 1,262 QA pairs | Free | paper <https://arxiv.org/abs/2405.09980> |
| **FiQA (Task 1 & 2)** | Aspect-based sentiment + opinion QA (microblogs, news) | ~1k sentiment + QA pairs | Free | <https://sites.google.com/view/fiqa/> · HF mirrors under TheFinAI |
| **FinDER** | Expert-generated financial information-retrieval / RAG benchmark | 5,703 query–evidence–answer triplets | Free | paper <https://arxiv.org/abs/2504.15800> · ICLR 2025 Fin-AI workshop |

🇧🇷 *Brazil:* no public B3-filing QA set; closest is NER over Brazilian earnings-call transcripts (§9). CVM filings (IPE/DFP) are an open, untapped corpus to build one.

---

## 3. Numerical / Quantitative reasoning

| Dataset | Task | Size | Free | Link |
|---------|------|------|------|------|
| **BizBench** (Kensho/S&P) | 8 quantitative-reasoning tasks from exams, earnings reports, programs; includes SEC-Num span ID | 8 tasks | Free | paper <https://arxiv.org/abs/2311.06602> · S&P AI Benchmarks <https://benchmarks.kensho.com/> |
| **FinQA / ConvFinQA / TAT-QA / MultiHiertt** | (see §2 — all are numerical-reasoning sets too) | — | Free | — |
| **DocFinQA** | Long-context numerical reasoning + program synthesis | 7,437 | Free | <https://arxiv.org/abs/2401.06915> |
| **FAMMA** | Financial **multilingual + multimodal** QA (charts, tables, formulas); EN/ZH/FR, 8 finance subfields | 1,945 (Basic) + 103 (LivePro, held-out) | Free | paper <https://arxiv.org/abs/2410.04526> · project <https://famma-bench.github.io/famma/> |
| **FinEval** | Chinese financial knowledge MCQ (finance/econ/accounting/certs) | 4,661 Qs, 34 subjects | Free | <https://arxiv.org/abs/2308.09975> |
| **CFLUE** | Chinese financial language-understanding eval (knowledge + NLP tasks) | multi-task | Free | paper <https://arxiv.org/abs/2405.10542> |

> SEC-Num (inside BizBench) is one of the few **span-identification** numeric tasks — useful when you care about *finding* the right figure in a noisy 10-K, not synthesizing it.

---

## 4. Summarization & generation

| Dataset | Task | Size | Free | Link |
|---------|------|------|------|------|
| **ECTSum** | Bullet-point summarization of long earnings-call transcripts (Reuters-derived gold) | 2,425 transcript–summary pairs (7:1:2 split) | Free | <https://github.com/rajdeep345/ECTSum> · paper <https://arxiv.org/abs/2210.12467> · EMNLP 2022 |
| **EDGAR-CORPUS** | Raw 10-K filing corpus (pre-training / summarization base), 1993–2020, item-split | ~220k records, ~6.5B tokens | Free | HF <https://huggingface.co/datasets/eloukas/edgar-corpus> · crawler <https://github.com/lefterisloukas/edgar-crawler> · paper <https://arxiv.org/abs/2109.14394> |
| **EDGAR-CORPUS Financial Summarization** | Derived 10-K summarization split | ~1k records | Free | HF <https://huggingface.co/datasets/kritsadaK/EDGAR-CORPUS-Financial-Summarization> |
| **FNS / FinLLM Challenge (text-gen)** | Shared-task financial text summarization | shared-task splits | Free | task pages / FinLLM workshop |

🇧🇷 *Brazil:* earnings-call transcripts of Brazilian banks have been compiled for NER (384 transcriptions, §9) but not released as a summarization benchmark — a clear gap.

---

## 5. Sentiment & classification

| Dataset | Task | Size | Free | Link |
|---------|------|------|------|------|
| **Financial PhraseBank** | 3-class sentiment of finance-news sentences (retail-investor view); agreement-tiered | 4,840 sentences | CC BY-NC-SA 3.0 (non-commercial) | HF <https://huggingface.co/datasets/takala/financial_phrasebank> · mirror <https://huggingface.co/datasets/lmassaron/FinancialPhraseBank> |
| **FiQA SA** | Aspect-based sentiment (microblogs + news headlines) | ~961 annotated | Free | <https://sites.google.com/view/fiqa/> |
| **Headline (Gold news / Gold commodity)** | Multi-label news-headline tagging + price-direction (up/down/stable) on gold, 2000–2019 | 11,412 headlines | Free | via FLARE `flare-headlines`; paper "Impact of News on Commodity Prices" |
| **FinBERT pretraining corpus** | (model, not dataset) financial-comms LM; common baseline | — | Free | <https://arxiv.org/abs/2006.08097> |

> Financial PhraseBank ships four splits by annotator agreement (50%, 66%, 75%, 100%). Always report *which* split — scores are not comparable across them.

🇧🇷 *Brazil:* see §9 — Carosia (B3 news), B2T (bank tweets), and Kaggle PT-BR sentiment sets.

---

## 6. NER & relation extraction

| Dataset | Task | Size | Free | Link |
|---------|------|------|------|------|
| **FiNER-139** | XBRL numeric-entity tagging in 10-K/10-Q; 139 fine-grained tags | 1.1M sentences | Free | HF <https://huggingface.co/datasets/nlpaueb/finer-139> · repo <https://github.com/nlpaueb/finer> · ACL 2022 |
| **FiNER-ORD** | Open-research financial NER (PER/ORG/LOC in financial news) | manually annotated news | Free | <https://arxiv.org/abs/2302.11157> · <https://github.com/gtfintechlab/FiNER-ORD> |
| **REFinD** | Relation extraction over 10-X filings; 22 relations, 8 entity-pair types | ~28,676 instances | Free | paper <https://arxiv.org/abs/2305.18322> · SIGIR 2023 |
| **FinRED** | Relation extraction in financial domain (news + earnings), distant-supervised to Wikidata | 6,767 instances · 29 relations | Free | paper <https://arxiv.org/abs/2306.03736> · repo <https://github.com/soummyaah/FinRED> · WWW 2022 Companion |
| **FinNum (1/2/3)** | Numeral understanding/attachment in financial social media | shared-task splits | Free | FinNum shared tasks (NTCIR / IJCAI-FinNLP) |
| **BUSTER** | Business-transaction entity recognition | annotated docs | Free | <https://arxiv.org/abs/2402.09916> |

> REFinD is purpose-built to break models on **directional/numeric relational ambiguity** (e.g. "A acquired B" vs "B acquired A") that general-domain RE sets miss.

---

## 7. Tone, monetary policy & event signals

| Dataset | Task | Size | Free | Link |
|---------|------|------|------|------|
| **Trillion Dollar Words** | Hawkish/dovish/neutral classification of FOMC sentences + market analysis | ~40k sentences (1996–2022); ~2.5k labeled | Free | repo <https://github.com/gtfintechlab/fomc-hawkish-dovish> · model <https://huggingface.co/gtfintechlab/FOMC-RoBERTa> · paper <https://arxiv.org/abs/2305.07972> · ACL 2023 |
| **FOMC communication (HF)** | FOMC minutes/speeches/press-conf classification corpus | timestamped sentences | Free | HF <https://huggingface.co/datasets/gtfintechlab/fomc_communication> |
| **SubjECTive-QA** | Subjectivity of earnings-call Q&A across 6 features (Assertive, Cautious, Optimistic, Specific, Clear, Relevant) | 49,446 annotations over long-form QA pairs | Free | <https://arxiv.org/abs/2410.20651> · NeurIPS 2024 D&B |
| **Op-Fed** | Opinion/stance + monetary-policy annotations on FOMC transcripts (active learning) | 1,044 annotated sentences | Free | <https://arxiv.org/abs/2509.13539> |

---

## 8. Stock-movement forecasting from text (use with bias caveats)

| Dataset | Task | Size | Free | Link |
|---------|------|------|------|------|
| **StockNet / ACL18** | Binary movement prediction from tweets + prices; 88 S&P tickers, 2014–2016 | 87 stocks · 106,271 tweets · 696 days | Free | <https://github.com/yumoxu/stocknet-dataset> · ACL 2018 |
| **BigData22 / CIKM18** | Movement prediction; 50 tickers, 2019–2020 (sparse, noisy tweets) | 50 stocks · 272,762 tweets · 362 days | Free | SLOT repo <https://github.com/deeptrade-public/slot> · BigData 2022 |
| **FNSPID** | Time-series alignment of financial news to prices; S&P500, 1999–2023 | 29.7M prices · 15.7M news · 4,775 cos | CC BY-NC-4.0 | HF <https://huggingface.co/datasets/Zihan1004/FNSPID> · repo <https://github.com/Zdong104/FNSPID_Financial_News_Dataset> · paper <https://arxiv.org/abs/2402.06698> |

> These appear inside FinBen/FLARE as forecasting tasks (`flare-sm-*`). Treat reported accuracy as a *text-signal* benchmark, **not** a backtest — no transaction costs, survivorship-filtered, and label leakage is easy.

---

## 9. 🇧🇷 Brazil / Portuguese-language financial NLP

| Resource | Task | Notes | Link |
|----------|------|-------|------|
| **Carosia et al. — B3 financial news sentiment** | PT-BR sentiment on Brazilian-stock news + code | Foundational PT-BR finance-sentiment work | IEEE LATAM <https://latamt.ieeer9.org/index.php/transactions/article/view/5977> |
| **B2T** | Tweets about Brazilian banks (PT-BR); 375,912 comments, 1,096 sentiment-labeled | Bank-domain social sentiment dataset (SBC DSW 2024) | <https://sol.sbc.org.br/index.php/dsw/article/view/30610> |
| **BraFiNER (Brazilian earnings-call NER)** | NER on 384 PT-BR transcripts from 10 banks (~118k sentences); mono/multilingual transformer comparison | Rare PT-BR finance NER corpus | <https://arxiv.org/abs/2403.12212> |
| **PT-BR sentiment datasets (Kaggle)** | 5 standardized PT-BR sentiment splits (general, not finance-only) | Useful base for fine-tuning | <https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets> |
| **CVM open data / Dados Abertos** | DFP/ITR/IPE filings, fund registries (raw corpus to *build* PT-BR finance benchmarks) | Government open data; no labels | <https://dados.cvm.gov.br/> |

> **Build-it-yourself note:** the CVM open-data portal + B3's market-data files are an unlabeled goldmine; the missing piece for Brazil is *annotation*, not raw text. A PT-BR FinQA/FinBen analog is an open research opportunity.

---

## 10. Newer & specialized suites (2024–2026)

| Suite | Focus | Free | Link |
|-------|-------|------|------|
| **FailSafeQA** (Writer) | Robustness & compliance: perturbs Qs with misspellings, OCR noise, OOD rewrites, missing context; hallucination risk | Free | <https://arxiv.org/abs/2502.06329> |
| **InvestorBench** | LLM-**agent** financial decision-making (trading/allocation environments) | Free | paper <https://arxiv.org/abs/2412.18174> · ACL 2025 |
| **Finance Agent Benchmark** | Real-world financial-research agent tasks (EDGAR + web tools); 537 expert-vetted Qs, 9 categories | Free | <https://arxiv.org/abs/2508.00828> |
| **BizFinBench / BizFinBench.v2** | Business-driven, expert-level financial capability (v1: 6,781 Chinese queries; v2: 29,578 bilingual EN/ZH QA pairs) | Free | <https://arxiv.org/abs/2505.19457> · v2 <https://arxiv.org/abs/2601.06401> |
| **FinTrust** | Trustworthiness (safety, fairness, robustness) in finance | Free | <https://arxiv.org/abs/2510.15232> |
| **FinDABench** | Financial data-analysis ability (interpret + compute over data) | Free | <https://arxiv.org/abs/2401.02982> |
| **VisFinEval** | Chinese **multimodal** financial understanding (scenario-driven) | Free | <https://arxiv.org/abs/2508.09641> |
| **M³FinMeeting** | Multilingual/multi-sector/multi-task financial-meeting understanding | Free | <https://arxiv.org/abs/2506.02510> |
| **FinMTEB** | Finance **embedding** benchmark (MTEB analog): 64 datasets / 7 tasks (STS, retrieval, classification, clustering, reranking, etc.), EN+ZH; ships Fin-E5 model | Free | <https://arxiv.org/abs/2502.10990> · repo <https://github.com/yixuantt/FinMTEB> · EMNLP 2025 |
| **S&P AI Benchmarks (Kensho)** | Hosted finance/quant benchmark portal (BizBench host) | Free portal | <https://benchmarks.kensho.com/> |
| **FinGPT** | Open data + models for financial LLMs (instruction/RLHF pipelines) | Free | <https://github.com/AI4Finance-Foundation/FinGPT> |

---

## 11. Quick chooser

- **"Can my RAG answer 10-K questions?"** → FinanceBench (open slice) + DocFinQA + FinDER; for the *retriever/embeddings* use FinMTEB.
- **"Can it do the math in a report?"** → FinQA / ConvFinQA / TAT-QA / MultiHiertt / BizBench.
- **"Tables, charts, and multilingual?"** → FAMMA, VisFinEval, MultiHiertt.
- **"Summarize earnings calls?"** → ECTSum (+ EDGAR-CORPUS to pre-train).
- **"Sentiment / classification baseline?"** → Financial PhraseBank + FiQA + Headline.
- **"Extract entities / relations from filings?"** → FiNER-139, FiNER-ORD, REFinD, FinRED.
- **"Central-bank tone?"** → Trillion Dollar Words + Op-Fed + SubjECTive-QA.
- **"Is the model robust / safe / agentic?"** → FailSafeQA, FinTrust, InvestorBench, Finance Agent Benchmark.
- **"One score to rule them all?"** → FinBen (PIXIU) and the Open FinLLM Leaderboard.

---

### Sources

- The-FinAI / PIXIU & FinBen: <https://github.com/The-FinAI/PIXIU> · <https://arxiv.org/abs/2402.12659> · <https://huggingface.co/TheFinAI>
- Open FinLLM Leaderboard (FINOS / TheFinAI): <https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard> · <https://huggingface.co/spaces/TheFinAI/Open-FinLLM-Leaderboard> · <https://arxiv.org/abs/2501.10963> · <https://github.com/finos-labs/Open-Financial-LLMs-Leaderboard> · <https://finllm-leaderboard.readthedocs.io/>
- FinanceBench: <https://huggingface.co/datasets/PatronusAI/financebench> · <https://github.com/patronus-ai/financebench> · <https://arxiv.org/abs/2311.11944>
- FinQA: <https://github.com/czyssrs/FinQA> · <https://aclanthology.org/2021.emnlp-main.300/> — ConvFinQA: <https://arxiv.org/abs/2210.03849> — TAT-QA: <https://github.com/NExTplusplus/TAT-QA> — MultiHiertt: <https://github.com/psunlpgroup/MultiHiertt> — DocFinQA: <https://arxiv.org/abs/2401.06915>
- BizBench / Kensho: <https://arxiv.org/abs/2311.06602> · <https://benchmarks.kensho.com/> — FAMMA: <https://arxiv.org/abs/2410.04526> — FinEval: <https://arxiv.org/abs/2308.09975>
- ECTSum: <https://github.com/rajdeep345/ECTSum> · <https://arxiv.org/abs/2210.12467> — EDGAR-CORPUS: <https://huggingface.co/datasets/eloukas/edgar-corpus> · <https://github.com/lefterisloukas/edgar-crawler> · <https://arxiv.org/abs/2109.14394>
- Financial PhraseBank: <https://huggingface.co/datasets/takala/financial_phrasebank> — FiQA: <https://sites.google.com/view/fiqa/> — FinBERT: <https://arxiv.org/abs/2006.08097>
- FiNER-139: <https://huggingface.co/datasets/nlpaueb/finer-139> · <https://github.com/nlpaueb/finer> — FiNER-ORD: <https://arxiv.org/abs/2302.11157> · <https://github.com/gtfintechlab/FiNER-ORD> — REFinD: <https://arxiv.org/abs/2305.18322> — FinRED: <https://arxiv.org/abs/2306.03736> · <https://github.com/soummyaah/FinRED> — BUSTER: <https://arxiv.org/abs/2402.09916>
- Trillion Dollar Words: <https://github.com/gtfintechlab/fomc-hawkish-dovish> · <https://huggingface.co/datasets/gtfintechlab/fomc_communication> · <https://arxiv.org/abs/2305.07972> — SubjECTive-QA: <https://arxiv.org/abs/2410.20651> — Op-Fed: <https://arxiv.org/abs/2509.13539>
- StockNet/ACL18: <https://github.com/yumoxu/stocknet-dataset> — BigData22/SLOT: <https://github.com/deeptrade-public/slot> — FNSPID: <https://huggingface.co/datasets/Zihan1004/FNSPID> · <https://github.com/Zdong104/FNSPID_Financial_News_Dataset> · <https://arxiv.org/abs/2402.06698>
- FailSafeQA: <https://arxiv.org/abs/2502.06329> — InvestorBench: <https://arxiv.org/abs/2412.18174> — Finance Agent Benchmark: <https://arxiv.org/abs/2508.00828> — BizFinBench: <https://arxiv.org/abs/2505.19457> — FinTrust: <https://arxiv.org/abs/2510.15232> — FinDABench: <https://arxiv.org/abs/2401.02982> — VisFinEval: <https://arxiv.org/abs/2508.09641> — M³FinMeeting: <https://arxiv.org/abs/2506.02510> — FinMTEB: <https://arxiv.org/abs/2502.10990> · <https://github.com/yixuantt/FinMTEB> — FinGPT: <https://github.com/AI4Finance-Foundation/FinGPT>
- 🇧🇷 Brazil: <https://latamt.ieeer9.org/index.php/transactions/article/view/5977> · B2T <https://sol.sbc.org.br/index.php/dsw/article/view/30610> · BraFiNER <https://arxiv.org/abs/2403.12212> · <https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets> · <https://dados.cvm.gov.br/>

**Keywords:** financial LLM benchmark, financial NLP datasets, FinBen, FLARE, PIXIU, FinMA, Open FinLLM Leaderboard, FINOS, FinanceBench, FinQA, ConvFinQA, TAT-QA, MultiHiertt, DocFinQA, FinDER, BizBench, FAMMA, ECTSum, EDGAR-CORPUS, FNSPID, StockNet, ACL18, BigData22, Financial PhraseBank, FiQA, FiNER-139, REFinD, FinRED, Trillion Dollar Words, FOMC hawkish dovish, SubjECTive-QA, FailSafeQA, InvestorBench, FinMTEB, financial question answering, numerical reasoning, earnings call summarization, sentiment analysis, named entity recognition, relation extraction, central bank tone — em português: avaliação de modelos de linguagem financeiros, conjuntos de dados de PLN financeiro, perguntas e respostas financeiras, raciocínio numérico, sumarização de teleconferências de resultados, análise de sentimento, reconhecimento de entidades, extração de relações, tom de política monetária, B3, CVM, dados abertos.
