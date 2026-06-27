# Kaggle Winning Solutions — NLP & LLM

Curated index of top Kaggle solutions across classic NLP and LLM-era competitions. Each entry lists the competition (name + year), the final rank, the team, the model backbones, the key tricks, and at least one real public link (Kaggle writeup, official GitHub repo, or competition discussion). Sourced only from public material; unverifiable entries were omitted rather than invented.

> Conventions: `deberta-v3-large` and friends in `snake_case`. "MAP@k" = mean average precision at k. "PL" = pseudo-labeling. "KD" = knowledge distillation. "MLM" = masked-language-model pretraining on competition corpus. "MSD" = multi-sample dropout.

---

## TL;DR cheat-sheet

| Era | Dominant backbone | Recurring winning tricks |
|---|---|---|
| 2018–2020 (BERT era) | `bert-base/large`, `roberta`, `xlm-roberta`, `bart`, `bi-lstm`/`cnn` | Custom-vocab + MLM pretrain, multi-sample dropout, 2nd-stage char-level models, pseudo-labeling + TTA, post-processing on label noise |
| 2021–2023 (DeBERTa era) | `deberta-v3-large`, `deberta-v2-xlarge`, `longformer`, `funnel` | Pseudo-labeling, MLM domain-adapt, token-classification framing, diverse-architecture ensembles, GBDT stacking on OOF |
| 2023–2025 (LLM era) | `mistral-7b`, `llama-3-8b/70b`, `qwen2.5`, `gemma-2-9b`, `deberta-v3` | RAG/retrieval, LoRA/QLoRA fine-tune, KD (big→small), synthetic data generation, mean-prompt adversarial hacks, contrastive retrieve-then-rerank |

---

## Competition index (at a glance)

| Competition | Year | Best-documented rank | Backbones | Signature move | Link |
|---|---|---|---|---|---|
| Jigsaw — Toxic Comment Classification | 2018 | 1st (Toxic Crusaders) | `rnn`/`cnn`, transformer | TTA + PL, 8-fold OOF, tail-sentence models | [writeup](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/writeups/toxic-crusaders-1st-place-solution-overview) |
| Jigsaw — Unintended Bias in Toxicity | 2019 | 8th (Qishen Ha) | `bert`, `gpt-2`, `bi-lstm` | Identity sample-weighting loss, blend | [writeup](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/writeups/qishen-ha-8th-place-solution-4-models-simple-avg) |
| Jigsaw — Multilingual Toxic | 2020 | 1st (leecming) | `xlm-roberta-large` + monolingual | Translated-only data, iterative PL bootstrap | [repo](https://github.com/leecming82/jigsaw-multilingual) |
| Google QUEST Q&A Labeling | 2020 | 1st | `bert`, `roberta`, `bart` | StackExchange MLM, custom LaTeX/code vocab | [code](https://github.com/oleg-yaroshevskiy/quest_qa_labeling) |
| Tweet Sentiment Extraction | 2020 | 1st (Dark of the Moon) | `roberta`, `bert` + char models | Space-offset label-noise post-processing | [code](https://github.com/heartkilla/kaggle_tweet) |
| CommonLit Readability Prize | 2021 | 1st | `roberta`, `deberta` | Pooling heads, LLRD, many-seed averaging | [code](https://github.com/mathislucka/kaggle_clrp_1st_place_solution) |
| Feedback Prize — Evaluating Student Writing | 2021 | 2nd | `longformer`, `deberta-xlarge` | Token-classification + span post-proc | [writeup](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389) |
| NBME — Score Clinical Patient Notes | 2022 | top/gold | `deberta-v3-large` | Clinical MLM + iterative PL | [news](https://www.nbme.org/news/six-teams-recognized-nlp-advances-nbmes-patient-note-scoring-competition) |
| U.S. Patent Phrase-to-Phrase Matching | 2022 | top discussions | `deberta-v3-large` | Anchor-group prompt concat + CPC context | [discussion](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332492) |
| Feedback Prize — Predicting Effective Arguments | 2022 | 1st (Team Hydrogen) | `deberta-v3-large` | Token-classification framing, ensemble | [code](https://github.com/ybabakhin/kaggle-feedback-effectiveness-1st-place-solution) |
| Feedback Prize — English Language Learning | 2022 | 1st (AutoX) | `deberta-v3-large/base` | Multi-stage PL, differential LR | [code](https://github.com/rohitsingh02/kaggle-feedback-english-language-learning-1st-place-solution) |
| Learning Equality — Curriculum Recommendations | 2023 | 1st (Psi) | `xlm-roberta`/`mpnet` bi-encoder | Retrieve → cross-encoder rerank, multilingual | [discussion](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812) |
| CommonLit — Evaluate Student Summaries | 2023 | gold writeup | `deberta-v3-large` | DeBERTa OOF → GBDT stack | [discussion](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446712) |
| Kaggle — LLM Science Exam | 2023 | 1st | `deberta-v3`, `mistral-7b` | BM25 retrieve → DeBERTa rerank → MC | [discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436674) |
| LLM — Detect AI Generated Text | 2023–24 | 1st | LLM zoo (CLM) | Generate PERSUADE essays; TF-IDF+trees | [writeup](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/comprehensive-1st-place-write-up) |
| PII Data Detection | 2024 | 1st (Fold Zero) | 5× `deberta-v3-large` | Diverse heads + KD + synthetic PII | [code](https://github.com/bogoconic1/pii-detection-1st-place) |
| LLM Prompt Recovery | 2024 | top | `gemma`/T5 proxy | Mean-prompt embedding-cosine hack | [code](https://github.com/ironbar/prompt_recovery) |
| LLM 20 Questions | 2024 | 1st (c-number) | agent + keyword vocab | Binary search over keyword list | [writeup](https://www.kaggle.com/competitions/llm-20-questions/writeups/c-number-1st-place-solution) |
| LMSYS — Chatbot Arena Human Preference | 2024 | 1st | `gemma-2-9b` (distilled) | 70B teachers → Gemma KD, LoRA-avg | [discussion](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629) |
| Learning Agency Lab — Automated Essay Scoring 2.0 | 2024 | 1st | `deberta-v3` | Data-source classification head vs shift | [discussion](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/498478) |
| Eedi — Mining Misconceptions in Mathematics | 2024 | 1st (MTH 101) | `qwen2.5` | Bi-encoder retrieve → LLM listwise rerank | [writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution) |
| WSDM Cup — Multilingual Chatbot Arena | 2025 | 1st | Qwen/Gemma LoRA | Preference-pair seq-cls + external arena | [writeup](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/whitefebruary-1st-place-solution) |
| MAP — Charting Student Math Misunderstandings | 2025 | 1st | LLM classifiers/rerankers | Misconception classification + ensembling | [writeup](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution) |

---

## 1. Toxicity / Content classification

### Jigsaw — Toxic Comment Classification Challenge (2018) — 1st place
- **Team**: "Toxic Crusaders".
- **Task**: multi-label classification of comment toxicity (6 labels: toxic, severe_toxic, obscene, threat, insult, identity_hate), mean column-wise ROC-AUC.
- **Models**: the team had no prior NLP-specific edge and systematically tested standard DL — best `rnn` slightly beat best `cnn`; a Transformer ("Attention Is All You Need") matched RNN level but trained much slower. Final winner was a large blend.
- **Key tricks**: **TTA** + **pseudo-labeling** were the recurring levers; 6 GPUs, 8-fold OOF over 1M+ samples (TTA+PL) ~2h/model; **tail-sentence models** (last 25–50 chars) added because many comments were toxic only in the final sentence (~0.0015 stacking boost).
- Link (1st-place overview): https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/writeups/toxic-crusaders-1st-place-solution-overview
- Link (community review of top methods): https://blog.ceshine.net/post/kaggle-toxic-comment-classification-challenge/
- Competition: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

### Jigsaw — Unintended Bias in Toxicity Classification (2019)
- **Task**: predict toxicity while minimizing unintended bias across identity subgroups; custom bias-AUC metric.
- **Top solutions** (BERT era): fine-tuned `bert-base/large` dominated as best single models, blended with `gpt-2` and `bi-lstm` heads; custom **sample-weighting loss** (up-weight identity mentions + non-toxic content), and logistic-regression / blend ensembling were the recurring winning levers.
- **Verified references**: official 8th-place writeup (Qishen Ha, 4-model simple average) and a widely-cited solution-notes post.
- Link (8th-place writeup): https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/writeups/qishen-ha-8th-place-solution-4-models-simple-avg
- Link (overview of top-solution methods): https://blog.ceshine.net/post/kaggle-jigsaw-toxic-2019/
- Competition: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification

### Jigsaw — Multilingual Toxic Comment Classification (2020) — 1st place
- **Team**: leecming + rafiko1.
- **Task**: classify toxicity in 6 non-English languages (es, it, tr, pt, ru, fr); train mostly on English + translations; ROC-AUC.
- **Models**: `xlm-roberta-large` was the strongest single model; ensemble with monolingual pretrained models (per-language).
- **Key tricks**: the winning recipe used **translated data only**; **iterative pseudo-labeling** was essential — bootstrap test-set PL with multilingual XLM-R, train monolingual/multilingual models on data+PL, then feed monolingual PL back to improve XLM-R. TPU training (rafiko1's part targeted Kaggle TPU instances).
- Link (1st-place code): https://github.com/leecming82/jigsaw-multilingual
- Link (competition review): https://blog.ceshine.net/post/multilingual-toxic-classification/
- Competition: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification

---

## 2. Readability / Writing quality (CommonLit + Feedback Prize family)

### CommonLit Readability Prize (2021) — 1st place
- **Team**: Mathis Lucka et al.
- **Task**: regress passage reading-difficulty (single continuous target), RMSE.
- **Models**: ensemble of `roberta-large` / `roberta-base` and `deberta` variants; careful pooling heads (attention/mean pooling), layer-wise learning-rate decay, and many-seed averaging to fight the tiny noisy dataset.
- Link (1st-place code): https://github.com/mathislucka/kaggle_clrp_1st_place_solution
- Link (2nd-place code, Takoi): https://github.com/TakoiHirokazu/kaggle_commonLit_readability_prize
- Competition: https://www.kaggle.com/c/commonlitreadabilityprize

### CommonLit — Evaluate Student Summaries (2023) — winning solution
- **Team**: Ivan Isaev / "Aerlic" (gold-zone writeup).
- **Task**: score student summaries on `content` and `wording`, MCRMSE.
- **Models**: ensemble of `deberta-v3-large` (and larger DeBERTa) first-stage models feeding a **2nd-stage stacking** model (GBDT / RF on OOF preds); prompt-text concatenation and length/feature engineering.
- Link (winning recap): https://www.linkedin.com/pulse/our-kaggle-winning-solution-recap-commonlit-evaluate-ivan-isaev--a4bgf
- Link (solution discussions index): https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446712
- Competition: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries

### Learning Agency Lab — Automated Essay Scoring 2.0 (2024) — 1st place
- **Task**: holistic 1–6 essay scoring on the ASAP 2.0 / PERSUADE-derived corpus; quadratic weighted kappa (QWK). 2,700+ teams.
- **Key insight**: the competition was fundamentally a **distribution-shift** problem — the train set mixed two sources (PERSUADE 2.0 essays vs Kaggle-only essays) with different grading criteria. The 1st-place author jumped from ~619th public to 1st private by adding a **data-source classification head** so models could distinguish and calibrate across sources, plus threshold/rounding tuning against the shifted private set.
- Link (1st-place discussion): https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/498478
- Link (top-solutions report): https://hippocampus-garden.com/kaggle_aes2/
- Competition: https://www.kaggle.com/c/learning-agency-lab-automated-essay-scoring-2

### Feedback Prize — Evaluating Student Writing (2021) — top solutions
- **Task**: token-span identification of argumentative discourse elements (NER-style), span-overlap F1.
- **Models**: `longformer-large-4096`, `deberta-xlarge`, `deberta-v3-large`, `funnel-large`, `deberta-large` ensembles; framed as token-classification + span post-processing. The "two Longformers are better than one" pattern was a community baseline; top teams scaled with large DeBERTa ensembles and threshold tuning.
- **Verified references**: official 2nd-place writeup; 2nd-place code (ubamba98); tascj solution repo.
- Link (2nd-place writeup): https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389
- Link (2nd-place code): https://github.com/ubamba98/feedback-prize
- Link (tascj solution): https://github.com/tascj/kaggle-feedback-prize-2021
- Competition: https://www.kaggle.com/competitions/feedback-prize-2021

### Feedback Prize — Predicting Effective Arguments (2022) — 1st place
- **Team**: "Team Hydrogen" (Yuri Babakhin / ybabakhin et al.).
- **Task**: classify each argumentative discourse element as `Ineffective` / `Adequate` / `Effective`, multi-class log-loss.
- **Models**: `deberta-v3-large` ensemble; element-level classification with surrounding-context framing (special tokens marking the target span), strong cross-validation and blend; reused token-span insight from Feedback 2021.
- Link (1st-place code): https://github.com/ybabakhin/kaggle-feedback-effectiveness-1st-place-solution
- Link (1st-place writeup): https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536
- Competition: https://www.kaggle.com/competitions/feedback-prize-effectiveness

### Feedback Prize — English Language Learning / ELL (2022) — 1st place
- **Team**: AutoX (Rohit Singh, Yevhenii) — official gold.
- **Task**: regress 6 language-proficiency dimensions (cohesion, syntax, vocabulary, phraseology, grammar, conventions), MCRMSE.
- **Models**: `deberta-v3-large` / `deberta-v3-base` ensemble; **multi-stage pseudo-labeling** (pretrain on PL from prior model / ensemble), differential LR, mean pooling + attention heads; trained largely on free A6000 machines.
- Link (1st-place writeup): https://www.kaggle.com/competitions/feedback-prize-english-language-learning/writeups/autox-rohit-yevhenii-1st-place-solution
- Link (1st-place code): https://github.com/rohitsingh02/kaggle-feedback-english-language-learning-1st-place-solution
- Competition: https://www.kaggle.com/competitions/feedback-prize-english-language-learning

---

## 3. Semantic matching / QA labeling / NLI / retrieval

### Google QUEST Q&A Labeling (2020) — 1st place
- **Team**: Dmitriy Danevskiy, Yury Kashnitsky, Oleg Yaroshevskiy, Dmitry Abulkhanov.
- **Task**: predict 30 subjective question/answer quality targets (Spearman).
- **Models**: blend of OOF preds from 2× `bert-base`, 1× `roberta-base`, 1× `bart-large`.
- **3 key tricks**: (1) **custom-domain MLM + sentence-order pretrain** on ~7M extra StackExchange questions; (2) **extended cased vocabulary** with LaTeX/math/code tokens; (3) ELMo-like softmax-weighted layer pooling + multi-sample dropout.
- Link (1st-place code): https://github.com/oleg-yaroshevskiy/quest_qa_labeling
- Link (winners' interview): https://medium.com/kaggle-blog/the-3-ingredients-to-our-success-winners-dish-on-their-solution-to-googles-quest-q-a-labeling-c1a63014b88
- Competition: https://www.kaggle.com/c/google-quest-challenge

### U.S. Patent Phrase-to-Phrase Matching / USPPPM (2022) — top solutions
- **Task**: score semantic similarity between anchor/target phrases within a CPC context class (Pearson).
- **Winning patterns**: **prompt-style grouping** — concatenate `anchor [SEP] target [SEP] context-title`, and group all targets sharing an anchor+context so the model sees sibling phrases; ensembles of `deberta-v3-large` + BERT-family variants; CPC-code-to-text mapping as added context.
- **Verified references**: competition solution-discussion index (top writeups). Note: third-party repos below are community references, not the official winning solution.
- Link (solutions discussion): https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332492
- Link (community repo, reference only): https://github.com/vadimtimakin/Patent-Matching-Kaggle
- Competition: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching

### Learning Equality — Curriculum Recommendations (2023) — 1st place
- **Team**: "Psi" (Philipp Singer / psinger) — won both the accuracy and the efficiency tracks.
- **Task**: match K-12 curriculum `topics` to the most relevant learning `content` items across many languages; mean F2.
- **Pipeline (retrieve → rerank)**: a **bi-encoder retriever** (multilingual transformer, e.g. `paraphrase-multilingual`/`xlm-roberta` style) embeds topics and content to fetch top candidates per topic; a **cross-encoder re-ranker** (transformer classifier on topic+content pairs) scores candidates; careful per-language handling and threshold tuning. Efficiency-track win came from a lean retrieval + compact reranker design.
- Link (1st-place discussion): https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812
- Link (1st-place code): https://github.com/psinger/kaggle-curriculum-solution
- Competition: https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations

### Contradictory, My Dear Watson (Getting-Started)
- **Note**: a **Getting Started** competition (no prize money / no official "winning" writeup). Multilingual NLI over 15 languages (premise/hypothesis → entailment / neutral / contradiction).
- **Community-standard approach**: fine-tune `xlm-roberta-large` on combined XNLI/MNLI + competition data; back-translation augmentation; TPU training. Treat as tutorial reference, not a prize solution.
- Link (representative XLM-R solution repo, reference only): https://github.com/SumitM0432/XLM-RoBERTa-for-Textual-Entailment
- Competition: https://www.kaggle.com/c/contradictory-my-dear-watson

### Tweet Sentiment Extraction (2020) — 1st place
- **Team**: "Dark of the Moon" (heartkilla et al.).
- **Task**: extract the support phrase for a given sentiment (Jaccard).
- **The "magic"**: discovered the **label noise came from removed consecutive spaces** offsetting annotations — a deterministic post-processing alignment recovered the true spans and gave a large boost.
- **Architecture**: 1st-stage transformers (`roberta`, `bert`) emit token-level start/end probabilities → fed as features into **2nd-stage character-level models** (CNN / RNN / WaveNet) with multi-sample dropout and custom losses; **pseudo-labeling** on public data with a threshold.
- Link (1st-place code): https://github.com/heartkilla/kaggle_tweet
- Link (char-level "magic" notebook): https://www.kaggle.com/theoviel/character-level-model-magic
- Competition: https://www.kaggle.com/competitions/tweet-sentiment-extraction

---

## 4. Clinical / PII / education extraction

### NBME — Score Clinical Patient Notes (2022) — top/gold solutions
- **Teams**: multiple gold teams recognized by NBME (incl. Ryuichi & currypurin gold work; ~900 hours open-sourced).
- **Task**: map scoring-rubric clinical concepts to token spans in student-written patient notes (token-level F1).
- **Models**: `deberta-v3-large` token-classification with **MLM domain pretrain** + **iterative pseudo-labeling** (M1 labels unlabeled data → retrain → M2); span-threshold post-processing; ensembling.
- Link (NBME recognition / winners): https://www.nbme.org/news/six-teams-recognized-nlp-advances-nbmes-patient-note-scoring-competition
- Link (PL methodology write-up): https://arxiv.org/html/2401.12994v1
- Competition: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes

### The Learning Agency Lab — PII Data Detection (2024) — 1st place
- **Team**: "Fold Zero" (Nicholas Broad / bogoconic1 et al.).
- **Task**: detect PII tokens in student essays (micro F5, recall-heavy).
- **Models**: ensemble of **5× `deberta-v3-large`** with diverse heads — a **multi-sample-dropout** custom model and a **BiLSTM** head model — for diversity; **knowledge distillation** (best MSD models as teachers → student); heavy synthetic-PII data generation and rule-based post-processing (regex for emails/phones/URLs).
- Link (1st-place writeup): https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/writeups/fold-zero-1st-place-solution-ensemble-of-diverse-d
- Link (1st-place code): https://github.com/bogoconic1/pii-detection-1st-place
- Competition: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data

---

## 5. LLM-era competitions

### Kaggle — LLM Science Exam (2023) — 1st place
- **Task**: answer GPT-3.5-written science MCQs (5 options, rank by correctness), MAP@5.
- **Pipeline (RAG)**: Wikipedia (`graelo/wikipedia 20230601.en`) chunked into overlapping passages, indexed with **Apache Lucene BM25**; retrieved chunks **reranked by a `deberta-v3` reranker**; answer models (`deberta-v3` multiple-choice + `mistral-7b`) trained as multi-class; outputs mixed by a custom **XGBRanker**.
- Link (1st-place solution discussion): https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436674
- Link (competition report w/ top-solution breakdown): https://hippocampus-garden.com/kaggle_llm/
- Competition: https://www.kaggle.com/competitions/kaggle-llm-science-exam

### LLM Prompt Recovery (2024) — top solution
- **Task**: recover the rewrite-instruction prompt given original + Gemma-rewritten text; metric = mean **Sharpened Cosine Similarity** of T5 (sentence-t5) embeddings.
- **The adversarial "mean prompt" hack**: because scoring is embedding cosine, top teams optimized a single fixed/"mean" prompt (plus token suffixes like the infamous `lucrarea`/"please improve this text..." style strings) maximizing expected similarity across the hidden set, rather than truly recovering per-sample prompts. Beam/greedy search over candidate phrases against a local T5 proxy.
- Link (solution repo, ironbar): https://github.com/ironbar/prompt_recovery
- Competition: https://www.kaggle.com/competitions/llm-prompt-recovery

### LLM — Detect AI Generated Text (2023–2024) — 1st place
- **Team**: rbiswasfc & team (official "Comprehensive 1st Place Write-Up").
- **Task**: classify essays as human vs LLM-generated (ROC-AUC), with a tiny public train set and a large hidden distribution → generalization was the whole game.
- **Key moves**: fine-tuned a **wide zoo of LLMs with the CLM objective on the PERSUADE corpus** to mass-generate diverse student-like essays (the real "data" advantage); trained classifiers on this generated corpus. The runner-up "LLMLab" path showed the **custom-trained byte/BPE tokenizer + TF-IDF + tree models (SGD/LightGBM/CatBoost) ensemble** that dominated the public LB. Multi-GPU (4× A100) DDP via HF Accelerate.
- Link (1st-place writeup): https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/comprehensive-1st-place-write-up
- Link (1st-place code): https://github.com/rbiswasfc/llm-detect-ai
- Link (1st-public/9th-private "LLMLab"): https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/llmlab-1st-public-9th-private-llmlab-solution-summ
- Competition: https://www.kaggle.com/competitions/llm-detect-ai-generated-text

### LLM 20 Questions (2024) — 1st place
- **Team**: "c-number" (official 1st-place writeup).
- **Task**: simulation competition — agents play 20-questions (guesser + answerer LLMs) to deduce a secret keyword in ≤20 yes/no questions.
- **Winning strategy**: **binary search over a pre-built keyword vocabulary** (20 questions ≈ 2^20 ≈ 1M words covers English), with robust question phrasing the paired answerer could answer reliably; deterministic alphabetical/bisection questioning beat free-form LLM guessing.
- Link (1st-place writeup): https://www.kaggle.com/competitions/llm-20-questions/writeups/c-number-1st-place-solution
- Competition: https://www.kaggle.com/competitions/llm-20-questions

### LMSYS — Chatbot Arena Human Preference Predictions (2024) — 1st place
- **Team**: "Distill is all you need" writeup authors.
- **Task**: given a prompt + two model responses, predict human preference (winner-A / winner-B / tie), log-loss.
- **Winning method — distillation**: train large teachers (`llama-3-70b`, `qwen2-72b`) in a 5-fold setup, then **distill (KL-divergence) into `gemma-2-9b`** per fold → 5 distilled Gemmas; **average the LoRA adapters** across folds into one model. 8× A100-80G. Heavy use of extra arena/preference data + careful truncation of long conversations.
- Link (1st-place "Distill is all you need"): https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629
- Link (tascj solution code): https://github.com/tascj/kaggle-lmsys-chatbot-arena
- Link (4th-place code, DaoyuanLi): https://github.com/DaoyuanLi2816/Kaggle-4th-Place-Solution-LMSYS-Chatbot-Arena-Human-Preference-Predictions
- Competition: https://www.kaggle.com/competitions/lmsys-chatbot-arena

### WSDM Cup — Multilingual Chatbot Arena (2025) — 1st place
- **Related follow-up** to LMSYS, multilingual preference prediction.
- **Models**: large decoder LLMs (Qwen/Gemma family) fine-tuned with LoRA as sequence classifiers; preference-pair framing + extensive external arena data.
- Link (1st-place writeup): https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/whitefebruary-1st-place-solution
- Competition: https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena

### Eedi — Mining Misconceptions in Mathematics (2024) — 1st place
- **Team**: "MTH 101" (official 1st-place detailed solution).
- **Task**: for each multiple-choice math question + wrong answer (distractor), retrieve the top-25 likely misconceptions, MAP@25.
- **Pipeline (retrieve → rerank)**: two **`qwen2.5`-based bi-encoder retrievers** fine-tuned with **LoRA + contrastive learning** on a mix of ~10k synthetic + ~1.8k real pairs to fetch top candidate misconceptions; then **LLM listwise reranking** (chain-of-thought with Qwen2.5) to reorder the top candidates; heavy **synthetic misconception data generation** to cover the long tail.
- Link (1st-place writeup): https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution
- Link (Eedi recap): https://www.eedi.com/news/from-wrong-answers-to-real-insights-how-we-used-a-kaggle-challenge-to-map-student-misconceptions
- Competition: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics

### MAP — Charting Student Math Misunderstandings (2025) — 1st place
- **Team**: 1st place won by a Kaggle Grandmaster from Rist (official "1st Place Solution" writeup); hosted by Vanderbilt University + The Learning Agency, $55k pool, 10/2025 deadline.
- **Task**: from a student's free-text math explanation, predict the underlying math misconception(s) / faulty reasoning to assist teacher feedback; code competition.
- **Approach**: LLM-based classification/ranking of misconception categories over student explanations, ensembling and careful handling of the long-tail label space (consistent with the broader misconception-mining recipe of Eedi).
- Link (1st-place writeup): https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution
- Competition: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings

---

## Patterns worth stealing (synthesis)

- **Data > model in the LLM era**: Detect-AI (generate your own essays from PERSUADE), Eedi / MAP (synthetic misconceptions), NBME/ELL (pseudo-labels), Jigsaw-Multilingual (iterative PL bootstrap). The win came from manufacturing in-distribution training data.
- **Read the distribution shift before the model**: AES 2.0 (data-source classification head to bridge two grading regimes; 619th public → 1st private). When public and private come from different sources, calibration beats raw accuracy.
- **Distill big → small for inference budgets**: LMSYS (70B teachers → `gemma-2-9b` student, LoRA-averaged), PII (MSD teachers → student). Keeps accuracy under the GPU submission limit.
- **Retrieve-then-rerank** is the canonical LLM-QA/matching recipe: LLM Science Exam (BM25 → DeBERTa rerank → MC), Eedi (bi-encoder retrieve → LLM listwise rerank), Learning Equality (bi-encoder retrieve → cross-encoder rerank).
- **Exploit the metric**: Prompt Recovery (mean-prompt cosine hack), Tweet Sentiment (space-offset label-noise post-processing), LLM 20 Questions (binary search ≈ 2^20). Read the metric before the data.
- **Multi-stage stacking still wins regression**: CommonLit Evaluate / ELL (DeBERTa OOF → GBDT stack), Tweet Sentiment (transformer → char-CNN/RNN/WaveNet).
- **Domain MLM + custom vocab** for niche corpora: QUEST (StackExchange LaTeX/code tokens), NBME (clinical MLM).
- **Diverse-architecture ensembles + KD** for token tasks: PII (5 DeBERTa with MSD/BiLSTM heads + KD), Feedback 2021 (Longformer + multiple DeBERTa sizes), Jigsaw 2018 (RNN/CNN/Transformer + tail-sentence models).
- **TTA + pseudo-labeling** is the oldest reliable lever: Jigsaw 2018, Jigsaw-Multilingual 2020, Tweet Sentiment 2020.

---

## Sources

- https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/writeups/toxic-crusaders-1st-place-solution-overview
- https://blog.ceshine.net/post/kaggle-toxic-comment-classification-challenge/
- https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/writeups/qishen-ha-8th-place-solution-4-models-simple-avg
- https://blog.ceshine.net/post/kaggle-jigsaw-toxic-2019/
- https://github.com/leecming82/jigsaw-multilingual
- https://blog.ceshine.net/post/multilingual-toxic-classification/
- https://github.com/mathislucka/kaggle_clrp_1st_place_solution
- https://github.com/TakoiHirokazu/kaggle_commonLit_readability_prize
- https://www.linkedin.com/pulse/our-kaggle-winning-solution-recap-commonlit-evaluate-ivan-isaev--a4bgf
- https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446712
- https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/498478
- https://hippocampus-garden.com/kaggle_aes2/
- https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389
- https://github.com/ubamba98/feedback-prize
- https://github.com/tascj/kaggle-feedback-prize-2021
- https://github.com/ybabakhin/kaggle-feedback-effectiveness-1st-place-solution
- https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536
- https://www.kaggle.com/competitions/feedback-prize-english-language-learning/writeups/autox-rohit-yevhenii-1st-place-solution
- https://github.com/rohitsingh02/kaggle-feedback-english-language-learning-1st-place-solution
- https://github.com/oleg-yaroshevskiy/quest_qa_labeling
- https://medium.com/kaggle-blog/the-3-ingredients-to-our-success-winners-dish-on-their-solution-to-googles-quest-q-a-labeling-c1a63014b88
- https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332492
- https://github.com/psinger/kaggle-curriculum-solution
- https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812
- https://github.com/heartkilla/kaggle_tweet
- https://www.kaggle.com/theoviel/character-level-model-magic
- https://www.nbme.org/news/six-teams-recognized-nlp-advances-nbmes-patient-note-scoring-competition
- https://arxiv.org/html/2401.12994v1
- https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/writeups/fold-zero-1st-place-solution-ensemble-of-diverse-d
- https://github.com/bogoconic1/pii-detection-1st-place
- https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436674
- https://hippocampus-garden.com/kaggle_llm/
- https://github.com/ironbar/prompt_recovery
- https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/comprehensive-1st-place-write-up
- https://github.com/rbiswasfc/llm-detect-ai
- https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/llmlab-1st-public-9th-private-llmlab-solution-summ
- https://www.kaggle.com/competitions/llm-20-questions/writeups/c-number-1st-place-solution
- https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629
- https://github.com/tascj/kaggle-lmsys-chatbot-arena
- https://github.com/DaoyuanLi2816/Kaggle-4th-Place-Solution-LMSYS-Chatbot-Arena-Human-Preference-Predictions
- https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/writeups/whitefebruary-1st-place-solution
- https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution
- https://www.eedi.com/news/from-wrong-answers-to-real-insights-how-we-used-a-kaggle-challenge-to-map-student-misconceptions
- https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution

---
_Curated via public-source research (Kaggle Discussions, official solution repos, arXiv, blogs). Verify any specific link before relying on it; gold write-ups live in each competition's Discussion tab._
