# The Kaggle Grandmaster Playbook — Recurring Winning Techniques

A dense, source-grounded synthesis of the meta-techniques that repeatedly win Kaggle competitions. Every entry links to a real, public artifact (Kaggle write-up/discussion, GitHub, blog, paper, or docs). Competition names, ranks, and key methods are cited. Snake_case-friendly throughout.

> The single highest-leverage statement in competitive ML, repeated by virtually every grandmaster: **build a trustworthy local validation, then trust it over the public leaderboard.** Everything below is downstream of that.

---

## TL;DR — Patterns → When to use → Example competition

| pattern | when_to_use | example_competition (rank) | source |
|---|---|---|---|
| `StratifiedKFold` | i.i.d. rows, classification, class imbalance | Home Credit Default Risk 2018 (#1, stratified k-fold) | [#1 write-up](https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution) |
| `GroupKFold` / `StratifiedGroupKFold` | repeated entities (patient/user/image-source) leak across folds | medical imaging / multi-image-per-patient tasks | [sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) |
| time-based / `Purged K-Fold` | temporal data, look-ahead leakage | Optiver — Trading at the Close 2023 (#1, purged CV) | [#1 write-up](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution) |
| adversarial_validation | suspected train/test distribution shift | tabular shake-up risk detection | [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) |
| hill_climbing ensemble | many diverse OOF predictions to blend | Playground S5E12 (#1, hill climb + ridge) | [#1 write-up](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl) |
| stacking (2nd-level model) | hill climb plateaus; rich OOF matrix | cuML stacking pro-tip (#1) | [NVIDIA stacking](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/) |
| pseudo_labeling | abundant unlabeled / test data | Instant Gratification (QDA, 0.969) | [Deotte notebook](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969) |
| test_time_augmentation (TTA) | vision/audio, cheap inference budget | RSNA Pneumonia 2018 (flips + multi-fold avg) | [solution repo](https://github.com/tatigabru/kaggle-rsna) |
| leak / "magic" exploitation | synthetic data, ordering, metadata artifacts | Santander Customer Transaction 2019 (#1, fake-sample detection) | [#1 discussion](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003) |
| target/feature_engineering | tabular dominance | Home Credit 2018 (#1: FE > tuning) | [#1 write-up](https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution) |
| post_processing | metric-specific calibration / rounding | ordinal & ranking metrics | [Kaggle Book ch.10](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/Text/Chapter_10.xhtml) |
| seed/full-data retraining | squeeze final variance reduction | NVIDIA "extra training" technique | [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) |

---

## 1. Robust cross-validation design

The foundation. A leaky or misaligned CV produces a model that overfits the public LB and collapses on the private split ("the shake-up").

### 1.1 Match the split to the data-generating process
- **`StratifiedKFold`** — default for classification, preserves class ratios per fold. Essential under imbalance. Used across countless top solutions including Home Credit Default Risk (#1, 2018), where LightGBM/XGBoost were trained with stratified k-fold. [Write-up](https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution)
- **`GroupKFold`** — when a single entity (patient, user, document, image source) appears in multiple rows, group it so the *same entity never spans train and validation*. Otherwise the model memorizes the entity, not the signal.
- **`StratifiedGroupKFold`** — preserve class distribution *and* keep groups disjoint; ideal for imbalanced grouped data. [sklearn reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) · [CV overview](https://scikit-learn.org/stable/modules/cross_validation.html)
- **Time-based / purged splits** — for temporal targets, validate only on *future* data and **purge** rows near the train/val boundary to kill look-ahead leakage. The Optiver — Trading at the Close (#1, 2023) solution used a purged k-fold variant for the last-10-minutes NASDAQ prediction. [#1 write-up](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)

### 1.2 Adversarial validation — detect train/test drift
Train a binary classifier to distinguish train vs. test rows (target = `is_test`). If AUC ≈ 0.5, distributions match and random CV is safe; if AUC is high, the sets differ and you must (a) drop the most-discriminative features, (b) reweight/select training rows that look like test, or (c) build a validation set resembling the test distribution. Named explicitly in the NVIDIA Grandmasters playbook's "smarter EDA" step (check train-vs-test shift before trusting CV). [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) · concept primer: [Ilias Antonopoulos](https://ilias-ant.github.io/blog/adversarial-validation/)

### 1.3 The cardinal rule
> "If CV improves but LB drops, investigate data structure." The Playground S5E12 (#1) winner explicitly frames distribution shift as a "competition killer," concluding *trust your CV and trust the process* — a slightly lower CV with better generalization often wins. [#1 write-up](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)

**Pitfalls to avoid:** fitting any preprocessing (target encoding, scalers, imputers, feature selection) on the full dataset before splitting → leakage; choosing fold count without regard to data size; ignoring that public LB is itself a small, noisy held-out set.

---

## 2. Ensembling & stacking

Diversity beats individual strength. The reliable recipe: many *decorrelated* models → combine.

### 2.1 Blending (weighted average)
Simple weighted average of model predictions. Cheap, robust, hard to overfit. The base case for every ensemble.

### 2.2 Hill climbing
Start from the single strongest model; greedily test adding each candidate at various weights; keep only additions that improve OOF/validation; repeat until no gain. It is the grandmaster default for combining a large bag of OOF predictions and is technique #4 of the NVIDIA 7. [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- Reusable implementation: [`Matt-OP/hillclimbers`](https://github.com/Matt-OP/hillclimbers) — Python module that iteratively blends model predictions via hill climbing.
- Reference notebook: [Hillclimb ensembling](https://www.kaggle.com/code/hhstrand/hillclimb-ensembling)
- Winning use: Playground S5E12 (#1) — hill climbing refined over many competitions, then **Ridge stacking** on top once the hill climb plateaued. [#1 write-up](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)

### 2.3 Stacking (second-level meta-model)
Train a meta-learner on the **OOF predictions** of base models. When hill climbing stalls, a two-stage ensemble (e.g., Ridge / cuML on top of the strongest base OOFs) often breaks the plateau. Technique #5 of the NVIDIA 7; demonstrated as a #1-place pro-tip with cuML. [NVIDIA stacking](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/) · also Playground S4E9 (#3) "Gather, Ridge, Repeat". [#3 write-up](https://www.kaggle.com/competitions/playground-series-s4e9/writeups/optimistix-3rd-place-solution-an-open-secret-gathe)
- **Critical:** generate level-1 features with the *same* fold scheme so the meta-model never sees in-fold predictions (otherwise stacking leaks). Discussed in [CV strategy when blending/stacking](https://www.kaggle.com/general/18793) and [Kaggle Book ch.10](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/Text/Chapter_10.xhtml).

### 2.4 Optimized blend weights
Use `Optuna` / `scipy.optimize` to learn blend weights that maximize the competition metric on OOF predictions (more flexible than greedy hill climbing, but more prone to overfitting the OOF — regularize, and validate weight stability across folds).

### 2.5 Diversity sources (what to ensemble)
GBDTs (LightGBM, XGBoost, CatBoost) + NNs (MLP, transformers/CNN) + different seeds + different feature subsets + different folds. The NVIDIA playbook's technique #2 is literally "build a diverse set of baselines across model types right away." [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

---

## 3. Pseudo-labeling & knowledge distillation

Turn unlabeled / test data into training signal.

- **Mechanism:** train on labeled data → predict on unlabeled/test → add high-confidence predictions as pseudo-labels → retrain. Acts like a form of self-distillation; ensembles or multi-round pseudo-labeling usually beat single-pass. Technique #6 of the NVIDIA 7. [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- **Canonical demo:** Chris Deotte's *Pseudo Labeling — QDA — [0.969]* from Instant Gratification, the reference implementation copied across the community. [Notebook](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969)
- **In a winning solution:** Santander Customer Transaction (#1, 2019) added pseudo-labels — the 2700 highest-predicted test points as class 1, the 2000 lowest as class 0 — on top of the magic features. [#1 discussion](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
- **Cautions:** only fold pseudo-labels in *inside* each CV fold to avoid leaking test into validation; confidence-threshold or use soft labels; beware reinforcing your own errors (confirmation bias).
- **Knowledge distillation:** train a smaller/faster "student" on a strong ensemble's soft predictions — recovers most of the ensemble's accuracy under a tight inference/latency budget (common in code-competitions with runtime limits).

---

## 4. Test-time augmentation (TTA)

Augment each test sample (flips, crops, scales, multi-window, audio time-shifts), predict on each view, average. Nearly free accuracy in vision/audio when the inference budget allows.
- **Example:** RSNA Pneumonia Detection (2018) solutions averaged predictions across folds and horizontal flips (e.g., 3 models × 5 folds × 2 flips). [solution repo](https://github.com/tatigabru/kaggle-rsna) · [ChallengePneumo repo](https://github.com/emptyewer/kaggle-rsna2018)
- **Tips:** TTA transforms should mirror *train-time* augmentations; average in probability/logit space appropriate to the metric; balance the number of views against the runtime cap.

---

## 5. Handling leakage & exploiting "magic"

The dual nature of leakage: defend against it in your CV, and (where allowed) detect data artifacts that competitors miss.

- **Santander Customer Transaction (#1, 2019) — the canonical "magic":** half the test set was synthetic. The community kernel "List of Fake Samples and Public/Private LB split" (YaG320) flagged ~100k fake rows via unique-value counts. Real test rows let teams compute genuine per-feature value-frequency features; the #1 team rode this from ~0.901 to ~0.92x AUC, plus shuffle augmentation (duplicating/shuffling target==1 rows 16×, target==0 4×) and pseudo-labels. [#1 discussion](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003) · [fake-samples discussion](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/87057)
- **General leakage classes to hunt or defend:** row-ordering / index leaks, ID or timestamp leaks, duplicate near-rows across train/test, target leakage via aggregates computed over the full set, and metadata (file size, group IDs) correlated with the target.
- **Survival heuristic:** treat any sudden CV/LB gap as a signal — either you have leakage or distribution shift. [Shake-up survival guide](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-tips-tricks-to-survive-a-kaggle-shake-up-23675beed05e)

---

## 6. Target & feature engineering

On tabular problems this is still the single biggest differentiator — "feature engineering proved more useful than model tuning and stacking" (Home Credit #1).

- **Home Credit Default Risk (#1, 2018):** multi-table aggregations over time windows, weighted moving averages, ratio/product features, and a standout `neighbors_target_mean_500` (mean target of 500 nearest neighbors in `EXT_SOURCE` × `CREDIT_ANNUITY_RATIO` space). Feature reduction via **forward selection with Ridge**, then LightGBM/XGBoost with stratified k-fold. [#1 write-up](https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution) · community repo [`kozodoi/Kaggle_Home_Credit`](https://github.com/kozodoi/Kaggle_Home_Credit)
- **Target encoding** (mean/likelihood encoding) for high-cardinality categoricals — *must* be computed out-of-fold / with smoothing to avoid leakage.
- **Categorical interactions:** combining categorical columns to expose hidden interaction signal is called out explicitly in the NVIDIA playbook (technique #3 — generate hundreds/thousands of features). [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- **Domain alpha (finance):** Optiver — Trading at the Close treated feature engineering as "the key to alpha discovery" (imbalance, microprice, rolling/cross-sectional stats) feeding LightGBM. [#1 write-up](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
- **Target transforms:** log/rank/Box-Cox transforms of regression targets; reframing the loss to match the metric.

---

## 7. Post-processing

Metric-aware adjustments applied *after* model inference — often a few free positions.
- Probability **calibration** (Platt/isotonic) when the metric rewards calibrated outputs.
- **Rounding / thresholding** for ordinal or quantized targets; optimize the decision threshold on OOF for F1/accuracy.
- **Rank normalization** before averaging models when the metric is rank-based (AUC) and model score scales differ.
- Domain **constraints** (clip to physical/known ranges, enforce monotonicity/ordering). See ensembling+post-processing treatment in [Kaggle Book ch.10](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/Text/Chapter_10.xhtml).

---

## 8. Squeeze techniques (variance reduction)

NVIDIA "extra training" (technique #7): after fixing hyperparameters, **retrain across multiple random seeds and average**, then **retrain on 100% of the data** for the final submission. Cheap, reliable variance reduction. [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

---

## 9. Hardware, efficiency & tooling

| tool | role | source |
|---|---|---|
| `LightGBM` | leaf-wise histogram GBDT; fast, memory-light, tabular workhorse | [Optiver #1 uses LGBM](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution) |
| `XGBoost` / `CatBoost` | GBDT diversity; CatBoost for native categoricals | [Home Credit #1](https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution) |
| `Optuna` | hyperparameter & blend-weight optimization | [Kaggle Book ch.10](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/Text/Chapter_10.xhtml) |
| `cuDF` / RAPIDS | GPU-accelerated feature engineering on huge tables | [cuDF pandas #1 pro-tip](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/) |
| `cuML` | GPU stacking / classical ML at scale | [cuML stacking #1 pro-tip](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/) |
| `hillclimbers` | drop-in hill-climbing blender | [`Matt-OP/hillclimbers`](https://github.com/Matt-OP/hillclimbers) |
| `scikit-learn` CV objects | `StratifiedKFold`/`GroupKFold`/`StratifiedGroupKFold`/`TimeSeriesSplit` | [CV docs](https://scikit-learn.org/stable/modules/cross_validation.html) |

**Efficiency tricks:** mixed precision + gradient checkpointing for large NN models; knowledge distillation to fit code-competition runtime caps; caching OOF prediction matrices so ensembling iterates in seconds; GPU dataframes for FE on multi-million-row tables.

---

## 10. The repeatable grandmaster loop

1. **EDA + adversarial validation** → quantify train/test shift and temporal structure. [NVIDIA playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
2. **Design a leakage-proof CV** that mirrors how the private set differs from train (group/time/stratify).
3. **Diverse baselines** across GBDTs and NNs; lock the validation harness.
4. **Heavy feature engineering** (and hunt for leaks/magic where legitimate).
5. **Pseudo-labeling / TTA** to extract more from unlabeled & test data.
6. **Hill climb → stack** over a large bag of decorrelated OOFs.
7. **Post-process** to the exact metric; **seed-average + full-data retrain** for the final sub.
8. **Trust CV over public LB** when selecting final submissions.

Curated meta-references: a community compendium of solution methods across hundreds of competitions ([`kyaiooiayk/Kaggle-Competitions-Analysis`](https://github.com/kyaiooiayk/Kaggle-Competitions-Analysis)) and Abhishek Thakur's *Approaching (Almost) Any Machine Learning Problem* ([repo + free PDF](https://github.com/abhishekkrthakur/approachingalmost), [PDF](https://github.com/abhishekkrthakur/approachingalmost/blob/master/AAAMLP.pdf)).

---

## Sources

- NVIDIA — The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data (Onodera, Viel, Titericz, Deotte): https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- NVIDIA — Grandmaster Pro Tip: Stacking with cuML (#1): https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/
- NVIDIA — Grandmaster Pro Tip: Feature Engineering with cuDF pandas (#1): https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/
- Kaggle — Home Credit Default Risk, 1st place write-up: https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution
- Kaggle — Santander Customer Transaction Prediction, #1 solution discussion: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003
- Kaggle — Santander fake-samples / public-private split discussion: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/87057
- Kaggle — Optiver, Trading at the Close, 1st place write-up: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- Kaggle — Playground S5E12, 1st place (Hill Climbing + Ridge Ensemble): https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl
- Kaggle — Playground S4E9, 3rd place (Gather, Ridge, Repeat): https://www.kaggle.com/competitions/playground-series-s4e9/writeups/optimistix-3rd-place-solution-an-open-secret-gathe
- Chris Deotte — Pseudo Labeling QDA [0.969] notebook: https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969
- Hillclimb ensembling notebook (hhstrand): https://www.kaggle.com/code/hhstrand/hillclimb-ensembling
- Matt-OP/hillclimbers (GitHub): https://github.com/Matt-OP/hillclimbers
- scikit-learn — StratifiedGroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- scikit-learn — Cross-validation guide: https://scikit-learn.org/stable/modules/cross_validation.html
- The Kaggle Book (2nd ed.), Ch.10 — Ensembling with Blending and Stacking: https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/Text/Chapter_10.xhtml
- Adversarial Validation primer (Ilias Antonopoulos): https://ilias-ant.github.io/blog/adversarial-validation/
- RSNA Pneumonia Detection solution repo (tatigabru): https://github.com/tatigabru/kaggle-rsna
- RSNA Pneumonia ChallengePneumo solution repo: https://github.com/emptyewer/kaggle-rsna2018
- Abhishek Thakur — Approaching (Almost) Any Machine Learning Problem (repo + PDF): https://github.com/abhishekkrthakur/approachingalmost
- Kaggle Competitions Analysis compendium (kyaiooiayk): https://github.com/kyaiooiayk/Kaggle-Competitions-Analysis
- Kaggle Handbook — Tips & Tricks to Survive a Shake-up: https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-tips-tricks-to-survive-a-kaggle-shake-up-23675beed05e
- CV strategy when blending/stacking (Kaggle discussion): https://www.kaggle.com/general/18793

---
_Curated via public-source research (Kaggle Discussions, official solution repos, arXiv, blogs). Verify any specific link before relying on it; gold write-ups live in each competition's Discussion tab._
