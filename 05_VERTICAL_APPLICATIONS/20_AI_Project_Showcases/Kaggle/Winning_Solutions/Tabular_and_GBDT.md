# Kaggle Winning Solutions — Tabular / Feature Engineering / GBDT

A dense, link-verified index of winning and top solutions for the canonical tabular / GBDT Kaggle competitions. Emphasis: **feature engineering**, **GBDT** (LightGBM / XGBoost / CatBoost), **CV strategy**, **leak handling**, and **ensembling / stacking**. Every entry links to a real, public source (Kaggle discussion writeup, official solution repo, HF/arXiv, or Kaggle blog interview). Recurring meta-patterns are summarized at the end.

> Conventions: `comp` = competition, `LB` = leaderboard, `OOF` = out-of-fold, `FE` = feature engineering, `DAE` = denoising autoencoder, `AV` = adversarial validation, `TE` = target encoding.

---

## Quick reference table

| Competition (year) | Winning core | Signature trick | Link |
|---|---|---|---|
| Porto Seguro Safe Driver (2017) | DAE + NN blend (+1 LGB) | Denoising autoencoder representation, "swap noise", GaussRank | [writeup](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629) |
| Home Credit Default Risk (2018) | LGB + XGB stack | Time-window aggregations, forward feature selection | [1st writeup](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821) |
| Santander Value (2018) | Leak reconstruction | Time-series leak across rows/cols (Giba) | [leak thread](https://www.kaggle.com/competitions/santander-value-prediction-challenge/discussion/61329) |
| ELO Merchant (2019) | LGB single + stack | Outlier binary classifier + linear interpolation blend | [1st writeup](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/discussion/82036) |
| Microsoft Malware (2019) | LGB + AV | Heavy train/test drift; adversarial validation | [overview](https://www.kaggle.com/competitions/microsoft-malware-prediction) |
| Santander Cust. Transaction (2019) | "Magic" count + LGB/NN | Real-vs-fake test split, per-value frequency feature | [1st writeup](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003) |
| IEEE-CIS Fraud (2019) | XGB+LGB+CatBoost | UID = card1+addr1+(day−D1), group aggregations | [1st writeup](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284) |
| MoA / lish-moa (2020) | NN blend + tree | Multi-label NN ensemble, label smoothing, PCA/QT | [1st writeup](https://www.kaggle.com/competitions/lish-moa/discussion/201510) |
| Jane Street (2021) | Supervised AE + MLP | Bottleneck autoencoder feeding MLP, purged CV | [1st writeup](https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/224348) |
| Optiver Realized Vol. (2021) | LGB/NN + GRU | time_id reordering via tick size; nearest-neighbor aggs | [1st writeup](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970) |
| Ubiquant Market (2022) | NN/GBDT ensemble | Market-average feature; "betting strategy" CV | [1st writeup](https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/337470) |
| AMEX Default (2022) | LGB + GRU/Transformer | "after-pay" features, last/mean/diff aggs, knowledge distillation | [1st repo](https://github.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution) |
| Optiver Trading-at-the-Close (2023) | LGB / LGB+NN | Imbalance & triplet-imbalance features, online retrain | [1st writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446) |
| Home Credit Risk Stability (2024) | GBDT + stability | Gini-stability metric, weekly-decay robustness | [1st writeup](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337) |

---

## Detailed entries

### 1. Porto Seguro's Safe Driver Prediction (2017)
- **Winner / rank:** Michael Jahrer, 1st / 5,169 teams. Metric: normalized Gini.
- **Core:** Blend of **6 models — 1× LightGBM + 5× neural nets**. A rare tabular comp where pure XGBoost did *not* win.
- **Signature FE:** Neural nets trained on the hidden activations of a **denoising autoencoder (DAE)** built on **train+test** features (unsupervised representation learning). DAE: 221 in/out features, 1 bottleneck + 4 deep stacked layers.
- **Tricks:** **"swap noise"** data augmentation (corrupt 7–20% of inputs by swapping values within a column), **GaussRank** normalization. No missing-value imputation, minimal manual FE.
- **Why it matters:** Canonical reference for representation learning on tabular data; the swap-noise DAE pattern recurs in later comps (Jane Street, AMEX NN tracks).
- **Links:** [Kaggle writeup](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629) · [GINK03/kaggle-dae (reimpl)](https://github.com/GINK03/kaggle-dae)

### 2. Home Credit Default Risk (2018)
- **Winner / rank:** Team **Home Aloan**, 1st / 7,176 teams. Metric: AUC.
- **Core:** Many **LightGBM + XGBoost** models with stratified K-fold, then stacking. Conclusion: **FE > tuning/stacking**.
- **Signature FE:** Time-window aggregations over the auxiliary tables (bureau, previous applications, installments, POS, credit card) computed over *specific* time periods; **weighted moving averages** on time-based features.
- **Feature selection:** **Forward feature selection with Ridge regression** to prune thousands of generated aggregations.
- **CV:** Stratified K-fold; OOF stacking.
- **Links:** [1st place writeup](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821) · [kozodoi/Kaggle_Home_Credit](https://github.com/kozodoi/Kaggle_Home_Credit)

### 3. Santander Value Prediction Challenge (2018)
- **Outcome:** Comp became a **leak hunt**. **Giba** (then #1 global) discovered the anonymized data was a **time series in two dimensions** — values repeated across rows and columns.
- **Signature trick:** Reconstruct the hidden time series ("lag" structure) by brute-force matching repeated value sequences; GBDT on raw features stalled above RMSLE 1.0 until the leak was exploited.
- **Lesson:** Classic example of **leak detection** dominating an anonymized-tabular regression task — top scores were driven by reconstructing the series, not by modeling.
- **Links:** [Giba leak / "property" thread](https://www.kaggle.com/competitions/santander-value-prediction-challenge/discussion/61329) · [comp](https://www.kaggle.com/c/santander-value-prediction-challenge)

### 4. ELO Merchant Category Recommendation (2019)
- **Metric:** RMSE on loyalty score; dominated by a handful of extreme **outliers** (loyalty ≈ −33.22).
- **Winning idea:** Train a separate **binary classifier for outliers**, then blend via **linear interpolation** on the classifier probability instead of hard thresholding:
  `final = p_outlier·(−33.21928) + (1 − p_outlier)·pred_no_outlier`
  → reported **~0.015 local CV boost** vs. training on the raw target directly.
- **Models:** LightGBM (+ stacked LGB/XGB); aggregation FE over historical & new merchant transactions (counts, recency, category aggregates).
- **Links:** [1st place writeup](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/discussion/82036)

### 5. Microsoft Malware Prediction (2019)
- **Challenge:** Severe **train/test distribution shift** over time — `AvSigVersion`, `EngineVersion`, `AppVersion`, `Census_OSVersion`, `Census_OSBuildRevision` carried values in test unseen in train. Strong public→private LB shakeup.
- **Key technique:** **Adversarial validation** — a LightGBM classifier separating train vs. test reached ~0.93 AUC, proving the sets were nearly separable; top teams down-weighted / dropped time-leaking version features and validated against a time-forward split.
- **Lesson:** Canonical **non-stationary tabular** comp; AV and feature-drift handling matter more than raw FE.
- **Links:** [comp overview](https://www.kaggle.com/competitions/microsoft-malware-prediction) · [AV walkthrough](https://osfork.com/2019/03/10/microsoft-kaggle-challenge-adversarial-validation/)

### 6. Santander Customer Transaction Prediction (2019)
- **Metric:** AUC; 200 anonymized numeric features `var_0…var_199`.
- **The "magic":** Test set was **half synthetic** (real + fake rows); fake rows could be detected because their per-column values lacked the **uniqueness/frequency structure** of real rows. The winning insight: add **per-value frequency counts**, computed using only the *real* test rows + train, so each feature contributes independently.
- **Models:** LightGBM on ~400 engineered (value + count) features; plus a custom **NN on ~600 features** with per-feature structure. Used **pseudo-labeling** and data augmentation.
- **Reported:** Winning NN ≈ 0.92687 public / 0.92546 private.
- **Links:** [1st place writeup](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)

### 7. IEEE-CIS Fraud Detection (2019)
- **Winner:** Team **FraudSquad** (Chris Deotte et al.), 1st / 6,381 teams. Private AUC ≈ **0.9459**. Metric: AUC.
- **The magic UID:** Construct a latent **client/card identity** `uid = card1_addr1_D1n` where `D1n = transaction_day − D1`. Group-aggregate (mean/std/counts) numeric & categorical columns **by uid** → lets the model classify *clients* rather than isolated transactions.
- **Models:** Ensemble of **XGBoost + LightGBM + CatBoost**.
- **CV / leak handling:** Strict **time-respecting** aggregations to avoid future leakage; validation aligned with the temporal train→test split.
- **Links:** [1st writeup pt.1](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284) · [1st writeup pt.2](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111321) · [NVIDIA blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)

### 8. Mechanisms of Action (MoA / lish-moa) (2020)
- **Winner:** Team **Hungry for Gold**, 1st / 4,373 teams. Metric: mean column-wise log loss (multi-label, 206 targets).
- **Core:** **NN-heavy blend** — multiple feed-forward nets, **TabNet**, and tree models, blended with a **weight search** over OOF predictions.
- **Signature FE / tricks:** **PCA + QuantileTransformer** features on gene-expression (`g-`) and cell-viability (`c-`) columns, variance thresholding, **label smoothing**, multi-seed bagging, and careful handling of the non-scored auxiliary targets for pretraining.
- **CV:** **MultilabelStratifiedKFold** (iterative stratification) — essential for the multi-label target.
- **Links:** [1st place writeup](https://www.kaggle.com/competitions/lish-moa/discussion/201510) · [guitarmind/kaggle_moa_winner_hungry_for_gold](https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold)

### 9. Jane Street Market Prediction (2021)
- **Winner:** **Yirun Zhang** (team "cats trading"), 1st. Metric: utility on a high-noise financial response.
- **Core:** **Supervised denoising autoencoder + MLP** — a bottleneck AE is trained jointly with the supervised head; the AE-reconstructed/encoded features feed the MLP, with the auxiliary reconstruction loss acting as regularization on noisy market features.
- **Tricks:** Gaussian noise injection (DAE lineage from Porto Seguro), multi-target (action over multiple resp horizons), heavy seed bagging.
- **CV:** **Purged / grouped time-series CV** to avoid leakage across overlapping financial windows.
- **Links:** [1st writeup (supervised AE + MLP)](https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/224348) · [scaomath/kaggle-jane-street](https://github.com/scaomath/kaggle-jane-street)

### 10. Optiver Realized Volatility Prediction (2021)
- **Winner / rank:** 1st / 3,852 teams. Metric: RMSPE.
- **The leak/reordering trick:** Reverse-engineer the chronological order of the anonymized `time_id`s by recovering real price levels from normalized prices + **tick size**, building a directed graph over time_ids and approximating a **shortest-Hamiltonian-path** ordering. This enables **time-aware features** on otherwise shuffled buckets.
- **FE:** Order-book & trade features (WAP, log-returns, realized vol over sub-windows, bid-ask spread, depth imbalance), plus **nearest-neighbor aggregations** across stocks/time_ids.
- **Models:** LightGBM + NN (incl. GRU/1D-CNN on book sequences), blended.
- **Links:** [1st place writeup](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970) · [michaelpoluektov/orvp (7th, reordering)](https://github.com/michaelpoluektov/orvp)

### 11. Ubiquant Market Prediction (2022)
- **Winner:** Team **K_I_Y**, 1st. Metric: mean Pearson correlation per time_id.
- **Core idea ("Our Betting Strategy"):** Treat LB probing/selection as a bet — careful **CV-vs-LB trust** and submission selection on a noisy target.
- **Signature FE:** Add **market-average (cross-sectional mean) features** per time_id so each investment is modeled relative to the market state; ensemble of NN + GBDT.
- **CV:** Time-aware / grouped by time_id to respect the forward-looking nature.
- **Links:** [1st place writeup](https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/337470) · [pinouche/ubiquant-kaggle-competition (top 1%)](https://github.com/pinouche/ubiquant-kaggle-competition)

### 12. American Express — Default Prediction (2022)
- **Winner:** Team **jxzly** (1st / 4,874 teams). Metric: custom AMEX metric (0.5·G + 0.5·D@4%).
- **Pipeline (repo stages S1–S7):** data denoising → manual FE → **time-series aggregations** (last / mean / std / min / max / **diff / lag**) over each customer's monthly statements → feature consolidation → **LightGBM** → **NN (GRU/Transformer over the statement sequence)** → ensemble.
- **Signature FE:** **"after-pay" features** (e.g., balance minus payment ratios) and last-statement emphasis; **knowledge distillation** from LightGBM OOF into NNs was widely used by top teams (e.g., Chris Deotte's transformer + LGB distillation).
- **CV:** Customer-level StratifiedKFold.
- **Links:** [1st place repo](https://github.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution) · [1st writeup](https://www.kaggle.com/competitions/amex-default-prediction/discussion/348111) · [DeepWiki summary](https://deepwiki.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution)

### 13. Optiver — Trading at the Close (2023)
- **Winner:** Team **hyd**, 1st / 4,436 teams. Metric: MAE on closing-auction price move. NASDAQ closing-auction data.
- **Core:** Strong **single LightGBM** (and LGB+MLP ensembles among other top teams), with **FE as the alpha source**.
- **Signature FE:** Order-imbalance features, **triplet-imbalance** combinations across {bid, ask, wap, reference, near, far} prices, rolling/lagged returns, stock-level and global aggregates; **online retraining / inference** within the time-series API.
- **CV:** Time-forward validation; careful to avoid look-ahead in lagged features.
- **Links:** [1st place writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446) · [nimashahbazi/optiver-trading-close (top, NN/LGB)](https://github.com/nimashahbazi/optiver-trading-close)

### 14. Home Credit — Credit Risk Model Stability (2024)
- **Metric:** **Gini-stability** — mean AUC across weeks penalized by a downward trend and variance over time (rewards *stable* models, not just high AUC).
- **Winning theme ("My Betting Strategy"):** Optimize directly for stability — favor robust GBDT features that don't decay week-to-week, prune unstable features, and select submissions on the stability-weighted metric rather than raw AUC.
- **Models:** GBDT (LightGBM/CatBoost) with depth/regularization tuned for temporal robustness; sober FE on the relational feature-definition schema.
- **Links:** [1st place writeup](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337) · [comp](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)

### 15. Two Sigma Financial Modeling Challenge (2017)
- **Note:** No single canonical *1st-place* public writeup; the best-documented top finishes:
  - **2nd place — Nima Shahbazi & Chahhou Mohamed:** Kaggle winners' interview detailing ridge/ensemble modeling on the anonymized financial panel with strict overfitting control on a tiny-signal target.
  - **5th place — Team Best Fitting** (only team top-5 on *both* public & private LBs): emphasis on robust validation and conservative ensembling on a noisy code-competition target.
- **Lesson:** Extreme low signal-to-noise → **trust CV over LB**, regularize hard, avoid overfitting the public LB (a recurring tabular-finance theme).
- **Links:** [2nd place interview (Kaggle blog)](http://blog.kaggle.com/2017/05/25/two-sigma-financial-modeling-challenge-winners-interview-2nd-place-nima-shahbazi-chahhou-mohamed/) · [5th place Best Fitting interview](https://medium.com/kaggle-blog/two-sigma-financial-modeling-code-competition-5th-place-winners-interview-team-best-fitting-279a493c76bd)

---

## Tabular Playground Series / Playground Series — winning patterns

The synthetic **Playground Series (PS)** comps are a laboratory for tabular meta-technique. Recurring winning recipes (verified across multiple PS writeups and the NVIDIA Grandmaster playbook):

- **Multi-level stacking:** Top solutions routinely build **L1 zoos of 75–150+ diverse models selected from 500–850 candidates**, feeding 3–4 stacking levels; final meta-layer often **Ridge / Logistic Regression on logit-transformed OOF preds**.
- **Hill-climbing ensembles:** Start from the strongest single model, greedily add weighted models that improve OOF, repeat — a staple of recent PS 1st places.
- **Diversity over single-model perfection:** Vary model family (LGB/XGB/CatBoost/NN), depth, learning rate, regularization, seeds, **and FE policy** (different feature blocks per model) — ensembling across *FE policies* yields large gains.
- **FE candidate blocks + selection:** Generate ratios, polynomials, target/frequency encodings, then **block-wise feature selection** rather than dumping all features into one model.
- **"Original data" trick:** Many PS datasets are CTGAN-generated from a real source; concatenating the **original real dataset** (when public) as extra rows is a frequent boost.
- **GPU-accelerated stacking** (cuML/cuDF) to iterate hundreds of base models cheaply.

**Representative PS writeups:**
- [PS S5E12 — 1st: Hill Climbing + Ridge ensemble](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)
- [PS S6E4 — 4th: "more ensemblers than models"](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/4th-place-more-ensemblers-than-models)
- [PS S6E5 — 8th: L5 ensemble](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/l5-ensemble)
- [NVIDIA: Kaggle Grandmasters Playbook — 7 tabular techniques](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [NVIDIA: Winning 1st with stacking (cuML)](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/)

---

## Cross-cutting meta-patterns (what actually wins tabular)

**Feature engineering**
- **Entity/UID reconstruction** dominates fraud & customer comps: recover a latent client id, then **group-aggregate** every column by it (IEEE-CIS, AMEX, ELO).
- **Time aggregations:** last / mean / std / min / max / **diff / lag** over a customer or time window (AMEX, Home Credit); weighted moving averages.
- **Frequency / count encoding** of categorical & even numeric values (Santander CTP "magic", IEEE-CIS).
- **Cross-sectional / market-relative features** in finance (Ubiquant market average, Optiver imbalance & nearest-neighbor aggs).
- **Representation learning** when trees plateau: **denoising autoencoders + swap noise** (Porto Seguro, Jane Street, AMEX NN tracks).

**GBDT choices**
- **LightGBM** is the default workhorse; **CatBoost** shines with high-cardinality categoricals; **XGBoost** adds ensemble diversity. Winning blends usually contain **all three**.

**CV strategy**
- Match CV to the leakage structure: **StratifiedKFold** (Home Credit), **MultilabelStratified** (MoA), **GroupKFold by customer** (AMEX), **purged/time-forward CV** (Jane Street, Optiver, Ubiquant, Two Sigma).
- **Trust CV over a noisy public LB** in low-signal finance comps.

**Leak handling**
- Hunt leaks explicitly (Santander Value time-series, Optiver `time_id` reordering) — sometimes the leak *is* the competition.
- Use **adversarial validation** to detect train/test drift and drift-prone features (Microsoft Malware).
- Enforce **time-respecting aggregations** so client-level features never see the future (IEEE-CIS).

**Ensembling / stacking**
- **Multi-level stacking** + **hill-climbing weight search** on OOF; meta-model = Ridge/LogReg on logits.
- **Knowledge distillation** GBDT→NN (AMEX) to fuse tree signal into sequence models.
- Diversity across **model family × hyperparams × FE policy × seeds** beats over-tuning one model.

---

## Sources

- Porto Seguro 1st: https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629 · https://github.com/GINK03/kaggle-dae
- Home Credit 1st: https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821 · https://github.com/kozodoi/Kaggle_Home_Credit
- Santander Value leak: https://www.kaggle.com/competitions/santander-value-prediction-challenge/discussion/61329
- ELO 1st: https://www.kaggle.com/competitions/elo-merchant-category-recommendation/discussion/82036
- Microsoft Malware: https://www.kaggle.com/competitions/microsoft-malware-prediction · https://osfork.com/2019/03/10/microsoft-kaggle-challenge-adversarial-validation/
- Santander CTP 1st: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003
- IEEE-CIS 1st: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284 · https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111321 · https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
- MoA 1st: https://www.kaggle.com/competitions/lish-moa/discussion/201510 · https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold
- Jane Street 1st: https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/224348 · https://github.com/scaomath/kaggle-jane-street
- Optiver Realized Vol 1st: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970 · https://github.com/michaelpoluektov/orvp
- Ubiquant 1st: https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/337470 · https://github.com/pinouche/ubiquant-kaggle-competition
- AMEX 1st: https://github.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution · https://www.kaggle.com/competitions/amex-default-prediction/discussion/348111
- Optiver Trading-at-the-Close 1st: https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446 · https://github.com/nimashahbazi/optiver-trading-close
- Home Credit Stability 1st: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337
- Two Sigma Financial Modeling: http://blog.kaggle.com/2017/05/25/two-sigma-financial-modeling-challenge-winners-interview-2nd-place-nima-shahbazi-chahhou-mohamed/ · https://medium.com/kaggle-blog/two-sigma-financial-modeling-code-competition-5th-place-winners-interview-team-best-fitting-279a493c76bd
- Playground Series patterns: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ · https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ · https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl
- Solution index: https://farid.one/kaggle-solutions/ · https://github.com/faridrashidi/kaggle-solutions

---
_Curated via public-source research (Kaggle Discussions, official solution repos, arXiv, blogs). Verify any specific link before relying on it; gold write-ups live in each competition's Discussion tab._
