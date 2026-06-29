# ML Evaluation Metrics

> Quantitative measures of how well a model's predictions match ground truth, used to compare models, tune them, and decide whether they are fit for deployment.

## Why it matters

The metric you optimize defines what "good" means: a model that maximizes accuracy on a 99%-negative dataset can be useless, while one tuned for recall or AUC may be exactly right. Choosing a metric aligned with the business cost structure and the data distribution (class imbalance, ranking position, calibrated probabilities) is often more consequential than the choice of model. Metrics also drive model selection, early stopping, and hyperparameter search, so a mismatched metric silently steers the entire pipeline in the wrong direction.

## Core concepts

- **Confusion matrix** (binary): counts of TP, FP, TN, FN. Most classification metrics are functions of these four numbers.
- **Accuracy** = (TP+TN)/(TP+TN+FP+FN). Misleading under class imbalance; compare against the majority-class baseline.
- **Precision** = TP/(TP+FP) (purity of positive predictions). **Recall / Sensitivity / TPR** = TP/(TP+FN) (coverage of actual positives). **Specificity / TNR** = TN/(TN+FP).
- **F-beta** = (1+β²)·(precision·recall)/(β²·precision+recall); **F1** (β=1) is their harmonic mean. β>1 weights recall more.
- **Threshold-free ranking metrics:** **ROC-AUC** is the area under TPR-vs-FPR as the decision threshold sweeps; it equals P(score(positive) > score(negative)), i.e. the probability a random positive outranks a random negative. **PR-AUC / Average Precision** integrates precision over recall and is far more informative than ROC-AUC under heavy class imbalance (Davis & Goadrich, 2006).
- **Multiclass averaging:** *macro* (unweighted mean over classes, treats classes equally), *micro* (pool all decisions, dominated by frequent classes), *weighted* (by support).
- **Regression:** **MAE** = mean|y−ŷ| (robust, same units); **MSE/RMSE** = mean (y−ŷ)² (penalizes large errors, RMSE in original units); **R²** = 1 − SS_res/SS_tot (fraction of variance explained, can be negative); **MAPE** = mean|（y−ŷ)/y| (scale-free but unstable near 0); **Huber / pinball (quantile) loss** for robustness and quantile regression.
- **Ranking / IR:** **Precision@k**, **Recall@k**, **MAP**, **MRR**, and **NDCG@k** = DCG@k / IDCG@k, where DCG discounts graded relevance by log position (Järvelin & Kekäläinen, 2002).
- **Clustering:** *external* (vs ground-truth labels) — **ARI** (chance-adjusted Rand index), **NMI**, **homogeneity / completeness / V-measure**, **Fowlkes–Mallows**; *internal* (no labels) — **Silhouette**, **Davies–Bouldin**, **Calinski–Harabasz**.
- **Calibration:** a model is calibrated if predicted probability p occurs ≈p of the time. Measured with **reliability diagrams**, **Expected Calibration Error (ECE)**, and proper scoring rules — **Brier score** = mean (p−y)² and **log loss / cross-entropy** = −mean[y·log p + (1−y)·log(1−p)]. Proper scoring rules are minimized only by the true probabilities, so they jointly reward discrimination *and* calibration. Fix miscalibration post-hoc with **Platt scaling**, **isotonic regression**, or **temperature scaling**.

## Variants

| Task | Metric | Formula / Definition | Use when |
|---|---|---|---|
| Classification | Accuracy | (TP+TN)/N | Balanced classes, equal error costs |
| Classification | Precision | TP/(TP+FP) | False positives are costly |
| Classification | Recall (TPR) | TP/(TP+FN) | False negatives are costly |
| Classification | F1 / Fβ | harmonic mean of P & R | Imbalanced, single threshold, trade-off |
| Classification | Balanced accuracy | mean(TPR, TNR) | Imbalanced classes |
| Classification | MCC | corr. of true & pred labels, [−1,1] | Imbalanced, want one robust scalar |
| Classification | Cohen's κ | agreement corrected for chance | Inter-rater / vs-chance agreement |
| Ranking (binary) | ROC-AUC | area under TPR–FPR curve | Threshold-free, moderate imbalance |
| Ranking (binary) | PR-AUC / AP | area under precision–recall curve | Severe imbalance, rare positives |
| Probabilistic | Log loss | −Σ y log p | Need calibrated probabilities |
| Probabilistic | Brier score | mean (p−y)² | Calibration + accuracy, decomposable |
| Probabilistic | ECE | Σ \|acc_b − conf_b\|·n_b/N | Auditing calibration error |
| Regression | RMSE / MSE | √mean(e²) / mean(e²) | Penalize large errors |
| Regression | MAE | mean\|e\| | Robust to outliers |
| Regression | R² | 1 − SS_res/SS_tot | Variance explained |
| Regression | Pinball loss | quantile loss | Quantile / interval forecasts |
| Ranking (IR) | NDCG@k | DCG@k / IDCG@k | Graded relevance, position matters |
| Ranking (IR) | MAP / MRR | mean AP / mean 1/rank | Binary relevance, first-hit |
| Clustering (ext.) | ARI / NMI | chance-adjusted Rand / norm. MI | Have ground-truth labels |
| Clustering (ext.) | V-measure | harmonic mean homogeneity/completeness | Have ground-truth labels |
| Clustering (int.) | Silhouette | (b−a)/max(a,b) | No labels, assess cohesion/separation |

## Tools & libraries

| Tool | What it covers | URL |
|---|---|---|
| scikit-learn `metrics` | Classification, regression, ranking, clustering, calibration | https://scikit-learn.org/stable/modules/model_evaluation.html |
| TorchMetrics | GPU/distributed metrics for PyTorch & Lightning | https://lightning.ai/docs/torchmetrics/stable/ |
| Keras Metrics | Built-in metrics for TF/Keras training | https://keras.io/api/metrics/ |
| Hugging Face `evaluate` | Unified metric hub (incl. NLP/IR) | https://huggingface.co/docs/evaluate |
| TREC `pytrec_eval` | Standard IR metrics (NDCG, MAP, MRR) | https://github.com/cvangysel/pytrec_eval |
| netcal | Calibration metrics & methods (ECE, temperature scaling) | https://github.com/EFS-OpenSource/calibration-framework |
| statsmodels | Regression diagnostics, fit statistics | https://www.statsmodels.org/ |
| Yellowbrick | Visual diagnostics (ROC, PR, confusion, residuals) | https://www.scikit-yb.org/ |
| TensorFlow Model Analysis | Sliced, fairness-aware metrics at scale | https://www.tensorflow.org/tfx/guide/tfma |

## Learning resources

- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman) — free PDF: https://hastie.su.domains/ElemStatLearn/
- **An Introduction to Statistical Learning** (James, Witten, Hastie, Tibshirani) — free PDF: https://www.statlearning.com/
- **Probabilistic Machine Learning** (Kevin Murphy) — free PDFs: https://probml.github.io/pml-book/
- **scikit-learn User Guide — Metrics and scoring**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **Google ML Crash Course — Classification (ROC, AUC, accuracy, precision/recall)**: https://developers.google.com/machine-learning/crash-course/classification
- **StatQuest (Josh Starmer)** — intuitive videos on ROC/AUC, confusion matrices, R²: https://www.youtube.com/c/joshstarmer
- **Information Retrieval (Manning, Raghavan, Schütze)** — IR metrics chapter, free: https://nlp.stanford.edu/IR-book/
- **Forecasting: Principles and Practice** (Hyndman & Athanasopoulos) — forecast accuracy metrics, free: https://otexts.com/fpp3/

## Key papers

- Davis, J. & Goadrich, M. (2006). *The Relationship Between Precision-Recall and ROC Curves.* ICML. https://doi.org/10.1145/1143844.1143874
- Järvelin, K. & Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR Techniques* (DCG/NDCG). ACM TOIS. https://doi.org/10.1145/582415.582418
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.* ICML. https://doi.org/10.1145/1102351.1102430 — PDF: https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks* (temperature scaling, ECE). ICML / arXiv:1706.04599. https://arxiv.org/abs/1706.04599
- Rosenberg, A. & Hirschberg, J. (2007). *V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure.* EMNLP-CoNLL. https://aclanthology.org/D07-1043/
- Hubert, L. & Arabie, P. (1985). *Comparing Partitions* (Adjusted Rand Index). Journal of Classification. https://doi.org/10.1007/BF01908075

## Cross-references in AIForge

- [../../Machine_Learning](../../Machine_Learning) — supervised/unsupervised algorithms whose outputs these metrics score
- [../../Deep_Learning](../../Deep_Learning) — loss functions, calibration of neural nets
- [../../Optimization_Algorithms](../../Optimization_Algorithms) — metrics as objectives for hyperparameter search
- [../../Bayesian_and_Probabilistic_ML](../../Bayesian_and_Probabilistic_ML) — proper scoring rules and probabilistic calibration

## Sources

- scikit-learn — Metrics and scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
- Davis & Goadrich (2006), PR vs ROC: https://doi.org/10.1145/1143844.1143874
- Järvelin & Kekäläinen (2002), Cumulated Gain / NDCG: https://doi.org/10.1145/582415.582418
- Niculescu-Mizil & Caruana (2005), Predicting Good Probabilities: https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- Guo et al. (2017), On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
- Rosenberg & Hirschberg (2007), V-Measure: https://aclanthology.org/D07-1043/
- Google ML Crash Course — Classification: https://developers.google.com/machine-learning/crash-course/classification
