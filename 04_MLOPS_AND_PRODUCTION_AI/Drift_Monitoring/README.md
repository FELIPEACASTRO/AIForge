# Drift Monitoring

> Drift monitoring is the practice of continuously detecting and quantifying changes in production data distributions and model behavior — covariate (data) drift, concept drift, label/prior shift — so silent model degradation is caught before it harms decisions.

## Why it matters

Models are trained on a snapshot of the world, but the world keeps moving: user behavior, upstream pipelines, pricing, and seasonality all shift inputs away from the training distribution. When the input distribution or the input→label relationship changes, accuracy can decay silently, often without any error or alert. Drift monitoring provides early-warning signals (statistical distances, two-sample tests, performance estimation) that trigger investigation, retraining, or rollback. It is the bridge between one-off model validation and durable, trustworthy production ML.

## Taxonomy

| Drift type | What changes | Formal view | Typical signal |
|---|---|---|---|
| **Covariate / data drift** | Input feature distribution `P(X)` | `P_t(X) ≠ P_train(X)`, `P(y\|X)` stable | PSI, KS, Wasserstein per feature |
| **Concept drift** | Relationship `P(y\|X)` | mapping input→target shifts | error/accuracy rise (DDM, ADWIN) |
| **Label / prior shift** | Target distribution `P(y)` | class balance changes | chi-square on label counts |
| **Prediction drift** | Distribution of model outputs `P(ŷ)` | proxy when labels are delayed | KS/PSI on score distribution |
| **Upstream/data-quality drift** | Schema, nulls, ranges, units | non-statistical breakage | range/null/cardinality checks |

By detection mode: **supervised** (needs ground-truth labels — error-rate monitors like DDM/EDDM/Page-Hinkley) vs **unsupervised** (label-free — distribution distances, density-ratio, PCA reconstruction, performance estimation). By cadence: **batch** (compare a reference window to an analysis window) vs **streaming/online** (update detector one sample at a time). By temporal pattern: sudden, gradual, incremental, and recurring drift.

## Key tools & libraries

| Tool | Focus | Link |
|---|---|---|
| Evidently AI | Drift + ML/LLM monitoring, reports & tests (open source) | https://github.com/evidentlyai/evidently |
| NannyML | Performance estimation without labels (CBPE/DLE) + PCA multivariate drift | https://github.com/NannyML/nannyml |
| Alibi Detect (Seldon) | Drift, outlier & adversarial detection (MMD, KS, LSDD, classifier drift) | https://github.com/SeldonIO/alibi-detect |
| deepchecks | Data + model validation & drift test suites | https://github.com/deepchecks/deepchecks |
| river | Online learning with built-in concept-drift detectors (ADWIN, DDM, PH) | https://github.com/online-ml/river |
| Frouros | Drift-only library (31+ data & concept drift methods) | https://github.com/IFCA-Advanced-Computing/frouros |
| whylogs / WhyLabs | Data logging & profile-based drift monitoring | https://github.com/whylabs/whylogs |
| scikit-multiflow | Stream learning + drift detectors (legacy, merged into river) | https://github.com/scikit-multiflow/scikit-multiflow |
| Eurybia (MAIF) | Data & model drift validation/reports over time | https://github.com/MAIF/eurybia |
| TorchDrift | Drift detection for PyTorch models (MMD, kernel tests) | https://github.com/TorchDrift/TorchDrift |

## Key methods

| Method | Type | Use when | Reference |
|---|---|---|---|
| **PSI** (Population Stability Index) | Distance | Binned numeric/categorical features; >0.2–0.25 = significant | https://aigents.co/learn/Statistical-Techniques-for-Drift-Detection |
| **Kolmogorov–Smirnov (KS)** | Two-sample test | Continuous univariate features | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html |
| **Chi-square** | Two-sample test | Categorical / label distributions | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html |
| **KL / JS divergence** | Distance | Compare reference vs current densities (JS bounded, symmetric) | https://link.springer.com/article/10.1007/s42488-024-00119-y |
| **Wasserstein / Earth Mover** | Distance | Ordered numeric features, magnitude-aware | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html |
| **MMD** (Maximum Mean Discrepancy) | Kernel two-sample | Multivariate / high-dim, embeddings | https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html |
| **DDM / EDDM** | Supervised, streaming | Error-rate monitoring with labels | https://riverml.xyz/latest/api/drift/binary/DDM/ |
| **ADWIN** | Adaptive window | Streaming, false-positive/negative bounds | https://epubs.siam.org/doi/10.1137/1.9781611972771.42 |
| **Page-Hinkley** | Sequential CUSUM | Detect mean shifts in a metric stream | https://riverml.xyz/latest/api/drift/PageHinkley/ |
| **PCA reconstruction error** | Unsupervised multivariate | Label-free multivariate drift (NannyML) | https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html |
| **Domain-classifier drift** | Discriminative | Can a classifier separate reference vs current? | https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html |

## Benchmarks & datasets

| Resource | Use | Link |
|---|---|---|
| Electricity (Elec2) | Canonical real-world concept-drift stream | https://www.openml.org/d/151 |
| Airlines | Large concept-drift classification stream | https://www.openml.org/d/1169 |
| USP DS Repository | Curated synthetic + real drifting streams | https://sites.google.com/view/uspdsrepository |
| river `datasets` / synth generators | SEA, Agrawal, Hyperplane, RBF drift generators | https://riverml.xyz/latest/api/datasets/synth/ |
| Failing Loudly benchmark | Empirical dataset-shift detection study & code | https://github.com/steverab/failing-loudly |

## Key papers

| Paper | Year | Link |
|---|---|---|
| Gama et al. — A Survey on Concept Drift Adaptation (ACM CSUR) | 2014 | https://doi.org/10.1145/2523813 |
| Bifet & Gavaldà — Learning from Time-Changing Data with Adaptive Windowing (ADWIN, SDM) | 2007 | https://epubs.siam.org/doi/10.1137/1.9781611972771.42 |
| Lu et al. — Learning under Concept Drift: A Review (IEEE TKDE) | 2020 | https://arxiv.org/abs/2004.05785 |
| Rabanser et al. — Failing Loudly: Methods for Detecting Dataset Shift (NeurIPS) | 2019 | https://arxiv.org/abs/1810.11953 |
| Cerqueira et al. — STUDD: Student-Teacher Unsupervised Concept Drift Detection | 2021 | https://arxiv.org/abs/2103.00903 |
| Céspedes-Sisniega & López — Frouros: open-source drift detection library | 2022 | https://arxiv.org/abs/2208.06868 |
| Gözüaçık et al. — Concept Drift & Covariate Shift Detection with Lagged Labels | 2020 | https://arxiv.org/abs/2012.04759 |
| Open-Source Drift Detection Tools in Action: Insights from Two Use Cases | 2024 | https://arxiv.org/abs/2404.18673 |

## Cross-references in AIForge

- [AI Observability](../AI_Observability/) — dashboards, tracing, and alerting that consume drift signals.
- [CI/CD for ML](../CI_CD_for_ML/) — wiring drift checks and retraining triggers into pipelines.
- [Model Registry Solutions](../Model_Registry_Solutions/) — versioning datasets/models for safe rollback after drift.
- [Guardrails and Safety](../Guardrails_and_Safety/) — runtime safety controls complementary to drift alerts.

## Sources

- https://github.com/evidentlyai/evidently
- https://github.com/NannyML/nannyml
- https://github.com/SeldonIO/alibi-detect
- https://github.com/IFCA-Advanced-Computing/frouros
- https://riverml.xyz/latest/api/drift/
- https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html
- https://doi.org/10.1145/2523813
- https://arxiv.org/abs/2004.05785
- https://arxiv.org/abs/1810.11953
- https://arxiv.org/abs/2208.06868
- https://epubs.siam.org/doi/10.1137/1.9781611972771.42
- https://link.springer.com/article/10.1007/s42488-024-00119-y
- https://aigents.co/learn/Statistical-Techniques-for-Drift-Detection
- https://arxiv.org/abs/2404.18673

_Expanded from a verified high-value gap sweep. Contributions welcome (see CONTRIBUTING.md)._
