# Machine Learning Frameworks And Research Source Atlas - 2026-07-07

This atlas enriches the machine-learning directory with source-first routes for algorithms, libraries, papers, benchmarks, and reproducible implementation references. Use it as a triage layer before adding new files deeper in `Classical_ML_Algorithms/`, `Deep_Learning/`, `Optimization_Algorithms/`, `AutoML/`, `Model_Evaluation/`, or domain-specific application directories.

## Core ML Frameworks

| Source | Evidence to capture | Local routing |
|---|---|---|
| [scikit-learn documentation](https://scikit-learn.org/stable/) | Estimators, preprocessing, model selection, pipelines, metrics, and examples. | `Classical_ML_Algorithms/`, `Feature_Engineering/`, `Model_Evaluation/` |
| [PyTorch documentation](https://pytorch.org/docs/stable/index.html) | Tensor APIs, autograd, distributed training, compilation, data loading, and model examples. | `Deep_Learning/`, `Computer_Vision/`, `Natural_Language_Processing/` |
| [TensorFlow documentation](https://www.tensorflow.org/) | Keras workflows, TensorFlow serving, TensorFlow Lite, model optimization, and tutorials. | `Deep_Learning/`, `Edge_and_On_Device_Deployment/`, production notes. |
| [Keras documentation](https://keras.io/) | Multi-backend deep-learning APIs, examples, callbacks, model saving, and transfer learning. | `Deep_Learning/architectures/`, `Transfer_Learning/` |
| [JAX documentation](https://docs.jax.dev/en/latest/) | Differentiable programming, transformations, just-in-time compilation, and accelerator workflows. | `Optimization_Algorithms/`, `Deep_Learning/`, research prototypes. |
| [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/) | Gradient boosting, distributed training, categorical support, and GPU acceleration. | `Classical_ML_Algorithms/Gradient_Boosting/` |
| [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/) | Gradient boosting, ranking, distributed learning, and parameter references. | `Classical_ML_Algorithms/Gradient_Boosting/` |
| [CatBoost documentation](https://catboost.ai/docs/) | Categorical-feature handling, ranking, Python/R/CLI workflows, and model analysis. | `Classical_ML_Algorithms/Gradient_Boosting/` |
| [Ray Tune documentation](https://docs.ray.io/en/latest/tune/index.html) | Distributed hyperparameter tuning and integrations with common ML frameworks. | `AutoML/Hyperparameter_Optimization/` |
| [Optuna documentation](https://optuna.readthedocs.io/en/stable/) | Bayesian and adaptive hyperparameter search, pruning, and experiment studies. | `AutoML/Hyperparameter_Optimization/` |

## Research And Paper Sources

| Source | Evidence to capture | Local routing |
|---|---|---|
| [arXiv cs.LG recent](https://arxiv.org/list/cs.LG/recent) | New machine-learning papers, preprint ids, author versions, and code links when present. | `Research_Platforms_and_Preprints/`, method-specific folders. |
| [Journal of Machine Learning Research](https://www.jmlr.org/) | Peer-reviewed ML papers, special topics, and survey articles. | Theory and algorithms folders. |
| [NeurIPS proceedings](https://proceedings.neurips.cc/) | Conference papers, datasets/benchmarks track entries, and workshop references. | New algorithm and evaluation evidence. |
| [Proceedings of Machine Learning Research](https://proceedings.mlr.press/) | ICML, AISTATS, COLT, UAI, and other ML proceedings. | Statistical learning, optimization, probabilistic ML. |
| [ICLR on OpenReview](https://openreview.net/group?id=ICLR.cc) | Open peer review, accepted papers, reviews, and author responses. | Modern deep learning and representation learning. |
| [ACL Anthology](https://aclanthology.org/) | NLP papers, datasets, shared tasks, and benchmark papers. | `Natural_Language_Processing/`, `Prompt_Engineering/` |
| [CVF Open Access](https://openaccess.thecvf.com/) | CVPR, ICCV, ECCV, WACV papers and supplementary material. | `Computer_Vision/`, `Vision_Transformers/`, `Vision_Language_Models/` |
| [Papers with Code](https://paperswithcode.com/) | Paper-code links, task pages, datasets, leaderboards, and method tags. | Benchmark evidence and reproducibility routing. |
| [OpenML](https://www.openml.org/) | Datasets, tasks, runs, flows, and benchmark suites. | `Datasets/Machine_Learning/`, `Model_Evaluation/` |

## Evidence Rules

| Rule | Requirement |
|---|---|
| Separate method from implementation | Record the paper/source for the method and the library/repo used to implement it. |
| Preserve benchmark context | Dataset split, metric, task, leaderboard date, and baseline must travel with any score. |
| Prefer primary docs | Use official framework docs for APIs and behavior; use blog posts only as secondary interpretation. |
| Capture reproducibility state | Code URL, license, dependency versions, hardware assumptions, and data availability must be explicit. |
| Route by technical claim | Algorithms go to theory folders, libraries to tooling folders, datasets to dataset folders, and applications to `05_VERTICAL_APPLICATIONS/`. |

## Local Expansion Queue

| Priority | Target | Source seed |
|---|---|---|
| 1 | Gradient boosting guide with XGBoost, LightGBM, CatBoost comparison. | Official docs plus OpenML/Papers with Code tasks. |
| 2 | Hyperparameter optimization guide. | Ray Tune, Optuna, scikit-learn model selection. |
| 3 | Deep-learning framework comparison. | PyTorch, TensorFlow, Keras, JAX primary docs. |
| 4 | Reproducible ML paper intake checklist. | arXiv, OpenReview, JMLR, NeurIPS, PMLR, Papers with Code. |
