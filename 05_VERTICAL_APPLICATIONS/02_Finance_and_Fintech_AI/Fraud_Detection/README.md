# Fraud Detection

This directory covers AI/ML systems for detecting, preventing, investigating, and monitoring fraud across payments, banking, e-commerce, insurance, identity, account takeover, bot activity, and financial crime.

## Scope

- Supervised, semi-supervised, unsupervised, graph, sequence, and streaming fraud models.
- Transaction scoring, device fingerprinting, behavioral signals, identity signals, sanctions/AML adjacency, and case-management queues.
- Class imbalance, delayed labels, concept drift, adversarial behavior, leakage, and false-positive cost.

## Reference Links

- IEEE-CIS Fraud Detection competition: https://www.kaggle.com/competitions/ieee-fraud-detection
- IEEE-CIS data description: https://www.kaggle.com/competitions/ieee-fraud-detection/data
- Amazon Fraud Dataset Benchmark: https://github.com/amazon-science/fraud-dataset-benchmark
- NVIDIA fraud detection Kaggle solution overview: https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
- FINRA AI key challenges: https://www.finra.org/rules-guidance/key-topics/fintech/report/artificial-intelligence-in-the-securities-industry/key-challenges

## Documentation Standard

Record label source, fraud typology, delay window, entity keys, leakage risks, threshold policy, investigation workflow, drift checks, and human-review process.

## Routing Rules

- Put credit-decisioning models in `../Credit_Scoring/`.
- Put securities and market-surveillance AI in `../Risk_Management/` or `../Algorithmic_Trading/`.
- Put generic anomaly-detection theory in the AI fundamentals section.
