# Active Learning - Uncertainty Sampling

## Description

**Active Learning** is a Machine Learning technique that aims to optimize the data labeling process, allowing the model to iteratively select the most informative unlabeled examples to be annotated by an oracle (usually a human). The goal is to achieve high accuracy with a significantly smaller number of labeled samples, reducing costs and development time. **Uncertainty Sampling** is the most popular query strategy within Active Learning. Its unique value proposition lies in its ability to identify and prioritize samples that are close to the model's decision boundary, where classification is most ambiguous. By focusing on these "hard cases," the model maximizes the information gain with each new labeling, accelerating convergence and improving the model's robustness with less annotation effort.

## Statistics

The main performance indicator is **labeling efficiency**. Studies show that Active Learning, using Uncertainty Sampling, can achieve the same accuracy as a model trained with 70% of randomly labeled data, using only **20%** of the labeled samples. This represents a **71%** reduction in annotation effort. The key metric is the **Learning Curve**, which shows the model's accuracy as a function of the number of labeled samples, where a successful Active Learning curve rises much faster than the random sampling curve. Effectiveness is often measured by the **Area Under the Learning Curve (AL-AUC)**.

## Features

Uncertainty Sampling manifests through three main uncertainty measures:
*   **Least Confident**: Selects the sample for which the probability of the most likely class is the lowest. Uncertainty is computed as $U(x) = 1 - P(\hat{x}|x)$, where $\hat{x}$ is the most likely class.
*   **Margin Sampling**: Selects the sample with the smallest probability difference between the two most likely classes, $M(x) = P(\hat{x}_1|x) - P(\hat{x}_2|x)$. A small margin indicates that the model is undecided between the two best options.
*   **Entropy Sampling**: Selects the sample with the highest entropy, $H(x) = -\sum_{k} p_k \log(p_k)$. Entropy measures the randomness of the probability distribution across all classes, and is maximal when the distribution is uniform (highest uncertainty).

## Use Cases

*   **Natural Language Processing (NLP)**: Efficient labeling of large volumes of text for tasks such as sentiment classification, named entity recognition (NER), and word disambiguation. The model queries sentences where the classification is ambiguous.
*   **Computer Vision**: Annotation of images and videos for object detection and semantic segmentation. Active Learning is used to select images that contain boundary examples (for example, partially occluded objects or objects at unusual angles).
*   **Medical Diagnosis**: Reduction of the labeling cost of medical images (X-rays, MRIs) by specialists. The model prioritizes cases where the diagnosis is less clear, optimizing the physician's time.
*   **Fraud Detection**: Identification of financial transactions that lie on the "boundary" between fraud and non-fraud, allowing human analysts to review only the most ambiguous and informative cases.

## Integration

The integration of Active Learning with Uncertainty Sampling is facilitated by Python libraries such as **modAL** and **scikit-activeml**.

**Integration Example with modAL (Python):**

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

# 1. Data Preparation
X, y = load_iris(return_X_y=True)
# Initially, we label only 10 samples
n_initial = 10
initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
X_initial, y_initial = X[initial_idx], y[initial_idx]
X_pool = np.delete(X, initial_idx, axis=0)
y_pool = np.delete(y, initial_idx, axis=0)

# 2. Initialize the Active Learner with Uncertainty Sampling
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=uncertainty_sampling, # Uncertainty Sampling strategy
    X_training=X_initial, y_training=y_initial
)

# 3. Active Learning Loop
n_queries = 20
for idx in range(n_queries):
    # The model queries the most uncertain sample
    query_idx, query_instance = learner.query(X_pool)

    # Labeling simulation (the oracle provides the label)
    X_new, y_new = X_pool[query_idx], y_pool[query_idx]

    # The model learns from the new labeled sample
    learner.teach(X_new, y_new)

    # Remove the labeled sample from the pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    print(f"Iteration {idx+1}: Current accuracy = {learner.score(X, y):.4f}")
```

## URL

https://modal-python.readthedocs.io/en/latest/content/query_strategies/uncertainty_sampling.html