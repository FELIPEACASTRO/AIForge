# SHAP (SHapley Additive exPlanations)

## Description

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explaining the output of any machine learning model. Its unique value proposition lies in the unification of model interpretability methods, connecting optimal credit allocation with local explanations. SHAP assigns each feature an importance value for a specific prediction, ensuring consistency and local accuracy. It solves the black-box problem in ML by providing transparency and trust in the model's decisions. The framework is model-agnostic, meaning it can be applied to linear models, tree-based models, and deep neural networks. The SHAP value of a feature is the average change in the model's prediction when that feature is included in the feature set, compared to when it is excluded.

## Statistics

- **Citations of the Original Paper:** The seminal paper "A Unified Approach to Interpreting Model Predictions" by Lundberg and Lee (2017) has more than **44,000 citations** (source: arXiv, ACM, NeurIPS), indicating its vast academic influence.
- **GitHub Stars:** The official `shap/shap` repository on GitHub has approximately **24,679 stars**, reflecting strong adoption and support from the open-source community.
- **PyPI Downloads:** The `shap` package on PyPI records a high volume of downloads, with the all-time total surpassing **100 million** (estimate based on third-party data, such as PyPI Stats), which demonstrates its popularity and widespread use in production and research.

## Features

- **Model-Agnostic:** Can be applied to any machine learning model (linear, tree-based, neural networks).
- **Consistency and Local Accuracy:** Ensures that explanations are locally accurate and consistent with game theory.
- **Multiple Explainers:** Includes KernelSHAP (for any model), TreeSHAP (optimized for tree-based models such as XGBoost, LightGBM, CatBoost), and DeepExplainer/GradientExplainer (for deep learning models such as TensorFlow and PyTorch).
- **Comprehensive Visualizations:** Offers summary plots, dependence plots, force plots, and interaction plots for global and local insights.
- **Unification:** Unifies earlier methods such as LIME, DeepLIFT, and Layer-wise Relevance Propagation (LRP) under a single theoretical framework.

## Use Cases

- **Model Diagnostics:** Identify and correct biases in ML models, such as gender or racial discrimination, by analyzing the contribution of sensitive features.
- **Regulatory Approval and Compliance:** In sectors such as finance and healthcare, SHAP is used to explain credit decisions or medical diagnoses, meeting regulatory transparency requirements (e.g., GDPR, AI regulations).
- **Business Optimization:** In marketing, explaining why a specific customer received an offer or why they are likely to churn, enabling targeted and more effective interventions.
- **Feature Engineering:** Understanding which features are most important globally to the model, guiding the process of selecting and creating new features.
- **Cybersecurity Analysis:** Explaining the classification of network traffic as malicious or benign, identifying the network features that contributed most to the decision.

## Integration

SHAP integration is typically done through the Python library `shap`. The integration method varies depending on the type of model (agnostic or tree-based/deep learning).

**Integration Example with Scikit-learn (KernelSHAP):**
```python
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 1. Load data and train model
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# 2. Initialize the SHAP explainer (KernelSHAP for model-agnostic models)
# KernelSHAP requires a background dataset
explainer = shap.KernelExplainer(model.predict_proba, X_train)

# 3. Compute the SHAP values for a test sample
shap_values = explainer.shap_values(X_test.iloc[0,:])

# 4. Visualize the explanation
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[0,:])
```

**Integration Example with Tree-Based Models (TreeSHAP):**
```python
import shap
import xgboost
from sklearn.datasets import load_boston

# 1. Load data and train XGBoost model
X, y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# 2. Initialize the SHAP explainer (TreeSHAP for tree-based models)
explainer = shap.TreeExplainer(model)

# 3. Compute the SHAP values
shap_values = explainer.shap_values(X)

# 4. Visualize the summary plot
shap.summary_plot(shap_values, X)
```

## URL

https://shap.readthedocs.io/ | https://github.com/shap/shap