# PersonalCareNet (CNN with Attention Mechanism and XAI)

## Description

**PersonalCareNet** is an explainable artificial intelligence (XAI) model based on Convolutional Neural Networks (CNNs) and an attention mechanism, designed for personalized health monitoring and disease risk prediction. Published in 2025, it stands out for integrating the attention mechanism to increase interpretability, allowing the model to dynamically focus on the most relevant clinical features. Interpretability is further enhanced by incorporating the **SHAP (Shapley Additive exPlanations)** framework, which provides global and patient-specific explanations, building trust for clinical decision-making. The model was trained and evaluated using a subset of the **MIMIC-III** database, focused on ICU patients.

## Statistics

- **Maximum Accuracy:** 97.86%
- **AUC (Area Under the Curve):** 98.3%
- **Comparison:** Outperforms state-of-the-art models such as TabNet, AutoGluon Tabular, and NODE in accuracy.
- **Dataset:** Subset of MIMIC-III (ICU patients).
- **Citations:** Cited in 3 articles (as of the research date).
- **Publication Year:** 2025.

## Features

- **Optional Attention Mechanism (CHARMS):** Allows the model to assign dynamic weights to clinical features, increasing interpretability by highlighting the most influential factors in the prediction.
- **Multi-Level Interpretability (SHAP):** Provides explanations at the local level (for each individual prediction) and global level (overall feature importance).
- **Hybrid Architecture:** Combines dense layers with regularization components for structured clinical data.
- **Overfitting Prevention:** Uses Dropout layers and L2 regularization to ensure model generalization.

## Use Cases

- **Personalized Health Monitoring:** Continuous assessment of patients' health status.
- **Disease Risk Prediction:** Prediction of clinical conditions or health risks in ICU patients based on structured clinical data.
- **Clinical Decision Support:** Providing physicians not only with a prediction, but also the rationale (which clinical features were most important) behind the decision, promoting trust in the AI system.

## Integration

Although the direct source code for PersonalCareNet was not found, the methodology suggests a standard Deep Learning implementation with the addition of an attention module and post-hoc integration of the SHAP framework.

**Conceptual Integration Example (Python/PyTorch or TensorFlow):**

1.  **Model Definition:** Implement a CNN or dense network with an attention layer (for example, *Self-Attention* or *Additive Attention*) applied to the outputs of the intermediate layers.
2.  **Training:** Train the model with EHR data (such as MIMIC-III) using standard optimizers (Adam/SGD) and a cross-entropy loss function.
3.  **Interpretability (SHAP):** After training, use the `shap` library to compute Shapley values for each prediction.

```python
# Example of using SHAP for interpretability
import shap
# Assuming 'model' is the trained PersonalCareNet model and 'data' is the input data
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(data)

# Visualizing the explanation for a single instance (patient)
shap.force_plot(explainer.expected_value[0], shap_values[0][0], data.iloc[0])
```

## URL

https://pmc.ncbi.nlm.nih.gov/articles/PMC12397250/