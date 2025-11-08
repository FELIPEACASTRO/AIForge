# 🐍 Python Libraries - Credit AI

**[EN]** Curated collection of 15+ Python libraries for credit scoring, risk assessment, and model development.

**[PT]** Coleção curada de 15+ bibliotecas Python para credit scoring, avaliação de risco e desenvolvimento de modelos.

---

## ⭐ Top 5 Essential Libraries

### 1. ScoringPy ⭐⭐⭐⭐⭐
**PyPI:** https://pypi.org/project/ScoringPy/  
**Install:** `pip install ScoringPy`  

**[EN]** Complete credit scorecard library with WoE, IV, binning, and deployment.

**[PT]** Biblioteca completa de scorecard de crédito com WoE, IV, binning e deployment.

**Features:**
- ✅ WoE (Weight of Evidence) calculation
- ✅ IV (Information Value) analysis
- ✅ Optimal binning algorithms
- ✅ Scorecard development
- ✅ Model deployment tools

---

### 2. XGBoost ⭐⭐⭐⭐⭐
**Docs:** https://xgboost.readthedocs.io/  
**Install:** `pip install xgboost`  

**[EN]** Best performing ML algorithm (75% of winning solutions).

**[PT]** Algoritmo de ML com melhor desempenho (75% das soluções vencedoras).

**Features:**
- ✅ Gradient boosting
- ✅ 75%+ of Kaggle winners
- ✅ GPU acceleration
- ✅ Built-in regularization
- ✅ Feature importance

---

### 3. LightGBM ⭐⭐⭐⭐⭐
**Docs:** https://lightgbm.readthedocs.io/  
**Install:** `pip install lightgbm`  

**[EN]** Fastest gradient boosting, often outperforms XGBoost.

**[PT]** Gradient boosting mais rápido, frequentemente supera XGBoost.

**Features:**
- ✅ Faster than XGBoost
- ✅ Lower memory usage
- ✅ Categorical features support
- ✅ GPU acceleration
- ✅ High accuracy

---

### 4. pyratings ⭐⭐⭐⭐⭐ (HSBC)
**Docs:** https://hsbc.github.io/pyratings/  
**Install:** `pip install pyratings`  

**[EN]** Professional credit ratings library by HSBC.

**[PT]** Biblioteca profissional de ratings de crédito da HSBC.

**Features:**
- ✅ Credit rating calculations
- ✅ HSBC-developed
- ✅ Production-grade
- ✅ Well-documented
- ✅ Industry standard

---

### 5. SHAP ⭐⭐⭐⭐⭐ (EXPLAINABILITY)
**Docs:** https://shap.readthedocs.io/  
**Install:** `pip install shap`  

**[EN]** Explainable AI library (mandatory for credit models).

**[PT]** Biblioteca de IA explicável (obrigatória para modelos de crédito).

**Features:**
- ✅ Model explainability
- ✅ SHAP values
- ✅ Feature importance
- ✅ Regulatory compliance
- ✅ Visualization tools

---

## 📚 Additional Libraries

### 6. scikit-learn
**Install:** `pip install scikit-learn`  
**Use:** Baseline models (Logistic Regression, Random Forest)

### 7. imbalanced-learn
**Install:** `pip install imbalanced-learn`  
**Use:** SMOTE, undersampling, oversampling

### 8. category_encoders
**Install:** `pip install category-encoders`  
**Use:** WoE encoding, target encoding

### 9. optbinning
**Install:** `pip install optbinning`  
**Use:** Optimal binning for scorecards

### 10. LIME
**Install:** `pip install lime`  
**Use:** Local explainability

---

## 🚀 Quick Start

```python
# Install essential libraries
!pip install xgboost lightgbm scoringpy shap pandas scikit-learn

# Basic credit scoring pipeline
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap

# Load data
df = pd.read_csv('credit_data.csv')
X = df.drop('default', axis=1)
y = df['default']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train XGBoost
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f'AUC: {auc:.4f}')

# Explain with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

**Last Updated:** November 8, 2025  
**Total Libraries:** 15+  
**Maintained by:** AIForge Community
