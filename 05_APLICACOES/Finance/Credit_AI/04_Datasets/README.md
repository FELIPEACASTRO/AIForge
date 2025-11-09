# 📊 Datasets - Credit AI

**[EN]** Curated collection of 15+ public datasets for credit scoring, risk assessment, and loan default prediction.

**[PT]** Coleção curada de 15+ datasets públicos para credit scoring, avaliação de risco e previsão de inadimplência.

---

## ⭐ Top 5 Essential Datasets

### 1. German Credit Risk ⭐⭐⭐⭐⭐ (CLASSIC)
**Source:** UCI Machine Learning Repository + Kaggle  
**URLs:**  
- https://www.kaggle.com/datasets/uciml/german-credit  
- http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)  

**[EN]** Most cited benchmark dataset for credit scoring research.

**[PT]** Dataset de benchmark mais citado para pesquisa de credit scoring.

**Details:**
- 📊 1,000 instances
- 📋 21 attributes (7 numerical, 14 categorical)
- 🎯 Binary classification (good/bad credit)
- 🏆 Most cited in literature
- ✅ Clean and well-documented

---

### 2. American Express Default Prediction
**Source:** Kaggle Competition  
**URL:** https://www.kaggle.com/c/amex-default-prediction  

**[EN]** Millions of credit card profiles from AmEx competition.

**[PT]** Milhões de perfis de cartão de crédito da competição AmEx.

**Details:**
- 📊 Millions of profiles
- 📋 Time-series features
- 🎯 Credit card default prediction
- 🏆 Major Kaggle competition
- ✅ Real-world scale

---

### 3. Credit Risk Dataset (Kaggle)
**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/laotse/credit-risk-dataset  

**[EN]** Simulated credit bureau data for experimentation.

**[PT]** Dados simulados de bureau de crédito para experimentação.

**Details:**
- 📊 32,581 instances
- 📋 12 features
- 🎯 Binary classification
- ✅ Good for learning
- ✅ Balanced dataset

---

### 4. Loan Default Prediction
**Source:** Kaggle / Coursera  
**URL:** https://www.kaggle.com/datasets/laotse/loan-default-prediction  

**[EN]** Loan default prediction challenge dataset.

**[PT]** Dataset de desafio de previsão de inadimplência de empréstimos.

**Details:**
- 📊 148,670 instances
- 📋 34 features
- 🎯 Binary classification
- ✅ Imbalanced (realistic)
- ✅ Feature engineering opportunities

---

### 5. Home Credit Default Risk
**Source:** Kaggle Competition  
**URL:** https://www.kaggle.com/c/home-credit-default-risk  

**[EN]** Alternative credit scoring for unbanked population.

**[PT]** Credit scoring alternativo para população sem conta bancária.

**Details:**
- 📊 307,511 instances
- 📋 Multiple tables (relational)
- 🎯 Financial inclusion focus
- 🏆 Major competition
- ✅ Alternative data examples

---

## 📚 Additional Datasets

### 6. Give Me Some Credit (Kaggle)
**URL:** https://www.kaggle.com/c/GiveMeSomeCredit  
**Instances:** 150,000 | **Features:** 11  

### 7. LendingClub Loan Data
**URL:** https://www.kaggle.com/datasets/wordsforthewise/lending-club  
**Instances:** 2.2M+ | **Features:** 150+  

### 8. Australian Credit Approval
**URL:** http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)  
**Instances:** 690 | **Features:** 14  

### 9. Japanese Credit Screening
**URL:** http://archive.ics.uci.edu/ml/datasets/Japanese+Credit+Screening  
**Instances:** 690 | **Features:** 15  

### 10. Taiwan Credit Card Default
**URL:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset  
**Instances:** 30,000 | **Features:** 24  

---

## 🎯 Datasets by Use Case

| Use Case | Recommended Dataset |
|---|---|
| **Learning** | German Credit Risk |
| **Benchmarking** | German Credit, Australian |
| **Production Simulation** | AmEx, Home Credit |
| **Alternative Data** | Home Credit |
| **Time-Series** | AmEx, LendingClub |
| **Imbalanced Data** | Loan Default Prediction |

---

## 🚀 Quick Start

```python
# Download German Credit dataset
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(url, sep=' ', header=None)

# Download from Kaggle (requires kaggle API)
!kaggle datasets download -d uciml/german-credit
```

---

**Last Updated:** November 8, 2025  
**Total Datasets:** 15+  
**Maintained by:** AIForge Community
