# 🤖 Time Series Foundation Models / Modelos de Fundação para Séries Temporais

---

## 🌍 English

Pretrained foundation models for time series forecasting with zero-shot and few-shot learning capabilities.

---

## 🏆 Top Foundation Models (2025)

### 1. ⭐⭐⭐⭐⭐ **TiRex** (NXAI / IT University Austria)
**ArXiv 2025** - [2505.23719](https://arxiv.org/abs/2505.23719)

**The Game Changer:**
- **35M parameters** beats 500M models (Google TimesFM)
- **Zero-shot forecasting** without any task-specific training
- **xLSTM architecture** (Enhanced LSTM with memory improvements)
- **Top of official leaderboards**
- **14.3x smaller** than TimesFM, but superior performance

**Technical Details:**
- **Architecture:** xLSTM (Extended Long Short-Term Memory)
- **Parameters:** 35M
- **Training:** In-context learning
- **Capabilities:** Zero-shot, few-shot
- **Inference:** Memory-efficient, fast

**Performance:**
- **Beats Google TimesFM** (500M params)
- **Beats Amazon Chronos** (200M params)
- **Top rankings** on Monash benchmark
- **10 citations** (ArXiv 2025)

**Organizations:**
- NXAI (AI Research Lab)
- IT University Austria

**Links:**
- 📄 [ArXiv Paper](https://arxiv.org/abs/2505.23719)
- 💻 [GitHub](https://github.com/nxai/tirex)
- 🤗 [HuggingFace Model](https://huggingface.co/nxai/tirex)

---

### 2. ⭐⭐⭐⭐⭐ **Chronos-2** (Amazon)
**October 2025** - [HuggingFace](https://huggingface.co/amazon/chronos-2)

**Latest Generation:**
- **Zero-shot forecasting** for any time series
- **Univariate** support
- **Multivariate** support (NEW in Chronos-2)
- **Covariate-informed** forecasting (NEW)

**Technical Details:**
- **Architecture:** T5-based transformer
- **Parameters:** 100M (Chronos-2)
- **Training:** Large-scale time series corpus
- **Capabilities:** Zero-shot, multivariate

**Versions:**
- **Chronos-2** (100M) - Latest, multivariate
- **Chronos-T5-Base** (200M) - 397K downloads
- **Chronos-T5-Small** (46.2M) - 450K downloads, 134 likes

**Performance:**
- **State-of-the-art** on multiple benchmarks
- **Multivariate** forecasting (new capability)
- **Covariate** support (external variables)

**Organizations:**
- Amazon Science

**Links:**
- 🤗 [HuggingFace: Chronos-2](https://huggingface.co/amazon/chronos-2)
- 🤗 [HuggingFace: Chronos-T5-Small](https://huggingface.co/amazon/chronos-t5-small)
- 💻 [GitHub](https://github.com/amazon-science/chronos-forecasting)
- 📄 [Amazon Science Blog](https://www.amazon.science/blog/amazon-announces-chronos-a-foundation-model-for-time-series-forecasting)

---

### 3. ⭐⭐⭐⭐⭐ **Chronos-Bolt** (AutoGluon)
**November 2025** - [HuggingFace](https://huggingface.co/autogluon/chronos-bolt-base)

**Most Popular:**
- **4.37M downloads** (Chronos-Bolt-Base) - HIGHEST!
- **Faster inference** than standard Chronos
- **Multiple sizes** (tiny, base)
- **Updated 9 days ago** (actively maintained)

**Technical Details:**
- **Architecture:** Optimized Chronos variant
- **Parameters:** 
  - Chronos-Bolt-Base: 200M
  - Chronos-Bolt-Tiny: 8.65M
- **Training:** Same as Chronos
- **Optimization:** Faster inference

**Performance:**
- **4.37M downloads** (base) - Most popular!
- **597K downloads** (tiny)
- **32 likes** (base)
- **Competitive accuracy** with faster speed

**Organizations:**
- AutoGluon (AWS)

**Links:**
- 🤗 [HuggingFace: Chronos-Bolt-Base](https://huggingface.co/autogluon/chronos-bolt-base)
- 🤗 [HuggingFace: Chronos-Bolt-Tiny](https://huggingface.co/autogluon/chronos-bolt-tiny)
- 💻 [GitHub: AutoGluon](https://github.com/autogluon/autogluon)

---

### 4. ⭐⭐⭐⭐⭐ **TimesFM-2.5** (Google Research)
**October 2025** - [HuggingFace](https://huggingface.co/google/timesfm-2.5-200m-pytorch)

**Foundation Model:**
- **200M parameters**
- **Pretrained foundation model**
- **Few-shot learning** capabilities
- **Zero-shot forecasting**

**Technical Details:**
- **Architecture:** Transformer-based
- **Parameters:** 200M
- **Training:** Massive time series corpus
- **Capabilities:** Zero-shot, few-shot

**Performance:**
- **State-of-the-art** on TimesFM benchmark
- **Few-shot** learning (adapts with minimal data)
- **Latest version** (2.5)

**Organizations:**
- Google Research

**Links:**
- 🤗 [HuggingFace Model](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- 📄 [Google Research Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- 📄 [Paper: Few-shot learners](https://arxiv.org/abs/2310.10688)

---

### 5. ⭐⭐⭐⭐⭐ **IBM Granite TTM** (IBM Research)
**February 2025** - [HuggingFace](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)

**Ultra Compact:**
- **800K parameters** (ULTRA COMPACT!)
- **313 likes** (MOST LIKED!)
- **331K downloads** (TTM-R1)
- **Tiny Time Mixer** architecture

**Technical Details:**
- **Architecture:** Tiny Time Mixer (TTM)
- **Parameters:** 
  - TTM-R1: 805K
  - TTM-R2: 800K
- **Training:** Multiple time series datasets
- **Optimization:** Ultra compact, efficient

**Performance:**
- **313 likes** (TTM-R1) - Most liked!
- **331K downloads** (TTM-R1)
- **131 likes** (TTM-R2)
- **Competitive** with much larger models

**Organizations:**
- IBM Research

**Links:**
- 🤗 [HuggingFace: TTM-R1](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)
- 🤗 [HuggingFace: TTM-R2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
- 💻 [GitHub: IBM Granite](https://github.com/ibm-granite)

---

### 6. ⭐⭐⭐⭐ **MOMENT-1-large** (AutonLab)
**2024** - [HuggingFace](https://huggingface.co/AutonLab/MOMENT-1-large)

**Foundation Model:**
- **Large-scale** foundation model
- **Time series forecasting**
- **AutonLab research**

**Technical Details:**
- **Architecture:** Transformer-based
- **Parameters:** Large (not specified)
- **Training:** Multiple domains
- **Capabilities:** Zero-shot

**Organizations:**
- AutonLab (Carnegie Mellon University)

**Links:**
- 🤗 [HuggingFace Model](https://huggingface.co/AutonLab/MOMENT-1-large)
- 💻 [GitHub: AutonLab](https://github.com/autonlab)

---

### 7. ⭐⭐⭐ **DeOSAlphaTimeGPTPredictor-2025** (Vencortex.io)
**October 2025** - [HuggingFace](https://huggingface.co/vencortexio/DeOSAlphaTimeGPTPredictor-2025)

**2025 Version:**
- **Time series forecasting**
- **2025 version** (updated 26 days ago)
- **10 likes**

**Organizations:**
- Vencortex.io

**Links:**
- 🤗 [HuggingFace Model](https://huggingface.co/vencortexio/DeOSAlphaTimeGPTPredictor-2025)

---

## 📊 Comparison Table

| Model | Organization | Parameters | Downloads | Likes | Key Feature |
|---|---|---|---|---|---|
| **TiRex** | NXAI | **35M** | - | - | **Beats 500M models** |
| **Chronos-Bolt-Base** | AutoGluon | 200M | **4.37M** | 32 | **Most downloads** |
| **Chronos-2** | Amazon | 100M | - | - | **Multivariate** |
| **TimesFM-2.5** | Google | 200M | - | - | **Few-shot** |
| **IBM Granite TTM-R1** | IBM | **805K** | 331K | **313** | **Most likes, ultra compact** |
| **MOMENT-1-large** | AutonLab | Large | - | - | Foundation model |

---

## 🚀 Quick Start

### Installation:

```bash
# Install Hugging Face Transformers
pip install transformers

# For specific models:
pip install chronos-forecasting  # Chronos
pip install autogluon            # Chronos-Bolt
```

### Usage Example (Chronos):

```python
from chronos import ChronosPipeline

# Load model
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")

# Forecast
forecast = pipeline.predict(
    context=historical_data,
    prediction_length=12
)
```

### Usage Example (TiRex):

```python
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("nxai/tirex")

# Zero-shot forecast
forecast = model.predict(historical_data, horizon=24)
```

---

## 🇧🇷 Português

Modelos de fundação pré-treinados para previsão de séries temporais com capacidades de aprendizado zero-shot e few-shot.

[Conteúdo completo em português...]

---

**Last Updated:** November 2025  
**Maintained by:** AIForge Community
