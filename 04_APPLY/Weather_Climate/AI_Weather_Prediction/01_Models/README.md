# 🤖 AI Weather Prediction Models / Modelos de IA para Previsão Meteorológica

---

## 🌍 English

State-of-the-art AI models that revolutionize weather forecasting by achieving superior accuracy with dramatically reduced computational requirements.

---

## 🏆 Revolutionary Models

### 1. ⭐⭐⭐⭐⭐ **Aardvark Weather** (University of Cambridge)
**Nature 2025** - DOI: [10.1038/s41586-025-08897-0](https://doi.org/10.1038/s41586-025-08897-0)

**The Game Changer:**
- **Thousands of times faster** than traditional NWP systems
- Runs on **desktop computer** instead of supercomputer
- Uses only **10% of the data** required by existing systems
- Developed in **18 months** (traditional systems take decades)
- **Outperforms** US National Forecasting System (GFS)

**Technical Details:**
- **Architecture:** End-to-end AI system (replaces entire NWP pipeline)
- **Training Data:** ERA5 reanalysis dataset (reduced subset)
- **Inference:** Real-time on consumer hardware
- **Resolution:** Global weather prediction
- **Forecast Range:** Medium-range (up to 10 days)

**Organizations:**
- University of Cambridge
- Alan Turing Institute
- Microsoft Research
- ECMWF (European Centre for Medium-Range Weather Forecasts)

**Impact:**
- **Democratization:** Enables developing countries to run world-class forecasting
- **Cost Reduction:** Eliminates need for expensive supercomputers
- **Speed:** Faster response to extreme weather events
- **Accessibility:** Desktop deployment for research and education

**Links:**
- 📄 [Nature Paper](https://www.nature.com/articles/s41586-025-08897-0)
- 🌐 [Cambridge News](https://www.cam.ac.uk/research/news/fully-ai-driven-weather-prediction-system-could-start-revolution-in-forecasting)
- 🏛️ [Alan Turing Institute](https://www.turing.ac.uk/)

---

### 2. ⭐⭐⭐⭐⭐ **GenCast** (Google DeepMind)
**Nature 2024** - DOI: [10.1038/s41586-024-08252-9](https://doi.org/10.1038/s41586-024-08252-9)

**State-of-the-Art Performance:**
- **97.2% better** than ENS (ECMWF Ensemble System) across 1,320 targets
- Generates **15-day forecast in 8 minutes**
- **0.25° resolution** (approximately 25km)
- Predicts **80+ variables** (surface + atmospheric)

**Technical Details:**
- **Architecture:** Probabilistic diffusion model
- **Training Data:** ERA5 reanalysis (1979-2018, 40 years)
- **Ensemble:** Generates 50+ ensemble members
- **Resolution:** 0.25° latitude/longitude
- **Variables:** Temperature, wind, precipitation, pressure, humidity, etc.

**Capabilities:**
- **Extreme Weather Prediction:** Hurricanes, floods, heatwaves
- **Tropical Cyclone Tracking:** Superior track prediction
- **Wind Power Forecasting:** Renewable energy optimization
- **Probabilistic Forecasting:** Uncertainty quantification

**Performance Metrics:**
- **97.2%** of targets outperform ENS
- **174 citations** (in less than 1 year)
- **335k+ accesses**
- **1475 Altmetric score**

**Organizations:**
- Google DeepMind
- ECMWF (collaboration)

**Links:**
- 📄 [Nature Paper](https://www.nature.com/articles/s41586-024-08252-9)
- 💻 [GitHub (Open Source)](https://github.com/google-deepmind/graphcast)
- 🌐 [Google DeepMind Blog](https://deepmind.google/discover/blog/)

---

### 3. ⭐⭐⭐⭐ **GraphCast** (Google DeepMind)
**Science 2023** - DOI: [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)

**Breakthrough:**
- **Graph Neural Network** (GNN) for weather forecasting
- **10-day forecasts in under 1 minute**
- **0.25° resolution**
- **More accurate** than HRES (ECMWF high-resolution system)

**Technical Details:**
- **Architecture:** Graph Neural Network (GNN)
- **Training Data:** ERA5 reanalysis (1979-2017)
- **Inference:** <1 minute on TPU
- **Variables:** 227 variables
- **Levels:** 37 atmospheric levels

**Performance:**
- **90%** of targets more accurate than HRES
- **Extreme weather:** Superior cyclone tracking
- **Long-range:** Maintains accuracy up to 10 days

**Links:**
- 📄 [Science Paper](https://www.science.org/doi/10.1126/science.adi2336)
- 💻 [GitHub](https://github.com/google-deepmind/graphcast)
- 🤗 [HuggingFace Demo](https://huggingface.co/spaces/google/graphcast)

---

### 4. ⭐⭐⭐⭐ **FourCastNet** (NVIDIA)
**Preprint 2022** - ArXiv: [2202.11214](https://arxiv.org/abs/2202.11214)

**Innovation:**
- **Fourier Neural Operator** (FNO) architecture
- **Global forecast in 2 seconds**
- **0.25° resolution**
- **GPU-optimized** for real-time inference

**Technical Details:**
- **Architecture:** Adaptive Fourier Neural Operator (AFNO)
- **Training Data:** ERA5 reanalysis
- **Inference:** 2 seconds on A100 GPU
- **Variables:** 20+ atmospheric variables

**Performance:**
- **Competitive** with IFS (ECMWF)
- **2000x faster** than traditional NWP
- **Energy efficient:** 12,400x less energy

**Organizations:**
- NVIDIA
- Lawrence Berkeley National Laboratory

**Links:**
- 📄 [ArXiv Paper](https://arxiv.org/abs/2202.11214)
- 💻 [GitHub](https://github.com/NVlabs/FourCastNet)
- 🌐 [NVIDIA Blog](https://developer.nvidia.com/blog/)

---

## 📊 Comparison Table

| Model | Organization | Speed | Accuracy | Resolution | Forecast Range |
|---|---|---|---|---|---|
| **Aardvark** | Cambridge | **Desktop** | > GFS | Global | 10 days |
| **GenCast** | DeepMind | **8 min** | **97.2% > ENS** | 0.25° | 15 days |
| **GraphCast** | DeepMind | <1 min | 90% > HRES | 0.25° | 10 days |
| **FourCastNet** | NVIDIA | **2 sec** | ≈ IFS | 0.25° | 10 days |

---

## 🚀 Getting Started

### For Researchers:
1. **Read Papers:** Start with GenCast and Aardvark (Nature)
2. **Download Code:** GraphCast and FourCastNet are open source
3. **Experiment:** Use ERA5 dataset for training

### For Developers:
1. **GraphCast:** Most accessible (HuggingFace demo available)
2. **FourCastNet:** Best for GPU deployment
3. **GenCast:** Best for ensemble forecasting

### For Businesses:
1. **Evaluate:** Compare models for your use case
2. **Deploy:** GraphCast for production (open source)
3. **Integrate:** Use APIs or self-host

---

## 📚 Additional Resources

### Datasets:
- **ERA5:** [ECMWF Reanalysis](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- **WeatherBench:** [Benchmark Dataset](https://github.com/pangeo-data/WeatherBench)

### Tutorials:
- **GraphCast Tutorial:** [HuggingFace](https://huggingface.co/spaces/google/graphcast)
- **FourCastNet Tutorial:** [NVIDIA Developer](https://developer.nvidia.com/)

### Communities:
- **ECMWF:** [AI for Weather](https://www.ecmwf.int/)
- **WMO:** [World Meteorological Organization](https://public.wmo.int/)

---

## 🇧🇷 Português

Modelos de IA de última geração que revolucionam a previsão meteorológica ao alcançar precisão superior com requisitos computacionais drasticamente reduzidos.

---

## 🏆 Modelos Revolucionários

### 1. ⭐⭐⭐⭐⭐ **Aardvark Weather** (University of Cambridge)
**Nature 2025** - DOI: [10.1038/s41586-025-08897-0](https://doi.org/10.1038/s41586-025-08897-0)

**O Divisor de Águas:**
- **Milhares de vezes mais rápido** que sistemas NWP tradicionais
- Roda em **computador desktop** em vez de supercomputador
- Usa apenas **10% dos dados** requeridos por sistemas existentes
- Desenvolvido em **18 meses** (sistemas tradicionais levam décadas)
- **Supera** o Sistema Nacional de Previsão dos EUA (GFS)

[Conteúdo completo em português espelhando a versão em inglês...]

---

**Last Updated:** November 2025  
**Maintained by:** AIForge Community
