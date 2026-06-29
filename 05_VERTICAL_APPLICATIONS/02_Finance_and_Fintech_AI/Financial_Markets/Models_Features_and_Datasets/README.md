# 🧰 Models, Features & Datasets for Markets (Global)

> Country-agnostic ML building blocks usable on **any market in the world** — a model zoo, a feature-engineering catalog, and a global dataset directory. Every entry was fact-checked to confirm the model/library/dataset actually exists with a working link.

These three catalogs are the "toolbox" layer of [Financial Markets](../): pick a model, engineer features, point it at data — for equities, ETFs, futures, options, FX, bonds, or crypto, in any region.

| Catalog | What's inside |
|---|---|
| [Models for Markets](./Models_for_Markets.md) | Gradient boosting (XGBoost/LightGBM/CatBoost), classical TS (ARIMA/GARCH), deep TS (TFT, N-BEATS, N-HiTS, PatchTST), **time-series foundation models** (Chronos, TimesFM, Moirai, Lag-Llama, TTM, MOMENT), LOB models (DeepLOB/TLOB), RL (FinRL), finance LLMs (FinGPT, PIXIU/FinMA, FinBERT, **FinBERT-PT-BR**), Qlib. |
| [Features & Feature Engineering](./Features_and_Feature_Engineering.md) | Returns/momentum, technical indicators (TA-Lib/pandas-ta), volatility estimators (Garman-Klass/Yang-Zhang), microstructure (OFI/OBI/VPIN), cross-sectional factors (**WorldQuant 101 Alphas**, Qlib Alpha158/360), calendar, fundamentals, NLP/alt-data, auto-FE (tsfresh/Featuretools), alphalens; with leakage/point-in-time guidance. |
| [Global Datasets for Markets](./Global_Datasets_for_Markets.md) | Prices/fundamentals (Stooq, Nasdaq Data Link, Tiingo, EOD, **Fama-French & JKP Global Factor Data**), macro (FRED/World Bank/OECD/BIS), LOB/tick (LOBSTER/Databento), options/vol (ORATS/Deribit), FX, crypto (CCXT/Kaiko/Glassnode), news/sentiment (FNSPID/FiQA), Numerai/paperswithbacktest. |

## Related in AIForge
- Parent: [`../`](../) (Financial Markets) · [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Backtesting & Frameworks](../Backtesting_and_Frameworks/) · [Key Papers & Research](../Key_Papers_and_Research/)
- Platform pulls: [HuggingFace Finance](../Datasets_APIs_and_Data_Vendors/HuggingFace_Finance_Datasets_and_Models.md) · [Kaggle Finance](../Datasets_APIs_and_Data_Vendors/Kaggle_Finance_Competitions_and_Datasets.md)
- Theory: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/)

**Keywords:** financial markets ML models, time-series foundation models, feature engineering finance, alpha factors, WorldQuant 101 alphas, Qlib, market datasets, modelos para mercado financeiro, engenharia de atributos, datasets de mercado.
