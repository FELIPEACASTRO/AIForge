# Kaggle — Finance Competitions & Datasets

> Curated index of **financial-markets** competitions and datasets on Kaggle (pulled live via the Kaggle API). The flagship quant competitions here are some of the best public benchmarks for market prediction; the datasets are ready-made for stocks, options, crypto, FX, and sentiment.

> How this list was built: queried the Kaggle API (`competitions_list`, `dataset_list`) across finance terms and de-duplicated. Use these as starting points — always check each competition's rules/license before use.

## 🏆 Landmark quant competitions (the canonical benchmarks)

These are the most cited market-prediction competitions on Kaggle (host-run, large prize pools):

| Competition | What it is | Link |
|---|---|---|
| **Jane Street Market Prediction** | Real anonymized market data; predict a trading signal — the reference modern quant comp | https://www.kaggle.com/competitions/jane-street-market-prediction |
| **Jane Street Real-Time Market Data Forecasting** (2024) | Successor, time-series API, live forecasting | https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting |
| **Optiver — Trading at the Close** | Predict closing-auction price moves from order-book/imbalance | https://www.kaggle.com/competitions/optiver-trading-at-the-close |
| **Optiver Realized Volatility Prediction** | Predict short-term realized vol from order-book & trade data | https://www.kaggle.com/competitions/optiver-realized-volatility-prediction |
| **G-Research Crypto Forecasting** | Forecast returns of 14 cryptoassets | https://www.kaggle.com/competitions/g-research-crypto-forecasting |
| **Ubiquant Market Prediction** | Chinese-market quant return prediction | https://www.kaggle.com/competitions/ubiquant-market-prediction |
| **JPX Tokyo Stock Exchange Prediction** | Rank Japanese stocks by future return | https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction |
| **Two Sigma: Using News to Predict Stock Movements** | News + market data signal | https://www.kaggle.com/competitions/two-sigma-financial-news |
| **Two Sigma Financial Modeling Challenge** | Classic anonymized-feature return prediction | https://www.kaggle.com/competitions/two-sigma-financial-modeling |

> 🔎 The **winning-solution write-ups** for these live in AIForge's Kaggle index — see [Project Showcases / Kaggle](../../../20_AI_Project_Showcases/Kaggle/).

### More market competitions (community + course)
- HKU QIDS Quantitative Investment Competition — https://www.kaggle.com/competitions/hku-qids-2023-quantitative-investment-competition
- High Frequency Price Prediction of Index Futures (Caltech CS155) — https://www.kaggle.com/competitions/caltech-cs155-2020
- Market Movement Prediction — https://www.kaggle.com/competitions/market-movement-prediction
- AIA Forex Prediction — https://www.kaggle.com/competitions/aiatc6-forex-prediction
- Financial News Headlines (NLP) — https://www.kaggle.com/competitions/me-ph-i-m-25-ml-financial-news-headlines

## 📦 Datasets — prices, fundamentals, options, sentiment

### Equities / indices (US & global)
| Dataset | Content | Link |
|---|---|---|
| Huge Stock Market Dataset | Price+volume for all US stocks & ETFs | https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs |
| S&P 500 stock data | Daily OHLCV for S&P 500 | https://www.kaggle.com/datasets/camnugent/sandp500 |
| S&P 500 Stocks (daily updated) | Refreshed S&P 500 prices + companies | https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks |
| Stock Market Data (NASDAQ, NYSE, S&P500) | Multi-exchange historical | https://www.kaggle.com/datasets/paultimothymooney/stock-market-data |
| NYSE | Prices + fundamentals | https://www.kaggle.com/datasets/dgawlik/nyse |
| Daily Historical Stock Prices (1970–2018) | Long history | https://www.kaggle.com/datasets/ehallmar/daily-historical-stock-prices-1970-2018 |
| 200+ Financial Indicators of US stocks (2014–2018) | Fundamentals for ML | https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018 |
| US historical stock prices with earnings | Prices + earnings | https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data |

### India (NSE/Nifty — very active community)
| Dataset | Link |
|---|---|
| NIFTY-50 Stock Market Data (2000–2021) | https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data |
| NSE Nifty50 Index minute-level data | https://www.kaggle.com/datasets/tomtillo/nse-nifty50-index-daily-minute-level-data |
| ALGO TRADING DATA — Nifty intraday w/ indicators | https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators |
| NSE Future and Options dataset 2024 | https://www.kaggle.com/datasets/kaalicharan9080/nse-future-and-options-data |
| Historical Nifty Options 2024 (all expiries) | https://www.kaggle.com/datasets/senthilkumarvaithi/historical-nifty-options-2024-all-expiries |

### Options, futures, crypto, FX, gold
| Dataset | Link |
|---|---|
| options market trades | https://www.kaggle.com/datasets/bendgame/options-market-trades |
| Options in Korean Stock Market (VKOSPI) | https://www.kaggle.com/datasets/ninetyninenewton/vkospi |
| High-Frequency Trading (HFT) trades | https://www.kaggle.com/datasets/mahmoudshaheen1134/trades-dataset |
| Gold Price Prediction Dataset | https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset |
| Currency Rate Euro→USD (2003–2022) | https://www.kaggle.com/datasets/meetnagadia/currency-rate-euro-to-usd |
| Stock Exchange Data (global indices) | https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data |

### News & sentiment (NLP for markets)
| Dataset | Link |
|---|---|
| Daily News for Stock Market Prediction | https://www.kaggle.com/datasets/aaron7sun/stocknews |
| Daily Financial News for 6000+ Stocks | https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests |
| Sentiment Analysis for Financial News | https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news |
| Financial Sentiment Analysis | https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis |
| Stock-Market Sentiment Dataset | https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset |
| Tweets about the Top Companies (2015–2020) | https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020 |
| r/wallstreetbets submissions | https://www.kaggle.com/datasets/shergreen/wallstreetbets-subreddit-submissions |

> ℹ️ **Brazil/B3 note:** Kaggle's Brazilian-market coverage is thin; for B3 data use **brapi.dev** (https://brapi.dev/), **MetaTrader5**, `yfinance` with the `.SA` suffix (e.g. `PETR4.SA`), and B3/Status Invest.

## 🔌 How to pull this yourself (API)

```bash
pip install kaggle   # put kaggle.json in ~/.kaggle/
kaggle competitions list -s "stock"
kaggle datasets list -s "stock market" --sort-by votes
kaggle datasets download -d camnugent/sandp500
```
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
resp = api.competitions_list(search="trading")   # -> resp.competitions
for d in api.dataset_list(search="options trading", sort_by="votes"):
    print(d.ref, d.title)
```

## Related in AIForge
- [Datasets, APIs & Data Vendors](../Datasets_APIs_and_Data_Vendors/) · [Backtesting & Frameworks](../Backtesting_and_Frameworks/) · [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Key Papers & Research](../Key_Papers_and_Research/)
- Winning solutions: [`../../../20_AI_Project_Showcases/Kaggle/`](../../../20_AI_Project_Showcases/Kaggle/)

**Keywords:** Kaggle finance, Jane Street market prediction, Optiver realized volatility, G-Research crypto, JPX Tokyo, Two Sigma, quant competition, stock dataset, options dataset, financial sentiment dataset, dataset de ações.
