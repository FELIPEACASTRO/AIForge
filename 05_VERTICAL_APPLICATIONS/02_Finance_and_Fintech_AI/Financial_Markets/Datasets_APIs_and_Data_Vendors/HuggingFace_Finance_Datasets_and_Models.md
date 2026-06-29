# HuggingFace — Finance Datasets & Models

> Curated map of the most-used **finance/markets** datasets and models on the HuggingFace Hub (pulled live via the HF API, ranked by downloads). Use these for sentiment, NLP on filings/news, financial QA, and time-series. Includes **Portuguese (PT-BR)** finance models for the Brazilian market.

> How this list was built: queried the HF Hub API (`huggingface_hub.HfApi.list_datasets / list_models`, `sort="downloads"`) across finance terms and de-duplicated. Download counts are point-in-time and shift over time — treat as relative popularity.

## 🤖 Models — financial sentiment, tone & NLP

| Model | Why it matters | Link |
|---|---|---|
| **ProsusAI/finbert** | The canonical FinBERT — financial sentiment (positive/negative/neutral); ~7.4M downloads | https://huggingface.co/ProsusAI/finbert |
| **yiyanghkust/finbert-tone** | FinBERT fine-tuned for analyst-tone classification on financial text | https://huggingface.co/yiyanghkust/finbert-tone |
| **mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis** | Lightweight financial-news sentiment | https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis |
| **ahmedrachid/FinancialBERT-Sentiment-Analysis** | FinancialBERT trained on financial phrasebank | https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis |
| **human-centered-summarization/financial-summarization-pegasus** | Summarize financial news / earnings text | https://huggingface.co/human-centered-summarization/financial-summarization-pegasus |
| **nickmuchi/finbert-tone-finetuned-finance-topic-classification** | Finance topic classification | https://huggingface.co/nickmuchi/finbert-tone-finetuned-finance-topic-classification |

### 🇧🇷 Portuguese / Brazil & other languages
| Model | Language | Link |
|---|---|---|
| **lucas-leme/FinBERT-PT-BR** | 🇧🇷 Brazilian-Portuguese financial sentiment (FinBERT-PT-BR) | https://huggingface.co/lucas-leme/FinBERT-PT-BR |
| **snunlp/KR-FinBert-SC** | 🇰🇷 Korean financial sentiment | https://huggingface.co/snunlp/KR-FinBert-SC |

> Also worth knowing (search the Hub): **FinGPT** (AI4Finance), **FinMA / PIXIU**, and instruction-tuned finance LLMs. Browse the tag pages: https://huggingface.co/models?search=finance and https://huggingface.co/models?search=fingpt

## 📊 Datasets — sentiment, news, QA, prices

| Dataset | Content | Link |
|---|---|---|
| **takala/financial_phrasebank** | The classic finance-sentiment benchmark (sentences labeled by agreement level) | https://huggingface.co/datasets/takala/financial_phrasebank |
| **zeroshot/twitter-financial-news-sentiment** | Finance tweets labeled for sentiment | https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment |
| **zeroshot/twitter-financial-news-topic** | Finance tweets labeled by topic | https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic |
| **PatronusAI/financebench** | FinanceBench — QA over real financial documents (filings) | https://huggingface.co/datasets/PatronusAI/financebench |
| **defeatbeta/yahoo-finance-data** | Yahoo Finance market data on the Hub (~136k downloads) | https://huggingface.co/datasets/defeatbeta/yahoo-finance-data |
| **paperswithbacktest/Stocks-Daily-Price** | Daily stock prices packaged for backtesting | https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price |
| **Multimodal Dataset (Image+Text+Table+TimeSeries) for Financial Time-Series Forecasting** | Multimodal financial forecasting | https://huggingface.co/datasets/tishtakalita/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting |
| **artefactory/Argimi-Ardian-Finance-10k-text** | Finance long-form text corpus | https://huggingface.co/datasets/artefactory/Argimi-Ardian-Finance-10k-text |
| crypto-market datasets, **trentmkelly/polymarket_crypto_derivatives** | Crypto/derivatives & prediction-market data | https://huggingface.co/datasets?search=crypto |

## 🔌 How to pull this yourself (API)

```python
from huggingface_hub import HfApi
api = HfApi()  # add token=... for higher rate limits
# top finance datasets by downloads
for d in api.list_datasets(search="financial", sort="downloads", limit=25):
    print(d.id, d.downloads, d.likes)
# top finance models
for m in api.list_models(search="finance", sort="downloads", limit=25):
    print(m.id, m.downloads, m.likes)
# load a dataset
from datasets import load_dataset
ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
```

- Hub search UIs: [models](https://huggingface.co/models?search=finance) · [datasets](https://huggingface.co/datasets?search=finance) · [HF Papers](https://huggingface.co/papers)

## Related in AIForge
- [Datasets, APIs & Data Vendors](../Datasets_APIs_and_Data_Vendors/) · [Alternative Data & Sentiment Analysis](../Alternative_Data_and_Sentiment_Analysis/) · [Technical & Fundamental Analysis](../Technical_and_Fundamental_Analysis/) · [Key Papers & Research](../Key_Papers_and_Research/)
- [`../../../../02_LLM_AND_AI_MODELS/`](../../../../02_LLM_AND_AI_MODELS/) · [`../../../../03_DATASETS_TOOLS_AND_RESOURCES/`](../../../../03_DATASETS_TOOLS_AND_RESOURCES/)

**Keywords:** HuggingFace finance, FinBERT, FinBERT-PT-BR, financial sentiment dataset, financial_phrasebank, FinanceBench, FinGPT, finance LLM, dataset de finanças, modelo de sentimento financeiro português.
