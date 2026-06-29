# Alternative Data & Sentiment Analysis

> "Alt-data" is non-traditional information used to predict fundamentals and prices before they show up in official numbers. Sentiment analysis turns text/social signals into features. This is where NLP and LLMs meet markets.

## Alternative data categories

| Category | Examples | Signal |
|---|---|---|
| Text / news | Newswires, Bloomberg, RavenPack, filings | Sentiment, events, surprise |
| Social | Twitter/X, Reddit (WallStreetBets), StockTwits | Crowd sentiment, retail flow, meme risk |
| Web / search | Web scraping, Google Trends, app downloads | Demand nowcasting |
| Satellite / geospatial | Parking lots, oil tanks, crop health, shipping | Retail traffic, supply, ag yields |
| Transaction / card | Aggregated card spend, receipts | Revenue nowcasting |
| Corporate disclosures | 10-K/10-Q, earnings-call transcripts | Tone, guidance, risk factors |
| ESG / supply chain | Sustainability data, shipping, logistics | Operational signals |

## NLP & LLMs for finance

- **FinBERT** and finance-tuned transformers for sentiment/classification.
- **LLMs** (GPT/Claude) to summarize filings, extract entities/events, score earnings-call tone, and answer questions over disclosures (RAG over EDGAR). See [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/).
- **Loughran-McDonald** finance lexicon (generic sentiment dictionaries misclassify finance text).
- **Event studies & nowcasting**: map text events to abnormal returns; nowcast GDP/sales.
- **Caveats**: alpha decay as data commoditizes, look-ahead via revised data, point-in-time discipline, data costs, and **survivorship/coverage bias** in alt-data panels.

## Data & tools

- News/sentiment: RavenPack, Refinitiv News Analytics. Social: PRAW (Reddit), `snscrape`/X API, StockTwits. Search: `pytrends`. Filings: SEC EDGAR. Satellite: Orbital Insight, Planet, RS Metrics. NLP: HuggingFace `transformers`, `FinBERT`, spaCy.

## Key references

- Loughran & McDonald (2011) — finance sentiment. https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2010.01625.x
- Tetlock, *Giving Content to Investor Sentiment* (JF 2007). https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01232.x
- Araci, *FinBERT* (2019). https://arxiv.org/abs/1908.10063
- Lopez-Lira & Tang, *Can ChatGPT Forecast Stock Price Movements?* (2023). https://arxiv.org/abs/2304.07619

## Related in AIForge
- [Technical & Fundamental Analysis](../Technical_and_Fundamental_Analysis/) · [Equities & Stock Markets](../Equities_and_Stock_Markets/) · [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Crypto & Digital Assets](../Crypto_and_Digital_Assets/)
- Fundamentals: [`../../../../02_LLM_AND_AI_MODELS/`](../../../../02_LLM_AND_AI_MODELS/) · [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/RAG_and_Retrieval/)

**Keywords:** alternative data, financial sentiment analysis, FinBERT, LLM finance, satellite data trading, card transaction nowcasting, Google Trends signals, earnings call NLP, RavenPack, WallStreetBets sentiment.
