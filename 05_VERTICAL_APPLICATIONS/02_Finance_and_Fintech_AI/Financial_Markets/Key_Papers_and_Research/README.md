# Key Papers & Research

> A curated reading list for ML in the financial markets — foundations, asset pricing with ML, deep learning, RL, pricing/hedging, and the methodology that keeps you honest.

## Foundations & theory

- Markowitz, *Portfolio Selection* (1952). https://www.jstor.org/stable/2975974
- Sharpe, *Capital Asset Prices* (CAPM, 1964).
- Fama, *Efficient Capital Markets* (1970) — the EMH null you must beat.
- Black & Scholes (1973); Merton (1973) — option pricing.
- Fama & French, *Common Risk Factors* (1993) — the factor revolution.

## Methodology (avoid fooling yourself)

- López de Prado, *Advances in Financial Machine Learning* (2018) — the modern bible: purged/embargoed CV, meta-labeling, fractional differentiation, deflated Sharpe. https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- Bailey, Borwein, López de Prado, Zhu, *Probability of Backtest Overfitting* (2014). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Harvey, Liu, Zhu, *…and the Cross-Section of Expected Returns* (2016) — the multiple-testing crisis in factor research.

## Empirical asset pricing with ML

- Gu, Kelly, Xiu, *Empirical Asset Pricing via Machine Learning* (RFS 2020). https://academic.oup.com/rfs/article/33/5/2223/5758276
- Kelly, Pruitt, Su, *Characteristics Are Covariances* (IPCA, 2019).
- Bianchi, Büchner, Tamoni, *Bond Risk Premia with Machine Learning* (RFS 2021). https://academic.oup.com/rfs/article/34/2/1046/5821387

## Deep learning & limit order books

- Zhang, Zohren, Roberts, *DeepLOB* (2019). https://arxiv.org/abs/1808.03668
- Sirignano & Cont, *Universal features of price formation* (2019). https://arxiv.org/abs/1803.06917
- Lim, Zohren, Roberts, *Enhancing Time Series Momentum with Deep Neural Networks* (2019). https://arxiv.org/abs/1904.04912

## Reinforcement learning & execution

- Almgren & Chriss, *Optimal Execution of Portfolio Transactions* (2000). https://www.math.nyu.edu/~almgren/papers/optliq.pdf
- Nevmyvaka, Feng, Kearns, *Reinforcement Learning for Optimized Trade Execution* (ICML 2006). https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf
- Liu et al., *FinRL* (2020). https://arxiv.org/abs/2011.09607
- Deng et al., *Deep Direct Reinforcement Learning for Financial Signal Representation and Trading* (2017).

## Pricing, hedging & volatility

- Buehler, Gonon, Teichmann, Wood, *Deep Hedging* (2019). https://arxiv.org/abs/1802.03042
- Horvath, Muguruza, Tomas, *Deep Learning Volatility* (2019). https://arxiv.org/abs/1901.09647
- Han, Jentzen, E, *Solving high-dimensional PDEs using deep learning* (Deep BSDE, 2018). https://www.pnas.org/doi/10.1073/pnas.1718942115
- Huge & Savine, *Differential Machine Learning* (2020). https://arxiv.org/abs/2005.02347

## NLP / LLMs in finance

- Loughran & McDonald (2011) — finance sentiment lexicon. https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2010.01625.x
- Tetlock, *Giving Content to Investor Sentiment* (2007). https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01232.x
- Araci, *FinBERT* (2019). https://arxiv.org/abs/1908.10063
- Wu et al., *BloombergGPT: A Large Language Model for Finance* (2023). https://arxiv.org/abs/2303.17564
- Lopez-Lira & Tang, *Can ChatGPT Forecast Stock Price Movements?* (2023). https://arxiv.org/abs/2304.07619

## Surveys

- Sezer, Gudelek, Ozbayoglu, *Financial Time Series Forecasting with Deep Learning: A Survey* (2020). https://arxiv.org/abs/1911.13288
- Ozbayoglu, Gudelek, Sezer, *Deep Learning for Financial Applications: A Survey* (2020). https://arxiv.org/abs/2002.05786
- Hambly, Xu, Yang, *Recent Advances in Reinforcement Learning in Finance* (2021). https://arxiv.org/abs/2112.04553

## 🔎 Where to find the research (platforms & search)
| Page | What's inside |
|---|---|
| [arXiv q-fin & Preprint Servers](./arXiv_q-fin_and_Preprints.md) | All nine q-fin subcategories (CP/GN/MF/PM/PR/RM/ST/TR/EC), arXiv API & listing/RSS URLs, ar5iv/alphaXiv/SciRate, general & Brazil preprints (SciELO Preprints). |
| [Economics & Finance Working Papers](./Economics_and_Finance_Working_Papers.md) | SSRN/FEN, NBER, RePEc/IDEAS/EconPapers, CEPR, EconStor, MPRA, Fed/IMF/World Bank, 🇧🇷 BCB & RBFin/SBFin — with how-to-search & JEL tips. |
| [Research Search Engines & Aggregators](./Research_Search_Engines_and_Aggregators.md) | Papers with Code (→ HF Papers), Semantic Scholar API, OpenAlex API, DBLP, CORE, Crossref, Unpaywall — example queries for ML-finance literature pipelines. |

## 📖 Learn the field
- [Books, Courses & Learning Resources](./Books_Courses_and_Learning_Resources.md) — Hull, Natenberg, Sinclair, Taleb, Gatheral, López de Prado, Chan, Jansen; courses (QuantInsti EPAT, WorldQuant University, Coursera/Georgia Tech, CQF, Damodaran); blogs/podcasts; 🇧🇷 QuantBrasil, Asimov Academy.

## Related in AIForge
- [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Backtesting & Frameworks](../Backtesting_and_Frameworks/) · [Options & Derivatives](../Options_and_Derivatives/) · [Market Microstructure & HFT](../Market_Microstructure_and_HFT/) · [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Reinforcement_Learning/) · [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/)

**Keywords:** financial machine learning papers, empirical asset pricing ML, DeepLOB, deep hedging, deep BSDE, FinRL, BloombergGPT, FinBERT, López de Prado, backtest overfitting, optimal execution.
