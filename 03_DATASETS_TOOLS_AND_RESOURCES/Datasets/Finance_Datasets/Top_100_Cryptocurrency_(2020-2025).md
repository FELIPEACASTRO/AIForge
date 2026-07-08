# Top 100 Cryptocurrency (2020-2025)

## Description
This is a comprehensive dataset of daily cryptocurrency prices, covering the top 100 cryptocurrencies by market capitalization. It provides OHLC (Open, High, Low, Close) data essential for time series analysis and the development of predictive models. The dataset is valuable for researchers and developers interested in quantitative finance and machine learning applied to the digital asset market. The inclusion of the blockchain network associated with each asset adds a useful layer of information for deeper analyses.

## Statistics
**Dataset Size:** 11.9 MB (single CSV file).
\n**Sample Count:** More than 200,000 rows of data.
\n**Versions:** Version 1 (updated 3 months ago, as of the research date).
\n**Coverage Period:** 2020 to 2025.
\n**Granularity:** Daily.
\n**Assets:** 100 cryptocurrencies.

## Features
Daily OHLC price data (Open, High, Low, Close); Covers the top 100 cryptocurrencies by market capitalization; Includes the "Blockchain Network" field for each asset; Easy-to-use CSV format.

## Use Cases
Time series forecasting and cryptocurrency price prediction; Development of AI/ML-based trading strategies; Correlation analysis and the impact of news sentiment on the market; Portfolio optimization based on historical trends; Academic research on digital asset market dynamics.

## Integration
The dataset is hosted on Kaggle and can be downloaded directly via the "Download" button on the resource page. For Kaggle users, integration is facilitated by using the Kaggle API or directly within the platform's Notebooks. For external use, the CSV file must be downloaded and imported into development environments such as Python (with Pandas) or R.
\n\n**Example usage in Python (after download):**
\n```python
\nimport pandas as pd
\n
\n# Load the CSV file
\ndf = pd.read_csv('top_100_cryptos_with_correct_network.csv')
\n
\n# View the first rows
\nprint(df.head())
\n
\n# Convert the 'date' column to datetime format
\ndf['date'] = pd.to_datetime(df['date'])
\n
\n# Basic analysis (e.g., average closing price)
\nprint(df['close'].mean())
\n```

## URL
[https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025](https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025)
