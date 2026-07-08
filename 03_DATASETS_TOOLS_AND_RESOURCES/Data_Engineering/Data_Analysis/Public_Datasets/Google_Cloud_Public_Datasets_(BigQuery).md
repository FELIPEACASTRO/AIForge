# Google Cloud Public Datasets (BigQuery)

## Description

The **Google Cloud Public Datasets** are a vast repository of public datasets hosted on **BigQuery**, Google Cloud's serverless, highly scalable, and low-cost data warehouse. The unique value lies in allowing users and data scientists to query petabytes of complex data (such as genomics, climate, financial, and traffic data) using standard SQL, without the need to set up infrastructure, upload data, or pay for storage. The program is maintained by Google, which covers storage costs, and offers free access to processing the first 1 TB of data per month, making it an invaluable tool for research, analysis, and Machine Learning prototyping.

## Statistics

**Cost:** Google pays for storage. Query processing is free up to 1 TB per month per user. **Size:** The repository contains petabytes of data. Some individual datasets, such as Common Crawl, can exceed 170 TB. **Accessibility:** Immediate access to hundreds of high-quality public datasets. **Location:** Datasets are stored in multi-regional locations, such as `US` or `EU`.

## Features

**SQL Access:** Allows querying petabytes of data using standard SQL (GoogleSQL or Legacy SQL). **Managed Infrastructure:** Requires no infrastructure setup, data upload, or storage management. **Data Variety:** Includes data from various areas such as science, finance, healthcare, climate, and government. **Native Integration:** Direct access via Google Cloud Console, the `bq` command-line tool, and REST APIs. **Cost Model:** Google pays for storage; the user pays only for queries (with 1 TB of free processing per month).

## Use Cases

**Scientific Research:** Analysis of genomic data (such as the NIH chest x-ray dataset) and climate data (such as NOAA's GSOD) for scientific discoveries. **Financial Analysis:** Use of market and transaction data for predictive modeling and backtesting. **Machine Learning:** Training ML models at scale with ready-to-use datasets, such as the Wikipedia revision dataset or the GitHub traffic dataset. **Education and Prototyping:** Allows students and developers to experiment with BigQuery and SQL on large volumes of data without initial infrastructure costs. **Media Analysis:** Querying large volumes of text data, such as the trigram dataset or the news dataset.

## Integration

Access and integration are performed through the BigQuery API, supported by various client libraries. The most common method is via Python, using the `google-cloud-bigquery` library.

**Python Integration Example:**
```python
# Installation: pip install google-cloud-bigquery
from google.cloud import bigquery

# Initialize the BigQuery client
client = bigquery.Client()

# SQL query for the US names dataset
query = """
    SELECT
        name,
        sum(number) AS total_births
    FROM
        `bigquery-public-data.usa_names.usa_1910_2013`
    WHERE
        state = 'TX'
    GROUP BY
        name
    ORDER BY
        total_births DESC
    LIMIT 10
"""

query_job = client.query(query)  # Send the request to the API

print("The 10 most popular names in Texas (1910-2013):")
for row in query_job:
    print(f"Name: {row['name']}, Births: {row['total_births']}")
```

**Access via Console:**
1.  Navigate to BigQuery in the Google Cloud Console.
2.  Click **+ ADD DATA** and select **Explore public datasets**.
3.  Search for and add the desired dataset to your project.

**Access via Command Line (`bq`):**
```bash
# Runs a query and saves the result to a CSV file
bq query --use_legacy_sql=false --format=csv "SELECT word, word_count FROM \`bigquery-public-data.samples.shakespeare\` WHERE word = 'love'" > love_count.csv
```

## URL

https://docs.cloud.google.com/bigquery/public-data