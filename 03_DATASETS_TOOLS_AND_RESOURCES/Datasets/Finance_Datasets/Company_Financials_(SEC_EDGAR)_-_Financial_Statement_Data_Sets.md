# Company Financials (SEC EDGAR) - Financial Statement Data Sets

## Description
The "Financial Statement Data Sets" from the U.S. SEC (Securities and Exchange Commission) provide numerical information extracted from the financial statements of all companies that file corporate reports using the XBRL (eXtensible Business Reporting Language) language through the EDGAR (Electronic Data Gathering, Analysis, and Retrieval) system. The dataset presents the data in a "flattened" format to facilitate the analysis and comparison of financial information over time and across different registrants. It includes data from balance sheets, income statements, and cash flow statements.

## Statistics
The dataset is updated quarterly and covers data from January 2009 through the most recent quarter (June 2025, according to the source). Each quarterly file has an average size of approximately 100 MB (ranging from 75 MB to 122 MB in the recent quarters of 2023-2025). The total number of samples (records) is massive, with millions of entries per quarter, depending on the file (submissions, numeric data, etc.). The most recent version is the second quarter of 2025 (2025 Q2).

## Features
Numerical financial statement data (balance sheet, income statement, cash flow) from all companies listed in the U.S. The data is extracted from XBRL (eXtensible Business Reporting Language) reports filed in the EDGAR system. The format is "flattened" to facilitate analysis and comparison. It includes additional fields such as the company's Standard Industrial Classification (SIC). The dataset is composed of four main files: `sub` (submission information), `num` (numeric data), `tag` (XBRL tag definitions), and `pre` (presentation structure).

## Use Cases
Predictive modeling of bankruptcy or financial performance. Sentiment analysis and market risk based on financial disclosures. Training Large Language Models (LLMs) for understanding financial documents. Academic research in corporate finance and accounting. Development of financial data analysis and regulatory compliance tools.

## Integration
The data is made available in quarterly ZIP files containing tab-delimited text files (TXT). To use it, you need to download the ZIP files for the desired quarter and extract them. The SEC also provides an API (Application Programming Interface) for programmatic access to the data, allowing developers to query EDGAR submissions by company and extracted XBRL data. There are third-party libraries in languages such as Python and R (for example, `edgar` in R) that facilitate downloading and analyzing the data.

## URL
[https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets)
