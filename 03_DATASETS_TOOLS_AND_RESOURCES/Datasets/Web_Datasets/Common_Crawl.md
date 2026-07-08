# Common Crawl

## Description
Common Crawl is a non-profit organization (501(c)(3)) founded in 2007 that maintains an open, free repository of web crawl data. Its goal is to make large-scale extraction, transformation, and analysis of open web data accessible to researchers. The total corpus contains more than 300 billion pages and is cited in over 10,000 research papers. It is the largest source of pre-training data for most modern Large Language Models (LLMs).

## Statistics
Frequency: Monthly. Latest Version (October 2025): CC-MAIN-2025-43. Size (October 2025): 2.61 billion web pages, or 468 TiB of uncompressed content. Total Size (Corpus): More than 300 billion pages, spanning 18 years.

## Features
The dataset is made available in raw and processed formats, including WARC (Web ARChive), WAT (Web Archive Transformation), and WET (Web Extracted Text) files. It includes raw web data, metadata, and extracted text. The organization also regularly releases Web Graphs (host and domain level) and quality annotations such as GneissWeb for content filtering.

## Use Cases
Training Large Language Models (LLMs) such as GPT and LLaMA. Academic research in web data mining, trend analysis, digital preservation studies, and computational linguistics. Trend analysis and monitoring the evolution of the web.

## Integration
Data access is free and can be done in two main ways: 1. **Amazon S3:** The data is hosted in the `s3://commoncrawl/` bucket in AWS's US-East-1 (Northern Virginia) region. Accessing via the S3 API requires authentication. 2. **HTTP(S) Download:** To download files directly without an AWS account, use the `https://data.commoncrawl.org/` prefix followed by the file path. Tools such as `wget` or `curl` can be used. Common Crawl also provides tools and libraries for processing the data.

## URL
[https://commoncrawl.org/](https://commoncrawl.org/)
