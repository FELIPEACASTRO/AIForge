# AWS Open Data Registry

## Description

The **Registry of Open Data on AWS** is a centralized catalog that facilitates the discovery and access of **high-value, cloud-optimized public datasets** that are available through AWS resources such as Amazon S3. Its unique value proposition lies in **cloud optimization**, with data stored in formats that enable direct analysis (e.g., Parquet, Zarr), and in the **AWS Open Data Sponsorship Program**, which covers storage and data transfer (egress) costs for providers, making access free to the public. Recently, all Registry datasets became **discoverable in AWS Data Exchange**, unifying the search for open, free, and commercial data.

## Statistics

*   **Data Volume:** More than **300 Petabytes (PB)** of high-value, cloud-optimized data are made available through the program.
*   **Number of Datasets:** The catalog lists **more than 800 datasets** (the main page indicates "currently 818 matching datasets").
*   **Providers:** Includes data from organizations such as NASA, NOAA, NIH, Allen Institute for AI (AI2), and Common Crawl.

## Features

*   **Centralized Discovery:** Search and browse interface to find datasets by keyword, category (e.g., genomics, climate, transportation), and provider.
*   **Direct S3 Access:** The data is accessible directly from public Amazon S3 buckets, enabling the use of AWS analytics services (such as Amazon Athena, Amazon EMR, Amazon SageMaker) without the need to move the data.
*   **Usage Examples:** Provides tutorials and Usage Examples, including SageMaker Studio Lab notebooks, to accelerate project kickoff.
*   **Rich Metadata:** Each dataset has a details page with metadata, description, tags, and licensing information.

## Use Cases

*   **Scientific Research:** Analysis of genomic data (The Cancer Genome Atlas - TCGA), climate data (NOAA), and remote sensing data (Landsat, Sentinel).
*   **Machine Learning:** Large-scale training of ML models, using datasets such as Common Crawl (for NLP) or satellite imagery datasets.
*   **Cloud Data Analysis:** Running complex queries and Big Data processing using AWS services, such as Amazon Athena to query data in S3.
*   **Application Development:** Building applications that consume public data in real time or near real time, such as weather forecasting or environmental monitoring apps.

## Integration

Primary integration is done through direct access to public Amazon S3 buckets, using native AWS tools or SDKs.

**A. Access via AWS CLI (Command Line Interface):**
```bash
# Listar o conteúdo do bucket Common Crawl
aws s3 ls s3://commoncrawl/
```

**B. Access via Python (Boto3 - AWS SDK):**
```python
import boto3

# O acesso a buckets públicos não requer credenciais
s3 = boto3.client('s3')
bucket_name = 'commoncrawl'
key = 'crawl-data/CC-MAIN-2023-50/segments/1702130000000.12345/warc/CC-MAIN-2023-50-12345-warc-00000.warc.gz'

try:
    response = s3.get_object(Bucket=bucket_name, Key=key)
    file_content = response['Body'].read().decode('utf-8')
    print(f"Conteúdo do arquivo: {file_content[:500]}...")
except Exception as e:
    print(f"Erro ao acessar o S3: {e}")
```

**C. Query via Amazon Athena (SQL):**
To query structured data (e.g., Parquet) directly in S3 (requires prior configuration):
```sql
-- Exemplo conceitual de consulta a um dataset público
SELECT col1, COUNT(*)
FROM my_open_data
WHERE col2 > 100
GROUP BY 1;
```

## URL

https://registry.opendata.aws/
