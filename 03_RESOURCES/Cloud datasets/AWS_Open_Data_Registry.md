# AWS Open Data Registry

## Description

O **Registry of Open Data on AWS** (Registro de Dados Abertos na AWS) é um catálogo centralizado que facilita a descoberta e o acesso a **datasets públicos de alto valor e otimizados para a nuvem** que estão disponíveis por meio de recursos da AWS, como o Amazon S3. Sua proposta de valor única reside na **otimização para a nuvem**, com dados armazenados em formatos que permitem a análise direta (ex: Parquet, Zarr), e no **AWS Open Data Sponsorship Program**, que cobre os custos de armazenamento e transferência de dados (egress) para os provedores, tornando o acesso gratuito ao público. Recentemente, todos os datasets do Registry se tornaram **descobertos no AWS Data Exchange**, unificando a busca por dados abertos, gratuitos e comerciais.

## Statistics

*   **Volume de Dados:** Mais de **300 Petabytes (PB)** de dados de alto valor e otimizados para a nuvem são disponibilizados através do programa.
*   **Número de Datasets:** O catálogo lista **mais de 800 datasets** (a página principal indica "atualmente 818 datasets correspondentes").
*   **Provedores:** Inclui dados de organizações como NASA, NOAA, NIH, Allen Institute for AI (AI2), e Common Crawl.

## Features

*   **Descoberta Centralizada:** Interface de busca e navegação para encontrar datasets por palavra-chave, categoria (ex: genômica, clima, transporte) e provedor.
*   **Acesso Direto ao S3:** Os dados são acessíveis diretamente de buckets públicos do Amazon S3, permitindo o uso de serviços de análise da AWS (como Amazon Athena, Amazon EMR, Amazon SageMaker) sem a necessidade de mover os dados.
*   **Exemplos de Uso:** Fornece tutoriais e exemplos de uso (Usage Examples), incluindo notebooks do SageMaker Studio Lab, para acelerar o início dos projetos.
*   **Metadados Ricos:** Cada dataset possui uma página de detalhes com metadados, descrição, tags e informações de licenciamento.

## Use Cases

*   **Pesquisa Científica:** Análise de dados genômicos (The Cancer Genome Atlas - TCGA), dados climáticos (NOAA), e dados de sensoriamento remoto (Landsat, Sentinel).
*   **Aprendizado de Máquina (Machine Learning):** Treinamento de modelos de ML em grande escala, utilizando datasets como Common Crawl (para NLP) ou datasets de imagens de satélite.
*   **Análise de Dados em Nuvem:** Execução de consultas complexas e processamento de Big Data usando serviços da AWS, como o Amazon Athena para consultar dados em S3.
*   **Desenvolvimento de Aplicações:** Criação de aplicações que consomem dados públicos em tempo real ou quase real, como aplicativos de previsão do tempo ou monitoramento ambiental.

## Integration

A integração primária é feita através do acesso direto aos buckets públicos do Amazon S3, utilizando ferramentas nativas da AWS ou SDKs.

**A. Acesso via AWS CLI (Interface de Linha de Comando):**
```bash
# Listar o conteúdo do bucket Common Crawl
aws s3 ls s3://commoncrawl/
```

**B. Acesso via Python (Boto3 - SDK da AWS):**
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

**C. Consulta via Amazon Athena (SQL):**
Para consultar dados estruturados (ex: Parquet) diretamente no S3 (requer configuração prévia):
```sql
-- Exemplo conceitual de consulta a um dataset público
SELECT col1, COUNT(*)
FROM my_open_data
WHERE col2 > 100
GROUP BY 1;
```

## URL

https://registry.opendata.aws/