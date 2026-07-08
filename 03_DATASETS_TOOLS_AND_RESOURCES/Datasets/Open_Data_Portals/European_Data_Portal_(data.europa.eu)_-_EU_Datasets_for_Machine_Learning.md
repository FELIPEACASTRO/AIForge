# European Data Portal (data.europa.eu) - EU Datasets for Machine Learning

## Description

The **European Data Portal (data.europa.eu)** is the official, centralized access point to open data published by the institutions, agencies, and other bodies of the European Union, as well as by national, regional, and local data portals of 35 countries. Its unique value proposition lies in facilitating the discovery and reuse of open data for innovation, including Machine Learning (ML) and Artificial Intelligence (AI) applications. The portal aims to foster a data community, promote data literacy through the **data.europa academy**, and provide tools for exploring, standardizing, and publishing data. It replaced the EU Open Data Portal, consolidating access to a vast collection of data resources.

## Statistics

The portal consolidates access to a vast collection of open data, with the following key metrics (as of November 2025):
*   **Datasets:** 1,973,135
*   **Catalogues:** 203
*   **Countries:** 35
*   **Resources related to "machine learning":** More than 6,800 (including 6,443 datasets).
*   **Publications:** 688

## Features

Programmatic access via search API and SPARQL; tools for exploring, standardizing, and publishing data; data.europa academy with free courses for data literacy; publication of Open Data Maturity reports; advanced filters for searching datasets by theme, source, format, and metadata quality.

## Use Cases

The EU datasets available on the portal are used in various Machine Learning and Artificial Intelligence applications, including:
*   **Fake News Detection (Fake News Shield):** Using ML to verify the credibility of news sources and detect disinformation.
*   **Environmental and Deforestation Monitoring (Digital Dryads):** Analysis of aerial and multispectral satellite imagery to identify and protect forests against illegal deforestation, a task that benefits from computer vision models.
*   **Biodiversity Assessments:** Applying Machine Learning algorithms (such as `randomForest` in R) to forest inventory data to improve biodiversity assessments in Central Europe.
*   **Coaching and Optimization Systems:** Using user feedback data to improve the effectiveness of coaching systems, such as in the SAAM Sleep Coaching project, using preference learning.

## Integration

The portal offers programmatic access to its data and metadata through an HTTP GET-based search API, which returns results in JSON format.

**Integration Example (Python with `requests`):**
Searching for resources related to "machine learning" can be performed directly on the portal's search API.

```python
import requests
import json

# URL da API de busca para "machine learning"
api_url = "https://data.europa.eu/api/hub/search/search?q=machine%20learning&filters=catalogue,dataset,resource&resource=editorial-content"

try:
    response = requests.get(api_url)
    response.raise_for_status() # Levanta um erro para códigos de status HTTP ruins
    data = response.json()

    # Exemplo de extração de estatísticas
    total_resources = data.get('result', {}).get('count', {}).get('total', 0)
    datasets_count = data.get('result', {}).get('count', {}).get('dataset', 0)

    print(f"Total de recursos encontrados para 'machine learning': {total_resources}")
    print(f"Total de conjuntos de dados (datasets): {datasets_count}")

except requests.exceptions.RequestException as e:
    print(f"Erro ao acessar a API: {e}")
except json.JSONDecodeError:
    print("Erro ao decodificar a resposta JSON.")
```

## URL

https://data.europa.eu/en
