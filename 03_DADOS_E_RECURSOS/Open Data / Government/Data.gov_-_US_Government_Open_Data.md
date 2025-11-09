# Data.gov - US Government Open Data

## Description

O Data.gov é o portal de dados abertos do Governo dos Estados Unidos, lançado em 2009 e gerenciado pela U.S. General Services Administration (GSA). Sua missão é liberar o poder dos dados abertos do governo para **informar decisões do público e de formuladores de políticas, impulsionar a inovação e a atividade econômica, alcançar as missões das agências e fortalecer a base de um governo aberto e transparente**. É o catálogo central para dados federais, estaduais e locais, servindo como um recurso fundamental para o desenvolvimento de modelos de Inteligência Artificial e Machine Learning (IA/ML) com base em informações governamentais verificadas.

**English:** Data.gov is the United States Government's open data portal, launched in 2009 and managed by the U.S. General Services Administration (GSA). Its mission is to unleash the power of government open data to **inform decisions by the public and policymakers, drive innovation and economic activity, achieve agency missions, and strengthen the foundation of an open and transparent government**. It serves as the central catalog for federal, state, and local data, making it a fundamental resource for developing Artificial Intelligence and Machine Learning (AI/ML) models based on verified government information.

## Statistics

**Português:**
*   **Total de Conjuntos de Dados:** Mais de **364.000** conjuntos de dados (dado de 2025).
*   **Visualizações de Página (Mensal):** Aproximadamente **1,95 milhão** de visualizações de página (dados de Outubro de 2025).
*   **Principais Contribuintes:** U.S. Census Bureau (mais de 111 mil conjuntos de dados), NOAA (mais de 102 mil conjuntos de dados).
*   **Tecnologia:** Alimentado por duas aplicações de código aberto: CKAN e 11ty.

**English:**
*   **Total Datasets:** Over **364,000** datasets (2025 data).
*   **Monthly Pageviews:** Approximately **1.95 million** pageviews (October 2025 data).
*   **Top Contributors:** U.S. Census Bureau (over 111k datasets), NOAA (over 102k datasets).
*   **Technology:** Powered by two open-source applications: CKAN and 11ty.

## Features

**Português:**
*   **Catálogo Centralizado:** Agrega dados de mais de 170 agências federais, estaduais e locais.
*   **Base CKAN:** Utiliza a plataforma de código aberto CKAN (Comprehensive Knowledge Archive Network) para gerenciamento de dados e funcionalidade de API.
*   **Dados Legíveis por Máquina:** Foco na disponibilização de dados em formatos abertos e legíveis por máquina, conforme exigido pela Lei OPEN Government Data.
*   **Pesquisa e Filtragem:** Ferramentas robustas para pesquisar e filtrar conjuntos de dados por tópico, formato, agência e tags (incluindo "artificial-intelligence" e "machine-learning").
*   **Métricas de Uso:** Painel de métricas que fornece insights sobre os conjuntos de dados mais visualizados e baixados.

**English:**
*   **Centralized Catalog:** Aggregates data from over 170 federal, state, and local agencies.
*   **CKAN Foundation:** Built on the open-source CKAN (Comprehensive Knowledge Archive Network) platform for data management and API functionality.
*   **Machine-Readable Data:** Focus on making data available in open, machine-readable formats, as required by the OPEN Government Data Act.
*   **Search and Filtering:** Robust tools to search and filter datasets by topic, format, agency, and tags (including "artificial-intelligence" and "machine-learning").
*   **Usage Metrics:** Metrics dashboard providing insights into the most viewed and downloaded datasets.

## Use Cases

**Português:**
*   **Treinamento de Modelos de IA/ML:** Utilização de grandes volumes de dados governamentais (ex: dados climáticos da NOAA, dados demográficos do Census Bureau) para treinar modelos de Machine Learning em tarefas de previsão e classificação.
*   **Análise de Políticas Públicas:** Modelos de IA para analisar o impacto de políticas de saúde (usando dados do HHS) ou ambientais (usando dados da EPA), permitindo a tomada de decisões baseada em evidências.
*   **Detecção de Anomalias e Fraudes:** Aplicação de ML em dados financeiros e de contratos governamentais para identificar padrões incomuns e prevenir fraudes.
*   **Desenvolvimento de Aplicações Cívicas:** Criação de aplicativos móveis e web que utilizam dados abertos (ex: dados de trânsito, segurança pública, qualidade do ar) para melhorar a vida dos cidadãos.
*   **Pesquisa Científica:** Uso de conjuntos de dados de alta qualidade (ex: dados da NASA e do Departamento de Energia) para pesquisa acadêmica e desenvolvimento de novas tecnologias.

**English:**
*   **AI/ML Model Training:** Utilizing large volumes of government data (e.g., NOAA climate data, Census Bureau demographic data) to train Machine Learning models for forecasting and classification tasks.
*   **Public Policy Analysis:** AI models to analyze the impact of health (using HHS data) or environmental policies (using EPA data), enabling evidence-based decision-making.
*   **Anomaly and Fraud Detection:** Applying ML to government financial and contract data to identify unusual patterns and prevent fraud.
*   **Civic Application Development:** Creating mobile and web applications that use open data (e.g., traffic data, public safety, air quality) to improve citizens' lives.
*   **Scientific Research:** Using high-quality datasets (e.g., NASA and Department of Energy data) for academic research and new technology development.

## Integration

**Português:**
A integração é primariamente feita através da **API CKAN** (Comprehensive Knowledge Archive Network), que é a plataforma de código aberto que alimenta o Data.gov. A API permite a pesquisa programática, recuperação de metadados e acesso direto aos recursos de dados.

**Exemplo de Código (Python para Pesquisa de Conjuntos de Dados):**
```python
import requests
import json

# URL base da API CKAN do Data.gov
CKAN_API_URL = "https://catalog.data.gov/api/3/action/"

def search_datasets(query):
    """
    Pesquisa conjuntos de dados no Data.gov usando a API CKAN.
    """
    action = "package_search"
    # Limita a 5 resultados para o exemplo
    params = {"q": query, "rows": 5} 
    
    try:
        response = requests.get(CKAN_API_URL + action, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data["success"]:
            print(f"Encontrados {data['result']['count']} conjuntos de dados para '{query}':\n")
            for pkg in data["result"]["results"]:
                print(f"  Título: {pkg['title']}")
                print(f"  Organização: {pkg['organization']['title'] if pkg.get('organization') else 'N/A'}")
                print(f"  URL: https://catalog.data.gov/dataset/{pkg['name']}")
                print("-" * 20)
        else:
            print("Falha na pesquisa da API:", data.get("error"))
            
    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar à API: {e}")

if __name__ == "__main__":
    # Exemplo de pesquisa por conjuntos de dados relacionados a "artificial intelligence"
    search_datasets("artificial intelligence")
```

**English:**
Integration is primarily done through the **CKAN API** (Comprehensive Knowledge Archive Network), which is the open-source platform powering Data.gov. The API allows for programmatic searching, metadata retrieval, and direct access to data resources.

**Code Example (Python for Dataset Search):**
```python
import requests
import json

# Base URL for the Data.gov CKAN API
CKAN_API_URL = "https://catalog.data.gov/api/3/action/"

def search_datasets(query):
    """
    Searches for datasets on Data.gov using the CKAN API.
    """
    action = "package_search"
    # Limits to 5 results for the example
    params = {"q": query, "rows": 5} 
    
    try:
        response = requests.get(CKAN_API_URL + action, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data["success"]:
            print(f"Found {data['result']['count']} datasets for '{query}':\n")
            for pkg in data["result"]["results"]:
                print(f"  Title: {pkg['title']}")
                print(f"  Organization: {pkg['organization']['title'] if pkg.get('organization') else 'N/A'}")
                print(f"  URL: https://catalog.data.gov/dataset/{pkg['name']}")
                print("-" * 20)
        else:
            print("API search failed:", data.get("error"))
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")

if __name__ == "__main__":
    # Example search for datasets related to "artificial intelligence"
    search_datasets("artificial intelligence")
```

## URL

https://data.gov/