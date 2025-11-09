# European Data Portal (data.europa.eu) - EU Datasets for Machine Learning

## Description

O **European Data Portal (data.europa.eu)** é o ponto de acesso oficial e centralizado aos dados abertos publicados pelas instituições, agências e outros órgãos da União Europeia, bem como por portais de dados nacionais, regionais e locais de 35 países. Sua proposta de valor única reside em facilitar a descoberta e o reuso de dados abertos para inovação, incluindo aplicações de Machine Learning (ML) e Inteligência Artificial (IA). O portal visa fomentar uma comunidade de dados, promover a alfabetização em dados através da **data.europa academy** e fornecer ferramentas para a exploração, padronização e publicação de dados. Ele substituiu o EU Open Data Portal, consolidando o acesso a uma vasta coleção de recursos de dados.

## Statistics

O portal consolida o acesso a uma vasta coleção de dados abertos, com as seguintes métricas chave (em Novembro de 2025):
*   **Conjuntos de Dados (Datasets):** 1.973.135
*   **Catálogos (Catalogues):** 203
*   **Países (Countries):** 35
*   **Recursos relacionados a "machine learning":** Mais de 6.800 (incluindo 6.443 conjuntos de dados).
*   **Publicações:** 688

## Features

Acesso programático via API de busca e SPARQL; Ferramentas de exploração, padronização e publicação de dados; data.europa academy com cursos gratuitos para alfabetização em dados; Publicação de relatórios de Maturidade de Dados Abertos (Open Data Maturity); Filtros avançados para pesquisa de conjuntos de dados por tema, origem, formato e qualidade de metadados.

## Use Cases

Os conjuntos de dados da UE disponíveis no portal são utilizados em diversas aplicações de Machine Learning e Inteligência Artificial, incluindo:
*   **Detecção de Notícias Falsas (Fake News Shield):** Uso de ML para verificar a credibilidade de fontes de notícias e detectar desinformação.
*   **Monitoramento Ambiental e Desmatamento (Digital Dryads):** Análise de imagens aéreas e de satélite multiespectral para identificar e proteger florestas contra desmatamento ilegal, uma tarefa que se beneficia de modelos de visão computacional.
*   **Avaliações de Biodiversidade:** Aplicação de algoritmos de Machine Learning (como `randomForest` em R) em dados de inventário florestal para aprimorar as avaliações de biodiversidade na Europa Central.
*   **Sistemas de Coaching e Otimização:** Uso de dados de feedback de usuários para melhorar a eficácia de sistemas de coaching, como no projeto SAAM Sleep Coaching, utilizando aprendizado por preferência.

## Integration

O portal oferece acesso programático aos seus dados e metadados através de uma API de busca baseada em HTTP GET, que retorna resultados em formato JSON.

**Exemplo de Integração (Python com `requests`):**
A busca por recursos relacionados a "machine learning" pode ser realizada diretamente na API de busca do portal.

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