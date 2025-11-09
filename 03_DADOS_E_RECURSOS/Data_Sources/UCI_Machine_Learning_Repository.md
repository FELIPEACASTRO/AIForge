# UCI Machine Learning Repository

## Description

O **UCI Machine Learning Repository** é uma coleção de bases de dados, teorias de domínio e geradores de dados mantida pela Universidade da Califórnia, Irvine (UCI). É um dos repositórios de dados mais antigos e populares, servindo como um recurso fundamental para a comunidade de aprendizado de máquina para a análise empírica de algoritmos de ML. Seu valor único reside em fornecer um catálogo padronizado e diversificado de conjuntos de dados do mundo real, essenciais para o **benchmarking** e a **validação** de novos modelos e técnicas. A recente introdução da biblioteca oficial `ucimlrepo` simplifica drasticamente o acesso programático aos dados e metadados, tornando-o mais acessível para pesquisa e desenvolvimento moderno. (English: The **UCI Machine Learning Repository** is a collection of databases, domain theories, and data generators maintained by the University of California, Irvine (UCI). It is one of the oldest and most popular data repositories, serving as a fundamental resource for the machine learning community for the empirical analysis of ML algorithms. Its unique value lies in providing a standardized and diverse catalog of real-world datasets, essential for **benchmarking** and **validation** of new models and techniques. The recent introduction of the official `ucimlrepo` library drastically simplifies programmatic access to data and metadata, making it more accessible for modern research and development.)

## Statistics

**Total de Datasets:** Mais de 688 conjuntos de dados (em 2025). **Popularidade:** Um dos repositórios mais citados em artigos de pesquisa de ML. **Tipos de Tarefa:** Predominantemente Classificação (mais de 400), seguido por Regressão, Agrupamento e Outros. **Domínios:** Abrange uma ampla gama de domínios, incluindo Ciências da Vida, Negócios, Engenharia e Ciências Sociais. **Formato de Dados:** Principalmente dados tabulares (CSV, texto delimitado), com a maioria dos atributos sendo numéricos ou categóricos. (English: **Total Datasets:** Over 688 datasets (as of 2025). **Popularity:** One of the most cited repositories in ML research papers. **Task Types:** Predominantly Classification (over 400), followed by Regression, Clustering, and Others. **Domains:** Covers a wide range of domains, including Life Sciences, Business, Engineering, and Social Sciences. **Data Format:** Primarily tabular data (CSV, delimited text), with most attributes being numerical or categorical.)

## Features

**Catálogo Abrangente e Diversificado:** Mais de 688 conjuntos de dados (em 2025) cobrindo tarefas como Classificação, Regressão, Agrupamento e Séries Temporais. **Metadados Ricos:** Cada conjunto de dados é acompanhado por metadados detalhados, incluindo número de instâncias, atributos, tipos de atributos e informações de domínio. **Acesso Programático Simplificado:** A biblioteca oficial `ucimlrepo` (Python/R) permite a busca, importação e manipulação direta de dados e metadados em ambientes de notebook. **Formato Padronizado:** Os dados são frequentemente fornecidos em formatos simples (como CSV ou arquivos de texto delimitados), facilitando o carregamento em diversas plataformas. (English: **Comprehensive and Diverse Catalog:** Over 688 datasets (as of 2025) covering tasks like Classification, Regression, Clustering, and Time Series. **Rich Metadata:** Each dataset is accompanied by detailed metadata, including the number of instances, attributes, attribute types, and domain information. **Simplified Programmatic Access:** The official `ucimlrepo` library (Python/R) allows for direct search, import, and manipulation of data and metadata in notebook environments. **Standardized Format:** Data is often provided in simple formats (like CSV or delimited text files), facilitating loading into various platforms.)

## Use Cases

**Benchmarking de Algoritmos:** O caso de uso principal é fornecer um conjunto padronizado de dados para testar e comparar o desempenho de novos algoritmos de aprendizado de máquina. **Educação e Treinamento:** Amplamente utilizado em cursos universitários e tutoriais para ensinar os fundamentos da ciência de dados e ML, devido ao tamanho gerenciável e à documentação clara dos datasets. **Projetos de Prova de Conceito (PoC):** Ideal para prototipagem rápida e validação inicial de ideias de modelos antes de escalar para conjuntos de dados maiores e mais complexos. **Pesquisa em Domínios Específicos:** Datasets como o "Wine Quality" ou "Adult Income" são usados para pesquisa aplicada em áreas como química, sociologia e finanças. (English: **Algorithm Benchmarking:** The primary use case is to provide a standardized set of data to test and compare the performance of new machine learning algorithms. **Education and Training:** Widely used in university courses and tutorials to teach the fundamentals of data science and ML, due to the manageable size and clear documentation of the datasets. **Proof-of-Concept (PoC) Projects:** Ideal for rapid prototyping and initial validation of model ideas before scaling to larger, more complex datasets. **Research in Specific Domains:** Datasets like "Wine Quality" or "Adult Income" are used for applied research in areas such as chemistry, sociology, and finance.)

## Integration

A integração moderna é feita primariamente através da biblioteca oficial **`ucimlrepo`** (Python), que permite buscar e carregar conjuntos de dados diretamente em *dataframes* do Pandas, incluindo dados e metadados.

**Exemplo de Integração Python (`ucimlrepo`):**

```python
# Instalação (se necessário): pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# 1. Buscar um dataset pelo ID (ex: Iris, ID=53)
iris = fetch_ucirepo(id=53)

# 2. Acessar os dados (features e target) como DataFrames do Pandas
X = iris.data.features
y = iris.data.targets

# 3. Acessar os metadados
print(iris.metadata)
print(iris.variables)

# Exemplo de uso:
# print(X.head())
# print(y.head())
```

**Melhores Práticas:**
1.  **Priorizar `ucimlrepo`:** Use a biblioteca oficial para garantir o acesso aos metadados corretos e a estrutura de dados mais limpa.
2.  **Revisar Metadados:** Sempre inspecione `iris.metadata` e `iris.variables` para entender o contexto, o tipo de tarefa (classificação/regressão) e a descrição dos atributos antes de pré-processar.
3.  **Limpeza Manual:** Para datasets mais antigos, pode ser necessário realizar etapas adicionais de limpeza, como tratar valores ausentes ou converter tipos de dados, mesmo após o carregamento. (English: Modern integration is primarily done through the official **`ucimlrepo`** library (Python), which allows searching and loading datasets directly into Pandas dataframes, including data and metadata. **Python Integration Example (`ucimlrepo`):** [Code example as above]. **Best Practices:** 1. **Prioritize `ucimlrepo`:** Use the official library to ensure access to correct metadata and the cleanest data structure. 2. **Review Metadata:** Always inspect `iris.metadata` and `iris.variables` to understand the context, task type (classification/regression), and attribute description before preprocessing. 3. **Manual Cleaning:** For older datasets, additional cleaning steps may be necessary, such as handling missing values or converting data types, even after loading.)

## URL

https://archive.ics.uci.edu/