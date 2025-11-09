# ETL Tools for ML - Apache Spark, Dask, Pandas

## Description

**Apache Spark** é um motor de análise unificado e de código aberto para processamento de dados em larga escala, com módulos integrados para SQL, streaming, aprendizado de máquina (MLlib) e processamento de grafos (GraphX). Sua proposta de valor única reside na sua capacidade de processamento em memória, que o torna significativamente mais rápido que o Hadoop MapReduce para cargas de trabalho iterativas e interativas. Ele é projetado para ser um motor de análise multi-linguagem (Scala, Java, Python, R, SQL) que pode ser executado em máquinas de nó único ou em clusters, tornando-o uma plataforma robusta e escalável para engenharia de dados, ciência de dados e ML em escala massiva.

**Dask** é uma biblioteca Python de código aberto para computação paralela e distribuída. Sua proposta de valor única é a capacidade de escalar o ecossistema Python existente (como NumPy, Pandas e Scikit-learn) para conjuntos de dados maiores que a memória RAM, sem a necessidade de reescrever o código. Ele faz isso dividindo grandes estruturas de dados (como DataFrames ou Arrays) em coleções de blocos menores, que podem ser processados em paralelo em um único laptop ou em um cluster distribuído. É a solução ideal para cientistas de dados que desejam escalar seu código Python familiar.

**Pandas** é uma biblioteca Python de código aberto que fornece estruturas de dados de alto desempenho e fáceis de usar e ferramentas de análise de dados. Sua proposta de valor única é a introdução do `DataFrame`, uma estrutura de dados bidimensional rotulada com colunas de tipos potencialmente diferentes, que é o padrão de fato para manipulação e análise de dados em Python. Ele é otimizado para análise de dados em um único nó (in-memory) e é a base para a maioria dos fluxos de trabalho de ciência de dados em Python, permitindo a limpeza, transformação, exploração e manipulação rápida de dados.

## Statistics

**Apache Spark:**
*   **Velocidade:** Até 100x mais rápido que o Hadoop MapReduce para processamento em memória.
*   **Ecossistema:** Mais de 80% das empresas Fortune 500 usam Spark.
*   **Linguagens:** Suporte nativo para Scala, Java, Python, R e SQL.
*   **Comunidade:** Um dos projetos de código aberto mais ativos na área de Big Data.

**Dask:**
*   **Escalabilidade:** Permite processar DataFrames e Arrays que excedem a memória RAM de um único nó.
*   **Integração:** Projetado para ser 100% compatível com as APIs do NumPy e Pandas.
*   **Flexibilidade:** Pode ser executado em laptops, clusters HPC, e nuvens (AWS, GCP, Azure).

**Pandas:**
*   **Padrão da Indústria:** A biblioteca de fato para manipulação de dados em Python.
*   **Performance:** Altamente otimizado para operações em memória, com base em NumPy.
*   **Comunidade:** Vasta documentação, tutoriais e uma comunidade de usuários massiva.
*   **Limitação:** Projetado para dados que cabem na memória de um único computador.

## Features

**Apache Spark:**
*   **Processamento em Memória:** Utiliza cache em memória para acelerar cargas de trabalho iterativas.
*   **APIs Unificadas:** Suporta Spark SQL, Spark Streaming, MLlib (Machine Learning) e GraphX (Processamento de Grafos).
*   **Suporte Multi-Linguagem:** APIs em Scala, Java, Python (PySpark), R e SQL.
*   **Conectividade Ampla:** Pode ser executado em Hadoop YARN, Apache Mesos, Kubernetes ou de forma autônoma.

**Dask:**
*   **Paralelização de Bibliotecas Python:** Estende NumPy, Pandas e Scikit-learn para computação paralela.
*   **Estruturas de Dados Paralelas:** Oferece `Dask Array`, `Dask DataFrame` e `Dask Bag` para lidar com dados maiores que a memória.
*   **Agendador Dinâmico de Tarefas:** Otimiza a execução de gráficos de tarefas complexos em paralelo.
*   **Dashboard de Monitoramento:** Fornece métricas de desempenho detalhadas em tempo real.

**Pandas:**
*   **Estrutura de Dados DataFrame:** Estrutura de dados tabular rotulada e poderosa.
*   **Manipulação de Dados:** Funções ricas para indexação, fatiamento, agrupamento, junção e remodelação de dados.
*   **Limpeza de Dados:** Ferramentas robustas para lidar com dados ausentes (`NaN`), valores duplicados e transformações de tipo.
*   **Análise Estatística:** Funcionalidades para calcular estatísticas descritivas e aplicar funções arbitrárias.

## Use Cases

**Apache Spark:**
*   **Processamento de Logs em Tempo Real:** Análise de logs de websites e servidores para detecção de fraudes e monitoramento.
*   **Análise Genômica:** Processamento de grandes volumes de dados de sequenciamento de DNA.
*   **Sistemas de Recomendação:** Treinamento de modelos de filtragem colaborativa em grandes conjuntos de dados de usuários e itens (usando MLlib).
*   **ETL em Larga Escala:** Transformação e carregamento de petabytes de dados em data warehouses.

**Dask:**
*   **Escalando o Pandas:** Execução de operações de Pandas em conjuntos de dados maiores que a memória RAM de um único computador.
*   **Computação Científica:** Paralelização de cálculos complexos em Arrays NumPy (Dask Array) para simulações climáticas e astrofísicas.
*   **Processamento de Imagens Grandes:** Manipulação de imagens médicas ou de satélite de alta resolução.
*   **Treinamento de ML Distribuído:** Uso de Dask-ML para paralelizar o treinamento de modelos Scikit-learn.

**Pandas:**
*   **Análise Exploratória de Dados (EDA):** Limpeza, resumo e visualização inicial de conjuntos de dados.
*   **Preparação de Dados para ML:** Engenharia de features, tratamento de valores ausentes e normalização de dados para modelos.
*   **Análise Financeira:** Processamento de séries temporais e cálculo de métricas estatísticas.
*   **Relatórios e BI:** Geração de relatórios e painéis a partir de dados estruturados.

## Integration

**Apache Spark:**
A integração com Python é feita via **PySpark**. O código a seguir demonstra a leitura de um arquivo CSV e a execução de uma operação de MLlib (exemplo conceitual):

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# 1. Inicializar a Spark Session
spark = SparkSession.builder.appName("SparkML_Example").getOrCreate()

# 2. Carregar Dados
data = spark.read.csv("caminho/para/dados.csv", header=True, inferSchema=True)

# 3. Preparação de Features (ETL)
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
data = assembler.transform(data)

# 4. Treinamento do Modelo
lr = LinearRegression(featuresCol="features", labelCol="target")
model = lr.fit(data)

# 5. Parar a Spark Session
spark.stop()
```

**Dask:**
A integração é nativa com o ecossistema Python. O código a seguir demonstra a criação de um `Dask DataFrame` e a execução de uma operação paralela:

```python
import dask.dataframe as dd
from dask.distributed import Client

# 1. Inicializar o Cliente Dask (opcional, mas recomendado para clusters)
client = Client(n_workers=4)

# 2. Criar um Dask DataFrame a partir de múltiplos arquivos CSV
ddf = dd.read_csv('caminho/para/arquivos/*.csv')

# 3. Executar uma operação paralela (ex: calcular a média de uma coluna)
media_coluna = ddf['coluna_numerica'].mean().compute()

# 4. Fechar o Cliente
client.close()
```

**Pandas:**
A integração é o padrão para a maioria das bibliotecas Python de ML (Scikit-learn, TensorFlow, PyTorch). O código a seguir demonstra a leitura de dados e a limpeza básica:

```python
import pandas as pd
import numpy as np

# 1. Ler Dados
df = pd.read_csv('caminho/para/dados.csv')

# 2. Limpeza de Dados (ETL)
# Preencher valores ausentes com a média
df['idade'].fillna(df['idade'].mean(), inplace=True)

# Remover duplicatas
df.drop_duplicates(inplace=True)

# 3. Preparação para ML (ex: One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['categoria'])

# O DataFrame 'df_encoded' está pronto para ser usado em um modelo de ML.
```

## URL

Apache Spark: https://spark.apache.org/ | Dask: https://www.dask.org/ | Pandas: https://pandas.pydata.org/