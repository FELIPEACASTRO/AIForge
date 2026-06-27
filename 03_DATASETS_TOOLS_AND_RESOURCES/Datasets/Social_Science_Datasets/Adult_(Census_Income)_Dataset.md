# Adult (Census Income) Dataset

## Description
O dataset Adult, também conhecido como Census Income Dataset, é um conjunto de dados clássico extraído do banco de dados do Censo dos EUA de 1994. O objetivo principal é prever se a renda anual de um indivíduo excede $50.000 por ano com base em 14 atributos demográficos e de emprego. É amplamente utilizado em tarefas de classificação e em pesquisas sobre aprendizado de máquina justo (Fair Machine Learning) devido à sua inclusão de atributos sensíveis como raça e sexo. Embora o nome da tarefa fosse "Employment Data", a pesquisa identificou que o dataset mais relevante e popular nesse contexto, especialmente em Machine Learning, é o "Adult (Census Income)".

## Statistics
O dataset contém 48.842 instâncias (amostras) e 14 atributos (características). O arquivo de dados compactado tem um tamanho de 605.7 KB. A versão original foi extraída do Censo de 1994.

## Features
O dataset possui 14 atributos, incluindo: idade (contínua), classe de trabalho (categórica), escolaridade (categórica e numérica), estado civil (categórica), ocupação (categórica), relacionamento (categórica), raça (categórica), sexo (binário), ganho de capital (contínuo), perda de capital (contínuo), horas por semana (contínuo) e país nativo (categórico). A variável alvo é binária: se a renda excede $50K/ano ou não. Contém valores ausentes.

## Use Cases
- **Classificação Binária:** Previsão de renda (>50K ou <=50K).
- **Aprendizado de Máquina Justo (Fair ML):** Avaliação de viés e discriminação em modelos de IA, utilizando atributos sensíveis como raça e sexo.
- **Análise de Dados:** Estudo de fatores demográficos e de emprego que influenciam a renda.
- **Testes de Algoritmos:** Benchmark para novos algoritmos de classificação.

## Integration
O dataset pode ser baixado diretamente do repositório UCI Machine Learning. Para usuários de Python, a integração mais recomendada é através do pacote `ucimlrepo`.
1.  **Instalação:** `pip install ucimlrepo`
2.  **Uso em Python:**
    ```python
    from ucimlrepo import fetch_ucirepo 
    
    # Busca o dataset Adult (ID=2)
    adult = fetch_ucirepo(id=2) 
    
    # Dados (como dataframes pandas)
    X = adult.data.features 
    y = adult.data.targets 
    
    # Metadados e informações de variáveis também estão disponíveis
    # print(adult.metadata) 
    # print(adult.variables)
    ```

## URL
[https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)
