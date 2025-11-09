# Active Learning - Uncertainty Sampling

## Description

**Active Learning** é uma técnica de Machine Learning que visa otimizar o processo de rotulagem de dados, permitindo que o modelo selecione iterativamente os exemplos não rotulados mais informativos para serem anotados por um oráculo (geralmente um humano). O objetivo é alcançar alta precisão com um número significativamente menor de amostras rotuladas, reduzindo custos e tempo de desenvolvimento. A **Amostragem por Incerteza (Uncertainty Sampling)** é a estratégia de consulta mais popular dentro do Active Learning. Sua proposta de valor única reside na sua capacidade de identificar e priorizar amostras que estão próximas da fronteira de decisão do modelo, onde a classificação é mais ambígua. Ao focar nesses "casos difíceis", o modelo maximiza o ganho de informação a cada nova rotulagem, acelerando a convergência e melhorando a robustez do modelo com menos esforço de anotação.

## Statistics

O principal indicador de desempenho é a **eficiência de rotulagem**. Estudos demonstram que o Active Learning, utilizando Amostragem por Incerteza, pode alcançar a mesma precisão de um modelo treinado com 70% dos dados rotulados aleatoriamente, utilizando apenas **20%** das amostras rotuladas. Isso representa uma redução de **71%** no esforço de anotação. A métrica chave é a **Curva de Aprendizado**, que mostra a precisão do modelo em função do número de amostras rotuladas, onde uma curva de Active Learning bem-sucedida se eleva muito mais rapidamente do que a curva de amostragem aleatória. A eficácia é frequentemente medida pela **Área Sob a Curva de Aprendizado (AL-AUC)**.

## Features

A Amostragem por Incerteza se manifesta em três principais medidas de incerteza:
*   **Incerteza de Classificação (Least Confident)**: Seleciona a amostra para a qual a probabilidade da classe mais provável é a menor. A incerteza é calculada como $U(x) = 1 - P(\hat{x}|x)$, onde $\hat{x}$ é a classe mais provável.
*   **Margem de Classificação (Margin Sampling)**: Seleciona a amostra com a menor diferença de probabilidade entre as duas classes mais prováveis, $M(x) = P(\hat{x}_1|x) - P(\hat{x}_2|x)$. Uma margem pequena indica que o modelo está indeciso entre as duas melhores opções.
*   **Entropia de Classificação (Entropy Sampling)**: Seleciona a amostra com a maior entropia, $H(x) = -\sum_{k} p_k \log(p_k)$. A entropia mede a aleatoriedade da distribuição de probabilidade de todas as classes, sendo máxima quando a distribuição é uniforme (maior incerteza).

## Use Cases

*   **Processamento de Linguagem Natural (PLN)**: Rotulagem eficiente de grandes volumes de texto para tarefas como classificação de sentimentos, reconhecimento de entidades nomeadas (NER) e desambiguação de palavras. O modelo consulta frases onde a classificação é ambígua.
*   **Visão Computacional**: Anotação de imagens e vídeos para detecção de objetos e segmentação semântica. O Active Learning é usado para selecionar imagens que contêm exemplos de fronteira (por exemplo, objetos parcialmente ocluídos ou em ângulos incomuns).
*   **Diagnóstico Médico**: Redução do custo de rotulagem de imagens médicas (raios-X, ressonâncias magnéticas) por especialistas. O modelo prioriza casos onde o diagnóstico é menos claro, otimizando o tempo do médico.
*   **Detecção de Fraudes**: Identificação de transações financeiras que estão na "fronteira" entre fraude e não-fraude, permitindo que analistas humanos revisem apenas os casos mais ambíguos e informativos.

## Integration

A integração de Active Learning com Amostragem por Incerteza é facilitada por bibliotecas Python como **modAL** e **scikit-activeml**.

**Exemplo de Integração com modAL (Python):**

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

# 1. Preparação dos Dados
X, y = load_iris(return_X_y=True)
# Inicialmente, rotulamos apenas 10 amostras
n_initial = 10
initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
X_initial, y_initial = X[initial_idx], y[initial_idx]
X_pool = np.delete(X, initial_idx, axis=0)
y_pool = np.delete(y, initial_idx, axis=0)

# 2. Inicialização do Active Learner com Amostragem por Incerteza
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=uncertainty_sampling, # Estratégia de Amostragem por Incerteza
    X_training=X_initial, y_training=y_initial
)

# 3. Ciclo de Active Learning
n_queries = 20
for idx in range(n_queries):
    # O modelo consulta a amostra mais incerta
    query_idx, query_instance = learner.query(X_pool)

    # Simulação de rotulagem (o oráculo fornece o rótulo)
    X_new, y_new = X_pool[query_idx], y_pool[query_idx]

    # O modelo aprende com a nova amostra rotulada
    learner.teach(X_new, y_new)

    # Remove a amostra rotulada do pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    print(f"Iteração {idx+1}: Precisão atual = {learner.score(X, y):.4f}")
```

## URL

https://modal-python.readthedocs.io/en/latest/content/query_strategies/uncertainty_sampling.html