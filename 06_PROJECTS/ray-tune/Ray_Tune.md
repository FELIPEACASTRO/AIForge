# Ray Tune

## Description

Ray Tune é uma biblioteca Python para execução de experimentos e **otimização escalável de hiperparâmetros** em qualquer escala, construída sobre o framework de computação distribuída Ray. Sua proposta de valor única reside na capacidade de **paralelizar transparentemente** a busca de hiperparâmetros em múltiplas GPUs e nós, permitindo a escalabilidade de experimentos em até 100 vezes e a redução de custos em até 10 vezes através do uso de instâncias preemptivas. Ele oferece uma API unificada para integrar algoritmos de ponta e se conecta perfeitamente com os principais frameworks de Machine Learning (ML), removendo a complexidade do gerenciamento de sistemas distribuídos para otimização de ML.

## Statistics

**Escalabilidade e Desempenho**: Permite a escalabilidade de experimentos em até **100 vezes** mais rápido do que soluções de nó único. **Otimização de Custo**: Pode reduzir os custos de computação em até **10 vezes** ao utilizar instâncias preemptivas de baixo custo. **Adoção**: É um componente chave do ecossistema Ray, que possui mais de **35.000 estrelas** no GitHub, indicando uma forte adoção e comunidade. **Artigo Científico**: Baseado no artigo "Tune: A Research Platform for Distributed Model Selection and Training" (arXiv:1807.05118), publicado em 2018.

## Features

**Algoritmos de Otimização de Ponta**: Suporta algoritmos como Population Based Training (PBT) e HyperBand/ASHA, além de integrar-se com ferramentas externas como Ax, BayesOpt, BOHB, Nevergrad e Optuna. **Produtividade do Desenvolvedor**: Permite a otimização de modelos com a adição de apenas algumas linhas de código, suportando múltiplas opções de armazenamento para resultados de experimentos (NFS, cloud storage) e registro de logs em ferramentas como MLflow e TensorBoard. **Treinamento Distribuído e Multi-GPU**: Oferece paralelização transparente em múltiplas GPUs e múltiplos nós, com tolerância a falhas e suporte nativo a ambientes de nuvem. **Integração com Frameworks de ML**: Compatibilidade nativa com PyTorch, TensorFlow/Keras, XGBoost, scikit-learn e outros.

## Use Cases

**Otimização de Modelos de Linguagem Grande (LLMs)**: Usado para ajustar hiperparâmetros de modelos de transformadores e LLMs em ambientes distribuídos. **Visão Computacional**: Otimização de arquiteturas de redes neurais convolucionais (CNNs) para tarefas como classificação e segmentação de imagens. **Aprendizado por Reforço (RL)**: Utilizado em frameworks de RL como o RLlib (também parte do Ray) para otimizar políticas e algoritmos de RL. **Previsão de Séries Temporais**: Otimização de modelos complexos de previsão, como NeuroCard, para estimativa de cardinalidade em bancos de dados. **Pesquisa em ML**: Plataforma fundamental para pesquisadores que buscam comparar e desenvolver novos algoritmos de otimização de hiperparâmetros de forma eficiente e escalável.

## Integration

A integração com o Ray Tune é realizada definindo uma função de treinamento (objective function) que aceita um dicionário de configuração de hiperparâmetros e reporta métricas através da API `tune.report()`. O `tune.Tuner` é então usado para iniciar a busca, definindo o espaço de busca (`param_space`) e o algoritmo/agendador.

**Exemplo de Integração com PyTorch (Conceitual):**
```python
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig

def train_model(config):
    # 1. Carregar dados e modelo
    # 2. Definir hiperparâmetros a partir de 'config'
    lr = config["lr"]
    epochs = config["epochs"]
    
    for epoch in range(epochs):
        # Lógica de treinamento
        loss = ...
        accuracy = ...
        
        # Reportar métricas para o Ray Tune
        tune.report(loss=loss, accuracy=accuracy)

# Definir o espaço de busca
search_space = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "epochs": tune.choice([5, 10, 15]),
}

# Configurar o Tuner
tuner = Tuner(
    train_model,
    param_space=search_space,
    tune_config=TuneConfig(
        metric="accuracy",
        mode="max",
        num_samples=10, # Número de amostras a serem testadas
        scheduler=tune.schedulers.ASHAScheduler(), # Exemplo de agendador
    ),
    run_config=RunConfig(name="my_tune_experiment")
)

results = tuner.fit()
best_result = results.get_best_result()
print(f"Melhor Configuração: {best_result.config}")
```

## URL

https://docs.ray.io/en/latest/tune/index.html