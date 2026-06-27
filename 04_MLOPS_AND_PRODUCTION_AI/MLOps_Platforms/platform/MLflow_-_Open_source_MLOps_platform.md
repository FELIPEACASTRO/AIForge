# MLflow - Open source MLOps platform

## Description

**MLflow** é uma plataforma de código aberto projetada para gerenciar o ciclo de vida completo do aprendizado de máquina (ML), desde a experimentação até a implantação. Sua proposta de valor única é fornecer uma **solução unificada** para MLOps, abrangendo Rastreamento de Experimentos, Empacotamento de Código, Gerenciamento de Modelos e Registro Centralizado de Modelos. Recentemente, expandiu-se para incluir suporte robusto para **aplicações de IA Generativa (GenAI)** e **LLMs (Large Language Models)**, oferecendo ferramentas para avaliação e rastreamento de LLMs, consolidando-se como a base para MLOps em escala.

## Statistics

**Downloads Mensais:** Mais de 30 Milhões (Estatística de 2025); **Estrelas no GitHub:** Mais de 19.000; **Contribuidores:** Mais de 850 desenvolvedores; **Empresas Usuárias Notáveis:** BNP Paribas, Thales, Unilever France, Carmax, DoorDash, Walmart, Oracle.

## Features

MLflow Tracking (API e UI para registrar parâmetros, métricas e artefatos); MLflow Projects (formato padrão para empacotar código de ciência de dados, garantindo reprodutibilidade); MLflow Models (convenção para empacotar modelos para implantação em diversas plataformas); MLflow Model Registry (repositório centralizado para gerenciar modelos, versões e estágios de transição); Rastreamento de Métricas do Sistema (registro de estatísticas de CPU, GPU, memória); Avaliação de LLM (API simples para avaliar LLMs com métricas personalizadas).

## Use Cases

**Rastreamento de Experimentos:** Comparação e otimização de hiperparâmetros em centenas de execuções de modelos; **MLOps em Escala:** Gerenciamento do ciclo de vida de modelos em produção, desde o treinamento até a implantação e monitoramento; **Reprodutibilidade:** Garantia de que qualquer membro da equipe possa reproduzir os resultados de um experimento anterior; **Implantação Multi-Plataforma:** Empacotamento de um modelo para implantação em diversos ambientes (Docker, Azure ML, AWS SageMaker, Databricks); **Aplicações de LLM:** Avaliação e monitoramento do desempenho de modelos de linguagem.

## Integration

A integração principal é feita através da API Python, compatível com a maioria dos frameworks de ML (Scikit-learn, TensorFlow, PyTorch).

**Exemplo de Integração (Python - MLflow Tracking):**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Inicia uma nova execução do MLflow
with mlflow.start_run():
    # Define e registra parâmetros
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)

    # Simulação de treinamento e métrica
    mse = 0.55 # Valor simulado
    
    # Registra a métrica
    mlflow.log_metric("mse", mse)

    # Registra o modelo (descomentar para uso real)
    # mlflow.sklearn.log_model(rf, "random_forest_model")
    
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
```

## URL

http://mlflow.org/