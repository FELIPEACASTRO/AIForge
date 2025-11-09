# AWS SageMaker

## Description

O **Amazon SageMaker** é uma plataforma de aprendizado de máquina (ML) totalmente gerenciada que permite a cientistas de dados e desenvolvedores construir, treinar e implantar modelos de ML em escala de forma rápida e fácil. Sua proposta de valor única reside em unificar todo o ciclo de vida do ML (MLOps) em um único ambiente integrado, o SageMaker Studio, simplificando o processo de experimentação à produção. A plataforma elimina a complexidade de gerenciar a infraestrutura subjacente, permitindo que as equipes se concentrem na inovação e na resolução de problemas de negócio. A nova geração do SageMaker, com o **Unified Studio**, oferece uma experiência integrada para análise e IA com acesso unificado a todos os dados, construído sobre o Amazon DataZone [1] [2].

## Statistics

**Adoção e Escala:** Em 2022, o AWS SageMaker contava com mais de **100.000 clientes** de praticamente todos os setores, com **milhões de modelos criados** e modelos treinados com **bilhões de parâmetros** [8].
**Otimização de Custos:** Clientes relatam uma redução média de **50% nos custos de implantação de modelos** ao usar recursos avançados como o Multi-Model Endpoints e o SageMaker Inference Recommender [9]. O uso de **SageMaker Savings Plans** pode reduzir os custos de treinamento e inferência em **50% ou mais** [10].
**Eficiência:** Empresas como a **bp** e a **ENGIE Digital** utilizam o SageMaker para escalar suas operações de ML, com a bp usando-o para construir, treinar e implantar modelos de ML como parte de sua solução de infraestrutura como código [11] [12].

## Features

**SageMaker Studio:** Ambiente de desenvolvimento unificado e baseado em web para todo o fluxo de trabalho de ML.
**SageMaker JumpStart:** Hub de ML com modelos pré-treinados, notebooks de solução e algoritmos para implantação rápida.
**SageMaker Autopilot:** Criação automática de modelos de ML de alta qualidade, sem necessidade de conhecimento em codificação.
**SageMaker Feature Store:** Repositório centralizado para armazenar, descobrir e compartilhar recursos de ML para treinamento e inferência.
**SageMaker Clarify:** Ajuda a detectar viés em modelos e a fornecer explicações sobre as previsões.
**SageMaker Pipelines:** Ferramenta de MLOps para criar, gerenciar e automatizar fluxos de trabalho de ML de ponta a ponta.
**SageMaker Inference:** Oferece diversas opções de implantação (Real-time, Serverless, Assíncrona e Batch) para otimizar custos e desempenho [3] [4].

## Use Cases

**Manutenção Preditiva:** Empresas de energia (como a ENGIE Digital) usam o SageMaker para prever falhas em equipamentos de usinas, otimizando a manutenção e reduzindo o tempo de inatividade [12].
**Análise de Imagem e Visão Computacional:** Classificação de imagens, detecção de objetos e segmentação semântica para controle de qualidade em manufatura ou análise de imagens médicas.
**Processamento de Linguagem Natural (NLP):** Construção de chatbots, análise de sentimentos, sumarização de texto e tradução.
**Sistemas de Recomendação:** Criação de modelos que sugerem produtos, filmes ou conteúdo personalizado para usuários em plataformas de e-commerce e streaming.
**Previsão de Séries Temporais:** Previsão de demanda de produtos, preços de ações ou consumo de energia [3].
**ML No-Code:** O SageMaker Canvas permite que analistas de negócios criem modelos de ML de alta precisão sem escrever código, democratizando o acesso ao ML [13].

## Integration

A integração com o SageMaker é primariamente realizada através do **AWS SDK para Python (Boto3)** ou do **SageMaker Python SDK**, que oferece uma interface de alto nível e orientada a objetos para interagir com os recursos do SageMaker [5].

**Exemplo de Treinamento e Implantação (SageMaker Python SDK):**

```python
import sagemaker
from sagemaker.tensorflow import TensorFlow

# 1. Configurar a sessão e o bucket S3
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# 2. Definir o Estimador (para treinamento)
# O script 'entry_point' contém o código de treinamento do modelo
estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='2.11',
    py_version='py39',
    sagemaker_session=sagemaker_session
)

# 3. Iniciar o job de treinamento
# 'inputs' aponta para a localização dos dados no S3
estimator.fit({'training': 's3://seu-bucket/seus-dados/'})

# 4. Implantação do modelo em um endpoint de inferência em tempo real
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# 5. Fazer uma previsão
# response = predictor.predict(data)

# 6. Limpar o endpoint (muito importante para evitar custos)
# predictor.delete_endpoint()
```
O SDK simplifica o provisionamento de recursos, o upload de dados para o S3 e a criação de jobs de treinamento e endpoints de inferência [6] [7].

## URL

https://aws.amazon.com/sagemaker/