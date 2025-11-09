# TensorFlow Serving

## Description

O TensorFlow Serving é um sistema de serviço de alto desempenho e flexível, projetado especificamente para ambientes de produção de aprendizado de máquina. Ele permite a implantação de modelos treinados de forma segura e eficiente, facilitando a transição do treinamento para a inferência em tempo real. Sua principal proposta de valor reside na capacidade de gerenciar o ciclo de vida do modelo, suportar o serviço de múltiplas versões simultaneamente e fornecer uma arquitetura robusta para escalabilidade e monitoramento.

## Statistics

**Performance:** Projetado para lidar com milhões de solicitações por segundo (QPS) em escala de produção. **Adoção:** Amplamente utilizado por grandes empresas de tecnologia e startups para servir modelos de ML em escala. **Métricas:** Suporta mais de 50 métricas padrão para regressão, classificação binária e multiclasse/multilabel, essenciais para o monitoramento de modelos em tempo real. **Latência:** Otimizado para baixa latência, crucial para aplicações de inferência em tempo real. **Otimização:** Otimizado para tempos de execução do TensorFlow, como o tempo de execução otimizado do Vertex AI, para inferência mais rápida e de menor custo.

## Features

**Arquitetura Flexível e de Alto Desempenho:** Projetado para ambientes de produção, suportando alta taxa de transferência e baixa latência. **Gerenciamento de Versões de Modelo:** Permite o carregamento, descarregamento e alternância segura entre diferentes versões de um modelo, facilitando o rollback e o teste A/B. **Suporte a Múltiplos Modelos:** Capacidade de servir múltiplos modelos e subtarefas simultaneamente. **APIs Padrão:** Oferece APIs de inferência via gRPC e RESTful, permitindo a integração com diversas plataformas e linguagens. **Batching Otimizado:** Inclui opções de loteamento (batching) para otimizar a utilização de hardware e aumentar a taxa de transferência. **Monitoramento e Métricas:** Integração com sistemas de monitoramento (como Prometheus/Grafana) para rastrear métricas de desempenho e saúde do modelo.

## Use Cases

**Sistemas de Recomendação:** Servir modelos de recomendação em tempo real para sugestões de produtos ou conteúdo. **Detecção de Objetos e Visão Computacional:** Implantar modelos de detecção de objetos para inferência em aplicativos móveis (Android/iOS) ou serviços de backend. **Processamento de Linguagem Natural (NLP):** Servir modelos de tradução, análise de sentimento ou classificação de texto em APIs de serviço. **Previsão Financeira e de Séries Temporais:** Implantar modelos para previsões de mercado ou demanda. **Testes A/B e Rollout Gradual:** Utilizar o gerenciamento de versões para testar novas versões de modelos com um subconjunto de usuários antes do rollout completo.

## Integration

A integração com o TensorFlow Serving é tipicamente realizada através de chamadas de API gRPC ou RESTful. O modelo treinado é exportado no formato `SavedModel` e carregado pelo servidor.

**Exemplo de Exportação de Modelo (Python):**
```python
import tensorflow as tf
import os

# Supondo que 'model' é o seu modelo treinado
export_path = os.path.join('/tmp/tf_serving_model', '1') # '1' é a versão
tf.saved_model.save(model, export_path)
print(f"Modelo exportado para: {export_path}")
```

**Exemplo de Chamada de Inferência (RESTful - Shell):**
```bash
# O servidor TF Serving deve estar rodando na porta 8501
curl -d '{"instances": [[1.0, 2.0, 5.0]]}' \
    -X POST http://localhost:8501/v1/models/my_model:predict
```

**Exemplo de Chamada de Inferência (gRPC - Python):**
A integração gRPC requer a instalação do cliente TensorFlow Serving e a geração de stubs.

```python
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# ... código de configuração do stub gRPC ...

channel = implementations.insecure_channel('localhost', 8500)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.model_spec.signature_name = 'serving_default'
request.inputs['input_name'].CopyFrom(
    tf.make_tensor_proto(data_point, dtype=tf.float32))

result = stub.Predict(request, timeout=10.0)
# Processar o resultado
```

## URL

https://www.tensorflow.org/tfx/guide/serving