# ONNX Runtime

## Description

O ONNX Runtime (ORT) é um motor de inferência e treinamento de aprendizado de máquina de alto desempenho e código aberto, projetado para acelerar a execução de modelos ONNX (Open Neural Network Exchange) em diversas plataformas de hardware e software. Sua proposta de valor única reside na **interoperabilidade e otimização de desempenho**, permitindo que modelos treinados em qualquer framework (como PyTorch, TensorFlow, Keras, scikit-learn) sejam exportados para o formato ONNX e executados de forma eficiente em ambientes de produção, desde a nuvem (Azure, AWS, GCP) até dispositivos de borda e móveis (iOS, Android, Web). O ORT aplica otimizações de grafo e utiliza provedores de execução (Execution Providers - EPs) específicos de hardware para maximizar a velocidade de inferência e treinamento, reduzindo a latência e aumentando a taxa de transferência.

## Statistics

**Aceleração de Desempenho:** Relatos de ganhos de desempenho de inferência de até **9x** em comparação com frameworks nativos em cenários específicos, como em serviços de modelo com alta taxa de transferência.
**Otimização de Latência:** Projetado para minimizar a latência de inferência, crucial para aplicações em tempo real.
**Uso em Produção:** Utilizado em produtos da Microsoft, incluindo Windows, Office, Azure Cognitive Services e Bing, demonstrando escalabilidade e confiabilidade de nível de produção.
**Flexibilidade de Hardware:** Suporta mais de 10 Provedores de Execução (EPs) diferentes para otimizar o desempenho em uma ampla gama de hardware.

## Features

**Aceleração Multi-Plataforma:** Suporte para Windows, Linux, macOS, iOS, Android e Web.
**Ampla Compatibilidade de Hardware:** Otimizado para CPU, GPU (CUDA, TensorRT, OpenVINO, ROCm), e NPUs (Qualcomm QNN).
**Provedores de Execução (EPs):** Interface flexível para integrar bibliotecas específicas de hardware, como TensorRT, OpenVINO, DirectML, e Core ML, para aceleração máxima.
**Otimização de Grafo:** Aplica transformações e fusões de nós para melhorar a eficiência do modelo antes da execução.
**Suporte a Treinamento:** Capacidade de treinamento de modelos grandes e treinamento no dispositivo (on-device training) para personalização e privacidade.
**Suporte a Linguagens:** APIs disponíveis para Python, C++, C#, Java e JavaScript.

## Use Cases

**Serviços de Inferência em Nuvem:** Aceleração de modelos de visão computacional, processamento de linguagem natural (NLP) e modelos de recomendação em plataformas como Azure Machine Learning.
**Aplicações de Borda (Edge Computing):** Implantação de modelos de IA em dispositivos de baixa potência e latência, como câmeras inteligentes e gateways de IoT.
**Aplicações Móveis:** Integração de recursos de IA em aplicativos iOS e Android (ONNX Runtime Mobile) para experiências personalizadas e processamento no dispositivo.
**Integração em Aplicações Desktop:** Uso em produtos como o Microsoft Office e Windows para recursos de IA integrados, como reconhecimento de imagem e processamento de texto.
**Modelos de Linguagem Grande (LLMs):** Otimização da inferência de LLMs e modelos de IA Generativa para reduzir custos e latência.

## Integration

A integração com o ONNX Runtime é direta, exigindo a instalação do pacote e o carregamento do modelo ONNX para a criação de uma `InferenceSession`. O exemplo a seguir demonstra a inferência básica em Python:

```python
import onnxruntime as ort
import numpy as np

# 1. Carregar o modelo ONNX
model_path = "caminho/para/seu/modelo.onnx"
session = ort.InferenceSession(model_path)

# 2. Preparar os dados de entrada (exemplo com um tensor float32)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
# Cria um tensor de entrada simulado
input_data = np.random.rand(*input_shape).astype(np.float32)

# 3. Executar a inferência
# O segundo argumento é um dicionário {nome_da_entrada: dados_da_entrada}
outputs = session.run(None, {input_name: input_data})

# 4. Processar a saída
print("Saída da Inferência:", outputs[0])
```

Para C++, a integração envolve a utilização da API C++ (um *wrapper* da API C) para criar um ambiente, uma sessão e executar o modelo. Exemplos detalhados estão disponíveis no repositório `microsoft/onnxruntime-inference-examples`.

## URL

https://onnxruntime.ai/