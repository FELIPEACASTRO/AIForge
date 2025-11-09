# ONNX - Open Neural Network Exchange

## Description

O **Open Neural Network Exchange (ONNX)** é um formato aberto e padrão da indústria, apoiado por uma comunidade de empresas de tecnologia e organizações de pesquisa, projetado para representar modelos de aprendizado de máquina. Sua proposta de valor única reside em sua capacidade de **interoperabilidade** e **portabilidade**, permitindo que os desenvolvedores treinem modelos em qualquer framework (como PyTorch, TensorFlow, Keras) e os executem em qualquer runtime ou dispositivo (como ONNX Runtime, CPUs, GPUs, ou aceleradores de hardware dedicados). O ONNX define um conjunto comum de operadores e um formato de arquivo para o grafo de computação, atuando como uma linguagem intermediária que elimina a dependência de framework para a fase de inferência. Isso simplifica a transição do desenvolvimento para a produção, garantindo que o desempenho do modelo seja maximizado em diversos ambientes de implantação.

## Statistics

- **Adoção Ampla:** O ONNX é um projeto graduado da LF AI & Data Foundation, com contribuições de grandes empresas como Microsoft, Meta, Amazon e Nvidia.
- **Ecossistema Robusto:** O ecossistema inclui conversores para mais de 15 frameworks de ML (incluindo PyTorch, TensorFlow, Keras, Scikit-learn, MATLAB), e é suportado por mais de 40 runtimes e aceleradores de hardware.
- **Popularidade no Hugging Face:** Modelos no formato ONNX (frequentemente quantizados ou otimizados) são amplamente utilizados para inferência de alto desempenho, com muitos modelos ONNX populares acumulando centenas de milhares de downloads (ex: modelos de embeddings como `bge-m3-onnx-o4` com mais de 230 mil downloads).
- **ONNX Runtime (ORT):** O ORT é o motor de inferência mais comum para modelos ONNX, suportando múltiplas plataformas (Windows, Linux, macOS, Android, iOS) e linguagens (Python, C++, C#, Java, JavaScript).

## Features

- **Formato de Grafo de Computação:** Define o modelo como um grafo de computação, onde os nós são operadores e as bordas são tensores de dados.
- **Conjunto de Operadores Padrão (Opset):** Possui um conjunto rico e extensível de operadores (como `Conv`, `Relu`, `Add`) que são os blocos de construção de modelos de ML e DL.
- **Portabilidade de Framework:** Permite a conversão de modelos de frameworks populares (PyTorch, TensorFlow, Keras, Scikit-learn) para o formato ONNX.
- **Otimização de Modelo:** Modelos ONNX podem ser otimizados por ferramentas como o ONNX Optimizer para fusão de nós e outras transformações antes da implantação.
- **Extensibilidade:** Suporta operadores personalizados (Custom Ops) para estender a funcionalidade do formato.

## Use Cases

- **Implantação Multiplataforma:** Implantação de modelos treinados em um framework em ambientes de produção que exigem um runtime diferente (ex: treinar em PyTorch e implantar em um servidor C# usando ONNX Runtime).
- **Otimização de Inferência:** Aceleração da inferência de modelos de Visão Computacional (como ResNet, YOLO), Processamento de Linguagem Natural (como BERT, GPT) e modelos Tabulares, aproveitando otimizações específicas do hardware via ONNX Runtime.
- **Edge Computing e IoT:** Implantação de modelos de ML em dispositivos de borda com recursos limitados, onde a eficiência e o baixo consumo de energia são cruciais.
- **Conversão de Framework:** Uso do ONNX como formato intermediário para converter modelos entre diferentes frameworks de ML (ex: de TensorFlow para PyTorch ou vice-versa) para fins de pesquisa ou migração.
- **Serviços de IA em Nuvem:** Plataformas de nuvem (como Azure ML) usam ONNX para fornecer inferência otimizada e escalável para modelos de clientes.

## Integration

A integração com ONNX geralmente envolve duas etapas: **Exportação** do modelo treinado para o formato ONNX e **Inferência** usando o ONNX Runtime (ORT).

**1. Exportação de um Modelo PyTorch para ONNX (Exemplo):**
```python
import torch
import torch.onnx as onnx
import torchvision.models as models

# 1. Carregar ou definir o modelo
model = models.resnet18(pretrained=True)
model.eval()

# 2. Definir uma entrada de exemplo (dummy input)
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

# 3. Exportar o modelo
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",  # Nome do arquivo de saída
    export_params=True,
    opset_version=17, # Versão do Opset (deve ser compatível com o runtime)
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Modelo PyTorch exportado com sucesso para resnet18.onnx")
```

**2. Inferência com ONNX Runtime (ORT) em Python:**
```python
import onnxruntime as ort
import numpy as np

# 1. Carregar o modelo ONNX
ort_session = ort.InferenceSession("resnet18.onnx")

# 2. Preparar a entrada (deve corresponder ao formato de entrada do modelo)
# O ONNX Runtime espera entradas como um dicionário de nomes de entrada para arrays NumPy
input_name = ort_session.get_inputs()[0].name
input_shape = ort_session.get_inputs()[0].shape
# Criar um array NumPy de exemplo
input_data = np.random.randn(*input_shape).astype(np.float32)
input_feed = {input_name: input_data}

# 3. Executar a inferência
output = ort_session.run(None, input_feed)

# 4. Processar a saída
print("Inferência executada com sucesso.")
# A saída é uma lista de arrays NumPy, um para cada saída do modelo
# print(output[0].shape)
```

## URL

https://onnx.ai/