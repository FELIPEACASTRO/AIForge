# Ferramentas de Quantização de Modelos (PyTorch e TensorFlow Lite)

## Description

A quantização de modelos é uma técnica de otimização que reduz a precisão numérica dos pesos e/ou ativações de uma rede neural, tipicamente de ponto flutuante de 32 bits (FP32) para inteiros de 8 bits (INT8) ou ponto flutuante de 16 bits (FP16). Seu principal valor reside na **redução drástica do tamanho do modelo** e no **aumento da velocidade de inferência**, especialmente em dispositivos de borda (edge devices) e hardware com restrições de recursos. O **PyTorch** oferece uma API flexível para quantização dinâmica, estática e Quantization-Aware Training (QAT), focada em desempenho em CPUs e GPUs. O **TensorFlow Lite (TFLite)**, por sua vez, é o framework de implantação do TensorFlow para dispositivos móveis e embarcados, e sua quantização é crucial para otimizar modelos para esses ambientes, com forte ênfase em INT8 para máxima aceleração em aceleradores de hardware.

## Statistics

Redução de Tamanho: Tipicamente 4x (de FP32 para INT8). Redução de Latência/Aumento de Velocidade: 2x a 4x em CPUs e aceleradores compatíveis. A quantização para FP16 (float16) geralmente resulta em uma redução de 2x no tamanho e um aumento de velocidade moderado, com menor perda de precisão. A quantização INT8 é o padrão para máxima otimização em TFLite.

## Features

PyTorch:
- **Quantização Dinâmica (Dynamic Quantization):** Quantiza apenas os pesos antes da inferência e as ativações dinamicamente durante a inferência. Ideal para modelos com poucas operações de peso (como LSTMs) ou quando a calibração de dados é inviável.
- **Quantização Estática Pós-Treinamento (Post-Training Static Quantization - PTQ):** Quantiza pesos e ativações. Requer um pequeno conjunto de dados de calibração para determinar os parâmetros de quantização das ativações. Oferece maior aceleração do que a dinâmica.
- **Treinamento com Conscientização de Quantização (Quantization-Aware Training - QAT):** Simula a quantização durante o treinamento, resultando na menor perda de precisão e, frequentemente, no melhor desempenho. É o método mais complexo.

TensorFlow Lite:
- **Quantização Pós-Treinamento (Post-Training Quantization - PTQ):** Converte um modelo FP32 treinado para INT8 ou FP16. Inclui opções para quantização de pesos apenas, ou quantização completa (pesos e ativações) com calibração.
- **Quantização de Escala Total (Full Integer Quantization):** Garante que todas as operações no modelo usem inteiros, o que é essencial para implantação em microcontroladores e aceleradores que não suportam operações de ponto flutuante.
- **QAT:** Semelhante ao PyTorch, insere nós de quantização no grafo de treinamento para simular o efeito da quantização.

## Use Cases

PyTorch:
- Otimização de modelos de NLP (como LSTMs e Transformers) usando Quantização Dinâmica para implantação em servidores.
- Aceleração de modelos de visão computacional (CNNs) em CPUs de servidor ou dispositivos móveis de alto desempenho usando Quantização Estática ou QAT.
- Redução do consumo de memória em GPUs para permitir o processamento de lotes maiores (batch size).

TensorFlow Lite:
- Implantação de modelos de Visão Computacional (detecção de objetos, classificação de imagens) em smartphones (Android/iOS) e dispositivos de borda (Raspberry Pi, Coral Edge TPU).
- Execução de modelos de Machine Learning em microcontroladores (TensorFlow Lite Micro) com restrições extremas de memória e processamento.
- Aplicações de tempo real que exigem baixa latência, como reconhecimento de fala no dispositivo ou filtros de câmera.

## Integration

PyTorch (Exemplo de Quantização Dinâmica):
```python
import torch
from torch.quantization import quantize_dynamic

# Carregar o modelo (exemplo: um modelo LSTM)
model_fp32 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
model_fp32.eval()

# Aplicar Quantização Dinâmica
model_quantized = quantize_dynamic(
    model_fp32, 
    {torch.nn.Linear, torch.nn.LSTM}, 
    dtype=torch.qint8
)

# O modelo quantizado está pronto para inferência
print(model_quantized)
```

TensorFlow Lite (Exemplo de Quantização Pós-Treinamento para INT8):
```python
import tensorflow as tf

# Carregar o modelo Keras treinado
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Configurar a otimização para quantização INT8
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Função de conjunto de dados de calibração (necessária para quantização completa)
def representative_dataset_gen():
    # Gerar dados de entrada de exemplo
    for _ in range(num_calibration_steps):
        yield [input_data]

converter.representative_dataset = representative_dataset_gen

# Garantir que apenas operações INT8 sejam usadas (opcional, para microcontroladores)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Converter o modelo
tflite_quant_model = converter.convert()

# Salvar o modelo TFLite quantizado
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
```

## URL

PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html | TensorFlow Lite Quantization: https://www.tensorflow.org/model_optimization/guide/quantization