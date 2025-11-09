# LiteRT (formerly TensorFlow Lite)

## Description

LiteRT (abreviação de Lite Runtime), anteriormente conhecido como TensorFlow Lite, é o runtime de alto desempenho do Google para IA em dispositivos. Ele permite a implantação de modelos de machine learning para inferência em dispositivos de borda, abordando restrições críticas como latência, privacidade, conectividade, tamanho e consumo de energia. Suporta modelos de múltiplos frameworks (TensorFlow, PyTorch, JAX) convertidos para o formato eficiente FlatBuffers (.tflite).

## Statistics

Implantado em mais de 4 bilhões de dispositivos (dado de 2020, para TensorFlow Lite). Otimizado para restrições de machine learning em dispositivo (ODML): baixa latência (sem ida e volta ao servidor), privacidade aprimorada (nenhum dado pessoal sai do dispositivo), não requer conectividade com a internet, tamanho reduzido do modelo e binário, e consumo de energia eficiente.

## Features

Otimizado para ML em dispositivo; Suporte multiplataforma (Android, iOS, Linux embarcado, microcontroladores); Opções de modelo multiframework (conversão TensorFlow, PyTorch, JAX); Suporte a diversas linguagens (Java/Kotlin, Swift, Objective-C, C++, Python); Alto desempenho com aceleração de hardware (GPU Delegate, iOS Core ML Delegate).

## Use Cases

Visão Computacional (detecção de objetos, classificação de imagens) em dispositivos móveis/embarcados; Assistentes de Voz (detecção de palavra de ativação, PNL) no dispositivo; Automação Residencial Inteligente; Monitoramento de Saúde (dispositivos vestíveis); Veículos Autônomos (tomada de decisão em tempo real).

## Integration

Os modelos são convertidos para o formato FlatBuffers .tflite. A integração utiliza a API LiteRT Interpreter para modelos sem metadados ou a LiteRT Support Library para modelos com metadados. SDKs estão disponíveis para Android (Java/Kotlin), iOS (Swift) e Micro dispositivos (C++).

## URL

https://ai.google.dev/edge/litert