# NVIDIA TensorRT

## Description

O NVIDIA TensorRT é um Software Development Kit (SDK) de alto desempenho para otimização e aceleração da inferência de modelos de deep learning em produção, exclusivamente em GPUs NVIDIA. Ele atua como um compilador de inferência que transforma modelos treinados em frameworks como TensorFlow, PyTorch e ONNX em um "motor" (engine) otimizado. Sua proposta de valor única é maximizar a eficiência da GPU, resultando em baixa latência, alto throughput e máxima utilização do hardware para aplicações críticas em tempo real. É uma ferramenta de pós-treinamento, complementando os frameworks de treinamento.

## Statistics

Aceleração de Inferência: Relatos comuns indicam um aumento de velocidade de 3x a 5x em comparação com a execução do modelo original em frameworks como PyTorch ou TensorFlow. Comparação com CPU: Em alguns casos, a inferência com TensorRT em GPUs pode ser até 40x mais rápida do que em plataformas somente com CPU. Métricas de Desempenho: Throughput (imagens/segundo ou tokens/segundo), Latência (tempo de ponta a ponta para uma única inferência) e Utilização da GPU.

## Features

Fusão de Camadas (Layer Fusion): Combina várias camadas do modelo em um único kernel da GPU. Otimização de Precisão (Precision Calibration): Suporte a precisão mista (FP32, FP16 e INT8) com quantização para maximizar o desempenho. Alocação de Memória Otimizada: Reutiliza a memória da GPU de forma eficiente, minimizando a pegada de memória. Seleção Automática de Kernel (Kernel Auto-Tuning): Seleciona o algoritmo de kernel mais rápido para a arquitetura de GPU e o tamanho de lote específicos. Otimização de Grafos: Remove camadas desnecessárias e reordena o grafo de computação. Suporte a Modelos Dinâmicos: Permite otimizar o motor para diferentes tamanhos de entrada em tempo de execução.

## Use Cases

Veículos Autônomos: Processamento em tempo real de dados de sensores (visão computacional, LiDAR) para detecção de objetos e planejamento de rotas. Processamento de Linguagem Natural (NLP) e LLMs: Aceleração de modelos de linguagem grandes (LLMs) com o TensorRT-LLM, permitindo respostas mais rápidas e maior throughput. Visão Computacional: Aplicações como reconhecimento facial, vigilância por vídeo e análise de imagens médicas (detecção de câncer). Geração de Conteúdo em Tempo Real: Aplicações de IA generativa que se beneficiam da baixa latência para interações mais fluidas. Sistemas de Recomendação: Sistemas de alto volume que exigem baixa latência para resultados instantâneos.

## Integration

A integração do TensorRT geralmente envolve a conversão de um modelo treinado em um framework para o formato otimizado do TensorRT. A NVIDIA fornece bibliotecas específicas para os principais frameworks:

1.  **PyTorch (Torch-TensorRT):** Permite compilar modelos PyTorch diretamente para o TensorRT.
    ```python
    import torch
    import torch_tensorrt
    # ... (código de carregamento do modelo)
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[input_tensor],
        enabled_precisions={torch.float16}, # Otimização para FP16
        workspace_size=1 << 25
    )
    ```
2.  **TensorFlow (TF-TRT):** Uma integração que permite otimizar modelos do TensorFlow (SavedModel).
    ```python
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    # ... (código de configuração da conversão)
    converter = trt.TrtConverterV2(input_saved_model_dir=input_dir, conversion_params=params)
    converter.convert()
    converter.save(output_dir)
    ```
3.  **API C++:** Para máxima performance e controle em ambientes de produção, o TensorRT oferece uma API C++ para construção, calibração e execução de motores de inferência, frequentemente importando modelos no formato ONNX.

## URL

https://developer.nvidia.com/tensorrt