# NVIDIA Triton Inference Server

## Description

Software de código aberto para servir inferência de IA, desenvolvido pela NVIDIA. O Triton Inference Server padroniza a implantação e execução de modelos de IA em todas as cargas de trabalho, permitindo que as equipes implantem qualquer modelo de IA de múltiplas estruturas de aprendizado profundo e aprendizado de máquina (incluindo TensorRT, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, e mais). Ele é otimizado para alto desempenho (alta taxa de transferência e baixa latência) e suporta implantação em nuvem, data center, edge e dispositivos embarcados em GPUs NVIDIA, CPUs x86 e ARM, ou AWS Inferentia. O Triton faz parte da plataforma de software NVIDIA AI Enterprise.

## Statistics

Otimizado para alto desempenho (alta taxa de transferência e baixa latência). Fornece métricas Prometheus para estatísticas de GPU e requisições, incluindo contagens de requisições, inferências e execuções, permitindo o cálculo do tamanho médio do lote (Inference Count / Execution Count). Oferece métricas detalhadas de latência (tempo de espera na fila, tempo de execução, tempo de envio/recebimento).

## Features

Suporte a múltiplas estruturas de IA (TensorRT, PyTorch, ONNX, etc.). Suporte a diversas plataformas de hardware (NVIDIA GPUs, CPUs x86/ARM, AWS Inferentia). Protocolos de inferência HTTP/REST e GRPC. Loteamento de sequência (Sequence batching) e gerenciamento de estado implícito para modelos com estado. Pipelines de modelo usando Ensembling ou Business Logic Scripting (BLS). API C e Java para casos de uso in-process e edge. Métricas detalhadas (GPU, throughput, latência) via Prometheus. Extensibilidade via Backend API para operações personalizadas de pré/pós-processamento.

## Use Cases

Implantação de modelos de IA em produção em escala. Servir modelos em tempo real, em lote, e em fluxos de áudio/vídeo. Aplicações de IA com estado (como modelos de linguagem sequenciais) usando loteamento de sequência. Criação de pipelines de inferência complexos (Ensembling/BLS). Implantação em ambientes de nuvem, data center, edge e embarcados.

## Integration

Protocolos de inferência HTTP/REST e GRPC baseados no protocolo KServe. Bibliotecas cliente Python (tritonclient) e C++ para comunicação. API C e Java para integração in-process. Integração com Prometheus para monitoramento de métricas. Exemplo de uso do cliente Python para inferência:

```python
import tritonclient.http as httpclient

# Configuração do cliente
triton_client = httpclient.InferenceServerClient(url='localhost:8000')

# Preparação dos dados de entrada (exemplo simplificado)
input_data = httpclient.InferInput('INPUT_NAME', [1, 224, 224, 3], 'FP32')
input_data.set_data_from_numpy(numpy_array, binary_data=True)

# Envio da requisição de inferência
result = triton_client.infer(
    model_name='my_model',
    inputs=[input_data],
    outputs=[httpclient.InferRequestedOutput('OUTPUT_NAME')]
)

# Obtenção do resultado
output_data = result.as_numpy('OUTPUT_NAME')
print(output_data)
```

## URL

https://developer.nvidia.com/dynamo-triton