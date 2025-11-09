# TorchServe

## Description

TorchServe é uma ferramenta de código aberto para servir modelos PyTorch em produção, desenvolvida e mantida em colaboração pela AWS e pela Meta (Facebook). Sua proposta de valor única reside em fornecer uma solução robusta, flexível e fácil de usar para a implantação de modelos de Deep Learning em escala, com foco em alto desempenho e gerenciamento completo do ciclo de vida do modelo. **Nota Importante:** O projeto está atualmente em modo de **Manutenção Limitada** (Limited Maintenance), o que significa que não há atualizações, correções de bugs ou patches de segurança planejados, e os usuários devem estar cientes de que vulnerabilidades podem não ser resolvidas.

## Statistics

**Métricas de Desempenho e Monitoramento:** O TorchServe oferece um sistema de métricas abrangente, classificadas em:
*   **Métricas de Frontend:** Incluem status de solicitação da API (2XX, 4XX, 5XX), métricas de solicitação de inferência e métricas de utilização do sistema (coletadas periodicamente).
*   **Métricas de Backend:** Incluem métricas padrão e métricas personalizadas do modelo (via API).
*   **Modos de Métricas:** Suporta três modos de coleta e exposição de métricas: `log` (padrão, logs para arquivos), `prometheus` (expõe métricas em formato Prometheus via endpoint da API) e `legacy`.
*   **Latência:** Métricas detalhadas de latência, como `ts_inference_latency_microseconds` e `ts_queue_latency_microseconds`, são rastreadas para otimização de desempenho.

## Features

**APIs de Gerenciamento e Inferência:** Oferece APIs RESTful dedicadas para gerenciamento de modelos (registro, descarregamento, escalonamento) e inferência (previsões em tempo real e em lote).
**Suporte a Modelos:** Suporta modelos PyTorch no modo Eager e Scripted (TorchScript).
**Manipuladores Personalizados (Custom Handlers):** Permite a criação de lógica de pré-processamento e pós-processamento personalizada para atender a requisitos específicos do modelo.
**Escalabilidade e Desempenho:** Suporta Dynamic Batching (agrupamento dinâmico) para otimizar a utilização de recursos e microbatching.
**Segurança:** Suporte para Servir Modelos de Forma Segura (SSL/TLS).
**Suporte a Hardware:** Integração e otimização para diversos hardwares, incluindo AWS Inferentia2, AWS Graviton, OpenVINO e Intel Extension for PyTorch.
**Gerenciamento de Modelos:** Permite servir múltiplas versões do mesmo modelo e realizar testes A/B.

## Use Cases

**Implantação de Modelos de Visão Computacional:** Servir modelos de classificação de imagens, detecção de objetos e segmentação (eager e scripted mode).
**Serviço de Modelos em Escala:** Utilização de Dynamic Batching e otimizações de hardware (como AWS Inferentia2 e Graviton) para servir modelos com alta taxa de transferência e baixa latência.
**Aplicações de IA Generativa (GenAI):**
*   **LLM Serving com RAG Compilado:** Implantação de endpoints de Geração Aumentada por Recuperação (RAG) usando `torch.compile` para aumentar o throughput e otimizar o uso de recursos (ex: RAG endpoint em CPU Graviton e LLM endpoint em GPU).
*   **Geração Multi-Imagens em Cadeia:** Criação de aplicações complexas que encadeiam múltiplos modelos (ex: Llama para geração de prompt e Stable Diffusion para geração de imagem) usando TorchServe, `torch.compile` e OpenVINO para otimização de desempenho.
**Serviço Seguro:** Implantação de modelos com segurança aprimorada usando configurações SSL/TLS.

## Integration

A integração com o TorchServe é realizada principalmente através de sua API RESTful. O fluxo típico envolve a criação de um arquivo de arquivo de modelo (MAR) e o uso de comandos `curl` para interagir com as APIs de Gerenciamento e Inferência.

**1. Criação do Arquivo MAR (Model Archive):**
```bash
torch-model-archiver --model-name my_model --version 1.0 \
--model-file my_model.py --serialized-file my_model.pth \
--handler image_classifier --extra-files index_to_name.json
```

**2. Início do TorchServe:**
```bash
torchserve --start --ncs --model-store model_store
```

**3. Registro do Modelo (API de Gerenciamento):**
```bash
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=my_model.mar"
```

**4. Inferência (API de Inferência):**
```bash
# Exemplo de chamada de inferência com um arquivo de entrada
curl http://localhost:8080/predictions/my_model -T input_data.json
```

**5. Escalabilidade (API de Gerenciamento):**
```bash
# Aumentar o número de workers para escalabilidade
curl -v -X PUT "http://localhost:8081/models/my_model?min_worker=4&synchronous=true"
```

## URL

https://github.com/pytorch/serve