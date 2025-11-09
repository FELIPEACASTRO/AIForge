# Apache NiFi

## Description

Apache NiFi é um sistema de fluxo de dados (dataflow) poderoso, confiável e fácil de usar, projetado para automatizar o fluxo de dados entre sistemas. Ele é baseado nos conceitos de programação baseada em fluxo, permitindo a criação de gráficos direcionados escaláveis para roteamento e processamento de dados. Sua proposta de valor única reside na sua interface de usuário visual (UI) para construir e monitorar fluxos de dados em tempo real, garantindo proveniência de dados completa e rastreabilidade.

## Statistics

Adotado por mais de 8.000 empresas globais. Um único cluster NiFi pode processar trilhões de eventos e petabytes de dados por dia, com proveniência e linhagem de dados completas. Possui uma comunidade ativa e é um projeto de código aberto de nível superior da Apache Software Foundation.

## Features

Interface de usuário visual para design e monitoramento; Garantia de entrega (QoS configurável); Proveniência de dados completa (rastreabilidade); Extensibilidade (desenvolvimento de processadores personalizados); Arquitetura escalável e tolerante a falhas; Suporte a diversos protocolos e sistemas (HTTP, S3, Kafka, etc.).

## Use Cases

Automação de ETL (Extração, Transformação, Carga); Ingestão de dados em tempo real de diversas fontes; Integração de sistemas heterogêneos; Coleta e processamento de métricas e logs; Fluxos de trabalho de segurança cibernética e observabilidade.

## Integration

A integração é primariamente feita através da interface de usuário visual, conectando Processadores. Para automação e gerenciamento externo, utiliza-se a **REST API** para iniciar/parar processadores, monitorar filas e consultar proveniência. Exemplo de uso da API (via `curl`): \n\n```bash\n# Exemplo de chamada à REST API para obter o status do cluster\ncurl -X GET 'http://localhost:8080/nifi-api/flow/cluster/summary'\n```\n\nOu usando uma biblioteca Python para a API: \n\n```python\nimport requests\n\nNIFI_URL = 'http://localhost:8080/nifi-api'\n\ndef get_cluster_summary():\n    response = requests.get(f'{NIFI_URL}/flow/cluster/summary')\n    response.raise_for_status()\n    return response.json()\n\n# print(get_cluster_summary())\n```

## URL

https://nifi.apache.org/