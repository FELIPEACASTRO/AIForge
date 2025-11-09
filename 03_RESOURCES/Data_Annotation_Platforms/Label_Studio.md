# Label Studio

## Description

**Label Studio** é uma plataforma de anotação de dados de código aberto e flexível, projetada para preparar dados de treinamento de alta qualidade para modelos de Machine Learning (ML), incluindo LLMs (Large Language Models), Visão Computacional, Processamento de Linguagem Natural (NLP) e séries temporais [1] [2]. Sua proposta de valor única reside na sua **flexibilidade e suporte a múltiplos tipos de dados** (áudio, texto, imagens, vídeos e séries temporais) e na sua capacidade de se integrar perfeitamente ao pipeline de ML, permitindo a implementação de estratégias como **Active Learning** e pré-rotulagem por modelos [3] [4]. A plataforma é mantida pela HumanSignal e se destaca por ser uma ferramenta agnóstica a dados e modelos, fornecendo uma interface de usuário simples e configurável para anotadores e cientistas de dados.

## Statistics

*   **Estrelas no GitHub:** Mais de **10.000 estrelas** no repositório principal (`HumanSignal/label-studio`), indicando uma forte adoção e popularidade na comunidade de código aberto (dado de 2022, mas serve como base) [3].
*   **Comunidade:** Possui uma comunidade ativa com milhares de membros, incluindo um canal Slack e fóruns de discussão [6].
*   **Organização Mantenedora:** Desenvolvido e mantido pela **HumanSignal** [2].
*   **Licença:** Código aberto sob a licença **Apache 2.0** [2].

## Features

*   **Suporte a Múltiplos Tipos de Dados:** Anotação de áudio, texto, imagens (incluindo detecção de objetos e segmentação), vídeos e dados de séries temporais [2].
*   **Interface Configurável:** Permite a criação de interfaces de anotação personalizadas usando XML, adaptando-se a qualquer tarefa de rotulagem [2].
*   **Integração com ML (Active Learning):** Capacidade de conectar modelos de ML para pré-rotulagem e Active Learning, acelerando o processo de anotação [3].
*   **Gerenciamento de Projetos e Usuários:** Suporte a múltiplos projetos, usuários e equipes, com recursos de controle de qualidade e fluxo de trabalho [4].
*   **Formato de Saída Padronizado:** Exporta anotações em um formato JSON padronizado, facilitando a ingestão por modelos de ML [2].
*   **SDK Python:** Oferece um SDK robusto para integração programática em pipelines de dados [3].

## Use Cases

O Label Studio é amplamente utilizado em diversos domínios de Machine Learning para a criação de conjuntos de dados rotulados [1] [4]:

*   **Visão Computacional:**
    *   **Detecção de Objetos:** Criação de *bounding boxes* e máscaras de segmentação para imagens e vídeos.
    *   **Classificação de Imagens:** Rotulagem de imagens para tarefas de classificação.
*   **Processamento de Linguagem Natural (NLP):**
    *   **Análise de Sentimento:** Rotulagem de texto para identificar polaridade (positivo, negativo, neutro).
    *   **Reconhecimento de Entidades Nomeadas (NER):** Identificação e rotulagem de entidades (pessoas, locais, organizações) em texto.
    *   **Sistemas de Perguntas e Respostas (QA):** Rotulagem de pares de perguntas e respostas para fine-tuning de LLMs [7].
*   **Áudio e Fala:**
    *   **Transcrição:** Rotulagem de segmentos de áudio para transcrição de fala.
    *   **Classificação de Áudio:** Identificação de sons ou eventos em gravações.
*   **Séries Temporais:**
    *   **Monitoramento de Saúde:** Anotação de dados de sensores ou sinais vitais (e.g., ECG) para detecção de anomalias.
    *   **Finanças:** Rotulagem de dados de mercado para análise preditiva.
*   **Robótica:**
    *   Tradução de comportamento do mundo real em entendimento estruturado e legível por máquina para treinamento de modelos de robótica [8].

## Integration

A integração com o Label Studio é primariamente realizada através de sua **API REST** e do **SDK Python**. O SDK permite que cientistas de dados e engenheiros incorporem a plataforma diretamente em seus pipelines de ML para automatizar tarefas como criação de projetos, importação de dados e exportação de anotações [3].

**Exemplo de Integração com Python SDK (Criação de Projeto e Importação de Tarefas):**

```python
from label_studio_sdk import Client

# 1. Inicializar o cliente
# Substitua 'YOUR_LABEL_STUDIO_URL' e 'YOUR_API_KEY'
LS_URL = "http://localhost:8080"
API_KEY = "YOUR_API_KEY"
ls = Client(url=LS_URL, api_key=API_KEY)

# 2. Criar um novo projeto (opcional, se já não existir)
project = ls.create_project(title="Meu Projeto de Anotação de Imagens")

# 3. Definir tarefas de anotação (exemplo de importação de dados)
tasks = [
    {"data": {"image": "https://example.com/image1.jpg"}},
    {"data": {"image": "https://example.com/image2.jpg"}},
]

# 4. Importar tarefas para o projeto
project.import_tasks(tasks=tasks)

print(f"Projeto '{project.title}' criado e {len(tasks)} tarefas importadas.")
```

**Integração com Modelos de ML (ML Backend):**
O Label Studio suporta a conexão de um *ML Backend* (geralmente um servidor web Python) que usa o `label-studio-ml-backend` SDK. Este backend é responsável por receber dados do Label Studio, gerar predições (pré-rotulagem) e enviá-las de volta para a plataforma, facilitando o Active Learning [5].

## URL

https://labelstud.io/