# Data Labeling Platforms: Scale AI, Labelbox, Supervisely

## Description

Uma análise comparativa das principais plataformas de rotulagem de dados (Data Labeling Platforms) — Scale AI, Labelbox e Supervisely — essenciais para o desenvolvimento e aprimoramento de modelos de Inteligência Artificial. Cada plataforma oferece uma proposta de valor única, desde soluções full-stack para grandes empresas (Scale AI) até ambientes especializados em visão computacional (Supervisely) e fábricas de dados focadas em qualidade e avaliação de modelos (Labelbox).

## Statistics

**Scale AI:** Avaliação de mercado de **$14 Bilhões** (dado de 2025). Relato de **97,8% de precisão** em mais de 10 milhões de imagens médicas.
**Labelbox:** Foco em métricas de qualidade como **precisão, recall e F-1** auto-calculadas. Utiliza um painel de desempenho para monitorar **throughput, eficiência e qualidade** do processo de rotulagem.
**Supervisely:** Oferece **Estatísticas Interativas Avançadas** para análise de conjunto de dados, incluindo estatísticas de classes, objetos de vídeo e co-ocorrência de classes, essenciais para **Garantia de Qualidade (QA)**.

## Features

**Scale AI:** Plataforma full-stack para dados de IA, oferecendo rotulagem, avaliação de modelos (Scale Evaluation) e software para desenvolvimento de aplicações de IA. Focada em dados de alta qualidade para decisões críticas.
**Labelbox:** Plataforma completa de avaliação de modelos e rotulagem de dados (Data Engine). Inclui Model-Assisted Labeling, Model Foundry, e o Editor de Preferência Humana LLM, com foco em qualidade e avaliações humanas (Alignerrs).
**Supervisely:** Plataforma especializada em visão computacional, oferecendo ferramentas de anotação para imagens, vídeos, nuvens de pontos e DICOM. Possui recursos avançados de curadoria de dados, estatísticas interativas e implantação de redes neurais.

## Use Cases

**Scale AI:** Carros autônomos, mapeamento, Realidade Aumentada/Virtual (AR/VR), robótica e imagens médicas, focando em dados para sistemas de IA de missão crítica.
**Labelbox:** Geração de dados de alta qualidade para modelos de **IA Generativa (GenAI)** e modelos específicos de tarefas. Avaliação e comparação de saídas de modelos de **LLM (Large Language Models)** através do Human Preference Editor.
**Supervisely:** Tarefas de **Visão Computacional** como detecção de objetos, segmentação semântica e análise de nuvens de pontos (Point Clouds) e dados DICOM. Automação de implantação e inferência de modelos de redes neurais.

## Integration

**Scale AI:**
Integração via API e SDK oficial em Python (`scaleapi`). Permite a criação programática de tarefas de rotulagem e o gerenciamento de projetos.
```python
# Instalação: pip install --upgrade scaleapi
# import scaleapi
# from scaleapi.client import ScaleClient
# client = ScaleClient("YOUR_SCALE_API_KEY")
# # Exemplo de uso: client.create_task(...)
```

**Labelbox:**
Integração robusta via Python SDK (`labelbox`). Utilizado para gerenciamento de datasets, upload de dados, criação de ontologias e projetos, e exportação de anotações.
```python
# Instalação: pip install labelbox
# import labelbox as lb
# client = lb.Client("YOUR_LABELBOX_API_KEY")
# # Exemplo de uso: dataset = client.create_dataset(name="My New Dataset")
```

**Supervisely:**
Integração via Python SDK (`supervisely`) e API. Projetado para automação de tarefas de visão computacional, como upload de imagens, criação de anotações e implantação de modelos.
```python
# Instalação: pip install supervisely
# import supervisely as sly
# api = sly.Api("https://app.supervise.ly", "YOUR_SUPERVISELY_API_TOKEN")
# # Exemplo de uso: api.image.upload_np(...)
```

## URL

Scale AI: https://scale.com/ | Labelbox: https://labelbox.com/ | Supervisely: https://supervisely.com/