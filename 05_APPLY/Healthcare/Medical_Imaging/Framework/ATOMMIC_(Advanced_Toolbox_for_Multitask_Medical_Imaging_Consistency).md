# ATOMMIC (Advanced Toolbox for Multitask Medical Imaging Consistency)

## Description

**ATOMMIC (Advanced Toolbox for Multitask Medical Imaging Consistency)** é um *framework* de código aberto e modular baseado em PyTorch Lightning, projetado para facilitar a pesquisa e aplicação de métodos de Inteligência Artificial em **Ressonância Magnética (RM)**. Ele suporta múltiplas tarefas, incluindo **reconstrução acelerada de RM (REC)**, **segmentação de RM (SEG)**, **imagem de RM quantitativa (qMRI)** e, crucialmente, **Aprendizado Multi-Tarefa (MTL)** para realizar tarefas simultaneamente, como reconstrução e segmentação conjuntas. O framework visa padronizar fluxos de trabalho, garantir a interoperabilidade de dados e permitir a avaliação de modelos de *Deep Learning* em diversos conjuntos de dados e esquemas de subamostragem. O MTL é um componente central, permitindo que o modelo melhore o desempenho em tarefas conjuntas ao aprender representações compartilhadas. Além disso, o MTL tem sido aplicado com sucesso em modelos multimodais para a **previsão simultânea de múltiplas doenças crônicas** (como diabetes, doenças cardíacas, AVC e hipertensão) a partir de registros médicos eletrônicos, demonstrando sua eficácia na avaliação de risco clínico.

## Statistics

- **Ano de Publicação:** 2024 (Artigo: Computer Methods and Programs in Biomedicine, Volume 256, Novembro de 2024).
- **Citações:** 1 (no artigo original do ScienceDirect, em Nov 2025).
- **Modelos Suportados:** Mais de 25 modelos de *Deep Learning* de última geração.
- **Datasets Suportados:** Suporte nativo para datasets públicos como CC359, fastMRI, BraTS 2023, ISLES 2022 e SKM-TEA.
- **Desempenho MTL:** Demonstra melhoria no desempenho de tarefas conjuntas (e.g., reconstrução e segmentação) em comparação com modelos de tarefa única.
- **Métricas de Avaliação:** Utiliza métricas padrão da área como SSIM, PSNR, NMSE (para reconstrução) e DICE, F1, IOU, HD95 (para segmentação).

## Features

- **Modularidade e Extensibilidade:** Projetado para fácil adição de novas tarefas, modelos e conjuntos de dados.
- **Suporte a Múltiplas Tarefas de RM:** Inclui coleções para REC, SEG, qMRI e MTL.
- **Mais de 25 Modelos SOTA:** Implementa modelos de última geração como CIRIM, VarNet, UNet, e modelos específicos para MTL (e.g., SERANet, MTLRS).
- **Treinamento de Alto Desempenho:** Utiliza PyTorch Lightning para treinamento multi-GPU, precisão mista e otimização de hiperparâmetros.
- **Interoperabilidade de Dados:** Suporta dados de valor complexo e real, com extensas transformações de pré-processamento.
- **Integração com HuggingFace:** Modelos pré-treinados e *checkpoints* podem ser carregados e baixados diretamente do HuggingFace.

## Use Cases

- **Reconstrução Acelerada de RM:** Redução do tempo de aquisição de imagens de RM.
- **Segmentação de Imagens Médicas:** Segmentação de estruturas anatômicas e lesões em exames de RM (e.g., tumores cerebrais no BraTS).
- **Imagem de RM Quantitativa:** Estimação de mapas de parâmetros quantitativos.
- **MTL em Imagem Médica:** Realização simultânea de reconstrução e segmentação para consistência e melhoria de desempenho.
- **Previsão de Doenças Crônicas:** Aplicação de MTL em redes multimodais (MAND) para prever simultaneamente os riscos de **diabetes mellitus, doenças cardíacas, AVC e hipertensão** a partir de registros médicos eletrônicos.
- **Diagnóstico e Prognóstico em Oncologia:** Uso de MTL com CNNs e Vision Transformers para melhorar a previsão de resultados em pacientes com câncer de cabeça e pescoço.

## Integration

A instalação é recomendada via Conda e Pip:

```shell
conda create -n atommic python=3.10
conda activate atommic
pip install atommic
```

Para uso em pesquisa, o treinamento e teste são realizados através de um arquivo de configuração `.yaml` (utilizando Hydra e OmegaConf) e um comando simples:

```shell
atommic run -c path-to-config-file.yaml
```

**Exemplo de Configuração (Conceitual para MTL):**
O arquivo `.yaml` define o *pipeline* completo, incluindo a tarefa (`MTL`), o modelo (e.g., `MTLRS`), o conjunto de dados (e.g., `SKM-TEA`), os parâmetros de subamostragem, transformações, otimizador e agendador.

**Integração com Docker:**
Uma imagem Docker está disponível para facilitar a implantação em diferentes ambientes:

```shell
docker pull wdika/atommic
docker run --gpus all -it --rm -v /home/user/configs:/config atommic:latest atommic run -c /config/config.yaml
```

## URL

https://github.com/wdika/atommic