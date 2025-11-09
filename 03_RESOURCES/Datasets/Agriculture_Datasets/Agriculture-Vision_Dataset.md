# Agriculture-Vision Dataset

## Description

O Agriculture-Vision é um extenso conjunto de dados de imagens aéreas de alta resolução e multi-banda, projetado para análise de padrões agrícolas e detecção de anomalias em campos de cultivo. O dataset original (2020) e suas extensões subsequentes (2021, 2022, 2023) foram criados para o desafio anual CVPR Agriculture-Vision. Ele se destaca por ser multi-banda (RGB e Infravermelho Próximo - NIR) e por incluir anotações detalhadas de nove tipos de anomalias de campo por especialistas em agronomia. As anomalias incluem: sombra de nuvem, planta dupla, seca, fim de linha, deficiência de nutrientes, falha de plantio, danos por tempestade, água, curso d'água e aglomerado de ervas daninhas. O dataset é fundamental para o desenvolvimento de modelos de Visão Computacional em Agricultura de Precisão.

## Statistics

**Dataset Original (2020):**
*   **Total de Imagens:** 94.986 imagens de 512x512 pixels.
*   **Campos de Cultivo:** Amostrado de 3.432 fazendas nos EUA.
*   **Divisão:** 56.944 (Treino) / 18.334 (Validação) / 19.708 (Teste).
*   **Canais:** 4 canais (RGB e NIR).
*   **Tamanho do Arquivo:** O subconjunto do desafio (2020) é de aproximadamente 4.4 GB.

**Extensões (2021 em diante):**
*   **Resolução:** Imagens de até 10 cm/pixel.
*   **Dados Adicionais:** Sequências de imagens de campo completo (full-field imagery) de 52 campos, totalizando 261 imagens de alta resolução, para métodos fracamente supervisionados.
*   **Tamanho do Arquivo (2021):** O novo dataset de 2021 é de aproximadamente 20 GB.

## Features

Imagens aéreas de alta resolução (até 10 cm/pixel); Multi-banda (RGB e Infravermelho Próximo - NIR); Anotações de segmentação semântica para 9 tipos de anomalias de campo; Imagens de 512x512 pixels; Inclui dados de campo completo (full-field imagery) para métodos fracamente supervisionados (a partir de 2021).

## Use Cases

**Detecção de Anomalias Agrícolas:** Identificação e localização precisa de problemas como falhas de plantio, plantas duplas, deficiência de nutrientes e danos por água.
**Segmentação Semântica:** Treinamento de modelos para segmentar diferentes padrões e anomalias em imagens aéreas.
**Agricultura de Precisão:** Suporte à tomada de decisão para otimizar o uso de recursos (água, fertilizantes) e aumentar a produtividade das colheitas.
**Desenvolvimento de Modelos Multi-banda:** Pesquisa e desenvolvimento de arquiteturas de visão computacional que utilizam informações de canais além do RGB (NIR).

## Integration

O dataset pode ser acessado e baixado diretamente do Amazon S3 Bucket, sem a necessidade de uma conta AWS, utilizando o AWS CLI com a flag `--no-sign-request`.
Exemplo de acesso ao dataset original (2020) via AWS CLI:
```bash
aws s3 ls --no-sign-request s3://intelinair-data-releases/agriculture-vision/cvpr_paper_2020/
```
O dataset também está disponível no Hugging Face Hub, permitindo o acesso via biblioteca `datasets` do Python, embora o download direto do arquivo `tar.gz` seja o método principal.
Exemplo de acesso ao dataset de 2021 via AWS CLI:
```bash
aws s3 ls --no-sign-request s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/
```

## URL

https://www.agriculture-vision.com/