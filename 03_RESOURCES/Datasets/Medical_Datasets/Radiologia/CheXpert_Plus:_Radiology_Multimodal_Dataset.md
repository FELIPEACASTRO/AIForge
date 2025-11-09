# CheXpert Plus: Radiology Multimodal Dataset

## Description

O CheXpert Plus é uma coleção abrangente e multimodal que estende o dataset CheXpert original, combinando imagens de raios-X de tórax com relatórios de radiologia alinhados, dados demográficos do paciente e formatos de imagem adicionais. Este dataset foi lançado em 2024 e representa um avanço significativo para o treinamento de modelos de IA multimodal, como Large Language Models (LLMs) aplicados à radiologia. Ele é notável por ser o maior dataset de texto liberado publicamente em radiologia, facilitando o desenvolvimento de sistemas que podem interpretar e gerar relatórios médicos com maior precisão e contexto clínico.

## Statistics

**Total de Pares Únicos Imagem-Relatório:** 223.462. **Estudos:** 187.711. **Pacientes:** 64.725. **Tamanho do Texto:** 36 milhões de tokens de texto, incluindo 13 milhões de tokens de impressão. **Formato da Imagem:** DICOM. **Patologias Anotadas:** 14. **Lançamento:** 2024.

## Features

Multimodal (Imagens de raios-X de tórax e Relatórios de Radiologia). Relatórios de radiologia meticulosamente divididos em 11 subseções. Inclui 47 elementos de metadados DICOM. Anotações para 14 patologias torácicas. 8 elementos de metadados sobre informações do paciente. Foco em alinhamento texto-imagem para treinamento de modelos de linguagem visual.

## Use Cases

**Treinamento de Modelos Multimodais:** Desenvolvimento de Large Language Models (LLMs) para radiologia, como o CXR-LLaVA. **Geração de Relatórios Radiológicos:** Criação de sistemas que geram relatórios a partir de imagens. **Classificação Multilabel:** Detecção e classificação de 14 patologias torácicas. **Pesquisa em Viés e Equidade:** Análise de vieses em IA de imagem devido à inclusão de dados demográficos. **Feature Engineering:** Extração de features de texto (NLP) e imagem (visuais) para tarefas de diagnóstico.

## Integration

O dataset é disponibilizado pelo Stanford Center for Artificial Intelligence in Medicine & Imaging (AIMI). O acesso requer a aceitação dos Termos e Condições e pode ser obtido através do portal AIMI. O repositório oficial no GitHub (Stanford-AIMI/chexpert-plus) fornece informações e potencialmente scripts para download e processamento. O uso em projetos de aprendizado de máquina geralmente envolve o alinhamento dos pares imagem-texto para tarefas como geração de relatórios e classificação multimodal. Exemplo de acesso (pseudocódigo Python):
```python
# O acesso requer registro e aprovação no portal AIMI
# Exemplo de uso após o download e descompactação:

import pandas as pd
import os

# Carregar o arquivo de metadados (exemplo)
metadata_path = 'path/to/chexpert_plus/metadata.csv'
df_metadata = pd.read_csv(metadata_path)

# Acessar um par imagem-relatório
for index, row in df_metadata.head().iterrows():
    image_path = os.path.join('path/to/chexpert_plus/images', row['dicom_id'] + '.dcm')
    report_text = row['full_report_text']
    
    print(f"Imagem: {image_path}")
    print(f"Relatório: {report_text[:200]}...")
    # Implementar lógica de carregamento de imagem DICOM e processamento de texto
```

## URL

https://aimi.stanford.edu/datasets/chexpert-plus