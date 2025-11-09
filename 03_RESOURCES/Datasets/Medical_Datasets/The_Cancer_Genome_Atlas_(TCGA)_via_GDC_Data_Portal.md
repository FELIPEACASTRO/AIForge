# The Cancer Genome Atlas (TCGA) via GDC Data Portal

## Description

O The Cancer Genome Atlas (TCGA) é um programa abrangente e coordenado para mapear as alterações genômicas do câncer. Os dados são acessados e harmonizados através do Genomic Data Commons (GDC) Data Portal, que fornece uma plataforma computacional para pesquisadores de câncer. O GDC harmoniza dados clínicos e genômicos, incluindo sequenciamento de genoma completo (WGS), sequenciamento de exoma completo (WES), RNA-Seq, miRNA-Seq, metilação e dados clínicos.

## Statistics

Dados da Data Release 44.0 (Outubro de 2025): 88 Projetos, 69 Sítios Primários, 48.763 Casos, 1.223.539 Arquivos, 22.580 Genes, 3.082.397 Mutações. Os dados são continuamente atualizados e expandidos com novos projetos como APOLLO-OV e CCDI-MCI.

## Features

Dados multi-ômicos harmonizados (genômica, transcriptômica, epigenômica, proteômica) e dados clínicos padronizados. Inclui variantes somáticas, expressão gênica, contagens de RNA-Seq, dados de metilação, e imagens de lâminas (WSIs).

## Use Cases

Identificação de novos biomarcadores e alvos terapêuticos, classificação de subtipos de câncer, estudos de sobrevivência e prognóstico, desenvolvimento de modelos de aprendizado de máquina para predição de resposta a medicamentos e progressão da doença.

## Integration

Acesso via GDC Data Portal (interface web) ou GDC API. Ferramentas como TCGADownloadHelper (Python) simplificam a extração e pré-processamento. O GDC Data Transfer Tool é recomendado para download de grandes volumes de dados. Exemplo de uso do TCGADownloadHelper:

```python
# Instalação (exemplo)
# pip install TCGADownloadHelper

from TCGADownloadHelper import TCGADownloadHelper

# Inicializa o helper
tcga_helper = TCGADownloadHelper(project_name='TCGA-BRCA', data_type='RNA-Seq', file_type='counts')

# Baixa os dados
tcga_helper.download_data()

# Prepara os dados para análise (exemplo)
df = tcga_helper.prepare_data()
print(df.head())
```

## URL

https://portal.gdc.cancer.gov/