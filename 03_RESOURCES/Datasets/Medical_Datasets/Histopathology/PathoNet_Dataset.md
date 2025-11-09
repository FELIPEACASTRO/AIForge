# PathoNet Dataset

## Description

**PathoNet** é um dataset de propósito geral para patologia digital, focado em imagens de histopatologia. Foi publicado em julho de 2023 e consiste em imagens extraídas do portal de dados **TCGA (The Cancer Genome Atlas)**. O dataset é composto por imagens de 12 classes de tecidos diferentes, sendo uma fonte valiosa para o desenvolvimento e avaliação de modelos de *Deep Learning* em patologia computacional. As imagens são patches de 256x256 pixels, extraídos de Whole Slide Images (WSI), e passaram por um processo de limpeza automatizado para remover conteúdo excessivo branco e imagens borradas. A divisão em conjuntos de Treinamento, Validação e Teste é feita considerando os casos (WSI) para evitar a mistura de imagens do mesmo caso em partições diferentes.

## Statistics

**Total de Imagens:** 4.462.156 imagens JPG.
**Resolução:** 256x256 pixels.
**Classes:** 12 classes de tecidos.
**Fonte:** TCGA (The Cancer Genome Atlas).
**Tamanho Total do Arquivo:** 131.6 GB (dividido em múltiplos arquivos .zip).
**Divisão de Dados (Exemplo - Bexiga):** Treinamento (308.677), Validação (38.927), Teste (39.166).

## Features

Dataset de propósito geral para patologia digital. Imagens de histopatologia (patches de 256x256 pixels). 12 classes de tecidos (Bexiga, Cérebro, Mama, Brônquios e Pulmão, Cólon, Corpo do Útero, Rim, Fígado e Ductos Biliares Intra-hepáticos, Próstata, Pele, Estômago, Glândula Tireoide). Extraído de Whole Slide Images (WSI) do TCGA. Divisão de dados baseada em casos para evitar vazamento de dados. Processo de limpeza automatizado.

## Use Cases

Treinamento e avaliação de modelos de Deep Learning para classificação de tecidos em patologia digital. Desenvolvimento de modelos de segmentação e detecção de objetos em imagens de histopatologia. Pesquisa em patologia computacional, especialmente para tarefas de classificação multi-classe de câncer baseada em tecidos. Estudos de Transfer Learning em patologia.

## Integration

O dataset está disponível para download no Zenodo, dividido em arquivos `.zip` por partição (Treinamento, Validação, Teste) e por classe para o conjunto de treinamento. O download pode ser realizado diretamente pela interface web do Zenodo.

**Exemplo de Download (Shell/cURL):**
Devido ao tamanho total de 131.6 GB, o download deve ser feito em partes. O arquivo `test.zip` (13.3 GB) pode ser baixado via `cURL` ou `wget` usando o link de download direto fornecido na página do Zenodo.

```bash
# Exemplo de download do arquivo de teste
wget https://zenodo.org/record/8116751/files/test.zip
# Exemplo de download de uma classe de treinamento (Bexiga)
wget https://zenodo.org/record/8116751/files/train_bladder.zip
```

**Nota:** A URL de download direto pode mudar, sendo a página do Zenodo a fonte primária para obter os links mais recentes. Não há um script de integração Python padrão fornecido, mas o download e a manipulação dos arquivos `.zip` podem ser facilmente integrados em um pipeline de ML usando bibliotecas como `zipfile` e `Pillow` no Python.

## URL

https://zenodo.org/records/8116751