# Pew Research Center Survey Datasets

## Description
O Pew Research Center é um *think tank* não partidário que informa o público sobre as questões, atitudes e tendências que moldam o mundo. Os **Survey Datasets** (Conjuntos de Dados de Pesquisa) do Pew Research Center são coleções de microdados em nível de caso, provenientes de suas pesquisas de opinião pública e estudos demográficos. Estes dados abrangem uma vasta gama de tópicos, incluindo política, religião, tendências sociais, internet e tecnologia, e questões globais. Os dados são disponibilizados ao público para análise secundária após um período de tempo, permitindo que pesquisadores, acadêmicos e o público em geral realizem suas próprias análises. O acesso é gratuito, mas requer um registro de conta no site.

## Statistics
**Número de Datasets:** Mais de 1000 conjuntos de dados disponíveis para download (em Novembro de 2025).
**Formato:** Arquivos SPSS (.sav) dentro de um arquivo compactado (.zip).
**Versões:** Os dados são lançados continuamente, refletindo as pesquisas mais recentes. Por exemplo, o "Spring 2024 Survey Data" foi lançado em 2025.
**Tamanho/Amostras:** Variável por pesquisa. Por exemplo, o *American Trends Panel* é uma amostra representativa nacional de adultos dos EUA, com o número de amostras (casos) variando por onda de pesquisa (tipicamente milhares de respondentes). O *Religious Landscape Survey* de 2023-24 inclui mais de 35.000 americanos.

## Features
Os conjuntos de dados são fornecidos como arquivos **SPSS (.sav)**, que são amplamente utilizados em ciências sociais. Cada download é um arquivo compactado (.zip) que inclui:
1.  **Dataset (.sav):** O arquivo de microdados em nível de caso.
2.  **Questionário Completo:** O instrumento de pesquisa original.
3.  **Codebook:** Um manual que detalha as variáveis, seus valores e a metodologia de amostragem.
Os arquivos de dados incluem **variáveis de peso** (*weight variables*) que devem ser usadas na análise para garantir a representatividade da amostra. Os dados cobrem pesquisas nos EUA (como o *American Trends Panel*) e estudos globais.

## Use Cases
**Pesquisa Acadêmica:** Análise de tendências sociais, políticas e religiosas em longo prazo.
**Jornalismo de Dados:** Criação de reportagens e visualizações baseadas em dados de opinião pública.
**Ciência de Dados e IA:** Utilização de dados de pesquisa para treinar modelos de processamento de linguagem natural (NLP) em tarefas de análise de sentimento ou classificação de tópicos, ou para estudos de viés e representatividade em IA.
**Políticas Públicas:** Informar o debate e a formulação de políticas com base nas atitudes e crenças do público.

## Integration
1.  **Acesso:** Os dados são acessíveis através da seção "Datasets" no site do Pew Research Center.
2.  **Registro:** É necessário **criar uma conta gratuita** no site para poder baixar os arquivos.
3.  **Download:** Após o login, o usuário pode baixar o arquivo compactado (.zip) que contém o dataset (.sav), o questionário e o codebook.
4.  **Uso:** O arquivo `.sav` é nativo do software estatístico **SPSS**, mas pode ser lido por outros softwares de análise estatística como **R** (usando pacotes como `haven` ou o pacote `Pew Research Methods R package` oficial), **Python** (usando bibliotecas como `pandas` com `pyreadstat`), ou **Stata**. O Codebook é essencial para a correta interpretação das variáveis e aplicação das variáveis de peso.

## URL
[https://www.pewresearch.org/datasets/](https://www.pewresearch.org/datasets/)
