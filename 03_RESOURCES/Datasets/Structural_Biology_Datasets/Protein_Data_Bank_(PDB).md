# Protein Data Bank (PDB)

## Description

O Protein Data Bank (PDB) é o principal repositório global de estruturas tridimensionais (3D) de macromoléculas biológicas, como proteínas e ácidos nucleicos, determinadas experimentalmente (por cristalografia de raios X, RMN, crio-EM). É um recurso fundamental para a biologia estrutural e a bioinformática. Recentemente (2023-2025), o foco tem se expandido para incluir Estruturas Integrativas e Modelos de Estrutura Computacional (CSM) de fontes como AlphaFold DB e ModelArchive, consolidando o PDB como um hub para dados de estrutura molecular.

## Statistics

Em novembro de 2025, o arquivo PDB continha 244.730 estruturas disponíveis. O crescimento anual em 2025 foi de 15.064 novas estruturas, seguindo a tendência de crescimento constante. O arquivo também inclui mais de 1 milhão de Modelos de Estrutura Computacional (CSM).

## Features

Estruturas 3D de Proteínas e Ácidos Nucleicos; Dados Experimentais (X-ray, NMR, Cryo-EM); Metadados de Biocuração; Estruturas Integrativas; Modelos de Estrutura Computacional (CSM).

## Use Cases

Modelagem molecular e simulação; Descoberta e design de medicamentos (drug discovery); Análise de interações proteína-ligante; Estudos de dobramento e função de proteínas; Treinamento de modelos de Machine Learning para predição de estrutura e função.

## Integration

O acesso pode ser feito via interface web (rcsb.org) ou programaticamente. A RCSB PDB oferece o pacote Python `rcsb-api` (disponível no PyPI) para acesso simplificado aos serviços de busca e dados, permitindo consultas complexas e recuperação de dados via API REST e GraphQL. Exemplo de instalação: `pip install rcsb-api`. Exemplo de uso da API de Dados: `https://data.rcsb.org/rest/v1/core/entry/1A2C`.

## URL

https://www.rcsb.org/