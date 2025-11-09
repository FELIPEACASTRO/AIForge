# DeepBioFusion (Modelo), BioMassters (Dataset), DeepLiDARPlanet (Implementação)

## Description

Pesquisa abrangente sobre o uso de LiDAR e técnicas de aprendizado profundo para a estimativa de Biomassa Acima do Solo (AGB), focando em desenvolvimentos recentes (2023-2025). Foram identificados três recursos chave: um modelo de aprendizado profundo multimodal (DeepBioFusion), um dataset de benchmark (BioMassters) e um repositório de código para implementação (DeepLiDARPlanet). Os resultados detalham as características, estatísticas de desempenho, casos de uso e informações de integração para cada recurso.

## Statistics

- **DeepBioFusion:** Coeficiente de correlação de 0.57 (banda L do SAR) entre valores de biomassa estimados e previstos. Erros de previsão local (AGB) dentro da faixa de ±5.1%.
- **BioMassters:** Dataset lançado em 2023, citado 17 vezes (NeurIPS 2023). Contém dados de 5 anos (2016-2021) para quase 13.000 patches de floresta na Finlândia.
- **DeepLiDARPlanet:** Repositório com 7 estrelas e 1 fork (em 2025). Foca em modelos U-Net para previsão de altura e biomassa em 100m de resolução espacial.

## Features

- **DeepBioFusion:** Framework de aprendizado profundo multimodal (SAR e óptico) para estimativa de AGB em nível de árvore. Utiliza dados derivados de LiDAR para geração de verdade terrestre (ground truth) e incorpora modelagem de espécies.
- **BioMassters:** Dataset de benchmark com séries temporais multimodais (Sentinel-1 SAR e Sentinel-2 MSI) para estimativa de AGB em florestas finlandesas.
- **DeepLiDARPlanet:** Repositório com ferramentas e modelos para estimativa de altura e biomassa florestal, integrando dados LiDAR e Planet, com foco em modelos U-Net para previsão em 100m de resolução.

## Use Cases

- **DeepBioFusion:** Monitoramento de biomassa em larga escala, mitigação de mudanças climáticas, gestão florestal sustentável e avaliação de risco de incêndios florestais.
- **BioMassters:** Treinamento e benchmark de modelos de aprendizado profundo para estimativa de AGB usando dados de satélite multimodais.
- **DeepLiDARPlanet:** Criação de mapas contínuos de vegetação para gestão florestal, monitoramento de reflorestamento/desmatamento e classificação de cobertura do solo em regiões tropicais (ex: Kalimantan).

## Integration

- **DeepBioFusion:** Não há código de integração direta disponível no artigo, mas o framework é baseado em aprendizado profundo e utiliza dados de satélite (SAR e óptico) e dados de referência derivados de LiDAR.
- **BioMassters:** O dataset está disponível no Hugging Face e a competição foi hospedada no DrivenData. O código de baseline está acessível no GitHub (link no URL).
- **DeepLiDARPlanet:** O repositório GitHub fornece a estrutura de pastas e scripts Python para pré-tratamento de dados (CHM, Planet) e modelagem (U-Net, Random Sampling), com requisitos de instalação (Python 3.11.4, PyTorch 11.7, Rasterio, GDAL). O código de integração é baseado em Python/PyTorch.

## URL

DeepBioFusion: https://www.sciencedirect.com/science/article/pii/S1574954125002869 | BioMassters: https://nascetti-a.github.io/BioMasster/ | DeepLiDARPlanet: https://github.com/fraware/LiDAR-Planet-Deep-Learning