# Global Maps of Aboveground Biomass, Canopy Height and Cover for 2023, Cropland Data Layer (CDL) 2024 - 10m Resolution, Geospatial Feature Engineering Techniques (GLCM and CAE-based Features)

## Description

Conjunto de dados global de densidade de biomassa acima do solo (AGBD), altura de dossel (CH) e cobertura de dossel (CC) com resolução de 10 metros, derivado de um Modelo Unificado de Deep Learning aplicado a imagens de satélite multi-sensor de alta resolução. Os dados são acompanhados por mapas de erro padrão para cada variável. O Cropland Data Layer (CDL) é um mapa de cobertura terrestre raster, geo-referenciado e específico para culturas, produzido anualmente pelo USDA National Agricultural Statistics Service (NASS). A versão de 2024 marca a transição da resolução espacial de 30 metros para 10 metros, fornecendo um detalhamento de campo sem precedentes. Técnicas avançadas de engenharia de features espaciais, como a Matriz de Co-ocorrência de Nível de Cinza (GLCM) para extração de textura e features baseadas em Autoencoders Convolucionais (CAE) para extração de features latentes, são cruciais para melhorar a precisão em modelos de Biomassa e Agropecuária.

## Statistics

Ano: 2023. Resolução: 10 metros. Cobertura: Global (57° S a 67° N). Tamanho Total: 5 TB (Conjunto de Dados Completo). Variáveis (6 bandas GeoTIFF): AGBD (Mg/ha), CH (cm), CC (%), Erro Padrão de AGBD, Erro Padrão de CH, Erro Padrão de CC. Ano: 2024 (Lançado em Fev/2025). Resolução: 10 metros (Nova resolução) e 30 metros (versão reamostrada). Cobertura: Estados Unidos (Continental e Havaí). Classes: Mais de 110 categorias de culturas e 14 categorias não-agrícolas. GLCM: Usada em estudos de 2024 e 2025 para classificação de culturas e avaliação de carbono orgânico. CAE-based Features: Estudos de 2025 mostram melhoria de até 10% na previsão de rendimento de culturas em comparação com autoencoders tradicionais.

## Features

Alta resolução (10m); Fusão multi-sensor (Modelo de Deep Learning); Inclui estimativas de incerteza (Erro Padrão); Cobertura global para 2023. Alta resolução (10m) para detalhamento de campo; Específico para culturas (identificação de tipos de culturas); Base para o Agricultural Cropland Tracking and Interactive Visualization Environment (ACTIVE) no Google Earth Engine; Consistência histórica com dados de 30m. GLCM (Gray-Level Co-occurrence Matrix): Extrai features de textura (Contraste, Homogeneidade, Entropia, Correlação) capturando a relação espacial entre pixels. CAE (Convolutional Autoencoder) Latent Features: Aprende representações compactas de dados de satélite de alta dimensão, separando features de genótipo e ambiente.

## Use Cases

Mitigação de mudanças climáticas e contabilidade de carbono (AGBD); Gestão e inventário florestal (CH, CC); Modelagem ecológica e estudos de biodiversidade; Monitoramento agrícola (Cobertura de Dossel). Avaliação de área plantada e produção agrícola; Monitoramento de mudanças na cobertura terrestre; Modelagem de rendimento de culturas; Análise de políticas agrícolas e ambientais. Classificação de culturas e mapeamento de uso da terra (GLCM); Estimativa de biomassa e carbono orgânico (GLCM); Previsão de rendimento de culturas em estágios iniciais (CAE); Detecção e classificação de doenças em plantas (CAE).

## Integration

Formato: Arquivos GeoTIFF (divisões de 3° x 3°). Acesso: Zenodo (subamostra de 48 GB) ou serviço Requester-Pays do AWS S3 (conjunto de dados completo). Comando AWS S3: aws s3 sync s3://eda-appsci-open-access/biomass/ DESTINATION_PATH --request-payer requester. Acesso: Disponível no aplicativo web CroplandCROS. Download: Versão de 30m disponível para download. A versão de 10m pode ser acessada via CroplandCROS e possivelmente via Google Earth Engine (ACTIVE). GLCM: Implementada em bibliotecas como `scikit-image` (Python) ou Google Earth Engine. CAE-based Features: Requer implementação e treinamento de modelos de Autoencoder Convolucional usando frameworks de Deep Learning (TensorFlow/PyTorch).

## URL

https://zenodo.org/records/15269923, https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php, Referências de artigos científicos (e.g., MDPI, Frontiers in Plant Science, IEEE Xplore) de 2024-2025.