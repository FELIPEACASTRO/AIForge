# PyRadiomics (Feature Extraction Tool)

## Description

Pacote Python de código aberto e amplamente adotado para a extração de features radiômicas de imagens médicas (2D e 3D) e máscaras binárias. O PyRadiomics implementa as definições de features padronizadas pela Iniciativa de Padronização de Biomarcadores de Imagem (**IBSI**), garantindo reprodutibilidade. Ele é a ferramenta fundamental para a engenharia de features de **Textura**, **Forma** e **Intensidade** (Primeira Ordem) em projetos de Radiômica. O pacote é ativamente mantido e é compatível com formatos de imagem médica comuns como DICOM, NIfTI e NRRD.

## Statistics

O PyRadiomics é uma ferramenta de software, não um dataset.
*   **Features Extraídas:** Aproximadamente 1500 features por imagem (dependendo das configurações e filtros aplicados).
*   **Classes de Features:** 8 classes principais (1ª Ordem, 3D/2D Shape, GLCM, GLRLM, GLSZM, NGTDM, GLDM).
*   **Padrão:** Em conformidade com o padrão IBSI (Image Biomarker Standardization Initiative).
*   **Compatibilidade:** Suporta imagens 2D e 3D em formatos como DICOM, NIfTI e NRRD.
*   **Datasets de Referência:** Utilizado em conjunto com datasets padronizados como **Open-radiomics** (que inclui BraTS 2020/2023 e TCIA NSCLC) para garantir a reprodutibilidade.

## Features

O PyRadiomics extrai um grande número de features (aproximadamente 1500 por imagem, dependendo das configurações), categorizadas em:
*   **Features de Primeira Ordem (Intensidade):** 19 features, incluindo Média, Mediana, Desvio Padrão, Entropia, Skewness e Kurtosis.
*   **Features de Forma (Shape):** 16 features 3D e 10 features 2D, como Volume, Área de Superfície, Esfericidade e Compacidade.
*   **Features de Textura (Higher Order):** Incluem features de matrizes de co-ocorrência de nível de cinza (GLCM - 24 features), matrizes de comprimento de execução de nível de cinza (GLRLM - 16 features), matrizes de zona de tamanho de nível de cinza (GLSZM - 16 features), matrizes de diferença de tom de cinza vizinho (NGTDM - 5 features) e matrizes de dependência de nível de cinza (GLDM - 14 features).

## Use Cases

*   **Previsão de Prognóstico e Sobrevivência:** Utilização de features radiômicas para prever a resposta do paciente ao tratamento e a sobrevida em diversos tipos de câncer (ex: Câncer de Pulmão de Não Pequenas Células - NSCLC, Gliomas).
*   **Classificação de Tumores:** Distinção entre tumores de alto e baixo grau (ex: HGG vs. LGG no dataset BraTS 2023) com base nas características de textura e intensidade.
*   **Avaliação da Heterogeneidade Tumoral:** Quantificação da variação espacial e de intensidade dentro de uma Região de Interesse (ROI) para melhor caracterizar o fenótipo do tumor.
*   **Desenvolvimento de Biomarcadores de Imagem:** Criação de modelos preditivos e descritivos que correlacionam features de imagem com dados genômicos e clínicos (Radiogenômica).

## Integration

A extração de features é tipicamente realizada via Python, utilizando a classe `RadiomicsFeatureExtractor`.

```python
from radiomics import featureextractor
import SimpleITK as sitk

# 1. Inicializar o extrator de features
extractor = featureextractor.RadiomicsFeatureExtractor()

# Opcional: Configurar o extrator (ex: desabilitar features, mudar binWidth)
# extractor.disableAllFeatures()
# extractor.enableFeatureClassByName('FirstOrder')
# extractor.settings['binWidth'] = 25

# 2. Carregar a imagem e a máscara (ROI)
imageName = 'path/to/image.nrrd'
maskName = 'path/to/mask.nrrd'
image = sitk.ReadImage(imageName)
mask = sitk.ReadImage(maskName)

# 3. Extrair as features
result = extractor.execute(image, mask)

# 4. Imprimir o resultado
print('Extração concluída. Features extraídas:')
for featureName in result.keys():
    print(f'  {featureName}: {result[featureName]}')
```

O pacote também oferece uma interface de linha de comando (`pyradiomics`) para extração em lote. O código de exemplo e dados estão disponíveis no repositório oficial do GitHub.

## URL

https://pyradiomics.readthedocs.io/