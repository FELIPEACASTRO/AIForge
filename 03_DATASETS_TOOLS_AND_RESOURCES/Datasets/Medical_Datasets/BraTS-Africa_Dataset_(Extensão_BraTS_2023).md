# BraTS-Africa Dataset (Extensão BraTS 2023)

## Description

O BraTS-Africa Dataset é uma extensão crucial do desafio BraTS (Brain Tumor Segmentation), lançado em 2023, com o objetivo de expandir a diversidade dos dados de ressonância magnética (RM) de tumores cerebrais para incluir populações da África Subsaariana (SSA). É o primeiro conjunto de dados de imagem cerebral anotado e publicamente disponível da África, abordando a falta de representação geográfica nos datasets de IA médica. O dataset foi curado a partir de seis centros de diagnóstico na Nigéria e é fundamental para o desenvolvimento de modelos de segmentação de tumores cerebrais mais generalizáveis e equitativos, especialmente em ambientes com recursos limitados.

## Statistics

**Tamanho:** 3.7 GB. **Pacientes:** 146 pacientes. **Origem:** Seis centros de diagnóstico na Nigéria. **Período de Aquisição:** Janeiro de 2010 a Dezembro de 2022. **Modalidades:** T1, T1 CE, T2, T2 FLAIR. **Formato:** NIfTI. **Anotações:** Segmentações de sub-regiões do tumor (Núcleo, Edema, Aprimorado).

## Features

**Imagens de RM Multiparamétricas (mpMRI):** Inclui T1, T1 com contraste (T1 CE), T2 e T2 FLAIR. **Anotações de Segmentação:** Segmentações de sub-regiões do tumor (Núcleo do Tumor Necrótico/Não Aprimorado, Edema Peritumoral e Tumor Aprimorado) anotadas por especialistas. **Características Radiômicas:** A pesquisa recente (2024-2025) demonstra o uso de técnicas de **Radiômica** para extrair centenas de características quantitativas (como forma, intensidade e textura) das regiões tumorais e peritumorais, visando a predição de subtipos de glioma e a avaliação de resposta ao tratamento. Ferramentas como `pyradiomics` são comumente usadas para essa extração.

## Use Cases

**Segmentação de Tumores Cerebrais:** Treinamento e avaliação de modelos de Deep Learning (como U-Net e MedNeXt) para segmentação automática de gliomas em populações africanas. **Generalização de Modelos:** Uso como dataset de teste para avaliar a robustez e a equidade de modelos treinados em dados predominantemente ocidentais. **Pesquisa em Radiômica:** Extração de características radiômicas para predição de prognóstico, subtipos moleculares de glioma e avaliação de resposta à terapia. **Desenvolvimento de Ferramentas de Baixo Recurso:** Criação de modelos leves e eficientes para uso em ambientes clínicos com recursos computacionais limitados.

## Integration

O dataset está disponível através do **The Cancer Imaging Archive (TCIA)**. O acesso aos dados de imagem e segmentação requer o uso do plugin IBM-Aspera-Connect para download. Alternativamente, o dataset pode ser acessado via **Kaggle** (versões não oficiais ou menores) ou através de repositórios de projetos de pesquisa que o utilizam.

**Exemplo de Acesso (TCIA - Requer Download):**
1.  Acessar a página do dataset no TCIA.
2.  Utilizar o plugin IBM-Aspera-Connect para baixar o arquivo `Radiology Images and Segmentations - BraTS 2023 Challenge` (que inclui os dados do BraTS-Africa).

**Exemplo de Integração (Python - Conceitual para Radiômica):**
```python
# Exemplo conceitual de extração de características radiômicas
# Requer o download prévio dos arquivos NIfTI do TCIA
import SimpleITK as sitk
from radiomics import featureextractor

# 1. Carregar a imagem e a máscara de segmentação
image_path = 'caminho/para/imagem_T1CE.nii.gz'
mask_path = 'caminho/para/mascara_segmentacao.nii.gz'
image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

# 2. Configurar o extrator de características
extractor = featureextractor.RadiomicsFeatureExtractor()
# Configurações podem ser ajustadas para extrair características específicas (e.g., GLCM, GLRLM)

# 3. Executar a extração
result = extractor.execute(image, mask)

# 4. Exibir as características extraídas
# print(result)
```

## URL

https://www.cancerimagingarchive.net/collection/brats-africa/