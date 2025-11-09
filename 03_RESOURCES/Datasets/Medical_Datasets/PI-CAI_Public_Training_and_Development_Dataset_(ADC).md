# PI-CAI Public Training and Development Dataset (ADC)

## Description

O PI-CAI (Prostate Cancer Artificial Intelligence) é um desafio de grande escala que sucedeu o PROSTATEx, focado na detecção e diagnóstico de câncer de próstata clinicamente significativo (csPCa) usando ressonância magnética multiparamétrica (mpMRI). O dataset público para treinamento e desenvolvimento contém 1500 exames de mpMRI, incluindo dados de T2-weighted (T2W), DWI (Diffusion-Weighted Imaging) e mapas de Coeficiente de Difusão Aparente (ADC). Embora o dataset público não inclua Ktrans (Volume Transfer Constant) ou SUV (Standardized Uptake Value), ele é fundamental para a pesquisa em biomarcadores de imagem, pois o ADC é um dos três biomarcadores solicitados e o dataset é o mais recente e robusto em seu domínio. O desafio PI-CAI é um recurso essencial para o desenvolvimento de algoritmos de IA em radiologia.

## Statistics

**Tamanho do Dataset Público:** 1500 exames de mpMRI. **Tamanho Total do Cohort:** Mais de 10.000 exames. **Origem:** Multi-centro (4 centros na Holanda e Noruega) e multi-vendor (Siemens e Philips). **Sequências:** T2W, DWI (high b-value), ADC. **Licença:** CC BY-NC 4.0. **Status:** Ativo e em uso para pesquisa de ponta (2023-2025).

## Features

**Biomarcadores de Imagem Incluídos:** Coeficiente de Difusão Aparente (ADC). **Sequências de Imagem:** T2-weighted (T2W), DWI (high b-value). **Rótulos:** Anotações de csPCa (câncer de próstata clinicamente significativo) e informações clínicas básicas (idade do paciente, volume da próstata, PSA, densidade de PSA). **Características:** Dataset multi-centro e multi-vendor, com 1500 casos públicos para treinamento e desenvolvimento. O dataset total inclui mais de 10.000 exames. Não inclui Ktrans ou SUV no conjunto de dados de treinamento público para algoritmos de IA.

## Use Cases

**Desenvolvimento de Algoritmos de IA:** Treinamento e validação de modelos de *Deep Learning* para detecção e diagnóstico de câncer de próstata clinicamente significativo (csPCa). **Radiômica:** Extração de features radiômicas de mapas ADC para predição de escore de Gleason e resposta ao tratamento. **Comparação Humano-Máquina:** Benchmarking do desempenho de algoritmos de IA contra radiologistas experientes. **Pesquisa em Biomarcadores:** Estudo da utilidade do ADC como biomarcador quantitativo em mpMRI.

## Integration

O dataset público de treinamento e desenvolvimento (1500 casos) está disponível via Zenodo e as anotações via GitHub.

**Acesso ao Dataset:**
`zenodo.org/record/6624726` (DOI: 10.5281/zenodo.6624726)

**Acesso às Anotações (Labels):**
`github.com/DIAGNijmegen/picai_labels`

**Exemplo de Extração de Features (Conceitual, usando PyRadiomics para ADC):**
A extração de features radiômicas de imagens ADC (e Ktrans/SUV, se disponíveis em outros conjuntos de dados) pode ser realizada com a biblioteca PyRadiomics.

```python
from radiomics import featureextractor
import SimpleITK as sitk

# 1. Carregar a imagem ADC e a máscara de segmentação
# image_path deve ser o caminho para o arquivo .mha do ADC
# mask_path deve ser o caminho para o arquivo .mha da máscara de segmentação
image = sitk.ReadImage("caminho/para/imagem_adc.mha")
mask = sitk.ReadImage("caminho/para/mascara_segmentacao.mha")

# 2. Configurar o extrator de features
# Um arquivo de parâmetros YAML pode ser usado para especificar quais features extrair
extractor = featureextractor.RadiomicsFeatureExtractor()

# 3. Executar a extração
result = extractor.execute(image, mask)

# 4. Imprimir as features extraídas
print("Extração de Features Radiômicas do ADC:")
for key, val in result.items():
    print(f"\t{key}: {val}")
```

## URL

https://pi-cai.grand-challenge.org/