# TITAN (Transformer-based pathology Image and Text Alignment Network)

## Description

O TITAN (Transformer-based pathology Image and Text Alignment Network) é um modelo de fundação multimodal de última geração para patologia digital. Ele foi pré-treinado usando 335.645 imagens de lâminas inteiras (WSIs) e alinhamento visão-linguagem com relatórios de patologia correspondentes, além de 423.122 legendas sintéticas geradas por um copiloto de IA generativa multimodal. Embora utilize uma arquitetura Vision Transformer (ViT), ele se baseia em recursos extraídos por modelos como o CONCHv1.5, que são essenciais para a análise de imagens de patologia em larga escala. O TITAN gera representações de lâminas de propósito geral e pode gerar relatórios de patologia, generalizando para cenários clínicos com recursos limitados, como a recuperação de doenças raras e o prognóstico de câncer.

## Statistics

- **Base de Dados de Pré-treinamento**: 335.645 Imagens de Lâminas Inteiras (WSIs) e 182.862 relatórios médicos.
- **Desempenho (Subtipagem)**: Supera modelos de fundação anteriores (como PRISM, GigaPath e CHIEF) com uma melhoria média de **+8,4%** em tarefas de subtipagem multiclasse e **+6,7%** em tarefas binárias.
- **Citações**: 79 citações (em novembro de 2025, conforme Nature Medicine).
- **Publicação**: *Nature Medicine* (2025).
- **Recuperação de Câncer Raro**: Demonstra desempenho superior em precisão@K e MVAcc@K em conjuntos de dados de câncer raro.

## Features

- **Multimodalidade**: Alinha representações visuais de WSIs com descrições morfológicas e relatórios clínicos (texto).
- **Representação de Propósito Geral**: Extrai *embeddings* de lâminas que podem ser implantados em diversas configurações clínicas sem a necessidade de *fine-tuning* extensivo.
- **Modelagem de Contexto Longo**: Utiliza ALiBi para modelagem de contexto longo, permitindo previsões precisas em contextos de tecido inteiros.
- **Geração de Relatórios**: Capacidade de gerar relatórios de patologia de alta qualidade, capturando local do tecido, diagnóstico e grau do tumor.
- **Arquitetura**: Baseado em Vision Transformer (ViT) com destilação de conhecimento aluno-professor, pré-treinado em três estágios (visão unimodal, alinhamento *cross-modal* ROI-nível e WSI-nível).

## Use Cases

- **Classificação e Subtipagem de Câncer**: Identificação de subtipos morfológicos e classificação molecular em WSIs.
- **Previsão de Biomarcadores**: Previsão de biomarcadores moleculares diretamente a partir de imagens histopatológicas.
- **Recuperação de Casos Raros**: Recuperação de casos clinicamente relevantes e morfologicamente semelhantes para doenças raras ou complexas.
- **Prognóstico de Câncer**: Auxílio na previsão de resultados clínicos e sobrevivência.
- **Geração de Relatórios**: Geração automatizada de relatórios de patologia de alta qualidade.

## Integration

A integração é feita através da biblioteca Python do modelo, que permite a extração de *embeddings* de lâminas a partir de recursos de *patch* pré-extraídos (por exemplo, usando CONCHv1.5 ou CLAM).

**Exemplo de Extração de *Embedding* de Lâmina (Python/PyTorch):**

```python
import torch
import h5py
from huggingface_hub import hf_hub_download

# O modelo TITAN deve ser carregado previamente
# model = ... 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Carregar dados de amostra (recursos de patch pré-extraídos)
demo_h5_path = hf_hub_download(
    "MahmoodLab/TITAN", 
    filename="TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5",
)
file = h5py.File(demo_h5_path, 'r')
features = torch.from_numpy(file['features'][:])
coords = torch.from_numpy(file['coords'][:])
patch_size_lv0 = file['coords'].attrs['patch_size_level0']

# Extrair o embedding da lâmina
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    features = features.to(device)
    coords = coords.to(device)
    slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)

# 'slide_embedding' contém a representação vetorial da lâmina inteira.
```

O projeto também recomenda o uso do **TRIDENT** para extração de recursos em larga escala. O acesso aos pesos do modelo é feito através da página do Hugging Face do MahmoodLab.

## URL

https://www.nature.com/articles/s41591-025-03982-3