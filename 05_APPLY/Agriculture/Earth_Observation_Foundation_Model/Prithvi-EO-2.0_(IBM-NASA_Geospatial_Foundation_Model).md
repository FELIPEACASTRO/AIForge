# Prithvi-EO-2.0 (IBM-NASA Geospatial Foundation Model)

## Description

O Prithvi-EO-2.0 é um modelo fundamental geoespacial de segunda geração, desenvolvido em conjunto pela IBM, NASA e Jülich Supercomputing Centre. É um modelo de Visão (ViT) baseado em Transformer, pré-treinado usando uma abordagem de autoencoder mascarado (MAE). Foi treinado em 4.2M de amostras de séries temporais globais do arquivo Harmonized Landsat and Sentinel-2 (HLS) da NASA com resolução de 30m. O modelo incorpora embeddings temporais e de localização para um desempenho aprimorado em várias tarefas geoespaciais. É disponibilizado em diferentes tamanhos de parâmetros (300M e 600M), com e sem embeddings de tempo/localização (TL).

## Statistics

**Desempenho:** A versão 600M supera o modelo anterior Prithvi-EO em 8% no GEO-Bench. Supera outros seis modelos fundamentais geoespaciais em tarefas de sensoriamento remoto. **Downloads:** 54.018 downloads no último mês (dados de 2024). **Citação:** Artigo técnico (arXiv:2412.02732) publicado em 2024, com mais de 60 citações.

## Features

Arquitetura ViT com modificações 3D para dados espaço-temporais. Suporte a embeddings temporais e de localização. Pré-treinamento com dados HLS (Landsat e Sentinel-2) de 30m. Disponível em versões de 300M e 600M, com e sem TL (Temporal/Location embeddings). Licença Apache-2.0. Utiliza a estrutura IBM TerraTorch para fine-tuning.

## Use Cases

Mapeamento de uso da terra e culturas (crop mapping). Monitoramento da dinâmica de ecossistemas. Aplicações de alta resolução (0.1m a 15m). Resposta a desastres. Previsão de Fluxo de Carbono (Regressão). Segmentação de deslizamentos de terra. Segmentação multitemporal de culturas.

## Integration

O modelo pode ser utilizado para fine-tuning usando a estrutura TerraTorch da IBM. Para construir o backbone em um pipeline PyTorch personalizado:

```python
from terratorch.registry import BACKBONE_REGISTRY

# Exemplo para a versão tiny com embeddings de tempo/localização
model = BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl", pretrained=True)
```

Exemplo de script de inferência para reconstrução de imagem:

```bash
python inference.py --data_files t1.tif t2.tif t3.tif t4.tif --input_indices <optional, space separated 0-based indices of the six Prithvi channels in your input>
```

Notebooks de exemplo para fine-tuning estão disponíveis no GitHub do projeto.

## URL

https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M