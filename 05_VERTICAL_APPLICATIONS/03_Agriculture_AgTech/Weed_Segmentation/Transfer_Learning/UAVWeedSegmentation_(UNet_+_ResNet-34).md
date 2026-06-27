# UAVWeedSegmentation (UNet + ResNet-34)

## Description

O recurso é um repositório de código-fonte que implementa uma solução de Deep Learning para a **Segmentação Precoce de Ervas Daninhas** em imagens de VANT (Veículo Aéreo Não Tripulado) de campos de sorgo. O foco principal é lidar com imagens com desfoque de movimento (motion blur), um desafio comum em capturas aéreas. O modelo principal utilizado é uma arquitetura **UNet** com um *backbone* **ResNet-34**, que é um exemplo de aplicação de *Transfer Learning* (Aprendizado por Transferência) para tarefas de segmentação semântica em agricultura de precisão. O repositório fornece o código para treinamento, retreinamento, previsão e comparação de resultados com o *Ground Truth*. A publicação associada é de 2022, mas o repositório é um recurso chave para implementação prática de modelos de DL em agricultura.

## Statistics

- **Modelo de Melhor Desempenho**: UNet + ResNet-34.
- **Métricas por Classe (Test-Set)**:
    - **Background**: Precisão (99.80%), Recall (99.93%), F1-Score (99.86%).
    - **Sorghum**: Precisão (91.58%), Recall (86.10%), F1-Score (88.76%).
    - **Weed**: Precisão (87.64%), Recall (72.71%), F1-Score (79.48%).
- **Média Macro**: Precisão (93.01%), Recall (86.25%), F1-Score (89.37%).
- **Citação (Artigo de 2022)**: O artigo "Deep Learning-based Early Weed Segmentation using Motion Blurred UAV Images of Sorghum Fields" foi publicado em *Computers and Electronics in Agriculture* (DOI: 10.1016/j.compag.2022.107388). Embora a publicação seja de 2022, o recurso é uma implementação de DL relevante e atualizada para a área.

## Features

- **Modelo de Segmentação Semântica**: UNet com ResNet-34 como extrator de características.
- **Foco em Imagens com Desfoque de Movimento**: O modelo foi treinado e avaliado em um conjunto de dados que inclui imagens com diferentes graus de desfoque de movimento.
- **Segmentação Multiclasse**: Classifica pixels em três classes: Fundo (Background), Sorgo (Sorghum) e Erva Daninha (Weed).
- **Scripts de Implementação**: Inclui scripts para geração de patches, treinamento (incluindo retreinamento com o conjunto completo), previsão e avaliação de desempenho.
- **Transfer Learning**: Utiliza a arquitetura ResNet-34, pré-treinada em grandes conjuntos de dados, como base para a tarefa de segmentação.

## Use Cases

- **Mapeamento de Ervas Daninhas**: Criação de mapas de segmentação de alta resolução para identificar a localização exata das ervas daninhas.
- **Aplicação de Herbicidas de Taxa Variável**: Utilização dos mapas de segmentação para guiar a aplicação de herbicidas apenas nas áreas infestadas, otimizando o uso de insumos (Agricultura de Precisão).
- **Monitoramento de Culturas**: Avaliação da saúde e densidade de culturas (sorgo) em estágios iniciais.
- **Mitigação de Desfoque de Movimento**: Demonstração de um modelo robusto para lidar com a qualidade de imagem variável de capturas de VANT.

## Integration

O repositório fornece um guia de instalação e uso detalhado. A integração envolve:
1.  **Clonagem do Repositório**: `git clone git@github.com:grimmlab/UAVWeedSegmentation.git`
2.  **Instalação de Dependências**: `pip install -r requirements.txt` (Python 3.8, PyTorch 1.11.0, CUDA).
3.  **Download do Modelo Pré-treinado**: Baixar o modelo do Mendeley Data e salvar em `/models` com o nome `model_unet_resnet34_dil0_bilin1_retrained.pt`.
4.  **Previsão em Novas Imagens**:
    ```bash
    python3 save_patches.py
    python3 predict_testset.py models/model_unet_resnet34_dil0_bilin1_retrained.pt test
    ```
5.  **Treinamento de Novos Modelos**: O script `train.py` permite treinar novas arquiteturas (fcn, unet, dlplus) com diferentes *feature extractors* (resnet18, resnet34, resnet50, resnet101).

## URL

https://github.com/grimmlab/UAVWeedSegmentation