# RoboNet

## Description
O RoboNet é um banco de dados aberto e diversificado para o compartilhamento de experiências robóticas, focado em aprendizado em larga escala e multi-robô. Ele foi projetado para permitir que modelos de alta capacidade, como redes neurais profundas, generalizem de forma eficaz para uma ampla gama de ambientes do mundo real. O dataset é composto por interações robô-objeto capturadas de múltiplos pontos de vista e plataformas robóticas. O objetivo principal é pré-treinar modelos de aprendizado por reforço em um conjunto de dados diversificado e, em seguida, transferir o conhecimento para novos robôs e tarefas com muito menos dados específicos.

## Statistics
- **Tamanho do Dataset (Completo):** Download de 36.20 GiB, tamanho em disco de 144.90 GiB (configuração 128x128).
- **Amostras (Trajetórias):** Aproximadamente 162.417 trajetórias de treinamento (na versão completa).
- **Quadros de Vídeo:** Mais de 15 milhões de quadros de vídeo de interações robô-objeto.
- **Plataformas Robóticas:** 7 plataformas robóticas diferentes.
- **Pontos de Vista:** 113 pontos de vista de câmera únicos.
- **Versão (TFDS):** `4.0.1` (padrão).
- **Resoluções Disponíveis (TFDS):** 64x64 e 128x128.

## Features
- **Aprendizado Multi-Robô em Larga Escala:** O dataset permite o treinamento de modelos generalizáveis para manipulação robótica baseada em visão em diversas plataformas.
- **Diversidade de Plataformas:** Inclui dados de 7 plataformas robóticas diferentes, desde braços industriais Kuka até braços de baixo custo WidowX.
- **Múltiplos Pontos de Vista:** As interações são capturadas de 113 pontos de vista de câmera únicos, aumentando a robustez visual dos modelos treinados.
- **Dados de Ação e Estado:** Além dos quadros de vídeo, o dataset inclui ações (deltas de posição e rotação do efetor final, mais a junta da garra) e estados (espaço de ação de controle cartesiano do efetor final e junta da garra).
- **Transferência de Conhecimento:** Demonstra a capacidade de pré-treinar em RoboNet e ajustar (fine-tuning) em dados de um robô específico (como Franka ou Kuka) para superar o treinamento específico do robô com 4x a 20x mais dados.

## Use Cases
- **Pré-treinamento para Aprendizado por Reforço (RL):** Usado para pré-treinar modelos de RL em um conjunto de dados diversificado antes de ajustar para tarefas específicas.
- **Previsão de Vídeo:** Treinamento de modelos de previsão de vídeo para antecipar interações robóticas.
- **Modelos Inversos Supervisionados:** Treinamento de modelos para inferir ações a partir de observações visuais.
- **Generalização de Habilidades:** Estudo da capacidade de generalizar controladores robóticos para novos objetos, tarefas, cenas, pontos de vista de câmera, garras ou até mesmo robôs inteiramente novos.
- **Pesquisa em Visão Computacional para Robótica:** Fornece um benchmark para o desenvolvimento de algoritmos de visão para manipulação robótica.

## Integration
O dataset RoboNet pode ser acessado e utilizado de várias maneiras:

1.  **TensorFlow Datasets (TFDS):** A maneira mais fácil de usar o dataset, especialmente para modelos em TensorFlow.
    *   **Instalação:** `pip install tensorflow-datasets`
    *   **Uso:** O dataset pode ser carregado diretamente no código Python:
        ```python
        import tensorflow_datasets as tfds
        ds = tfds.load('robonet/robonet_128', split='train', shuffle_files=True)
        ```
    *   **Configurações:** Estão disponíveis diferentes configurações (e.g., `robonet_sample_64`, `robonet_128`) que variam em tamanho e resolução.

2.  **Download Direto (HDF5):** O dataset completo (36 GB) ou uma amostra (~100 MB) pode ser baixado diretamente usando a ferramenta `gdown` e descompactado.
    *   **Instalação:** `pip install gdown`
    *   **Download Completo (36 GB):**
        ```bash
        gdown https://drive.google.com/a/andrew.cmu.edu/uc?id=1BkqHzfRkfzgzCfc73NbNnPMK_rg3i1n9&export=download
        tar -xzvf robonet_v3.tar.gz
        ```
    *   **Download Amostra (~100 MB):**
        ```bash
        gdown https://drive.google.com/uc?id=1YX2TgT8IKSn9V4wGCwdzbRnS53yicV2P&export=download
        tar -xvzf robonet_sampler.tar.gz
        ```

3.  **Código-Fonte e Utilitários:** O repositório oficial do GitHub fornece código para carregar, manipular e treinar modelos (como modelos inversos supervisionados e modelos de previsão de vídeo) no dataset.
    *   **Repositório:** `git clone https://github.com/SudeepDasari/RoboNet.git`
    *   **Instalação:** `pip install -r requirements.txt` e `python setup.py develop`

## URL
[https://www.robonet.wiki/](https://www.robonet.wiki/)
