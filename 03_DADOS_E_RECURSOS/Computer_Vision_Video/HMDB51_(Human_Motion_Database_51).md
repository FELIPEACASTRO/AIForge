# HMDB51 (Human Motion Database 51)

## Description
O HMDB51 (Human Motion Database 51) é um dataset de grande escala projetado para o reconhecimento de ações humanas em vídeos. Foi criado para abordar a limitação de datasets anteriores, que continham poucas categorias de ação e eram coletados sob condições controladas. O HMDB51 é composto por clipes de vídeo extraídos de diversas fontes, como filmes e vídeos do YouTube, o que introduz variações significativas em termos de movimento de câmera, ponto de vista, qualidade de vídeo e oclusão, tornando-o um benchmark desafiador e realista para algoritmos de visão computacional.

## Statistics
- **Número de Clipes:** 6.766 vídeos (cada categoria contém no mínimo 101 clipes).
- **Número de Classes:** 51 categorias de ações humanas.
- **Tamanho Aproximado:** Cerca de 2 GB (dados brutos de vídeo).
- **Versão:** A versão original foi publicada em 2011, mas continua sendo um benchmark padrão e é frequentemente reempacotada em plataformas como Hugging Face e Kaggle.

## Features
- **51 Categorias de Ação:** Inclui ações diversas como "jump", "drink", "kiss", "laugh", "climb", "shake hands", entre outras.
- **Vídeos Não-Controlados:** Os clipes foram coletados de fontes do mundo real, resultando em variações complexas de fundo, iluminação e movimento.
- **Formato:** Clipes de vídeo no formato `.avi`.
- **Divisões Oficiais:** O dataset fornece três divisões de treinamento/teste (splits) para avaliação de desempenho padronizada.

## Use Cases
- **Reconhecimento de Ação Humana (HAR):** Classificação de ações em sequências de vídeo.
- **Visão Computacional em Vídeo:** Desenvolvimento e avaliação de modelos de *deep learning* 3D (como I3D, R(2+1)D) para processamento de vídeo.
- **Transfer Learning:** Uso do dataset para *fine-tuning* de modelos pré-treinados em datasets maiores (como Kinetics).
- **Análise de Movimento:** Pesquisa sobre a robustez de algoritmos sob variações de câmera, oclusão e fundo.

## Integration
O dataset HMDB51 pode ser acessado e utilizado de diversas formas, sendo a mais comum através de bibliotecas de aprendizado de máquina que o integram:

1.  **Hugging Face Datasets:** Pode ser carregado diretamente usando a biblioteca `datasets` do Hugging Face, o que facilita o pré-processamento e o uso em modelos de *deep learning*.
    ```python
    from datasets import load_dataset
    dataset = load_dataset("jili5044/hmdb51")
    ```
2.  **PyTorch Torchvision:** A biblioteca `torchvision` oferece uma classe dedicada (`torchvision.datasets.HMDB51`) para download e carregamento do dataset, exigindo que o usuário baixe os arquivos brutos do site oficial e os organize em uma estrutura específica.
3.  **Download Direto:** Os arquivos brutos (vídeos e divisões) podem ser baixados do site oficial do Serre Lab (Brown University) após a aceitação dos termos de uso. O dataset é frequentemente usado em conjunto com o UCF101 para benchmarks de reconhecimento de ação.

## URL
[http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
