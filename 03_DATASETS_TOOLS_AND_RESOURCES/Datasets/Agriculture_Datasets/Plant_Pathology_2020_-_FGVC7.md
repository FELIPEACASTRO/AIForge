# Plant Pathology 2020 - FGVC7

## Description

Conjunto de dados de imagens de folhas de macieira (apple leaves) utilizado na competição Kaggle Fine-Grained Visual Categorization 7 (FGVC7) em 2020. O objetivo é classificar as imagens em quatro categorias de saúde/doença: folha saudável, ferrugem da macieira (apple rust), sarna da macieira (apple scab) e múltiplas doenças (combinations). O dataset é um recurso fundamental para o desenvolvimento de modelos de Visão Computacional (CNNs, Vision Transformers) para diagnóstico de doenças em plantas.

## Statistics

Tamanho: 823.79 MB. Contém 3645 arquivos (imagens JPG e arquivos CSV de metadados). As classes de destino são: 'healthy', 'rust', 'scab' e 'combinations'. O conjunto de treinamento possui 3645 imagens, e o conjunto de teste possui 1821 imagens (no dataset original da competição).

## Features

Imagens RGB de alta resolução de folhas de macieira. As técnicas de feature engineering mais recentes (2023-2025) envolvem o uso de modelos de Deep Learning pré-treinados (como ResNet, EfficientNet, Vision Transformers - ViT) para extração automática de características (feature extraction), além de técnicas de aumento de dados (data augmentation) como rotação, zoom, e ajustes de cor para melhorar a robustez do modelo.

## Use Cases

Diagnóstico automatizado de doenças em plantas por meio de imagens. Desenvolvimento de sistemas de alerta precoce para agricultores. Aplicações em agricultura de precisão para monitoramento da saúde das culturas. Pesquisa em Visão Computacional e Deep Learning para classificação fina (Fine-Grained Visual Categorization).

## Integration

O dataset pode ser acessado diretamente via API do Kaggle (requer autenticação) ou baixado da página da competição. A integração em projetos de código geralmente envolve bibliotecas como TensorFlow ou PyTorch. Exemplo de código para download via Kaggle CLI:\n\n```bash\nkaggle competitions download -c plant-pathology-2020-fgvc7\nunzip plant-pathology-2020-fgvc7.zip\n```\n\nEm Python, a integração para treinamento de modelos utiliza DataLoaders para carregar as imagens e os rótulos do arquivo `train.csv`.

## URL

https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7