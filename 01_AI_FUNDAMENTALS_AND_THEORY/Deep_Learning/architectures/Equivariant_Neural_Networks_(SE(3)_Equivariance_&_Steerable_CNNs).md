# Equivariant Neural Networks (SE(3) Equivariance & Steerable CNNs)

## Description

Redes Neurais Equivariantes (ENNs) são uma classe de arquiteturas de aprendizado profundo projetadas para incorporar simetrias geométricas inerentes aos dados, como rotação e translação. A **equivariância SE(3)** refere-se à equivariância sob o grupo de movimentos rígidos tridimensionais (Special Euclidean Group in 3D), que inclui rotações e translações. Isso garante que a representação interna de um objeto mude de forma previsível (equivariante) quando o objeto é transformado, e que a saída final da rede seja invariante ou equivariante à transformação de entrada. As **Steerable CNNs (CNNs Direcionáveis)** são um framework que implementa essa equivariância, onde os filtros convolucionais são restritos a serem 'direcionáveis', ou seja, qualquer versão rotacionada de um filtro pode ser expressa como uma combinação linear de um conjunto base de filtros. Isso resulta em redes mais eficientes em termos de dados e com melhor capacidade de generalização.

## Statistics

O principal benefício estatístico é a **eficiência de dados** e a **melhor generalização**. Em benchmarks como o CIFAR, as Steerable CNNs alcançaram resultados de ponta com um uso de parâmetros muito mais eficiente do que as CNNs tradicionais. Em tarefas de modelagem molecular (como com o NequIP), a equivariância E(3) (que inclui SE(3)) demonstrou ganhos de desempenho significativos, especialmente em cenários de dados limitados. O custo computacional pode ser maior em comparação com CNNs simples devido à complexidade das operações de convolução equivariante, mas o ganho em eficiência de parâmetros e generalização geralmente compensa.

## Features

1. **Equivariância Geométrica Intrínseca:** Garante que a rede respeite as simetrias dos dados (e.g., rotação, translação). 2. **Filtros Direcionáveis (Steerable Filters):** Permite que os filtros convolucionais sejam transformados de forma previsível, reduzindo o número de parâmetros independentes. 3. **Eficiência de Parâmetros:** Utiliza os parâmetros de forma mais eficiente, pois o conhecimento da simetria é codificado na arquitetura. 4. **Melhor Generalização:** A capacidade de generalizar para transformações de entrada não vistas é significativamente melhorada. 5. **Aplicações 3D:** Essencial para dados 3D, como nuvens de pontos e estruturas moleculares, onde a orientação é arbitrária.

## Use Cases

1. **Química e Física Computacional:** Modelagem de potenciais interatômicos (e.g., NequIP) e previsão de propriedades moleculares, onde a energia e outras propriedades devem ser invariantes à rotação e translação. 2. **Visão Computacional 3D:** Análise de nuvens de pontos 3D, reconhecimento de objetos e segmentação, onde a pose do objeto é arbitrária. 3. **Robótica e Manipulação:** Representações de objetos SE(3)-Equivariantes para tarefas de manipulação e planejamento de movimento. 4. **Análise de Imagens Biomédicas:** Segmentação de imagens microscópicas e análise de estruturas biológicas, onde a orientação da amostra pode variar. 5. **Classificação de Imagens 2D:** Embora mais comuns em 3D, as CNNs direcionáveis (e.g., E(2)-equivariantes) melhoram a classificação em 2D ao lidar com rotações.

## Integration

A integração é tipicamente feita através de bibliotecas de aprendizado profundo que implementam as operações equivariantes. \n\n**Exemplo de Biblioteca:** `escnn` (Equivariant Steerable CNNs) para PyTorch, sucessora da `e2cnn`.\n\n**Instalação (esquemática):**\n```bash\npip install escnn\n```\n\n**Exemplo de Código (Convolução Equivariante em PyTorch com `escnn`):**\n```python\nimport torch\nfrom escnn import gspaces, nn\n\n# 1. Definir o espaço de simetria (e.g., C4 para rotações de 90 graus)\ngspace = gspaces.Rot2dOnR2(N=4)\n\n# 2. Definir o tipo de campo de entrada (e.g., um campo escalar com 1 canal)\nin_type = nn.FieldType(gspace, [gspace.trivial_repr])\n\n# 3. Definir o tipo de campo de saída (e.g., um campo regular com 8 canais)\nout_type = nn.FieldType(gspace, [gspace.regular_repr] * 8)\n\n# 4. Criar a camada convolucional equivariante\n# A camada garante que a saída se transforme de forma previsível sob as rotações do grupo C4\nconv = nn.R2Conv(in_type, out_type, kernel_size=5, padding=2)\n\n# 5. Criar um tensor de entrada (batch_size, channels, height, width)\nx = torch.randn(1, 1, 32, 32)\n\n# 6. Converter o tensor em um objeto FieldType para a rede\nx_field = in_type(x)\n\n# 7. Aplicar a convolução\ny_field = conv(x_field)\n\n# 8. Obter o tensor de saída\ny = y_field.tensor\n\nprint(f\"Formato de entrada: {x.shape}\")\nprint(f\"Formato de saída: {y.shape}\")\n```

## URL

https://github.com/QUVA-Lab/escnn (escnn - Sucessor de e2cnn)