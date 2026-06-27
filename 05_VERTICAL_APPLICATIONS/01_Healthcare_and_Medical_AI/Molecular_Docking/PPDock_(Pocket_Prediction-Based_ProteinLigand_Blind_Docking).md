# PPDock (Pocket Prediction-Based Protein–Ligand Blind Docking)

## Description

**PPDock** (Pocket Prediction-Based Protein–Ligand Blind Docking) é um modelo de *docking* cego de proteína-ligante baseado em Redes Neurais Gráficas (GNNs), proposto em 2025. Ele aborda a limitação dos métodos tradicionais de *docking* que exigem a pré-definição do sítio de ligação (bolsão). O PPDock opera em duas etapas principais: primeiro, ele prevê o sítio de ligação (bolsão) na proteína, e em seguida, realiza o *docking* do ligante dentro da região prevista. A arquitetura utiliza uma combinação de Redes Neurais Gráficas Euclidianas (EGNNs) para capturar as relações espaciais e estruturais complexas entre a proteína e o ligante, resultando em maior precisão e eficiência, especialmente em cenários de *docking* cego.

## Statistics

- **Ano de Publicação:** 2025
- **Métrica Chave:** Ligand RMSD (Root Mean Square Deviation) e Centroid Distance.
- **Desempenho:** Alcançou uma taxa de Ligand RMSD < 2 Å de **45.2%** no conjunto de teste original de 363 complexos, superando significativamente os métodos de *docking* cego baseados em aprendizado de máquina de última geração.
- **Arquitetura:** EGNN (Rede Neural Gráfica Euclidiana) e GNN.
- **Citações:** 5 (em novembro de 2025, indicando ser um trabalho muito recente).
- **Dataset de Treinamento/Teste:** PDBbind.

## Features

- **Docking Cego Baseado em Predição de Bolsão:** Não requer conhecimento prévio do sítio de ligação da proteína.
- **Arquitetura EGNN:** Utiliza Redes Neurais Gráficas Euclidianas (EGNNs) para modelar as interações 3D entre proteína e ligante, preservando a invariância de rotação e translação.
- **Alta Precisão:** Supera métodos de *docking* cego de última geração em conjuntos de dados de referência.
- **Eficiência Computacional:** Oferece um equilíbrio entre precisão e velocidade, crucial para triagem virtual em larga escala.

## Use Cases

- **Triagem Virtual (Virtual Screening):** Identificação rápida e precisa de novos candidatos a medicamentos, prevendo a pose de ligação de milhões de moléculas a um alvo proteico.
- **Otimização de Chumbo (Lead Optimization):** Refinamento de compostos promissores, prevendo as poses de ligação mais estáveis e a afinidade.
- **Docking Cego:** Aplicação em casos onde o sítio de ligação da proteína é desconhecido ou difícil de determinar experimentalmente.
- **Previsão de Afinidade de Ligação Droga-Alvo (DTA):** Embora focado em *docking*, a pose prevista é crucial para a estimativa precisa da afinidade.

## Integration

O modelo PPDock é implementado em Python e requer bibliotecas de GNN, como PyTorch Geometric (PyG). O repositório GitHub fornece pesos de modelo pré-treinados para as duas partes (Predição de Bolsão e Docking).

**Exemplo de Uso (Conceitual):**
```python
# Instalação (assumindo ambiente Python/PyTorch)
# pip install torch_geometric
# git clone https://github.com/JieDuTQS/PPDock
# cd PPDock

# Carregar pesos do modelo (disponíveis no repositório)
# model_pocket = load_model('pocket_prediction_weights.pth')
# model_dock = load_model('docking_weights.pth')

# 1. Preparação dos dados (estrutura da proteína e ligante)
# protein_graph = preprocess_protein('protein.pdb')
# ligand_graph = preprocess_ligand('ligand.mol2')

# 2. Predição do Bolsão
# pocket_coords = model_pocket.predict(protein_graph)

# 3. Docking Cego
# docked_pose = model_dock.dock(protein_graph, ligand_graph, pocket_coords)

# 4. Avaliação da Pose
# rmsd = calculate_rmsd(docked_pose, true_pose)
```
O repositório GitHub (`JieDuTQS/PPDock`) é a fonte primária para o código e instruções de integração.

## URL

https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c01373