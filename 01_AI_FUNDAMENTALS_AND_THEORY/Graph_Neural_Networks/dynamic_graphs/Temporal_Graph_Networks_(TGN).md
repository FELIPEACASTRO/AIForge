# Temporal Graph Networks (TGN)

## Description

Temporal Graph Networks (TGNs) são uma estrutura genérica e eficiente para aprendizado profundo em grafos dinâmicos, representados como sequências de eventos temporizados. O TGN aborda a limitação das Redes Neurais Gráficas (GNNs) tradicionais, que não conseguem modelar efetivamente a natureza evolutiva de muitos sistemas do mundo real (como redes sociais, transações financeiras e interações biológicas). Sua proposta de valor única reside na combinação de **módulos de memória** e **operadores baseados em grafos** para gerar embeddings temporais de nós que capturam o estado de cada nó em qualquer ponto no tempo, permitindo que o modelo se adapte e preveja mudanças na estrutura e nas características do grafo ao longo do tempo. O TGN generaliza vários modelos anteriores de aprendizado de grafos dinâmicos, como JODIE e DyRep, como instâncias específicas de sua estrutura.

## Statistics

**Desempenho de Estado da Arte (SOTA):** O TGN alcançou desempenho SOTA em várias tarefas de previsão transdutiva e indutiva em grafos dinâmicos no momento de sua publicação. **Eficiência:** Demonstrou ser mais eficiente computacionalmente do que modelos anteriores como JODIE e DyRep. **Citações:** O artigo original (`arXiv:2006.10637`) possui mais de 1.100 citações (em 2024), indicando sua ampla adoção e influência na pesquisa de aprendizado de grafos dinâmicos. **Implementação:** O repositório GitHub oficial possui mais de 1.100 estrelas e 220 forks, refletindo sua popularidade na comunidade de pesquisa. **Linguagem:** Implementado 100% em Python, utilizando a biblioteca PyTorch.

## Features

**Módulos de Memória:** Armazenam e atualizam um estado de memória para cada nó, permitindo que o modelo retenha informações sobre o histórico de interações do nó. A memória é atualizada após cada evento temporal (interação). **Função de Agregação de Mensagens:** Agrega mensagens de vizinhos temporais para formar uma nova mensagem para o nó de destino. O TGN suporta diferentes agregadores, como agregação de atenção (TGN-attn) e agregação de soma. **Módulo de Incorporação Temporal:** Codifica a diferença de tempo entre o evento atual e o último evento armazenado na memória, permitindo que o modelo capture a importância da recência. **Generalização:** A estrutura TGN é flexível e pode ser configurada para replicar o comportamento de modelos anteriores de grafos dinâmicos, como JODIE e DyRep, por meio de diferentes configurações de seus componentes. **Eficiência Computacional:** Projetado para ser mais eficiente computacionalmente do que as abordagens anteriores, especialmente em tarefas transdutivas e indutivas.

## Use Cases

**Previsão de Link (Link Prediction):** Prever a formação de novas arestas (interações) em um grafo dinâmico, como prever futuras conexões em redes sociais ou transações em redes financeiras. **Classificação de Nós (Node Classification):** Classificar o tipo ou o estado de um nó em um determinado momento, como identificar usuários maliciosos em uma rede social ou classificar o tipo de entidade em uma rede de conhecimento em evolução. **Detecção de Anomalias em Grafos (Graph Anomaly Detection):** Identificar eventos ou nós incomuns que se desviam do comportamento normal do grafo, como detecção de fraude em transações financeiras ou identificação de ataques em redes de comunicação. **Sistemas de Recomendação:** Modelar as interações temporais entre usuários e itens para fornecer recomendações mais precisas e sensíveis ao tempo. **Modelagem de Sistemas Físicos:** Prever a dinâmica de longo prazo de sistemas físicos complexos, como em simulações de física.

## Integration

A implementação mais popular e mantida do TGN está disponível no **PyTorch Geometric (PyG)**, uma biblioteca líder para GNNs em PyTorch. A classe `torch_geometric.nn.models.TGN` fornece uma implementação pronta para uso.

**Exemplo de Integração (PyTorch Geometric):**

```python
import torch
from torch_geometric.nn import TGN
from torch_geometric.datasets import TGNExample

# 1. Carregar um conjunto de dados de grafo dinâmico de exemplo
dataset = TGNExample(name='wikipedia')
data = dataset[0]

# 2. Inicializar o modelo TGN
# Parâmetros chave:
# - num_nodes: Número total de nós
# - raw_msg_dim: Dimensão da característica da mensagem (geralmente a dimensão da característica da aresta)
# - memory_dim: Dimensão do estado de memória de cada nó
# - time_dim: Dimensão da incorporação temporal
# - num_layers: Número de camadas de atenção do grafo
# - use_memory: Se deve usar o módulo de memória
# - aggregator: Tipo de agregador de mensagens ('last', 'mean', 'attn')

tgn = TGN(
    num_nodes=data.num_nodes,
    raw_msg_dim=data.msg.size(-1),
    memory_dim=100,
    time_dim=100,
    num_layers=1,
    use_memory=True,
    aggregator='attn'
)

# 3. Exemplo de uso (Previsão de Link)
# O TGN é normalmente usado em um loop de treinamento que processa eventos em lotes.

# Simulação de um lote de eventos (interações)
src = data.src[:100]
dst = data.dst[:100]
t = data.t[:100]
msg = data.msg[:100]

# 4. Obter embeddings de nós temporais
# O TGN retorna os embeddings de nós *antes* e *depois* da atualização da memória.
# O módulo de memória é atualizado internamente.
z, last_update = tgn(src, dst, t, msg)

# z contém os embeddings de nós para src e dst.
# last_update contém o tempo da última atualização para cada nó.

# 5. Resetar o estado da memória (necessário antes de um novo epoch ou teste)
# tgn.memory.reset_state()
```

**Dependências:**
*   `torch`
*   `torch-geometric` (PyG)
*   `pandas`
*   `scikit-learn`

## URL

https://github.com/twitter-research/tgn