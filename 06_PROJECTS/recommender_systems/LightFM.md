# LightFM

## Description

LightFM é uma biblioteca Python de código aberto que implementa um modelo de fatoração de matriz híbrido, combinando as abordagens de **Filtragem Colaborativa** e **Filtragem Baseada em Conteúdo**. Seu principal diferencial é a capacidade de incorporar metadados (recursos/features) de usuários e itens, permitindo a criação de sistemas de recomendação robustos que lidam eficazmente com o problema do 'cold-start' (usuários ou itens novos com poucas interações). A biblioteca é otimizada para desempenho, utilizando implementações em Cython.

## Statistics

A LightFM é uma das bibliotecas de recomendação mais populares em Python, com mais de 4.000 estrelas no GitHub (em novembro de 2025). É amplamente utilizada em projetos de ciência de dados e produção devido à sua capacidade de lidar com grandes volumes de dados esparsos e feedback implícito (como cliques ou visualizações) e explícito (como avaliações).

## Features

1. **Recomendação Híbrida:** Combina Filtragem Colaborativa (aprendendo com interações) e Filtragem Baseada em Conteúdo (usando metadados de usuários/itens).
2. **Suporte a Feedback Implícito e Explícito:** Pode ser treinada com dados de avaliações (explícito) ou interações binárias (implícito).
3. **Solução para Cold-Start:** A inclusão de recursos de conteúdo permite gerar recomendações para novos usuários ou itens antes que haja dados de interação suficientes.
4. **Algoritmos Integrados:** Implementa modelos populares como *Weighted Approximate-Rank Pairwise* (WARP) e *BPR* (Bayesian Personalized Ranking) para feedback implícito, e regressão logística para feedback explícito.
5. **Otimização de Desempenho:** Utiliza Cython para otimizar o treinamento do modelo, tornando-o escalável para grandes conjuntos de dados.

## Use Cases

1. **E-commerce e Varejo:** Recomendar produtos para clientes, sugerir itens relacionados ou personalizar a página inicial.
2. **Mídia e Entretenimento:** Sugerir filmes, músicas, artigos ou vídeos com base no histórico de consumo e nas características do conteúdo.
3. **Redes Sociais:** Recomendar conexões, grupos ou conteúdo a seguir.
4. **Sistemas de Busca Personalizada:** Melhorar a relevância dos resultados de busca incorporando o perfil do usuário e as características dos itens.

## Integration

A integração é feita via Python, utilizando o `pip` para instalação e a API intuitiva da biblioteca. O exemplo a seguir demonstra a criação de um modelo híbrido simples:

```python
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score

# 1. Preparação dos Dados
# Suponha que você tenha listas de (usuário, item, interação)
# e listas de (usuário, recurso) e (item, recurso)

# Exemplo de dados de interação (feedback implícito)
interactions = [
    (1, 101, 1), (1, 102, 1), (2, 101, 1), (3, 103, 1), (4, 104, 1)
]

# Exemplo de recursos de usuário e item
user_features = [
    (1, 'premium'), (2, 'standard'), (3, 'premium'), (4, 'standard')
]
item_features = [
    (101, 'categoria_A'), (102, 'categoria_B'), (103, 'categoria_A'), (104, 'categoria_C')
]

# 2. Criação do Objeto Dataset
dataset = Dataset()
dataset.fit(
    (u for u, _, _ in interactions), # Usuários
    (i for _, i, _ in interactions), # Itens
    user_features=(f for u, f in user_features), # Recursos de usuário
    item_features=(f for i, f in item_features)  # Recursos de item
)

# 3. Construção das Matrizes
(interactions_matrix, weights_matrix) = dataset.build_interactions(interactions)
user_features_matrix = dataset.build_user_features(user_features)
item_features_matrix = dataset.build_item_features(item_features)

# 4. Treinamento do Modelo Híbrido
# loss='warp' é ideal para feedback implícito
model = LightFM(loss='warp')
model.fit(
    interactions_matrix,
    user_features=user_features_matrix,
    item_features=item_features_matrix,
    epochs=30, num_threads=2
)

# 5. Avaliação (Exemplo)
# train_auc = auc_score(model, interactions_matrix, num_threads=2).mean()
# print(f\"AUC do Treinamento: {train_auc:.4f}\")

# 6. Geração de Recomendações (Exemplo para o usuário 1)
user_id = 1
user_x = dataset.get_user_representations(user_id)

scores = model.predict(
    user_x,
    np.arange(dataset.n_items),
    user_features=user_features_matrix,
    item_features=item_features_matrix
)

top_items = np.argsort(-scores)

print(f\"Recomendações para o Usuário {user_id}:\")
for item_id in top_items[:3]:
    print(f\"- Item {dataset.mapping()[2][item_id]}\")
```

## URL

https://github.com/lyst/lightfm