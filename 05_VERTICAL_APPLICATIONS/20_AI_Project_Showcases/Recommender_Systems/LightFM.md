# LightFM

## Description

LightFM is an open-source Python library that implements a hybrid matrix factorization model, combining the **Collaborative Filtering** and **Content-Based Filtering** approaches. Its main differentiator is the ability to incorporate metadata (features) of users and items, enabling the creation of robust recommendation systems that effectively handle the 'cold-start' problem (new users or items with few interactions). The library is optimized for performance, using implementations in Cython.

## Statistics

LightFM is one of the most popular recommendation libraries in Python, with more than 4,000 stars on GitHub (as of November 2025). It is widely used in data science and production projects due to its ability to handle large volumes of sparse data and both implicit feedback (such as clicks or views) and explicit feedback (such as ratings).

## Features

1. **Hybrid Recommendation:** Combines Collaborative Filtering (learning from interactions) and Content-Based Filtering (using user/item metadata).
2. **Support for Implicit and Explicit Feedback:** Can be trained with rating data (explicit) or binary interactions (implicit).
3. **Cold-Start Solution:** The inclusion of content features makes it possible to generate recommendations for new users or items before there is sufficient interaction data.
4. **Built-in Algorithms:** Implements popular models such as *Weighted Approximate-Rank Pairwise* (WARP) and *BPR* (Bayesian Personalized Ranking) for implicit feedback, and logistic regression for explicit feedback.
5. **Performance Optimization:** Uses Cython to optimize model training, making it scalable to large datasets.

## Use Cases

1. **E-commerce and Retail:** Recommending products to customers, suggesting related items, or personalizing the home page.
2. **Media and Entertainment:** Suggesting movies, music, articles, or videos based on consumption history and content characteristics.
3. **Social Networks:** Recommending connections, groups, or content to follow.
4. **Personalized Search Systems:** Improving the relevance of search results by incorporating the user profile and item characteristics.

## Integration

Integration is done via Python, using `pip` for installation and the library's intuitive API. The following example demonstrates the creation of a simple hybrid model:

```python
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score

# 1. Data Preparation
# Suppose you have lists of (user, item, interaction)
# and lists of (user, feature) and (item, feature)

# Example interaction data (implicit feedback)
interactions = [
    (1, 101, 1), (1, 102, 1), (2, 101, 1), (3, 103, 1), (4, 104, 1)
]

# Example user and item features
user_features = [
    (1, 'premium'), (2, 'standard'), (3, 'premium'), (4, 'standard')
]
item_features = [
    (101, 'categoria_A'), (102, 'categoria_B'), (103, 'categoria_A'), (104, 'categoria_C')
]

# 2. Creation of the Dataset Object
dataset = Dataset()
dataset.fit(
    (u for u, _, _ in interactions), # Users
    (i for _, i, _ in interactions), # Items
    user_features=(f for u, f in user_features), # User features
    item_features=(f for i, f in item_features)  # Item features
)

# 3. Building the Matrices
(interactions_matrix, weights_matrix) = dataset.build_interactions(interactions)
user_features_matrix = dataset.build_user_features(user_features)
item_features_matrix = dataset.build_item_features(item_features)

# 4. Training the Hybrid Model
# loss='warp' is ideal for implicit feedback
model = LightFM(loss='warp')
model.fit(
    interactions_matrix,
    user_features=user_features_matrix,
    item_features=item_features_matrix,
    epochs=30, num_threads=2
)

# 5. Evaluation (Example)
# train_auc = auc_score(model, interactions_matrix, num_threads=2).mean()
# print(f\"Training AUC: {train_auc:.4f}\")

# 6. Generating Recommendations (Example for user 1)
user_id = 1
user_x = dataset.get_user_representations(user_id)

scores = model.predict(
    user_x,
    np.arange(dataset.n_items),
    user_features=user_features_matrix,
    item_features=item_features_matrix
)

top_items = np.argsort(-scores)

print(f\"Recommendations for User {user_id}:\")
for item_id in top_items[:3]:
    print(f\"- Item {dataset.mapping()[2][item_id]}\")
```

## URL

https://github.com/lyst/lightfm