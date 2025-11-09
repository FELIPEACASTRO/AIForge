# SHAP (SHapley Additive exPlanations)

## Description

SHAP (SHapley Additive exPlanations) é uma abordagem baseada na teoria dos jogos para explicar a saída de qualquer modelo de aprendizado de máquina. Sua proposta de valor única reside na unificação de métodos de interpretabilidade de modelos, conectando a alocação de crédito ideal com explicações locais. O SHAP atribui a cada característica um valor de importância para uma previsão específica, garantindo consistência e precisão local. Ele resolve o problema de caixas-pretas em ML, fornecendo transparência e confiança nas decisões do modelo. O framework é agnóstico ao modelo, o que significa que pode ser aplicado a modelos lineares, baseados em árvores e redes neurais profundas. O valor SHAP de uma característica é a mudança média na previsão do modelo quando essa característica é incluída no conjunto de características, em comparação com quando ela é excluída.

## Statistics

- **Citações do Artigo Original:** O artigo seminal "A Unified Approach to Interpreting Model Predictions" de Lundberg e Lee (2017) possui mais de **44.000 citações** (fonte: arXiv, ACM, NeurIPS), indicando sua vasta influência acadêmica.
- **Estrelas do GitHub:** O repositório oficial `shap/shap` no GitHub tem aproximadamente **24.679 estrelas**, refletindo uma forte adoção e suporte da comunidade de código aberto.
- **Downloads PyPI:** O pacote `shap` no PyPI registra um alto volume de downloads, com o total de downloads em todos os tempos ultrapassando **100 milhões** (estimativa baseada em dados de terceiros, como PyPI Stats), o que demonstra sua popularidade e uso generalizado em produção e pesquisa.

## Features

- **Agnóstico ao Modelo:** Pode ser aplicado a qualquer modelo de aprendizado de máquina (linear, baseado em árvore, redes neurais).
- **Consistência e Precisão Local:** Garante que as explicações sejam localmente precisas e consistentes com a teoria dos jogos.
- **Vários Explicadores:** Inclui KernelSHAP (para qualquer modelo), TreeSHAP (otimizado para modelos baseados em árvores como XGBoost, LightGBM, CatBoost) e DeepExplainer/GradientExplainer (para modelos de aprendizado profundo como TensorFlow e PyTorch).
- **Visualizações Abrangentes:** Oferece gráficos de resumo (summary plots), gráficos de dependência (dependence plots), gráficos de força (force plots) e gráficos de interação (interaction plots) para insights globais e locais.
- **Unificação:** Unifica métodos anteriores como LIME, DeepLIFT e Layer-wise Relevance Propagation (LRP) sob uma única estrutura teórica.

## Use Cases

- **Diagnóstico de Modelos:** Identificar e corrigir vieses (bias) em modelos de ML, como discriminação de gênero ou raça, ao analisar a contribuição das características sensíveis.
- **Aprovação Regulatória e Conformidade:** Em setores como finanças e saúde, o SHAP é usado para explicar decisões de crédito ou diagnósticos médicos, atendendo a requisitos regulatórios de transparência (e.g., GDPR, regulamentações de IA).
- **Otimização de Negócios:** Em marketing, explicar por que um cliente específico recebeu uma oferta ou por que ele provavelmente cancelará (churn), permitindo intervenções direcionadas e mais eficazes.
- **Engenharia de Características (Feature Engineering):** Entender quais características são mais importantes globalmente para o modelo, guiando o processo de seleção e criação de novas características.
- **Análise de Segurança Cibernética:** Explicar a classificação de tráfego de rede como malicioso ou benigno, identificando as características de rede que mais contribuíram para a decisão.

## Integration

A integração do SHAP é tipicamente feita através da biblioteca Python `shap`. O método de integração varia dependendo do tipo de modelo (agnóstico ou baseado em árvore/deep learning).

**Exemplo de Integração com Scikit-learn (KernelSHAP):**
```python
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 1. Carregar dados e treinar modelo
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# 2. Inicializar o explicador SHAP (KernelSHAP para modelos agnósticos)
# O KernelSHAP requer um conjunto de dados de fundo (background dataset)
explainer = shap.KernelExplainer(model.predict_proba, X_train)

# 3. Calcular os valores SHAP para uma amostra de teste
shap_values = explainer.shap_values(X_test.iloc[0,:])

# 4. Visualizar a explicação
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[0,:])
```

**Exemplo de Integração com Modelos Baseados em Árvores (TreeSHAP):**
```python
import shap
import xgboost
from sklearn.datasets import load_boston

# 1. Carregar dados e treinar modelo XGBoost
X, y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# 2. Inicializar o explicador SHAP (TreeSHAP para modelos baseados em árvores)
explainer = shap.TreeExplainer(model)

# 3. Calcular os valores SHAP
shap_values = explainer.shap_values(X)

# 4. Visualizar o gráfico de resumo
shap.summary_plot(shap_values, X)
```

## URL

https://shap.readthedocs.io/ | https://github.com/shap/shap