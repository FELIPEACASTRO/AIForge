# PersonalCareNet (CNN com Mecanismo de Atenção e XAI)

## Description

O **PersonalCareNet** é um modelo de inteligência artificial explicável (XAI) baseado em Redes Neurais Convolucionais (CNNs) e um mecanismo de atenção, projetado para o monitoramento personalizado de saúde e predição de risco de doenças. Publicado em 2025, ele se destaca por integrar o mecanismo de atenção para aumentar a interpretabilidade, permitindo que o modelo foque dinamicamente nas características clínicas mais relevantes. A interpretabilidade é ainda mais aprimorada pela incorporação do framework **SHAP (Shapley Additive exPlanations)**, que fornece explicações globais e específicas para cada paciente, construindo confiança para a tomada de decisão clínica. O modelo foi treinado e avaliado usando um subconjunto do banco de dados **MIMIC-III**, focado em pacientes de UTI.

## Statistics

- **Acurácia Máxima:** 97.86%
- **AUC (Area Under the Curve):** 98.3%
- **Comparação:** Supera modelos de ponta como TabNet, AutoGluon Tabular e NODE em precisão.
- **Dataset:** Subconjunto do MIMIC-III (pacientes de UTI).
- **Citações:** Citado em 3 artigos (até a data da pesquisa).
- **Ano de Publicação:** 2025.

## Features

- **Mecanismo de Atenção Opcional (CHARMS):** Permite que o modelo atribua pesos dinâmicos às características clínicas, aumentando a interpretabilidade ao destacar os fatores mais influentes na predição.
- **Interpretabilidade Multi-Nível (SHAP):** Oferece explicações no nível local (para cada predição individual) e global (importância geral das características).
- **Arquitetura Híbrida:** Combina camadas densas com componentes de regularização para dados clínicos estruturados.
- **Prevenção de Overfitting:** Utiliza camadas de Dropout e regularização L2 para garantir a generalização do modelo.

## Use Cases

- **Monitoramento Personalizado de Saúde:** Avaliação contínua do estado de saúde de pacientes.
- **Predição de Risco de Doenças:** Previsão de condições clínicas ou riscos de saúde em pacientes de UTI com base em dados clínicos estruturados.
- **Suporte à Decisão Clínica:** Fornecer aos médicos não apenas uma predição, mas também a justificativa (quais características clínicas foram mais importantes) para a decisão, promovendo a confiança no sistema de IA.

## Integration

Embora o código-fonte direto do PersonalCareNet não tenha sido encontrado, a metodologia sugere uma implementação padrão de Deep Learning com a adição de um módulo de atenção e a integração pós-hoc do framework SHAP.

**Exemplo Conceitual de Integração (Python/PyTorch ou TensorFlow):**

1.  **Definição do Modelo:** Implementar uma CNN ou rede densa com uma camada de atenção (por exemplo, *Self-Attention* ou *Additive Attention*) aplicada às saídas das camadas intermediárias.
2.  **Treinamento:** Treinar o modelo com dados de EHRs (como o MIMIC-III) usando otimizadores padrão (Adam/SGD) e função de perda de entropia cruzada.
3.  **Interpretabilidade (SHAP):** Após o treinamento, usar a biblioteca `shap` para calcular os valores de Shapley para cada predição.

```python
# Exemplo de uso do SHAP para interpretabilidade
import shap
# Supondo que 'model' é o modelo PersonalCareNet treinado e 'data' são os dados de entrada
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(data)

# Visualização da explicação para uma instância (paciente)
shap.force_plot(explainer.expected_value[0], shap_values[0][0], data.iloc[0])
```

## URL

https://pmc.ncbi.nlm.nih.gov/articles/PMC12397250/