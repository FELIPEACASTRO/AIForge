# LIME - Local Interpretable Model-agnostic Explanations

## Description

LIME (Local Interpretable Model-agnostic Explanations) é uma técnica inovadora de eXplainable AI (XAI) que visa tornar as previsões de modelos de Machine Learning (ML) de "caixa-preta" compreensíveis para os humanos. Sua proposta de valor única reside em sua **agnosticidade de modelo** e **interpretabilidade local**. A agnosticidade de modelo significa que o LIME pode ser aplicado a *qualquer* modelo de ML, independentemente de sua arquitetura (Redes Neurais, Random Forests, SVMs, etc.). A interpretabilidade local garante que, em vez de tentar explicar o modelo inteiro (o que é inviável para modelos complexos), o LIME explica a previsão de uma *única* instância de dados, ajustando um modelo substituto interpretável (como regressão linear) em torno dessa previsão específica. Isso aumenta a **confiança** e a **auditabilidade** dos sistemas de ML, permitindo que os usuários entendam *por que* uma decisão específica foi tomada.

## Statistics

**Citações Acadêmicas:** O artigo original ("Why Should I Trust You?": Explaining the Predictions of Any Classifier) possui mais de **27.000 citações** (em 2024), destacando seu impacto fundamental no campo de XAI.
**Popularidade:** É uma das bibliotecas de interpretabilidade de modelo mais populares e amplamente adotadas na comunidade de Machine Learning.
**Desempenho:** Embora o LIME seja localmente fiel, a fidelidade global do modelo substituto é tipicamente baixa, o que é uma característica inerente ao seu design local.
**Velocidade:** A geração de explicações pode ser computacionalmente intensiva, especialmente para modelos complexos ou grandes conjuntos de dados, devido à necessidade de perturbar a instância e reavaliar o modelo.

## Features

**Agnosticismo de Modelo:** Pode explicar as previsões de qualquer modelo de Machine Learning, sem acesso à sua estrutura interna ou parâmetros.
**Interpretabilidade Local:** Foca em explicar previsões individuais, ajustando um modelo simples e interpretável (por exemplo, regressão linear) localmente em torno da instância de interesse.
**Suporte a Múltiplos Tipos de Dados:** Funciona com dados tabulares, texto e imagens, usando diferentes métodos de perturbação para gerar amostras vizinhas.
**Fidelidade:** O modelo substituto local é projetado para ser fiel à previsão do modelo de caixa-preta na vizinhança da instância explicada.
**Transparência e Confiança:** Ajuda a identificar vieses e falhas do modelo, permitindo que os desenvolvedores e usuários decidam se devem confiar em uma previsão específica.

## Use Cases

**Avaliação de Confiança:** Decidir se uma previsão individual deve ser confiável, especialmente em domínios de alto risco (saúde, finanças).
**Seleção de Modelo:** Comparar modelos de ML diferentes com base em suas explicações para escolher o mais robusto e menos enviesado.
**Depuração de Modelo:** Identificar falhas e vieses em modelos de "caixa-preta", revelando que o modelo pode estar usando características irrelevantes ou espúrias para fazer previsões.
**Conformidade Regulatória:** Atender aos requisitos de transparência e explicabilidade em setores regulamentados (por exemplo, GDPR, leis de discriminação algorítmica).
**Melhoria de Modelo:** Usar as explicações para obter *insights* sobre o modelo e o conjunto de dados, levando a melhorias no *feature engineering* ou na arquitetura do modelo.

## Integration

A integração do LIME é feita através de sua biblioteca Python, que requer que o modelo de Machine Learning forneça uma função de previsão de probabilidade. O exemplo a seguir demonstra o uso do `LimeTabularExplainer` com um modelo `RandomForestClassifier` do Scikit-learn para explicar uma previsão no conjunto de dados Iris.

**Pré-requisitos:**
```bash
sudo pip3 install lime scikit-learn
```

**Exemplo de Código Python (lime_example.py):**
```python
import lime
import lime.lime_tabular
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Carregar e preparar os dados
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = [str(x) for x in iris.target_names]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. Treinar um modelo "caixa-preta" (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Criar o explicador LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    kernel_width=0.25 # Controla o tamanho da vizinhança local
)

# 4. Escolher uma instância para explicar (o primeiro ponto de teste)
instance_to_explain = X_test[0]
prediction = class_names[model.predict(instance_to_explain.reshape(1, -1))[0]]

print(f"Instância a ser explicada: {instance_to_explain}")
print(f"Previsão do Modelo: {prediction}\n")

# 5. Gerar a explicação
# 'num_features' define quantas características mais importantes serão mostradas
explanation = explainer.explain_instance(
    data_row=instance_to_explain,
    predict_fn=model.predict_proba,
    num_features=2
)

# 6. Exibir a explicação
print("Explicação LIME (Top 2 Características):")
for feature, weight in explanation.as_list():
    print(f" - {feature}: {weight:.4f}")
```

**Saída do Exemplo:**
```
Instância a ser explicada: [6.1 2.8 4.7 1.2]
Previsão do Modelo: versicolor

Explicação LIME (Top 2 Características):
 - 0.30 < petal width (cm) <= 1.30: 0.0095
 - 4.30 < petal length (cm) <= 5.10: 0.0073
```
A saída mostra que a previsão para a classe "versicolor" foi influenciada positivamente pela largura e comprimento da pétala dentro de faixas específicas.

## URL

https://github.com/marcotcr/lime