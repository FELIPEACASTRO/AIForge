# Telco Customer Churn Dataset

## Description
Um conjunto de dados focado em programas de retenção de clientes, originalmente da IBM. Ele contém informações detalhadas sobre clientes de uma empresa de telecomunicações fictícia, incluindo se eles cancelaram o serviço no último mês (Churn), os serviços que contrataram, informações da conta (tempo de permanência, tipo de contrato, método de pagamento, cobrança mensal e total) e dados demográficos (gênero, faixa etária, se têm parceiros e dependentes). É amplamente utilizado para modelagem de previsão de rotatividade (churn prediction).

## Statistics
O conjunto de dados bruto contém 7.043 linhas (clientes) e 21 colunas (recursos). O arquivo principal é `WA_Fn-UseC_Telco-Customer-Churn.csv`, com um tamanho de 977.5 kB. A versão mais recente no Kaggle é a "Version 1". O dataset é amplamente utilizado, com mais de 2.59 milhões de visualizações e 446 mil downloads.

## Features
O dataset possui 21 colunas, incluindo: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure` (meses de permanência), `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` e a variável alvo `Churn`.

## Use Cases
Previsão de Rotatividade de Clientes (Customer Churn Prediction) em empresas de telecomunicações. Análise Exploratória de Dados (EDA) para identificar fatores que levam ao cancelamento. Desenvolvimento e comparação de modelos de Machine Learning (como Regressão Logística, Árvores de Decisão, Random Forest e XGBoost) para classificação binária. Criação de programas de retenção de clientes focados.

## Integration
O dataset pode ser baixado diretamente do Kaggle no formato CSV. Para uso em Python, é comum utilizar a biblioteca `pandas` para carregar o arquivo e iniciar a análise e modelagem. Exemplo de carregamento: `import pandas as pd; df = pd.read_csv('WA_Fn-UseC_Telco-Customer-Churn.csv')`. É frequentemente necessário pré-processamento, como tratamento de valores ausentes e codificação de variáveis categóricas, antes de aplicar modelos de Machine Learning.

## URL
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
