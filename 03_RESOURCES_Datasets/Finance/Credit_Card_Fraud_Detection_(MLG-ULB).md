# Credit Card Fraud Detection (MLG-ULB)

## Description
Este conjunto de dados apresenta transações de cartão de crédito realizadas por titulares de cartões europeus em setembro de 2013. Ele contém 284.807 transações, das quais 492 são fraudes. O dataset é altamente desbalanceado, com a classe positiva (fraude) representando apenas 0,172% de todas as transações. Devido a questões de confidencialidade, as características (features) são transformadas usando Análise de Componentes Principais (PCA), exceto 'Time' (tempo decorrido desde a primeira transação) e 'Amount' (valor da transação).

## Statistics
284.807 transações no total. 492 casos de fraude (0,172%). 31 colunas (features + label). Tamanho do arquivo: aproximadamente 150 MB. Versão clássica e mais citada, utilizada em inúmeros artigos de 2017 até 2025. Uma versão mais recente (2023) com mais de 550.000 registros também está disponível no Kaggle.

## Features
Características numéricas transformadas por PCA (V1-V28); 'Time' e 'Amount' originais. A variável de resposta é 'Class' (1 para fraude, 0 para legítima). É notável o alto desbalanceamento de classes.

## Use Cases
Desenvolvimento e avaliação de modelos de detecção de fraude, como Redes Neurais, Máquinas de Vetores de Suporte (SVM) e métodos de *ensemble* (e.g., XGBoost, Random Forest). Pesquisa sobre técnicas de tratamento de dados desbalanceados e aprendizado sensível ao custo. Análise de risco e segurança de transações financeiras.

## Integration
O dataset pode ser baixado diretamente do Kaggle. Para uso em Python, é comum a utilização das bibliotecas `pandas` para carregamento e `scikit-learn` para pré-processamento e modelagem. Devido ao desbalanceamento, técnicas como *oversampling* (SMOTE) ou *undersampling* são frequentemente aplicadas. É necessário ter uma conta no Kaggle para baixar o arquivo `creditcard.csv`.

## URL
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
