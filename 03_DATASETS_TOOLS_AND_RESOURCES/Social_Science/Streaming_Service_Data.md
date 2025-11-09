# Streaming Service Data

## Description
O dataset "Streaming Service Data" é uma coleção abrangente de 5.000 registros de clientes de um serviço de streaming, projetado para análise de comportamento do consumidor e previsão de *churn* (abandono). Ele captura diversos aspectos demográficos, comportamentais e transacionais dos clientes, como idade, gênero, duração da assinatura, região, método de pagamento, número de tickets de suporte abertos, pontuação de satisfação, descontos oferecidos, atividade recente e gasto mensal. O principal objetivo do dataset é fornecer *insights* valiosos sobre os padrões de retenção e satisfação do cliente.

## Statistics
**Amostras:** 5.000 registros de clientes.
**Variáveis:** 12 colunas.
**Tamanho do Arquivo:** 314.59 kB (Streaming.csv).
**Versões:** A versão atual é a V1, atualizada há cerca de um ano (informação baseada na data de atualização da fonte em 2024).

## Features
**Estrutura de Dados:**
*   **Registros:** 5.000 clientes.
*   **Variáveis:** 12 colunas, incluindo Customer_ID, Age, Gender, Subscription_Length, Region, Payment_Method, Support_Tickets_Raised, Satisfaction_Score, Discount_Offered, Last_Activity, Monthly_Spend e Churned.
*   **Tipos de Dados:** Contém variáveis categóricas (Gender, Region, Payment_Method) e numéricas (Age, Subscription_Length, Scores, Spend).
*   **Características:** Inclui valores ausentes em algumas variáveis (Age e Satisfaction_Score), o que é útil para a prática de pré-processamento de dados. O campo `Churned` (1 = sim, 0 = não) é a variável alvo principal para modelos de classificação.

## Use Cases
*   **Previsão de Churn:** Construção de modelos de Machine Learning (como Regressão Logística, Árvores de Decisão ou *Random Forests*) para prever quais clientes têm maior probabilidade de cancelar a assinatura.
*   **Análise de Sentimento e Satisfação:** Estudo da correlação entre a Pontuação de Satisfação e outras variáveis comportamentais.
*   **Segmentação de Clientes:** Utilização de algoritmos de *clustering* para identificar grupos de clientes com perfis e comportamentos de gasto semelhantes.
*   **Otimização de Marketing:** Avaliação da eficácia dos descontos oferecidos e da frequência de atividade do cliente.

## Integration
O dataset pode ser baixado diretamente da plataforma Kaggle. Após o download, o arquivo `Streaming.csv` pode ser carregado em qualquer ambiente de análise de dados (como Python com Pandas ou R) para exploração e modelagem. A integração é simples, exigindo apenas a leitura do arquivo CSV.

**Exemplo de Download (Kaggle CLI):**
```bash
kaggle datasets download -d akashanandt/streaming-service-data
```
**Exemplo de Uso (Python/Pandas):**
```python
import pandas as pd
df = pd.read_csv('Streaming.csv')
print(df.head())
```

## URL
[https://www.kaggle.com/datasets/akashanandt/streaming-service-data](https://www.kaggle.com/datasets/akashanandt/streaming-service-data)
