# Google AutoML - Cloud AutoML services (via Vertex AI)

## Description

O Google AutoML é um conjunto de produtos de Machine Learning (ML) que permite a desenvolvedores com **pouca ou nenhuma experiência em ML** treinar modelos personalizados de alta qualidade com o mínimo de esforço. Sua proposta de valor única é a **democratização do ML**, automatizando o ciclo de vida do desenvolvimento de modelos (desde a seleção de algoritmos até o ajuste de hiperparâmetros) e permitindo que as empresas criem modelos específicos para suas necessidades de negócio em minutos, utilizando uma interface gráfica amigável. Atualmente, o Cloud AutoML está integrado e acessível através do **Vertex AI**, a plataforma unificada de ML do Google Cloud.

## Statistics

As métricas de avaliação são específicas para cada tipo de modelo e são acessíveis via console do Google Cloud ou API. Para **Modelos de Classificação e Regressão (Tabular)**, as métricas incluem **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, **Matrizes de Confusão** e valores de **Precisão-Recall**. O AutoML é projetado para **acelerar o treinamento e a implantação de modelos**, reduzindo o tempo de desenvolvimento e a necessidade de cientistas de dados especializados. O custo é variável, baseado em fatores como o tempo de treinamento (ex: aproximadamente $21.252 por hora de nó para treinamento de dados tabulares) e o volume de previsões.

## Features

**Plataforma Unificada (Vertex AI):** Prepara e armazena datasets, oferece acesso às ferramentas de ML do Google, e gerencia modelos com confiança. **AutoML Tabular:** Constrói e implanta modelos de ML para dados estruturados (classificação, regressão, previsão). **AutoML Image:** Deriva insights de detecção de objetos e classificação de imagens (com rótulos personalizados). Suporta implantação de modelos na borda (edge). **AutoML Video:** Permite análise de vídeo em streaming, detecção de mudança de cena e rastreamento de objetos. **AutoML Text:** Revela a estrutura e o significado do texto, oferecendo extração de entidade e análise de sentimento personalizadas. **AutoML Translation:** Detecta e traduz dinamicamente entre idiomas, suportando 50 pares de idiomas e modelos personalizados. **APIs:** Suporte a APIs REST, RPC e gRPC para interações programáticas.

## Use Cases

**Análise de Imagem:** Classificação de produtos em e-commerce, detecção de defeitos em linhas de produção (controle de qualidade). **Análise de Vídeo:** Anotação de conteúdo para descoberta aprimorada, rastreamento de objetos em vídeos de segurança. **Processamento de Linguagem Natural (NLP):** Extração de entidades personalizadas em documentos legais ou médicos, análise de sentimento de avaliações de clientes. **Dados Tabulares:** Previsão de demanda de vendas (como no exemplo do Colab), detecção de fraudes, previsão de rotatividade de clientes. **Exemplos Reais:** **Twitter** (ajuda clientes a encontrar "Spaces" significativos), **Imagia** (usa AutoML para descobrir marcadores para doenças degenerativas).

## Integration

A integração é feita principalmente através da plataforma **Vertex AI** e suas APIs. O cliente Python mais recente e recomendado é o **Vertex AI SDK for Python**, que substitui o cliente `google-cloud-automl` mais antigo e permite o treinamento e a previsão programática.

**Exemplo de Integração (Vertex AI SDK for Python para Previsão em Lote):**

```python
from google.cloud import bigquery
from google.cloud import aiplatform

# Inicializa o SDK do Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Define o caminho do modelo (assumindo que 'model' é uma instância de Model)
# model = aiplatform.Model(model_name='projects/PROJECT_ID/locations/REGION/models/MODEL_ID')

# Define as configurações de entrada e saída do BigQuery
PREDICTION_DATASET_BQ_PATH = "bq://bigquery-public-data:iowa_liquor_sales_forecasting.2021_sales_predict"
batch_predict_bq_output_uri_prefix = "bq://{}.{}".format(
    PROJECT_ID, "iowa_liquor_sales_predictions"
)

# Realiza a previsão em lote
batch_prediction_job = model.batch_predict(
    job_display_name="iowa_liquor_sales_forecasting_predictions",
    bigquery_source=PREDICTION_DATASET_BQ_PATH,
    instances_format="bigquery",
    bigquery_destination_prefix=batch_predict_bq_output_uri_prefix,
    predictions_format="bigquery",
    generate_explanation=True,
    sync=False, # Executa de forma assíncrona
)

print(f"Batch Prediction Job Name: {batch_prediction_job.resource_name}")
```

## URL

https://cloud.google.com/automl