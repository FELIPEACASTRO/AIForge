# Google AutoML - Cloud AutoML services (via Vertex AI)

## Description

Google AutoML is a suite of Machine Learning (ML) products that enables developers with **little or no ML experience** to train high-quality custom models with minimal effort. Its unique value proposition is the **democratization of ML**, automating the model development lifecycle (from algorithm selection to hyperparameter tuning) and allowing companies to create models tailored to their business needs in minutes using a user-friendly graphical interface. Cloud AutoML is now integrated and accessible through **Vertex AI**, Google Cloud's unified ML platform.

## Statistics

Evaluation metrics are specific to each model type and are accessible via the Google Cloud console or API. For **Classification and Regression Models (Tabular)**, the metrics include **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, **Confusion Matrices**, and **Precision-Recall** values. AutoML is designed to **accelerate model training and deployment**, reducing development time and the need for specialized data scientists. Cost is variable, based on factors such as training time (e.g., approximately $21.252 per node hour for tabular data training) and prediction volume.

## Features

**Unified Platform (Vertex AI):** Prepares and stores datasets, provides access to Google's ML tools, and manages models with confidence. **AutoML Tabular:** Builds and deploys ML models for structured data (classification, regression, forecasting). **AutoML Image:** Derives insights from object detection and image classification (with custom labels). Supports edge model deployment. **AutoML Video:** Enables streaming video analysis, scene change detection, and object tracking. **AutoML Text:** Reveals the structure and meaning of text, offering custom entity extraction and sentiment analysis. **AutoML Translation:** Dynamically detects and translates between languages, supporting 50 language pairs and custom models. **APIs:** Support for REST, RPC, and gRPC APIs for programmatic interactions.

## Use Cases

**Image Analysis:** Product classification in e-commerce, defect detection on production lines (quality control). **Video Analysis:** Content annotation for improved discovery, object tracking in security videos. **Natural Language Processing (NLP):** Custom entity extraction in legal or medical documents, sentiment analysis of customer reviews. **Tabular Data:** Sales demand forecasting (as in the Colab example), fraud detection, customer churn prediction. **Real-World Examples:** **Twitter** (helps customers find meaningful "Spaces"), **Imagia** (uses AutoML to discover markers for degenerative diseases).

## Integration

Integration is done primarily through the **Vertex AI** platform and its APIs. The latest and recommended Python client is the **Vertex AI SDK for Python**, which replaces the older `google-cloud-automl` client and enables programmatic training and prediction.

**Integration Example (Vertex AI SDK for Python for Batch Prediction):**

```python
from google.cloud import bigquery
from google.cloud import aiplatform

# Initialize the Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION)

# Define the model path (assuming 'model' is a Model instance)
# model = aiplatform.Model(model_name='projects/PROJECT_ID/locations/REGION/models/MODEL_ID')

# Define the BigQuery input and output configurations
PREDICTION_DATASET_BQ_PATH = "bq://bigquery-public-data:iowa_liquor_sales_forecasting.2021_sales_predict"
batch_predict_bq_output_uri_prefix = "bq://{}.{}".format(
    PROJECT_ID, "iowa_liquor_sales_predictions"
)

# Run the batch prediction
batch_prediction_job = model.batch_predict(
    job_display_name="iowa_liquor_sales_forecasting_predictions",
    bigquery_source=PREDICTION_DATASET_BQ_PATH,
    instances_format="bigquery",
    bigquery_destination_prefix=batch_predict_bq_output_uri_prefix,
    predictions_format="bigquery",
    generate_explanation=True,
    sync=False, # Run asynchronously
)

print(f"Batch Prediction Job Name: {batch_prediction_job.resource_name}")
```

## URL

https://cloud.google.com/automl