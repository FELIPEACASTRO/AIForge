# AWS SageMaker

## Description

**Amazon SageMaker** is a fully managed machine learning (ML) platform that enables data scientists and developers to build, train, and deploy ML models at scale quickly and easily. Its unique value proposition lies in unifying the entire ML lifecycle (MLOps) in a single integrated environment, SageMaker Studio, simplifying the process from experimentation to production. The platform eliminates the complexity of managing the underlying infrastructure, allowing teams to focus on innovation and solving business problems. The new generation of SageMaker, with **Unified Studio**, offers an integrated experience for analytics and AI with unified access to all data, built on Amazon DataZone [1] [2].

## Statistics

**Adoption and Scale:** In 2022, AWS SageMaker had more than **100,000 customers** across virtually every industry, with **millions of models created** and models trained with **billions of parameters** [8].
**Cost Optimization:** Customers report an average reduction of **50% in model deployment costs** when using advanced features such as Multi-Model Endpoints and the SageMaker Inference Recommender [9]. The use of **SageMaker Savings Plans** can reduce training and inference costs by **50% or more** [10].
**Efficiency:** Companies such as **bp** and **ENGIE Digital** use SageMaker to scale their ML operations, with bp using it to build, train, and deploy ML models as part of its infrastructure-as-code solution [11] [12].

## Features

**SageMaker Studio:** Unified, web-based development environment for the entire ML workflow.
**SageMaker JumpStart:** ML hub with pre-trained models, solution notebooks, and algorithms for rapid deployment.
**SageMaker Autopilot:** Automatic creation of high-quality ML models, with no coding knowledge required.
**SageMaker Feature Store:** Centralized repository to store, discover, and share ML features for training and inference.
**SageMaker Clarify:** Helps detect bias in models and provide explanations for predictions.
**SageMaker Pipelines:** MLOps tool to create, manage, and automate end-to-end ML workflows.
**SageMaker Inference:** Offers multiple deployment options (Real-time, Serverless, Asynchronous, and Batch) to optimize cost and performance [3] [4].

## Use Cases

**Predictive Maintenance:** Energy companies (such as ENGIE Digital) use SageMaker to predict equipment failures in power plants, optimizing maintenance and reducing downtime [12].
**Image Analysis and Computer Vision:** Image classification, object detection, and semantic segmentation for quality control in manufacturing or analysis of medical images.
**Natural Language Processing (NLP):** Building chatbots, sentiment analysis, text summarization, and translation.
**Recommendation Systems:** Creating models that suggest products, movies, or personalized content for users on e-commerce and streaming platforms.
**Time Series Forecasting:** Forecasting product demand, stock prices, or energy consumption [3].
**No-Code ML:** SageMaker Canvas allows business analysts to create high-accuracy ML models without writing code, democratizing access to ML [13].

## Integration

Integration with SageMaker is primarily done through the **AWS SDK for Python (Boto3)** or the **SageMaker Python SDK**, which offers a high-level, object-oriented interface to interact with SageMaker resources [5].

**Training and Deployment Example (SageMaker Python SDK):**

```python
import sagemaker
from sagemaker.tensorflow import TensorFlow

# 1. Configure the session and the S3 bucket
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# 2. Define the Estimator (for training)
# The 'entry_point' script contains the model training code
estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='2.11',
    py_version='py39',
    sagemaker_session=sagemaker_session
)

# 3. Start the training job
# 'inputs' points to the data location in S3
estimator.fit({'training': 's3://seu-bucket/seus-dados/'})

# 4. Deploy the model to a real-time inference endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# 5. Make a prediction
# response = predictor.predict(data)

# 6. Clean up the endpoint (very important to avoid costs)
# predictor.delete_endpoint()
```
The SDK simplifies resource provisioning, uploading data to S3, and creating training jobs and inference endpoints [6] [7].

## URL

https://aws.amazon.com/sagemaker/
