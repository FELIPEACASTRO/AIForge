# TorchServe

## Description

TorchServe is an open-source tool for serving PyTorch models in production, developed and maintained collaboratively by AWS and Meta (Facebook). Its unique value proposition lies in providing a robust, flexible, and easy-to-use solution for deploying Deep Learning models at scale, with a focus on high performance and complete management of the model lifecycle. **Important Note:** The project is currently in **Limited Maintenance** mode, which means there are no planned updates, bug fixes, or security patches, and users should be aware that vulnerabilities may not be resolved.

## Statistics

**Performance and Monitoring Metrics:** TorchServe offers a comprehensive metrics system, classified into:
*   **Frontend Metrics:** Include API request status (2XX, 4XX, 5XX), inference request metrics, and system utilization metrics (collected periodically).
*   **Backend Metrics:** Include standard metrics and custom model metrics (via API).
*   **Metrics Modes:** Supports three modes of metric collection and exposure: `log` (default, logs to files), `prometheus` (exposes metrics in Prometheus format via API endpoint), and `legacy`.
*   **Latency:** Detailed latency metrics, such as `ts_inference_latency_microseconds` and `ts_queue_latency_microseconds`, are tracked for performance optimization.

## Features

**Management and Inference APIs:** Offers dedicated RESTful APIs for model management (registration, unloading, scaling) and inference (real-time and batch predictions).
**Model Support:** Supports PyTorch models in Eager and Scripted (TorchScript) mode.
**Custom Handlers:** Allows the creation of custom pre-processing and post-processing logic to meet specific model requirements.
**Scalability and Performance:** Supports Dynamic Batching to optimize resource utilization and microbatching.
**Security:** Support for Serving Models Securely (SSL/TLS).
**Hardware Support:** Integration and optimization for various hardware, including AWS Inferentia2, AWS Graviton, OpenVINO, and Intel Extension for PyTorch.
**Model Management:** Allows serving multiple versions of the same model and performing A/B testing.

## Use Cases

**Deployment of Computer Vision Models:** Serving image classification, object detection, and segmentation models (eager and scripted mode).
**Model Serving at Scale:** Use of Dynamic Batching and hardware optimizations (such as AWS Inferentia2 and Graviton) to serve models with high throughput and low latency.
**Generative AI (GenAI) Applications:**
*   **LLM Serving with Compiled RAG:** Deployment of Retrieval-Augmented Generation (RAG) endpoints using `torch.compile` to increase throughput and optimize resource usage (e.g., RAG endpoint on Graviton CPU and LLM endpoint on GPU).
*   **Chained Multi-Image Generation:** Creation of complex applications that chain multiple models (e.g., Llama for prompt generation and Stable Diffusion for image generation) using TorchServe, `torch.compile`, and OpenVINO for performance optimization.
**Secure Serving:** Deployment of models with enhanced security using SSL/TLS configurations.

## Integration

Integration with TorchServe is done primarily through its RESTful API. The typical flow involves creating a model archive file (MAR) and using `curl` commands to interact with the Management and Inference APIs.

**1. Creating the MAR File (Model Archive):**
```bash
torch-model-archiver --model-name my_model --version 1.0 \
--model-file my_model.py --serialized-file my_model.pth \
--handler image_classifier --extra-files index_to_name.json
```

**2. Starting TorchServe:**
```bash
torchserve --start --ncs --model-store model_store
```

**3. Model Registration (Management API):**
```bash
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=my_model.mar"
```

**4. Inference (Inference API):**
```bash
# Example inference call with an input file
curl http://localhost:8080/predictions/my_model -T input_data.json
```

**5. Scalability (Management API):**
```bash
# Increase the number of workers for scalability
curl -v -X PUT "http://localhost:8081/models/my_model?min_worker=4&synchronous=true"
```

## URL

https://github.com/pytorch/serve