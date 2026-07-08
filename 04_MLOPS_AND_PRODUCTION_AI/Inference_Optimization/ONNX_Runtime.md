# ONNX Runtime

## Description

ONNX Runtime (ORT) is a high-performance, open-source machine learning inference and training engine, designed to accelerate the execution of ONNX (Open Neural Network Exchange) models across diverse hardware and software platforms. Its unique value proposition lies in **interoperability and performance optimization**, allowing models trained in any framework (such as PyTorch, TensorFlow, Keras, scikit-learn) to be exported to the ONNX format and executed efficiently in production environments, from the cloud (Azure, AWS, GCP) to edge and mobile devices (iOS, Android, Web). ORT applies graph optimizations and uses hardware-specific Execution Providers (EPs) to maximize inference and training speed, reducing latency and increasing throughput.

## Statistics

**Performance Acceleration:** Reports of inference performance gains of up to **9x** compared to native frameworks in specific scenarios, such as in high-throughput model serving.
**Latency Optimization:** Designed to minimize inference latency, crucial for real-time applications.
**Production Use:** Used in Microsoft products, including Windows, Office, Azure Cognitive Services, and Bing, demonstrating production-grade scalability and reliability.
**Hardware Flexibility:** Supports more than 10 different Execution Providers (EPs) to optimize performance across a wide range of hardware.

## Features

**Cross-Platform Acceleration:** Support for Windows, Linux, macOS, iOS, Android, and Web.
**Broad Hardware Compatibility:** Optimized for CPU, GPU (CUDA, TensorRT, OpenVINO, ROCm), and NPUs (Qualcomm QNN).
**Execution Providers (EPs):** Flexible interface to integrate hardware-specific libraries, such as TensorRT, OpenVINO, DirectML, and Core ML, for maximum acceleration.
**Graph Optimization:** Applies node transformations and fusions to improve model efficiency before execution.
**Training Support:** Capability for training large models and on-device training for personalization and privacy.
**Language Support:** APIs available for Python, C++, C#, Java, and JavaScript.

## Use Cases

**Cloud Inference Services:** Acceleration of computer vision, natural language processing (NLP), and recommendation models on platforms such as Azure Machine Learning.
**Edge Computing Applications:** Deployment of AI models on low-power, low-latency devices, such as smart cameras and IoT gateways.
**Mobile Applications:** Integration of AI features into iOS and Android applications (ONNX Runtime Mobile) for personalized experiences and on-device processing.
**Desktop Application Integration:** Use in products such as Microsoft Office and Windows for built-in AI features, such as image recognition and text processing.
**Large Language Models (LLMs):** Optimization of LLM and Generative AI model inference to reduce costs and latency.

## Integration

Integration with ONNX Runtime is straightforward, requiring installation of the package and loading of the ONNX model to create an `InferenceSession`. The following example demonstrates basic inference in Python:

```python
import onnxruntime as ort
import numpy as np

# 1. Load the ONNX model
model_path = "path/to/your/model.onnx"
session = ort.InferenceSession(model_path)

# 2. Prepare the input data (example with a float32 tensor)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
# Create a simulated input tensor
input_data = np.random.rand(*input_shape).astype(np.float32)

# 3. Run the inference
# The second argument is a dictionary {input_name: input_data}
outputs = session.run(None, {input_name: input_data})

# 4. Process the output
print("Inference Output:", outputs[0])
```

For C++, integration involves using the C++ API (a *wrapper* of the C API) to create an environment, a session, and run the model. Detailed examples are available in the `microsoft/onnxruntime-inference-examples` repository.

## URL

https://onnxruntime.ai/