# ONNX - Open Neural Network Exchange

## Description

The **Open Neural Network Exchange (ONNX)** is an open, industry-standard format, backed by a community of technology companies and research organizations, designed to represent machine learning models. Its unique value proposition lies in its capacity for **interoperability** and **portability**, allowing developers to train models in any framework (such as PyTorch, TensorFlow, Keras) and run them on any runtime or device (such as ONNX Runtime, CPUs, GPUs, or dedicated hardware accelerators). ONNX defines a common set of operators and a file format for the computation graph, acting as an intermediate language that eliminates framework dependency for the inference phase. This simplifies the transition from development to production, ensuring that model performance is maximized across diverse deployment environments.

## Statistics

- **Broad Adoption:** ONNX is a graduated project of the LF AI & Data Foundation, with contributions from major companies such as Microsoft, Meta, Amazon, and Nvidia.
- **Robust Ecosystem:** The ecosystem includes converters for more than 15 ML frameworks (including PyTorch, TensorFlow, Keras, Scikit-learn, MATLAB), and is supported by more than 40 runtimes and hardware accelerators.
- **Popularity on Hugging Face:** Models in the ONNX format (often quantized or optimized) are widely used for high-performance inference, with many popular ONNX models accumulating hundreds of thousands of downloads (e.g., embedding models such as `bge-m3-onnx-o4` with more than 230 thousand downloads).
- **ONNX Runtime (ORT):** ORT is the most common inference engine for ONNX models, supporting multiple platforms (Windows, Linux, macOS, Android, iOS) and languages (Python, C++, C#, Java, JavaScript).

## Features

- **Computation Graph Format:** Defines the model as a computation graph, where the nodes are operators and the edges are data tensors.
- **Standard Operator Set (Opset):** Has a rich and extensible set of operators (such as `Conv`, `Relu`, `Add`) that are the building blocks of ML and DL models.
- **Framework Portability:** Enables the conversion of models from popular frameworks (PyTorch, TensorFlow, Keras, Scikit-learn) to the ONNX format.
- **Model Optimization:** ONNX models can be optimized by tools such as the ONNX Optimizer for node fusion and other transformations before deployment.
- **Extensibility:** Supports custom operators (Custom Ops) to extend the functionality of the format.

## Use Cases

- **Cross-Platform Deployment:** Deployment of models trained in one framework to production environments that require a different runtime (e.g., training in PyTorch and deploying to a C# server using ONNX Runtime).
- **Inference Optimization:** Acceleration of inference for Computer Vision models (such as ResNet, YOLO), Natural Language Processing models (such as BERT, GPT), and Tabular models, leveraging hardware-specific optimizations via ONNX Runtime.
- **Edge Computing and IoT:** Deployment of ML models on resource-constrained edge devices, where efficiency and low power consumption are crucial.
- **Framework Conversion:** Use of ONNX as an intermediate format to convert models between different ML frameworks (e.g., from TensorFlow to PyTorch or vice versa) for research or migration purposes.
- **Cloud AI Services:** Cloud platforms (such as Azure ML) use ONNX to provide optimized and scalable inference for customer models.

## Integration

Integration with ONNX generally involves two steps: **Exporting** the trained model to the ONNX format and **Inference** using ONNX Runtime (ORT).

**1. Exporting a PyTorch Model to ONNX (Example):**
```python
import torch
import torch.onnx as onnx
import torchvision.models as models

# 1. Load or define the model
model = models.resnet18(pretrained=True)
model.eval()

# 2. Define a sample input (dummy input)
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

# 3. Export the model
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",  # Output file name
    export_params=True,
    opset_version=17, # Opset version (must be compatible with the runtime)
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("PyTorch model successfully exported to resnet18.onnx")
```

**2. Inference with ONNX Runtime (ORT) in Python:**
```python
import onnxruntime as ort
import numpy as np

# 1. Load the ONNX model
ort_session = ort.InferenceSession("resnet18.onnx")

# 2. Prepare the input (must match the model's input format)
# ONNX Runtime expects inputs as a dictionary of input names to NumPy arrays
input_name = ort_session.get_inputs()[0].name
input_shape = ort_session.get_inputs()[0].shape
# Create a sample NumPy array
input_data = np.random.randn(*input_shape).astype(np.float32)
input_feed = {input_name: input_data}

# 3. Run the inference
output = ort_session.run(None, input_feed)

# 4. Process the output
print("Inference executed successfully.")
# The output is a list of NumPy arrays, one for each model output
# print(output[0].shape)
```

## URL

https://onnx.ai/