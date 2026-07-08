# NVIDIA TensorRT

## Description

NVIDIA TensorRT is a high-performance Software Development Kit (SDK) for optimizing and accelerating deep learning model inference in production, exclusively on NVIDIA GPUs. It acts as an inference compiler that transforms models trained in frameworks such as TensorFlow, PyTorch, and ONNX into an optimized "engine." Its unique value proposition is to maximize GPU efficiency, resulting in low latency, high throughput, and maximum hardware utilization for critical real-time applications. It is a post-training tool, complementing training frameworks.

## Statistics

Inference Acceleration: Common reports indicate a speedup of 3x to 5x compared to running the original model in frameworks such as PyTorch or TensorFlow. Comparison with CPU: In some cases, inference with TensorRT on GPUs can be up to 40x faster than on CPU-only platforms. Performance Metrics: Throughput (images/second or tokens/second), Latency (end-to-end time for a single inference), and GPU Utilization.

## Features

Layer Fusion: Combines multiple model layers into a single GPU kernel. Precision Calibration: Support for mixed precision (FP32, FP16, and INT8) with quantization to maximize performance. Optimized Memory Allocation: Reuses GPU memory efficiently, minimizing the memory footprint. Kernel Auto-Tuning: Selects the fastest kernel algorithm for the specific GPU architecture and batch size. Graph Optimization: Removes unnecessary layers and reorders the computation graph. Dynamic Model Support: Allows optimizing the engine for different input sizes at runtime.

## Use Cases

Autonomous Vehicles: Real-time processing of sensor data (computer vision, LiDAR) for object detection and route planning. Natural Language Processing (NLP) and LLMs: Acceleration of large language models (LLMs) with TensorRT-LLM, enabling faster responses and higher throughput. Computer Vision: Applications such as facial recognition, video surveillance, and medical image analysis (cancer detection). Real-Time Content Generation: Generative AI applications that benefit from low latency for smoother interactions. Recommendation Systems: High-volume systems that require low latency for instant results.

## Integration

TensorRT integration generally involves converting a model trained in a framework to the optimized TensorRT format. NVIDIA provides specific libraries for the main frameworks:

1.  **PyTorch (Torch-TensorRT):** Allows compiling PyTorch models directly to TensorRT.
    ```python
    import torch
    import torch_tensorrt
    # ... (model loading code)
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[input_tensor],
        enabled_precisions={torch.float16}, # Optimization for FP16
        workspace_size=1 << 25
    )
    ```
2.  **TensorFlow (TF-TRT):** An integration that allows optimizing TensorFlow models (SavedModel).
    ```python
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    # ... (conversion configuration code)
    converter = trt.TrtConverterV2(input_saved_model_dir=input_dir, conversion_params=params)
    converter.convert()
    converter.save(output_dir)
    ```
3.  **C++ API:** For maximum performance and control in production environments, TensorRT offers a C++ API for building, calibrating, and running inference engines, frequently importing models in the ONNX format.

## URL

https://developer.nvidia.com/tensorrt