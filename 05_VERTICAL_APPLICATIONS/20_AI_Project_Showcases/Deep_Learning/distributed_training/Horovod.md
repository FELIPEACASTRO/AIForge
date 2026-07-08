# Horovod

## Description

Horovod is an open-source framework for distributed deep learning training, developed by Uber Engineering. Its unique value proposition is to make distributed training of deep learning models (across frameworks such as TensorFlow, Keras, PyTorch, and Apache MXNet) **fast, easy, and portable**. It achieves this by minimizing the code changes required to migrate a single-GPU training script to a distributed environment, using the **Ring All-reduce** algorithm for efficient gradient communication between nodes.

## Statistics

- **Scaling Efficiency:** Achieved **nearly 90% scaling efficiency** for models such as Inception V3 and ResNet-101 on 128 GPUs.
- **Core Algorithm:** Based on the **Ring All-reduce** algorithm, which optimizes gradient communication, outperforming older approaches such as the parameter server.
- **Origin:** Developed and maintained by **Uber Engineering**.

## Features

- **Multi-Framework Support:** Compatibility with TensorFlow, Keras, PyTorch, and Apache MXNet.
- **Easy to Use:** Requires only a few lines of code to adapt an existing training script to distributed mode.
- **Communication Optimization:** Implements the **Ring All-reduce** algorithm for gradient aggregation, which is crucial for large-scale performance.
- **Portability:** Based on MPI (Message Passing Interface) concepts, making it easy to run across diverse cluster environments.
- **HorovodRunner:** A general API for running distributed deep learning workloads on Spark clusters (such as Azure Databricks).

## Use Cases

- **Training Scaling:** The primary use case is scaling single-GPU training scripts to multiple nodes or GPUs, drastically reducing training time for large models and massive datasets.
- **High-Performance Research:** Used in high-performance research environments (such as NASA/HECC) to accelerate deep learning model iteration.
- **Integration with ML Platforms:** Used in conjunction with platforms such as **Ray Train**, **AWS SageMaker**, and **Azure Databricks** for orchestrating and managing distributed training in production environments.

## Integration

Integration with Horovod is straightforward and requires only a few modifications to existing training code. The principle is to initialize Horovod, pin the process to the local GPU, and wrap the optimizer with `hvd.DistributedOptimizer`.

**Integration Example with PyTorch:**

```python
import torch
import horovod.torch as hvd

# 1. Initialize Horovod
hvd.init()

# 2. Pin the GPU to the local process
torch.cuda.set_device(hvd.local_rank())

# 3. Define the model and the optimizer (example)
model = ...
optimizer = optim.Adadelta(model.parameters())

# 4. Wrap the optimizer with Horovod
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters())

# 5. Broadcast the initial weights from rank 0 to all others
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

# The training loop follows the standard pattern, where optimizer.step() performs the All-reduce
```

**Integration Example with TensorFlow/Keras:**

```python
import tensorflow as tf
import horovod.tensorflow.keras as hvd

# 1. Initialize Horovod
hvd.init()

# 2. Configure visible GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# 3. Wrap the optimizer with Horovod
opt = tf.optimizers.Adadelta(1.0 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

# 4. Compile and train the model
model.compile(optimizer=opt, ...)
model.fit(...)
```

## URL

https://horovod.ai/