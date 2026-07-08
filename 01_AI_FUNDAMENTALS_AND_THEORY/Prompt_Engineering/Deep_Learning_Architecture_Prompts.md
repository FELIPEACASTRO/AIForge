# Deep Learning Architecture Prompts

## Description
"Deep Learning Architecture Prompts" (DLAP) is an advanced Prompt Engineering technique that uses Large Language Models (LLMs) to assist, automate, or guide the process of **Neural Architecture Search (NAS)**. Instead of manually designing the structure of a neural network (such as the number of layers, type of convolution, activation functions), the prompt is used to instruct an LLM to generate, modify, or suggest the code or structure of the architecture.

This approach relies on the ability of LLMs to reason about code and complex structures, transforming the high-level specification of the problem (defined in the prompt) into a functional Deep Learning architecture. The concept is closely tied to methodologies such as **EvoPrompting**, where the LLM acts as an adaptive mutation and crossover operator in an evolutionary NAS algorithm, optimizing the architecture at the code level. DLAP represents a shift in focus from manual architecture engineering to the engineering of prompts that generate the architecture.

## Examples
```
1. **CNN Generation for Classification:** "Generate the Python code for a Convolutional Neural Network (CNN) architecture using PyTorch. The network should be optimized for CIFAR-10 image classification (10 classes, 32x32x3 images). Include 3 convolutional blocks, each followed by ReLU and Max Pooling. The final model should have fewer than 500,000 parameters."

2. **GNN Design for Graph Data:** "Design a Graph Neural Network (GNN) architecture using the PyG (PyTorch Geometric) framework for a node classification task on a citation graph (Cora dataset). The architecture should use two Graph Convolutional Network (GCN) layers and a *dropout* layer of 0.5. Generate the complete code for the model class."

3. **Optimization of an Existing Architecture (EvoPrompting):** "Analyze the following neural network architecture (code provided). The objective is to reduce the number of parameters by 20% without losing more than 1% accuracy on the MNIST dataset. Suggest a modification to the depth or width of the convolutional layers and generate the modified code."

4. **Transformer Architecture Specification:** "Create a detailed prompt for an LLM that generates a Transformer architecture for English-to-Portuguese translation. The architecture should have 6 encoder layers and 6 decoder layers, with 8 attention heads (multi-head attention) and an embedding size of 512. The prompt should request the code in TensorFlow/Keras."

5. **Recurrent Neural Network (RNN) Design for Time Series:** "Generate a neural network architecture for time series forecasting (stock prices). Use a bidirectional LSTM layer with 128 units, followed by a dense output layer. The model should be implemented in PyTorch and accept input sequences of 60 timesteps."

6. **Hardware Constraint Prompt:** "Develop a neural network architecture for real-time object detection (bounding boxes), suitable for deployment on an Edge device with 4GB of RAM and a maximum latency of 50ms per inference. Suggest a lightweight variant of YOLO or MobileNet and provide the architecture specification in YAML format."
```

## Best Practices
**Task and Data Specificity:** The prompt should detail the type of task (e.g., image classification, time series regression) and the characteristics of the dataset (size, dimensionality, data type). **Constraint Definition:** Include clear hardware and performance constraints, such as maximum latency, available memory, and the desired number of parameters. **Use of Structured Methodologies:** Employ techniques such as EvoPrompting (for evolutionary optimization) or Cognitive Prompt Architecture (for cognitive structuring of the prompt) to guide the LLM more effectively. **Validation and Iterative Refinement:** Use the generated code or architecture as a starting point. Validate the performance and use the results (e.g., accuracy, loss) to refine the prompt in subsequent iterations. **Programming Language and Framework:** Clearly specify the language (Python) and the framework (TensorFlow, PyTorch) to ensure the generation of functional code.

## Use Cases
**Accelerated Neural Architecture Search (NAS):** Using LLMs to explore the architecture search space more efficiently than traditional NAS methods based on genetic algorithms or reinforcement learning. **Optimization of Architectures for Edge Computing:** Generating lightweight, efficient architectures that meet strict latency and energy consumption constraints for IoT and Edge devices. **Rapid Prototyping Code Generation:** Quickly creating the base code of a complex architecture (e.g., Transformer, GNN) for prototyping and initial testing. **Domain Knowledge Transfer:** Incorporating *insights* from recent research articles (via RAG) into the prompt to guide the LLM to design architectures that use the latest innovations in the field. **Education and Exploration:** Allowing students and researchers to quickly explore architecture variations for different tasks, understanding the impact of hyperparameter changes.

## Pitfalls
**Vagueness in Specification:** Prompts that do not clearly define the task, the dataset, or the performance constraints will result in generic and ineffective architectures. **Ignoring Hardware Constraints:** Generating a complex architecture without considering the memory or processing power limitations of the deployment environment. **Lack of Validation:** Blindly trusting the code generated by the LLM without performing performance tests and *benchmarking* on the real dataset. **Excessive Dependence:** Using the prompt to generate the entire architecture without applying human knowledge to refine or adjust the structure, missing the opportunity to incorporate domain-specific *insights*. **Focus Only on Code:** Concentrating solely on generating the architecture code and neglecting the data pipeline, the loss function, and the optimization scheme, which are crucial for training.

## URL
[https://proceedings.neurips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html)
