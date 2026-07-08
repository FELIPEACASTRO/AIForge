# CLEVR (Compositional Language and Elementary Visual Reasoning)

## Description
CLEVR (Compositional Language and Elementary Visual Reasoning) is a synthetic diagnostic dataset designed to test a wide range of visual and linguistic reasoning skills. It was created to address the biases present in earlier Visual Question Answering (VQA) benchmarks, ensuring that models cannot exploit statistical shortcuts to answer correctly without actually reasoning. The dataset consists of rendered images of simple 3D scenes containing objects (cubes, cylinders, spheres) with varied attributes (color, shape, size, material) and complex programmatically generated questions about those scenes. Each question is accompanied by a "functional program" representation that defines the exact sequence of logical and visual operations needed to arrive at the answer, enabling a precise evaluation of the reasoning capabilities of models.

## Statistics
**Main Version:** CLEVR v1.0. **Download Size (v1.0):** Approximately 18 GB (with images) or 86 MB (annotations only). **Dataset Size (TFDS):** 17.75 GiB. **Splits (Main Dataset):** Training (70,000 images, 699,989 questions), Validation (15,000 images, 149,991 questions), Test (15,000 images, 14,988 questions). **CoGenT Version:** 24 GB (with images) or 106 MB (annotations only), with specific splits for testing compositional generalization. **TFDS Version:** 3.1.0 (Adds question/answer text).

## Features
**Composition and Attributes:** Synthetic 3D scenes with simple objects (cubes, cylinders, spheres) varying in color (8), shape (3), size (2), and material (2). **Complex Questions:** Programmatically generated questions that require compositional reasoning, including attribute identification, counting, comparison, spatial relationships, and logical operations. **Detailed Annotations:** Includes scene graphs with object location, attributes, and relationships, and functional programs for each question, which serve as ground truth for the reasoning. **Minimal Bias:** Designed to have minimal bias, forcing models to perform genuine visual and linguistic reasoning. **Extensions:** Has extensions such as CLEVR-CoGenT (for compositional generalization) and CLEVR-X (for natural language explanations).

## Use Cases
**VQA Model Evaluation:** Serves as a rigorous diagnostic benchmark for Visual Question Answering (VQA) models, focusing on reasoning ability rather than statistical shortcuts. **Compositional Reasoning:** Used to test the ability of models to combine visual and linguistic concepts systematically. **Generalization:** The CoGenT version is used to evaluate the generalization of models to new attribute combinations not seen during training. **Interpretability:** The use of functional programs aids in the development of more interpretable VQA models, where the reasoning process can be traced. **Model Development:** Used to train and validate neural network architectures focused on attention, memory, and explicit reasoning modules.

## Integration
The CLEVR dataset can be downloaded directly from the official project page (links to the v1.0 and CoGenT versions, with and without images). For integration into machine learning projects, it is recommended to use dataset libraries such as TensorFlow Datasets (TFDS) or Hugging Face Datasets, which offer the `clevr` version for simplified loading and preprocessing.

**Example usage with TensorFlow Datasets (Python):**
```python
import tensorflow_datasets as tfds

# Load the dataset
ds = tfds.load('clevr', split='train', shuffle_files=True)

# Iterate over the examples
for example in ds.take(1):
    print(example)
```
Alternatively, the raw files (images and question/annotation JSONs) can be downloaded and processed manually. The dataset generation code is also available on GitHub to render new images or generate new questions.

## URL
[https://cs.stanford.edu/people/jcjohns/clevr/](https://cs.stanford.edu/people/jcjohns/clevr/)
