# CIFAR-10/100 (Image Classification)

## Description
**CIFAR-10** and **CIFAR-100** are computer vision datasets widely used for training image classification models. Both are labeled subsets of the 80 million tiny images dataset and were created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton [1].

**CIFAR-10** consists of 60,000 color images (32x32 pixels) in 10 mutually exclusive classes, with 6,000 images per class. The classes include: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

**CIFAR-100** is similar, but has 100 classes, each with 600 images. The 100 classes are grouped into 20 superclasses, providing a "fine" label (the specific class) and a "coarse" label (the superclass) for each image. This makes it ideal for hierarchical classification tasks [1].

Although its creation predates 2023, the dataset remains a fundamental reference and is constantly used in recent research (2023-2025) for benchmarking new neural network architectures, such as ResNet and Wide Residual Networks, and for studies on learning with noisy labels and transfer learning [2] [3] [4].

## Statistics
- **Total Size:** 60,000 color images (32x32 pixels).
- **Split:** 50,000 training images and 10,000 test images.
- **Resolution:** 32x32 pixels (RGB).
- **CIFAR-10:** 10 classes, 6,000 images per class.
- **CIFAR-100:** 100 classes (fine labels) and 20 superclasses (coarse labels), 600 images per fine class.
- **File Size (Python Version):** CIFAR-10 (163 MB), CIFAR-100 (161 MB).
- **Notable Versions:** CIFAR-10.1 (new test set for CIFAR-10) [5].

## Features
- **Color Images:** All images are color (RGB).
- **Low Resolution:** Fixed resolution of 32x32 pixels, which makes it ideal for rapid testing and proof-of-concept development.
- **Class Structure:**
    - **CIFAR-10:** 10 object classes.
    - **CIFAR-100:** 100 object classes, grouped into 20 superclasses (fine and coarse labels).
- **Standard Split:** 50,000 training images and 10,000 test images.
- **Extended Versions:** Extended versions exist, such as CIFAR-10.1, which offers a new test set for more robust model evaluation [5].

## Use Cases
CIFAR-10/100 is a benchmark dataset for the development and evaluation of computer vision algorithms, especially in:

- **Image Classification:** This is the primary use case, testing a model's ability to correctly assign an image to one of the defined classes.
- **CNN Architecture Development:** Used to test the effectiveness of new Convolutional Neural Network (CNN) architectures and Deep Learning models, such as ResNet, VGG, and Wide Residual Networks [3].
- **Transfer Learning:** Although small, it is frequently used as a target dataset for models pre-trained on larger datasets, such as ImageNet, to evaluate knowledge transfer capability [8].
- **Learning with Noisy Labels:** Modified versions, such as CIFAR-10/100N, are used to research and develop robust machine learning methods that can handle annotation errors [2].
- **Image Generation:** Used to train and evaluate generative models, such as GANs (Generative Adversarial Networks) and VAEs (Variational Autoencoders), due to its well-defined class structure.
- **Model Quantization and Optimization:** Used to test the efficiency of model compression and optimization techniques for deployment on resource-constrained devices.

## Integration
The most common and recommended way to integrate CIFAR-10/100 is through high-level machine learning libraries, such as PyTorch and TensorFlow, which provide utility functions for automatically downloading and loading the dataset.

**Integration Example (PyTorch):**
`torchvision.datasets` allows downloading and loading the dataset with just a few lines of code, eliminating the need to manage files manually [6].

```python
import torchvision
import torchvision.transforms as transforms

# Define the transformations to be applied to the images
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training dataset (CIFAR-10)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Download and load the test dataset (CIFAR-10)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
```

**Integration Example (TensorFlow/Keras):**
Keras also offers direct access to the dataset [7].

```python
from tensorflow.keras.datasets import cifar10

# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

For manual download, the files are available in Python `pickle`, Matlab, and binary formats on the official site [1].

## URL
[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
