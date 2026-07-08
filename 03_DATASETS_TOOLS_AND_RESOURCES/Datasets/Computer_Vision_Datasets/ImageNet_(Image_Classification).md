# ImageNet (Image Classification)

## Description
ImageNet is a vast visual database designed for use in visual object recognition software research. It is an ongoing research effort to provide image data for training large-scale object recognition models. The most widely used version is the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** 2012-2017 subset, which has become the de facto standard for benchmarking in computer vision. ImageNet is organized according to the WordNet hierarchy, where each node (synset) is illustrated by hundreds of thousands of images. ImageNet's impact was fundamental to the advancement of Deep Learning, especially after AlexNet's victory in ILSVRC 2012. Recent research (2023-2025) continues to use ImageNet as a foundation, but also explores variations such as **ImageNet-D** (CVPR 2024), which uses generative models to create synthetic images and test the robustness of neural networks against more challenging data distributions.

## Statistics
- **Total Size (Original):** 14,197,122 images.
- **Synsets (Categories):** 21,841 indexed synsets (nodes of the WordNet hierarchy).
- **Most Widely Used Version (ILSVRC 2012/ImageNet-1K):**
    - **Classes:** 1,000 object classes.
    - **Training Images:** 1,281,167 images.
    - **Validation Images:** 50,000 images.
    - **Test Images:** 100,000 images.
- **Recent Variations (2024):** ImageNet-D (for robustness), ImageNet-BG (for background variations).

## Features
- **Scale and Diversity:** Contains millions of images and tens of thousands of categories (synsets).
- **WordNet Hierarchy:** Organized in a hierarchical structure that allows training at different levels of granularity.
- **ILSVRC-2012 (ImageNet-1K):** The most popular subset, with 1,000 classes and more than 1.2 million training images.
- **Robustness Benchmarking:** The emergence of variations such as ImageNet-D (2024) and ImageNet-C/P/A/R/Sketch/Adversarial demonstrates the continued use of ImageNet as a foundation for evaluating the robustness and generalization of Computer Vision models.
- **Annotated Images:** Manually annotated images for object classification and localization.

## Use Cases
- **Training Classification Models:** The primary use case, being the reference dataset for training and evaluating image classification models (e.g., ResNet, VGG, EfficientNet).
- **Transfer Learning:** Using models pre-trained on ImageNet as *backbones* for tasks in other domains (e.g., object detection, semantic segmentation, medical classification) through *fine-tuning*.
- **Robustness Benchmarking:** Evaluating the ability of models to maintain performance under different types of corruption, distortion, or domain variation (e.g., using ImageNet-C, ImageNet-D).
- **Computer Vision Research:** Development of new neural network architectures and training methods.
- **Text-to-Image Models (2025):** Recent research explores the use of ImageNet to train text-to-image generation models, despite its relatively small size for this task.

## Integration
The most common subset, ImageNet ILSVRC 2012 (ImageNet-1K), is available for download on **Kaggle** (after accepting the terms of use and license). To access the full dataset and other subsets, it is necessary to log in or request access on the official website, agreeing to the terms of use that restrict use to **non-commercial research and education purposes**.

**Usage Example (PyTorch):**
The `torchvision` library provides a `torchvision.datasets.ImageNet` class for easy integration, but the user must first download and organize the data locally.

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# The directory must contain the 'train' and 'val' folders with the images
traindir = '/path/to/imagenet/train'
valdir = '/path/to/imagenet/val'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageNet(
    traindir, split='train', download=False,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
```

## URL
[https://www.image-net.org/](https://www.image-net.org/)
