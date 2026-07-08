# Places365 (Scene Recognition)

## Description
Places365 is a subset of the Places2 database, a vast repository of images conceived for **Scene Recognition** and understanding. Its main objective is to provide a fundamental resource for training artificial intelligence systems on high-level visual understanding tasks, such as scene context, object recognition, and action prediction. The semantic categories of the dataset are defined by their function, representing the entry level of an environment (for example, distinguishing between different types of bedrooms: home, hotel, or nursery). The dataset is widely used for training Convolutional Neural Networks (CNNs) to extract *deep scene features* and to establish new *benchmarks* in scene-centered tasks. The use of the dataset and the pre-trained models remains relevant, being referenced in recent research (2024-2025) for validating new *machine learning* methods.

## Statistics
**Main Versions:** Places365-Standard and Places365-Challenge. **Categories:** 365 unique scene classes. **Places365-Standard:** Approximately 1.8 million training images, with a maximum of 5,000 images per category. **Places365-Challenge:** Approximately 8 million training images (including those from Standard), with a maximum of 40,000 images per category. **Places Base (Total):** More than 10 million images in total, with more than 400 scene categories.

## Features
**Large-Scale Scene Recognition:** Contains 365 unique scene categories. **Functional Categories:** The classes are defined by the function of the environment, aligned with human visual cognition. **Pre-trained Models:** Provides pre-trained CNNs (AlexNet, VGG, ResNet, GoogLeNet) for Caffe and PyTorch. **Output Capabilities:** The trained models can predict scene categories, environment type (indoor/outdoor), scene attributes, and generate Class Activation Maps (CAM) for visualization. **Two Versions:** Places365-Standard (1.8M images) and Places365-Challenge (8M images) for different training needs.

## Use Cases
**Scene Recognition:** Classification of images into one of the 365 scene categories (e.g., airport, bedroom, forest). **Transfer Learning:** Use of the pre-trained models (Places365-CNNs) as *feature extractors* for related computer vision tasks, such as object detection and semantic segmentation, due to their ability to learn generic scene *features*. **Geolocation:** Analysis of visual content to infer geographic location. **Video Analysis:** Classification of scenes in videos for indexing or summarization (e.g., in sports videos or movies). **Robotics and Autonomous Vehicles:** Assistance in understanding the environment for navigation and decision-making. **Content Moderation:** Automatic classification of images based on scene type.

## Integration
The dataset and the pre-trained models are available on the official project website. For integration into *deep learning* projects, the most common approach is through popular *framework* libraries: **PyTorch:** The dataset is accessible directly through `torchvision.datasets.Places365`. **Pre-trained Models:** The weights of the Places365-CNNs models (for AlexNet, VGG, ResNet, etc.) can be downloaded from the official GitHub repository and used with Caffe or PyTorch. **Usage:** The GitHub repository provides Python code examples (`run_placesCNN_basic.py`, `run_placesCNN_unified.py`) for performing scene prediction and CAM visualization. **Citation:** Use of the dataset requires citing the paper: Zhou et al., "Places: A 10 million Image Database for Scene Recognition," IEEE T-PAMI, 2017.

## URL
[http://places2.csail.mit.edu/](http://places2.csail.mit.edu/)
