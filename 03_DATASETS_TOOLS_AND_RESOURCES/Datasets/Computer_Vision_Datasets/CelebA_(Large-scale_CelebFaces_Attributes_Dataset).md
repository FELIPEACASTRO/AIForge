# CelebA (Large-scale CelebFaces Attributes Dataset)

## Description
The **CelebFaces Attributes Dataset (CelebA)** is a large-scale dataset of facial attributes, containing more than **200,000** celebrity images. Each image is annotated with **40 binary attributes** (such as "smiling", "blond hair", "eyeglasses") and **5 facial landmark locations**. The dataset is notable for its great diversity, covering significant variations in pose and background clutter. It is widely used in computer vision research for face-related tasks. The primary source is MMLAB at the Chinese University of Hong Kong. Although the original dataset is from 2015, it remains a foundational resource, with more recent related versions such as CelebA-HQ (high quality) and Multi-Modal-CelebA-HQ (with textual descriptions).

## Statistics
- **Images**: 202,599 face images.
- **Identities**: 10,177 unique identities.
- **Annotations**: 40 binary attributes and 5 landmarks per image.
- **Versions**: The original version is from 2015. Related high-quality versions include **CelebA-HQ** (30,000 high-resolution images) and **Multi-Modal-CelebA-HQ** (30,000 images with textual descriptions).
- **Size**: The total size of the dataset (images and annotations) is approximately 1.6 GB (for the aligned and cropped version) or larger for the "in-the-wild" version.

## Features
- **Rich Attributes**: 40 binary attributes per image, enabling the training of facial attribute recognition models.
- **Landmark Localization**: 5 facial landmarks (eyes, nose, mouth) annotated for each image.
- **Large Scale**: More than 200,000 images and 10,000 unique identities.
- **Diversity**: Significant variations in pose, expression, illumination, and background.
- **Aligned and Cropped Images**: Availability of "in-the-wild" images and preprocessed versions (aligned and cropped) to facilitate use.

## Use Cases
- **Facial Attribute Recognition**: Training models to identify attributes such as age, gender, presence of a beard, eyeglasses, etc.
- **Face Recognition**: Development and evaluation of systems for identifying individuals.
- **Landmark Localization**: Training models to locate key points on the face.
- **Facial Editing and Synthesis**: Generating new faces or modifying facial attributes (for example, using GANs or VAEs).
- **Bias and Fairness Research**: Analysis of bias in AI models due to the distribution of attributes in the dataset.

## Integration
The dataset can be downloaded directly from the links provided by the official source (Google Drive or Baidu Drive). For use in machine learning frameworks, libraries such as **Torchvision** (for PyTorch) and **TensorFlow Datasets (TFDS)** offer APIs for simplified downloading and loading of CelebA, facilitating integration into training pipelines. For example, in PyTorch, the `torchvision.datasets.CelebA` class can be used to download and load the dataset automatically. Use requires acceptance of the non-commercial-use agreement.

## URL
[https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
