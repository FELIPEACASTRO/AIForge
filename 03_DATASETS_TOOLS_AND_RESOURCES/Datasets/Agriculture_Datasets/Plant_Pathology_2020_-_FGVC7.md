# Plant Pathology 2020 - FGVC7

## Description

A dataset of apple leaf images used in the Kaggle Fine-Grained Visual Categorization 7 (FGVC7) competition in 2020. The goal is to classify the images into four health/disease categories: healthy leaf, apple rust, apple scab, and multiple diseases (combinations). The dataset is a fundamental resource for developing Computer Vision models (CNNs, Vision Transformers) for plant disease diagnosis.

## Statistics

Size: 823.79 MB. Contains 3645 files (JPG images and CSV metadata files). The target classes are: 'healthy', 'rust', 'scab', and 'combinations'. The training set has 3645 images, and the test set has 1821 images (in the original competition dataset).

## Features

High-resolution RGB images of apple leaves. The most recent feature engineering techniques (2023-2025) involve using pre-trained Deep Learning models (such as ResNet, EfficientNet, Vision Transformers - ViT) for automatic feature extraction, in addition to data augmentation techniques such as rotation, zoom, and color adjustments to improve model robustness.

## Use Cases

Automated plant disease diagnosis through images. Development of early warning systems for farmers. Applications in precision agriculture for monitoring crop health. Research in Computer Vision and Deep Learning for Fine-Grained Visual Categorization.

## Integration

The dataset can be accessed directly via the Kaggle API (requires authentication) or downloaded from the competition page. Integration into code projects generally involves libraries such as TensorFlow or PyTorch. Example code for downloading via the Kaggle CLI:\n\n```bash\nkaggle competitions download -c plant-pathology-2020-fgvc7\nunzip plant-pathology-2020-fgvc7.zip\n```\n\nIn Python, the integration for model training uses DataLoaders to load the images and labels from the `train.csv` file.

## URL

https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7